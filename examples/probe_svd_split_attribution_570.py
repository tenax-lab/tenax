#!/usr/bin/env python3
"""Per-SVD (M1 / M2 / M_prime) cost split of the #570 backward — go/no-go number.

`probe_bwd_subop_attribution_570.py` showed the CTM-AD compile wall is the
block-sparse **SVD VJP**, and `_ctm_tensor_projector_2x2.py`'s symmetric 2×2
Fishman projector (`_compute_2x2_projector_symmetric`) performs THREE
differentiated SVDs per call:

  * **M1**  — Stage 2 SVD of M1_T  (`new_bond_label="m1_bond"`,  call site ~L879)
  * **M2**  — Stage 2 SVD of M2_T  (`new_bond_label="m2_bond"`,  call site ~L889)
  * **M′**  — Stage 4 truncation SVD of M_prime_T
              (`new_bond_label="chi_new"`, call sites ~L943 / L962 / L971)

The planned #570 lever is a drop-in **QR projector** that replaces M1 and M2 with
QR (cheap backward) but must KEEP M′ as SVD (QR cannot truncate to χ). So the win
is the share of the backward SVD-VJP attributable to **M1+M2**, NOT M′.

This probe reuses the source-attribution machinery of
`probe_bwd_subop_attribution_570.py` (same fused-backward jaxpr; same
``svd_vjp`` categorization) and adds a finer split: every ``svd_vjp``-attributed
backward equation is sub-bucketed into {M1, M2, M_prime} by the SOURCE LINE
NUMBER of its originating call inside ``_ctm_tensor_projector_2x2.py``.

Robust mapping (verified empirically — see ``_SVD_REGIONS``): every svd_vjp eqn
carries at least one traceback frame in the projector file's FUNCTION BODY
(lineno ≥ 804, i.e. below the helper defs at the top of the module). Taking the
smallest such body-lineno lands on one of exactly six lines:

    879 (M1 SVD call) / 886 (M1 gauge-fix)   -> M1
    889 (M2 SVD call) / 896 (M2 gauge-fix)   -> M2
    943/962/971 (M′ SVD call) / 981 (M′ gauge-fix) -> M_prime

We bucket by proximity to the three known call-site regions rather than
hard-coding only the bare SVD lines, so the gauge-fix VJP ops (which are part of
each M1/M2/M′ decomposition and would be replaced/kept along with it) are
credited to the right SVD.

GO/NO-GO: GO if (M1+M2) ops are ≳ the M′ share; NO-GO otherwise.

RECONCILIATION WITH #589 (post-#593 baseline shift — verified 2026-06-09)
------------------------------------------------------------------------
The #589 handoff measured ``svd_vjp`` = 61.3 % of the backward (92,368 ops) at
D=4/χ=12 and 36.2 % (27,856 ops) at D=2/χ=12, GROWING with χ. THIS probe reports
``svd_vjp`` = 5,184 ops / ~10 % of the backward, FLAT in χ and D. That is NOT an
undercount of the split: it is the genuine effect of **PR #593**, which merged
between #589 and this branch and vectorized ``_gauge_fix_symmetric_svd`` from a
**per-column** scatter loop to a **per-sector** batched op.

The gauge-fix backward is categorized ``svd_vjp`` (it is part of each SVD
decomposition). Pre-#593 its per-column emission was the DOMINANT svd_vjp mass
and the χ-scaling driver: it scaled with the number of surviving singular
values (≈ χ × block size). #593 collapsed it to one op per charge sector, cutting
svd_vjp from 92,368 → 5,184 at D=4/χ=12 (−87,184 ops, −18×) and the whole
backward from 150,621 → 51,581. Empirically re-verified by swapping the pre-#593
projector back in: it reproduces #589's 150,621 / 61.3 % (D=4/χ=12) and 76,893 /
36.2 % (D=2/χ=12) EXACTLY (within trace nondeterminism). So #589 is correct for
its source snapshot; it is simply STALE — the lever it sized (per-column
gauge-fix) is already gone, and what remains of svd_vjp is the per-sector SVD
backward proper. ``svd/bwd%`` is printed alongside the split as the anchor.

The op count is **block-COUNT driven** (16 charge blocks, identical for even D
and saturated already at χ=4), so the traced jaxpr is byte-identical across D∈{2,4}
and all χ — exactly #589 fact #2 ("structural … constant across all χ AND
identical at D=2 and D=4"). Post-#593 svd_vjp now obeys the same flatness.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_svd_split_attribution_570.py \
        --D 4 --chi 8 12 16 --full
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

# Reuse the existing probe's fused-backward construction + source attribution
# verbatim so the unit profiled is the SAME compile unit. Ensure this examples/
# dir is importable when run as ``uv run python examples/probe_...py``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_bwd_subop_attribution_570 import (  # noqa: E402
    _categorize,
    _signif_frames,
    backward_vjp_jaxprs,
    make_site,
    walk_eqns,
)

_PROJ_FILE = "_ctm_tensor_projector_2x2.py"

# Body-region linenos (function body of `_compute_2x2_projector_symmetric`,
# which starts at L804; helper defs live above it, so a >=804 cutoff excludes
# the shared gauge-fix/qr helpers that the per-SVD code dispatches into).
_BODY_CUTOFF = 804

# Each known SVD's "region" of source lines: the bare SVD call site plus its
# immediately-following per-sector gauge fix / re-truncation. A body-frame
# lineno is bucketed to the region whose start it falls in (intervals are
# half-open and ordered). Verified against the actual emitted tracebacks:
#   879=M1 SVD, 886=M1 gauge-fix; 889=M2 SVD, 896=M2 gauge-fix;
#   943/962/971=M′ SVD, 978=M′ retruncate, 981=M′ gauge-fix.
# Bucket by proximity: M1 owns [879, 889), M2 owns [889, 902), M_prime owns
# [902, ∞). This is robust to small line drift since it keys off the three
# call-site *order*, not exact line numbers.
def _bucket_lineno(lineno: int) -> str:
    if lineno < 889:
        return "M1"
    if lineno < 902:
        return "M2"
    return "M_prime"


def _svd_source_bucket(eqn) -> str | None:
    """Sub-bucket an svd_vjp eqn into {M1, M2, M_prime} by its projector-file
    source line, or ``None`` if no projector-body frame is present."""
    tb = eqn.source_info.traceback
    if tb is None:
        return None
    body_lines = [
        fr.line_num
        for fr in tb.frames
        if fr.file_name.endswith(_PROJ_FILE) and fr.line_num >= _BODY_CUTOFF
    ]
    if not body_lines:
        return None
    # Smallest body lineno is the originating call site (gauge-fix/retruncate
    # frames sit on the same or higher line in the same region).
    return _bucket_lineno(min(body_lines))


def split_svd(jaxprs):
    """Return (totals dict, total_backward_ops, total_svd_vjp_ops).

    totals: Counter over {"M1", "M2", "M_prime", "unattributed"} of svd_vjp ops.
    """
    totals: collections.Counter = collections.Counter()
    total_backward = 0
    total_svd = 0
    for jx in jaxprs:
        for eqn in walk_eqns(jx):
            total_backward += 1
            frames = _signif_frames(eqn)
            if _categorize(frames) != "svd_vjp":
                continue
            total_svd += 1
            bucket = _svd_source_bucket(eqn)
            totals[bucket if bucket is not None else "unattributed"] += 1
    return totals, total_backward, total_svd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument(
        "--full",
        action="store_true",
        help="Include the params-sweep VJP (whole fused backward), not just "
        "the env-sweep VJP.",
    )
    args = ap.parse_args()

    os.environ.setdefault("TENAX_BATCH_BLOCKSPARSE", "0")
    unit = (
        "env+params sweep VJP (full fused backward)"
        if args.full
        else "env-sweep VJP (apply_Jt / Neumann matvec)"
    )
    print(f"# #570 per-SVD (M1/M2/M_prime) split | D={args.D} | unit={unit}")
    print(f"# x64={jax.config.read('jax_enable_x64')} | projector=svd\n")

    A = make_site(args.D)

    # #589-style anchor columns (total_backward + svd/bwd%) are printed alongside
    # the split so this output is directly comparable to the #589 baseline table
    # (D=4/χ=12: total≈150,663, svd_vjp=61.3% PRE-#593; ≈51,581 / ~10% POST-#593).
    hdr = (
        f"{'chi':>5} {'M1_ops':>8} {'M2_ops':>8} {'Mprime_ops':>11} "
        f"{'total_svd_vjp':>14} {'total_bwd':>10} {'(M1+M2)%':>9} {'svd/bwd%':>9}"
    )
    print(hdr)
    rows = []
    for chi in args.chi:
        jaxprs = backward_vjp_jaxprs(A, chi, full=args.full, projector="svd")
        totals, total_bwd, total_svd = split_svd(jaxprs)
        m1 = totals["M1"]
        m2 = totals["M2"]
        mp = totals["M_prime"]
        unattr = totals["unattributed"]
        m12 = m1 + m2
        m12_share = 100 * m12 / total_svd if total_svd else 0.0
        svd_bwd_share = 100 * total_svd / total_bwd if total_bwd else 0.0
        rows.append((chi, m1, m2, mp, total_svd, total_bwd, unattr))
        print(
            f"{chi:>5} {m1:>8} {m2:>8} {mp:>11} {total_svd:>14} {total_bwd:>10} "
            f"{m12_share:>8.1f}% {svd_bwd_share:>8.1f}%"
        )

    # Any unattributed svd_vjp ops would undermine the split — surface them.
    tot_unattr = sum(r[6] for r in rows)
    if tot_unattr:
        print(f"\n# WARNING: {tot_unattr} svd_vjp ops had NO projector-body frame "
              "(unattributed; split may be incomplete).")

    print("\n# RECONCILIATION vs #589 (PR #593 baseline shift):")
    print(
        "#  - #589 (pre-#593): D=4/chi=12 total_bwd=150,663, svd_vjp=61.3%, GROWING "
        "with chi."
    )
    print(
        "#  - Now (post-#593): total_bwd=51,581, svd_vjp~10%, FLAT in chi AND D. "
        "Not an undercount."
    )
    print(
        "#  - Cause: #593 vectorized _gauge_fix_symmetric_svd (per-column -> "
        "per-sector). The gauge-fix backward is svd_vjp; its per-column emission "
        "WAS the dominant svd_vjp mass + the chi driver. #593 cut svd_vjp "
        "92,368 -> 5,184 at D=4/chi=12 (re-verified by swapping the pre-#593 "
        "projector back in -> reproduces #589 exactly)."
    )
    print(
        "#  - Op count is block-COUNT driven (16 blocks for even D, saturated at "
        "chi>=4) -> jaxpr identical across D in {2,4} and all chi (#589 fact #2)."
    )

    print("\n# READOUT (#570 QR-drop-in go/no-go):")
    print(
        "#  - QR drop-in replaces M1+M2 SVDs with (cheap-backward) QR; M_prime "
        "STAYS SVD (QR can't truncate to chi)."
    )
    print(
        "#  - The reachable win is the M1+M2 share of the svd_vjp ops. GO if "
        "(M1+M2) >~ M_prime; NO-GO if M1+M2 is a small slice."
    )
    print(
        "#  - Expected backward-op reduction ~ (M1+M2 ops)/total_backward_ops x "
        "(1 - 1/2.6)  [2.6x = per-sector QR-vs-SVD VJP op ratio from Task 1]."
    )
    # Print the reduction estimate for the largest chi.
    chi, m1, m2, mp, total_svd, total_bwd, _ = rows[-1]
    m12 = m1 + m2
    est = (m12 / total_bwd) * (1 - 1 / 2.6) if total_bwd else 0.0
    print(
        f"#  - At chi={chi}: (M1+M2)={m12} of total_backward={total_bwd} -> "
        f"est. backward-op reduction ~= {100 * est:.1f}%."
    )


if __name__ == "__main__":
    main()
