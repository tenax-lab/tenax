# #570 — drop-in QR projector is NO-GO (3 spikes); pivot to faithful reduced-corner QR-CTMRG

**Date:** 2026-06-09 · **Branch:** `feat/qr-projector-2x2-570` · **Issue:** #570 (umbrella #566)

## TL;DR

We designed a **drop-in QR projector** for the symmetric 2×2 Fishman path
(`_compute_2x2_projector_symmetric`): replace the two non-truncating M1/M2 half-SVDs with
QR, keep the M′ truncation SVD. Three cheap pre-build spikes (no projector built) **killed it**,
and together they say the real compile win lives in the **faithful reduced-corner QR-CTMRG**
(remove *all* SVDs), which is now the chosen direction. Spec:
`docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md` (status NO-GO). Plan:
`docs/superpowers/plans/2026-06-09-qr-projector-2x2-ad-drop-in.md` (not executed past spikes).

## The three spikes

**Spike 1 — QR VJP stability** (`examples/probe_qr_vjp_stability_570.py`, commit `0c2c731`).
`check_grads` on gauge-fixed `jnp.linalg.qr` reduced to Q: PASS well-conditioned + tall,
**FAIL near-rank-deficient** (cotangent blows up as `diag(R)→0` — the regime projector sectors
hit near truncation). ⇒ a drop-in would need a `regularized_qr` custom-VJP (analogous to
`regularized_svd`). Not a killer alone, but added scope.

**Spike 2 — per-SVD cost attribution + #589 reconciliation**
(`examples/probe_svd_split_attribution_570.py`, commits `a520210`, `66d32ca`). Within the
backward SVD-VJP, **M1+M2 = 70.8%**, M′ = 29.2% (good: drop-in targets the right two). BUT the
SVD-VJP is only **~10.1% of the whole backward** post-#593 — **not** #589's 61%. Reconciled
**empirically**: swapping the pre-#593 projector source back in reproduces #589 *exactly*
(D=4/χ=12 → 150,621 / 61.3%). **PR #593's gauge-fix vectorization already banked the bulk of the
old SVD-VJP mass** (the per-column gauge-fix backward was categorized under `svd_vjp`, and #593
collapsed it per-sector). So drop-in QR saves only `(M1+M2)/total × (1−1/2.6) ≈ 4.4%` of backward
ops. The spec's original "~25%" estimate was pre-#593 and is obsolete.

**Spike 3 — SVD-vs-QR compile micro-benchmark**
(`examples/probe_svd_vs_qr_compile_570.py`, commit `e46b6d9`). The remaining justification was
compile-time (SVD custom-call lowerings ≫ QR). Measured (CPU, x64, realistic per-sector sizes):

| level | SVD/QR backward HLO | SVD/QR backward compile-time |
|---|---|---|
| single decomposition (D=4/χ=12, 48×48) | **2.24×** (2.52× at D=2) | 1.47× (1.43×) |
| per-projector composite (3×SVD vs 2×QR+1×SVD) | **1.58–1.68×** | **1.32–1.42×** |

Verdict **MARGINAL**: the premise is real (SVD backward = 2.2–2.5× QR backward in HLO), but the
**drop-in is capped at ~1.4× per-projector compile because it keeps the M′ truncation SVD** and
pays for two QR backwards (net ~37% fewer HLO, not 56%). Above the <1.3× NO-GO floor, below the
≥2× slam-dunk. (Methodology note: composite needed three *distinct* inputs to defeat XLA CSE,
which had falsely reported ~1.0×.)

## Decision

**Drop-in: NO-GO.** ~4.4% op-count + ~1.4× capped compile, and #593 already took the decomposition
win. **Pivot to the faithful reduced-corner QR-CTMRG** (Yang/Zhang/Corboz, arXiv 2505.00494): a
rank-χ *reduced* corner + unpivoted QR makes Q the projector with **no truncation SVD at all** —
removing all three SVD lowerings, so it captures the full ~2.2–2.5×-per-decomposition compile win
that the drop-in cannot. Trade-off: larger rewrite (replaces the Fishman two-projector with the
standard single-isometry scheme), a new *approximate* fixed point, warm-up regime switch
(`qr_warmup_steps` already anticipates it), and full energy/convergence revalidation. Gets its own
brainstorm → spec → plan.

## Reusable artifacts (keep)

- `regularized_qr` is still needed for any QR-AD path (Spike 1) — carry forward.
- `_gauge_fix_symmetric_qr` design (diag(R)≥0 per sector) — carry forward.
- The three probes are good #570 diagnostics; `probe_svd_vs_qr_compile_570.py` directly informs
  the faithful design's expected compile win.

## Method lessons (recorded)

1. **Spike before building.** Three cheap trace/compile probes (minutes each, no projector built)
   converted a plausible "~25% win" into a measured NO-GO before any implementation cost.
2. **A landed PR can move the baseline.** #589's 61% was stale the moment #593 merged; always
   re-anchor a cost estimate against the *current* tree (the empirical pre/post-#593 swap settled
   it). op-count ≠ HLO ≠ compile-time — measure the artifact the decision depends on.
