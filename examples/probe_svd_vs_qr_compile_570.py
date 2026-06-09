#!/usr/bin/env python3
"""#570 decision spike: symmetric SVD-vs-QR *compile-time* micro-benchmark.

The single question this answers
--------------------------------
The "drop-in QR projector" (Task 3-9 of the #570 plan) replaces the two
**non-truncating** half-system SVDs (M1, M2) of ``_compute_2x2_projector_symmetric``
with **QR**, keeping only the third (M', truncating) SVD. Task 2 already showed
the *backward op-count* win is only ~4.4% post-#593 — so op count alone is a
NO-GO. The ONLY remaining justification is **compile time**: the belief that XLA
lowers a per-sector symmetric **SVD** (and especially its VJP) into far more /
far costlier HLO than a symmetric **QR**, so removing 2 of 3 per-projector SVDs
across the differentiated CTM sweep is a real *compile* win even when op-count is
flat.

This spike tests that belief in isolation, cheaply, on CPU, with NO projector
build. It compares, on a representative (χD × χD) block-sparse matrix M:

  * **A: symmetric SVD** exactly as the projector calls M1/M2 —
    ``tenax.linalg.svd(M, ..., max_singular_values=None)`` (untruncated).
  * **B: symmetric QR** — ``tenax.linalg.qr(M, ...)``.

For each we build a jitted scalar-output ``f(M)`` (sum of |factor entries|²,
a differentiable real scalar), then measure for BOTH ``jax.jit(f)`` (forward) and
``jax.jit(jax.grad(f))`` (backward / VJP):

  1. **Lowered-HLO instruction count** (deterministic) — ``.lower().compile()``
     then count ``' = '`` assignment lines (same helper as
     ``profile_570_sweepvjp_compile._hlo_instr_count``).
  2. **Cold wall compile time** — fresh on-disk-cache dir per compile (the
     #584/#585 lesson: ``clear_caches()`` alone lets a repeat compile load from
     disk and corrupt the cold timing), warm once, then time a fresh
     ``.lower().compile()``.

It reports SVD vs QR for forward and backward, the **SVD/QR backward ratios**
(the headline), and a **composite** per-projector estimate: 3×SVD backward
(current path) vs 2×QR+1×SVD backward (drop-in path).

Representative M
----------------
The real M1/M2 are (χD × χD) 2-leg block-sparse SymmetricTensors (χ env leg × D
virtual leg, grouped). At the production bar D=4, χ=12 → χD=48. We build a
**synthetic-representative** M: a 4-leg ``(chi_L, d_L, chi_R, d_R)`` U(1)
SymmetricTensor with χ-leg charges spread over {-1,0,0,1} (repeated to length χ)
and D-leg charges {-1,0,0,1} (D=4) / {0,1} (D=2) — so the grouped (χD × χD)
matrix carries the multi-sector block structure of a real M (9 fused-charge
sectors at D=4/χ=12). It is decomposed by the SAME ``tenax.linalg.svd`` /
``tenax.linalg.qr`` the projector uses, so the per-sector lowering is
production-faithful even though the entries are random. (Driving the full
enlarged-corner fixture to get a byte-real M1_T is unnecessary for an *isolated
decomposition* compile measurement and was avoided to keep the spike cheap.)

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_svd_vs_qr_compile_570.py \
        --D 4 --chi 12
    JAX_PLATFORMS=cpu uv run python examples/probe_svd_vs_qr_compile_570.py \
        --D 2 --chi 12 --reps 3
"""

from __future__ import annotations

import argparse
import atexit
import platform
import shutil
import statistics
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import SymmetricTensor  # noqa: E402
from tenax.linalg import qr as tensor_qr  # noqa: E402
from tenax.linalg import svd as tensor_svd  # noqa: E402

# --------------------------------------------------------------------------- #
# Fresh persistent-cache dir per cold compile (the #584/#585 lesson; lifted
# from profile_570_sweepvjp_compile.py): tenax enables JAX's on-disk compile
# cache, so clear_caches() alone lets a repeat compile load from disk and
# corrupt the cold timing.
# --------------------------------------------------------------------------- #
_PRIOR_CACHE_DIRS: list[str] = []
atexit.register(
    lambda: [shutil.rmtree(d, ignore_errors=True) for d in _PRIOR_CACHE_DIRS]
)


def _fresh_cache_dir() -> None:
    while _PRIOR_CACHE_DIRS:
        shutil.rmtree(_PRIOR_CACHE_DIRS.pop(), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="jax_cc_svdqr570_")
    _PRIOR_CACHE_DIRS.append(d)
    jax.config.update("jax_compilation_cache_dir", d)


_LEFT = ("chi_L", "d_L")
_RIGHT = ("chi_R", "d_R")


def make_M(D: int, chi: int, seed: int = 0) -> SymmetricTensor:
    """A (χD × χD) representative of a projector M1/M2 (4 legs, 2×2 grouped).

    χ-leg charges are {-1,0,0,1} repeated to length χ; D-leg charges are
    {-1,0,0,1} (D=4) / {0,1} (D=2). The grouped (chi_L,d_L)×(chi_R,d_R) matrix
    then carries multiple fused-charge sectors, like a real M.
    """
    sym = U1Symmetry()
    if D == 4:
        dch = np.array([-1, 0, 0, 1], dtype=np.int32)
    elif D == 2:
        dch = np.array([0, 1], dtype=np.int32)
    else:
        dch = np.zeros(D, dtype=np.int32)
    base = np.array([-1, 0, 0, 1], dtype=np.int32)
    reps = int(np.ceil(chi / len(base)))
    chch = np.tile(base, reps)[:chi].astype(np.int32)
    idx = (
        TensorIndex.from_charges(sym, chch.copy(), FlowDirection.IN, label="chi_L"),
        TensorIndex.from_charges(sym, dch.copy(), FlowDirection.IN, label="d_L"),
        TensorIndex.from_charges(sym, chch.copy(), FlowDirection.OUT, label="chi_R"),
        TensorIndex.from_charges(sym, dch.copy(), FlowDirection.OUT, label="d_R"),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


def _block_sqsum(t: SymmetricTensor) -> jax.Array:
    """Differentiable real scalar: Σ |entries|² over all factor blocks."""
    return jnp.real(sum(jnp.sum(jnp.abs(b) ** 2) for b in t.blocks.values()))


def f_svd(m: SymmetricTensor) -> jax.Array:
    """Scalar through an UNtruncated symmetric SVD (as M1/M2 call it)."""
    U, S, Vh, _ = tensor_svd(
        m,
        left_labels=_LEFT,
        right_labels=_RIGHT,
        new_bond_label="b",
        max_singular_values=None,
    )
    return jnp.sum(jnp.abs(S) ** 2) + _block_sqsum(U) + _block_sqsum(Vh)


def f_qr(m: SymmetricTensor) -> jax.Array:
    """Scalar through a symmetric QR (the drop-in replacement for M1/M2)."""
    Q, R = tensor_qr(
        m, left_labels=_LEFT, right_labels=_RIGHT, new_bond_label="b"
    )
    return _block_sqsum(Q) + _block_sqsum(R)


def f_3svd(m: SymmetricTensor) -> jax.Array:
    """Composite CURRENT path: M1, M2, M' all via SVD (3×SVD), DISTINCT inputs.

    The three decompositions run on *distinct* rescaled copies of ``m`` (M1, M2,
    M' are three different matrices in the real projector). Reusing one ``m``
    would let XLA CSE 3×SVD -> 1×SVD and report a falsely-cheap current path.
    """
    return f_svd(1.0 * m) + f_svd(1.3 * m) + f_svd(0.7 * m)


def f_2qr_1svd(m: SymmetricTensor) -> jax.Array:
    """Composite DROP-IN path: M1, M2 via QR, M' via SVD (2×QR + 1×SVD)."""
    return f_qr(1.0 * m) + f_qr(1.3 * m) + f_svd(0.7 * m)


def _hlo_instr_count(compiled) -> int:
    """HLO instruction count = ``' = '`` assignment lines (profile_570 helper)."""
    try:
        txt = compiled.as_text()
    except Exception:  # noqa: BLE001
        return -1
    return sum(1 for ln in txt.splitlines() if " = " in ln)


def _cold_compile(fn, M):
    """Fresh-cold ``lower().compile()`` of ``jax.jit(fn)``; (compile_s, hlo)."""
    _fresh_cache_dir()
    jax.clear_caches()
    jitted = jax.jit(fn)
    lowered = jitted.lower(M)  # trace+lower (not timed; we want XLA compile)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0
    return t_compile, _hlo_instr_count(compiled)


def measure(fn, M, reps: int):
    """Warm once (prime any process caches), then `reps` cold compiles.

    HLO count is deterministic (verified); we report it once. Compile time is
    the median of the cold reps.
    """
    _cold_compile(fn, M)  # warm-up; discard
    times, hlo = [], -1
    for _ in range(reps):
        tc, ic = _cold_compile(fn, M)
        times.append(tc)
        hlo = ic
    return statistics.median(times), hlo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--chi", type=int, default=12)
    ap.add_argument(
        "--reps", type=int, default=3, help="Cold-compile reps (median of timings)."
    )
    ap.add_argument(
        "--no-composite",
        action="store_true",
        help="Skip the 3×SVD vs 2×QR+1×SVD composite.",
    )
    args = ap.parse_args()

    dev = jax.devices()[0]
    M = make_M(args.D, args.chi)
    chiD = M.indices[0].dim * M.indices[1].dim
    n_sectors_M = M.n_blocks
    U, S, _Vh, _ = tensor_svd(
        M, left_labels=_LEFT, right_labels=_RIGHT, new_bond_label="b",
        max_singular_values=None,
    )
    n_sectors_fused = U.n_blocks

    print("=" * 92)
    print("# #570 SVD-vs-QR symmetric DECOMPOSITION compile micro-spike")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(f"# x64={jax.config.read('jax_enable_x64')}  D={args.D} chi={args.chi}  "
          f"reps={args.reps}")
    print(f"# M: representative (synthetic) (chi_L,d_L)x(chi_R,d_R)  chiD={chiD}  "
          f"input_blocks={n_sectors_M}  fused_sectors={n_sectors_fused}")
    print("=" * 92)

    hdr = (
        f"{'decomp':>8} {'size':>10} {'fwd_HLO':>9} {'bwd_HLO':>9} "
        f"{'fwd_cmp_s':>10} {'bwd_cmp_s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    size = f"{chiD}x{chiD}"
    results = {}
    for nm, ffwd in [("SVD", f_svd), ("QR", f_qr)]:
        fwd_cmp, fwd_hlo = measure(ffwd, M, args.reps)
        bwd_cmp, bwd_hlo = measure(jax.grad(ffwd), M, args.reps)
        results[nm] = dict(
            fwd_hlo=fwd_hlo, bwd_hlo=bwd_hlo, fwd_cmp=fwd_cmp, bwd_cmp=bwd_cmp
        )
        print(
            f"{nm:>8} {size:>10} {fwd_hlo:>9} {bwd_hlo:>9} "
            f"{fwd_cmp:>10.3f} {bwd_cmp:>10.3f}"
        )

    if not args.no_composite:
        print("-" * len(hdr))
        for nm, ffull in [("3xSVD", f_3svd), ("2QR+1SVD", f_2qr_1svd)]:
            cmp_s, hlo = measure(jax.grad(ffull), M, args.reps)
            results[nm] = dict(bwd_hlo=hlo, bwd_cmp=cmp_s)
            print(
                f"{nm:>8} {size:>10} {'-':>9} {hlo:>9} "
                f"{'-':>10} {cmp_s:>10.3f}   (composite bwd)"
            )

    print("-" * len(hdr))
    s, q = results["SVD"], results["QR"]
    hlo_ratio = s["bwd_hlo"] / max(q["bwd_hlo"], 1)
    cmp_ratio = s["bwd_cmp"] / max(q["bwd_cmp"], 1e-9)
    fwd_hlo_ratio = s["fwd_hlo"] / max(q["fwd_hlo"], 1)
    fwd_cmp_ratio = s["fwd_cmp"] / max(q["fwd_cmp"], 1e-9)
    print(f"\nHEADLINE  SVD/QR backward  HLO ratio = {hlo_ratio:.2f}x   "
          f"compile-time ratio = {cmp_ratio:.2f}x")
    print(f"          SVD/QR forward   HLO ratio = {fwd_hlo_ratio:.2f}x   "
          f"compile-time ratio = {fwd_cmp_ratio:.2f}x")
    if "3xSVD" in results:
        c3, cd = results["3xSVD"], results["2QR+1SVD"]
        comp_hlo = c3["bwd_hlo"] / max(cd["bwd_hlo"], 1)
        comp_cmp = c3["bwd_cmp"] / max(cd["bwd_cmp"], 1e-9)
        print(f"COMPOSITE per-projector backward  3xSVD / (2QR+1SVD)  "
              f"HLO ratio = {comp_hlo:.2f}x   compile-time ratio = {comp_cmp:.2f}x")

    print("\nDECISION RULE (per the #570 spec):")
    print("  GO       : backward compile-time (or HLO) ratio >= ~2x  -> build the QR drop-in.")
    print("  MARGINAL : 1.3x - 2x  -> small end-to-end A100 check before committing.")
    print("  NO-GO    : < ~1.3x    -> #593 already captured the win; stop the drop-in.")


if __name__ == "__main__":
    main()
