# QR vs SVD projector at large χ on GPU — Findings (revisiting #570)

**Date:** 2026-06-22
**Trigger:** the chunked-einsum spike flagged the projector SVD (`χD²×χD²`-class) as the remaining
large-χ bottleneck; QR is more GPU-friendly. **Revisits** [[570-svd-vjp-wall]] /
[[570-phase3-kickoff]] (which reached a "wash" verdict) in the regime #570 did **not** test.
**Hardware:** single A100-80GB, f64.
**Verdict (ISOLATED decomposition): QR is ~1.4× fwd / ~1.3× bwd faster than SVD at large χ.** The
#570 "wash" was a different regime (small-D / compile-bound / block-sparse `"qr"`=eigh).
**Verdict (END-TO-END forward CTM — CORRECTS the takeaway, see the End-to-end section below): the
real large-χ lever is the reduced-corner `1×1` SCHEME (≈109× faster than the default `2×2` full-SVD);
within `1×1`, QR is ~1.25× *slower* per-sweep than SVD but uses ~1.4× *less memory* (higher χ
ceiling). QR's end-to-end contribution is MEMORY, not speed.**

## Grounding (verified in code)

- The dense reduced-corner QR projector `_ctm_projector._reduced_qr_projector` (#570 Phase 1,
  Yang/Zhang/Corboz arXiv:2505.00494) is the **real** QR. Its docstring + code: concat both reduced
  corners → **thin QR of `(χD² × 2χ)`** → tiny `2χ×2χ` Hermitian eigh → `P=Q@V`. **"No large `χD²`
  SVD anywhere."** Truncation-free / faithful.
- The **symmetric** `"qr"` (`_qr_projector_symmetric`) is per-sector **eigh** ("'qr' label retained
  for API compat") — NOT a real QR. So block-sparse "QR" measurements never tested real QR.
- Both QR and the default SVD projector decompose a tall-skinny `(χD² × χ)`-class matrix at
  `O(χ³D²)` FLOPs — **same complexity**. The difference is the kernel: QR = direct Householder
  (BLAS-3, GPU-friendly); SVD = iterative bidiagonalization (poor GPU utilization).

## Measurement (`examples/spike_qr_vs_svd_projector.py`, D=8 → D²=64, f64, A100)

Isolated projector decomposition, warm (compile-subtracted), forward AND the VJP (#570's wall):

| χ | fused χD² | svd_fwd | qr_fwd | **fwd↑** | svd_bwd | qr_bwd | **bwd↑** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 8.1 ms | 6.5 ms | 1.25× | 8.1 ms | 6.8 ms | 1.19× |
| 128 | 8192 | 22.3 | 7.1 | 3.12× | 13.8 | 7.7 | 1.78× |
| 256 | 16384 | 34.5 | 22.3 | 1.55× | 35.4 | 25.6 | 1.38× |
| 384 | 24576 | 64.4 | 47.5 | 1.36× | 66.8 | 56.4 | 1.18× |

QR is faster at **every** χ, fwd and bwd. The χ=128 fwd 3.12× is an outlier (SVD hit a slow GPU
path at that size); the robust signal is **~1.3–1.5× fwd / ~1.2–1.4× bwd**, roughly stable in χ.

## Interpretation

- **#570's "wash" doesn't transfer.** #570 chased the compile/SVD-VJP wall at small D and measured
  block-sparse (eigh-as-"qr"). At large-χ dense on GPU, real reduced-corner QR is a clear ~1.4× win
  on the projector decomposition — **including the backward** (the QR-VJP via `regularized_qr` beats
  the SVD-VJP, contra the #570 framing where SVD-VJP was the obstacle).
- **It's the right tool for the large-χ regime** the chunked-einsum spike exposed: chunking bounds
  the `χ²·D⁶` edge-growth peak; QR makes the projector decomposition (the *other* large-χ cost) ~1.4×
  cheaper and more GPU-efficient. They **compose** (chunked contractions + QR projector + GSPMD).

## Caveats / what this does NOT yet show

1. **Isolated decomposition, not end-to-end.** In the full CTM step the projector is a *fraction* of
   the work; the contractions (`χ²·D⁶`) dominate at large **D**, the projector (`χ³D²`) dominates at
   large **χ / moderate D**. So the ~1.4× projector win maps to an end-to-end win **only in the
   large-χ regime** — needs an end-to-end `recipe="1x1"` (QR) vs `"2x2"` (SVD) full-CTM measurement
   at large χ to size the real gain.
2. **Requires `recipe="1x1"`** (the reduced-corner QR-CTMRG scheme, #595/#596/#597) — a different CTM
   scheme from the default 2×2 Fishman. Built + AD-validated already; correctness is not in question.
3. Numbers are noisy at the ~×1.3 level; the claim is "QR consistently faster," not a precise factor.

## End-to-end forward CTM — the result that CORRECTS the takeaway

`examples/bench_qr_vs_svd_ctm_e2e.py`, A100 f64, warm per-sweep, three configs:
`svd2x2` (default = 2×2 Fishman, full-corner SVD), `svd1x1` and `qr1x1` (reduced-corner 1-site
scheme; **same scheme, only the projector differs** → clean isolation).

| D | χ | svd2x2 /sweep | svd1x1 /sweep | qr1x1 /sweep | svd1x1 peak | qr1x1 peak |
|---|---|---:|---:|---:|---:|---:|
| 4 | 64 | **2002 ms** | 7.7 ms | 31.7 ms | 0.54 GB | 0.68 GB |
| 4 | 128 | 2821 ms | 16.4 ms | 93.5 ms | 1.64 GB | 1.51 GB |
| 8 | 48 | **9731 ms** | 89 ms | 111 ms | 14.5 GB | 10.2 GB |
| 8 | 64 | — | 145 ms | 183 ms | 25.8 GB | 17.8 GB |
| 8 | 96 | — | **OOM** | 477 ms | OOM (>80) | **39.3 GB** |

**Three findings:**
1. **The dominant large-χ lever is the reduced-corner `1×1` SCHEME, not QR.** The default `svd2x2`
   is **~100–260× slower** (`svd1x1` 89 ms vs `svd2x2` 9731 ms at D=8/χ=48) — it does the **full
   `χD²×χD²` SVD**, which is catastrophic on GPU at large χ; `1×1` reduced-corner does only a
   `χD²×χ` decomposition. This dwarfs the QR-vs-SVD effect.
2. **Within `1×1`, QR is ~1.25× SLOWER per-sweep than SVD** (D=8: 111 vs 89, 183 vs 145). The
   isolated decomposition's 1.4× QR *speed* win does **not** survive end-to-end — the projector is a
   fraction of the sweep, and the `1×1` reduced SVD is already efficient. (At small D=4 it's worse,
   ~4×, where QR's 3-op overhead on a tiny `χD²` dominates.)
3. **QR's real end-to-end win is MEMORY:** `qr1x1` peak < `svd1x1` consistently (D=8/χ=64: 17.8 vs
   25.8 GB), and at **χ=96 `svd1x1` OOMs while `qr1x1` runs** (39 GB). The QR construction avoids the
   SVD's larger workspace/intermediates → higher χ ceiling.

So the corrected story: the isolated projector finding (QR faster) is real **but doesn't transfer to
wall-clock end-to-end**; QR instead buys **memory headroom** (higher χ before OOM), and the big
*speed* win at large χ is switching off the default 2×2 full-SVD to the reduced-corner 1×1 scheme.

## Recommendation (revised)

- **Large-χ dense CTM speed: use `recipe="1x1"` (reduced-corner), not the default `2×2`.** That is
  the ~100× lever — the default's full `χD²×χD²` SVD is the real large-χ wall (the user's "SVD is
  GPU-unfriendly" instinct, but the fix is the reduced-corner *scheme*, of which QR is one variant).
- **QR vs SVD projector is then a memory↔speed knob within 1×1:** SVD for ~1.25× faster sweeps; QR
  for ~1.4× less memory / higher χ ceiling (e.g. χ=96 fits with QR, OOMs with SVD). Pick by the
  binding constraint.
- Composes with chunked-einsum (edge peak) and GSPMD (multi-GPU). Targets the large-**χ** axis; does
  not change the truly-large-**D** verdict (eager/YASTN).
- **Caveat to verify before any default change:** confirm `2×2` vs `1×1` reach the **same converged
  energy/accuracy** (different CTMRG schemes; #570 AD-validated 1×1 correctness, but the
  scheme-vs-scheme accuracy/convergence-rate comparison at large χ is not measured here). **The
  `svd2x2` 100× is ROOT-CAUSED (not an artifact)** — see the Root-cause section below.

## Root cause of the `svd2x2` ~100× (debugged — RESOLVED, it is NOT an artifact)

Systematic debugging (`JAX_LOG_COMPILES` + differential `t(1)` vs `t(4)` + SVD-shape trace +
standalone SVD timing) pinned it:

- **Execution-bound, not compile.** Differential timing (shared jit-step cache): `t(1)=4.9 s`,
  `t(4)=42.5 s` → ~10–14 s/sweep of *cached execution*, compile ≈ 0.
- **The cost is the projector SVD.** Tracing the SVD shapes in the jitted step:
  - `recipe="2x2"`: **12 × `jnp.linalg.svd(3072×3072)`** per sweep (+1 tiny metric SVD).
  - `recipe="1x1"`: **5 × `svd(48×48)`** per sweep.
- **GPU f64 SVD is slow:** standalone A100 — `svd(48×48)=3.7 ms`, `svd(1536²)=315 ms`,
  **`svd(3072²)=1225 ms`**. So 12 × 1.2 s ≈ **14 s/sweep** for 2×2 vs ~18 ms for 1×1. Matches.
- **Why 3072 vs 48:** `_compute_projector_tensor` SVDs `M = C1g^H @ C4g = (col1×col2)`. In `1×1`
  the corners are **reduced** (`col=χ=48`); in the tensor `2×2` path they are **un-reduced**
  (`col=χD²=3072`). The *raw* 2×2 path (`_svd_projector_raw`) DOES reduce to χ×χ — so the tensor
  2×2 path's full-corner SVD looks like a **missed reduction**, not an inherent scheme cost.

**Two compounding causes:** (a) the default tensor 2×2 SVDs the **full** χD²×χD² corner (12×/sweep)
where a reduced χ×χ would do; (b) cuSOLVER f64 GPU SVD is intrinsically slow at these sizes. The
1×1 reduced-corner scheme fixes (a); QR / a faster SVD path would help (b).

**Fix options (not yet implemented — architectural choice):** (1) **use `recipe="1x1"`** for large-χ
dense — already works, ~100×; (2) **make the tensor 2×2 projector reduce the corner** (χ×χ
cross-product, like the raw path) before the SVD — could fix the *default* at large χ, needs
correctness verification against the 2×2 multisite scheme; (3) faster decomposition (QR, or even a
CPU-SVD callback — the GPU SVD is so slow a host SVD may beat 1.2 s).

## Artifacts (branch `spike/chunked-einsum-ctm`)

- `examples/spike_qr_vs_svd_projector.py` — isolated SVD-vs-reduced-corner-QR projector timing
  (fwd + VJP) over χ.
- `examples/bench_qr_vs_svd_ctm_e2e.py` — end-to-end forward CTM, svd2x2 / svd1x1 / qr1x1 per-sweep
  + peak (one config/process).
