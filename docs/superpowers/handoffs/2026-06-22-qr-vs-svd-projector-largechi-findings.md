# QR vs SVD projector at large χ on GPU — Findings (revisiting #570)

**Date:** 2026-06-22
**Trigger:** the chunked-einsum spike flagged the projector SVD (`χD²×χD²`-class) as the remaining
large-χ bottleneck; QR is more GPU-friendly. **Revisits** [[570-svd-vjp-wall]] /
[[570-phase3-kickoff]] (which reached a "wash" verdict) in the regime #570 did **not** test.
**Hardware:** single A100-80GB, f64.
**Verdict: QR WINS at large χ on GPU — ~1.4× fwd, ~1.3× bwd, consistently.** The #570 "wash" was a
different regime (small-D / compile-bound / block-sparse `"qr"`=eigh). The user's instinct holds.

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

## Recommendation

**Revisit confirmed — pursue QR projector for the large-χ dense regime.** Next: an end-to-end
`recipe="1x1"` vs `"2x2"` forward-CTM benchmark at large χ on GPU to size the real speedup, then
consider a "prefer QR projector when χ is large" policy (or default for the large-χ dense path).
Composes with chunked-einsum (edge peak) and GSPMD (multi-GPU). Does not change the truly-large-**D**
verdict (eager/YASTN); it targets the large-**χ** axis.

## Artifacts (branch `spike/chunked-einsum-ctm`)

- `examples/spike_qr_vs_svd_projector.py` — isolated SVD-vs-reduced-corner-QR projector timing
  (fwd + VJP) over χ.
