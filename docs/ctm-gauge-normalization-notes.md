# Gauge Selection and Normalization for Differentiable CTMRG

> Note assembled 2026-04-28 from a third-party design summary, cross-checked against
> the current Tenax CTM AD code (`src/tenax/algorithms/_ctm_*.py`,
> `ipeps_config.py`, `ipeps_ad_policy.py`, `ad_utils.py`) and the in-house
> benchmarks accumulated through PRs #322, #334, #341.
>
> The original summary text is preserved as a baseline; **Tenax cross-check**
> blocks add code citations and benchmark numbers, and flag places where the
> generic recommendation needs to be qualified for Tenax's 1-site vs 2-site
> code paths.

## 1. Core challenge: gauge ambiguity and AD stability

During the CTMRG power method, truncation projectors computed via eigensolvers
(or SVD) introduce arbitrary phase degrees of freedom at each iteration.
While physical observables converge, the environment tensors fluctuate
element-wise. This lack of element-wise convergence prevents the establishment
of a smooth fixed point, causing chaotic VJPs and destabilising implicit
differentiation (adjoint methods).

**Tenax cross-check.** Confirmed empirically. Element-wise convergence is the
operative criterion for both implicit (GMRES adjoint) and explicit AD paths in
Tenax. Diagnostic measurements after 49 CTM iterations on 2-site D=2 χ=8
Heisenberg:

| Projector | Element-wise diff at iter 49 |
|-----------|------------------------------|
| eigh      | 9.73e-01 (never converges)   |
| **svd**   | **0.00e+00**                 |
| qr        | 8.49e-01                     |

Only Fishman SVD projectors achieve element-wise convergence; eigh's
eigenvector sign ambiguity prevents it regardless of the gauge fix applied
afterwards.

## 2. Gauge fixing strategies

### 2.1 Phase gauge (a.k.a. σ-gauge in the original summary) — Tenax default

*Mechanism.* Apply a per-tensor phase fix after every CTM absorption: divide
by the Frobenius norm, then multiply by a unit complex phase chosen so that a
designated reference element becomes real and positive. variPEPS and Tenax
both pick that reference element using a **first-above-threshold** rule
(first index whose `|x| ≥ 0.1·max|x|`), not `argmax|x|`.

*AD impact.* Yields smooth iteration-to-iteration tensors, which makes the
fixed-point equation continuously differentiable in the local PEPS site
tensor.

**Tenax cross-check.**

- Default in `CTMConfig.forward_gauge` is `"phase"`
  (`src/tenax/algorithms/ipeps_config.py:68`).
- `build_ad_ctm_config` does **no** silent gauge promotion
  (`src/tenax/algorithms/ipeps_ad_policy.py:67–78`); user choice is preserved.
- The first-above-threshold convention is implemented in two places and they
  must stay in sync:
  - per-absorption: `_phase_fix_normalize_tensor`,
    `src/tenax/algorithms/_ctm_tensor_moves.py:36–59`
    (`threshold = 0.1 * max(|x|)`, `argmax(abs_flat >= threshold)`).
  - per-sweep: `_phase_fix_ctm_tensor`,
    `src/tenax/algorithms/ad_utils.py:881–934`
    (`EPS_PHASE = 0.1`, identical logic at lines 903–911).
- Switching from `argmax(|x|)` to first-above-threshold turned a
  non-variational result into a variational one on 2-site non-C4v complex128,
  D=2 χ=16: **E_best −0.7128 → −0.6406** (4.3% above exact −0.6694) at
  ≈1 min/step on GPU after the per-absorption fix landed.
- Per-tensor phase fix is applied **inside the sweep, after each absorption**,
  not just once after the full sweep — matches variPEPS
  `_post_process_CTM_tensors`.

> **Naming note.** The original summary uses the label "σ-gauge" for what
> Tenax (and variPEPS) call **phase gauge**. In Tenax, "sigma gauge" refers
> to a different scheme (§2.3 below) — aligning the current environment to
> the previous iteration via transfer-matrix eigenvectors. Do not conflate
> them.

### 2.2 QR gauge — discouraged for AD

*Mechanism.* Enforce uniqueness by constraining the diagonal of the upper
triangular factor R from QR to be real and positive: `S_ii = R_ii / |R_ii|`,
then `Q ← Q · diag(S)`, `R ← diag(S) · R`.

*AD impact.* Highly unstable in continuous optimisation. The correction is
evaluated independently of iteration history, so microscopic perturbations can
flip signs of small diagonal entries, triggering chaotic phase wrapping.
Gradients of `R_ii / |R_ii|` blow up like `1/|R_ii|` near zero.

**Tenax cross-check.**

- Sign-fixed QR is implemented (`_sign_fixed_qr` and `_gauge_fix_ctm_tensor`,
  `src/tenax/algorithms/ad_utils.py:819–878`) but is treated as legacy.
- Empirically chaotic for AD: D=2 χ=8 with eigh projectors and QR gauge
  oscillates by ±0.1 in energy with spikes to −4.0 and never converges. With
  QR gauge the implicit-AD adjoint diverges regardless of solver.
- `build_ad_ctm_config` rewrites `forward_gauge="qr"` to `"phase"` for AD
  paths (the only auto-promotion that exists), preserving behavior for
  legacy non-AD callers.

### 2.3 σ-gauge in Tenax (transfer-matrix alignment) — 1-site only

*Mechanism (Tenax-specific, distinct from §2.1).* Align the current
environment to the previous iteration's environment by computing an overlap
between corresponding transfer-matrix eigenvectors, then solving for a
similarity matrix σ that rotates the new environment back into the old basis.
This is the YASTN convention (arXiv:2311.11894).

**Tenax cross-check.**

- Works for **1-site** C4v iPEPS: D=2 χ=8 reaches E=−0.6601 with both
  implicit (GMRES) and explicit AD; matches literature −0.6625.
- **Breaks for 2-site non-C4v.** σ-gauge produces inconsistent A/B
  alignments because the per-site transfer matrix eigenvectors are no longer
  related by symmetry. Energy drifts non-variationally: best −1.04 at D=2
  χ=8, −1.816 at D=2 χ=16. This is why phase gauge — not σ-gauge — is the
  default and why `build_ad_ctm_config` does not auto-promote to σ.
- σ-gauge requires two `stop_gradient`s in `_ctm_energy_ad.py` to avoid NaN
  / vanishing gradients for χ ≥ 8:
  1. on the σ matrices themselves (QR of rank-deficient transfer-matrix
     eigenvectors when χ > D² makes the QR VJP diverge);
  2. on the reference environment passed into `_jit_apply_Jt`, otherwise the
     Jacobian picks up spurious reference-path derivatives that push
     eigenvalues toward 1 and make `(I − Jᵀ)` near-singular.

  See `src/tenax/algorithms/ad_utils.py:746–749` (σ stop_gradient) and
  `:1232` (reference env stop_gradient), mirrored in `_ctm_energy_ad.py`.

## 3. Normalization schemes for complex tensors

### 3.1 Frobenius (L2) norm — Tenax default

Normalising by `‖T‖_F = √Σ|T_ijk…|²` yields a real positive scalar, scales
all entries uniformly, and preserves relative complex phases. JAX handles
the non-holomorphic conjugate via Wirtinger calculus without requiring
real/imaginary splitting.

**Tenax cross-check.** Implemented as `arr / (jnp.linalg.norm(arr) + 1e-30)`
in both `_phase_fix_normalize_tensor`
(`src/tenax/algorithms/_ctm_tensor_moves.py:49`) and `_phase_fix_ctm_tensor`
(`src/tenax/algorithms/ad_utils.py:901`). The `+ 1e-30` floor prevents a
divide-by-zero in the `T = 0` limit; in normal CTM operation the norm is
O(1) by construction.

### 3.2 Fixed-index normalisation — useful but brittle

Dividing by `T[i₀, j₀, …]` is purely holomorphic and pins the global phase,
but collapses if the chosen entry approaches zero. To stay AD-friendly the
choice of index must be static (no `jnp.where` branching on tensor values).

**Tenax cross-check.** Not used as the primary normalisation in the AD path.
The closest analogue is the first-above-threshold reference element used by
the **phase fix** (§2.1), which dynamically picks the index per call but
isolates the dynamic choice inside `argmax` — JAX traces this without
branching as long as the reference index itself is treated as data, not a
control-flow predicate.

### 3.3 Maximum absolute element — discouraged

`max|T|` combined with a max-finding op creates sharp ridges in the gradient
landscape; "max-crossings" cause discontinuous jumps that destabilise the
backward pass.

**Tenax cross-check.** Confirmed by the `argmax(|x|)` → first-above-threshold
fix described in §2.1: when the reference index for the **phase fix** was
chosen by raw argmax, two near-degenerate elements would exchange ranks
between iterations and flip the global phase, taking the optimiser
non-variational. Switching to first-above-threshold is what closed the
−0.7128 → −0.6406 gap. The same pathology applies to using `max|T|` as a
normaliser, so it is avoided.

## 4. Conclusion (Tenax-adjusted)

For stable optimisation of complex iPEPS via differentiable CTMRG, Tenax's
defaults are:

1. **`forward_gauge="phase"`** — per-absorption Frobenius normalisation +
   first-above-threshold phase fix. Works for 1-site and 2-site, matches
   variPEPS. Do **not** auto-promote to σ-gauge.
2. **`projector_method="svd"`** — Fishman two-projector with `S_safe =
   where(S > cutoff, S, 1.0)` masking. Required for element-wise CTM
   convergence and to avoid eigh's NaN gradients in the χ > D² regime.
3. **complex128 tensors** for non-C4v multi-site. Real float64 is
   pathological in this case: the implicit-diff linear system `(I − Jᵀ)λ =
   g` becomes ill-conditioned in the real subspace (40-step L-BFGS drifts
   to E = −1.498, non-variational). Doubling the parameter space with
   complex tensors well-conditions the system and the optimiser stays
   variational.
4. **σ-gauge (transfer-matrix alignment)** is opt-in for 1-site C4v users
   who want to mirror YASTN; it requires the two `stop_gradient`s described
   in §2.3 and must not be used on multi-site cells without C4v constraints.
5. **QR gauge** is retained for legacy non-AD callers only; AD configs
   silently rewrite it to `"phase"`.

### Tenax-specific items not in the original summary

- The first-above-threshold convention for the phase reference element is
  load-bearing — it is the single change responsible for staying variational
  in the 2-site non-C4v case.
- The phase fix must be applied **per absorption move**, not only once per
  full sweep.
- For 2-site non-C4v, complex128 is a correctness requirement, not a
  performance choice.
- σ-gauge needs `stop_gradient` on σ and on the reference environment to
  avoid NaN / vanishing gradients above χ = 8.
- SVD projectors with the `S_safe` mask are required for the χ > D² regime
  where eigh's degenerate-eigenvalue backward divides by zero.

### Headline benchmark (current state)

D=2 χ=16 2-site non-C4v Heisenberg, complex128, GPU, all defaults:
**E_best = −0.6406** (variational, 4.3 % above exact −0.6694), ≈1 min/step.
D=3 χ=16 2-site C4v Heisenberg: **E_best = −0.6521**, monotonic, 7.9 s/step.

### Pointers

- `CTMConfig` defaults: `src/tenax/algorithms/ipeps_config.py`
- AD config policy: `src/tenax/algorithms/ipeps_ad_policy.py`
- Phase / σ / QR gauge implementations: `src/tenax/algorithms/ad_utils.py`,
  `src/tenax/algorithms/_ctm_tensor_moves.py`,
  `src/tenax/algorithms/_ctm_energy_ad.py`
- Fishman SVD projectors with `S_safe`: `src/tenax/algorithms/_ctm_projector.py`
- Benchmark scripts: `benchmarks/bench_ipeps_ad.py`,
  `benchmarks/bench_gmres_precond.py`
- Related PRs: #322 (sigma + VJP), #334 (5-fix breakthrough), #341
  (Python-loop CTM AD + first-above-threshold + JAX GMRES).
