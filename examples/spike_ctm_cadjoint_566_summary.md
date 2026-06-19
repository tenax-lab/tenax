# #566 C-adjoint feasibility spike — findings (NO-GO)

**Platform:** NVIDIA A100-SXM4-80GB · x64 · 2026-06-19
**Design:** `docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`
**Script:** `examples/spike_ctm_cadjoint_566.py` · probes: `_probe_cadjoint_discrepancy.py`, `_probe_cadjoint_numpy_inner.py`

## Question

Does lifting the symmetric CTM-energy core behind one `jax.custom_vjp` whose
forward/backward are `jax.pure_callback`s (host runs production
`ctm_energy_implicit` under `jax.disable_jit()`) collapse the #566 compile wall
to O(1) in charge-block count, while staying AD-correct?

## Gate 1 — compile (A100, cold `value_and_grad`)

| arm | ferm D2 χ8 | ferm D3 χ12 | dense D3 χ12 |
|-----|-----------|------------|-------------|
| production baseline (jitted)        | 205.9s (n=270)  | ~2111s (recorded) | ~40s (recorded) |
| **spike** (prod-JAX eager in callback) | **12.1s** (n=549) | **165.9s** (n=6363) | 9.4s (n=259) |
| **numpy-inner probe** (zero JAX in callback) | **0.6s** (n=22) | **0.6s** (n=22) | — |

(`n` = number of XLA compiles captured via `jax_log_compiles`.)

**Verdict: NO-GO on the strict criterion** — spike fermionic D2→D3 ratio = 13.66
(GO needed <2×), D3 = 165.9s (GO needed <30s), and fermionic (166s) ≠ dense (9.4s).

## What the spike actually established (rigorous, not a flat fail)

1. **The architecture is sound.** The numpy-inner probe (callback does a pure-NumPy
   computation — zero JAX ops inside) compiles in **0.6s, perfectly flat** (22
   compiles, ferm D2 == D3). So `custom_vjp` + `pure_callback` *does* collapse the
   *outer* graph compile to O(1), independent of block count.

2. **The NO-GO is the inner stand-in, not the architecture.** Under
   `jax.disable_jit()` the production CTM dispatches **eager, op-by-op**, and each
   primitive still triggers a tiny XLA compile. The count scales with block-count×D
   (549 → 6363 compiles, D2→D3), summing to 166s. JAX-eager carries its own per-op
   compile tax that the eventual numpy/C kernel would not.

3. **The three ways to fill the callback, and why none is a tractable win:**
   - *jitted inner* → just moves the 2111s fused-compile *inside* the callback.
   - *eager-JAX inner* → 12–17× faster cold-compile, but still scales (166s @ D3)
     **and** warm runtime regresses badly (unfused eager). Not a production win.
   - *numpy/C inner* → the only path to true O(1) (the 0.6s floor), but it requires
     a **full non-JAX reimplementation** of the symmetric CTM-AD core (contraction +
     fusion + truncated SVD/eigh **with VJP** + projectors + implicit adjoint). Not a
     drop-in kernel — a parallel CTM-AD stack that forfeits JAX autodiff/XLA.

A root-cause note: an earlier in-spike "~7e-2 callback discrepancy" was a **shared
`env_cache` warm-start confound** (`max_iter=8`, not converged), not a
`pure_callback`/`disable_jit` artifact — verified by
`_probe_cadjoint_discrepancy.py` (fresh cache: all four paths agree to 6.7e-16).
Fixed with separate fresh caches; `disable_jit` is numerically harmless.

## Conclusion

Efficient symmetric CTM-AD at D≥3 is a **practical NO-GO in JAX**: the
trace-and-compile model makes block sparsity a liability (per-block op emission =
compile wall; per-block host dispatch = warm wall), and every tractable lever is
now exhausted (batched dispatch #568/#618/#627, stacking #566, env de-frag #610,
uniform-sector env #615, cuTensorNet/FFI #200, the PBA+scan port, and this
C-adjoint). The only remaining path leaves JAX entirely.

> **Correction (2026-06-19, `examples/probe_padded_vmap_566_summary.md`):** the
> clause "the PBA+scan port [is] exhausted" / "every tractable lever exhausted"
> was an **overclaim**. A follow-up probe measured that the PBA+`vmap`/`scan`
> port's feared obstacle — padding heterogeneous blocks to uniform *converges to
> dense* — is **false for even-D fermionic**: the entire converged CTM environment
> (corners + edges) is block-shape-uniform at even D (χ=8, 16), so a padded-`vmap`
> representation has **zero padding waste, O(1) compile, and the full 2× Z₂
> sparsity**. What is genuinely NO-GO is the *partial* batched-contraction form
> (#568/#627), which leaves fusion + the sweep loop eager. The **full** port
> (padded `_fuse_indices_symmetric` + `lax.scan` fixed-point as one jitted graph)
> is **untested** at even D and is the one un-refuted lever. Odd D=3 / U(1)-Sz
> still converge to dense (block fragmentation). See `566-padded-vmap-evenD` memory.

It is **not** an algorithm limit: the same iPEPS-AD is efficient in **eager**
frameworks — YASTN (PyTorch) wins symmetric-vs-dense at large D precisely because
its per-block loop is cheap C-level with no compile wall.

## Recommendation

- **tenax/JAX:** dense is the pragmatic D≥3 path; symmetric AD is correct and
  beneficial only at small D (D=2 wins 1.37–1.78×). Speed up dense (env warm-start,
  `chi_ramp`, per-step CTM reconvergence) where wins are reachable.
- **Large-D symmetric iPEPS-AD specifically:** use **YASTN** (mature, published,
  already benchmarked) rather than reimplementing it in JAX. Right tool per regime;
  JAX owns dense/DMRG/TRG/TPU/small-D, eager owns large-D block-sparse.
