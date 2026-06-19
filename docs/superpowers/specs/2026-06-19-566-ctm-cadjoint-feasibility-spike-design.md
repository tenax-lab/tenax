# C-adjoint feasibility spike for the #566 symmetric CTM-AD compile wall — design

**Date:** 2026-06-19
**Issue:** #566 (block-sparse `SymmetricTensor` AD compile cost scales with charge-block count)
**Status:** EXECUTED — **Gate 1 NO-GO** (A100, 2026-06-19). The architecture is sound
(numpy-inner probe → outer compile 0.6s, flat in block count), but the production-JAX
stand-in under `disable_jit` still scales (166s @ D3) — realizing O(1) needs a full
non-JAX CTM-AD core. Gate 2 not run (NO-GO branch). Findings:
`examples/spike_ctm_cadjoint_566_summary.md`.

## 1. Why this, and why now

The #566 symmetric CTM-AD speedup effort has accumulated ~10 documented NO-GOs
(batched dispatch, stacked block-sparse, env de-fragmentation, uniform-sector
env, cuTensorNet FFI, …). The last *un-refuted* lever in the active plan was
"port the DMRG `_jit_sweep` machinery (`PaddedBlockArray` + `lax.scan`) to the
CTM sweep so the symmetric path looks dense to XLA."

That lever rests on a **false premise**, surfaced during design review:

- **`PaddedBlockArray` + `lax.scan` ("pure JIT") is *not* what speeds up symmetric
  DMRG.** The production accelerator is a **Cython/C module** —
  `_cython_execute_plan` / `_blockwise_contract` doing raw BLAS (`dgemm`) per
  block, plus a fused C Lanczos (`cython_lanczos_ground`). The block-sparse
  matvec stays *eager* (Python loop over blocks); the win is each block's work
  dropping into C with zero per-block Python overhead. The symmetric PBA jit
  sweep exists but was never the production path and was never
  performance-benchmarked (only the *dense* jit sweep was, PR #209).
- The Cython path is **DMRG/CBE/DMRG3S-only**; `contractor.py`,
  `_ctm_tensor_moves.py`, `_tensor_utils.py` touch no Cython.
- `dmrg.py:1838` records *"cuTensorNet integration for `_blockwise_contract` is
  not beneficial"* — the same NO-GO as cuTensorNet #200.

**Why the C mechanism does not trivially transfer to CTM-AD:** DMRG's C matvec
sits inside a Lanczos eigensolve that is **never differentiated through XLA**.
CTM-AD must differentiate through the whole sweep, and the *measured* compile
wall (#585/#589) is the per-block op emission in the **AD backward**
(`_jit_fused_fixed_point_bwd`), not the forward kernel. Using a C block executor
under AD therefore requires `jax.custom_vjp` wrapping a C forward **and a
hand-written C adjoint**. The forward-only FFI form of this is already NO-GO
(#200 + the dmrg.py note).

This spike tests the one genuinely-untried architectural idea before any C is
written: **lift the entire CTM-energy core out of XLA AD behind a single
`custom_vjp` whose forward and backward are host callbacks, so XLA never emits
(or compiles) per-block ops in either direction.**

## 2. The measured wall (what we are attacking)

From the Phase-0 profiler (`examples/profile_566_a100_summary.md`,
`profile_566_a100_Dsweep.json`), A100, x64, implicit path, fermionic vs dense:

| sym       | D | χ  | blk | fwd_cmp | vg_cmp        | bwd_cmp |
|-----------|---|----|-----|---------|---------------|---------|
| fermionic | 2 | 8  | 16  | 63.8s   | **206.4s**    | 142.6s  |
| fermionic | 3 | 12 | 16  | 527.8s  | **2111.3s**   | 1583.5s |
| dense     | 3 | 12 | 1   | 27.8s   | **40.6s**     | 12.8s   |

Key facts the spike design depends on:

1. The wall is **compile**, not warm runtime (the implicit path's warm step uses
   jitted kernels and is fast; #618 measured the jitted backward at ~0.13s).
2. **Both** directions emit per-block ops at compile — the forward sweep-step
   compile alone is ~528s at D=3. So collapsing compile requires making *both*
   forward and backward opaque to XLA, not just the backward.
3. The compile cost is one-time per `(shape, code-version, jaxlib)` and the
   persistent on-disk cache (`jax_compilation_cache_dir`, enabled in
   `tenax/__init__.py`) only saves *unchanged* code. So the wall is paid on
   every code change, cold start, and CI run.

## 3. Architecture

A standalone `examples/spike_ctm_cadjoint_566.py`. **Zero production edits.** It
defines a parallel `ctm_energy_cb` that mirrors `ctm_energy_implicit`'s
`custom_vjp` interface but routes forward and backward through `jax.pure_callback`:

```
loss(A) = ctm_energy_cb( A_normalized )     # normalization stays in JAX, differentiated normally

@jax.custom_vjp
def ctm_energy_cb(A_data):                   # A_data = the flat SymmetricTensor _data buffer
    ...

def fwd(A_data):
    energy = pure_callback(host_energy, ShapeDtypeStruct((), f64), A_data)
    return energy, A_data                     # residual = A_data (env recomputed in bwd)

def bwd(A_data, ct):
    dA = pure_callback(host_grad, ShapeDtypeStruct(A_data.shape, A_data.dtype), A_data, ct)
    return (dA,)
```

**The simplification (cuts Gate-2 risk to plumbing only):** the host functions do
**not** reimplement CTM. They call the existing production
`ctm_energy_implicit` under **`jax.disable_jit()`**, which turns every internal
`@jax.jit` (`_make_jit_ctm_step`, `_jit_fused_fixed_point_bwd`) into eager
op-by-op dispatch. No fused per-block jaxpr is ever built ⇒ no compile wall
*inside* the callback, and:

```
def host_energy(A_data):
    with jax.disable_jit():
        A = reconstruct_symmetric(A_data, META)        # META = static block metadata, closure
        return np.asarray(ctm_energy_implicit_eager(A))

def host_grad(A_data, ct):
    with jax.disable_jit():
        A = reconstruct_symmetric(A_data, META)
        _, vjp = jax.vjp(ctm_energy_implicit_eager, A)
        (dA,) = vjp(jnp.asarray(ct))
        return np.asarray(dA._data)
```

`ctm_energy_implicit_eager` is just `ctm_energy_implicit` bound to the spike's
fixed kwargs (1×1 cell, `SINGLE_SITE_NEIGHBORS`, the gate). Its own `custom_vjp`
(the implicit adjoint) still applies under `disable_jit`, so `host_grad` returns
the *production implicit gradient*, computed eagerly. Only the flat `_data`
buffer crosses the boundary; static block metadata is captured in a closure and
never differentiated.

XLA's view of `value_and_grad(loss)` is therefore: `normalize → pure_callback →
scalar` forward, `pure_callback → dA` backward — one opaque op each direction,
**independent of block count**.

## 4. Gate 1 — compile collapse (primary test)

Reuse the `jax_log_compiles` capture from `examples/profile_ctm_ad_wall_566.py`
to measure `vg_compile_s` for the **spike** arm at:

- fermionic D=2 χ=8, fermionic D=3 χ=12, dense D=3 χ=12.

The decisive signal needs no expensive baseline rerun: because the callback is
opaque, **spike-fermionic compile should equal spike-dense compile** (both
block-count-independent), at a few seconds. Contrast against the recorded
baseline (`profile_566_a100_Dsweep.json`: fermionic vg_cmp 206s → 2111s from
D2→D3, ~10×).

**GO criterion:** spike `vg_compile_s` at fermionic D=3 is **< 30s** AND the
fermionic D2→D3 compile ratio is **< 2×** (baseline ~10×). Equivalently:
spike-fermionic-compile ≈ spike-dense-compile.

**NO-GO ⇒ stop.** ~100–150 lines written, no adjoint, C-adjoint direction closed.

## 5. Gate 2 — AD-correctness (only if Gate 1 is GO)

Because the host adjoint *is* production-under-`disable_jit`, the real risk is
the **boundary plumbing**, not the math: does the `custom_vjp` + `pure_callback`
wrapper thread the cotangent, the `_data` ↔ `SymmetricTensor` reconstruction, and
the outer normalization correctly end to end?

Validate `grad(loss_spike)(A)` against production
`grad(value_and_grad(ctm_energy_implicit))` at **fermionic D=2 χ=8** (one
~3.4-min reference compile, affordable), element-wise on `A._data`.

**GO criterion:** max-abs gradient difference **< 1e-6**.

## 6. Scope, deferred work, and the honest value proposition

- **Deferred (explicitly NOT gated): warm runtime.** The spike's callback is
  *eager* host, so its warm step is **slower** than production's jitted warm
  step. The spike proves only the **architecture** — compile collapses and AD is
  correct across a host callback. A double-GO is a **green light to build the
  Phase-2 C kernel** (the thing that delivers a fast warm step without a compile
  wall), not a speedup in itself.
- **Who a GO helps immediately:** removing the compile wall directly helps **dev
  iteration, cold-start, and CI** (the persistent cache only saves unchanged
  code). Phase-2 C extends the benefit to warm runtime.
- **Out of scope:** multisite (>1×1) cells, U(1)-Sz arm (fermionic is the
  validated AD path and the cleanest compile-wall demonstrator), the C kernel,
  and any production integration.

## 7. Risks

- **`pure_callback` host round-trip:** transfers `A._data` device→host and the
  result host→device each call. A *warm* tax, deferred, but noted so Phase-2
  budgets for it.
- **`disable_jit` must genuinely avoid the fused-jaxpr wall inside the callback.**
  Eager op-by-op dispatch should (each op is a tiny cached compile, no giant
  fused graph), but Gate 1 is precisely what confirms it.
- **Eager runtime at D=3 inside the callback** may be slow (seconds), acceptable
  for a spike — runtime does not gate.
- **Floating-point reorder** eager-vs-jit is ~1e-12, well under the 1e-6 Gate-2
  threshold.

## 8. Cost and outcome

- **Gate 1:** ~100–150 lines + minutes of A100 runtime (spike compiles in
  seconds; baseline reused from JSON).
- **Gate 2:** +~100 lines + one ~3.4-min reference run.
- **Total:** ~1 day for a decisive GO/NO-GO on the entire C-adjoint direction.

**On double-GO:** open a follow-on design for Phase-2 — a Cython/C block-sparse
CTM forward + hand-written adjoint behind the same `custom_vjp` boundary, plus
production integration as a new `adjoint_method`/path. **On NO-GO at either
gate:** record the finding; the C-adjoint long-shot is closed, and the fallback
is to formalize the symmetric-CTM-AD NO-GO and pivot speedup effort to the dense
D≥3 path (runtime-bound, env warm-start + `chi_ramp` apply).

A100 environment per the project notes; harness building blocks:
`examples/profile_ctm_ad_wall_566.py` (site/gate construction, loss dispatcher,
compile capture) and `examples/profile_warm_dispatch_618.py`.
