# Spec: PyTorch eager backend for Tenax block-sparse calculations

**Status:** draft for review · **Author:** Claude Code (for @yingjerkao) · **Date:** 2026-09-17
**Scope decisions (locked by YJ):** (a) **v1 focus is block-sparse autodiff** — the eager backend exists to break the `SymmetricTensor` AD wall (see §2), *not* to be a general second backend; (b) **AD must work through the torch backend** in v1; (c) block-sparse/PEPS **AD** families are the parity targets, MPS defaults to the CPython path (D7), and *full* single-device feature parity is the eventual **direction**, not a v1 gate.

> Provenance of the many specific requirements below (which review round surfaced each) is collected in **Appendix A** rather than inline, so the body reads as a design. Every code citation was independently verified against the tree at head `756f9e0`.

---

## 1. Summary

**The point of the eager backend is block-sparse autodiff.** The wall Tenax is
hitting is not contraction throughput or MPS speed — it is the **`SymmetricTensor`
AD path**: block-sparse VJP trace+compile cost dominates iPEPS/fPEPS/CTM
optimization, and it is worst on the fermionic path (§2). A PyTorch **eager** backend
attacks that wall directly — eager AD has no per-shape trace/compile, so the
block-sparse VJP cost that XLA pays up front simply isn't incurred. So v1 is scoped to
**block-sparse AD parity** (SymmetricTensor autodiff through CTM/iPEPS/fPEPS/PESS on
one device), plus exactly the forward and dense machinery that path depends on. The
seam is built to generalize (a full second backend is the eventual direction), but v1
is measured by the AD wall, not by breadth.

Concretely: a second array backend so block-sparse (`SymmetricTensor`) and its
underlying dense (`DenseTensor`) calculations run on **PyTorch eager** with the **same
public API, the same numerical results within f64/c128 tolerance, and working
autodiff** (`torch.autograd`), while JAX remains the default and is selected per
process.

JAX's numerical **behavior** is unchanged; what changes is that the array namespace
is rewritten behind a pass-through (`jnp.foo` → `B.foo`, where `B` on JAX is a thin
wrapper that calls today's `jnp`). So this is **behavior-additive**, not
edit-free — ~86 files are touched — but no existing JAX result moves.

The bet that makes this tractable: Tenax's **symmetry logic is already
backend-neutral**. Charge fusion, conservation, canonicalization, and all block
metadata (`core/symmetry.py`, `core/index.py`, `contraction/blocksparse_plan.py`)
are pure NumPy — they never touch `jnp`. Each tensor holds its numeric data as a
**single flat buffer** (`SymmetricTensor._data`, `DenseTensor._data`) plus static
NumPy metadata. So the port is a **backend seam around array ops + linalg + AD +
control flow**, not a rewrite of the hard part.

What is genuinely JAX-specific and must be redesigned (not shimmed): the
`custom_vjp` AD primitives, `jax.lax` control flow, pytree registration, the
complex-cotangent convention, and PRNG threading. Sections 5–8 handle each.

### Non-goals (v1)
- Not replacing JAX; not a default switch.
- No **real** `torch.compile`/tracing model — eager is the point (dynamic block
  shapes become *free*, §7). A no-op `jit` **shim** is provided (§7), but it does not
  trace.
- No distributed/sharded torch path. This is a genuine *feature* gap, not just a perf
  knob: JAX multi-GPU dense CTM (`ctm_sharding`, and the sharding block embedded in
  the default `_jit_sweep`, §3) stays JAX-only in v1 — hence "single-device parity".
- No cuTensorNet/Pallas torch kernels, and no torch port of the
  `blocksparse_backend` contraction-kernel *selector* — torch uses
  `torch.einsum`/`opt_einsum`. Also a feature excluded from "parity".

**What v1 delivers, in scope order:**
1. **Block-sparse AD (the target):** `SymmetricTensor` autodiff through the CTM /
   iPEPS / fPEPS / PESS optimizers on one device under torch, tolerance-equal to JAX —
   this is the wall (§2) and the acceptance bar.
2. **The dependencies of (1):** the dense per-block kernels (block-sparse decomps run
   dense kernels per sector), the forward CTM/PEPS paths AD runs on, and the shared
   array/linalg/control/tree seam. Dense is a *dependency* of block-sparse AD, not a
   separate deliverable.
3. **Oracle-only families — no AD wall (D7):** MPS (DMRG/iDMRG/TDVP) *and* the
   forward-only RG algorithms TRG/HOTRG/GILT. These have **no `SymmetricTensor` AD, and
   no jit compile either** — they already run **eager** under JAX with a measured
   `0.000` compile cost (§2 / benchmarks), so torch has no wall to break for them.
   JAX/CPython stays their path; torch carries only an optional correctness oracle,
   never a v1 parity or performance target.

*Full* single-device parity across *every* algorithm is the eventual direction the
seam is built toward — not a v1 gate. v1 does **not** promise the multi-GPU or
specialized-kernel paths in the non-goals above, and does not gate on MPS throughput.

**No-AD-wall families default to JAX/CPython, not torch (Decision D7).** Two groups do
not hit the block-sparse AD wall and so are *not* torch targets:
- **MPS (DMRG/iDMRG/TDVP).** The fastest path is the existing NumPy/Cython accelerator —
  `accelerator="auto"` already routes CPU-symmetric DMRG to the `numpy_blockwise` sweep
  with Cython-BLAS hot loops (#226: 2.7–5.3× vs TeNPy), *bypassing both JAX-jit and
  torch*.
- **Forward-only RG (TRG/HOTRG/GILT).** These have **no AD** (no `custom_vjp`/`grad`)
  *and* **no `jax.jit`** — they run eager (`hotrg.py`: *"Runs eager … not [jittable]"*),
  so the measured compile cost is **`0.000`** and JAX already runs them fast on CPU and
  GPU (e.g. HOTRG χ=8 ≈0.45 s CPU / 0.08 s GPU; χ=20 ≈8 s CPU / 3.5 s GPU). There is
  no VJP compile and no forward compile — nothing for eager torch to improve.

For both groups the torch backend offers **only an optional correctness oracle** (the
shared block-sparse ops still run under `B`, so a small check is a cheap cross-backend
sanity test), never a performance or production path. The torch backend's value is
strictly the block-sparse / CTM / iPEPS / fPEPS / PESS **AD** workloads. Parity for the
no-wall families is therefore oracle-level (§10.3), not a throughput commitment.

---

## 2. Motivation — the block-sparse AD wall

**The wall.** Tenax's iPEPS/fPEPS/CTM optimization is bottlenecked by the
**block-sparse autodiff** path, not by contraction or MPS speed. The cost is the
`SymmetricTensor` VJP under XLA: differentiating the per-sector decompositions and the
CTM fixed point compiles a large backward graph whose **trace+compile time** dominates
wall-clock — measured on the fermionic path, where it is worst, the AD backward is a
"slow one-time block-sparse compile," and the recorded conclusion is explicit that
**"the wall is block-sparse VJPs, not contraction"** (the #565/#566 fermionic-AD
compile-cost investigations; cuTensorNet was NO-GO). This is exactly the cost a
**PyTorch eager** backend does not pay: eager reverse-mode records the tape as it runs,
with no per-shape trace and no XLA compile of the backward, so the block-sparse VJP is
executed, not compiled. That is the wall this backend is built to break.

`docs/guide/capabilities.md` already concedes that for large-D fermionic systems
"an eager PyTorch fermionic-PEPS code (YASTN/peps-torch) is the better tool today" —
this backend is how Tenax stops conceding that. Secondary benefits (kept in view but
not the driver):

- **Debuggable eager execution** — real stack traces, `pdb` inside a CTM sweep,
  dynamic block shapes with no recompile penalty.
- **A second oracle** — cross-backend parity tests become a standing correctness
  check on the symmetry/AD math itself (§10).
- **Ecosystem** — torch optimizers, `torch.compile` later, the PEPS-torch/YASTN
  interop surface.

---

## 3. Current coupling surface (grounded in the code)

From a full sweep of `src/tenax` (120 files); counts re-verified at head `756f9e0`:

| Layer | Where | Backend-coupled? |
|---|---|---|
| Charge/symmetry arithmetic | `core/symmetry.py`, `core/index.py` | **No** — pure NumPy. Reuse as-is. |
| Block metadata & planning | `contraction/blocksparse_plan.py` (`BlockContractPlan`) | **No** — touches only block metadata. |
| Tensor data buffer | `core/tensor.py:719-803` (`_init_flat_buffer`, `_get_block`) | **Yes**, but tiny: one `concatenate` + slice/reshape. |
| Dense tensor ops | `core/tensor.py:512-544` (`conj`, `transpose`, `norm`) | Yes, small. |
| Array namespace | `jnp.*` in **~86 of 120 files** | Yes, **pervasive, no single chokepoint**. |
| Dense linalg kernels | `linalg.py:59-100` (svd), `:1573/1781/2537/2670` (qr/eigh) | Yes; isolatable behind ~3 functions. |
| Block-sparse decomps | `linalg.py` `_truncated_svd_symmetric` (:243), `_qr_symmetric` (:1452), `_eigh_symmetric` (:1663) | Yes; per-sector loops over dense kernels. |
| Contraction execution | `contractor.py:234,356,1124-1135`; `blocksparse_plan.py:373` (`jnp.einsum`+`segment_sum`) | Yes; opt_einsum *path* portable, execution not (`torch.einsum` cannot consume a precomputed path — replay as tensordot/matmul). |
| **AD primitives** | 6 leaf `custom_vjp`: `_ad_primitives.py:229-719` (5) + `_lorentzian_eigh.py:81` (1), +`blocksparse_backend.py:149-177` | **Yes — redesign** (§5). |
| **dtype introspection** | `jnp.iscomplexobj/issubdtype/result_type/finfo/complexfloating` (~18 sites) — incl. default Arnoldi `_arnoldi.py:37`, 2×2 projector, adjoint GMRES `_gmres_eager.py:117` | **Yes — backend predicates** (§4.2/§8). |
| **Differentiation-state checks** | `isinstance(x, jax.core.Tracer)` ×~18 across **7 files** (`core/tensor.py:1043`, `linalg.py:274/2045`, `_ctm_projector.py` ×6, `_ctm_tensor_projector_2x2.py:1001`, `_ctm_tensor_energy.py:130`, `contractor.py:897`, `cutensornet_backend.py:64`) | **Yes — backend predicate** (§4.2). |
| **Host reads** | truncation: `np.array(s_q)`/`np.asarray(block)` (`linalg.py:471/1112/1359`, `core/tensor.py:1044`); **AD-target diagnostics**: `fixed_point` backward `jax.device_get` (`_ctm_energy_ad.py:1694-1699`, gated `:1711/1714`, `:99`); **oracle-only families**: dense iDMRG `idmrg.py:976-977` + `:1071-1085`, GILT `gilt.py:209/222/256/265/267/274/284/318` | **Yes** — truncation → device-native (§4.3); AD-target diagnostics → `to_numpy`, **skipped under `torch.func`** (§4.3); oracle-only sites **CPU-only by design** (§10.3), not migrated for v1. |
| **Control flow** | `lax` `while_loop`×29, `scan`×22, `fori_loop`×11, `cond`×0, `stop_gradient`×58; `lax.map` (`_ctm_chunked_absorb.py`); union across 27 files | **Yes — redesign** (§7). |
| **Krylov / triangular solvers** | `_gmres_lax.py:171` (`solve_triangular`), `:299-336`; `_metric_precond.py:164`, `ad_utils.py:882` (`jax.scipy … gmres`); **`_krylov_bicgstab` (default `adjoint_solver`), `_ctm_tensor_c4v_reference_ad.py:166`** | **Yes — backend solvers incl. bicgstab** (§5.4). |
| **DMRG truncation ops** | `jax.lax.top_k`/`jax.nn.one_hot` (`_padded_linalg.py:128/142`), reached by `accelerator="auto"` → `_jit_sweep` | **Yes — `ArrayOps.top_k`/`one_hot`** (§4.2). |
| **Optimizer** | `optax` chains in `_ipeps_optimize_shared.py:112-136` + `pess_optimize.py:468/775` + `ipeps_optimize_root_implicit.py:637`; `optimizer.update`/`apply_updates` | **Yes — optimizer seam, all Optax users** (§5.6). |
| **Euclidean-grad conjugation** | `jax.tree.map(jnp.conj, grads)` (`_ipeps_optimize_shared.py:265`), applied at 6 sites (#957) | **Yes — convention-guard, identity under torch** (§5.3). |
| `jnp.linalg.inv` | root-implicit CTM AD (`_ctm_root_implicit_asym.py:240/241/1006`, `_ctm_root_implicit_multisite.py:648`) | Yes — `B.linalg.inv` (trivial, unenumerated). |
| `jnp.kron` (18) | default iDMRG + iPEPS + root-implicit | Yes — `B.kron` (trivial). |
| `static_argnums`/`static_argnames` on plain `jax.jit` (17) | `tdvp.py:103/126`, `ipeps_bp_gauge.py:1112`, `_ctm_energy_ad.py:1272/1308/1346/1531`, … | Yes — the identity-`jit` shim must **accept and ignore** these kwargs (§7). |
| **Pytree registration** | `core/tensor.py:441,624`; `stacked_tensor.py:41`; `_padded_block_array.py:229`; `pess.py:197,252` | **Yes — reinterpret** (§6). |
| PRNG | `jax.random`/PRNGKey in 17 files; **plus** transform-time randomness (§8) | Yes; key-threading → generator. |
| Global x64 | `__init__.py:45`; `jnp.float64` literals in factories | Yes; small but global. |
| Multi-GPU sharding | `_jit_sweep.py:769-806` (mesh/`device_put`), `ctm_sharding.py`, `_ctm_tensor_convergence.py:337` | **JAX-only** (non-goal) — but embedded in the default sweep, so the torch path must branch around it, not just skip a module. |
| TRG/HOTRG/GILT (forward-only RG) | `trg.py`/`hotrg.py`/`gilt.py` — **no AD, no `jax.jit`** (eager; `0.000` compile); `gilt.py:218/221/255` `jnp.linalg` | **No AD wall — JAX/CPython stays; torch oracle-only** (D7, §10.3). |
| GPU workarounds | `linalg.py:59-100` (cuSOLVER), `_einsum_compat.py` (cuBLASLt) | JAX-only; **drop** for torch. |

Note the existing `contraction/blocksparse_backend.py` seam selects **contraction
kernels** (stacked-JAX / cuTensorNet / per-block), all `jnp`. It is **not** an array
backend and cannot be reused as the torch seam — but its documented data contract
("the flat `_data` buffer + static block metadata is what every backend consumes")
is exactly the insertion point we build on.

---

## 4. Architecture: the `tenax.backend` seam

Introduce one new package, `src/tenax/backend/`, exposing a resolved **array
namespace** `B` and a small set of protocols. Every module that does numeric work
imports `B` instead of `jnp`:

```python
from tenax.backend import B          # resolved array namespace (jax or torch)
x = B.concatenate([...]); U, S, Vh = B.svd(m)
```

Backend resolution is process-global and explicit:
```python
import tenax
tenax.set_backend("torch")   # or "jax" (default); reads TENAX_BACKEND env as fallback
```

**`B` must be a live dispatch proxy, not a rebindable name.** `import tenax` eagerly
pulls in the numeric modules (`tenax/__init__.py:61-92` imports
`contraction.contractor`, `core.tensor`, `linalg`, `network.*` at load), so by the
time `set_backend("torch")` runs, every one has *already* done
`from tenax.backend import B`. If the setter merely **rebinds** `tenax.backend.B`,
those modules keep the old reference and silently stay on JAX. So `B` is a **stable
proxy whose *target* `set_backend` mutates in place**; the alternative (dereference
`tenax.backend.B.<op>` on every use) is rejected as per-op indirection and an
easy-to-violate rule over ~86 files. `set_backend` is valid **only before any tensor
is allocated**; a mid-process switch with live old-backend tensors is out of scope
(guarded with a clear error). To keep the per-op proxy indirection off the JAX hot
path, the proxy resolves its bound method table **once** at `set_backend` time.

### 4.1 Rollback & the seam-boundary invariant

Because `B` is load-bearing for the **default JAX path** from Phase 0 onward, a
`JaxBackend` bug would hit every existing JAX user, not just torch. Three things bound
that blast radius and make the migration safe to land incrementally:

- **`JaxBackend` is a thin pass-through with its own op-parity test** against raw
  `jnp`/`lax` — so "does `B.foo` equal `jnp.foo`?" is a standing check, and
  `TENAX_BACKEND=jax` (the default) keeps the proxy dispatching to that unchanged
  layer.
- **A CI grep-gate enforces the seam boundary**: no `import jax.numpy` / bare `jnp.`
  / `lax.` outside `src/tenax/backend/` (allow-list the few genuinely JAX-only
  modules), and no bare backend-array **method** calls torch lacks — `.at[` and
  `.astype(` (§4.3). This both prevents a *half-migrated* state where a
  not-yet-ported file calls raw `jnp` (or a JAX-only tensor method) on a torch
  tensor, and defines "migrated" mechanically.
- **Opt-out**: the whole effort is behind `set_backend`; reverting to raw `jnp` is a
  one-line default, and any phase can be shipped with the torch path dormant.

### 4.2 Layered design

```
        public API (unchanged): SymmetricTensor, contract, dmrg, ipeps, ...
                              │
        ┌─────────────────────┼──────────────────────────┐
   symmetry/metadata     backend seam (NEW)         algorithms
   (pure NumPy,          ┌──────────────┐           (import B + control + ad)
    reused as-is)        │ B: ArrayOps  │
                         │ linalg       │
                         │ ad (§5)      │
                         │ control (§7) │
                         │ random (§8)  │
                         └──────┬───────┘
                        ┌───────┴────────┐
                  JaxBackend         TorchBackend
                  (wraps jnp/lax,    (torch.*, python loops,
                   custom_vjp)        autograd.Function)
```

### 4.3 The `ArrayOps` protocol (Decision D1)

A **custom, explicit Protocol** enumerating the ~40 array ops Tenax actually uses
(`concatenate, reshape, transpose, conj, einsum, tensordot, stack, segment_sum,
where, pad, astype, zeros, top_k, one_hot, kron, ...`) plus the decomposition entry
points, the dtype/tracing predicates below, and functional indexed-updates. Two
implementations: `JaxBackend` (thin wrappers over today's `jnp`) and `TorchBackend`.
We reuse `array-api-compat` for the trivial elementwise subset, but the Protocol is
the contract — the standard doesn't cover `einsum`-with-paths, `segment_sum`,
algorithm-selected SVD, or complex dtypes portably.

**Functional indexed updates.** Tenax uses `x.at[idx].set/add/multiply(...)` **137
times across 25 files** (only `.set`/`.add`/`.multiply` — no exotic `.at` variants),
including `SymmetricTensor.todense()` (`core/tensor.py:1111`). Torch has no `.at`,
and in-place assignment there breaks leaf-autograd and — critically — the
**saved-tensor version counter** (any in-place mutation of a tensor saved for
backward raises "modified by an in-place operation"). So `ArrayOps` exposes
functional `index_set/index_add/index_mul` (`.at[...]` under JAX; out-of-place
`index_add`/`scatter`/masked `where` under torch), and the whole through-torch AD
path must stay mutation-free of saved tensors — migrating `.at[]` out-of-place is
necessary but not on its own sufficient.

**`.astype(dtype)` is a method the migration must not miss.** `astype` is in the op
list above, but the codebase overwhelmingly calls it as a **method on a backend
array** — `lambdas[i].astype(dtype)`, `T_u.astype(dtype)` (**31 sites in
`pess.py`**, e.g. `:438-440/563-564/707-717`), plus `_ad_primitives.py:371` and
`_ctm_energy_ad.py:1477/1485/1498`. Native torch tensors have **no `.astype`** (they
spell it `.to(dtype)`), so exposing `B.astype(x, dtype)` as a *function* does not
cover these method calls, and the §4.1 grep-gate — which keys on `jnp`/`lax` tokens
— will not flag `x.astype(`. So the migration must **audit and rewrite
backend-array `.astype(...)` sites to `B.astype(...)`**, and the seam-boundary gate
must additionally flag bare `.astype(`. (NumPy-array `.astype` — e.g. on a host gate
before `jnp.asarray`, `pess.py:80/970` — is out of scope; the audit is per-site and
distinguishes the two.)

**`top_k` / `one_hot` are load-bearing on the default DMRG path.**
`accelerator="auto"` (`dmrg.py:176`) routes dense-CPU and all GPU/TPU runs through
`_jit_sweep`, whose truncation uses `jax.lax.top_k` (`_padded_linalg.py:128`) and
`jax.nn.one_hot` (`:142`). Neither is a `lax` loop or a `custom_vjp`, so an identity
`jit` (§7) alone still feeds torch tensors into a JAX API. They map to `torch.topk` /
`torch.nn.functional.one_hot` — noting `one_hot` returns `int64` and takes no
`dtype`/`axis` arg, so a cast is needed.

**Differentiation-state detection is a backend predicate — and it must be
functorch-aware, not just `requires_grad`.** ~18 sites branch on
`isinstance(x, jax.core.Tracer)` to pick an AD-safe path over a concrete/validating
one (e.g. `core/tensor.py:1043` guards `np.asarray(data)`). A torch tensor with
`requires_grad=True` is never a JAX tracer, so a naive port takes the wrong branch.
But the obvious replacement `torch.is_grad_enabled() and x.requires_grad` is **also
wrong under `torch.func`**: inside `torch.func.grad`/`vmap` — the actual
through-torch AD path — grad tracking lives at the functorch level and the wrapper
tensor's `.requires_grad` is commonly `False`. So the predicate must additionally
probe functorch state (`torch._C._functorch.*` — private/unstable API, pinned to a
torch version floor, §D5). Two further cautions: (a) these sites conflate "AD-safe
branch?" with "no Python data-dependent branch?" — torch eager can `.item()`/branch
where a JAX tracer cannot, so the ~18 migrations need **per-site semantic review, not
a mechanical rename**; (b) host-convertibility is a *different* predicate (next).

**Host-convertibility is a separate concern, and device-native truncation is
mandatory under `torch.func`.** The AD-state predicate answers "AD-safe branch?", not
"can I `np.asarray` this?". The block-sparse decomposition host-reads its data —
`np.array(s_q)` (`linalg.py:471`), `np.asarray(block)` (`:1112/:1359`),
`np.asarray(data)` (`core/tensor.py:1044`). A `to_numpy(x)` helper
(`x.detach().cpu().numpy()` under torch; `np.asarray` under JAX) fixes the **eager**
`.backward()` case for CUDA tensors and `no_grad` grad-tensors. **But it does not fix
the transformed forward**: under `torch.func.grad`/`vmap` the inputs are
functorch-wrapped, and `.numpy()` — even after `.detach().cpu()` — **raises** on a
functorch-wrapped tensor. So for every host-read that sits inside a
`torch.func`-transformed forward (the SVD/eigh truncation is exactly this), a
**device-native truncation** (on-device `topk`/`sort`, no host transfer) is
**required**, not merely preferred. `to_numpy` remains for genuinely host-only,
non-transformed sites (e.g. final diagnostics).

**Diagnostic host-reads on the AD path are a third case — skip, don't just
convert.** The default `adjoint_method="fixed_point"` iPEPS backward host-syncs
convergence/residual scalars **unconditionally** via `jax.device_get(...)` —
`diverged`/`converged`/`n_iter`/`abs_resid`/`b_norm` at
`_ctm_energy_ad.py:1694-1699` (plus the `_F3_DIAG_COMPUTE_NORMS`-gated per-leaf
norms at `:1711/1714`, and `:99`). `jax.device_get` on a non-JAX value invokes
`__array__`, which a torch **CUDA** tensor rejects — so the CUDA iPEPS gradient path
crashes *after* a correct solve, on a value that never enters the gradient. Because
these are diagnostics written to a module global (`_F3_LAST_DIAGNOSTICS`), not the
returned cotangent, the fix is **not** device-native computation but routing through
a `backend` scalar host-read that is **skippable under a `torch.func`-transformed
backward** (where `to_numpy` itself raises on the functorch-wrapped scalar): compute
the diagnostic on the eager `.backward()` path, no-op it under transform. This site
is on an **AD-target** family (not oracle-only), so it is in scope for v1 — unlike
the iDMRG/GILT host-reads (§10.3).

**dtype-introspection predicates.** Branching on dtype —
`jnp.iscomplexobj/issubdtype/result_type/finfo/complexfloating` (~18 sites) — sits at
the core of the **default** Arnoldi/Lanczos eigensolver (`_arnoldi.py:37`), the 2×2
CTM projector (`_ctm_tensor_projector_2x2.py:100`, `ipeps_optimize.py:459`), and the
adjoint GMRES (`_gmres_eager.py:117`). Under torch a tensor's `.dtype` is a
`torch.dtype`, so `jnp.iscomplexobj(torch_tensor)` / `jnp.issubdtype(torch.complex128,
…)` return wrong or raise, mis-selecting the complex-vs-real branch on a default run.
`ArrayOps` therefore exposes `is_complex / is_floating / result_type / finfo`
(torch: `torch.is_complex`, `torch.is_floating_point`, `torch.result_type` — **binary,
vs jnp's variadic, so the wrapper folds**, `torch.finfo`). `issubdtype` has no torch
analog and is reduced to the `is_complex`/`is_floating` predicates at each site.

### 4.4 Tensor data contract

`SymmetricTensor`/`DenseTensor` keep their shape exactly. `_data` becomes "an array
of the active backend." `_init_flat_buffer`'s `jnp.concatenate` and `_get_block`'s
slice/reshape route through `B`. The static metadata (`_block_keys`, `_block_shapes`,
`_block_offsets`, `_indices`) is already NumPy and unchanged.

---

## 5. AD bridge (the spine)

This is where parity + through-torch-AD is won or lost. The principle: **share the
VJP *math*, wrap it per backend** — but note the wrapping is more than a thin shim
(per-primitive `nondiff_argnums`, hidden residuals, vmap rules, convention).

### 5.1 The split (same on both backends)
Everything composed from `B` ops is differentiable *automatically* — by XLA tracing
under JAX, by the eager tape under torch. **Only the linalg primitives need custom
gradients** (SVD/QR/eigh backward has degenerate-spectrum and truncation subtleties):
**6 leaf decomposition primitives across two files** — `truncated_svd_ad`,
`truncated_svd_ad_vh_only`, `regularized_svd`, `regularized_qr`, `regularized_eigh`
(`_ad_primitives.py`), **plus `truncated_eigh_regularized` (`_lorentzian_eigh.py:81`)**,
which `_ctm_projector.py:1146` invokes on `projector_backward="lorentzian"`. The
`*_converge`/`f(params_data_tuple)` `custom_vjp`s (`ad_utils.py`, `_ctm_energy_ad.py`,
`_split_ctm_energy_ad.py`, `_ctm_honeycomb_ad.py`, and the C4v-reference primitive)
are a **different tier** — the fixed-point adjoint family §5.4 owns.

### 5.2 Refactor each primitive to backend-neutral fwd/bwd
```python
def _svd_fwd(A, k):                 # pure B-ops
    (U, S, Vh) = B.svd(A); ... truncate ...
    return (s_trunc, Vh_trunc), Residuals(U, S, Vh, ...)   # residuals declared explicitly

def _svd_bwd(res, dU, dS, dVh):     # pure B-ops (F-matrix / gauge-fixed formula)
    return dA
```
A single `backend.ad.custom_vjp(fwd, bwd, nondiff_argnums=())` factory hides the
wrapper. But the wrapper carries **four** non-trivial contracts, each of which a naive
`autograd.Function` gets wrong:

1. **Static/nondifferentiable positions.** `truncated_svd_ad`,
   `truncated_svd_ad_vh_only` (`_ad_primitives.py:229,470`) and
   `truncated_eigh_regularized` (`_lorentzian_eigh.py:81`) are
   `nondiff_argnums=(1,)` — the rank `chi` controls slicing and output *shapes* and
   must never be traced. The factory forwards `nondiff_argnums` to `jax.custom_vjp`
   under JAX; under torch it binds those positions as static config (`ctx`), never as
   `.apply` tensor leaves. `backward` returns `None` in those slots. **This is a
   property of the primitives, so it belongs in Phase 2 with them, not later.**
2. **`torch.func`-compatibility.** A plain `Function` (`forward`/`backward` +
   `save_for_backward`) works under eager `.backward()` but **raises** the moment a
   primitive is reached through `backend.ad.grad`/`vjp`/`vmap` (§5.5) — the actual
   through-torch AD path. `torch.func` requires the split
   `forward` + `setup_context(ctx, inputs, output)` form, and `vmap` needs
   `generate_vmap_rule=True` or an explicit `vmap` staticmethod. Mandatory for all six
   primitives **and the fixed-point combinator** (§5.4).
3. **Hidden residuals — recompute from saved *inputs*, do not `mark_non_differentiable`.**
   `setup_context(ctx, inputs, output)` sees only inputs and the *returned* outputs,
   but the fwd rules save residuals absent from the public output:
   `_truncated_svd_ad_vh_only_fwd` returns only `(s, Vh)` while saving
   `(U_full, s_full, Vh_full, M, k)` (`_ad_primitives.py:489`), and
   `_truncated_eigh_regularized_fwd` returns `(w[:k], v[:,:k])` while saving the full
   eigensystem (`_lorentzian_eigh.py:92`). The tempting fix — return them as extra
   outputs tagged `ctx.mark_non_differentiable` — is **wrong for double-backward**:
   marking `U_full`/`s_full`/`Vh_full` non-differentiable makes the *first* backward
   treat their dependence on the input as constant, so the Hessian terms through
   SVD/eigh vanish — which silently breaks `compute_excitations`/HVP, the very
   second-order path §5.4 requires. So the contract is: **`setup_context` saves the
   raw *inputs* on `ctx`, and `backward` recomputes the decomposition from them**
   inside the recorded (differentiable) graph — or the primitive supplies a dedicated
   double-backward rule. A `gradgrad`/`torch.autograd.gradcheck(...,
   check_double_backward=True)` test (§10) guards it; `mark_non_differentiable` is
   only acceptable for residuals that are genuinely constant w.r.t. the input.
4. **`regularized_qr`'s backward is not pure `B`-ops — it calls `jax.vjp`.** Unlike
   the SVD/eigh backwards (hand-written F-matrix / gauge-fixed formulas),
   `_regularized_qr_bwd` delegates to JAX: it floors the diagonal to build
   `R_reg`, then returns `jax.vjp(_qr_tuple, Q @ R_reg)((dQ, dR))`
   (`_ad_primitives.py:650`). The `jax.vjp` mentions at `:618/627/634` are parity
   *comments*; `:650` is the **live call** — an earlier review round wrongly
   dismissed this finding as comment-only, corrected here. So this primitive's
   backward depends on `backend.ad.vjp` (VJP of a user function), which §5.5
   otherwise defers to Phase 3a. Two resolutions, either acceptable: **(a)** make
   `backend.ad.vjp` available for this primitive in **Phase 2** (a scoped
   exception — the primitive's *internal* vjp is a Phase-2 dependency even though
   the ~76 *algorithm-level* vjp sites migrate in 3a), or **(b, preferred)** give
   `regularized_qr` a hand-written backend-neutral QR backward (the standard
   `Q̄`/`R̄` triangular-solve formula) so it needs no `jax.vjp` at all and stays
   self-contained like the other five. Either way the §10 six-primitive gradient
   suite cannot go green in Phase 2 until this is closed.
5. **Convention** (§5.3) and **double-differentiability** (§5.4) — below.

### 5.3 ⚠️ Complex-cotangent convention — the highest-risk item
JAX and PyTorch use **different conjugation conventions** for complex gradients.
Tenax's VJP formulas are written to JAX's convention (cotangents pair *unconjugated*;
the objective reads `Re Σ g·dz`). PyTorch's autograd uses the Wirtinger/conjugate
convention (`g_torch = conj(g_jax)`). Dropping the shared `_bwd` into a torch
`Function` verbatim yields **silently wrong** complex gradients (right magnitude,
wrong phase) — no crash, just a bad optimizer direction. The per-primitive mitigation
is correct and sufficient: the `backend.ad` factory **conjugates incoming cotangents
and outgoing grads at the torch boundary** so `_bwd` always sees JAX-convention
inputs.

**But the convention must be reconciled at exactly ONE layer — and the codebase
already has one.** `_ipeps_optimize_shared.py:265` applies
`_euclidean_grads(grads) = jax.tree.map(jnp.conj, grads)` at **6** gradient-production
sites (`ipeps_optimize.py:1006/1762/2188/…`, `ipeps_optimize_root_implicit.py:540`) —
the #957 fix, because a JAX cotangent is `g = conj(∇E)` and steepest descent is
`-conj(g)`; forgetting it made the optimizer **ascend** the imaginary coordinates.
With §5.3's boundary conjugation, the torch end-to-end gradient handed to
`value_and_grad` is *already* the Euclidean `∇E`. If `_euclidean_grads` (`jnp.conj`)
is then kept on the torch path, it **double-conjugates back to `conj(∇E)` and
reproduces #957 on torch.** So `_euclidean_grads` must be **convention-guarded**:
identity under torch, `jnp.conj` under JAX. This is the single most subtle correctness
item in the port.

**Parity test.** Comparing raw `jax.grad` vs `torch.func.grad` is invalid — a correct
wrapper leaves them conjugated relative to each other. Compare **directional
derivatives** against a finite-difference reference, which is convention-free — but
the *pairing is per-backend*: JAX uses `dE = Re Σ g·v` (unconjugated), torch uses
`dE = Re Σ conj(g)·v` (Hermitian), since `g_torch = conj(g_jax)`. A single shared
`⟨g,v⟩` passes one backend and **rejects the correct implementation** of the other.
And the case must use **complex parameters through a full optimizer step** — a
real-tensor case is a no-op for `conj` and would hide the `_euclidean_grads`
double-conjugation entirely.

### 5.4 Fixed-point AD for CTM
Torch eager would unroll the whole CTM convergence tape → memory blowup. Parity
requires the **same implicit/fixed-point adjoint** JAX uses (`_ctm_energy_ad.py`,
root-implicit modules): backward solves the adjoint linear system at the converged
environment. Under torch this is an `autograd.Function` whose `backward` runs the
adjoint solve. Combinator signature:

```
fixed_point(step, params, init, adjoint)   # params = flattened tensors, differentiable
```

**The `.apply()` boundary carries only bare tensor leaves — for inputs *and*
outputs.** `torch.autograd.Function` registers as differentiable only **top-level
tensor positional arguments and returns**; tensors nested in a tuple/list/dict, or
Tenax `Tensor` objects, get no cotangents. So `control.fixed_point`:
- passes `params` as **splatted leaves** `Function.apply(leaf0, leaf1, …)` (a closure
  captured `step` would leave the iPEPS tensors with no gradient), and
- returns the environment as **flattened buffer leaves**, reconstructing the public
  `(C, T)` / env tree *outside* `.apply()` — the C4v-reference primitive returns
  `(C, T)` as `Tensor` objects (`_ctm_tensor_c4v_reference_ad.py:304-334`), so
  `g_c`/`g_t` must flow in through bare-tensor returns.
The pytree `treedef` (§6) rides along as static aux and is rebuilt inside `forward`.
JAX has neither constraint — this is a torch-side wrapper detail, invisible above the
seam.

**The fixed-point `Function` needs the same `torch.func` contract *and*
double-differentiability.** It is reached through `backend.ad.value_and_grad`
(`torch.func.grad_and_value`), so it needs the §5.2 `setup_context`/vmap form.
Moreover its `backward` computes `vjp(step)` at the fixed point: inside a `torch.func`
transform that vjp must be taken with `torch.func.vjp` (nesting `torch.autograd.grad`
inside functorch does **not** compose), and the backward must **not** be
`@once_differentiable` — `compute_excitations` and Hessian-vector products need
`create_graph`-safe (second-order) backwards.

**Solvers are part of the seam — and BiCGSTAB is the default.** The adjoint solve
calls JAX directly: `_gmres_lax.py:171` `solve_triangular`, `:299-336` GMRES machinery,
`ad_utils.py:882` and `_metric_precond.py:164` `jax.scipy … gmres`, and — crucially —
`CTMConfig.adjoint_solver` **defaults to `"bicgstab"`** (`ipeps_config.py:152`), which
the C4v-reference backward calls directly (`_ctm_tensor_c4v_reference_ad.py:166`)
before any GMRES fallback. So the seam adds `backend.linalg.gmres`,
`solve_triangular`, **and `bicgstab`** (matching the JAX default so both backends
solve the same system); a GMRES-only seam would fail the default adjoint path.

**The fixed-point family is more than `_ctm_energy_ad`.** The supported
`ctm_ad_mode="c4v_reference"` path calls the standalone
`ctm_tensor_c4v_reference_converge_reduced` (`_ctm_tensor_c4v_reference_ad.py:304`)
from `ipeps_optimize.py:981`; it routes through `control.fixed_point` on the same
contract with its own torch gradient test.

### 5.5 Reverse-mode transforms belong in the seam too
The algorithms call the JAX transform APIs *directly* — `jax.vjp` (44),
`jax.value_and_grad` (10, e.g. `ipeps_optimize.py:1761`), `jax.grad` (11), `jax.vmap`
(11): ~76 sites. Autodiff of `B` ops does not subsume these, so `backend.ad` exposes
the transforms:

| transform | JAX | Torch |
|---|---|---|
| `vjp(f, *primals)` | `jax.vjp` | `torch.func.vjp` (or `autograd.grad` over a taped forward) |
| `grad(f)` / `value_and_grad(f)` | `jax.grad` / `jax.value_and_grad` | `torch.func.grad` / `grad_and_value` (with the return-order adapter below) |
| `vmap(f)` | `jax.vmap` | `torch.func.vmap` |

**`value_and_grad` is an adapter, not an alias.** `jax.value_and_grad(f)` returns
`(value, grads)`; `torch.func.grad_and_value(f)` returns them **swapped**
`(grads, value)`, and the `has_aux` nesting differs (`jax`: `(value, aux), grads`;
`torch.func`: `grads, (value, aux)`). Raw exposure at
`energy_val, grads = value_and_grad(...)(params)` (`ipeps_optimize.py:1761`) would bind
the **gradient tree to `energy_val`** — a silent, catastrophic swap. So the torch
adapter calls `grad_and_value` and re-orders to JAX's `(value, grads)` and aux layout.
A parity test asserts the tuple order/aux structure, not just the values. Migrating the
~76 sites is a Phase-3a deliverable.

### 5.6 The optimizer step is backend-specific too
Gradient parity is not "full parity" — a real run continues into the *update* step,
built on **optax** (JAX-only): `_build_optimizer` (`_ipeps_optimize_shared.py:112`)
returns `optax.adam`/`scale_by_lbfgs`/`clip_by_global_norm` chains, and the loop calls
`optimizer.update`/`optax.apply_updates`. The seam adds a **backend optimizer**: optax
under JAX; a functional reimplementation over `backend.tree` leaves under torch.

**Contract: `update(grads, state, params) → (direction, state)`; the default L-BFGS
must be functional.** Tenax does not let the optimizer own the step — it takes the
returned `updates` as a **search direction** (`ipeps_optimize.py:2155`), runs its
**own** line search (Hager-Zhang / Armijo backtracking), applies it **functionally**
(`_normalize_params(_tree_add(params, _tree_scale(direction, alpha)))`), plus the #328
tangent projection and metric/CG variants. `torch.optim.LBFGS` breaks all of that
(owns+mutates params; drives its own closure line search; returns no direction). So
`gs_optimizer="lbfgs"` maps to a **functional two-loop L-BFGS** (Tenax already
hand-rolls one at `ipeps_optimize.py:2148` for the metric path). It is **not a verbatim
port**: it must (a) consume the Euclidean `∇E`, *not* re-conjugate (per §5.3), (b) use
`torch.vdot`/`Re(·)` for the Hermitian `s·y`, `y·r` curvature pairs
(`lbfgs_two_loop` uses `jnp.vdot`, `_metric_precond.py:236/246`), and (c) replicate the
`optax.scale_by_lbfgs` specifics that set the §10 tolerance — initial scaling
`γ=⟨s,y⟩/⟨y,y⟩`, the `s·y≤0` curvature-pair skip, and the sign convention Tenax's line
search consumes.

**Every direct Optax user routes through the seam**, not just `_build_optimizer`: the
PESS optimizers `optimize_pess_ad` / `optimize_pess_3site_multisite_ad`
(`pess_optimize.py:468/775`) and the root-implicit optimizer
(`ipeps_optimize_root_implicit.py:637`). §10 exercises one iPEPS **and** one PESS
default-mode update step.

---

## 6. Trees: a backend tree protocol, not just tensor flatten

Instance `flatten()/unflatten()` on the tensor classes is **necessary but not
sufficient**. The algorithms call the JAX tree API directly — **186
`tree_map`/`tree_leaves`/`tree_structure`/`tree_unflatten` uses** (e.g.
`_ctm_energy_ad.py:1212-1331`) — and several **standalone registered containers are
not tensor instances**: `IPESSState` (`pess.py:197`), `StackedTensor`
(`stacked_tensor.py:41`), `PaddedBlockArray` (`_padded_block_array.py:229`).

So the seam defines a **`backend.tree` protocol** — `map`, `leaves`, `structure`,
`flatten`, `unflatten`:
- **JAX:** `jax.tree_util.*` (unchanged).
- **Torch:** `torch.utils._pytree` (note: a private, `_`-prefixed, version-unstable
  module — pinned to the torch floor, §D5) or a small in-house registry. **Every
  custom container** (`SymmetricTensor`, `DenseTensor`, `IPESSState`, `StackedTensor`,
  `PaddedBlockArray`) registers with *both* systems, flattening to array leaves +
  static aux; the 186 sites migrate to `backend.tree.*` in Phase 3a.
- The `DenseTensor.tree_unflatten` `object()`-probe bypass (`tensor.py:491-493`) stays
  JAX-only; under torch, `torch.autograd`/`torch.func` differentiate w.r.t. the
  `_data` leaves natively.

---

## 7. Control flow

`lax.while_loop/scan/fori_loop/cond/map/stop_gradient` have no torch-*eager* analog
because eager *is* Python control flow. `backend.control` gives **two torch lowerings
behind one interface**: eager (Python loops) for v1, and a **compile-forward** one
(torch higher-order ops) a future `torch.compile` path selects unchanged.

| Combinator | JAX (`lax`) | Torch eager (v1) | Torch compile-forward (future) |
|---|---|---|---|
| `while_loop(cond, body, init)` | `lax.while_loop` | Python `while` | `torch.while_loop` |
| `scan(f, init, xs)` | `lax.scan` | Python `for`, stack outputs | `torch.while_loop` w/ index carry |
| `fori_loop(lo, hi, body, init)` | `lax.fori_loop` | Python `for` | `torch.while_loop` w/ counter |
| `map(f, xs, batch_size=…)` | `lax.map` (chunked; `_ctm_chunked_absorb.py`) | Python chunk loop + stack | HOP `scan`/chunk |
| `cond(p, t, f, x)` | `lax.cond` (0 real uses today) | Python `if` | `torch.cond` |
| `stop_gradient(x)` | `lax.stop_gradient` | `tree.map(detach, x)` (tree-aware) | same |
| `fixed_point(step, params, init, adjoint)` | `custom_vjp` + adjoint | `autograd.Function` (§5.4) + adjoint | same `Function` (opaque to Dynamo) |

**`torch.while_loop`/`torch.cond` are forward/compile-capture ops with no/limited
autograd** — `torch.while_loop` does not support backward. That is fine *because* every
differentiated loop routes through `fixed_point` (§5.4); the HOP lowerings are only for
the non-differentiated-inner, compiled-forward path. Both HOPs impose the same
discipline `lax` already requires (pure functional bodies, fixed carry structure), so
one functional-carry body serves the JAX compiled path, the torch eager path, and a
future torch compiled path — routing between lowerings is a single swap point.

**`jit`/`checkpoint` are backend ops too.** DMRG/iDMRG/TDVP flow through unconditional
`jax.jit` (`dmrg.py:1202`, `idmrg.py:418`, `tdvp.py:103`) + `jax.checkpoint`
decorators. So `backend.control` exposes `jit` (JAX: `jax.jit`; torch: an
**identity/eager** shim that must **accept and ignore `static_argnums`/`static_argnames`
kwargs**, 17 sites, or it raises `TypeError`) and `checkpoint` (JAX: `jax.checkpoint`;
torch: `torch.utils.checkpoint`). Two subtleties:
- **The identity-`jit` is necessary but not sufficient for DMRG**: the default
  `_jit_sweep` route calls `jax.lax.top_k`/`jax.nn.one_hot` directly, which must be
  `ArrayOps` ops (§4.3).
- **The wrappers must bind at CALL time, not import time.** `_matvec_jit =
  jax.jit(...)` runs at **module scope** (`dmrg.py:1202`, `idmrg.py:418`,
  `tdvp.py:126`), i.e. when the module is first imported. A user who does
  `import tenax.algorithms.dmrg` before `set_backend("torch")` therefore has a
  JAX-bound wrapper already installed, and `_ad_primitives`' `custom_vjp`
  factories can be captured the same way -- permanently. That sequence
  allocates **no tensors**, so §4.1's "valid only before any tensor is
  allocated" guard does **not** catch it. So `backend.control.jit` /
  `checkpoint` / `custom_vjp` must return a thin wrapper that resolves the
  lowering **on each call** (cache it per-backend-generation, so the JAX hot
  path still pays one dict lookup, not a re-trace), and `set_backend` must
  additionally refuse to switch once any backend-bound wrapper has been
  created. A decorator that closes over the backend live at import is the same
  class of bug as rebinding `B` instead of mutating the proxy.
- **`stop_gradient` and `checkpoint` are container-aware.** `stop_gradient` receives
  whole tensor objects/trees (`_ctm_root_implicit_symmetric.py:1944` a `SymmetricTensor`;
  `_ctm_energy_ad.py:348` the CTM env) — neither has `.detach()`, so the lowering is
  `backend.tree.map(lambda t: t.detach(), x)`. `checkpoint` is pinned to
  `use_reentrant=False`: `_ctm_energy_ad.py:346` checkpoints `(site_tensors, envs)`
  (tensors nested in containers), which the default **reentrant** variant does not
  treat as participating inputs — it would silently drop those gradients.

**Eager tape caveat.** The "dynamic block shapes are free" payoff is about *retracing*,
not tape memory. A differentiated non-fixed-point loop (`scan` → Python `for`) under
torch eager still materializes the full tape, so long DMRG/TDVP sweeps need
`checkpoint` for the memory win — unlike the CTM path, which `fixed_point` covers.

---

## 8. RNG, dtype & device

- **RNG:** `backend.random` abstracts key vs. generator (JAX `PRNGKey` split at 17
  sites; torch `torch.Generator`) via `random.normal(shape, seed_or_key, dtype)`.
  Determinism: same seed → reproducible *within* a backend; cross-backend bit-equality
  is **not** promised. **Transform-time randomness is a distinct hazard**: a stateful
  `torch.Generator` under `torch.func.vmap` errors unless the `randomness=` mode is
  set, so any random draw reached inside a transform must go through a
  randomness-mode-aware path.
- **dtype/x64:** JAX needs global `jax_enable_x64`; torch is per-tensor
  `float64`/`complex128`. The `jnp.float64` literals in factories route through
  `B.default_real/complex`. **Promotion diverges**: torch **raises** on mixed
  real×complex `matmul`/`einsum`, whereas JAX-with-x64 promotes implicitly — so the
  seam needs an explicit real→complex promotion policy at contraction/linalg
  boundaries (the JAX code relies on implicit promotion the torch path won't provide).
  See also the dtype-introspection predicates (§4.3).
- **device (GPU parity is a v1 aim — Decision D6):** torch device movement is trivial
  (`.to(device)`), so v1 targets **single-device GPU numeric parity**.
  `set_backend("torch", device="cuda")` sets the default device; `B` never hard-codes
  one. Because "GPU-parity" is where Tenax has historically diverged (CPU-green ≠
  GPU-green — #813 cuBLASLt, #803 CPU-energy vs GPU-gradient, the
  adjoint-needs-converged-forward class), two things must be designed in:
  1. **Deterministic scatter.** `jax.ops.segment_sum` → torch `index_add_`/`scatter_add_`
     is **non-deterministic on CUDA**. `torch.use_deterministic_algorithms(True)` is
     the right lever but **insufficient alone**: deterministic cuBLAS (backing
     `einsum`/`matmul`) additionally requires **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** in
     the environment, or it **raises at the first cuBLAS op**; and the switch is
     process-global and throws where no deterministic kernel exists. Both belong in the
     design, not "documented perf cost."
  2. **Torch's own GPU numerics** (`torch.linalg.svd` gesvdj vs gesvd, complex GEMM)
     are validated by *running the parity suite on CUDA* (§10), not assumed from
     CPU-green. Note determinism ≠ *same accumulation order as JAX's `segment_sum`*, so
     tight cross-backend `allclose` on scatter-accumulated ops may be unattainable —
     the tolerance policy for those ops is stated in §10, not aspired to as bit-parity.

---

## 9. Key decisions

**D1 — Array dispatch.** Custom `ArrayOps` Protocol (§4.3); `array-api-compat` only for
the trivial elementwise subset. *Rejected:* pure Array-API (incomplete for
einsum-paths/segment_sum/complex/algorithm-selected SVD); duck-typed module swap (no
type safety over ~86 files).

**D2 — AD sharing.** Shared backend-neutral `_fwd`/`_bwd`, per-backend wrapper (§5.2),
cotangent convention handled once (§5.3). *Rejected:* two independent gradient impls
(doubles the surface where the complex-convention bug hides).

**D3 — CTM/DMRG loop AD.** Fixed-point/implicit adjoint on both backends (§5.4).
*Rejected:* naive unrolled torch autograd (memory).

**D4 — Migration of the ~86-file `jnp` surface.** Incremental with a `JaxBackend`
pass-through so a module *behaves* identically before/after, layer by layer (core →
linalg → contraction → algorithms), full suite green at each step, enforced by the
seam-boundary CI gate (§4.1). *Rejected:* big-bang codemod (unreviewable).

**D5 — Optional torch dependency, pinned.** torch is an extra
(`pip install tenax[torch]`); `set_backend("torch")` raises a clear ImportError if
absent. **A minimum torch version is pinned** in the extra and in CI, because the
design leans on version-sensitive surfaces: `torch.func`
(`setup_context`/`generate_vmap_rule`), `torch.while_loop`/`torch.cond`
(prototype/experimental in many releases), `checkpoint(use_reentrant=False)`, and the
private `torch._C._functorch.*` / `torch.utils._pytree`. JAX stays a hard dependency.

**D6 — GPU numeric parity as a v1 aim.** _Recommend:_ **single-device GPU parity is a
v1 aim** (torch device movement is trivial), validated by a CUDA-gated parity suite.
_Alternative (rejected):_ CPU-only parity in v1, GPU "best-effort" — rejected because
CPU-green has repeatedly ≠ GPU-green here (§8). **Caveat:** the aim is only *enforced*
if CI has a CUDA runner; without one the D6 parity runs only out-of-band (see §10/M
note). Owner for GPU parity + cross-backend flake triage must be named.

**D7 — no-AD-wall families (MPS + forward-only RG) stay on JAX/CPython (§1).**
_Recommend:_ DMRG/iDMRG/TDVP keep the `numpy_blockwise`/Cython-BLAS accelerator, and
TRG/HOTRG/GILT keep the JAX eager path (no AD, no jit → `0.000` compile); torch offers
these only a correctness oracle. The torch backend is aimed strictly at the
block-sparse/PEPS **AD** workloads. _Alternative (rejected):_ make torch a first-class
execution path for these — rejected because there is no wall to break (MPS: CPython is
faster on CPU; RG: JAX already runs eager with zero compile), so it buys nothing over
the oracle.

---

## 10. Test / parity / AD-wall strategy

Acceptance = a **cross-backend parity suite** (correctness) **plus an AD-wall
benchmark** (the §2 driver — item 7):

1. **Op parity** — each `ArrayOps` method + block-sparse op (contract, permute,
   fuse/split), identical inputs both backends, `allclose` (f64/c128). **For SVD/QR/eigh,
   compare gauge-invariants, not raw factors** — `U`/`V` columns, `R` diagonal signs,
   eigenvectors carry gauge/phase freedom resolved differently by `torch.linalg.*` vs
   JAX, so `allclose(U_torch, U_jax)` fails on correct outputs. Assert singular/eigen
   values, `U diag(S) Vh` reconstruction, and — for a **degenerate** singular/eigen
   subspace — the **clustered projector** `V_a V_aᴴ` vs `V_b V_bᴴ` (invariant under a
   within-subspace unitary rotation), **not** the cross-overlap magnitude `|Vᴴ_a V_b|`
   (which two correct backends can make non-identity by rotating a degenerate block —
   it would reject valid decompositions in exactly the degenerate case).
2. **Gradient parity** — the 6 leaf primitives (§5.1) **and the fixed-point family**
   (incl. C4v-reference), plus composed objectives (DMRG/iPEPS energy). Compare via
   **directional derivatives with per-backend pairing** (§5.3), not raw grad `allclose`.
   Includes a **complex-parameter** case and a case run **through
   `backend.ad.value_and_grad`** so the fixed-point `Function` is exercised under a
   `torch.func` transform, not just eager `.backward()`.
3. **Algorithm parity — one representative case per *targeted* family, not a sample.**
   Per the v1 focus (§1), the parity target is the block-sparse/PEPS **AD** families;
   the suite must exercise **every** one the design commits to, or the checklist can
   pass while a targeted entry point stays JAX-bound. **Block-sparse/PEPS *AD* families
   (full parity, the torch target):** a small iPEPS energy+grad, **fPEPS**, **PESS** —
   each forward **and** AD end-to-end on torch vs the pinned JAX references.
   **No-AD-wall families (oracle-level, per D7):** MPS — DMRG (→ −0.4431 Heisenberg),
   iDMRG, TDVP — *and* the forward-only RG algorithms **TRG/HOTRG/GILT** (exported
   `gilt_tnr`/`gilt_plaquette`; no AD, no jit, `0.000` compile — JAX eager is already
   their fast path). These run only a small cross-backend **correctness oracle**, not a
   throughput or production gate; their production path stays JAX/CPython (D7). Any
   family YJ chooses to drop entirely must move to §1 non-goals, not be silently absent.
   **Oracle scope is CPU-only, and that is load-bearing, not incidental.** The
   oracle-only families still reach `np.array()`/`np.asarray()` on **tensors**,
   not just on index metadata, and those calls are unconditional -- they are not
   behind an `accelerator` switch:

   | Family | Host-read sites on the public path |
   |---|---|
   | dense iDMRG | `idmrg.py:976-977` (`np.array(A_L)`, `np.array(W)` feeding the fixed-point env solves) and `:1071-1085` (periodic re-orthogonalization: `np.array(A_L)`, `np.array(A_R)`, `np.array(s_center)`) |
   | GILT | `gilt.py:222` (`float(jnp.sum(...))`), `:256` (`np.asarray(s >= cut)`), `:318` (`float(jnp.max(...))`) -- plus `:209/265/267/274/284`, **ten sites, not three** |

   A torch **CPU** tensor converts through `__array__` and these all pass. A torch
   **CUDA** tensor does not, and `float(...)` on a functorch-wrapped tensor either
   raises or silently detaches a value the gradient needed. So: the cross-backend
   oracle for these families runs on **CPU torch only**, and the D6 CUDA matrix
   deliberately does **not** include them. Extending CUDA coverage to dense iDMRG
   or GILT is gated on first routing the sites above through `B.to_numpy` /
   device-native predicates -- it is not a test-matrix edit. In particular the
   iDMRG CUDA case must not be satisfied by a symmetric-only representative,
   which would leave the dense path above untested while reporting green.

4. **Optimizer-step parity** — one full `optimize_gs_ad` **and** one PESS update step
   **in the default L-BFGS mode** (build → `update` returns a direction → line search →
   functional apply), asserting parameters track the JAX/optax step. **Must use complex
   parameters** so the `_euclidean_grads` convention (§5.3) is actually exercised; an
   Adam-only or real-only test would leave both the direction contract and the
   double-conjugation trap invisible.
5. **Mutation discipline** — new parity tests must kill a seeded mutant (a dropped
   boundary conjugation must fail the complex-grad test).
6. **GPU parity (D6)** — the op/grad/algorithm/optimizer runs are **CUDA-gated
   variants** (JAX-GPU vs torch-GPU), skipping cleanly with no CUDA device. Non-optional
   given the CPU-green ≠ GPU-green history — **but only enforced if a CUDA CI runner
   exists** (D6 caveat); otherwise this validates out-of-band and must be owned.
7. **AD-wall benchmark — did we break the wall (the whole point, §2)?** Correctness
   parity alone does not prove v1 succeeded: the driver is the block-sparse VJP
   trace+compile cost. So v1 ships a **benchmark**, not just tests, on a representative
   **fermionic iPEPS/CTM AD step** (the worst case) comparing torch-eager backward vs
   JAX — reporting *first-call* wall-clock (where XLA pays the block-sparse-VJP compile)
   and steady-state per-step time. Acceptance is directional: torch eager must remove
   the one-time compile cost and be competitive per-step at the D/χ where the wall
   bites. This is a **benchmark artifact** (not a pass/fail gate — hardware-dependent),
   but a required v1 deliverable so "we broke the wall" is measured, not asserted.

Bucketing: fast op/grad parity → `core`; algorithm/optimizer → `algorithm`/`slow`;
GPU-gated skips without CUDA. Scatter-accumulated ops (`segment_sum`) use a **relaxed
tolerance** documented per op (§8), not bit-parity.

---

## 11. Phasing & effort

Re-cut by dependency (a prerequisite never lands after its consumer). Effort is
Risk × Bulk; rough person-weeks are indicative, not a commitment.

| Phase | Deliverable | Risk | Bulk | ~pw |
|---|---|---|---|---|
| **0. Seam + invariant** | `tenax.backend` package; `ArrayOps` Protocol incl. functional indexed-updates (137 `.at[]` / 25 files) **and dtype-introspection predicates**; `JaxBackend` pass-through + op-parity-vs-`jnp` test; **seam-boundary CI grep-gate** (§4.1); migrate `core/tensor.py` + `linalg.py` dense kernels behind `B`; export `set_backend`/`get_backend` in `__all__`. Suite green, zero behavior change. | Low | High | 3–5 |
| **1. Torch forward** | `TorchBackend` array ops + dense/symmetric linalg forward + contraction (`torch.einsum`/opt_einsum replay + segment-sum equiv); **RNG + default-dtype/promotion policy (§8)**; **`to_numpy` + device-native truncation, dtype predicates (§4.3)** — both are forward prerequisites, not polish. Op-parity green (gauge-invariant, §10). | Low–Med | Med | 3–4 |
| **2. Torch AD (leaf)** | Refactor the 6 leaf primitives to `_fwd/_bwd`; `torch.func`-compatible `Function` (`setup_context` + vmap rule) **incl. `nondiff_argnums` and hidden-residual returns (§5.2)** — intrinsic to these primitives; **`regularized_qr` needs a hand-written backend-neutral backward or `backend.ad.vjp` pulled forward from 3a (§5.2#4)** — its current bwd calls `jax.vjp`; complex-cotangent boundary + `_euclidean_grads` convention-guard + directional-derivative parity test (§5.3). Gradient-parity green (complex case). | **High** (§5.3) | Med | 3–5 |
| **3a. Control + trees + transforms** | `backend.control` combinators (incl. `map`) + `jit`/`checkpoint` (container-aware, §7); `backend.tree` protocol + register all containers (186 sites); `backend.ad` transforms + `value_and_grad` adapter (~76 sites); migrate the ~18 tracer checks to the functorch-aware predicate (per-site review). DMRG parity green. | Med | High | 4–6 |
| **3b. Fixed-point + solvers (the target)** | `fixed_point(step, params, …)` on the boundary-leaf + `setup_context` + double-differentiable contract (§5.4), incl. C4v-reference; `backend.linalg.gmres`/`solve_triangular`/`bicgstab` (bicgstab is the default). Small iPEPS energy+grad parity green **and the §10.7 AD-wall benchmark on a fermionic iPEPS/CTM AD step** — this is the deliverable the whole backend exists for (§2). | **High** | Med–High | 4–6 |
| **3c. Optimizer** | Backend optimizer (§5.6): functional default L-BFGS returning a direction; migrate **every** Optax user (`_build_optimizer` + PESS + root-implicit) to the `(direction, state)` contract. One iPEPS + one PESS default-mode step through torch (§10.4). | Med | Med | 2–3 |
| **4. Polish** | Drop GPU-only workarounds on torch path; docs + `capabilities.md`; `README.md` documents `set_backend`; example; CI torch job (**and, for D6, a CUDA runner or an explicit out-of-band owner**). | Low | Low–Med | 1–2 |

Phase 0 is the tedious-but-safe backbone; Phase 2 is the small-but-dangerous core;
3a/3b/3c were one overloaded "Phase 3" and are split because they are independently
reviewable and 3b carries the CTM-adjoint + Krylov risk that a single "Med–High" label
hid. Phases land as independent PRs; the torch path stays opt-in behind `set_backend`
until 3b/3c make an algorithm end-to-end usable.

**Public-API acceptance (repo rule):** `set_backend`/`get_backend` in `__all__` (Phase
0) + documented in `README.md` (Phase 4) are merge-blocking.

### 11.1 Definition of Done (v1 exit checklist)
- [ ] **AD-wall broken (the point, §2/§10.7):** the block-sparse VJP through a
  fermionic iPEPS/CTM AD step runs under torch eager with **no XLA backward compile**
  and competitive steady-state per-step time vs JAX, captured as a benchmark artifact.
- [ ] **Block-sparse/PEPS AD families** — iPEPS, fPEPS, PESS — run under
  `set_backend("torch")` forward **and** AD to tolerance-equal results vs JAX on **one
  device** (CPU and, if a CUDA runner exists, GPU), each with a representative
  end-to-end parity test (§10.3). **No-AD-wall families** (MPS DMRG/iDMRG/TDVP and the
  forward-only RG TRG/HOTRG/GILT) default to JAX/CPython (D7) and carry only an
  **oracle-level** torch sanity check, not a production/throughput gate. Any family
  dropped entirely is listed in §1 non-goals, not merely absent.
- [ ] Op/grad parity `core`-green; algorithm + optimizer parity `slow`-green; at least
  one **complex-parameter** optimizer step through torch (§10.4); a **double-backward
  (`gradgrad`)** test through the SVD/eigh primitives (§5.2) green.
- [ ] Seam-boundary CI gate green (no raw `jnp`/`lax` outside `backend/`).
- [ ] `set_backend`/`get_backend` exported + documented; torch version floor pinned.
- [ ] Named owner for GPU-parity + cross-backend flake triage.

---

## 12. Risks & open questions

- **Complex convention + `_euclidean_grads` double-conjugation (§5.3)** — the top
  hazard; mitigated by a one-layer convention guard + a complex-parameter
  mutation-checked parity test.
- **SVD/eigh backward on degenerate spectra, and *forward* gauge** — the shared `_bwd`
  uses Tenax's gauge-fixed formula, but grad parity also needs the **forward** SVD/eigh
  gauge pinned (driver-dependent `U/V` phases; `gesvd` vs `gesvdj`), not just the
  backward — especially at the known degenerate-SV ~5e-4 floor.
- **Non-finite-grad masking** — `jnp.where(isfinite, g, 0)` (`ipeps_optimize.py:1005`)
  → `torch.where` fixes the value but not its VJP (`0*NaN`) and deflates the
  convergence norm (a known trap here); must be ported deliberately, fencing the
  argument, not mechanically.
- **`segment_sum` accumulation order** — determinism ≠ same order as JAX; the tolerance
  policy for scatter-accumulated ops is explicit (§8/§10), not bit-parity.
- **Saved-tensor in-place** — the whole through-torch path must stay mutation-free of
  tensors saved for backward (§4.3), beyond the `.at[]` migration.
- **Migration surface (~86 files)** — cost is reviewer time; D4's incremental default +
  seam gate keep every step green.
- **GPU parity enforcement (D6)** — real only with a CUDA CI runner; otherwise
  out-of-band (§10.6).
- **Open — `torch.compile`** (post-v1 lever, not designed against): eager is v1's whole
  point. Room is left via the compile-forward control lowering (§7), `autograd.Function`s
  being opaque to Dynamo, and the sector-loop/dense-kernel split — but it must not shape
  the v1 API. Ship eager-only; revisit once parity is proven.

---

## 13. What we reuse unchanged (the leverage)

`core/symmetry.py`, `core/index.py`, all block metadata, and
`contraction/blocksparse_plan.py`'s planning logic are **already backend-independent**.
The port never touches the symmetry math — only how the per-block arrays are computed,
decomposed, looped, and differentiated. That is why single-device parity +
through-torch-AD is ambitious but bounded.

---

## Appendix A — review provenance & internal-review deltas

The specific requirements above were hardened across a Codex review (11 rounds) and a
four-lens internal review (citation-verification, torch/AD audit, completeness sweep,
design/consistency). Rather than tag each paragraph inline, the load-bearing findings
are listed here.

**Codex rounds (torch-boundary correctness):** explicit fixed-point params + splat
inputs/outputs; `torch.func` `setup_context` + vmap rule (leaf primitives *and*
fixed-point); `nondiff_argnums` in the factory; hidden-residual returns; `custom_vjp`
factory covers all transforms + `value_and_grad` return-order adapter; `backend.tree`
protocol for 186 sites; container-aware `stop_gradient`/`checkpoint` (`use_reentrant=
False`); backend `jit`/`checkpoint`; `bicgstab` as the default solver; C4v-reference
primitive; `ArrayOps.top_k`/`one_hot`; live `B` proxy; tracer→predicate; host-read
`to_numpy`; functional-indexed-updates; all-Optax-users; functional default L-BFGS.

**Internal-review deltas (new this pass, not from Codex):**
- **§5.3** `_euclidean_grads` double-conjugation (`_ipeps_optimize_shared.py:265`, #957)
  — the convention must be reconciled at one layer; complex-parameter optimizer test.
- **§4.3/§8** dtype-introspection predicate family (default Arnoldi/2×2-projector/GMRES).
- **§4.3** `to_numpy` is insufficient under `torch.func` — device-native truncation
  mandatory; the tracing predicate is unreliable under `torch.func` (requires_grad
  False) → functorch probe + per-site review.
- **§5.3/§10** per-backend directional-derivative pairing; §10.1 gauge-invariant op
  parity.
- **§5.4** functorch-`vjp` composition + double-differentiability (`compute_excitations`).
- **§5.6** functional L-BFGS is not a verbatim port (Hermitian `vdot`, `γ`-scaling,
  curvature skip, consume `∇E`).
- **§8** `CUBLAS_WORKSPACE_CONFIG` for CUDA determinism; real×complex promotion;
  `torch.func.vmap` randomness mode.
- **§4.3/§12** saved-tensor in-place version counter; forward SVD/eigh gauge; `where`
  `0*NaN` masking.
- **§1/§9/§11** scope reworded to *single-device* parity (sharded/cuTensorNet are
  feature exclusions); D6 added to the registry; rollback + seam-gate (§4.1);
  Phase 3 re-cut into 3a/3b/3c with dependencies fixed (`nondiff_argnums`→2,
  `to_numpy`/dtype→1); Definition of Done (§11.1); torch version floor (D5).
- **§7** `lax.map` combinator; `torch.while_loop` has no backward (fixed_point-only);
  identity-`jit` must accept/ignore `static_argnums`; eager tape ≠ free memory.
- **Citations corrected:** `ad_utils.py:915`→`:882`; "9 files"→7; "80/120"→~86;
  "stop_gradient 31 files"→17/27; "seven sites"→6; `:2154`→`:2155`; `:165`→`:166`;
  "fori_loop 10"→11.

**Codex rounds 9–11 (post-consolidation, torch-boundary correctness):**
- **R9** — double-backward residuals (recompute-from-inputs, not
  `mark_non_differentiable`, §5.2#3); degenerate-block gauge-invariant projector
  parity (§10.1); per-promised-family end-to-end coverage (§10.3).
- **R10** — call-time backend binding: module-scope `jax.jit` at `dmrg.py:1202`/
  `idmrg.py:418`/`tdvp.py:126` resolves at import under the §4.1 "no tensor yet"
  guard → `control.jit`/`checkpoint`/`custom_vjp` bind **per call**, cached per
  backend generation (§7); GILT & dense-iDMRG host-reads reclassified oracle-only
  (§10.3, D7).
- **R11** — **`regularized_qr`'s backward calls `jax.vjp`** (`_ad_primitives.py:650`,
  not comment-only — a **prior refutation reversed here**), so the QR primitive needs
  `backend.ad.vjp` in Phase 2 or a hand-written backward (§5.2#4); **`.astype`
  method calls** (31 in `pess.py` + `_ad_primitives.py:371` + `_ctm_energy_ad.py:1477/
  1485/1498`) are uncovered by a `B.astype` *function* and the grep-gate (§4.3);
  **`fixed_point` backward `jax.device_get` diagnostics** (`_ctm_energy_ad.py:1694-
  1699`) crash torch-CUDA host conversion — an AD-target host-read, skippable under
  `torch.func` (§4.3, §3 table).
