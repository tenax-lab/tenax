# Spec: PyTorch eager backend for Tenax block-sparse calculations

**Status:** draft for review · **Author:** Claude Code (for @yingjerkao) · **Date:** 2026-09-17
**Scope decisions (locked by YJ):** (a) **full feature parity** with the JAX path; (b) **AD must work through the torch backend** in v1.

---

## 1. Summary

Add a second array backend so every block-sparse (`SymmetricTensor`) and dense
(`DenseTensor`) calculation can run on **PyTorch eager** with the same public API,
same numerical results, and **working autodiff** (`torch.autograd`), while JAX
remains the default. This is **additive**: JAX is untouched and stays the default;
torch is selected per-process.

The bet that makes this tractable: Tenax's **symmetry logic is already
backend-neutral**. Charge fusion, conservation, canonicalization, and all block
metadata (`core/symmetry.py`, `core/index.py`, `contraction/blocksparse_plan.py`)
are pure NumPy — they never touch `jnp`. Each tensor holds its numeric data as a
**single flat buffer** (`SymmetricTensor._data`, `DenseTensor._data`) plus static
NumPy metadata. So the port is a **backend seam around array ops + linalg + AD +
control flow**, not a rewrite of the hard part.

What is genuinely JAX-specific and must be redesigned (not shimmed): the
`custom_vjp` AD primitives, `jax.lax` control flow, pytree registration, and PRNG
threading. Sections 5–8 handle each.

### Non-goals (v1)
- Not replacing JAX; not a default switch.
- No `jit`/compilation model for torch — eager is the point (dynamic block shapes
  become *free*, see §7).
- No distributed/sharded torch path (JAX `ctm_sharding` stays JAX-only for now).
- No cuTensorNet/Pallas torch kernels — torch uses `torch.einsum`/`opt_einsum`.

---

## 2. Motivation

`docs/guide/capabilities.md` already concedes that for large-D fermionic systems
"an eager PyTorch fermionic-PEPS code (YASTN/peps-torch) is the better tool today."
An in-house torch backend gives us:

- **Debuggable eager execution** — no trace, real stack traces, `pdb` inside a CTM
  sweep, dynamic shapes with no recompile penalty.
- **Ecosystem** — torch optimizers, `torch.compile` later, checkpointing utilities,
  the broader PEPS-torch/YASTN interop surface.
- **A second oracle** — cross-backend parity tests become a standing correctness
  check on the symmetry/AD math itself (§10).

---

## 3. Current coupling surface (grounded in the code)

From a full sweep of `src/tenax` (120 files):

| Layer | Where | Backend-coupled? |
|---|---|---|
| Charge/symmetry arithmetic | `core/symmetry.py`, `core/index.py` | **No** — pure NumPy. Reuse as-is. |
| Block metadata & planning | `contraction/blocksparse_plan.py` (`BlockContractPlan`) | **No** — "backend-agnostic, touches only block metadata." |
| Tensor data buffer | `core/tensor.py:719-803` (`_init_flat_buffer`, `_get_block`) | **Yes**, but tiny: one `concatenate` + slice/reshape. |
| Dense tensor ops | `core/tensor.py:512-544` (`conj`, `transpose`, `norm`) | Yes, small. |
| Array namespace | `jnp.*` in **80 of 120 files** | Yes, **pervasive, no single chokepoint**. |
| Dense linalg kernels | `linalg.py:59-100` (svd), `:1573/1781/2537/2670` (qr/eigh) | Yes; isolatable behind ~3 functions. |
| Block-sparse decomps | `linalg.py` `_truncated_svd_symmetric` (:243), `_qr_symmetric` (:1452), `_eigh_symmetric` (:1663) | Yes; per-sector loops over dense kernels. |
| Contraction execution | `contractor.py:234,356,1124-1135`; `blocksparse_plan.py:373` (`jnp.einsum`+`segment_sum`, `backend="jax"`) | Yes; opt_einsum *path* portable, execution not. |
| **AD primitives** | 6 leaf `custom_vjp`: `_ad_primitives.py:229-719` (5) + `_lorentzian_eigh.py:81` (1), +`blocksparse_backend.py:149-177`, ~60 sites/16 files | **Yes — redesign.** |
| **Differentiation-state checks** | `isinstance(x, jax.core.Tracer)` ×~18 across 9 files (`core/tensor.py:1043`, `linalg.py:274/2045`, `_ctm_projector.py`×7, …) | **Yes — backend predicate** (§4.2). |
| **Control flow** | `lax` `while_loop`×29, `scan`×22, `fori_loop`×10, `stop_gradient`×58 across 31 files | **Yes — redesign.** |
| **Krylov / triangular solvers** | `_gmres_lax.py:171` (`solve_triangular`), `:299-336`; `_metric_precond.py:164`, `ad_utils.py:915` (`jax.scipy … gmres`); **`_krylov_bicgstab` (default `adjoint_solver`), `_ctm_tensor_c4v_reference_ad.py:165`** | **Yes — backend solvers, incl. bicgstab** (§5.4). |
| **DMRG truncation ops** | `jax.lax.top_k`/`jax.nn.one_hot` (`_padded_linalg.py:128/142`), reached by `accelerator="auto"` → `_jit_sweep` | **Yes — `ArrayOps.top_k`/`one_hot`** (§4.2). |
| **Optimizer** | `optax` chains in `_ipeps_optimize_shared.py:112-136` + **`pess_optimize.py:468/775` and `ipeps_optimize_root_implicit.py:637`**; `optimizer.update`/`apply_updates` | **Yes — optimizer seam, all Optax users** (§5.6). |
| **Pytree registration** | `core/tensor.py:441,624`; `stacked_tensor.py:41`; `_padded_block_array.py:229`; `pess.py:252` | **Yes — reinterpret** (§6). |
| PRNG | `jax.random`/PRNGKey in 17 files | Yes; key-threading → generator. |
| Global x64 | `__init__.py:45`; `jnp.float64` literals in factories | Yes; small but global. |
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

**`B` must be a live dispatch proxy, not a rebindable name (Codex #1010 P1).**
`import tenax` eagerly pulls in the numeric modules — `tenax/__init__.py:61-92`
imports `contraction.contractor`, `core.tensor`, `linalg`, `network.*` at package
load — so by the time a user calls `set_backend("torch")`, every one of those
modules has *already* executed `from tenax.backend import B` and bound the
JAX target. If the setter merely **rebinds** `tenax.backend.B`, those modules keep
their original reference and silently stay on JAX — the public switch would be a
no-op for exactly the already-imported operations. So the contract is: **`B` is a
stable proxy object whose *target* `set_backend` mutates in place** (the imported
name keeps pointing at the same proxy, which now dispatches to torch). The
alternative — forcing every call site to dereference `tenax.backend.B.<op>`
dynamically on each use — is rejected as both a per-op indirection cost and an
easy-to-violate rule across the 80-file surface. `set_backend` is therefore only
valid **before any tensor is allocated**; switching mid-process with live tensors
of the old backend is out of scope (documented, and guarded with a clear error).

### 4.1 Layered design

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

### 4.2 The `ArrayOps` protocol (Decision D1 — see §9)

A **custom, explicit Protocol** enumerating the ~40 array ops Tenax actually uses
(`concatenate, reshape, transpose, conj, einsum, tensordot, stack, segment_sum,
where, pad, astype, zeros, top_k, one_hot, ...`) plus the decomposition entry
points. Two implementations: `JaxBackend` (thin wrappers over today's `jnp`,
behavior identical) and `TorchBackend`.

`top_k` and `one_hot` are called out because they are load-bearing on the **default
DMRG path**, not exotic (Codex #1010 P1): `accelerator="auto"` (`dmrg.py:176`) routes
dense-CPU and *all* GPU/TPU runs through `_jit_sweep`, and its truncation uses
`jax.lax.top_k` (`_padded_linalg.py:128`) and `jax.nn.one_hot` (`:142`) for
static-shape global truncation. Neither is a `lax` loop or a `custom_vjp`, so
replacing `jax.jit` with an identity wrapper (§7) still feeds torch tensors straight
into these JAX APIs. They map to `torch.topk` / `torch.nn.functional.one_hot`; the
alternative is to route torch DMRG to a backend-neutral executor instead of
`_jit_sweep`, but adding the two ops keeps the existing padded-truncation code
shared.

**Functional indexed updates are part of the surface (Codex #1010 P2).** Tenax
uses JAX's `x.at[idx].set/add/multiply(...)` **137 times across 25 files** —
including `SymmetricTensor.todense()` (`core/tensor.py:1111`) and the dense DMRG
kernels — and a torch tensor has *no* `.at` API (in-place assignment there also
breaks leaf-autograd and `torch.func` transform constraints). So `ArrayOps`
exposes functional scatter/indexed-update ops (`index_set/index_add/index_mul`,
mapping to `.at[...]` under JAX and out-of-place `index_add`/`scatter`/masked
`where` under torch), and migrating those 137 sites is an explicit part of the
Phase-0/1 mechanical work — without it, importing `B` for `jnp` cannot make even
`todense()` backend-neutral.

**Differentiation-state detection is a backend predicate (Codex #1010 P1).** The
code branches on `isinstance(x, jax.core.Tracer)` in **~18 sites across 9 files**
to select between a concrete/validating path and an AD-safe one — e.g.
`core/tensor.py:1043` skips `np.asarray(data)` (which *raises* on a grad-requiring
tensor) when the data is a tracer; `linalg.py:274`/`2045`, `_ctm_projector.py`
(seven sites: 899/946/1020/1028/1087/1126), `_ctm_tensor_projector_2x2.py:1001`,
`_ctm_tensor_energy.py:130`, `contraction/contractor.py:897`,
`cutensornet_backend.py:64` use the same predicate to pick the AD-safe
decomposition/contraction branch. **A torch tensor with `requires_grad=True` is
never a JAX tracer**, so every one of these checks reads `False` under torch and
sends a differentiable tensor down the concrete branch — either raising (the
`np.asarray` case) or silently taking the numerically-unstable eager path. So
`ArrayOps` exposes a predicate `is_tracing_or_requires_grad(x)` (JAX:
`isinstance(x, jax.core.Tracer)`; torch: `torch.is_grad_enabled() and
x.requires_grad`, plus a `torch.func` functorch-tracer check), and all ~18 sites
migrate to it — this is Phase-3 work called out alongside the transform-site
migration.

**Host-convertibility is a *separate* concern from AD-state — do not conflate them
(Codex #1010 P1).** `is_tracing_or_requires_grad` answers "take the AD-safe branch?",
**not** "can I `np.asarray` this?". The eager decomposition path host-reads the data
directly — `np.array(s_q)` (`linalg.py:471`), `np.asarray(block)` (`:1112/:1359`),
`np.asarray(data)` (`core/tensor.py:1044`) — and torch rejects a direct NumPy
conversion on **two** axes the AD predicate misses: (a) an *ordinary non-grad CUDA
tensor* (predicate `False`, but `.numpy()` still raises — needs `.cpu()` first), and
(b) a CPU `requires_grad=True` tensor read *inside* `torch.no_grad()` (predicate
`False`, but NumPy still rejects it — needs `.detach()`). So the seam exposes a
distinct **`to_numpy(x)`** (torch: `x.detach().cpu().numpy()`; JAX: `np.asarray`) —
or, better on the block-sparse path, a **backend-native truncation** that never
leaves the device — and the host-read sites use *that*, not the AD predicate. Without
this, CUDA HOTRG / public `truncated_svd` fail despite GPU parity being a v1 aim (D6).

Rationale over the Array-API standard: Tenax leans on ops the standard doesn't
cover portably — `einsum` with explicit contraction paths, `segment_sum`,
algorithm-selected SVD, and complex dtypes where array-api support is uneven. We
*do* reuse `array-api-compat` internally for the trivial elementwise subset to cut
boilerplate, but the Protocol is the contract.

### 4.3 Tensor data contract

`SymmetricTensor`/`DenseTensor` keep their shape exactly. `_data` becomes "an array
of the active backend." `_init_flat_buffer`'s `jnp.concatenate` and `_get_block`'s
slice/reshape route through `B`. The static metadata (`_block_keys`, `_block_shapes`,
`_block_offsets`, `_indices`) is already NumPy and unchanged.

---

## 5. AD bridge (the spine)

This is where parity + through-torch-AD is won or lost. The design principle:
**share the VJP *math*, wrap it per backend.**

### 5.1 The split (same on both backends)
Everything composed from `B` ops is differentiable *automatically* — by XLA tracing
under JAX, by the eager tape under torch. **Only the linalg primitives need custom
gradients**, because SVD/QR/eigh backward has degenerate-spectrum and truncation
subtleties. That is **6 leaf decomposition primitives across two files, not one
(Codex #1010 P1)**: `truncated_svd_ad`, `truncated_svd_ad_vh_only`,
`regularized_svd`, `regularized_qr`, `regularized_eigh` in `_ad_primitives.py`
(5), **plus `truncated_eigh_regularized` in `_lorentzian_eigh.py:81`** — a separate
`custom_vjp` that `_ctm_projector.py:1146` invokes whenever an AD iPEPS run selects
`projector_backward="lorentzian"` (a supported projector-backward mode). Scoping
the refactor to `_ad_primitives.py` alone would leave that path unable to run
through torch and absent from the parity suite, so the Lorentzian eigh primitive
goes through the *same* `_fwd/_bwd` + `backend.ad.custom_vjp` treatment (§5.2) and
into the §10 transform tests. (The `*_converge`/`f(params_data_tuple)` `custom_vjp`s
in `ad_utils.py`, `_ctm_energy_ad.py`, `_split_ctm_energy_ad.py`,
`_ctm_honeycomb_ad.py` are a different tier — the CTM **fixed-point adjoint** family
that §5.4's `fixed_point` combinator owns, not leaf decomposition math.)

### 5.2 Refactor each primitive to backend-neutral fwd/bwd
```python
def _svd_fwd(A, k):                 # pure B-ops
    U, S, Vh = B.svd(A); ... truncate ...
    return (U, S, Vh), Residuals(U, S, Vh, ...)

def _svd_bwd(res, dU, dS, dVh):     # pure B-ops (F-matrix / gauge-fixed formula)
    return dA
```
Then wrap:
- **JAX:** `custom_vjp` over `_svd_fwd`/`_svd_bwd` (mechanical move of today's code).
- **Torch:** a `torch.autograd.Function` wrapping `_svd_fwd`/`_svd_bwd`.

A single `backend.ad.custom_vjp(fwd, bwd)` factory hides which wrapper is used, so
the 6 primitives are written once.

**The factory must carry static/nondifferentiable argument positions (Codex #1010
P1).** Three of the six primitives — `truncated_svd_ad`, `truncated_svd_ad_vh_only`
(`_ad_primitives.py:229,470`) and `truncated_eigh_regularized`
(`_lorentzian_eigh.py:81`) — are declared `@partial(jax.custom_vjp,
nondiff_argnums=(1,))` because the truncation rank `chi` controls slicing and output
*shapes*; JAX passes it separately into the backward, and it must **never** become a
traced differentiable argument. A bare `custom_vjp(fwd, bwd)` that ignores this
would either change the backward arity or let `chi` be traced — breaking the
existing jitted JAX calls. So the factory signature is
`custom_vjp(fwd, bwd, nondiff_argnums=())`: under JAX it forwards to
`jax.custom_vjp(..., nondiff_argnums=...)` unchanged; under torch those positions are
**bound as static config** (closed over / passed as non-tensor `.apply` args that
`setup_context` stashes on `ctx`), never as differentiable leaves — so `chi` stays a
Python int on both backends and the backward sees the same argument split.

**The torch wrapper must be `torch.func`-compatible, not a plain Function (Codex
#1010 P1).** A bare `autograd.Function` with only `forward`/`backward` +
`save_for_backward` works under eager `.backward()` but **raises** the moment one
of these primitives is reached through a `backend.ad.grad`/`vjp`/`vmap` transform
(§5.5) — exactly the through-torch AD path. `torch.func` requires the split
`forward` + `setup_context(ctx, inputs, output)` form (no side-effecting saves in
`forward`), and `vmap` additionally needs `generate_vmap_rule = True` or an
explicit `vmap` staticmethod. So the wrapper contract mandates the
`setup_context` form and a vmap rule for all six primitives **and the fixed-point
combinator of §5.4** (which is likewise reached through `backend.ad.value_and_grad`),
and the parity suite (§10) transforms a composed objective through *each* of them to
prove it.

**`setup_context` only sees inputs + *returned* outputs — hidden residuals must be
returned, not stashed (Codex #1010 P1).** Today's `_fwd` rules save residuals that
are **not in their public outputs**: `_truncated_svd_ad_vh_only_fwd` returns only the
truncated `(s, Vh)` but saves `(U_full, s_full, Vh_full, M, k)`
(`_ad_primitives.py:489`), and `_truncated_eigh_regularized_fwd` returns `(w[:k],
v[:,:k])` but saves the *full* eigensystem `(w, v, k)` (`_lorentzian_eigh.py:92`).
JAX `custom_vjp` allows this (fwd returns `(output, residuals)`), but the torch
`setup_context(ctx, inputs, output)` form receives only inputs and the returned
outputs — it **cannot** recover `U_full`/full-`w`/full-`v` from a single `forward`.
So the wrapper contract requires each such primitive to return those residuals as
**hidden auxiliary outputs** of `forward` (marked non-differentiable via
`ctx.mark_non_differentiable`, so no cotangent is expected for them) which
`setup_context` then stashes on `ctx`, or to **recompute** them in `setup_context`.
The backend-neutral `_fwd` therefore declares its residual tuple explicitly so both
wrappers consume the same data; a transform test (§10) exercises exactly these two
truncating primitives through `grad`.

### 5.3 ⚠️ Complex-cotangent convention — the highest-risk item
JAX and PyTorch use **different conjugation conventions** for complex gradients.
Tenax's VJP formulas were written to JAX's convention (cotangents pair
*unconjugated*; the objective reads `Re Σ ḡ·dz`). PyTorch's autograd uses the
Wirtinger/conjugate convention (`.grad` holds the conjugate cotangent). If the
shared `_bwd` math is dropped into a torch `autograd.Function` verbatim, **complex
gradients will be silently wrong** (right magnitude, wrong phase/conjugate) — no
crash, just a bad optimizer direction.

**Mitigation:** the `backend.ad` factory conjugates incoming cotangents and outgoing
grads at the torch boundary so `_bwd` always sees JAX-convention inputs. This is
codified once, and **guarded by a complex-input gradient-parity test** (§10) that
would catch a convention regression. This item alone justifies a dedicated phase.

**The parity test cannot compare raw `jax.grad` vs `torch.func.grad` (Codex #1010
P1).** Even with a *correct* wrapper, the externally visible gradients of a real
objective stay **conjugated relative to each other** — that is the two frameworks'
convention, not a bug — so a naive `allclose(jax_grad, torch_grad)` would reject
the correct implementation and reward one that returns the wrong (conjugated)
direction to a torch optimizer. The test therefore compares **directional
derivatives** `Re⟨g, v⟩` against a finite-difference reference (backend-agnostic),
or `allclose` only *after* an explicit convention conversion at the boundary. The
descent-direction sign is what an optimizer consumes, so that is what is asserted.

### 5.4 Fixed-point AD for CTM — and threading the parameters
Torch eager would, by default, build the full unrolled tape through a CTM
convergence loop → memory blowup at the D/χ we care about. Parity requires the
torch path to use the **same implicit/fixed-point adjoint** the JAX path already
uses (`_ctm_energy_ad.py`, root-implicit modules): backward solves the adjoint
linear system (GMRES) at the converged environment instead of differentiating every
sweep. Under torch this is again an `autograd.Function` whose `backward` runs the
adjoint solve — structurally identical to the JAX `custom_vjp` on the fixed point,
re-expressed.

**The parameters must be explicit inputs, not closed over.** A
`torch.autograd.Function` returns gradients *only for the tensors passed to
`.apply()`*; anything a `step` closure captures runs outside that contract and
receives **no gradient** — which would silently break through-torch CTM AD for
exactly the iPEPS tensors being optimized. This is why the JAX side already makes
the flattened parameters the explicit custom-VJP primal
(`_ctm_energy_ad.py:1213`, `@jax.custom_vjp def f(params_data_tuple)`), not a
closed-over constant. The torch primitive must do the same, so the combinator
threads the parameters through its signature:

```
fixed_point(step, params, init, adjoint)   # params = flattened tensors, differentiable
```

`params` (the flat parameter buffers) are passed to `Function.apply`, and the
backward/adjoint returns *their* cotangents; `step`/`adjoint` are static config, not
gradient sources. `control.fixed_point(...)` (§7) owns this on both backends so
algorithms don't special-case it.

**Torch requires the leaves *splatted* as individual `.apply` arguments, not a
tuple (Codex #1010 P1).** PyTorch autograd only registers **top-level tensor
positional arguments** to a custom `Function` as differentiable inputs — tensors
nested inside a tuple/list/dict passed as one argument are treated as a non-tensor
constant, so `backward` returns *no* cotangents for them and the fixed-point path
would still yield zero parameter gradients. So the torch `fixed_point` wrapper
**flattens `params` to its leaves and calls `Function.apply(leaf0, leaf1, …)` with
one positional tensor per leaf** (the pytree `treedef` from `backend.tree` travels
as static aux and is rebuilt inside `forward`); `backward` then returns exactly one
cotangent per leaf argument, which the wrapper re-assembles into the parameter tree.
This splat/reassemble lives inside `control.fixed_point` so algorithms still pass a
single `params` tree. (JAX has no such constraint — `custom_vjp` differentiates the
whole pytree argument — so this is a torch-side wrapper detail, invisible above the
seam.)

**The same flatten/reassemble applies to the *outputs*, not just the inputs (Codex
#1010 P1).** `torch.autograd.Function` can expose differentiable **outputs** only as
top-level torch tensors, too — a `Function` that returns Tenax `Tensor` objects
cannot receive their output cotangents in `backward`. The C4v-reference fixed point
returns `(C, T)` as `Tensor` objects (`_ctm_tensor_c4v_reference_ad.py:304-334`,
`-> tuple[Tensor, Tensor]`), and `backward` needs `g_c`/`g_t` to flow in. So the
torch wrapper's `forward` returns the **flattened environment buffer leaves** (with
the output `treedef` as static aux), and `control.fixed_point` **reconstructs the
public `(C, T)` / env tensor tree outside `.apply()`**; `backward` then receives one
output cotangent per returned leaf. The rule is symmetric — *both* the parameter
inputs and the environment outputs cross the `.apply()` boundary as bare tensor
leaves, and the Tenax tensor trees are (dis)assembled on the Python side of it.

**The fixed-point `Function` must satisfy the *same* `torch.func` contract as the
leaf primitives (Codex #1010 P1).** §5.2's `setup_context` + vmap-rule requirement
was written for the six decomposition primitives, but the iPEPS loss is evaluated
through `backend.ad.value_and_grad` (→ `torch.func.grad_and_value`, §5.5), so this
fixed-point `autograd.Function` is *itself* reached by a `torch.func` transform — a
plain `forward`/`backward` wrapper would raise there before the adjoint ever runs.
So the `setup_context` form (and a vmap rule) is mandatory for the fixed-point
combinator too, and §10 transforms an iPEPS objective *through* it — not only
through the leaf primitives — to prove the through-torch CTM gradient path.

**The fixed-point family is more than the `_ctm_energy_ad` set (Codex #1010 P1).**
The supported `ctm_ad_mode="c4v_reference"` optimization path calls a *standalone*
`custom_vjp`, `ctm_tensor_c4v_reference_converge_reduced`
(`_ctm_tensor_c4v_reference_ad.py:304-334`), from `ipeps_optimize.py:981` inside
`jax.value_and_grad`. Migrating only the enumerated `_ctm_energy_ad.py` /
`_split_ctm_energy_ad.py` / `ad_utils.py` (`ctm_tensor_converge`) /
`_ctm_honeycomb_ad.py` fixed points would leave the C4v-reference mode JAX-bound
even after the generic work lands. So this primitive routes through
`control.fixed_point` on the same contract, and the reference-mode path gets its
own torch gradient test.

**The adjoint solve needs backend Krylov + triangular solvers (Codex #1010 P1).**
"backward solves the adjoint linear system (GMRES)" is not free under torch: the
solver itself calls JAX directly — `_gmres_lax.py:171`
`jax.scipy.linalg.solve_triangular`, and `jax.scipy.sparse.linalg.gmres` at
`_metric_precond.py:164` (metric preconditioner) plus the Arnoldi/GMRES machinery
in `_gmres_lax.py:299-336` and `ad_utils.py:915`. Migrating loop *combinators* (§7)
does not touch these — they would still receive torch tensors in a JAX API. So the
seam adds **`backend.linalg.gmres`**, **`backend.linalg.solve_triangular`**, and
**`backend.linalg.bicgstab`** (JAX: today's `jax.scipy.*` / `_krylov_bicgstab`;
torch: `torch.linalg.solve_triangular` + torch or `torch.func`-compatible
reimplementations of the existing `lax` Krylov solvers), and their call sites are
named in Phase 3 — without them the fixed-point primitive is not actually
backend-neutral.

**BiCGSTAB is the default adjoint solver, not GMRES (Codex #1010 P1).**
`CTMConfig.adjoint_solver` defaults to `"bicgstab"` (`ipeps_config.py:152`), and the
C4v-reference backward calls `_krylov_bicgstab` *directly*
(`_ctm_tensor_c4v_reference_ad.py:165`) before any GMRES fallback — so a solver seam
covering only GMRES would break on the **default** adjoint path with torch leaves,
before fallback selection even runs. `bicgstab` is therefore a first-class member of
the solver seam (matching the JAX default so torch and JAX solve the same system),
not an afterthought; the alternative — remapping the torch default to `gmres` — would
require its own explicit test and is rejected as diverging from the JAX path.

### 5.5 Reverse-mode transforms belong in the seam too
The `custom_vjp` factory (§5.2) is necessary but **not sufficient**: the algorithms
call the JAX transform APIs *directly* — `jax.vjp` (44 sites, e.g.
`_ctm_energy_ad.py:1269/1323`, `_ctm_root_implicit_asym.py:1462`),
`jax.value_and_grad` (10, e.g. `ipeps_optimize.py:1761`), `jax.grad` (11),
`jax.vmap` (11). Autodiff of `B` ops does **not** subsume these; leaving them
un-abstracted means the torch path feeds torch tensors into `jax.vjp` and parity is
impossible. So `backend.ad` must expose the transforms, not just `custom_vjp`:

| transform | JAX | Torch |
|---|---|---|
| `vjp(f, *primals)` | `jax.vjp` | `torch.func.vjp` (or `autograd.grad` over a taped forward) |
| `grad(f)` / `value_and_grad(f)` | `jax.grad` / `jax.value_and_grad` | `torch.func.grad` / `grad_and_value` (or `.backward()` + `.grad`) — **return order re-normalized, see below** |

**`value_and_grad` return order and `has_aux` must be re-normalized (Codex #1010
P1).** `jax.value_and_grad(f)` returns `(value, grads)`, but
`torch.func.grad_and_value(f)` returns them **swapped** — `(grads, value)` — and the
`has_aux` nesting differs too (`jax` gives `(value, aux), grads`; `torch.func` gives
`grads, (value, aux)`). Exposing the torch transform *raw* at a call site like
`energy_val, grads = value_and_grad(...)(params)` (`ipeps_optimize.py:1761`) would
bind the **gradient tree to `energy_val`** and hand the scalar energy to the
optimizer — a silent, catastrophic swap. So `backend.ad.value_and_grad` is an
**adapter, not an alias**: under torch it calls `grad_and_value` and re-orders to
JAX's `(value, grads)` (and un-nests `aux` to JAX's layout), so every migrated call
site keeps the JAX return contract unchanged. A parity test asserts the tuple
order/aux structure, not just the values.
| `vmap(f)` | `jax.vmap` | `torch.func.vmap` |

`torch.func` (functorch) gives functional, composable transforms that map closely
onto the existing JAX call sites. Migrating these ~76 uses is an **explicit
deliverable of Phase 3**, not something the `custom_vjp` factory covers for free.

### 5.6 The optimizer step is backend-specific too (Codex #1010 P1)
Gradient parity is necessary but **not the end of "full algorithm parity."** After
the gradient, a real `optimize_gs_ad` run continues into the *update* step, which is
built on **optax** — JAX-only: `_build_optimizer` (`_ipeps_optimize_shared.py:112`)
returns `optax.adam` / `optax.scale_by_lbfgs` / `optax.clip_by_global_norm` chains,
and the loop calls `optimizer.update(...)` (`ipeps_optimize.py:1053`) +
`optax.apply_updates`. None of the array/tree/AD/control abstractions replace optax;
`optax.update` cannot step a torch parameter tree. So the seam adds a **backend
optimizer** — under JAX today's optax chains; under torch a functional
reimplementation of the same update math over the `backend.tree` leaves.

**The seam's contract is `update(grads, state, params) → (direction, state)`, and
the default L-BFGS must be *functional* — `torch.optim.LBFGS` is not a drop-in
(Codex #1010 P1).** Tenax does *not* let the optimizer own the step: the loop takes
the returned `updates` as a **search direction** (`ipeps_optimize.py:2154`,
`direction = updates`), runs its **own** line search (Hager-Zhang / Armijo
backtracking) on it, and applies it **functionally**
(`_normalize_params(_tree_add(params, _tree_scale(direction, alpha)))`) — plus a
tangent-space projection (#328) and metric/CG variants. `torch.optim.LBFGS` breaks
every part of that: it *owns and mutates* the parameters in place and drives its own
line search through a loss-recomputing `closure` passed to `.step()`, with no way to
return a bare direction for Tenax's line search. So the default `gs_optimizer="lbfgs"`
maps to a **functional L-BFGS** (the two-loop recursion — Tenax already hand-rolls
one at `ipeps_optimize.py:2148` for the metric-preconditioned path) that returns a
direction matching the optax `scale_by_lbfgs` contract, *not* `torch.optim.LBFGS`.
Adam can use `torch.optim` or a functional form, but the direction-return contract
is the seam's interface either way.

Acceptance requires **at least one complete parameter-update step in the *default*
(L-BFGS) mode** run through torch in the parity suite (§10) — not just an energy+grad
comparison, and not only an unspecified optimizer — otherwise "full parity" stops at
gradient evaluation and the default torch optimization path could be silently
unusable. This is a Phase-3b deliverable.

**The seam must cover *every* direct Optax user, not just the shared iPEPS builder
(Codex #1010 P1).** `_build_optimizer` is not the only Optax call site: the public
PESS optimizers `optimize_pess_ad` / `optimize_pess_3site_multisite_ad` construct
their own `optax.chain(...)` and call `optimizer.update` (`pess_optimize.py:468/479`
and `:775/789`), and the root-implicit optimizer calls `optax.apply_updates`
(`ipeps_optimize_root_implicit.py:637`). Under the locked full-parity scope, all of
these must route through the backend optimizer seam — a test of only `optimize_gs_ad`
would leave the PESS and root-implicit paths silently on Optax with torch trees. So
Phase 3b migrates **every** direct Optax user and the parity suite (§10) exercises
**at least one PESS optimization step** in addition to the iPEPS one.

---

## 6. Trees: a backend tree protocol, not just tensor flatten

Instance `flatten()/unflatten()` on the tensor classes is **necessary but not
sufficient (Codex #1010 P1)**. The algorithms call the JAX *tree* API directly —
**186 `tree_map`/`tree_leaves`/`tree_structure`/`tree_unflatten` uses** (e.g.
`_ctm_energy_ad.py:1212-1331` reconstructs the fixed-point residual and does
optimizer tree arithmetic this way) — and several **standalone registered
containers are not tensor instances**: `IPESSState` (`pess.py:197`),
`StackedTensor` (`stacked_tensor.py:41`), `PaddedBlockArray`
(`_padded_block_array.py:229`). Exposing methods only on the two tensor classes
leaves all of that either JAX-bound or treating whole tensor objects as opaque
leaves.

So the seam defines a **`backend.tree` protocol** — `map`, `leaves`, `structure`,
`flatten`, `unflatten` — with:
- **JAX:** `jax.tree_util.*` (unchanged).
- **Torch:** `torch.utils._pytree` (torch's own registry), or a small in-house
  registry. **Every custom container** (`SymmetricTensor`, `DenseTensor`,
  `IPESSState`, `StackedTensor`, `PaddedBlockArray`) is registered with *both*
  systems, flattening to its array leaves + static aux; the 186 call sites migrate
  to `backend.tree.*` in Phase 3.
- The pytree registration and `DenseTensor.tree_unflatten`'s `object()`-probe
  bypass (`tensor.py:491-493`) stay **JAX-only**; under torch, `torch.autograd`
  differentiates w.r.t. the `_data` leaves natively (the eager tape carries the
  gradient), and `torch.func` transforms consume the torch pytree registration.

---

## 7. Control flow

`lax.while_loop/scan/fori_loop/cond/stop_gradient` have no torch-*eager* analog
because eager *is* Python control flow. Introduce `backend.control` — and give it
**two torch lowerings behind one interface**: an eager one (Python loops) for v1, and
a **compile-forward** one (torch higher-order ops) that a future `torch.compile` path
selects with no change to any algorithm.

| Combinator | JAX (`lax`) | Torch eager (v1) | Torch compile-forward (future) |
|---|---|---|---|
| `while_loop(cond, body, init)` | `lax.while_loop` | Python `while` | `torch.while_loop(cond, body, init)` |
| `scan(f, init, xs)` | `lax.scan` | Python `for`, stack outputs | `torch.while_loop` with an index carry (or the `scan`/`associative_scan` HOP where available) |
| `fori_loop(lo, hi, body, init)` | `lax.fori_loop` | Python `for` | `torch.while_loop` with a counter carry |
| `cond(p, t, f, x)` | `lax.cond` | Python `if` | `torch.cond(p, t, f, x)` |
| `stop_gradient(x)` | `lax.stop_gradient` | `tree.map(detach, x)` (tree-aware) | same (unchanged) |
| `fixed_point(step, params, init, adjoint)` | `custom_vjp` on `params` + GMRES adjoint | `autograd.Function` (`params` = `.apply` inputs, §5.4) + GMRES adjoint | same `autograd.Function` (opaque to Dynamo — §12) |

The CTM/DMRG/GMRES loops (already written in functional-carry style for `lax`) are
migrated to call these combinators. Functional style runs correctly under all three
lowerings — the torch-eager versions are near-trivial; the *constraint* comes from
keeping the bodies `lax`-compatible so the JAX path is unchanged.

**Why `torch.cond`/`torch.while_loop` are the compile-forward path.** These are
PyTorch higher-order ops that keep control flow *inside* the FX graph rather than
unrolling it, so `torch.compile`/Dynamo traces a loop **without a graph break** — the
torch analog of what `lax.while_loop` does for XLA. Crucially they impose the **same
discipline `lax` already requires**: pure functional bodies (no in-place mutation, no
side effects) and a **fixed carry structure/shape across iterations**. That is not an
extra tax — it is *exactly* the constraint we already pay to keep the JAX path
`jit`-able. So one loop body, written once in functional-carry style, serves the JAX
compiled path, the torch eager path, and a future torch compiled path. Routing
`backend.control` from the eager lowering to the HOP lowering is a **single swap
point**, not an algorithm rewrite — this is the "leaves room for `torch.compile`"
claim made concrete.

Boundary that stays symmetric: a loop whose *carry changes block count* mid-run
breaks under both `lax.while_loop` and `torch.while_loop` (data-dependent structure).
In practice CTM/DMRG carries hold χ (and thus block structure) fixed within a run, so
this is a non-issue — and where it isn't, it's a non-issue *identically* on both
backends, so no torch-specific handling is needed.

**`jit` and `checkpoint` are backend ops too, not only `lax` loops (Codex #1010
P1).** DMRG/iDMRG/TDVP flow through *unconditional* `jax.jit` wrappers —
`_matvec_jit = jax.jit(...)` (`dmrg.py:1202`), and the same pattern at
`idmrg.py:418`, `tdvp.py:103` — and the implicit-AD paths carry `jax.checkpoint`
decorators (3 files). Migrating only the `lax` loops leaves torch tensors flowing
into a live `jax.jit`, which fails before the control combinators are even
reached. So `backend.control` (or `backend.compile`) also exposes **`jit`** and
**`checkpoint`**: under JAX they are today's `jax.jit`/`jax.checkpoint`; under
torch `jit` is an **identity/eager** wrapper (no trace) and `checkpoint` maps to
`torch.utils.checkpoint` (eager rematerialization). Their call sites are named in
Phase 3 — without them the DMRG parity path breaks before reaching §7's loops.
Note the identity-`jit` is necessary but not *sufficient* for DMRG: the default
`accelerator="auto"` route runs `_jit_sweep`, whose truncation calls `jax.lax.top_k`
/ `jax.nn.one_hot` directly (`_padded_linalg.py:128/142`), so those must also be
`ArrayOps` ops (§4.2) — an identity `jit` alone would still hand torch tensors to a
JAX API inside the sweep.

**`stop_gradient` and `checkpoint` must be container-aware, not tensor-only (Codex
#1010 P1).** Two lowerings above look trivial but break on Tenax's containers:
- **`stop_gradient` takes whole tensor objects / trees, not raw arrays.**
  `_ctm_root_implicit_symmetric.py:1944` passes a `SymmetricTensor` and
  `_ctm_energy_ad.py` passes the nested CTM environment to `stop_gradient` — neither
  has a `.detach()` method, so a bare `x.detach()` lowering raises. The torch
  lowering is therefore **`backend.tree.map(lambda t: t.detach(), x)`**, detaching
  the array *leaves* through the tree protocol (§6); the symmetric root-implicit and
  truncated-backprop CTM paths depend on it.
- **`checkpoint` must be non-reentrant.** `_ctm_energy_ad.py:346` checkpoints
  `_step_envs_only(site_tensors, envs)` — a dict of tensors and a CTM-environment
  container whose differentiable torch buffers are **nested inside** Tenax tensor
  objects. PyTorch's default **reentrant** `torch.utils.checkpoint` does *not* treat
  tensors nested in structures as participating inputs, so rematerialized sweeps
  would silently lose their parameter gradients. The seam pins
  **`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`** (records the
  inner graph, supports nested structures) — or flattens the container args to
  leaves first — for this through-torch AD path.

**Eager payoff (v1):** under the *eager* lowering, dynamic block shapes across sweeps
cost nothing (no trace, no recompile), so the "static block keys" machinery that
exists only to keep JAX from retracing is simply bypassed. The compile-forward
lowering is opt-in later, for the batched/dense hot paths where fusion pays (§12),
not a v1 requirement.

---

## 8. RNG, dtype & device

- **RNG:** `backend.random` abstracts key vs. generator. JAX threads `PRNGKey`
  (split at 17 call sites); torch uses a `torch.Generator`. Provide
  `random.normal(shape, seed_or_key, dtype)` so factories in `core/tensor.py`,
  `core/mps.py`, `linalg.py` (rsvd) don't branch. Determinism contract: same seed →
  reproducible *within* a backend; cross-backend bit-equality is **not** promised
  (documented).
- **dtype/x64:** JAX needs the global `jax_enable_x64` (stays in `__init__.py`).
  Torch supports `float64`/`complex128` per-tensor with no global flag; the backend
  sets default real/complex dtypes and the `jnp.float64` literals in factories route
  through `B.default_real/complex`.
- **device (GPU parity is a v1 aim — Decision D6):** torch device movement is
  trivial (`.to(device)`), so v1 targets **GPU numeric parity**, not CPU-only.
  `set_backend("torch", device="cuda")` (or `TENAX_DEVICE`) sets the default device;
  factories allocate there, and `B` never hard-codes a device. Two consequences that
  are *not* free and must be designed in, because "GPU-parity" is where Tenax has
  historically diverged (CPU-green ≠ GPU-green — cf. #813 cuBLASLt mixed real/complex
  GEMM, #803 CPU-energy vs GPU-gradient split, the adjoint-needs-converged-forward
  CPU-green/GPU-red class):
  1. **Deterministic scatter.** The stacked block-sparse contraction's
     `jax.ops.segment_sum` maps to torch `index_add_`/`scatter_add_`, which is
     **non-deterministic on CUDA** by default — it perturbs both parity tolerances
     and gradient reproducibility. The torch backend enables
     `torch.use_deterministic_algorithms(True)` (documented perf cost) so the GPU
     path is reproducible and comparable.
  2. **Torch's own GPU numerics.** The JAX GPU workarounds (`linalg.py:59-100`
     cuSOLVER algorithm selection, `_einsum_compat.py` cuBLASLt promotion) are
     dropped on the torch path, but torch's own `torch.linalg.svd` (gesvdj vs gesvd)
     and complex GEMM have their own quirks. GPU parity is validated by *running the
     parity suite on CUDA* (§10), not assumed from CPU-green.

---

## 9. Key decisions

**D1 — Array dispatch mechanism.** _Recommend:_ **custom `ArrayOps` Protocol** (§4.2),
reusing `array-api-compat` only for the trivial elementwise subset.
_Alternatives:_ pure Array-API standard (rejected: incomplete for einsum-paths,
segment_sum, complex, algorithm-selected SVD); duck-typed module swap (rejected:
no type safety over an 80-file surface).

**D2 — AD sharing.** _Recommend:_ **shared backend-neutral `_fwd`/`_bwd`, per-backend
wrapper** (§5.2), with cotangent conjugation handled once at the torch boundary
(§5.3). _Alternative:_ two independent gradient implementations (rejected: doubles
the surface where the complex-convention bug hides).

**D3 — CTM/DMRG loop AD.** _Recommend:_ **fixed-point/implicit adjoint on both
backends** (§5.4). _Alternative:_ naive unrolled torch autograd (rejected: memory).

**D4 — Migration mechanism for the 80-file `jnp` surface.** _Recommend:_ **incremental
with a compatibility default** — `JaxBackend` is a pass-through, so a module reads
identically before/after migration; migrate layer by layer (core → linalg →
contraction → algorithms) with the full suite green at each step. _Alternative:_
big-bang codemod (rejected: unreviewable, no green checkpoint).

**D5 — Optional torch dependency.** _Recommend:_ torch is an **extra**
(`pip install tenax[torch]`); `set_backend("torch")` raises a clear ImportError if
absent; CI adds a torch job. JAX stays a hard dependency.

---

## 10. Test / parity strategy

The acceptance criterion for "parity" is a **cross-backend parity suite**:

1. **Op parity** — for each `ArrayOps` method and each block-sparse op (contract,
   svd/qr/eigh, permute, fuse/split), run identical inputs under both backends,
   assert `allclose` (f64/c128 tolerances).
2. **Gradient parity** — for the 6 leaf AD primitives (§5.1) **and the fixed-point
   family** (`_ctm_energy_ad`, split, honeycomb, and the C4v-reference primitive of
   §5.4), and for composed objectives (DMRG energy, iPEPS energy), compare
   `torch.autograd.grad` vs `jax.grad` on the same inputs. **Includes a complex-input
   case** targeting §5.3, and a case that evaluates the objective **through
   `backend.ad.value_and_grad`** so the fixed-point `Function` is exercised under a
   `torch.func` transform (§5.2/§5.4), not just eager `.backward()`.
3. **Algorithm parity** — DMRG (→ −0.4431 Heisenberg), iDMRG, a small iPEPS
   energy+grad, run end-to-end on torch, compared to the JAX reference values the
   existing benchmarks already pin.
4. **Optimizer-step parity (§5.6)** — at least **one complete `optimize_gs_ad`
   parameter-update step in the *default* L-BFGS mode** run through the torch
   optimizer seam (build optimizer → `update` returns a **direction** → Tenax line
   search → functional apply), asserting the updated parameters track the JAX/optax
   step within tolerance. The default-mode requirement is explicit: a test of only
   Adam would leave the functional-L-BFGS direction contract (§5.6) unexercised.
   Without this the suite would certify gradients but never a real torch
   optimization move.
5. **Mutation discipline** — new parity tests must kill a seeded mutant (e.g. a
   dropped conjugation in the torch AD boundary must make the complex-grad parity
   test fail), per the repo's testing rules.

5. **GPU parity (D6)** — the op/grad/algorithm parity runs are **CUDA-gated
   variants**, not CPU-only: JAX-GPU vs torch-GPU within tolerance. This is
   non-optional given the CPU-green ≠ GPU-green history; a CPU-only suite would give
   false confidence exactly where Tenax has diverged before.

Bucketing: fast op/grad parity → `core`; algorithm parity → `algorithm`/`slow`;
GPU-gated parity skips cleanly when no CUDA device is present.

---

## 11. Phasing & effort

| Phase | Deliverable | Conceptual risk | Bulk |
|---|---|---|---|
| **0. Seam** | `tenax.backend` package, `ArrayOps` Protocol **incl. functional indexed-updates** (§4.2; migrate the 137 `.at[...]` sites across 25 files), `JaxBackend` pass-through; migrate `core/tensor.py` + `linalg.py` dense kernels behind `B`; **export `set_backend`/`get_backend` in `__all__`**. Suite green, zero behavior change. | Low | **High** (mechanical, 80-file surface — but staged) |
| **1. Torch forward** | `TorchBackend` array ops + dense/symmetric linalg forward + contraction (`torch.einsum`/opt_einsum + segment-sum equiv). Op-parity suite green. | Low–Med | Med |
| **2. Torch AD** | Refactor the **6 leaf primitives across `_ad_primitives.py` (5) + `_lorentzian_eigh.py` (1, §5.1)** to `_fwd/_bwd`; **`torch.func`-compatible `autograd.Function` (`setup_context` + vmap rule, §5.2)**; **complex-cotangent boundary + directional-derivative parity test (§5.3)**. Gradient-parity green. | **High** (§5.3) | Med |
| **3. Control flow + algorithms** | `backend.control` combinators + **`jit`/`checkpoint` wrappers (§7)**; **`backend.tree` protocol + register all custom containers (§6; migrate the 186 `jax.tree` sites)**; **`backend.ad` transforms (§5.5) + `fixed_point(step, params, …)` on the `setup_context` contract, incl. the C4v-reference primitive (§5.4)**; **`backend.linalg.gmres`/`solve_triangular`/`bicgstab` for the adjoint solve (§5.4; bicgstab is the default, `_ctm_tensor_c4v_reference_ad.py:165`)**; **`ArrayOps.top_k`/`one_hot` for the default `_jit_sweep` DMRG truncation (§4.2; `_padded_linalg.py`)**; **`value_and_grad` return-order adapter + `custom_vjp` `nondiff_argnums` (§5.5/§5.2)**; **migrate the ~18 `jax.core.Tracer` checks to `is_tracing_or_requires_grad`, and the host-read sites (`np.array`/`np.asarray`) to `to_numpy` / backend-native truncation (§4.2)**; migrate CTM/DMRG/GMRES loops and the ~76 direct `jax.*` transform sites. DMRG + small iPEPS energy&grad parity green. | Med–High | High |
| **3b. Optimizer seam** | **backend optimizer (§5.6)** — optax under JAX; under torch a **functional L-BFGS** returning a direction (default mode), Adam via `torch.optim`/functional; migrate **every** direct Optax user (`_build_optimizer` + `pess_optimize.py` + `ipeps_optimize_root_implicit.py`) to the `(direction, state)` contract feeding Tenax's line search + functional apply. **One full `optimize_gs_ad` *and* one PESS update step in default L-BFGS mode run through torch** (§10 item 4). | Med | Med |
| **4. Polish** | RNG/dtype policy, drop GPU-only workarounds on torch path, docs, `capabilities.md` update, **`README.md` documents the `set_backend` signature**, example, CI torch job. | Low | Low–Med |

Phase 0 is the tedious-but-safe backbone; Phase 2 is the small-but-dangerous core.
Phases can land as independent PRs; the torch path stays behind `set_backend` and
opt-in until Phase 3 makes an algorithm end-to-end usable.

**Public-API acceptance (repo rule, CLAUDE.md/AGENTS.md):** `set_backend` /
`get_backend` are new public API, so it is a *merge-blocking acceptance item* — not
an afterthought — that Phase 0 adds them to `src/tenax/__init__.py`'s `__all__` and
Phase 4 documents their actual signature in `README.md`. Following the phase
deliverables must not leave the advertised entry point unexported or undocumented.

---

## 12. Risks & open questions

- **Complex gradient convention (§5.3)** — highest risk; mitigated by a boundary
  shim + a mutation-checked parity test. Everything else is comparatively mechanical.
- **SVD/eigh backward on degenerate spectra** — torch and JAX regularize
  differently; the shared `_bwd` uses Tenax's own gauge-fixed/regularized formula
  (not the library backward), so this is controlled — but the degenerate-SV parity
  case (cf. the AD-parity floor we already know at ~5e-4) must be in the suite.
- **`segment_sum` equivalent** — the stacked block-sparse contraction relies on
  `jax.ops.segment_sum`; torch uses `index_add_`/`scatter_add_`. Straightforward but
  must match accumulation order for bit-parity-adjacent tolerances.
- **Migration surface (80 files)** — real cost is reviewer time, not difficulty;
  D4's incremental default keeps every step green and reviewable.
- **Resolved (D6):** GPU numeric parity **is** a v1 aim (torch device movement is
  trivial); consequences in §8 (deterministic scatter, CUDA parity suite).
- **Open — `torch.compile` (deferred to a post-v1 lever, not designed against):**
  eager is v1's whole point (dynamic block shapes are free). `torch.compile` would
  later reintroduce a trace, so it must not shape the v1 API — but the design leaves
  room for it: (i) the `control` combinators expose a **compile-forward lowering to
  `torch.cond`/`torch.while_loop`** behind the same interface as the eager loops
  (§7) — a single swap point, no algorithm rewrite; (ii) `autograd.Function`s are
  treated as **opaque** by Dynamo, so the
  6 AD primitives compile as black boxes (correct, just not fused) — same posture as
  JAX seeing `custom_vjp`; (iii) the per-charge-sector Python loops in the
  block-sparse decomps are the part `torch.compile` would most want to fuse and the
  part most prone to graph breaks (data-dependent block counts/shapes) — so a
  compiled path would likely target the dense per-block kernels, not the sector
  dispatch. **Recommendation:** ship eager-only in v1; revisit `torch.compile` as an
  opt-in perf lever once parity is proven, measuring against the eager baseline.

---

## 13. What we reuse unchanged (the leverage)

`core/symmetry.py`, `core/index.py`, all block metadata (`_block_keys/_shapes/
_offsets`), and `contraction/blocksparse_plan.py`'s planning logic are **already
backend-independent**. The port never touches the symmetry math — only how the
resulting per-block arrays are computed, decomposed, looped, and differentiated.
That is why parity + through-torch-AD is ambitious but bounded.
