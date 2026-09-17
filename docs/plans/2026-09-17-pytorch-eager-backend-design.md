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
| **AD primitives** | `_ad_primitives.py:229-792` (6 `custom_vjp`), +`blocksparse_backend.py:149-177`, ~60 sites/16 files | **Yes — redesign.** |
| **Control flow** | `lax` `while_loop`×29, `scan`×22, `fori_loop`×10, `stop_gradient`×58 across 31 files | **Yes — redesign.** |
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
where, pad, astype, zeros, ...`) plus the decomposition entry points. Two
implementations: `JaxBackend` (thin wrappers over today's `jnp`, behavior
identical) and `TorchBackend`.

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
subtleties. That's the same 6 primitives in `_ad_primitives.py` today.

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
- **Torch:** a `torch.autograd.Function` whose `forward` calls `_svd_fwd` and
  `save_for_backward`s the residuals, and whose `backward` calls `_svd_bwd`.

A single `backend.ad.custom_vjp(fwd, bwd)` factory hides which wrapper is used, so
the 6 primitives are written once.

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

### 5.4 Fixed-point AD for CTM (parity, not naive unroll)
Torch eager would, by default, build the full unrolled tape through a CTM
convergence loop → memory blowup at the D/χ we care about. Parity requires the
torch path to use the **same implicit/fixed-point adjoint** the JAX path already
uses (`_ctm_energy_ad.py`, root-implicit modules): backward solves the adjoint
linear system (GMRES) at the converged environment instead of differentiating every
sweep. Under torch this is again an `autograd.Function` whose `backward` runs the
adjoint solve — structurally identical to the JAX `custom_vjp` on the fixed point,
re-expressed. The `control.fixed_point(...)` combinator (§7) owns this so algorithms
don't special-case the backend.

---

## 6. Pytrees → eager flatten/unflatten

The `@register_pytree_node_class` on the tensor classes exists to cross JAX
`jit`/`grad`/`vmap` boundaries. Torch eager has no tracing, so registration is inert
there — but the *flatten/unflatten* methods stay useful as plain
serialization/reconstruction helpers.

- Keep the pytree registration **active only under the JAX backend**.
- Expose `flatten()/unflatten()` as backend-neutral instance methods (the torch path
  calls them directly where JAX would rely on `tree_util`).
- `DenseTensor.tree_unflatten`'s validation bypass for JAX's `object()` probes
  (`tensor.py:491-493`) is JAX-specific and simply doesn't run under torch.
- `torch.autograd` differentiates w.r.t. the `_data` leaf natively once it's a
  `requires_grad` tensor, so no pytree is needed to carry gradients — the eager tape
  does it.

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
| `stop_gradient(x)` | `lax.stop_gradient` | `x.detach()` | `x.detach()` (unchanged) |
| `fixed_point(step, init, adjoint)` | `custom_vjp` + GMRES adjoint | `autograd.Function` + GMRES adjoint | same `autograd.Function` (opaque to Dynamo — §12) |

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
2. **Gradient parity** — for the 6 AD primitives and for composed objectives
   (DMRG energy, iPEPS energy), compare `torch.autograd.grad` vs `jax.grad` on the
   same inputs. **Includes a complex-input case** targeting §5.3.
3. **Algorithm parity** — DMRG (→ −0.4431 Heisenberg), iDMRG, a small iPEPS
   energy+grad, run end-to-end on torch, compared to the JAX reference values the
   existing benchmarks already pin.
4. **Mutation discipline** — new parity tests must kill a seeded mutant (e.g. a
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
| **0. Seam** | `tenax.backend` package, `ArrayOps` Protocol, `JaxBackend` pass-through; migrate `core/tensor.py` + `linalg.py` dense kernels behind `B`. Suite green, zero behavior change. | Low | **High** (mechanical, 80-file surface — but staged) |
| **1. Torch forward** | `TorchBackend` array ops + dense/symmetric linalg forward + contraction (`torch.einsum`/opt_einsum + segment-sum equiv). Op-parity suite green. | Low–Med | Med |
| **2. Torch AD** | Refactor 6 primitives to `_fwd/_bwd`; `autograd.Function` wrappers; **complex-cotangent boundary + parity test**. Gradient-parity green. | **High** (§5.3) | Med |
| **3. Control flow + algorithms** | `backend.control` combinators; migrate CTM/DMRG/GMRES loops; `fixed_point` adjoint on torch. DMRG + small iPEPS energy&grad parity green. | Med–High | High |
| **4. Polish** | RNG/dtype policy, drop GPU-only workarounds on torch path, docs, `capabilities.md` update, example, CI torch job. | Low | Low–Med |

Phase 0 is the tedious-but-safe backbone; Phase 2 is the small-but-dangerous core.
Phases can land as independent PRs; the torch path stays behind `set_backend` and
opt-in until Phase 3 makes an algorithm end-to-end usable.

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
