# cuTensorNet FFI handler (P-B5) — on-device kernel design (#200)

> **⛔ NO-GO (2026-06-07). The §0 gate FIRED NEGATIVE — do NOT build this.**
> P-B4 measured the contraction backend at the D=4/χ=12 bar: the compile wall
> (`_jit_fused_fixed_point_bwd`, ~549 s) is **invariant to the contraction backend**
> (548.7 s cutensornet vs 548.9 s stacked), so an on-device contraction handler cannot
> move it. cuTensorNet was also 1.29× *worse* on warm-step. This spec is retained as the
> record of the road not taken and the gate logic that rejected it. See
> `../handoffs/2026-06-07-cutensornet-pb4-finding-nogo.md`. Real lever: **#570** (block-
> sparse SVD/eigh VJP inside the implicit-diff backward).

**Date:** 2026-06-07
**Status:** NO-GO — gate (§0) fired negative at P-B4; design preserved for the record.
**Depends on:** P-B1–P-B3 (forward kernel, `custom_vjp` wrap, spine validation — all
done on the A100). Seam: `blocksparse_plan.py`, `blocksparse_backend.py`,
`blocksparse_cutensor.py`, `blocksparse_vjp.py`.
**Goal:** replace the `pure_callback` bridge inside `CuTensorNetBackend` with a
`jax.ffi.ffi_call` to a C++/CUDA handler that runs the batched block contraction on
the **device buffers in XLA's stream** — no host round-trip. This is the only piece
that can deliver the **runtime** win (#195/#200); the compile win is already route-
independent.

---

## 0. The decision gate (do NOT build until P-B4 says so)

P-B5 is conditional. Build it iff the P-B4 (callback) numbers show:

1. **Compile collapses** — `cutensornet` `bwd_cmp`/`vg_cmp` drops materially below the
   `stacked` baseline (546 s / 1061 s, D=4 χ=12). This proves the one-op route is
   real and is **preserved** by FFI (same single `custom_call` in the HLO).
2. **Warm-step is the (expected) loss** — the callback's device↔host round-trip per
   contraction dominates runtime (smoke: ~24 s warm step at D=2 depth=2). The size of
   that loss is the runtime headroom FFI would recover.

If compile does NOT collapse even with the callback, STOP — the problem is upstream
(graph structure / `n_compiles`), not the transport, and FFI buys nothing. If it
does, P-B5 is the on-device path; weigh it against Pallas (§8).

---

## 1. What changes — and what does NOT

The seam was built so this is a **one-function swap**. In `cutensor_forward`
(`blocksparse_cutensor.py:95`):

```python
gathered       = [jnp.take(operand_stacks[pos], rows, axis=0) for pos in (0, 1)]  # XLA, on-device — UNCHANGED
batched_result = _cutensor_batched_einsum(plan.batched_subscripts, *gathered, sd)  # ← the ONLY callback today
summed         = jax.ops.segment_sum(batched_result, segments, num_segments)       # XLA, on-device — UNCHANGED
return jnp.take(summed, canon_perm, axis=0)                                          # XLA, on-device — UNCHANGED
```

Only `_cutensor_batched_einsum` crosses to the host. Gather / `segment_sum` / reorder
are already on-device XLA ops (block-count-independent). **P-B5 replaces exactly that
one call** with `ffi_call`; everything else — including the `custom_vjp` wrap
(`_execute_opaque`), `backward_contraction`, and the whole plan — is untouched.

| Component | P-B5 status |
|---|---|
| `_cutensor_batched_einsum` (the batched contraction) | **replace** `pure_callback` → `ffi_call` |
| `cutensor_forward` gather/segment_sum/reorder | unchanged |
| `_execute_opaque` (`custom_vjp`) | unchanged — forward still calls `cutensor_forward` |
| `backward_contraction` (transposed plan) | unchanged — it is itself a contraction; see §5 |
| `BlockContractPlan` / `select_backend` / `cutensor_available` | unchanged (extend `available()` to also probe the FFI target is registered) |
| `tests/stacked/test_vjp_seam_cutensor.py` | unchanged — re-runs verbatim against the FFI route (§6) |

---

## 2. JAX-side wiring (`jax.ffi`, present in jax 0.10.1)

`jax.ffi` exposes `register_ffi_target`, `ffi_call`, `include_dir`, `pycapsule`.

```python
# one-time registration at import of the FFI module (guarded by availability)
import jax
from tenax_cutensornet_ffi import handler_capsule  # PyCapsule from the built .so
jax.ffi.register_ffi_target(
    "tenax_cutensornet_batched_contract", handler_capsule, platform="CUDA"
)

def _cutensor_batched_ffi(subscripts, a, b, out_shape_dtype):
    return jax.ffi.ffi_call(
        "tenax_cutensornet_batched_contract",
        out_shape_dtype,                      # jax.ShapeDtypeStruct (n_combos, *out_block_shape)
    )(a, b, subscripts=subscripts)            # subscripts passed as a static FFI attribute (string)
```

`ffi_call` returns a normal JAX array, is **one `custom_call` in the HLO** (compile win
preserved), and runs in XLA's CUDA stream. `out_shape_dtype` is already computed
statically in `cutensor_forward` (`out_sd`, `:130`) — reuse it verbatim.

Notes:
- The contraction is **static per call site** (subscripts + shapes are plan-derived,
  not data). Pass `batched_subscripts` and any layout metadata as FFI **attributes**
  (compile-time constants), so the handler can build/cache the cuTensorNet plan once
  per distinct contraction and the descriptor work stays out of the hot path.
- `available()` (`blocksparse_cutensor.py:43`) gains a third check: the FFI target is
  registered (import of the built extension succeeded). Off-GPU / no-extension hosts
  still fall back to `None` cleanly, and the callback path can remain as a
  `TENAX_CUTENSORNET_TRANSPORT=callback` escape hatch for debugging.

---

## 3. The C++/CUDA handler

Use XLA's typed FFI C++ API (header path from `jax.ffi.include_dir`):

```cpp
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

ffi::Error BatchedContractImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> a,        // (n_combos, *a_block_shape)  device ptr
    ffi::Buffer<ffi::F64> b,        // (n_combos, *b_block_shape)  device ptr
    std::string_view subscripts,    // static attribute, e.g. "Bij,Bjk->Bik"
    ffi::ResultBuffer<ffi::F64> out // (n_combos, *out_block_shape) device ptr
) {
    // 1. parse `subscripts` -> modes/extents (cache per (subscripts, shapes))
    // 2. cutensornetCreate(&handle) [once, cached]; describe tensors from buffer
    //    dims; cutensornetCreateContractionPlan(...) [cache keyed on the static
    //    contraction signature]
    // 3. cutensornetContraction(handle, plan, a.data, b.data, out.data,
    //                           workspace, stream)   // <-- XLA's stream
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BatchedContract, BatchedContractImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()        // a
        .Arg<ffi::Buffer<ffi::F64>>()        // b
        .Attr<std::string_view>("subscripts")
        .Ret<ffi::Buffer<ffi::F64>>());      // out
```

Key handler responsibilities:
- **Use the XLA stream.** Bind `cudaStream_t` via `Ctx<PlatformStream>` and pass it to
  every cuTensorNet call. Do NOT create your own stream or synchronize — that
  reintroduces the serialization FFI exists to avoid.
- **Cache the plan + handle.** `cutensornetCreate` and contraction-plan construction
  are expensive; key a cache on the static contraction signature (subscripts +
  extents + dtype) so steady-state CTM sweeps reuse it. (XLA may call the handler with
  an `instantiate`/`prepare` stage — use it for one-time setup if available in this
  XLA version; otherwise lazy-init + cache.)
- **Workspace.** Query `cutensornetWorkspaceComputeContractionSizes`, allocate via the
  XLA-provided scratch allocator if exposed, else a cached device buffer sized to the
  max seen. Never `cudaMalloc` per call in the hot path.
- **dtype templating.** Instantiate for `F64` and `C128` (the production dtypes; the
  seam is proven for both). Dispatch on `Buffer` element type. No real/complex
  conjugation in the handler — matches `backward_contraction`'s no-conj contract
  (`blocksparse_vjp.py` docstring).

### Batching strategy (the `B` mode)
The plan's batched contraction is `"B<a>,B<b>->B<out>"` over the survivor-combo axis
(`n_combos = len(group.segment_ids)`, `cutensor_forward:127`). Options, cheapest-risk
first:
1. **Loop the batch inside the handler** — one `cutensornetContraction` per combo on
   the shared stream. Zero transfer, trivially correct, async on the stream. Start
   here. (Per-combo launch overhead remains, but on-device and pipelined — far cheaper
   than the callback's per-contraction host round-trip.)
2. **Strided/batched contraction** — express `B` as a batched mode if cuTensorNet
   supports it for this shape, collapsing the loop into one descriptor. Optimization;
   only if (1)'s launch overhead shows up in the warm-step re-measure.

The output rows MUST stay in **survivor order** (i.e. exactly what `segment_sum`
downstream expects). Do not reorder in the handler — `canonical_perm` runs in JAX
after.

---

## 4. Output contract (unchanged, restated for the handler author)

The handler returns the **per-combo batched result** `(n_combos, *out_block_shape)` in
the SAME order `_cutensor_batched_einsum` returned today (survivor order, pre
`segment_sum`). It does NOT do accumulation or canonical reordering — those are the
JAX `segment_sum` + `jnp.take(canon_perm)` steps that already follow it. Honoring this
keeps the FFI swap byte-for-byte transparent to `cutensor_forward`.

---

## 5. The backward

`backward_contraction(operand_stacks, out_cotangent_stack, plan, wrt)` is **itself a
block-sparse contraction** expressed through the same plan machinery
(`blocksparse_vjp.py`): it builds a transposed plan and calls `stacked_execute`. Two
options for P-B5:

- **Minimal (recommended first):** leave the backward on `stacked_execute` (pure JAX /
  XLA). The forward goes through FFI; the backward stays on the XLA-fused path. This
  already removes the dominant forward round-trips and is the smallest correct step.
- **Full on-device:** route the backward's batched contraction through the SAME
  `ffi_call` (it is the same `"B<x>,B<y>->B<z>"` shape, transposed). Do this only if
  the backward contraction shows up in the warm-step re-measure after the forward is
  on FFI. The `custom_vjp` structure (`_execute_opaque`) does not change either way —
  `opaque_bwd` still calls `backward_contraction`; only what `backward_contraction`'s
  inner batched einsum dispatches to changes.

No conjugation anywhere (real + complex128) — see the `blocksparse_vjp.py` rationale;
it is proven against `jax.vjp` ground truth and must not be "fixed."

---

## 6. Validation — reuse the spine verbatim

The P-B3 test file (`tests/stacked/test_vjp_seam_cutensor.py`) is route-agnostic: it
asserts value + grad == per-block == `StackedJaxBackend`, real + c128, ferm_D2/D4,
opacity, fp 1e-12. **Run it unchanged against the FFI route** — it is the acceptance
test for P-B5. Add only:
- a transport switch so the test can force `TENAX_CUTENSORNET_TRANSPORT=ffi` (and a
  `skipif` if the extension is not built), and
- a one-op-count assertion that the FFI forward is still O(#groups) `custom_call`s in
  the jaxpr (the compile-collapse premise), mirroring the P-B1 op-count check.

GPU `segment_sum`/loop reduction order can drift ~5e-7 (bounded-fp tier); the f64
grad assertions still hold at 1e-12. Never compare raw SVD factors — N/A (contraction
only).

---

## 7. Build & packaging (the real cost of P-B5)

This is the step up from "transiently pip-install cuQuantum": P-B5 introduces a
compiled extension that must build in CI.

- **Toolchain:** CMake + `nvcc`, include `jax.ffi.include_dir`, link `cutensornet`
  (and `cutensor`), matched to the CUDA toolkit and the JAX/XLA ABI of the pinned jax.
- **Artifact:** a `.so` exposing the handler as a `PyCapsule` (`jax.ffi.pycapsule`),
  imported by `blocksparse_cutensor` lazily (so non-GPU hosts never import it).
- **Packaging:** an optional extra (e.g. `tenax[cutensornet]`) building the extension;
  `pyproject` build hook (scikit-build-core / setuptools+CMake). Default install stays
  pure-Python; the backend is still default-OFF (env-gated).
- **CI:** the FFI tests run ONLY on a CUDA runner with cuQuantum; everywhere else they
  skip (the existing `cutensor_available()` gate already does this). Document the A100
  build recipe in the handoff (cf. `cutensornet-200-env` memory).

ABI fragility is the main maintenance risk: the extension is pinned to a JAX/XLA
version. Pin jax tightly in the cutensornet extra and re-test on bump.

---

## 8. Alternative: Pallas (the TPU-also path)

Pallas is a JAX-native kernel DSL — **no separate C++ build**, stays in the XLA
toolchain, on-device, and is the same lever for **TPU**. But Pallas means hand-writing
the block contraction kernel; it **cannot call cuTensorNet** (no NVIDIA-tuned
contraction). Trade-off:

| | FFI + cuTensorNet | Pallas |
|---|---|---|
| Build | C++/CUDA extension in CI (heavy) | none (JAX-native) |
| Kernel quality | NVIDIA-tuned contraction | self-written (Triton/Mosaic) |
| Portability | CUDA only | GPU + TPU |
| Seam impact | swap `_cutensor_batched_einsum` | swap `_cutensor_batched_einsum` |

Both plug into the **same one-function seam** (§1), so the choice is deferrable and
reversible. If the P-B4 numbers justify on-device work, prototype the FFI loop-batch
handler (§3.1) first (smallest path to a real warm-step number), and keep Pallas as
the portability play once the lever is proven.

---

## 9. The re-measurement (P-B5 exit)

Re-run the P-B4 sweep with the FFI transport and compare warm-step against the
callback and the `stacked`/`perblock`/dense baselines:

```bash
for be in stacked cutensornet; do
  TENAX_BLOCKSPARSE_BACKEND=$be TENAX_CUTENSORNET_TRANSPORT=ffi \
  JAX_PLATFORMS=cuda,cpu uv run python examples/profile_ctm_ad_wall_566.py \
    --D 2 4 --sym fermionic --depth 8 --reps 3 --json profile_${be}_ffi.json
done
```

**Success:** warm-step approaches the dense floor (the #195 tiny-kernel-launch fix),
while compile stays collapsed (the #566 win, already banked by the callback). If FFI
removes the round-trip but warm-step is still far from dense, that is a real finding
about cuTensorNet per-block-contraction overhead at this problem size — bring the
numbers back before scaling to Pallas/TPU.

---

## 10. Build order (de-risk transport first)

1. **P-B5a** — minimal FFI: loop-batch handler (§3.1), forward only, backward stays on
   `stacked_execute` (§5 minimal). Build + register + pass `test_vjp_seam_cutensor`
   under `TRANSPORT=ffi`. Re-measure warm-step (§9).
2. **P-B5b** — only if warranted by §9: batched-mode handler (§3.2) and/or backward on
   FFI (§5 full).
3. **P-B5c** — packaging/CI hardening (§7): the `tenax[cutensornet]` extra + CUDA-only
   test lane.
```
