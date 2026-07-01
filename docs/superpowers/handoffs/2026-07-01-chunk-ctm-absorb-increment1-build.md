# Chunked dense-CTM absorb — Increment 1 build findings

**Date:** 2026-07-01
**Branch:** `feat/632-chunk-ctm-absorb-inc1`
**Gate:** `docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md` (STRONG GO)
**Plan:** `docs/superpowers/plans/2026-07-01-chunk-ctm-absorb-increment1.md`

## What shipped

- **`src/tenax/algorithms/_ctm_chunked_absorb.py`** — raw-array chunked edge-absorption core for all four
  1×1 CTM directions (left/right/top/bottom) + `_raw_in_label_order` helper. Each core
  function reproduces `T_new = conj(P1)ᵀ · (edge·a) · P2` via `jax.lax.map(batch_size=K)` over the
  boundary-χ axis without materializing the full χ²·D⁶ intermediate.
- **`src/tenax/algorithms/_ctm_tensor_moves.py`** — `chunk_size: int | None = None` added to all four
  1×1 `_ctm_tensor_move_*` functions; chunked branch is dense-gated
  (`isinstance(env_self.T*, DenseTensor)`); default `None` leaves the production path
  byte-for-byte unchanged.
- **`src/tenax/algorithms/_ctm_tensor_convergence.py`** — `chunk_size` threaded into
  `_ctm_tensor_sweep_multisite`; `_shard_a` hoisted and wired into the 1×1 branch (previously
  2×2-only) so chunking composes with the GSPMD device mesh.
- **`src/tenax/algorithms/_ctm_python_loop.py`** — `ctm_chunk_size` added to `_make_jit_ctm_step`
  (cache key + `static_argnames`) and `python_loop_ctm_converge` / `_python_loop_chi_ramp`.
- **`src/tenax/algorithms/ipeps_config.py`** — `CTMConfig.ctm_chunk_size: int | None = None` field.
- **`src/tenax/algorithms/ipeps_ad_policy.py`** — `ctm_chunk_size` forwarded through
  `ctm_converge_kwargs` and the `ctm_energy_implicit` call chain.
- **`tests/test_ctm_chunked_absorb.py`** — 19 `core`-marked parity tests.

## Design note: gate target vs production path

The gate (`2026-06-30-chunk-shard-ctm-move-findings.md`) validated the lever on `_compiled_move_*`
(test-only dead code). Production runs the label-based `Tensor.contract()` absorb path inside
`_apply_projector_tensor`. Increment 1 chunks that path via `_ctm_chunked_absorb.py`, called from
the four `_ctm_tensor_move_*` functions when the dense gate fires. The production relabel /
`_flip_leg_flow` / `_phase_fix_normalize_tensor` tail is reused verbatim — only the `T*_new` array
and its three `TensorIndex` objects are computed differently in the chunked branch.

## Parity guarantees

| Level | Oracle | Bound |
|---|---|---|
| core (chunked vs monolith einsum, 4 dirs, K=1/K>1/ragged, real) | `_monolith_{left,right,top,bottom}` | rel ≤ 1e-12 |
| core (complex projectors, LEFT) | same | rel ≤ 1e-12 |
| full-move (chunked vs default, 4 dirs) | `_ctm_tensor_move_*` default | rel ≤ 1e-12 on C/T fields |
| full-move (ragged chunk_size=5, chi=12) | same | rel ≤ 1e-12 |
| end-to-end converge (chunk on == off, 8 fixed 1×1 sweeps) | `python_loop_ctm_converge` default | rel ≤ 1e-10 |

Default-off (`chunk_size=None`) is byte-for-byte the original path; the existing test suite
confirms no regressions.

## Scope / deferrals

- **Forward only.** The implicit-AD backward (`_ctm_energy_ad.py`, `jit_step_bwd`) deliberately
  does NOT receive the `ctm_chunk_size` knob — that is Increment 2, gate-first.
- **1×1 recipe.** The 2×2 absorb path is untouched (each 2×2 move already has its own per-move
  sharding path; chunking there is separate work).
- **Dense only.** Symmetric and fermionic envs: when `chunk_size` is set but the env is not
  dense, the move functions emit `warnings.warn("chunk_size is set but env is not dense
  (SymmetricTensor); falling back to the standard fused-index CTM path.")` and fall back to the
  standard fused-index CTM path.
- **No optimize_gs_ad benchmark.** The D=10–12 / large-χ multi-GPU benchmark depends on Increment
  2 (correct + bounded-memory grads through the chunked backward). Do not wire `optimize_gs_ad`
  until Increment 2 gate passes.

## Next: Increment 2

Gate the chunked backward through the implicit-AD adjoint before any optimize wiring:
1. Verify AD through the chunked forward gives finite, correct gradients (compare to default at small D/χ).
2. Confirm peak memory through the backward also drops (the gain may be partial — lax.map
   rematerializes per chunk in backward, which is the intended behavior).
3. Only then wire `ctm_chunk_size` into `_ctm_energy_ad.py:jit_step_bwd` and `optimize_gs_ad`.
