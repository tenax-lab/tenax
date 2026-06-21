# Even-D padded-`vmap` CTM-AD port — feasibility spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide GO/NO-GO on the one un-refuted #566 lever — a fully-jitted even-D fermionic CTM sweep over padded uniform-block stacks — via staged, highest-risk-first gates, building no more than each gate needs.

**Architecture:** A standalone `examples/spike_evenD_padded_vmap_566.py`, zero production edits through Gates 0–2. It reuses the eager `_fuse_indices_symmetric` as a correctness oracle and replaces its per-block JAX scatter loop with ONE global scatter (Gate 0); assembles a single-jit forward sweep step from {global-scatter fuse, the existing batched contraction #568, batched SVD} (Gate 1); wraps it in a `lax.scan` fixed point (Gate 2); and adds the existing implicit adjoint (Gate 3). Each gate has a numeric GO bar and an explicit STOP.

**Tech stack:** JAX (x64), `jax.jit`/`jax.vmap`/`lax.scan`, A100. Harness reuse: `examples/profile_ctm_ad_wall_566.py` (`make_site_and_gate`, `_install_compile_capture`, `_cold`), `examples/probe_padded_vmap_566.py` (`block_stats`, env converge). Source under study: `src/tenax/algorithms/_tensor_utils.py:231` (`_fuse_indices_symmetric`), `src/tenax/contraction/contractor.py:241` (`_contract_symmetric_batched`).

**Design:** `docs/superpowers/specs/2026-06-19-566-evenD-padded-vmap-ctm-port-spike-design.md`.

**Gate flow (stop at first failure):**
```
Gate 0 padded fuse   -> correct + flat? --no--> NO-GO (chain-breaker unbreakable)
Gate 1 fwd sweep step-> compile collapse (A)? --no--> NO-GO (per-block ops survive)
Gate 2 fwd energy    -> warm beats dense (B)? --no--> PARTIAL GO (keep compile win)
Gate 3 + implicit bwd-> vg compile flat + grad ok --> FULL GO (production design)
```

**Standing rules (from CLAUDE.md):** run `uv run python ...` (not bare `python`); A100 runs use `JAX_PLATFORMS=cuda,cpu`; commit messages end with the `Co-Authored-By: Claude Opus 4.8` trailer; never `todense()` a large symmetric tensor (the fused tensors compared here are small — env corners/edges — so `todense()` for the oracle is allowed).

---

## File Structure

- `examples/spike_evenD_padded_vmap_566.py` — the entire spike (all gates, one file; the gates share the fuse/sweep machinery and the harness imports).
- `examples/spike_evenD_padded_vmap_566_summary.md` — findings (written in the final task).
- No production source is modified. The spike imports `_fuse_indices_symmetric` and `_contract_symmetric_batched` and builds its padded analogs alongside them.

---

## Task 1 — Gate 0: padded global-scatter fuse (the chain-breaker)

**Files:**
- Create: `examples/spike_evenD_padded_vmap_566.py`
- Reference (read, do not modify): `src/tenax/algorithms/_tensor_utils.py:231-403` (`_fuse_indices_symmetric`)
- Reuse (import): `examples/profile_ctm_ad_wall_566.py`, `examples/probe_padded_vmap_566.py`

**Context for the implementer:** `_fuse_indices_symmetric` splits into (a) STATIC numpy bookkeeping that computes, per input block key `(qa,qb)`, a transpose `perm`, a merged shape, a target charge `q_f`, and a `scatter_map[(qa,qb)]` of target offsets within the fused block; and (b) a per-block JAX loop that transposes+reshapes each block and does `out[q_f].at[offsets].set(block_flat)` — ONE scatter primitive **per block**, so the traced graph holds `n_blocks` scatter+transpose+reshape ops and compile scales with block count. Gate 0 collapses (b) to ONE scatter over the whole flat `_data` buffer, using a statically-precomputed global index map, so the op count is block-count-independent. The eager function is the exact correctness oracle.

- [ ] **Step 1: Module scaffold + imports.** Create the file with the x64 config, the harness imports (mirroring `examples/probe_padded_vmap_566.py` lines 1–45: `importlib`-load `profile_ctm_ad_wall_566.py` as `prof`, import `_build_double_layer_tensor`, `_ctm_tensor_multisite_fixed_point`, `SINGLE_SITE_NEIGHBORS`, `CTMConfig`), plus `from tenax.algorithms._tensor_utils import _fuse_indices_symmetric, fuse_indices` and `from tenax.core.tensor import SymmetricTensor`.

- [ ] **Step 2: Write the failing correctness test.** Add `test_padded_fuse_matches_eager()` that, for `(D, chi) in [(2, 8), (4, 16)]`, converges a 1×1 fermionic env (reuse `prof.make_site_and_gate("fermionic", D, 42)` + `_ctm_tensor_multisite_fixed_point`), then for each edge tensor `T` in `{env.T1, env.T2, env.T3, env.T4}` fuses its two non-χ... actually fuse the two legs that a sweep fuses (axes `(0, 1)` of the corner-grown edge). Compute `ref = _fuse_indices_symmetric(T, a, b, "f", OUT)` and `got = padded_fuse(T, a, b, "f", OUT)` and assert `jnp.max(jnp.abs(ref.todense() - got.todense())) < 1e-10`. (Corners/edges are small — `todense()` is fine.)

- [ ] **Step 3: Run the test to confirm it fails.** Run: `JAX_PLATFORMS=cuda,cpu uv run python -c "import examples.spike_evenD_padded_vmap_566 as s; s.test_padded_fuse_matches_eager()"`. Expected: `NameError: padded_fuse` (not yet defined).

- [ ] **Step 4: Implement `build_fuse_plan(T, a, b, fused_flow)` (static).** Re-derive, in numpy at build time, the same metadata the eager fuse computes (copy its lines 244–342 logic): `new_indices`, the per-output-block `fused_dim[q_f]`, and `scatter_map[(qa,qb)]`. Then compose, for the concrete `T`, two flat int arrays over the input `_data` buffer:
  - `src` — for each element of each input block (in `_data` order), its source index into `T._data`;
  - `dst` — that element's target index into the output flat buffer, accounting for the transpose `perm` (lines 363–366) + reshape + the per-`(qa,qb)` `scatter_map` offset + the output block's base offset.
  Return `(new_indices, out_block_keys, out_block_shapes, out_total, src, dst)`. This is the research core; the oracle (Step 2) makes it verifiable.

- [ ] **Step 5: Implement `padded_fuse(T, a, b, fused_label, fused_flow)` (the O(1)-op apply).** Call `build_fuse_plan`, then in ONE scatter:
  ```python
  out_data = jnp.zeros(out_total, dtype=T._data.dtype).at[dst].set(T._data[src])
  obj = object.__new__(SymmetricTensor)
  obj._indices = new_indices
  obj._init_flat_buffer(_unflatten_to_blocks(out_data, out_block_keys, out_block_shapes))
  return obj
  ```
  The graph now holds exactly one gather (`T._data[src]`) + one scatter, independent of `n_blocks`. (`src`/`dst` are static constants; only `T._data` is traced.)

- [ ] **Step 6: Run the test to verify it passes.** Run the Step-3 command. Expected: no assertion error (both D=2 and D=4 match eager to <1e-10). If it fails, debug `dst`/`perm` against the eager scatter (systematic-debugging: compare one block's targets).

- [ ] **Step 7: Write the compile-flatness measurement.** Add `gate0_compile()` that, for `D in (2, 4)`, builds the largest sweep-fused edge tensor, wraps `lambda data: padded_fuse(reconstruct(data), a, b, "f", OUT).norm()` and the eager `_fuse_indices_symmetric` analog, and uses `prof._install_compile_capture()` + `prof._cold(jax.jit(fn), data, cap)` to record `(compile_s, n_compiles)` for both. Print a table: `D | eager_compile_s eager_n | padded_compile_s padded_n`.

- [ ] **Step 8: Run Gate 0 and record the verdict.** Run: `JAX_PLATFORMS=cuda,cpu uv run python examples/spike_evenD_padded_vmap_566.py` (a `main()` calling `test_padded_fuse_matches_eager()` then `gate0_compile()`). **GO** iff padded matches eager (<1e-10, already asserted) AND padded compile is flat D=2→D=4 (ratio < 1.5×) at a few seconds while eager scales. **NO-GO ⇒ STOP** the plan here; the chain-breaker is unbreakable in-graph — record in the summary task and conclude dense-pragmatic stands.

- [ ] **Step 9: Commit.**
  ```bash
  git add examples/spike_evenD_padded_vmap_566.py
  git commit -m "spike(#566): Gate 0 — padded global-scatter fuse (correct + compile-flat)"
  ```

---

## Task 2 — Gate 1: forward sweep step in one jit (Gate A: compile collapse)

**Files:**
- Modify: `examples/spike_evenD_padded_vmap_566.py`
- Reference: `src/tenax/contraction/contractor.py:241` (`_contract_symmetric_batched`), `src/tenax/algorithms/ad_utils.py` (`_ctm_tensor_sweep_multisite`), `src/tenax/algorithms/_ctm_tensor_init.py` (env, double layer)

**Context:** A CTM sweep step grows corners/edges (contraction), builds projectors (truncated SVD of corner products), and renormalizes. The contraction already has a batched path (`_contract_symmetric_batched`, #568) — at even D, where all combos share a shape, it is one batched op. Truncated SVD over a uniform block stack is `jnp.linalg.svd` batched over the leading axis (native). The only previously-eager piece is the fuse, now Gate-0. This task assembles ONE jitted forward sweep step from these and measures whether per-block op emission is actually gone.

- [ ] **Step 1: Build the padded sweep step.** Add `padded_sweep_step(env_stacks, dbl_stack, chi)` that performs one full CTM directional sweep for the 1×1 cell using `padded_fuse` (Task 1), the batched contraction (call the production `contract` with `TENAX_BATCH_BLOCKSPARSE=1`, or `_contract_symmetric_batched` directly), and a batched truncated SVD helper `padded_svd(stack, chi)` (= `jnp.linalg.svd` over the stacked corner-product blocks + per-sector truncation to χ/2). Return the new env stacks. Keep it a pure function of the traced stacks (static metadata in closures).

- [ ] **Step 2: Write the failing warm-correctness test.** Add `test_sweep_step_matches_eager()`: for `(D,chi) in [(2,8),(4,16)]`, take a converged env, run ONE production eager sweep (`_ctm_tensor_sweep_multisite`) and ONE `padded_sweep_step`, and assert the renormalized corners/edges match to `< 1e-8` (dense-compare the small tensors). Run it; expect failure (function or shape mismatch) until Step 1 is correct; debug to green.

- [ ] **Step 3: Write `gate1_compile()` (Gate A).** Mirror `gate0_compile`: cold-`jit` `padded_sweep_step` and capture `(compile_s, n_compiles)` at D=2 χ=8 and D=4 χ=16. Print alongside the recorded eager baseline (`fwd_cmp` ferm D2 = 63.8s; D3 = 527.8s from `profile_566_a100_Dsweep.json`).

- [ ] **Step 4: Run Gate 1 and record the verdict.** Run the spike. **Gate A GO** iff `padded_sweep_step` compile is **< 30s AND flat D=2→D=4 (ratio < 2×)** — per-block emission is gone. **NO-GO ⇒ STOP**: even-D uniformity was necessary but the fused sweep still emits per-block ops; record which sub-op (capture by inspecting `jax.make_jaxpr`) and conclude.

- [ ] **Step 5: Record the warm step time** (for Gate B): in `gate1_compile`, after compile, time 20 warm `padded_sweep_step` calls (with `block_until_ready`) and the eager sweep, and print both. No gate here — this feeds Task 3.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/spike_evenD_padded_vmap_566.py
  git commit -m "spike(#566): Gate 1 — forward sweep step, one jit (Gate A compile collapse)"
  ```

---

## Task 3 — Gate 2: forward energy via `lax.scan` (Gate B: warm beats dense)

**Files:**
- Modify: `examples/spike_evenD_padded_vmap_566.py`
- Reference: `examples/profile_ctm_ad_wall_566.py` `build_loss` (dense baseline), `570-dense-largeD-study` memory (dense is χ^1.7 runtime-bound)

- [ ] **Step 1: Build the scan fixed point.** Add `padded_ctm_energy(A_data, chi, depth)` that reconstructs the site stack, builds the double-layer stack, initializes the env stacks, runs `lax.scan`/`lax.while_loop` over `padded_sweep_step` for `depth` iterations, and contracts the converged env for the 1×1 energy. Forward only (no AD).

- [ ] **Step 2: Write the failing energy-correctness test.** `test_energy_matches_production()`: at D=2 χ=8, assert `|padded_ctm_energy(A) - make_ctm_energy_fn(...)({(0,0):A})| < 1e-6`. Run; debug to green. (Reuse `prof.build_loss` for the production reference.)

- [ ] **Step 3: Write `gate2_warm()` (Gate B).** At D=4 χ=16: time the warm `padded_ctm_energy` step and the **dense** warm energy step (`prof.build_loss("dense", ...)` analog at matched D/χ), both after a warmup call + `block_until_ready`, averaged over ≥10 reps. Print `padded_warm_s`, `dense_warm_s`, ratio.

- [ ] **Step 4: Run Gate 2 and record the verdict.** **Gate B FULL GO** iff `padded_warm_s ≤ dense_warm_s` at D=4 (any real win; the partial form was 0.90×). **Gate B NO-GO ⇒ PARTIAL GO**: Gate-A compile win stands (worth landing for dev/CI/cold-start), but no production warm speedup — **STOP before Task 4** and record PARTIAL GO. Also log GPU saturation note (is `padded` host- or device-bound? — compare to the #627 host-bound regime).

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/spike_evenD_padded_vmap_566.py
  git commit -m "spike(#566): Gate 2 — scan fixed-point forward energy (Gate B warm vs dense)"
  ```

---

## Task 4 — Gate 3: implicit backward + full `value_and_grad` (only if Gates A+B GO)

**Files:**
- Modify: `examples/spike_evenD_padded_vmap_566.py`
- Reference: `src/tenax/algorithms/_ctm_energy_ad.py` (`ctm_energy_implicit`, the existing implicit adjoint)

- [ ] **Step 1: Wrap with the implicit adjoint.** Make `padded_ctm_energy` a `jax.custom_vjp` whose forward is the Task-3 scan and whose backward is the existing implicit fixed-point adjoint applied to the scan graph (the adjoint's `J^T` matvec is the scan's VJP — native). Reuse `ctm_energy_implicit`'s adjoint construction; only the forward step changes.

- [ ] **Step 2: Write the failing gradient test.** `test_grad_matches_production()`: at D=2 χ=8, assert `jnp.max(jnp.abs(grad(padded_ctm_energy)(A_data) - grad(production_loss)(A)._data)) < 1e-6`. Run; debug to green.

- [ ] **Step 3: Write `gate3_compile()`.** Cold-`jit` `value_and_grad(padded_ctm_energy)` and capture `(vg_compile_s, n_compiles)` at D=2 χ=8 and D=4 χ=16. Baseline: `vg_cmp` ferm D2 = 206.4s, D3 = 2111.3s.

- [ ] **Step 4: Run Gate 3 and record.** **FULL GO** iff vg compile is flat D=2→D=4 (≪ 2111s) AND the gradient matches (<1e-6). On FULL GO, the spike has proven the production path; the follow-on is a production-integration design (new even-D `adjoint_method`). Record either way.

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/spike_evenD_padded_vmap_566.py
  git commit -m "spike(#566): Gate 3 — implicit backward, full value_and_grad compile"
  ```

---

## Task 5 — Findings writeup + memory + PR

**Files:**
- Create: `examples/spike_evenD_padded_vmap_566_summary.md`
- Update (in `~/.claude/.../memory/`): `566-padded-vmap-evenD.md`, `MEMORY.md`

- [ ] **Step 1: Write the summary.** Mirror `examples/probe_padded_vmap_566_summary.md`: question, the gate table with measured numbers (Gate 0 compile flat?, Gate A collapse?, Gate B ratio, Gate 3 vg), verdict (FULL GO / PARTIAL GO / NO-GO and at which gate), and recommendation. Be explicit about which gate stopped it and what that means for production.

- [ ] **Step 2: Update memory.** Append the outcome to the `566-padded-vmap-evenD` memory (and the `MEMORY.md` one-liner): GO/PARTIAL/NO-GO, the gate that decided it, and the resulting recommendation (production design / land compile-only win / dense stays pragmatic).

- [ ] **Step 3: Open the PR.** Branch off main, commit the summary, push, and `gh pr create` with a 🤖 AI marker in the body (PreToolUse hook enforces) and the `🤖 Generated with [Claude Code]` trailer. Enable auto-merge: `gh pr merge <n> --squash --auto`.

---

## Self-Review

- **Spec coverage:** Gates 0/1/2/3 of the design map to Tasks 1/2/3/4; the staged GO bar (compile→warm) is Gate A (Task 2 Step 4) then Gate B (Task 3 Step 4); fusion-crux-first is Task 1; PARTIAL-GO fallback is Task 3 Step 4. Findings/record is Task 5. Covered.
- **Placeholder scan:** thresholds are concrete (1e-10, 1e-8, 1e-6, <30s, <2×, <1.5×, ≤dense); baselines cited (63.8/527.8/206.4/2111.3s). The one research-core step (Task 1 Step 4 `build_fuse_plan`) is specified by interface + oracle + construction recipe, not a placeholder — its correctness is gated by the Step-2 oracle test.
- **Type consistency:** `padded_fuse` signature matches `_fuse_indices_symmetric`; `build_fuse_plan` returns the tuple `padded_fuse` consumes; `padded_sweep_step`/`padded_ctm_energy`/`gate*` names are used consistently across tasks.
- **STOP discipline:** every gate task ends with an explicit GO/NO-GO/PARTIAL verdict and a STOP instruction, so an executor does not build Task N+1 after a NO-GO.
