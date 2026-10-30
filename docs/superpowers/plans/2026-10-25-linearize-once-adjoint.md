# Linearise Once, Jit Only the Transpose — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The implicit-AD backward linearises the gauge-fixed CTM sweep once per backward, and the Neumann loop and the joint GMRES solve compile only its transpose.

**Architecture:** A new closure program `_jit_linearize_joint` (inside `_make_implicit_vjp_fn`) returns `jax.vjp`'s pullback, a `jax.tree_util.Partial` of residuals. Two module-level consumers, `_jit_neumann_bwd` (new) and `_jit_gmres_joint_solve` (re-signed), take that Partial as an ordinary argument. Because a Partial holds no `SymmetricTensor`, the backward picks each consumer's CPU fusion variant from the site tensors through a new `_fusion_variant` helper. The `"separate"` chain rule keeps today's programs; the Neumann loop body moves into a shared `_neumann_loop` so both paths run the same loop.

**Tech Stack:** Python 3.11+, JAX 0.10.2 (`jax.vjp`, `jax.tree_util.Partial`, `lax.while_loop`), pytest, `jax.monitoring` trace events.

**Spec:** `docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md`

## Global Constraints

- Work in worktree `~/tenax-linonce`, branch `perf/linearize-once-adjoint`, base `main` @ `a0778e3`. Never push to `main`; no PR until the user approves; never arm `--auto` unasked; never pass `--delete-branch`.
- Use this worktree's own interpreter `.venv/bin/python`, never bare `python` or the main clone's venv (bare `python` imports the main clone's tenax).
- Run tests with `JAX_PLATFORMS=cpu taskset -c 0-15`, one suite at a time (the host also runs a 131 GB job; never run two suites at once).
- Never pipe a long run to `tail`; redirect to a file and read the file.
- GPU work uses `CUDA_VISIBLE_DEVICES=1` only. GPU 0 runs the S5 job (pid 869113): do not touch it.
- Gradients must equal today's to round-off: parity tests use `rtol=1e-10, atol=1e-12` against `adjoint_chain_rule="separate"` unless a step says otherwise.
- Count compiles with `/jax/core/compile/jaxpr_trace_duration` events, never `jit._cache_size()` (it reads 0 under pytest-cov).
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
  ```
- Pre-commit hooks (ruff, ruff-format) must pass: `~/.local/bin/pre-commit` is installed in the shared `.git`. After a hook failure, re-stage and commit again; never `--reuse-message=HEAD`.
- Mutation runs: commit first; each mutant anchor must occur exactly once; `PYTHONDONTWRITEBYTECODE=1`; delete `__pycache__` under `src/tenax/algorithms` before each run; restore with `git checkout -- <file>` after each.
- Not in scope: `"separate"` and its programs (`_jit_fused_fixed_point_bwd`, `_jit_gmres_one_matvec_solve`, `_jit_chain_rule`, `_jit_apply_Jt`); a joint energy VJP; the forward CTM; the split CTM; C4v.

## Review Focus

1. A complex-valued (complex128) state on the default path: the Partial's residuals are complex and the cotangent pairing must stay the one today's `jax.vjp` used — the existing `test_joint_gmres_matches_separate_complex` covers the gmres method; Task 3 Step 1 adds the fixed-point complex case.
2. A chi bump inside the run (`chi_post != chi`): `_jit_linearize_joint` recompiles for the new static `chi`, and the consumers must take the new Partial, not a stale one — Task 3 test `test_a_second_backward_with_new_values_matches_separate` pins stale reuse; a chi-bump run is exercised by the existing `test_vjp_cache_key_1049.py` end-to-end tests once Task 3 Step 6 adds the new names to `_BACKWARD_FNS`.
3. `projector_backward="flow"` (the #1028 default under `bond_phase`): its custom VJP residuals go into the Partial — the existing `test_joint_gmres_matches_separate_flow` covers gmres; Task 3 Step 1 adds the fixed-point flow case.
4. The Neumann loop diverges or runs out of iterations under `"joint"`: the joint GMRES still follows from `rhs` with the same Partial — covered by the existing `test_joint_fallback_matches_separate` (seed 2 trips the divergence guard).
5. A 2-site checkerboard cell (two site tensors, the production `optimize_gs_ad` shape): covered by the existing `test_joint_fused_matches_separate_two_site` and by Task 3's slow SU-state U(1) test.

---

### Task 1: Extract the Neumann loop into `_neumann_loop`

A pure refactor: the `lax.while_loop` body of `_jit_fused_fixed_point_bwd` moves to a module-level function that takes `apply_Jt` as an argument. Task 3's `_jit_neumann_bwd` reuses it.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (module level, after `_jit_gmres_joint_solve` at ~line 270; and the body of `_jit_fused_fixed_point_bwd` at ~lines 2188–2246)
- Create: `tests/test_linearize_once_adjoint.py`
- Modify: `tests/conftest.py` (`_FILE_MARKERS`)

**Interfaces:**
- Consumes: nothing.
- Produces: `_neumann_loop(apply_Jt, dE_denv, init_lam, use_init, *, tol, maxiter) -> (diverged, converged, n_iter, lam_final, abs_resid, b_norm)`, where `apply_Jt: tuple -> tuple` returns `J^T v`, `dE_denv`/`init_lam` are tuples of arrays, `use_init` is a traced bool array, `tol: float`, `maxiter: int`. Same semantics and return order as the first six outputs of today's `_jit_fused_fixed_point_bwd`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_linearize_once_adjoint.py`:

```python
"""Linearise once, jit only the transpose, for the implicit-AD adjoint.

The default backward (``adjoint_chain_rule="joint"``) builds the gauge-fixed
sweep's pullback once (``_jit_linearize_joint``) and hands it to the Neumann
loop (``_jit_neumann_bwd``) and the joint GMRES solve
(``_jit_gmres_joint_solve``), which then compile only the transpose.  The
oracle is ``"separate"``, whose programs are unchanged.  Spec:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md
"""

from __future__ import annotations

import collections
import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _block_sparse_jit
from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

# ----------------------------------------------------------- _neumann_loop


def _contractive(rho, n=6, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    M *= rho / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(n)
    return M, b


def _loop(M, b, init=None, use_init=False, tol=1e-13, maxiter=500):
    J = jnp.asarray(M)
    init = jnp.zeros(len(b)) if init is None else jnp.asarray(init)
    return cea._neumann_loop(
        lambda v: (J.T @ v[0],),
        (jnp.asarray(b),),
        (init,),
        jnp.array(use_init),
        tol=tol,
        maxiter=maxiter,
    )


def test_the_neumann_loop_solves_a_contractive_system():
    M, b = _contractive(0.5)
    diverged, converged, n_iter, lam, abs_resid, b_norm = _loop(M, b)
    assert bool(converged) and not bool(diverged)
    assert int(n_iter) > 5  # regime: it iterated
    np.testing.assert_allclose(
        np.asarray(lam[0]), np.linalg.solve(np.eye(6) - M.T, b), rtol=1e-10
    )
    np.testing.assert_allclose(float(b_norm), np.linalg.norm(b), rtol=1e-14)
    assert float(abs_resid) < 1e-12


def test_the_neumann_loop_flags_divergence():
    M, b = _contractive(1.5)
    diverged, converged, n_iter, *_ = _loop(M, b)
    assert bool(diverged) and not bool(converged)
    assert int(n_iter) <= 10


def test_the_warm_start_is_used_only_when_asked():
    M, b = _contractive(0.5)
    exact = np.linalg.solve(np.eye(6) - M.T, b)
    n_warm = int(_loop(M, b, init=exact, use_init=True)[2])
    n_cold = int(_loop(M, b, init=exact, use_init=False)[2])
    assert n_warm == 1
    assert n_cold > 5
```

Add to `_FILE_MARKERS` in `tests/conftest.py`, next to `"test_joint_chain_rule.py": "algorithm",`:

```python
    "test_linearize_once_adjoint.py": "algorithm",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t1_red.log 2>&1; tail -5 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t1_red.log`
Expected: 3 failed, each with `AttributeError: module 'tenax.algorithms._ctm_energy_ad' has no attribute '_neumann_loop'`.

- [ ] **Step 3: Move the loop**

In `src/tenax/algorithms/_ctm_energy_ad.py`, add after `_jit_gmres_joint_solve` (before the `# #1028: the flowing projector backward` comment):

```python
def _neumann_loop(apply_Jt, dE_denv, init_lam, use_init, *, tol, maxiter):
    """The adjoint fixed point ``λ_{k+1} = b + Jᵀλ_k``, ``b = dE_denv``, as one
    ``lax.while_loop``.

    Shared by ``_jit_fused_fixed_point_bwd`` (``"separate"``) and
    ``_jit_neumann_bwd`` (``"joint"``), so both chain rules run the same loop.
    ``apply_Jt`` returns ``Jᵀ v`` (not ``(I - Jᵀ) v``).  Cold start
    (``use_init`` false) seeds from ``dE_denv``; ``use_init`` is traced, so warm
    and cold calls share one compiled program.  Returns
    ``(diverged, converged, n_iter, lam_final, abs_resid, b_norm)``; see
    ``_jit_fused_fixed_point_bwd`` for what each one means.
    """
```

Then cut the body of `_jit_fused_fixed_point_bwd` from the line
`        # --- 3. Adjoint fixed-point inside lax.while_loop ---`
through the line
`        b_norm = jnp.sqrt(sum(jnp.sum(jnp.abs(b) ** 2) for b in dE_denv)).astype(`
and its continuation `            real_dtype\n        )`, and paste it as the body of `_neumann_loop`, dedented by 4 spaces. In the pasted body make exactly these two substitutions:

- `tol_arr = jnp.array(gmres_tol, dtype=real_dtype)` → `tol_arr = jnp.array(tol, dtype=real_dtype)`
- `maxiter_arr = jnp.array(gmres_maxiter, dtype=jnp.int32)` → `maxiter_arr = jnp.array(maxiter, dtype=jnp.int32)`

End `_neumann_loop` with:

```python
    return diverged, converged, n_iter, lam_final, abs_resid, b_norm
```

In `_jit_fused_fixed_point_bwd`, in place of the cut block, put:

```python
        # --- 3. Adjoint fixed-point inside lax.while_loop ---
        diverged, converged, n_iter, lam_final, abs_resid, b_norm = _neumann_loop(
            apply_Jt, dE_denv, init_lam, use_init, tol=gmres_tol, maxiter=gmres_maxiter
        )
```

and keep its existing last line
`        return diverged, converged, n_iter, lam_final, abs_resid, b_norm, dE_denv`.

- [ ] **Step 4: Run the new tests and the fused-path regression set**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py tests/test_shared_chain_rule_bwd.py tests/test_ctm_implicit_warm_start_adjoint.py tests/test_joint_chain_rule.py -m "not slow" -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t1_green.log 2>&1; tail -5 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t1_green.log`
Expected: all pass (0 failed). Record the pass count in the ledger.

- [ ] **Step 5: Commit**

```bash
cd ~/tenax-linonce
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_linearize_once_adjoint.py tests/conftest.py
git commit -m "refactor(ctm-ad): the Neumann adjoint loop is one shared function

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 2: `_fusion_variant` picks the CPU variant from the site tensors

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (import line 24; new function after `_neumann_loop`)
- Test: `tests/test_linearize_once_adjoint.py`

**Interfaces:**
- Consumes: `tenax.algorithms._block_sparse_jit._wants_no_fusion(args, kwargs, pinned=None, by_bond_dim=True) -> bool` (exists).
- Produces: `_fusion_variant(prog, params_data_tuple, *, by_bond_dim: bool)` returning `prog.block_sparse` or `prog.dense`, where `prog` is any `block_sparse_jit`-wrapped function.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_linearize_once_adjoint.py`:

```python
# --------------------------------------------------------- _fusion_variant

needs_no_fusion = pytest.mark.skipif(
    jax.default_backend() != "cpu"
    or not _block_sparse_jit.JIT_TAKES_COMPILER_OPTIONS
    or not _block_sparse_jit.no_fusion_compiles(),
    reason="the no-fusion variant is CPU-only and needs compiler_options",
)


@pytest.fixture
def any_size(monkeypatch):
    """Drop the size threshold, so a small U(1) state dispatches as a large one."""
    monkeypatch.setattr(_block_sparse_jit, "NO_FUSION_MIN_SITE_BLOCKS", 1)


@needs_no_fusion
@pytest.mark.parametrize("by_bond_dim", [True, False])
def test_a_large_symmetric_site_picks_the_no_fusion_variant(any_size, by_bond_dim):
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(prog, (A, B), by_bond_dim=by_bond_dim)
    assert got is prog.block_sparse


@needs_no_fusion
def test_dense_sites_pick_the_default_variant(any_size):
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(
        prog, (_wrap_as_dense_tensor(jnp.ones((2, 2, 2, 2, 2))),), by_bond_dim=True
    )
    assert got is prog.dense
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -k "variant" -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t2_red.log 2>&1; tail -5 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t2_red.log`
Expected: 3 failed with `AttributeError: ... has no attribute '_fusion_variant'`.

- [ ] **Step 3: Implement**

Change line 24 of `_ctm_energy_ad.py` from
`from tenax.algorithms._block_sparse_jit import block_sparse_jit`
to
`from tenax.algorithms._block_sparse_jit import _wants_no_fusion, block_sparse_jit`

Add after `_neumann_loop`:

```python
def _fusion_variant(prog, params_data_tuple, *, by_bond_dim):
    """``prog.block_sparse`` when the site tensors call for it, else ``prog.dense``.

    A program that takes the sweep's pullback (a ``jax.tree_util.Partial`` of
    plain arrays) has no ``SymmetricTensor`` argument, so ``block_sparse_jit``'s
    own dispatch would always pick the fused CPU variant.  The site tensors
    decide instead, by the rule the program would have used on
    ``(params, env)``: ``by_bond_dim=True`` for a joint program, ``False`` for
    one that is the sweep VJP only.
    """
    if _wants_no_fusion((params_data_tuple,), {}, None, by_bond_dim):
        return prog.block_sparse
    return prog.dense
```

- [ ] **Step 4: Run them to verify they pass**

Run the Step 2 command again.
Expected: 3 passed (or 3 skipped on a host without the no-fusion option; then record that in the ledger and treat the task as unverified on this host).

- [ ] **Step 5: Commit**

```bash
cd ~/tenax-linonce
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_linearize_once_adjoint.py
git commit -m "feat(ctm-ad): pick a backward program's CPU variant from the site tensors

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 3: Linearise once; the Neumann loop and the joint GMRES take the pullback

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`:
  - module level: re-sign `_jit_gmres_joint_solve` (~line 238); add `_jit_neumann_bwd` after `_fusion_variant`;
  - inside `_make_implicit_vjp_fn`: add `_jit_linearize_joint` after `_gauge_fixed_sweep_joint` (~line 2020); delete `_jit_apply_joint` (~lines 2040–2049);
  - inside `_bwd_core`: the fixed-point branch (~line 2401) and the `adjoint_method == "gmres"` joint branch (~line 2620).
- Modify: `tests/test_vjp_cache_key_1049.py` (`_BACKWARD_FNS`, ~line 207)
- Test: `tests/test_linearize_once_adjoint.py`

**Interfaces:**
- Consumes: `_neumann_loop` (Task 1); `_fusion_variant` (Task 2).
- Produces:
  - `_jit_linearize_joint(params_data_tuple, env_leaves, *, chi, pb) -> vjp_joint`, with `vjp_joint(v: tuple) -> (params_cotangent: tuple, env_cotangent: tuple)`.
  - `_jit_neumann_bwd(vjp_joint, dE_denv, init_lam, use_init, *, tol, maxiter) -> (diverged, converged, n_iter, lam_final, abs_resid, b_norm)`.
  - `_jit_gmres_joint_solve(vjp_joint, rhs, candidates, *, tol, maxiter, restart) -> (lam, abs_resid, b_norm, indirect)` (was `(params_data_tuple, env_leaves, rhs, candidates, *, apply_joint, chi, pb, tol, maxiter, restart)`).
  - Program names in trace events: `_jit_linearize_joint`, `_jit_neumann_bwd`, `_jit_gmres_joint_solve`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_linearize_once_adjoint.py`:

```python
# ------------------------------------------------------- the default backward

#: One gate object: the gate fingerprint is part of the dispatch cache key.
_GATE = heisenberg_gate()

#: tests/test_joint_chain_rule.py's fixture: at chi=4 the fused loop solves
#: seed 0.
_FUSED_SEED = 0

_SOLVERS = ("_jit_neumann_bwd", "_jit_gmres_joint_solve")
_NEW_PROGRAMS = ("_jit_linearize_joint", *_SOLVERS)


@pytest.fixture
def fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.fixture
def spy(monkeypatch):
    """Record every program the backward calls through ``_fusion_variant``:
    ``(name, "block_sparse" | "dense", variant, args, kwargs)``."""
    calls = []
    real = cea._fusion_variant

    def spying(prog, params_data_tuple, *, by_bond_dim):
        variant = real(prog, params_data_tuple, by_bond_dim=by_bond_dim)
        kind = "block_sparse" if variant is prog.block_sparse else "dense"

        def call(*args, **kwargs):
            calls.append((prog.__name__, kind, variant, args, kwargs))
            return variant(*args, **kwargs)

        return call

    monkeypatch.setattr(cea, "_fusion_variant", spying)
    return calls


def _site(seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((2, 2, 2, 2, 2))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A0 = A0 + 0.1j * rng.standard_normal((2, 2, 2, 2, 2))
    A0 = jnp.asarray(A0, dtype=dtype)
    return A0 / jnp.linalg.norm(A0)


def _energy(arrs, mode, *, method="fixed_point", tol=1e-11, neighbors=None, **kw):
    if not isinstance(arrs, (tuple, list)):
        arrs = (arrs,)
    neighbors = neighbors or SINGLE_SITE_NEIGHBORS
    coords = sorted(neighbors)
    return jnp.real(
        ctm_energy_implicit(
            {c: _wrap_as_dense_tensor(a) for c, a in zip(coords, arrs)},
            neighbors,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=method,
            gmres_tol=tol,
            adjoint_chain_rule=mode,
            **kw,
        )
    )


def _grad(arrs, mode, **kw):
    return jax.grad(lambda a: _energy(a, mode, **kw))(arrs)


_SVD = re.compile(r"\bsvd\[")


def _svd_count(call):
    _name, _kind, variant, args, kwargs = call
    jaxpr = jax.make_jaxpr(lambda *a: variant(*a, **kwargs))(*args)
    return len(_SVD.findall(str(jaxpr)))


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_adjoint_solvers_hold_no_forward_sweep(spy):
    """The forward projector SVDs live in the linearise program only.  A
    solver that re-linearised the sweep (main's GMRES matvec did) would carry
    every one of them again."""
    _grad(_site(_FUSED_SEED), "joint")
    by_name = {c[0]: c for c in spy}
    assert set(_NEW_PROGRAMS) <= set(by_name), sorted(by_name)
    forward = _svd_count(by_name["_jit_linearize_joint"])
    assert forward > 0  # regime: the forward sweep has SVDs to leak
    for name in _SOLVERS:
        assert _svd_count(by_name[name]) < forward, name


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_one_backward_linearises_once(spy):
    _grad(_site(_FUSED_SEED), "joint")
    names = [c[0] for c in spy]
    assert names.count("_jit_linearize_joint") == 1, names
    solver_args = [c[3][0] for c in spy if c[0] in _SOLVERS]
    assert len(solver_args) == 2 and solver_args[0] is solver_args[1], names


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_later_backwards_on_one_layout_trace_nothing_new():
    A0 = _site(_FUSED_SEED)
    mark0 = len(_TRACED)
    _grad(A0, "joint")
    assert set(_NEW_PROGRAMS) <= set(_TRACED[mark0:]), sorted(set(_TRACED[mark0:]))
    mark = len(_TRACED)
    for k in (1, 2):
        _grad(A0 * (1.0 + 1e-3 * k), "joint")
    retraced = set(_TRACED[mark:]) & set(_NEW_PROGRAMS)
    assert not retraced, sorted(retraced)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_second_backward_with_new_values_matches_separate():
    """A pullback reused from the first backward would give the first
    state's gradient."""
    A0 = _site(_FUSED_SEED)
    _grad(A0, "joint")
    A1 = 0.9 * A0 + 0.1 * _site(5)
    A1 = A1 / jnp.linalg.norm(A1)
    g_joint = _grad(A1, "joint")
    g_sep = _grad(A1, "separate")
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(dtype=jnp.complex128),
        dict(projector_backward="flow", forward_gauge="bond_phase"),
    ],
    ids=["complex", "flow"],
)
def test_the_fixed_point_path_matches_separate(kw):
    """Review Focus 1 and 3 on the default (fixed-point) method; the gmres
    method's cases live in tests/test_joint_chain_rule.py."""
    kw = dict(kw)
    dtype = kw.pop("dtype", jnp.float64)
    A0 = _site(_FUSED_SEED, dtype=dtype)
    g_sep = _grad(A0, "separate", **kw)
    g_joint = _grad(A0, "joint", **kw)
    if "projector_backward" in kw:  # regime: the flowing backward really ran
        used = cea.get_last_implicit_ad_diagnostics().get("projector_backward_used")
        assert used == "flow", used
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


@needs_no_fusion
@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_solvers_take_the_variant_the_site_tensors_call_for(spy, any_size):
    from tests.test_block_sparse_ctm_ad import _make_u1_symmetric_ipeps

    A_sym = _make_u1_symmetric_ipeps()
    jax.grad(
        lambda A: jnp.real(
            ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _GATE,
                chi=4,
                max_iter=30,
                conv_tol=1e-8,
                gmres_tol=1e-11,
                adjoint_chain_rule="joint",
            )
        )
    )(A_sym)
    kinds = {c[0]: c[1] for c in spy}
    assert kinds == dict.fromkeys(_NEW_PROGRAMS, "block_sparse"), kinds


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_dense_sites_keep_the_default_variant(spy):
    _grad(_site(_FUSED_SEED), "joint")
    kinds = {c[0]: c[1] for c in spy}
    assert kinds == dict.fromkeys(_NEW_PROGRAMS, "dense"), kinds


# ------------------------------------------------------------ SU states (slow)


def _assert_tree_close(a, b, rtol=1e-10, atol=1e-12):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb) and la
    for x, y in zip(la, lb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


@pytest.mark.slow
def test_a_u1_su_state_matches_separate():
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, ipeps, xxz_gate
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    gate = xxz_gate(0.3)
    A0, B0 = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    cfg = iPEPSConfig(
        max_bond_dim=2, num_imaginary_steps=100, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(gate, (A0, B0), cfg, compute_energy=False)
    A, B = A * (1.0 / A.norm()), B * (1.0 / B.norm())

    def grad(mode):
        return jax.grad(
            lambda p: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): p[0], (1, 0): p[1]},
                    CHECKERBOARD_NEIGHBORS,
                    gate,
                    chi=8,
                    max_iter=200,
                    conv_tol=1e-10,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )((A, B))

    _assert_tree_close(grad("joint"), grad("separate"))


@pytest.mark.slow
def test_a_fermionic_su_state_matches_separate():
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _fpeps_simple_update,
        _initialize_fpeps,
        spinless_fermion_gate,
    )

    cfg = FPEPSConfig(D=2, t=1.0, V=1.0, mu=2.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(0))
    B = _initialize_fpeps(cfg, jax.random.PRNGKey(1))
    A, B, _ = _fpeps_simple_update(A, H, max_D=2, dt=0.05, steps=100, B=B)

    def grad(mode):
        return jax.grad(
            lambda p: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): p[0], (1, 0): p[1]},
                    CHECKERBOARD_NEIGHBORS,
                    H,
                    chi=8,
                    max_iter=200,
                    conv_tol=1e-10,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )((A, B))

    _assert_tree_close(grad("joint"), grad("separate"))
```


- [ ] **Step 2: Run them to verify they fail**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -m "not slow" -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_red.log 2>&1; tail -15 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_red.log`
Expected: the Task 1/2 tests pass; these fail: `test_the_adjoint_solvers_hold_no_forward_sweep`, `test_one_backward_linearises_once`, `test_the_solvers_take_the_variant_the_site_tensors_call_for`, `test_dense_sites_keep_the_default_variant` (the spy sees no `_jit_linearize_joint`), and `test_later_backwards_on_one_layout_trace_nothing_new` (its premise: `_jit_linearize_joint` never traced). `test_a_second_backward_with_new_values_matches_separate` and `test_the_fixed_point_path_matches_separate` PASS on main — they are regression guards for Step 3, not RED tests; record that in the ledger.

- [ ] **Step 3: Re-sign `_jit_gmres_joint_solve` and add `_jit_neumann_bwd`**

Replace the whole of `_jit_gmres_joint_solve` (decorator through its `return gmres_single_matvec(...)`) with:

```python
@partial(block_sparse_jit, static_argnames=("tol", "maxiter", "restart"))
def _jit_gmres_joint_solve(vjp_joint, rhs, candidates, *, tol, maxiter, restart):
    """:func:`gmres_single_matvec` on the saved joint linearisation.

    ``vjp_joint`` is the gauge-fixed sweep's pullback over ``(params, env)``
    from ``_jit_linearize_joint``: ``vjp_joint(v) == (J_params^T v,
    J_env^T v)``.  The matvec is ``((I - J_env^T) v, J_params^T v)``, and
    :func:`gmres_single_matvec` (``aux=True``) carries ``J_params^T`` of its
    best iterate, so the chain rule's indirect term comes out of the solve
    and ``_jit_chain_rule`` never compiles (#1123).  This program holds only
    the transpose: the forward sweep compiled once, in the linearise program
    (spec 2026-10-25-linearize-once-adjoint-design.md).  Call it through
    :func:`_fusion_variant` with ``by_bond_dim=True`` (the chain rule's
    rule; this program contains the params-VJP): its arguments carry no
    ``SymmetricTensor``.  ``candidates`` must keep the same pytree structure
    from call to call or it recompiles (#1087).  Returns
    ``(lam, abs_resid, b_norm, indirect)``.
    """

    def matvec(v):
        params_ct, env_ct = vjp_joint(tuple(v))
        return tuple(vi - ji for vi, ji in zip(v, env_ct)), params_ct

    return gmres_single_matvec(
        matvec,
        rhs,
        list(candidates),
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        aux=True,
    )
```

Add after `_fusion_variant`:

```python
@partial(block_sparse_jit, by_bond_dim=False, static_argnames=("tol", "maxiter"))
def _jit_neumann_bwd(vjp_joint, dE_denv, init_lam, use_init, *, tol, maxiter):
    """The Neumann adjoint loop on the saved joint linearisation.

    ``J_env^T v = vjp_joint(v)[1]``; XLA drops the unused params half.  The
    program holds only the transpose.  Call it through
    :func:`_fusion_variant` with ``by_bond_dim=False`` (it is the sweep VJP
    only).  Returns :func:`_neumann_loop`'s tuple.
    """
    return _neumann_loop(
        lambda v: vjp_joint(tuple(v))[1],
        dE_denv,
        init_lam,
        use_init,
        tol=tol,
        maxiter=maxiter,
    )
```

- [ ] **Step 4: Add `_jit_linearize_joint`; delete `_jit_apply_joint`**

Inside `_make_implicit_vjp_fn`, replace the whole `_jit_apply_joint` definition (from `    @partial(block_sparse_jit, static_argnames=("chi", "pb"))\n    def _jit_apply_joint(` through its `return tuple(vi - ji for vi, ji in zip(v, jt_v)), pc`) with:

```python
    @partial(block_sparse_jit, by_bond_dim=False, static_argnames=("chi", "pb"))
    def _jit_linearize_joint(params_data_tuple, env_leaves, *, chi, pb):
        """The gauge-fixed sweep's pullback over ``(params, env)``, once per
        backward.

        Returns :func:`jax.vjp`'s pullback, a ``jax.tree_util.Partial`` whose
        leaves are the residuals.  ``_jit_neumann_bwd`` and
        ``_jit_gmres_joint_solve`` take it as an argument and compile only the
        transpose.  A cached call returns the same pullback function object,
        so they hit their caches too; a new ``chi``/``pb``/layout recompiles
        all three, as it recompiled their predecessors.
        """
        _, vjp_fn = jax.vjp(
            partial(_gauge_fixed_sweep_joint, chi=chi, pb=pb),
            params_data_tuple,
            env_leaves,
        )
        return vjp_fn
```

Run `command grep -n "_jit_apply_joint" src/tenax/algorithms/_ctm_energy_ad.py` — Expected after Step 5: no output.

- [ ] **Step 5: Rewire `_bwd_core`**

(a) In the fixed-point branch, replace

```python
            (
                diverged,
                _converged,
                _n_iter,
                lam_final,
                _abs_resid,
                _b_norm,
                fused_rhs,
            ) = _jit_fused_fixed_point_bwd(
                params_data_tuple, env_leaves, init_lam, use_init, chi=chi_post, pb=pb
            )
```

with

```python
            if adjoint_chain_rule == "joint":
                # Linearise once (spec 2026-10-25): the Neumann loop and the
                # joint GMRES solve below share this pullback and compile only
                # the transpose.
                vjp_joint = pick_variant(_jit_linearize_joint, False)(
                    params_data_tuple, env_leaves, chi=chi_post, pb=pb
                )
                fused_rhs = _eager_dE_denv()
                (
                    diverged,
                    _converged,
                    _n_iter,
                    lam_final,
                    _abs_resid,
                    _b_norm,
                ) = pick_variant(_jit_neumann_bwd, False)(
                    vjp_joint,
                    fused_rhs,
                    init_lam,
                    use_init,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                )
            else:
                (
                    diverged,
                    _converged,
                    _n_iter,
                    lam_final,
                    _abs_resid,
                    _b_norm,
                    fused_rhs,
                ) = _jit_fused_fixed_point_bwd(
                    params_data_tuple,
                    env_leaves,
                    init_lam,
                    use_init,
                    chi=chi_post,
                    pb=pb,
                )
```

and define, at the top of `_bwd_core` right after `run = _run_state(mutables, int(run_token))`:

```python
        def pick_variant(prog, by_bond_dim):
            # Looked up at call time, so tests can spy on _fusion_variant.
            return _fusion_variant(prog, params_data_tuple, by_bond_dim=by_bond_dim)
```

(b) In the `if adjoint_chain_rule == "joint":` block of the fixed-point branch, replace

```python
                lam, _abs, _bn, indirect = _jit_gmres_joint_solve(
                    params_data_tuple,
                    env_leaves,
                    fused_rhs,
                    [seed, fused_rhs],
                    apply_joint=_jit_apply_joint,
                    chi=chi_post,
                    pb=pb,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```

with

```python
                lam, _abs, _bn, indirect = pick_variant(_jit_gmres_joint_solve, True)(
                    vjp_joint,
                    fused_rhs,
                    [seed, fused_rhs],
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```

(c) In the `adjoint_method == "gmres"` branch, replace

```python
            if adjoint_chain_rule == "joint":
                lam, _abs, _bn, indirect = _jit_gmres_joint_solve(
                    params_data_tuple,
                    env_leaves,
                    rhs,
                    [warm, rhs],
                    apply_joint=_jit_apply_joint,
                    chi=chi_post,
                    pb=pb,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```

with

```python
            if adjoint_chain_rule == "joint":
                vjp_joint = pick_variant(_jit_linearize_joint, False)(
                    params_data_tuple, env_leaves, chi=chi_post, pb=pb
                )
                lam, _abs, _bn, indirect = pick_variant(_jit_gmres_joint_solve, True)(
                    vjp_joint,
                    rhs,
                    [warm, rhs],
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```

The rest of `_bwd_core` (diagnostics, `_report_adjoint`, `prev_lam_leaves`, `_jit_direct`, `_combine`, the `"separate"` fallback) stays as is.

- [ ] **Step 6: Register the new programs in the cache-key test**

In `tests/test_vjp_cache_key_1049.py`, change `_BACKWARD_FNS` to:

```python
_BACKWARD_FNS = {
    "_jit_fused_fixed_point_bwd",
    "_jit_dE_denv",
    "_jit_chain_rule",
    "_jit_apply_Jt",
    "_jit_gmres_one_matvec_solve",
    "_jit_gmres_joint_solve",  # the default chain rule ("auto" -> "joint")
    "_jit_linearize_joint",  # its pullback feeds the two solvers below
    "_jit_neumann_bwd",
    "_jit_direct",
}
```

- [ ] **Step 7: Run the task's tests**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py tests/test_joint_chain_rule.py tests/test_vjp_cache_key_1049.py tests/test_shared_chain_rule_bwd.py tests/test_ctm_implicit_warm_start_adjoint.py tests/test_adjoint_seed_858.py tests/test_gmres_adjoint_no_recompile_1087.py -m "not slow" -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_green.log 2>&1; tail -15 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_green.log`
Expected: 0 failed.

If `test_the_adjoint_solvers_hold_no_forward_sweep` fails on `forward > 0` (the CPU forward uses no `svd` primitive), print the primitive names of the linearise jaxpr (`sorted({e.primitive.name for e in jax.make_jaxpr(...)(...).eqns})` recursively is not needed — search `str(jaxpr)` for `eigh[`/`qr[`), switch `_SVD` to the decomposition the forward does use, and ledger a ruling.

- [ ] **Step 8: Run the slow SU-state tests**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py tests/test_joint_chain_rule.py -m slow -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_slow.log 2>&1; tail -10 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_slow.log`
Expected: 0 failed. If `ctm_energy_implicit` refuses the fermionic gate (an exception before any gradient), ledger `Ruling: the fermionic SU parity runs through <the call that works>` after finding the entry point `optimize_gs_ad` uses for a 2-site `FermionicU1` state (`command grep -n "ctm_energy_implicit\|_make_implicit_vjp_fn" src/tenax/algorithms/ipeps_optimize.py`), and call that entry point in the test instead.

- [ ] **Step 9: Run the core gate**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest -m core -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_core.log 2>&1; tail -5 /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_t3_core.log`
Expected: 0 failed (main at `a0778e3` gives the pass/skip/xfail counts to compare against; record both in the ledger).

- [ ] **Step 10: Commit**

```bash
cd ~/tenax-linonce
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_linearize_once_adjoint.py tests/test_vjp_cache_key_1049.py
git commit -m "perf(ctm-ad): linearise the sweep once; the adjoint solvers compile only the transpose

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 4: Mutation test the new tests

**Files:**
- Modify (temporarily, restored after each mutant): `src/tenax/algorithms/_ctm_energy_ad.py`
- Record: the plan's ledger (`.superpowers/sdd/2026-10-25-linearize-once-adjoint/progress.md`)

**Interfaces:**
- Consumes: Task 3's committed code and tests.
- Produces: one ledger line per mutant: `Task 4: mutant Mn <killed|SURVIVED> by <test names>`.

Mutant command (run from `~/tenax-linonce`, after applying each mutant):

```bash
find src/tenax/algorithms -name __pycache__ -type d -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py tests/test_joint_chain_rule.py -m "not slow" -q -p no:cacheprovider > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_mut_$M.log 2>&1
command grep -E "^(FAILED|ERROR)|passed|failed" /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_mut_$M.log | tail -12
git checkout -- src/tenax/algorithms/_ctm_energy_ad.py
```

Before each mutant, check its anchor is unique: `command grep -c -F '<anchor>' src/tenax/algorithms/_ctm_energy_ad.py` must print `1`.

- [ ] **Step 1: M1 — the pullback carries primals, so the solvers re-linearise inside their loops**

Anchor: the line `        return vjp_fn` inside `_jit_linearize_joint` (`command grep -c '        return vjp_fn$'` must print 1; if not, edit by the line number inside `_jit_linearize_joint`). Replace with:

```python
        return jax.tree_util.Partial(
            lambda p, e, v: jax.vjp(
                partial(_gauge_fixed_sweep_joint, chi=chi, pb=pb), p, e
            )[1](v),
            params_data_tuple,
            env_leaves,
        )
```

Run with `M=M1`. Expected: killed by `test_the_adjoint_solvers_hold_no_forward_sweep`.

- [ ] **Step 2: M2 — the Neumann loop takes the params half**

Anchor: `        lambda v: vjp_joint(tuple(v))[1],`. Replace `[1]` with `[0]`. Run with `M=M2`. Expected: killed (shape error or parity failure) by `test_the_fixed_point_path_matches_separate` and `tests/test_joint_chain_rule.py::test_joint_fused_matches_separate`.

- [ ] **Step 3: M3 — the pullback is cached across backwards**

Anchor: in the fixed-point branch,
`                vjp_joint = pick_variant(_jit_linearize_joint, False)(` (the first of the two occurrences — use `command grep -n` and edit that line number; the anchor test is the line number plus the text). Replace the 3-line call with:

```python
                vjp_joint = _M3_STALE.setdefault(
                    (chi_post, pb),
                    pick_variant(_jit_linearize_joint, False)(
                        params_data_tuple, env_leaves, chi=chi_post, pb=pb
                    ),
                )
```

and add `_M3_STALE: dict = {}` at module level after `_fusion_variant`. Run with `M=M3`. Expected: killed by `test_a_second_backward_with_new_values_matches_separate`.

- [ ] **Step 4: M4 — the fusion choice ignores the site tensors**

Anchor: `    if _wants_no_fusion((params_data_tuple,), {}, None, by_bond_dim):`. Replace `(params_data_tuple,)` with `()`. Run with `M=M4`. Expected: killed by `test_the_solvers_take_the_variant_the_site_tensors_call_for` and `test_a_large_symmetric_site_picks_the_no_fusion_variant`.

- [ ] **Step 5: M5 — the gauge reference is not under `stop_gradient`**

Anchor: inside `_gauge_fixed_sweep_joint`, the line
`        e_ref = jax.tree.map(jax.lax.stop_gradient, e)` — it occurs in several functions, so edit by line number inside `_gauge_fixed_sweep_joint` (find it with `command grep -n "def _gauge_fixed_sweep_joint" -A12`). Replace with `        e_ref = e`. Run with `M=M5`. Expected: killed by `test_the_fixed_point_path_matches_separate` and/or `tests/test_joint_chain_rule.py` parity tests.

- [ ] **Step 6: Restore and confirm a clean tree**

Run: `cd ~/tenax-linonce && git status --porcelain`
Expected: no output. Any SURVIVED mutant: add or tighten a test that kills it (RED first), commit, re-run that mutant, and ledger the result.

---

### Task 5: Docs — the backward's program list

**Files:**
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md` (the list under "So on CPU some implicit-AD backward programs ... are compiled unfused automatically:", ~lines 301–308)

**Interfaces:**
- Consumes: Task 3's program names.
- Produces: nothing code-facing.

- [ ] **Step 1: Edit the list**

Replace

```markdown
- `dE/denv`, the chain rule and the fused fixed-point backward go unfused
  when a site tensor has a virtual leg (`u`, `d`, `l`, `r`) of dimension 3
  or more, or 20 or more charge blocks.
- The `J^T` product and the two GMRES solves contain only the sweep VJP.
  They go unfused only when a site tensor has 20 or more charge blocks (a
  many-sector U(1) state; a D=3 U(1) site has 35).
```

with

```markdown
- `dE/denv`, the chain rule, the fused fixed-point backward (`"separate"`
  chain rule) and the joint GMRES solve go unfused when a site tensor has a
  virtual leg (`u`, `d`, `l`, `r`) of dimension 3 or more, or 20 or more
  charge blocks.
- The `J^T` product, the eager GMRES solve, the linearise program and the
  Neumann loop contain only the sweep or its VJP.  They go unfused only when
  a site tensor has 20 or more charge blocks (a many-sector U(1) state; a
  D=3 U(1) site has 35).

Under the default `"joint"` chain rule the backward linearises the
gauge-fixed sweep once (`_jit_linearize_joint`) and passes the saved
pullback to the Neumann loop (`_jit_neumann_bwd`) and the joint GMRES solve
(`_jit_gmres_joint_solve`), which compile only the transpose.  Those two
take no `SymmetricTensor` argument, so the backward picks their variant
from the site tensors by the rules above.
```

- [ ] **Step 2: Build check**

Run: `cd ~/tenax-linonce && command grep -n "_jit_neumann_bwd" docs/guide/algorithms/ipeps_ad_paths.md`
Expected: one line.

- [ ] **Step 3: Commit**

```bash
cd ~/tenax-linonce
git add docs/guide/algorithms/ipeps_ad_paths.md
git commit -m "docs(ad-paths): the default backward linearises the sweep once

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 6: GPU benchmark against main, and the acceptance gates

**Files:**
- Create: `benchmarks/bench_linearize_once.py`
- Modify: `docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md` (append §9 Results)

**Interfaces:**
- Consumes: the branch head (Tasks 1–5) and `main` @ `a0778e3`.
- Produces: `base_d3.json/.log`, `head_d3.json/.log` in the scratchpad; the §9 table.

- [ ] **Step 1: Write the benchmark**

Create `benchmarks/bench_linearize_once.py`:

```python
"""Compile and run cost of the implicit-AD backward, for an A/B against main.

U(1) 2-site XXZ (Delta=0.3) from a simple-update state, ``optimize_gs_ad``
for a few L-BFGS steps on the default backward.  Run it from the worktree
whose tenax you measure, with that worktree's interpreter, on GPU 1:

    CUDA_VISIBLE_DEVICES=1 JAX_LOG_COMPILES=1 JAX_ENABLE_COMPILATION_CACHE=false \
      /usr/bin/time -v .venv/bin/python benchmarks/bench_linearize_once.py \
      --D 3 --chi 16 --out OUT.json > OUT.log 2>&1

Per-program compile times come from the log (``JAX_LOG_COMPILES``); the
JSON holds energies, step times and device peak.  Spec:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig, ipeps, optimize_gs_ad  # noqa: E402
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate  # noqa: E402

_COMPILE_RE = re.compile(
    r"Finished XLA compilation of jit\((.+?)\) in ([0-9.eE+-]+) sec"
)


def parse_compiles(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, secs in _COMPILE_RE.findall(text):
        out[name] = out.get(name, 0.0) + float(secs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--su-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parse", help="print the compile table of a log and exit")
    a = ap.parse_args()
    if a.parse:
        table = parse_compiles(open(a.parse).read())
        for name, secs in sorted(table.items(), key=lambda kv: -kv[1])[:12]:
            print(f"{secs:10.1f}  {name}")
        print(f"{sum(table.values()):10.1f}  TOTAL")
        return
    print("devices", jax.devices(), flush=True)
    t0 = time.time()
    A0, B0 = heisenberg_u1sz_init_pair(a.D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=a.D,
        num_imaginary_steps=a.su_steps,
        dt=0.05,
        ctm=CTMConfig(chi=a.chi),
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A, B = A * (1.0 / A.norm()), B * (1.0 / B.norm())
    print(f"MARK su_done {time.time() - t0:.1f} s", flush=True)
    ctm = dataclasses.replace(
        CTMConfig(chi=a.chi),
        conv_tol=1e-10,
        max_iter=1000,
        plateau_patience=None,
        on_unconverged="warn",
    )
    cfg = iPEPSConfig(
        max_bond_dim=a.D,
        unit_cell="2site",
        ctm=ctm,
        gs_num_steps=a.steps,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-12,
        return_history=True,
        gs_verbose=True,
        gs_log_interval=1,
    )
    t1 = time.time()
    print("MARK optimize_start", flush=True)
    _, _envs, E, hist = optimize_gs_ad(xxz_gate(0.3), (A, B), cfg)
    wall = time.time() - t1
    print(f"MARK optimize_done {wall:.1f} s E={float(E):.12f}", flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    out = {
        "D": a.D,
        "chi": a.chi,
        "su_steps": a.su_steps,
        "steps": a.steps,
        "argv": sys.argv,
        "devices": [str(d) for d in jax.devices()],
        "energy_final": float(E),
        "energies": [float(e) for e in hist.get("energies", [])],
        "step_times": [float(s) for s in hist.get("step_times", [])],
        "optimize_wall_s": wall,
        "device_peak_bytes": stats.get("peak_bytes_in_use"),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("MARK json_written", a.out, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke it on CPU at D=2**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu JAX_LOG_COMPILES=1 taskset -c 0-15 .venv/bin/python benchmarks/bench_linearize_once.py --D 2 --chi 8 --su-steps 50 --steps 2 --out /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_smoke.json > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_smoke.log 2>&1; .venv/bin/python benchmarks/bench_linearize_once.py --out x --parse /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/lin_smoke.log`
Expected: the JSON exists with 2 step times; the table lists `_jit_linearize_joint`, `_jit_neumann_bwd` and `_jit_gmres_joint_solve` and does not list `_jit_fused_fixed_point_bwd`.

- [ ] **Step 3: Commit the benchmark**

```bash
cd ~/tenax-linonce
git add benchmarks/bench_linearize_once.py
git commit -m "bench: compile and run cost of the implicit-AD backward

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

- [ ] **Step 4: Make the base worktree**

```bash
cd ~/tenax && git worktree add -q --detach ~/tenax-linonce-base a0778e3
cd ~/tenax-linonce-base && uv venv -q .venv && uv pip install -q -p .venv/bin/python -e ".[cuda13,dev]"
cp ~/tenax-linonce/benchmarks/bench_linearize_once.py ~/tenax-linonce-base/benchmarks/
cd ~/tenax-linonce && ls .venv/bin/python || (uv venv -q .venv && uv pip install -q -p .venv/bin/python -e ".[cuda13,dev]")
```

Expected: both `.venv/bin/python -c "import tenax, jax; print(tenax.__file__, jax.devices())"` print their own worktree's path and list CUDA devices when run with `CUDA_VISIBLE_DEVICES=1`.

- [ ] **Step 5: Run base, then head, on GPU 1 (serially)**

Check first: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` shows GPU 1 free (no other job of this session), and `free -g` shows at least 80 GB available.

```bash
S=/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/linonce
mkdir -p $S
for arm in base head; do
  W=~/tenax-linonce; [ $arm = base ] && W=~/tenax-linonce-base
  (cd $W && CUDA_VISIBLE_DEVICES=1 JAX_LOG_COMPILES=1 JAX_ENABLE_COMPILATION_CACHE=false \
     timeout 21600 /usr/bin/time -v .venv/bin/python benchmarks/bench_linearize_once.py \
     --D 3 --chi 16 --out $S/${arm}_d3.json > $S/${arm}_d3.log 2>&1; echo "$arm rc=$?")
done
```

Run it in the background (it takes hours). Expected: `base rc=0` and `head rc=0`.

- [ ] **Step 6: Read the results and grade the gates**

```bash
S=/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/linonce
for arm in base head; do
  echo "== $arm"; ~/tenax-linonce/.venv/bin/python ~/tenax-linonce/benchmarks/bench_linearize_once.py --out x --parse $S/${arm}_d3.log
  command grep -E "step [0-9]+/3 E=|Maximum resident set size" $S/${arm}_d3.log
  ~/tenax-linonce/.venv/bin/python -c "import json; d=json.load(open('$S/${arm}_d3.json')); print(d['step_times'], d['device_peak_bytes'], d['energies'])"
done
```

Grade, using only the numbers read here:

- Backward compile total = sum of the backward programs' compile times: base = `_jit_fused_fixed_point_bwd` + `_jit_gmres_joint_solve` + `_jit_direct` + `_jit_dE_denv` (if present); head = `_jit_linearize_joint` + `_jit_neumann_bwd` + `_jit_gmres_joint_solve` + `_jit_direct` + `_jit_dE_denv`.
- A1: head / base ≤ 0.7. Above 0.9: NO-GO — stop, report to the user, do not continue to Step 7.
- A2: median(head `step_times[1:]`) ≤ 1.05 × median(base `step_times[1:]`). (Step time is forward + backward; the spec's "warm backward time" is the backward part of it. If A2 fails by less than 10%, report both medians and do not round.)
- A3: step-1 `E=` and `|g|=` in the head log equal base's to rel 1e-8 (E printed to 10 digits: equal digits; `|g|` printed to 4 significant digits: equal). The spec's adjoint-residual check (test 1, "matches main's to 1e-12") lives here, not in a unit test, because only a base run can supply main's value: the step-1 `GMRES ... resid=` values of head and base differ by at most 1e-12.
- A4: head `device_peak_bytes` ≤ 1.2 × base. Above: report it as a finding.

- [ ] **Step 7: Record §9 Results in the spec and commit**

Append to the spec:

```markdown
## 9. Results (<date>, GPU 1 RTX 4070 Ti SUPER, D=3 chi=16, base a0778e3 vs head <sha>)

| | base | head | ratio | gate |
|---|---|---|---|---|
| backward compile total (s) | <n> | <n> | <r> | A1 <PASS/FAIL> |
| median warm step (s) | <n> | <n> | <r> | A2 <PASS/FAIL> |
| step-1 E / abs(g) | <E> / <g> | <E> / <g> | — | A3 <PASS/FAIL> |
| device peak (bytes) | <n> | <n> | <r> | A4 <PASS/FAIL> |
| host max RSS (kB) | <n> | <n> | <r> | — |

Per-program compile (s): base <name=secs, ...>; head <name=secs, ...>.
```

Fill every `<...>` with the numbers read in Step 6 (none may be left; a missing number means the run did not produce it, and the cell says so in words).

```bash
cd ~/tenax-linonce
git add docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md
git commit -m "docs(spec): linearise-once benchmark results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

- [ ] **Step 8: Optional D=4 run**

Only if A1–A3 pass and GPU 1 is free: run the head arm with `--D 4 --chi 16 --steps 1` under `timeout 14400`, and compare its `_jit_linearize_joint` + `_jit_neumann_bwd` + `_jit_gmres_joint_solve` compile against the 2026-10-24 baseline on this host (`_jit_fused_fixed_point_bwd` 913.3 s + `_jit_gmres_joint_solve` 1,284.5 s). Add one row to §9 and commit as in Step 7.
