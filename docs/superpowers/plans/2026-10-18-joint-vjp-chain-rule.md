# Joint-VJP Chain Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in `CTMConfig.adjoint_chain_rule="joint"` under which the implicit-AD CTM backward gets the chain rule's indirect term `J_params^T λ` from the adjoint solve's own matvec, so `_jit_chain_rule` never compiles.

**Architecture:** One VJP of the gauge-fixed CTM sweep over `(params, env)` serves as both the adjoint matvec (`J_env^T v`) and the params cotangent (`J_params^T v`). `gmres_single_matvec` gains an `aux` mode that carries the cotangent of its best iterate; the fused Neumann loop gains a static `joint` flag that carries the cotangent of `λ_{K-1}` and takes the direct term from a joint energy VJP. `_bwd_core` picks the joint programs when the knob says so; the default `"separate"` runs today's programs unchanged.

**Tech Stack:** JAX (`jax.vjp`, `lax.while_loop`, `lax.switch`), tenax `block_sparse_jit`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-joint-vjp-chain-rule-design.md`

## Global Constraints

- Knob: `CTMConfig.adjoint_chain_rule: Literal["separate", "joint"] = "separate"`; any other value raises `ValueError` in `CTMConfig.__post_init__`.
- `"separate"` must trace exactly today's programs; `gmres_single_matvec(aux=False)` keeps its while-loop carry at 16 leaves.
- The knob is part of the `_ctm_energy_implicit_dispatch` cache key.
- PESS callers (`pess_optimize.py`) are not changed; they get the default.
- Joint programs use `block_sparse_jit` with the default `by_bond_dim=True` until Task 5 measures otherwise.
- Fused joint gradient uses `λ_{K-1}` (no extra iteration); `λ_K` is cached as the warm start.
- Run tests with `JAX_PLATFORMS=cpu uv run pytest ...` from `~/tenax-jvp`.
- New test files must be registered in `tests/conftest.py::_FILE_MARKERS`.
- Commits end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg`.

## Review Focus

1. Complex (`complex128`) site tensors under `"joint"` — gradient must equal `"separate"`; pinned in Task 3 (`test_joint_gmres_matches_separate_complex`).
2. Warm start across consecutive backwards under `"joint"` — the second fused solve must start from the cached `λ_K`; pinned in Task 4 (`test_joint_fused_warm_start_is_used`).
3. Multi-site unit cell (params tuple of length 2) — cotangent pytree per site; pinned in Task 4 (`test_joint_fused_matches_separate_two_site`).
4. `projector_backward="flow"` with its #1028 frozen retry — joint programs compile per `pb`; pinned in Task 3 (`test_joint_gmres_matches_separate_flow`).
5. Block-sparse (`SymmetricTensor`) params — cotangent is a `SymmetricTensor` pytree carried through `lax.while_loop`; pinned in Task 4 (`test_joint_fused_matches_separate_u1`, marked `slow`).

---

## File Structure

- `src/tenax/algorithms/_gmres_lax.py` — `gmres_single_matvec(..., aux=False)` (Task 1).
- `src/tenax/algorithms/ipeps_config.py` — the knob + validation (Task 2).
- `src/tenax/algorithms/ipeps_ad_policy.py` — pass the knob through (Task 2).
- `src/tenax/algorithms/_ctm_energy_ad.py` — plumbing + cache key (Task 2); `_jit_gmres_joint_solve`, `_gauge_fixed_sweep_joint`, `_jit_apply_joint`, `_jit_direct`, the GMRES branch (Task 3); the fused `joint` flag + fallback (Task 4).
- `tests/test_gmres_single_matvec.py` — aux tests (Task 1).
- `tests/test_joint_chain_rule.py` — new, `algorithm` marker (Tasks 2–4).
- `benchmarks/bench_chain_rule_compile.py` — new (Task 5).
- `docs/guide/algorithms/ipeps_ad_paths.md`, `CHANGELOG.md` — Task 6.

---

### Task 1: `gmres_single_matvec` aux mode

**Files:**
- Modify: `src/tenax/algorithms/_gmres_lax.py:346-584`
- Test: `tests/test_gmres_single_matvec.py`

**Interfaces:**
- Produces: `gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart, aux=False)`. With `aux=False`: unchanged, returns `(x_tree, abs_resid, b_norm)`. With `aux=True`: `matvec(v_tree)` returns `(Av_tree, a_tree)` with `a` linear in `v`; returns `(x_tree, abs_resid, b_norm, a_of_x_tree)` where `a_of_x_tree == a(x_tree)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gmres_single_matvec.py`)

```python
# --------------------------------------------------------------------------
# aux mode (joint-VJP chain rule): the solver carries a(x) for the returned x
# --------------------------------------------------------------------------

_P = jnp.asarray(np.random.default_rng(7).standard_normal((5, N)))


def _mv_aux(A, P=_P):
    def matvec(v):
        return (A @ v[0],), P @ v[0]

    return matvec


def _solve_aux(A, b, candidates=(), *, tol=1e-12, maxiter=10, restart=30, P=_P):
    return jax.jit(
        lambda bb, cc: gmres_single_matvec(
            _mv_aux(A, P),
            (bb,),
            [(c,) for c in cc],
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            aux=True,
        )
    )(b, list(candidates))


@pytest.mark.parametrize("kind", ["well", "ill", "nonnormal", "complex"])
def test_aux_is_the_linear_map_of_the_returned_iterate(kind):
    A, b = _system(kind)
    P = _P.astype(A.dtype)
    x, res, bn, a = _solve_aux(A, b, P=P)
    np.testing.assert_allclose(np.asarray(a), np.asarray(P @ x[0]), atol=1e-12)
    x0, res0, bn0 = _solve(A, b)
    np.testing.assert_array_equal(np.asarray(x[0]), np.asarray(x0[0]))
    assert float(res) == float(res0) and float(bn) == float(bn0)


def test_aux_follows_a_winning_seed():
    A, b = _system("well")
    exact = jnp.linalg.solve(A, b)
    x, _res, _bn, a = _solve_aux(A, b, [exact], maxiter=0)
    np.testing.assert_allclose(np.asarray(x[0]), np.asarray(exact), atol=1e-14)
    np.testing.assert_allclose(np.asarray(a), np.asarray(_P @ exact), atol=1e-12)


def test_aux_at_the_maxiter_cap():
    A, b = _system("ill")
    x, res, bn, a = _solve_aux(A, b, maxiter=1, restart=2)
    assert float(res) > 1e-12 * float(bn)  # premise: the cap stopped it
    np.testing.assert_allclose(np.asarray(a), np.asarray(_P @ x[0]), atol=1e-12)


def test_aux_with_maxiter_zero_and_no_seed_is_zero():
    A, b = _system("well")
    x, _res, _bn, a = _solve_aux(A, b, maxiter=0)
    assert not np.any(np.asarray(x[0]))
    assert not np.any(np.asarray(a))


def test_aux_stays_paired_with_x_when_the_operator_is_nan():
    A, b = _system("well")
    A = A.at[0, 0].set(jnp.nan)
    x, res, _bn, a = _solve_aux(A, b)
    assert np.isnan(float(res))
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(_P @ x[0]), atol=1e-12, equal_nan=True
    )


def _while_carry_sizes(fn, *args):
    jx = jax.make_jaxpr(fn)(*args)
    return [
        len(e.invars) - e.params["cond_nconsts"] - e.params["body_nconsts"]
        for e in jx.jaxpr.eqns
        if e.primitive.name == "while"
    ]


def test_aux_off_keeps_todays_loop_carry():
    """``aux=False`` must not grow the loop state: 16 carry leaves, measured
    on main ed5b8cf before aux existed."""
    A, b = _system("well")
    sizes = _while_carry_sizes(
        lambda bb: gmres_single_matvec(
            _mv(A), (bb,), [], tol=1e-10, maxiter=3, restart=2
        ),
        b,
    )
    assert sizes == [16]
    sizes_aux = _while_carry_sizes(
        lambda bb: gmres_single_matvec(
            _mv_aux(A), (bb,), [], tol=1e-10, maxiter=3, restart=2, aux=True
        ),
        b,
    )
    assert sizes_aux == [17]  # + best_aux (one leaf here)


def test_aux_mode_still_applies_the_matvec_at_one_site():
    A, b = _system("well")
    jx = jax.make_jaxpr(
        lambda bb: gmres_single_matvec(
            _mv_aux(A), (bb,), [], tol=1e-10, maxiter=3, restart=2, aux=True
        )
    )(b)
    # Two dot_generals per matvec (A @ v and P @ v), at one call site.
    assert str(jx).count("dot_general") == str(
        jax.make_jaxpr(
            lambda bb: gmres_single_matvec(
                _mv(A), (bb,), [], tol=1e-10, maxiter=3, restart=2
            )
        )(b)
    ).count("dot_general") + 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_gmres_single_matvec.py -k aux -v`
Expected: FAIL with `TypeError: gmres_single_matvec() got an unexpected keyword argument 'aux'`.

- [ ] **Step 3: Implement**

In `gmres_single_matvec`:

1. Signature: `def gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart, aux=False):`. Docstring: add under Args
   ```
        aux: when True, ``matvec`` returns ``(A v, a(v))`` with ``a`` linear,
            and the solver also returns ``a(x)`` for the returned ``x``.  Free:
            ``best_x`` changes only in ``_SEED`` and ``_RESID``, where the
            matvec input is exactly the vector adopted, so ``a(best_x)`` comes
            from the same call and is kept under the same strict-``<`` rule.
            The Krylov basis needs no copy.  Used by the joint-VJP chain rule,
            where ``a(v) = J_params^T v``.
   ```
   and under Returns: ``(x_tree, abs_resid, b_norm)``, plus ``a(x_tree)`` last when ``aux``.
2. The real-`b` complex-data check: when `aux`, evaluate only the first output:
   ```python
        out_shapes = jax.eval_shape(matvec, b_tree)
        if aux:
            out_shapes = out_shapes[0]
   ```
3. Replace `apply` with:
   ```python
    def apply(v):
        out = matvec(unflatten(v))
        if aux:
            y, a = out
            return flatten(y).astype(dtype), a
        return flatten(out).astype(dtype), None
   ```
4. After `coef_idx = ...`:
   ```python
    if aux:
        aux_shape = jax.eval_shape(lambda t: matvec(t)[1], b_tree)
        aux0 = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)

    def keep_aux(s, better, a):
        if not aux:
            return {}
        return {
            "best_aux": jax.tree.map(
                lambda new, old: jnp.where(better, new, old), a, s["best_aux"]
            )
        }
   ```
   and after building `state0`: `if aux: state0["best_aux"] = aux0`.
5. `seed(s, y)` → `seed(s, y, a)`; add `**keep_aux(s, better, a),` to its returned dict. `resid(s, y)` → `resid(s, y, a)`; add `**keep_aux(s, better, a),`. `arnoldi(s, y)` → `arnoldi(s, y, a)` (ignores `a`).
6. Body and return:
   ```python
    def body(s):
        y, a = apply(s["v_in"])  # the one and only matvec call site
        return lax.switch(s["phase"], (seed, resid, arnoldi), s, y, a)

    out = lax.while_loop(lambda s: ~s["done"], body, state0)
    abs_resid = jnp.where(out["nonfinite"], jnp.nan, out["best_res"])
    if aux:
        return unflatten(out["best_x"]), abs_resid, b_norm, out["best_aux"]
    return unflatten(out["best_x"]), abs_resid, b_norm
   ```
   (`lax.switch` with operand `None` is fine: `None` is an empty pytree.)

- [ ] **Step 4: Run the whole file**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_gmres_single_matvec.py -v`
Expected: all PASS (old tests unchanged, new aux tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_gmres_lax.py tests/test_gmres_single_matvec.py
git commit -m "feat(gmres): aux mode carries a linear map of the returned iterate"
```

---

### Task 2: The `adjoint_chain_rule` knob and its plumbing

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py` (field after `adjoint_method` at line 222; validation after line 564)
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py:596`
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (`ctm_energy_implicit` ~619/650/874, `_ctm_energy_implicit_dispatch` ~1336/1406/1454, `_make_implicit_vjp_fn` ~1474/1501)
- Create: `tests/test_joint_chain_rule.py`
- Modify: `tests/conftest.py` (`_FILE_MARKERS`, next to `"test_shared_chain_rule_bwd.py": "algorithm"`)

**Interfaces:**
- Produces: `CTMConfig.adjoint_chain_rule`; `ctm_energy_implicit(..., adjoint_chain_rule: str = "separate")`; `_make_implicit_vjp_fn(..., adjoint_chain_rule="separate")` with the name in scope for `_bwd_core`.

- [ ] **Step 1: Write the failing tests** (`tests/test_joint_chain_rule.py`)

```python
"""Opt-in joint-VJP chain rule (``CTMConfig.adjoint_chain_rule="joint"``).

The implicit-AD backward's chain rule needs ``J_params^T λ``, the params-VJP
of the CTM sweep, which ``_jit_chain_rule`` compiles as its own program -- the
largest single compile at D=3 (2,604 s on GPU).  Under ``"joint"`` the adjoint
solve's matvec is one VJP over ``(params, env)`` and yields that term itself.
These tests pin gradient parity with ``"separate"`` on every backward path and
that the separate chain rule is never compiled.  Spec:
docs/superpowers/specs/2026-10-18-joint-vjp-chain-rule-design.md
"""

from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

#: One gate object: the gate fingerprint is part of the dispatch cache key.
_GATE = heisenberg_gate()

#: Same fixture as tests/test_shared_chain_rule_bwd.py: at chi=4 the fused
#: loop solves seed 0 and its divergence guard fires on seed 2.  Asserted per
#: call, not assumed.
_FUSED_SEED, _FALLBACK_SEED = 0, 2


@pytest.fixture
def fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


def _site(seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((2, 2, 2, 2, 2))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A0 = A0 + 0.1j * rng.standard_normal((2, 2, 2, 2, 2))
    A0 = jnp.asarray(A0, dtype=dtype)
    return A0 / jnp.linalg.norm(A0)


def _energy(arrs, mode, *, method="fixed_point", tol=1e-11, neighbors=None,
            **kw):
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


# ----------------------------------------------------------------- config


def test_default_is_separate():
    assert CTMConfig().adjoint_chain_rule == "separate"


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="adjoint_chain_rule"):
        CTMConfig(adjoint_chain_rule="fused")


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_modes_get_separate_cache_entries():
    A0 = _site(_FUSED_SEED)
    _energy(A0, "separate")
    _energy(A0, "joint")
    keys = list(cea._VJP_CACHE)
    assert len(keys) == 2
    assert {"separate", "joint"} <= {p for k in keys for p in k if isinstance(p, str)}
```

Register the file: in `tests/conftest.py::_FILE_MARKERS` add `"test_joint_chain_rule.py": "algorithm",` right after `"test_shared_chain_rule_bwd.py": "algorithm",`.

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py -v`
Expected: FAIL — `AttributeError: 'CTMConfig' object has no attribute 'adjoint_chain_rule'` and `TypeError: ... unexpected keyword argument 'adjoint_chain_rule'`.

- [ ] **Step 3: Implement**

`ipeps_config.py`, after the `adjoint_method` field:

```python
    # How the implicit-AD backward forms the chain rule's indirect term
    # J_params^T lambda.
    #   "separate" (default) -- its own compiled program, ``_jit_chain_rule``:
    #     the params-VJP of the CTM sweep, the largest single compile at D=3
    #     (2,604 s at chi=16 on GPU).
    #   "joint" -- the adjoint solve's matvec is one VJP over (params, env),
    #     which yields J_params^T of the solve's own iterate; the separate
    #     program never compiles.  Spike at D=2 chi=8 CPU: GMRES-path compile
    #     -37% (Z2 -42%), gradient identical to 2e-16, warm matvec +16%.
    #     Opt-in until measured at D=3 (docs/superpowers/specs/
    #     2026-10-18-joint-vjp-chain-rule-design.md).
    adjoint_chain_rule: Literal["separate", "joint"] = "separate"
```

In `__post_init__`, after the `adjoint_method` check:

```python
        valid_chain_rules = {"separate", "joint"}
        if self.adjoint_chain_rule not in valid_chain_rules:
            raise ValueError(
                f"adjoint_chain_rule must be one of {valid_chain_rules}, "
                f"got {self.adjoint_chain_rule!r}"
            )
```

`ipeps_ad_policy.py:596`, after `adjoint_method=ctm_cfg.adjoint_method,`: `adjoint_chain_rule=ctm_cfg.adjoint_chain_rule,`.

`_ctm_energy_ad.py`:
- `ctm_energy_implicit` signature: add `adjoint_chain_rule: str = "separate",` after `mixing: float = 0.0,`; docstring Args: ``adjoint_chain_rule: ``"separate"`` (default) or ``"joint"``; see ``CTMConfig.adjoint_chain_rule``.``; pass it to `_ctm_energy_implicit_dispatch` as the last positional argument after `mixing,`.
- `_ctm_energy_implicit_dispatch`: add parameter `adjoint_chain_rule="separate",` after `mixing=0.0,`; add `adjoint_chain_rule,` to `cache_key` after `mixing,` with comment `# selects the backward's chain-rule programs`; pass `adjoint_chain_rule=adjoint_chain_rule,` to `_make_implicit_vjp_fn`.
- `_make_implicit_vjp_fn`: add parameter `adjoint_chain_rule: str = "separate",` after `mixing: float = 0.0,`. No use yet.

- [ ] **Step 4: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py tests/test_vjp_cache_key_1049.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_ad_policy.py \
  src/tenax/algorithms/_ctm_energy_ad.py tests/test_joint_chain_rule.py tests/conftest.py
git commit -m "feat(ctm-ad): adjoint_chain_rule knob, plumbed into the dispatch key"
```

---

### Task 3: Joint programs and the `adjoint_method="gmres"` branch

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (module level next to `_jit_gmres_one_matvec_solve` at ~200; inside `_make_implicit_vjp_fn` next to `_jit_apply_Jt` ~1887 and `_jit_chain_rule` ~1923; `_bwd_core` `else:` branch ~2440)
- Test: `tests/test_joint_chain_rule.py`

**Interfaces:**
- Consumes: `gmres_single_matvec(..., aux=True)` (Task 1); `adjoint_chain_rule` in `_make_implicit_vjp_fn` scope (Task 2).
- Produces (inside `_make_implicit_vjp_fn`):
  - `_gauge_fixed_sweep_joint(p_tuple, env_leaves_flat, *, chi, pb) -> tuple[leaves]` (plain function, not jitted).
  - `_jit_apply_joint(params_data_tuple, env_leaves, v, *, chi, pb) -> (tuple (I - J_env^T) v, params-cotangent tuple)`.
  - `_jit_direct(params_data_tuple, env_leaves) -> params-cotangent tuple` (dE/dparams at fixed env, unscaled).
  - `_combine(g, direct, indirect) -> (total,)` with `total = g * (direct + indirect)` leafwise.
- Produces (module level): `_jit_gmres_joint_solve(params_data_tuple, env_leaves, rhs, candidates, *, apply_joint, chi, pb, tol, maxiter, restart) -> (lam, abs_resid, b_norm, indirect)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_joint_chain_rule.py`)

```python
# ----------------------------------------------------------- GMRES method


def test_joint_gmres_matches_separate():
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate", method="gmres")
    g_joint = _grad(A0, "joint", method="gmres")
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-12, atol=1e-14
    )


def test_joint_gmres_matches_separate_complex():
    A0 = _site(_FUSED_SEED, dtype=jnp.complex128)
    g_sep = _grad(A0, "separate", method="gmres")
    g_joint = _grad(A0, "joint", method="gmres")
    assert g_joint.dtype == jnp.complex128
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-12, atol=1e-14
    )


def test_joint_gmres_matches_separate_flow():
    A0 = _site(_FUSED_SEED)
    kw = dict(method="gmres", projector_backward="flow", forward_gauge="bond_phase")
    g_sep = _grad(A0, "separate", **kw)
    used_sep = get_last_implicit_ad_diagnostics().get("projector_backward_used")
    g_joint = _grad(A0, "joint", **kw)
    used_joint = get_last_implicit_ad_diagnostics().get("projector_backward_used")
    assert used_sep == used_joint, (used_sep, used_joint)
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_gmres_never_traces_the_separate_chain_rule():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "joint", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert {"_jit_gmres_joint_solve", "_jit_direct"} <= traced, sorted(traced)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_separate_gmres_never_traces_the_joint_programs():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "separate", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" in traced
    assert not traced & {"_jit_gmres_joint_solve", "_jit_direct", "_jit_apply_joint"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py -k gmres -v`
Expected: the parity tests PASS (joint not wired yet — both modes run separate), `test_joint_gmres_never_traces_the_separate_chain_rule` FAILS (`_jit_chain_rule` traced). That failure is the red bar.

- [ ] **Step 3: Implement**

Module level, after `_jit_gmres_one_matvec_solve`:

```python
@partial(
    block_sparse_jit,
    static_argnames=("apply_joint", "chi", "pb", "tol", "maxiter", "restart"),
)
def _jit_gmres_joint_solve(
    params_data_tuple,
    env_leaves,
    rhs,
    candidates,
    *,
    apply_joint,
    chi,
    pb,
    tol,
    maxiter,
    restart,
):
    """:func:`_jit_gmres_one_matvec_solve` with the joint-VJP matvec.

    ``apply_joint`` returns ``((I - J_env^T) v, J_params^T v)`` from one VJP,
    and :func:`gmres_single_matvec` (``aux=True``) carries ``J_params^T`` of
    its best iterate, so the chain rule's indirect term comes out of the
    solve and ``_jit_chain_rule`` never compiles.  Gated with the default
    ``by_bond_dim=True`` (the chain rule's rule): this program contains the
    params-VJP too.  Returns ``(lam, abs_resid, b_norm, indirect)``.
    """

    def matvec(v):
        return apply_joint(params_data_tuple, env_leaves, v, chi=chi, pb=pb)

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

Inside `_make_implicit_vjp_fn`, after `_jit_chain_rule`:

```python
    def _gauge_fixed_sweep_joint(p_tuple, env_leaves_flat, *, chi, pb):
        """The gauge-fixed sweep as a function of (params, env).

        The gauge reference is ``stop_gradient(env)``, as in ``_jit_apply_Jt``;
        ``_jit_chain_rule`` uses the constant ``envs``, which has no params
        dependence, so both cotangents equal today's programs'.
        """
        st = dict(zip(coords, p_tuple))
        e = jax.tree.unflatten(_cached["env_treedef"], env_leaves_flat)
        e_ref = jax.tree.map(jax.lax.stop_gradient, e)
        e_out, _eps, _smin = jit_step_bwd(
            st,
            e,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=pb,
        )
        return tuple(jax.tree.leaves(_apply_gauge_fix(e_out, e_ref)))

    @partial(block_sparse_jit, static_argnames=("chi", "pb"))
    def _jit_apply_joint(params_data_tuple, env_leaves, v, *, chi, pb):
        """``((I - J_env^T) v, J_params^T v)`` from ONE VJP of the sweep."""
        _, vjp_fn = jax.vjp(
            partial(_gauge_fixed_sweep_joint, chi=chi, pb=pb),
            params_data_tuple,
            env_leaves,
        )
        pc, jt_v = vjp_fn(v)
        return tuple(vi - ji for vi, ji in zip(v, jt_v)), pc

    @block_sparse_jit
    def _jit_direct(params_data_tuple, env_leaves):
        """dE/dparams at fixed env -- the chain rule's direct term, unscaled."""
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        envs = jax.tree.unflatten(_cached["env_treedef"], env_leaves)

        def energy_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            if energy_fn_ is not None:
                return energy_fn_(st, envs, gate_)
            return _default_energy(st, envs, gate_, coords, neighbors)

        _, vjp_fn = jax.vjp(energy_from_params, params_data_tuple)
        return vjp_fn(jnp.ones(()))[0]

    def _combine(g, direct, indirect):
        """``(g * (direct + indirect),)`` -- ``_jit_chain_rule``'s return form."""
        return (jax.tree.map(lambda d, i: g * (d + i), direct, indirect),)
```

In `_bwd_core`, the `else:` (`adjoint_method == "gmres"`) branch: replace the `_jit_gmres_one_matvec_solve(...)` call and the final `return` with

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
            else:
                lam, _abs, _bn = _jit_gmres_one_matvec_solve(
                    ...unchanged arguments...
                )
                indirect = None
```

keep the `_report_adjoint`, diagnostics and `prev_lam_leaves` lines unchanged, and end with

```python
            if indirect is not None:
                direct = _jit_direct(params_data_tuple, env_leaves)
                return _combine(g, direct, indirect), _solved
            return (
                _jit_chain_rule(
                    params_data_tuple, env_leaves, lam_leaves, g, chi=chi_post, pb=pb
                ),
                _solved,
            )
```

- [ ] **Step 4: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py tests/test_gmres_single_matvec_wiring.py tests/test_adjoint_seed_858.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_joint_chain_rule.py
git commit -m "feat(ctm-ad): joint-VJP GMRES adjoint yields the chain rule's indirect term"
```

---

### Task 4: Fused Neumann loop `joint` flag and the GMRES fallback

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (`_jit_fused_fixed_point_bwd` ~1966-2140; `_bwd_core` fixed-point branch ~2263-2440)
- Test: `tests/test_joint_chain_rule.py`

**Interfaces:**
- Consumes: `_gauge_fixed_sweep_joint`, `_jit_apply_joint`, `_combine`, `_jit_gmres_joint_solve` (Task 3).
- Produces: `_jit_fused_fixed_point_bwd(params_data_tuple, env_leaves, init_lam, use_init, *, chi, pb, joint=False)`. `joint=False`: today's 7-tuple. `joint=True`: the 7-tuple plus `(direct, indirect)` — `direct = dE/dparams`, `indirect = J_params^T λ_{K-1}`, both unscaled.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ------------------------------------------------------- fused fixed point


def _fused_premise():
    d = get_last_implicit_ad_diagnostics()
    assert d.get("converged") and not d.get("diverged"), d
    return d


def test_joint_fused_matches_separate():
    """lambda_{K-1} vs lambda_K: within the solve tolerance."""
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate")
    _fused_premise()
    g_joint = _grad(A0, "joint")
    _fused_premise()
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-8, atol=1e-9
    )


def test_joint_fused_at_the_default_tolerance():
    A0 = _site(_FUSED_SEED)
    g_sep = np.asarray(_grad(A0, "separate", tol=1e-6))
    g_joint = np.asarray(_grad(A0, "joint", tol=1e-6))
    assert np.max(np.abs(g_joint - g_sep)) <= 1e-4 * np.max(np.abs(g_sep))


def test_joint_fallback_matches_separate():
    A0 = _site(_FALLBACK_SEED)
    g_sep = _grad(A0, "separate", tol=1e-6)
    d = get_last_implicit_ad_diagnostics()
    assert d.get("diverged") or not d.get("converged"), "premise: fallback ran"
    g_joint = _grad(A0, "joint", tol=1e-6)
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


def test_joint_fused_matches_separate_two_site():
    arrs = (_site(0), _site(1))
    kw = dict(neighbors=CHECKERBOARD_NEIGHBORS)
    g_sep = _grad(arrs, "separate", **kw)
    g_joint = _grad(arrs, "joint", **kw)
    for a, b in zip(g_joint, g_sep):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-7, atol=1e-9)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_fused_warm_start_is_used():
    A0 = _site(_FUSED_SEED)
    _grad(A0, "joint")
    n_cold = _fused_premise()["n_iter"]
    _grad(A0, "joint")
    d = _fused_premise()
    assert not d.get("warm_start_invalidated")
    assert d["n_iter"] < n_cold, (d["n_iter"], n_cold)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_fused_and_fallback_never_trace_the_separate_chain_rule():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "joint")
    _fused_premise()
    _grad(_site(_FALLBACK_SEED), "joint", tol=1e-6)
    d = get_last_implicit_ad_diagnostics()
    assert d.get("diverged") or not d.get("converged"), "premise: fallback ran"
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert "_jit_direct" not in traced, "the fallback must reuse the fused direct"
    assert "_jit_gmres_joint_solve" in traced


@pytest.mark.slow
def test_joint_fused_matches_separate_u1():
    """Block-sparse params: the cotangent is a SymmetricTensor pytree carried
    through the while loop.  Trivial-charge U(1) fixture of
    tests/test_block_sparse_ctm_ad.py (one block, full block-sparse path)."""
    from tests.test_block_sparse_ctm_ad import _make_u1_symmetric_ipeps

    A_sym = _make_u1_symmetric_ipeps()

    def grad(mode):
        return jax.grad(
            lambda A: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): A},
                    SINGLE_SITE_NEIGHBORS,
                    _GATE,
                    chi=4,
                    max_iter=30,
                    conv_tol=1e-8,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )(A_sym)

    g_sep, g_joint = grad("separate"), grad("joint")
    assert type(g_joint) is type(g_sep)
    np.testing.assert_allclose(
        np.asarray(g_joint._data), np.asarray(g_sep._data), rtol=1e-8, atol=1e-9
    )
```

If `from tests.test_block_sparse_ctm_ad import ...` fails (no `tests` package), copy `_make_u1_symmetric_ipeps` into this file verbatim instead.

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py -k "fused or fallback" -v`
Expected: `test_joint_fused_and_fallback_never_trace_the_separate_chain_rule` FAILS (`_jit_chain_rule` traced); parity tests pass trivially.

- [ ] **Step 3: Implement the fused flag**

`_jit_fused_fixed_point_bwd`: decorator `@partial(block_sparse_jit, static_argnames=("chi", "pb", "joint"))`; signature adds `joint=False` after `pb`. Docstring: add a paragraph

```
        ``joint=True`` (the ``adjoint_chain_rule="joint"`` path) differentiates
        the energy and the sweep over ``(params, env)``.  The energy VJP gives
        the direct term with ``dE_denv``; the loop body's one VJP gives
        ``J_env^T lambda_k`` and ``J_params^T lambda_k``, carried as ``pc``.
        On exit ``pc = J_params^T lambda_{K-1}``: the iterate whose residual
        ``diff``/``abs_resid`` certify (the step is ``b - (I - J^T)
        lambda_{K-1}``), at no extra iteration.  ``lambda_K`` is still returned
        for the warm start.  Returns the 7-tuple plus ``(direct, indirect)``.
```

Step 1 of the body becomes:

```python
        if joint:

            def energy_joint(p_tuple, env_leaves_flat):
                st = dict(zip(coords, p_tuple))
                e = jax.tree.unflatten(env_treedef, env_leaves_flat)
                if energy_fn_ is not None:
                    return energy_fn_(st, e, gate_)
                return _default_energy(st, e, gate_, coords, neighbors)

            _, vjp_energy = jax.vjp(energy_joint, params_data_tuple, env_leaves)
            direct, dE_denv = vjp_energy(jnp.ones(()))
        else:
            ...today's energy_from_env / dE_denv code, unchanged...
```

Step 2 becomes:

```python
        if joint:
            _, vjp_joint = jax.vjp(
                partial(_gauge_fixed_sweep_joint, chi=chi, pb=pb),
                params_data_tuple,
                env_leaves,
            )

            def step(v):
                pc_v, jt_v = vjp_joint(v)
                return jt_v, pc_v

        else:
            ...today's gauge_fixed_sweep_from_env / vjp_env_fn...

            def step(v):
                return vjp_env_fn(v)[0], None
```

(delete `apply_Jt`; keep its comment above `step`.)

Carry: after `init_resid`:

```python
        if joint:
            pc_shape = jax.eval_shape(lambda v: step(v)[1], init_lam_local)
            init_pc = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), pc_shape)
        else:
            init_pc = None
```

`cond_fn` unpacks `_lam, prev_diff, k, diverged, _resid, _pc = carry`. `body_fn`:

```python
        def body_fn(carry):
            lam, prev_diff, k, _diverged, _resid, _pc = carry
            jt_lam, pc = step(lam)  # pc = J_params^T lam (None unless joint)
            ...unchanged new_lam / delta / diff / resid / new_diverged...
            return (new_lam, diff, k + jnp.array(1, jnp.int32), new_diverged, resid, pc)
```

While-loop call and return:

```python
        lam_final, final_diff, n_iter, diverged, abs_resid, pc = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (init_lam_local, init_diff, init_k, init_diverged, init_resid, init_pc),
        )
        ...converged / b_norm unchanged...
        out = (diverged, converged, n_iter, lam_final, abs_resid, b_norm, dE_denv)
        if joint:
            # gmres_maxiter=0: the body never ran and pc is zero, but then
            # converged is False and the caller falls back -- never consumed.
            return out + (direct, pc)
        return out
```

- [ ] **Step 4: Implement the `_bwd_core` fixed-point branch**

Call:

```python
            joint = adjoint_chain_rule == "joint"
            fused_out = _jit_fused_fixed_point_bwd(
                params_data_tuple,
                env_leaves,
                init_lam,
                use_init,
                chi=chi_post,
                pb=pb,
                joint=joint,
            )
            (
                diverged,
                _converged,
                _n_iter,
                lam_final,
                _abs_resid,
                _b_norm,
                fused_rhs,
            ) = fused_out[:7]
            fused_direct, fused_indirect = fused_out[7:] if joint else (None, None)
```

Fallback (inside the `if diverged or not converged:` block): replace the `_jit_gmres_one_matvec_solve(...)` call with

```python
                if joint:
                    lam, _abs, _bn, indirect = _jit_gmres_joint_solve(
                        params_data_tuple,
                        env_leaves,
                        rhs,
                        [rhs],
                        apply_joint=_jit_apply_joint,
                        chi=chi_post,
                        pb=pb,
                        tol=gmres_tol,
                        maxiter=gmres_maxiter,
                        restart=gmres_restart,
                    )
                else:
                    lam, _abs, _bn = _jit_gmres_one_matvec_solve(
                        ...unchanged...
                    )
```

and its return with

```python
                if joint:
                    # The fused program already formed the direct term.
                    return _combine(g, fused_direct, indirect), _solved
                return (
                    _jit_chain_rule(...unchanged...),
                    _solved,
                )
```

Happy path: replace `grads_tuple = _jit_chain_rule(...)` with

```python
            if joint:
                grads_tuple = _combine(g, fused_direct, fused_indirect)
            else:
                grads_tuple = _jit_chain_rule(
                    params_data_tuple, env_leaves, lam_final, g, chi=chi_post, pb=pb
                )
```

Update the comment above it: under `"joint"` the indirect term comes from the fused loop's own VJP (λ_{K-1}).

- [ ] **Step 5: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py tests/test_shared_chain_rule_bwd.py tests/test_adjoint_convergence_gate.py tests/test_projector_backward_default_1028.py -v` then `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py -m slow -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_joint_chain_rule.py
git commit -m "feat(ctm-ad): fused Neumann loop carries J_params^T lambda under joint chain rule"
```

---

### Task 5: Compile benchmark driver and the CPU fusion-gating A/B

**Files:**
- Create: `benchmarks/bench_chain_rule_compile.py`
- Modify (only if the A/B says so): decorators of `_jit_gmres_joint_solve`, `_jit_apply_joint`, `_jit_fused_fixed_point_bwd`

**Interfaces:**
- Consumes: `ctm_energy_implicit(..., adjoint_chain_rule=...)`.
- Produces: `python benchmarks/bench_chain_rule_compile.py --mode {separate,joint} --D {2,3} --chi N --sym {dense,u1} --path {fused,fallback,gmres} --out FILE.json`, one mode per process (in-process trace caches contaminate comparisons).

- [ ] **Step 1: Write the driver**

```python
"""Per-program compile and warm time of the implicit-AD backward.

Compares ``adjoint_chain_rule="separate"`` with ``"joint"``.  Run ONE mode per
process: JAX's trace caches make a second in-process run look cheaper.

    JAX_PLATFORMS=cpu uv run python benchmarks/bench_chain_rule_compile.py \
        --mode joint --D 2 --chi 8 --sym dense --path gmres --out joint.json
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

COMPILE = collections.defaultdict(float)
_current = {"fun": "?"}


def _listener(event, duration, **kw):
    if event == "/jax/core/compile/jaxpr_trace_duration":
        _current["fun"] = kw.get("fun_name", "?")
    elif event == "/jax/core/compile/backend_compile_duration":
        COMPILE[_current["fun"]] += duration


jm.register_event_duration_secs_listener(_listener)


def _site(D, sym, seed):
    rng = np.random.default_rng(seed)
    a = jnp.asarray(rng.standard_normal((D, D, D, D, 2)))
    a = a / jnp.linalg.norm(a)
    if sym == "dense":
        return _wrap_as_dense_tensor(a)
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import SymmetricTensor

    u1 = U1Symmetry()
    virt = np.array([(-1) ** i * (i // 2 + i % 2) for i in range(D)], np.int32)
    phys = np.array([1, -1], np.int32)
    flows = (FlowDirection.OUT, FlowDirection.IN, FlowDirection.OUT,
             FlowDirection.IN, FlowDirection.IN)
    labels = ("u", "d", "l", "r", "phys")
    idx = tuple(
        TensorIndex.from_charges(u1, (phys if lab == "phys" else virt).copy(), f, label=lab)
        for f, lab in zip(flows, labels)
    )
    return SymmetricTensor.from_dense(a, idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["separate", "joint"], required=True)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--chi", type=int, default=8)
    p.add_argument("--sym", choices=["dense", "u1"], default="dense")
    p.add_argument("--path", choices=["fused", "fallback", "gmres"], default="gmres")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gate = heisenberg_gate()
    method = "gmres" if a.path == "gmres" else "fixed_point"
    # "fallback": a fused cap of 1 iteration forces the GMRES fallback.
    extra = {"gmres_maxiter": 1} if a.path == "fallback" else {}
    site = _site(a.D, a.sym, a.seed)

    def loss(A):
        return jnp.real(
            ctm_energy_implicit(
                {(0, 0): A}, SINGLE_SITE_NEIGHBORS, gate, chi=a.chi,
                max_iter=60, min_iter=8, conv_tol=1e-10,
                adjoint_method=method, adjoint_chain_rule=a.mode, **extra,
            )
        )

    g = jax.grad(loss)
    t0 = time.perf_counter()
    jax.block_until_ready(g(site))
    first = time.perf_counter() - t0
    diag = dict(get_last_implicit_ad_diagnostics())
    warm = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(g(site))
        warm.append(time.perf_counter() - t0)
    res = {
        "args": vars(a),
        "backend": jax.default_backend(),
        "first_grad_s": first,
        "warm_grad_s": sorted(warm)[len(warm) // 2],
        "compile_s_by_fun": dict(sorted(COMPILE.items(), key=lambda kv: -kv[1])),
        "compile_total_s": sum(COMPILE.values()),
        "diag": {k: v for k, v in diag.items() if isinstance(v, (int, float, bool, str))},
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("first_grad_s", "warm_grad_s", "compile_total_s")}))


if __name__ == "__main__":
    main()
```

Notes for the implementer: `gmres_maxiter` is shared by the fused loop and the GMRES fallback, so `--path fallback` (cap 1) also caps the fallback -- its compile numbers are valid, its warm number is not; say so in the JSON (`"warm_valid": false`).  If `SymmetricTensor.from_dense` rejects the `--sym u1` charges, build the site the way `_make_u1_symmetric_ipeps` in `tests/test_block_sparse_ctm_ad.py` does, with these charges.

- [ ] **Step 2: Smoke-run it**

Run: `JAX_PLATFORMS=cpu uv run python benchmarks/bench_chain_rule_compile.py --mode joint --D 2 --chi 4 --path gmres --out /tmp/claude-1007/-home-yjkao-tenax/9d5185cf-8f92-4cd5-aae3-5ba5b2d21977/scratchpad/jvp_smoke.json`
Expected: prints a JSON line; `compile_s_by_fun` has `_jit_gmres_joint_solve`, no `_jit_chain_rule`. Then `--mode separate`: `_jit_chain_rule` present.

- [ ] **Step 3: Measure (CPU, background, each in its own process)**

For `mode in separate joint`, `path in gmres fused fallback`: D=2 chi=8 dense. For the gating A/B: D=3 chi=8 `--sym u1 --path gmres` and `--path fused` with `--mode joint`, once as committed (`by_bond_dim=True`) and once with the three joint decorators temporarily set to `by_bond_dim=False`. Save JSONs under the scratchpad `inv/jvp/`. Expect hours at D=3; run in the background and do not block other work.

- [ ] **Step 4: Rule and record**

Keep `by_bond_dim=True` unless `False` lowers `compile_total_s` for both the gmres and fused paths at D=3 U(1) by >10%; if it does, set `by_bond_dim=False` on `_jit_gmres_joint_solve` and `_jit_apply_joint` (and on `_jit_fused_fixed_point_bwd` only for `joint=True`, which needs a second decorated copy — prefer leaving the fused one alone). Record the measured numbers in the `_jit_gmres_joint_solve` docstring either way.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_chain_rule_compile.py src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "bench: per-program compile/warm driver for the joint-VJP chain rule"
```

---

### Task 6: Docs and CHANGELOG

**Files:**
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md` (after the GMRES-adjoint paragraph at ~line 360)
- Modify: `CHANGELOG.md` (`## Unreleased` → `### Added`, line 255)

- [ ] **Step 1: Docs paragraph**

```markdown
**Joint-VJP chain rule (opt-in).** `CTMConfig(adjoint_chain_rule="joint")`
takes the chain rule's indirect term `J_params^T λ` from the adjoint solve's
own matvec — one VJP of the CTM sweep over `(params, env)` — so the separate
params-VJP program (`_jit_chain_rule`, the largest single compile at D=3) is
never built.  The GMRES solve returns `J_params^T` of its best iterate; the
fused Neumann loop returns it for `λ_{K-1}`, the iterate its residual
certifies, so its gradient can differ from `"separate"` by O(`gmres_tol`).
Measured at D=2 chi=8 on CPU: GMRES-path compile −37% (Z2 −42%), warm
matvec +16%.  The default stays `"separate"` until measured at D=3.
```

Fill in the D=2 numbers Task 5 measured if they differ from the spike's.

- [ ] **Step 2: CHANGELOG bullet** (top of `### Added` under `## Unreleased`)

```markdown
- **Opt-in joint-VJP chain rule for the implicit-AD backward.**
  `CTMConfig(adjoint_chain_rule="joint")` gets the chain rule's indirect term
  from the adjoint solve's own VJP over `(params, env)`, so the separate
  params-VJP program never compiles. D=2 chi=8 CPU: GMRES-path compile −37%,
  gradient equal to the default to 1e-12 on the GMRES paths and within
  `gmres_tol` on the fused path. Default unchanged (`"separate"`).
```

- [ ] **Step 3: Run the core suite and the new file**

Run: `JAX_PLATFORMS=cpu uv run pytest -m core -q -x` and `JAX_PLATFORMS=cpu uv run pytest tests/test_joint_chain_rule.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/guide/algorithms/ipeps_ad_paths.md CHANGELOG.md
git commit -m "docs: joint-VJP chain rule (opt-in)"
```

---

## After the plan (not tasks)

- Push, open the PR, ask Codex to review (labelled 🤖).
- Hand the peer session (4070 Ti) the D=3 chi=16 GPU run of `bench_chain_rule_compile.py`, both modes, `--path fused` and `--path gmres`.
- Default-flip PR only if D=3 compile drops and warm backward rises ≤ ~20%.
