# Single-matvec GMRES Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The eager-GMRES adjoint path of the implicit CTM-AD backward compiles
the J^T matvec once instead of about four times. The convergence contract and
the gradient (to the solve tolerance) stay the same.

**Architecture:** A new restarted GMRES, `gmres_single_matvec`, runs as one
`lax.while_loop`. Its body applies the matvec exactly once, then a
`lax.switch` on a phase consumes the result. The phases are seed candidate,
true residual and Arnoldi step. `_ctm_energy_ad._bwd_core` calls it through a
new compiled entry point `_jit_gmres_one_matvec_solve`. That entry point
replaces `_jit_eager_gmres_solve`, the separate seed-choice matvec and the
separate residual-measurement matvec. The fused Neumann module also returns
its `dE/denv` and takes the cold seed internally, so the fixed-point path no
longer compiles `_jit_dE_denv`.

**Tech Stack:** Python 3.11+, JAX (`lax.while_loop`, `lax.switch`,
`jax.scipy.linalg.solve_triangular`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-gmres-single-matvec-design.md`

## Global Constraints

- Acceptance is the same tolerance, not bit-identical. The convergence rule
  is `||r|| <= max(tol * ||b||, tol)` (`_adjoint_converged`), and it fails
  closed on NaN.
- The #858 seed rules: zero is the incumbent; a candidate must be strictly
  better; zero wins ties; an earlier candidate beats a later equal one.
- `gmres_maxiter` counts restart cycles of up to `gmres_restart` Arnoldi
  steps (the JAX GMRES meaning).
- Scope: only the two eager-GMRES call sites in `_bwd_core`. Leave
  `_jit_gmres_solve`, `gmres_lax`, `gmres_pytree_jax` and its other callers,
  C4v, `_best_adjoint_seed`, `_measure_adjoint` and
  `_warn_if_adjoint_unconverged` unchanged in behaviour.
- `_jit_apply_Jt` keeps being passed as a static argument (#1087). A second
  backward at the same static configuration must compile nothing.
- Re-point the `_jit_eager_gmres_solve` references in
  `tests/test_adjoint_seed_858.py` and `tests/test_vjp_cache_key_1049.py`. Do
  not delete those guards.
- The supported JAX floor is `jax>=0.4.35`. Use only APIs available there.
  `jax.extend.core.jaxprs_in_params` is not guaranteed there, so the jaxpr
  walk in the tests uses `eqn.params` values directly.
- Run tests with: `cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider <files>`
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg`.
- Do not push. Do not open a PR. The controller asks the user first.

## Review Focus

1. Complex adjoint vectors (complex128 `b`): the solver must use conjugating
   inner products and converge exactly as in the real case. Pinned in
   Task 1, `test_complex_system`.
2. A matvec that returns NaN (a blown-up sweep): the loop must terminate,
   report `abs_resid = nan`, and `_adjoint_converged` must say not converged.
   It must not hang. Pinned in Task 1, `test_nan_matvec_terminates_and_fails_closed`.
3. A zero right-hand side (`b = 0`): no division by zero. Return `x = 0` and
   `abs_resid = 0`. Pinned in Task 1, `test_zero_rhs`.
4. Multi-leaf pytrees with different leaf shapes (CTM env leaves): the
   flatten/unflatten round trip must preserve structure. Pinned in Task 1,
   `test_multi_leaf_pytree`.
5. A warm start present on one backward and absent on the next: the compiled
   solve must not recompile. Missing warm starts are passed as a zeros
   candidate, never `None`. Pinned in Task 2 by the existing
   `tests/test_gmres_adjoint_no_recompile_1087.py`, which must still pass,
   and by `test_gmres_branch_candidate_structure_is_fixed`.

## Ruling recorded against the spec

Spec §2 says the fallback stops calling `_eager_dE_denv()`. The fused branch
also calls it on a cold start (`init_lam = _eager_dE_denv()`), so
`_jit_dE_denv` would still compile on the first backward. To reach the spec's
stated outcome ("`_jit_dE_denv` is not compiled on that path"), Task 2 also
moves the cold seed inside the fused module.

- The fused module takes a `use_init` flag and seeds with its own `dE_denv`
  when the flag is false.
- The arithmetic is identical (same seed value).
- Cost if wrong: one extra compiled module on the first backward. No
  correctness impact.

---

### Task 1: `gmres_single_matvec` solver and unit tests

**Files:**
- Modify: `src/tenax/algorithms/_gmres_lax.py` (append a new function at the
  end of the file)
- Create: `tests/test_gmres_single_matvec.py`
- Modify: `tests/conftest.py`. Add `"test_gmres_single_matvec.py": "core",`
  next to `"test_gmres_adjoint_no_recompile_1087.py": "core",` (about line
  271).

**Interfaces:**
- Produces:
  `gmres_single_matvec(matvec, b_tree, candidates, *, tol: float, maxiter: int, restart: int) -> tuple[pytree, jax.Array, jax.Array]`
  - Returns `(x_tree, abs_resid, b_norm)`.
  - `abs_resid` is the measured `||b - A x||` of the returned `x`, a real
    scalar array.
  - `b_norm` is `||b||`.
  - `candidates` is a list of pytrees shaped like `b_tree`; `None` entries
    are skipped.
  - `maxiter=0` returns the best seed without any Krylov cycle.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gmres_single_matvec.py`:

```python
"""``gmres_single_matvec``: restarted GMRES with one matvec call site.

The implicit-AD adjoint used JAX's GMRES, whose program calls the matvec at
three places (initial residual, Arnoldi step, restart residual).  XLA inlines
each call, so the expensive CTM-sweep VJP was compiled three times there and
once more standalone for the seed choice and residual check: D=2 chi=8 on
CPU, 2,458 s + 435 s against 611 s for the fused Neumann module.  This solver
applies the matvec at exactly one site; these tests pin its numerics, its
#858 seed contract, and the one-site structure itself.
"""

from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import _adjoint_converged
from tenax.algorithms._gmres_lax import gmres_pytree_jax, gmres_single_matvec

N = 24


def _system(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "well":
        A = np.eye(N) + 0.3 * rng.standard_normal((N, N)) / np.sqrt(N)
    elif kind == "ill":
        Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
        A = Q @ np.diag(np.logspace(-6, 0, N)) @ Q.T
    elif kind == "nonnormal":
        A = np.eye(N) + np.triu(2.0 * rng.standard_normal((N, N)), k=1)
    elif kind == "complex":
        A = np.eye(N) + 0.3 * (
            rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        ) / np.sqrt(N)
    else:
        raise ValueError(kind)
    b = rng.standard_normal(N) + (1j * rng.standard_normal(N) if kind == "complex" else 0)
    return jnp.asarray(A), jnp.asarray(b)


def _mv(A):
    def matvec(v):
        return (A @ v[0],)

    return matvec


def _solve(A, b, candidates=(), *, tol=1e-12, maxiter=10, restart=30):
    x, res, bn = jax.jit(
        lambda bb, cc: gmres_single_matvec(
            _mv(A), bb, list(cc), tol=tol, maxiter=maxiter, restart=restart
        )
    )((b,), tuple(candidates))
    return np.asarray(x[0]), float(res), float(bn)


@pytest.mark.parametrize("kind", ["well", "ill", "nonnormal"])
def test_converges_to_the_tolerance(kind):
    A, b = _system(kind)
    tol = 1e-12
    x, res, bn = _solve(A, b, tol=tol, maxiter=20, restart=N)
    true = np.linalg.norm(np.asarray(b) - np.asarray(A) @ x)
    assert res == pytest.approx(true, rel=1e-6, abs=1e-14)
    assert true <= max(tol * bn, tol) * (1 + 1e-6)
    assert _adjoint_converged(res, bn, tol)
    if kind != "ill":
        np.testing.assert_allclose(x, np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-8, atol=1e-10)


def test_complex_system():
    A, b = _system("complex")
    x, res, bn = _solve(A, b, tol=1e-12, maxiter=20, restart=N)
    np.testing.assert_allclose(x, np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-8, atol=1e-10)
    assert _adjoint_converged(res, bn, 1e-12)


def test_small_budget_reports_an_honest_residual():
    A, b = _system("nonnormal")
    tol = 1e-14
    x, res, bn = _solve(A, b, tol=tol, maxiter=1, restart=2)
    true = np.linalg.norm(np.asarray(b) - np.asarray(A) @ x)
    assert res == pytest.approx(true, rel=1e-10)
    assert res > max(tol * bn, tol)
    assert not _adjoint_converged(res, bn, tol)


def test_restart_one_still_converges():
    A, b = _system("well")
    x, res, bn = _solve(A, b, tol=1e-10, maxiter=500, restart=1)
    assert _adjoint_converged(res, bn, 1e-10)


def test_a_seed_worse_than_zero_is_never_chosen():
    A, b = _system("well")
    bad = -10.0 * b  # residual ||b + 10 A b|| >> ||b||
    assert np.linalg.norm(np.asarray(b - A @ bad)) > np.linalg.norm(np.asarray(b))
    x, res, bn = _solve(A, b, [bad], maxiter=0)
    np.testing.assert_array_equal(x, np.zeros(N))
    assert res == pytest.approx(bn, rel=1e-12)


def test_a_good_warm_start_is_chosen_and_used():
    A, b = _system("nonnormal")
    exact = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(b)))
    good = exact + 1e-6 * jnp.ones(N)
    x0, _, _ = _solve(A, b, [good], maxiter=0)
    np.testing.assert_allclose(x0, np.asarray(good), rtol=0, atol=0)
    _, res_from_good, _ = _solve(A, b, [good], tol=1e-15, maxiter=1, restart=2)
    _, res_from_zero, _ = _solve(A, b, [], tol=1e-15, maxiter=1, restart=2)
    assert res_from_good < res_from_zero


def test_ties_go_to_zero():
    b = jnp.zeros(N).at[0].set(1.0)
    A = jnp.eye(N)
    tie = 2.0 * b  # ||b - 2b|| == ||b||
    x, res, bn = _solve(A, b, [tie], maxiter=0)
    np.testing.assert_array_equal(x, np.zeros(N))


def test_the_better_of_two_candidates_is_chosen():
    """Selection is by measured residual, not by position.  (That an earlier
    candidate beats a later *equal* one cannot be observed from the output:
    equal residuals need equal-quality seeds, and the strict ``<`` in the
    seed phase is what pins it, mirrored from ``_best_adjoint_seed``.)"""
    A, b = _system("well")
    exact = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(b)))
    worse = exact + 1e-2 * jnp.ones(N)
    better = exact + 1e-5 * jnp.ones(N)
    for cands in ([worse, better], [better, worse]):
        x, _, _ = _solve(A, b, cands, maxiter=0)
        np.testing.assert_array_equal(x, np.asarray(better))


def test_breakdown_returns_the_exact_solution():
    A = jnp.diag(jnp.arange(1.0, N + 1))
    b = jnp.zeros(N).at[0].set(1.0).at[1].set(1.0)  # invariant 2-d subspace
    x, res, bn = _solve(A, b, tol=1e-14, maxiter=3, restart=N)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, np.asarray(b) / np.arange(1.0, N + 1), atol=1e-13)
    assert res <= 1e-13


def test_zero_rhs():
    A, _ = _system("well")
    x, res, bn = _solve(A, jnp.zeros(N), maxiter=5)
    np.testing.assert_array_equal(x, np.zeros(N))
    assert res == 0.0 and bn == 0.0


def test_nan_matvec_terminates_and_fails_closed():
    _, b = _system("well")

    def nan_matvec(v):
        return (v[0] * jnp.nan,)

    x, res, bn = jax.jit(
        lambda bb: gmres_single_matvec(nan_matvec, bb, [bb], tol=1e-10, maxiter=50, restart=5)
    )((b,))
    assert not _adjoint_converged(float(res), float(bn), 1e-10)


def test_multi_leaf_pytree():
    rng = np.random.default_rng(5)
    shapes = [(3, 4), (5,), (2, 2, 2)]
    n = sum(int(np.prod(s)) for s in shapes)
    A = jnp.asarray(np.eye(n) + 0.2 * rng.standard_normal((n, n)) / np.sqrt(n))
    b = tuple(jnp.asarray(rng.standard_normal(s)) for s in shapes)

    def flat(t):
        return jnp.concatenate([x.ravel() for x in t])

    def matvec(v):
        y = A @ flat(v)
        out, off = [], 0
        for s in shapes:
            k = int(np.prod(s))
            out.append(y[off : off + k].reshape(s))
            off += k
        return tuple(out)

    x, res, bn = jax.jit(
        lambda bb: gmres_single_matvec(matvec, bb, [], tol=1e-12, maxiter=10, restart=n)
    )(b)
    assert [leaf.shape for leaf in x] == shapes
    np.testing.assert_allclose(flat(x), np.linalg.solve(np.asarray(A), np.asarray(flat(b))), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# The structural property this solver exists for.
# ---------------------------------------------------------------------------

_A_STRUCT, _B_STRUCT = _system("well", seed=9)


@jax.jit
def _named_matvec(v):
    return (_A_STRUCT @ v[0],)


def _stablehlo_call_sites(fn):
    text = jax.jit(fn).lower((_B_STRUCT,)).as_text()
    return len(re.findall(r"call @_named_matvec\w*\(", text))


def _jaxpr_call_sites(fn):
    closed = jax.make_jaxpr(fn)((_B_STRUCT,))
    count = 0

    def walk(jaxpr):
        nonlocal count
        for eqn in jaxpr.eqns:
            if eqn.params.get("name") == "_named_matvec":
                count += 1
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(closed.jaxpr)
    return count


def _new(b):
    return gmres_single_matvec(_named_matvec, b, [b], tol=1e-10, maxiter=5, restart=4)


def _old(b):
    return gmres_pytree_jax(_named_matvec, b, b, tol=1e-10, maxiter=5, restart=4)


def test_the_counter_sees_jax_gmres_call_sites():
    """Premise: the counters can see more than one site, else a pass below
    would mean nothing."""
    assert _stablehlo_call_sites(_old) >= 2
    assert _jaxpr_call_sites(_old) >= 2


def test_the_solver_applies_the_matvec_at_exactly_one_site():
    assert _jaxpr_call_sites(_new) == 1
    assert _stablehlo_call_sites(_new) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider tests/test_gmres_single_matvec.py`
Expected: collection ERROR with `ImportError: cannot import name 'gmres_single_matvec'`.

- [ ] **Step 3: Implement the solver**

Append to `src/tenax/algorithms/_gmres_lax.py`. Also add
`import numpy as np` to the imports at the top of that file if it is not
already there.

```python
# Phases of :func:`gmres_single_matvec`.  The loop body applies the matvec
# once and the phase decides what the product means.
_SEED, _RESID, _ARNOLDI = 0, 1, 2


def gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart):
    """Restarted GMRES whose compiled program applies ``matvec`` at ONE site.

    JAX's GMRES calls the matvec at three places (initial residual, Arnoldi
    step, residual after each restart), and XLA inlines every call.  When the
    matvec is a CTM-sweep VJP that is three compiled copies of the most
    expensive program in the backward -- measured at D=2 chi=8 on CPU:
    2,458 s for the GMRES program against 611 s for the fused Neumann module
    with one copy.  Here a single ``lax.while_loop`` body computes
    ``y = matvec(v_in)`` once and a ``lax.switch`` on ``phase`` consumes it:

    * ``_SEED``    -- ``y = A c_k`` for candidate ``k``; keep ``c_k`` if its
      residual is *strictly* below the incumbent's.  The incumbent starts as
      zero at ``||b||``, so zero wins ties and an earlier candidate beats a
      later equal one -- the :func:`_best_adjoint_seed` rules (#858).
    * ``_RESID``   -- ``y = A x``: the true residual.  Stop when
      ``||r|| <= max(tol*||b||, tol)``, when ``maxiter`` cycles are spent, or
      when ``||r||`` is not finite; otherwise start a Krylov cycle.
    * ``_ARNOLDI`` -- ``y = A V[j]``: CGS2 orthogonalisation, Givens update,
      and at the end of a cycle (``restart`` steps, estimated convergence, or
      breakdown) the triangular solve and the update of ``x``.

    Every exit passes through ``_RESID``, so the returned ``abs_resid`` is the
    measured ``||b - A x||`` of the returned ``x``, never an estimate.

    Args:
        matvec: pytree -> pytree, the operator ``A``.
        b_tree: right-hand side pytree.
        candidates: list of seed pytrees shaped like ``b_tree``; ``None``
            entries are skipped.  Zero is always in the running.
        tol: absolute and relative tolerance, as ``_adjoint_converged``.
        maxiter: maximum restart cycles.  ``0`` returns the best seed.
        restart: Arnoldi steps per cycle.

    Returns:
        ``(x_tree, abs_resid, b_norm)``.
    """
    b_leaves, treedef = jax.tree.flatten(b_tree)
    shapes = [leaf.shape for leaf in b_leaves]
    offsets = [int(o) for o in np.cumsum([leaf.size for leaf in b_leaves])[:-1]]

    def flatten(tree):
        return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])

    def unflatten(vec):
        parts = jnp.split(vec, offsets)
        return jax.tree.unflatten(
            treedef, [p.reshape(s) for p, s in zip(parts, shapes)]
        )

    b = flatten(b_tree)
    dtype = b.dtype
    rdtype = jnp.zeros((), dtype).real.dtype
    n = b.shape[0]
    m = int(restart)
    if m < 1:
        raise ValueError(f"restart must be >= 1, got {restart}")

    def apply(v):
        return flatten(matvec(unflatten(v))).astype(dtype)

    cands = [flatten(c).astype(dtype) for c in candidates if c is not None]
    n_cand = len(cands)
    C = jnp.stack(cands) if cands else jnp.zeros((1, n), dtype)

    def norm(v):
        return jnp.sqrt(jnp.sum(jnp.abs(v) ** 2)).astype(rdtype)

    b_norm = norm(b)
    tol_r = jnp.asarray(tol, rdtype)
    thresh = jnp.maximum(tol_r * b_norm, tol_r)
    eps = jnp.finfo(rdtype).eps
    zeros_n = jnp.zeros((n,), dtype)
    basis_idx = jnp.arange(m + 1)
    coef_idx = jnp.arange(m)

    state0 = {
        "phase": jnp.int32(_SEED if n_cand else _RESID),
        "k": jnp.int32(0),
        "v_in": C[0] if n_cand else zeros_n,
        "x": zeros_n,
        "best_x": zeros_n,
        "best_res": b_norm,
        "V": jnp.zeros((m + 1, n), dtype),
        "H": jnp.zeros((m + 1, m), dtype),
        "cs": jnp.zeros((m,), rdtype),
        "sn": jnp.zeros((m,), dtype),
        "g": jnp.zeros((m + 1,), dtype),
        "j": jnp.int32(0),
        "cycles": jnp.int32(0),
        "done": jnp.array(False),
        "abs_resid": jnp.asarray(jnp.inf, rdtype),
    }

    def seed(s, y):
        res = norm(b - y)
        better = res < s["best_res"]  # NaN compares False: never chosen
        best_x = jnp.where(better, C[s["k"]], s["best_x"])
        best_res = jnp.where(better, res, s["best_res"])
        k1 = s["k"] + 1
        last = k1 >= n_cand
        nxt = C[jnp.minimum(k1, max(n_cand - 1, 0))]
        return {
            **s,
            "k": k1,
            "best_x": best_x,
            "best_res": best_res,
            "x": jnp.where(last, best_x, s["x"]),
            "v_in": jnp.where(last, best_x, nxt),
            "phase": jnp.where(last, _RESID, _SEED).astype(jnp.int32),
        }

    def resid(s, y):
        r = b - y
        rn = norm(r)
        done = (rn <= thresh) | (s["cycles"] >= maxiter) | ~jnp.isfinite(rn)
        v0 = r / jnp.where(rn > 0, rn, 1).astype(dtype)
        return {
            **s,
            "abs_resid": rn,
            "done": done,
            "V": jnp.zeros_like(s["V"]).at[0].set(v0),
            "H": jnp.zeros_like(s["H"]),
            "cs": jnp.zeros_like(s["cs"]),
            "sn": jnp.zeros_like(s["sn"]),
            "g": jnp.zeros_like(s["g"]).at[0].set(rn.astype(dtype)),
            "j": jnp.int32(0),
            "v_in": v0,
            "phase": jnp.int32(_ARNOLDI),
        }

    def arnoldi(s, y):
        j = s["j"]
        V = s["V"]
        mask = (basis_idx <= j).astype(dtype)
        # Classical Gram-Schmidt applied twice (CGS2): vectorised over the
        # basis, as stable as modified Gram-Schmidt in practice.
        h1 = mask * (jnp.conj(V) @ y)
        w = y - h1 @ V
        h2 = mask * (jnp.conj(V) @ w)
        w = w - h2 @ V
        hnext = norm(w)
        col = (h1 + h2).at[j + 1].set(hnext.astype(dtype))

        def rotate(i, c):
            a, bb = c[i], c[i + 1]
            ci, si = s["cs"][i], s["sn"][i]
            return c.at[i].set(ci * a + si * bb).at[i + 1].set(
                -jnp.conj(si) * a + ci * bb
            )

        col = lax.fori_loop(0, j, rotate, col)
        a, bb = col[j], col[j + 1]
        denom = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(bb) ** 2)
        safe_denom = jnp.where(denom > 0, denom, 1)
        abs_a = jnp.abs(a)
        phase_a = jnp.where(abs_a > 0, a / jnp.where(abs_a > 0, abs_a, 1), 1)
        c_new = jnp.where(denom > 0, abs_a / safe_denom, 1).astype(rdtype)
        s_new = jnp.where(denom > 0, phase_a * jnp.conj(bb) / safe_denom, 0).astype(
            dtype
        )
        col = col.at[j].set(c_new * a + s_new * bb).at[j + 1].set(0)
        gj = s["g"][j]
        g = s["g"].at[j].set(c_new * gj).at[j + 1].set(-jnp.conj(s_new) * gj)
        H = s["H"].at[:, j].set(col)
        cs = s["cs"].at[j].set(c_new)
        sn = s["sn"].at[j].set(s_new)

        est = jnp.abs(g[j + 1])
        breakdown = hnext <= 10 * eps * norm(y)
        end = (j + 1 >= m) | (est <= thresh) | breakdown

        # End of cycle: solve the leading (j+1)x(j+1) upper-triangular system.
        active = coef_idx <= j
        Hs = jnp.where(
            active[:, None] & active[None, :], H[:m, :m], jnp.eye(m, dtype=dtype)
        )
        coef = jax.scipy.linalg.solve_triangular(
            Hs, jnp.where(active, g[:m], 0), lower=False
        )
        x_new = s["x"] + coef @ V[:m]
        V_next = V.at[j + 1].set(w / jnp.where(hnext > 0, hnext, 1).astype(dtype))
        return {
            **s,
            "H": H,
            "cs": cs,
            "sn": sn,
            "g": g,
            "V": jnp.where(end, V, V_next),
            "x": jnp.where(end, x_new, s["x"]),
            "v_in": jnp.where(end, x_new, V_next[j + 1]),
            "j": jnp.where(end, j, j + 1).astype(jnp.int32),
            "cycles": s["cycles"] + end.astype(jnp.int32),
            "phase": jnp.where(end, _RESID, _ARNOLDI).astype(jnp.int32),
        }

    def body(s):
        y = apply(s["v_in"])  # the one and only matvec call site
        return lax.switch(s["phase"], (seed, resid, arnoldi), s, y)

    out = lax.while_loop(lambda s: ~s["done"], body, state0)
    return unflatten(out["x"]), out["abs_resid"], b_norm
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider tests/test_gmres_single_matvec.py`
Expected: all PASS.

If `test_the_counter_sees_jax_gmres_call_sites` fails, inspect
`jax.jit(_old).lower(...).as_text()` for how nested jits are named in this
JAX version, and adjust the regex in `_stablehlo_call_sites` only. The
premise must hold before the main assertion means anything.

- [ ] **Step 5: Run pre-commit and commit**

```bash
cd ~/tenax-gmres1 && PATH=$HOME/.local/bin:$PATH pre-commit run --files src/tenax/algorithms/_gmres_lax.py tests/test_gmres_single_matvec.py tests/conftest.py
git add src/tenax/algorithms/_gmres_lax.py tests/test_gmres_single_matvec.py tests/conftest.py
git commit -m "feat(gmres): single-matvec restarted GMRES for the CTM adjoint

One lax.while_loop whose body applies the matvec once; a phase switch
consumes it as a seed residual, a true residual or an Arnoldi step.
JAX's GMRES has three inlined call sites.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg"
```

---

### Task 2: Wire the solver into `_bwd_core`

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`
  - imports (line 50);
  - `_jit_eager_gmres_solve` (about lines 195-226);
  - `_measure_adjoint` (about lines 265-291);
  - `_jit_fused_fixed_point_bwd` (about lines 1740-1888);
  - `_bwd_core` (about lines 1960-2250).
- Modify: `tests/test_adjoint_seed_858.py` (about lines 275-375)
- Modify: `tests/test_vjp_cache_key_1049.py` (about line 212)
- Create: `tests/test_gmres_single_matvec_wiring.py`
- Modify: `tests/conftest.py`. Add `"test_gmres_single_matvec_wiring.py": "core",`.

**Interfaces:**
- Consumes: `gmres_single_matvec` from Task 1.
- Produces (module level in `_ctm_energy_ad.py`):
  - `_jit_gmres_one_matvec_solve(params_data_tuple, env_leaves, rhs, candidates, *, apply_Jt, chi, pb, tol, maxiter, restart) -> (lam_tree, abs_resid, b_norm)`;
  - `_report_adjoint(abs_resid: float, b_norm: float, *, tol, maxiter, restart, warn, stacklevel=3) -> tuple[float, bool]`.
- `_jit_fused_fixed_point_bwd(params_data_tuple, env_leaves, init_lam, use_init, *, chi, pb)`:
  - it now returns 7 values, `(diverged, converged, n_iter, lam_final, abs_resid, b_norm, dE_denv)`;
  - `use_init` is a bool array, and when it is false the loop starts from
    `dE_denv`.

- [ ] **Step 1: Write the failing wiring tests**

Create `tests/test_gmres_single_matvec_wiring.py`:

```python
"""The eager-GMRES adjoint compiles the J^T matvec once (single-matvec GMRES).

Before: the GMRES program inlined three matvec call sites and the standalone
``_jit_apply_Jt`` compiled a fourth copy for the seed choice and residual
check; the fixed-point path also compiled a standalone ``_jit_dE_denv`` for
its cold seed and again for its fallback.  These tests count XLA backend
compiles by function name, so they hold whatever the solver's internals are.
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
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_COMPILED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _COMPILED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/backend_compile_duration"
        else None
    )
)

_GATE = heisenberg_gate()
#: Same configuration as tests/test_shared_chain_rule_bwd.py: the fused loop
#: solves seed 0 and gives up on seed 2 (asserted per call below).
_FUSED_SEED, _FALLBACK_SEED = 0, 2


def _energy(A_arr, adjoint_method, gmres_tol=1e-6):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)},
            SINGLE_SITE_NEIGHBORS,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=adjoint_method,
            gmres_tol=gmres_tol,
        )
    )


def _site(seed):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


@pytest.fixture
def _fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_fixed_point_path_compiles_no_standalone_matvec_or_rhs():
    grad = jax.grad(lambda a: _energy(a, "fixed_point"))
    mark = len(_COMPILED)
    grad(_site(_FUSED_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("converged") and not diag.get("diverged"), "premise: fused path"
    grad(_site(_FALLBACK_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("diverged") or not diag.get("converged"), "premise: fallback ran"
    compiled = set(_COMPILED[mark:])
    assert "_jit_gmres_one_matvec_solve" in compiled
    assert "_jit_apply_Jt" not in compiled, sorted(compiled)
    assert "_jit_dE_denv" not in compiled, sorted(compiled)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_gmres_branch_compiles_no_standalone_matvec():
    mark = len(_COMPILED)
    jax.grad(lambda a: _energy(a, "gmres"))(_site(_FUSED_SEED))
    compiled = set(_COMPILED[mark:])
    assert "_jit_gmres_one_matvec_solve" in compiled
    assert "_jit_apply_Jt" not in compiled, sorted(compiled)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_gmres_branch_candidate_structure_is_fixed():
    """Cold (no warm start) and warm backwards share one compiled solve: a
    missing warm start must be passed as zeros, not None, or the pytree
    structure of ``candidates`` changes and the solve recompiles."""
    grad = jax.grad(lambda a: _energy(a, "gmres"))
    grad(_site(_FUSED_SEED))  # cold
    mark = len(_COMPILED)
    grad(_site(_FUSED_SEED) * 1.001)  # warm, same shapes
    assert "_jit_gmres_one_matvec_solve" not in _COMPILED[mark:]


def test_fallback_gradient_matches_gmres_branch():
    a = _site(_FALLBACK_SEED)
    g_fb = jax.grad(lambda x: _energy(x, "fixed_point", gmres_tol=1e-11))(a)
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("diverged") or not diag.get("converged"), "premise: fallback ran"
    g_gm = jax.grad(lambda x: _energy(x, "gmres", gmres_tol=1e-11))(a)
    np.testing.assert_allclose(np.asarray(g_fb), np.asarray(g_gm), rtol=1e-7, atol=1e-10)


def test_fused_gradient_matches_gmres_branch():
    a = _site(_FUSED_SEED)
    g_fp = jax.grad(lambda x: _energy(x, "fixed_point", gmres_tol=1e-11))(a)
    g_gm = jax.grad(lambda x: _energy(x, "gmres", gmres_tol=1e-11))(a)
    np.testing.assert_allclose(np.asarray(g_fp), np.asarray(g_gm), rtol=1e-7, atol=1e-10)
```

Register it in `tests/conftest.py`: add
`"test_gmres_single_matvec_wiring.py": "core",` next to the Task 1 entry.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider tests/test_gmres_single_matvec_wiring.py`
Expected:
- the three compile-count tests FAIL (`_jit_gmres_one_matvec_solve` never
  compiled; `_jit_apply_Jt` compiled);
- the two parity tests PASS. They pin today's behaviour and must keep
  passing.

- [ ] **Step 3: Replace `_jit_eager_gmres_solve` with `_jit_gmres_one_matvec_solve`**

In `src/tenax/algorithms/_ctm_energy_ad.py`:
- Change line 50 to
  `from tenax.algorithms._gmres_lax import gmres_pytree, gmres_single_matvec`.
- Replace the whole `_jit_eager_gmres_solve` definition (the `@partial(...)`
  decorator through `return gmres_pytree_jax(...)`) with:

```python
@partial(
    block_sparse_jit,
    by_bond_dim=False,
    static_argnames=("apply_Jt", "chi", "pb", "tol", "maxiter", "restart"),
)
def _jit_gmres_one_matvec_solve(
    params_data_tuple,
    env_leaves,
    rhs,
    candidates,
    *,
    apply_Jt,
    chi,
    pb,
    tol,
    maxiter,
    restart,
):
    """The eager-GMRES adjoint solve: one compiled program, one J^T matvec.

    Replaces JAX's GMRES here, whose program inlined the matvec at three call
    sites, plus the standalone ``_jit_apply_Jt`` compile that chose the seed
    and measured the residual: four compiled copies of the CTM-sweep VJP,
    2,458 s + 435 s at D=2 chi=8 on CPU against 611 s for the fused Neumann
    module.  :func:`gmres_single_matvec` chooses the seed among
    ``candidates`` and zero by measured residual (#858) and returns the
    measured true residual, all with one matvec call site.

    ``apply_Jt`` is static and stable for the life of the dispatch entry, so a
    later backward with new values reuses the compiled loop (#1087).
    ``candidates`` must keep the same pytree structure from call to call --
    pass zeros, never ``None``, for a missing warm start -- or it recompiles.
    Returns ``(lam, abs_resid, b_norm)``.
    """

    def matvec(v):
        return apply_Jt(params_data_tuple, env_leaves, v, chi=chi, pb=pb)

    return gmres_single_matvec(
        matvec, rhs, list(candidates), tol=tol, maxiter=maxiter, restart=restart
    )
```

- [ ] **Step 4: Add `_report_adjoint` and make `_measure_adjoint` delegate to it**

Replace the body of `_measure_adjoint` (keep its signature and docstring)
and add `_report_adjoint` directly above it:

```python
def _report_adjoint(
    abs_resid: float,
    b_norm: float,
    *,
    tol: float,
    maxiter: int,
    restart: int,
    warn: bool,
    stacklevel: int = 3,
) -> tuple[float, bool]:
    """``(relative residual, converged)`` from a residual already measured.

    The threshold and the warning of :func:`_measure_adjoint`, for a solver
    that returns its own true residual (:func:`gmres_single_matvec`) and so
    needs no extra matvec.  Fails closed on NaN via ``_adjoint_converged``.
    """
    rel = abs_resid / b_norm if b_norm > 0 else abs_resid
    converged = _adjoint_converged(abs_resid, b_norm, tol)
    if warn and not converged:
        warnings.warn(
            f"Implicit-AD CTM: adjoint solve did not converge (relative "
            f"residual {rel:.3e} > gmres_tol {tol:.1e} after gmres_maxiter="
            f"{maxiter} iterations of a {restart}-dimensional Krylov space). "
            "The gradient contracts a λ that does not solve "
            "(I - Jᵀ)λ = dE/denv, so it is wrong by roughly that relative "
            "amount and nothing downstream can tell. Raise gmres_maxiter/"
            "gmres_restart, or loosen gmres_tol if this residual is "
            "acceptable for your use.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return rel, converged
```

New body of `_measure_adjoint`:

```python
    abs_resid, b_norm = _adjoint_residual_and_rhs_norm(matvec, lam, rhs)
    return _report_adjoint(
        abs_resid,
        b_norm,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        warn=warn,
        stacklevel=4,
    )
```

- [ ] **Step 5: Fused module returns `dE_denv` and takes the cold seed internally**

In `_jit_fused_fixed_point_bwd`:
- Change the signature to
  `def _jit_fused_fixed_point_bwd(params_data_tuple, env_leaves, init_lam, use_init, *, chi, pb):`.
- Replace `init_lam_local = init_lam  # lambda_0 (warm or cold seed)` with:

```python
        # Cold start seeds from this program's own dE_denv: passing it in
        # from the standalone _jit_dE_denv compiled a second copy of the
        # energy VJP just to make the seed.  ``use_init`` is traced, so warm
        # and cold calls share this compiled program.
        init_lam_local = tuple(
            jnp.where(use_init, w, d) for w, d in zip(init_lam, dE_denv)
        )
```

- Change the final `return` to
  `return diverged, converged, n_iter, lam_final, abs_resid, b_norm, dE_denv`.
- Add to the Returns section of its docstring:
  `dE_denv : tuple of leaves -- the RHS, returned so the GMRES fallback does not recompute it.`

- [ ] **Step 6: Rewire `_bwd_core`**

In the `adjoint_method == "fixed_point"` branch:
- Replace

```python
            if prev_lam is None:
                init_lam = _eager_dE_denv()
            else:
                init_lam = prev_lam
```

with

```python
            if prev_lam is None:
                init_lam = tuple(jnp.zeros_like(leaf) for leaf in env_leaves)
                use_init = jnp.array(False)
            else:
                init_lam = prev_lam
                use_init = jnp.array(True)
```

- Change the fused call and its unpacking to:

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

- In the fallback block, replace everything from `rhs = _eager_dE_denv()`
  through the `_measure_adjoint(...)` call and its
  `_F3_LAST_DIAGNOSTICS["adjoint_residual"] = _rel` line with:

```python
                rhs = fused_rhs
                _GMRES_LOGGER.debug(
                    "Eager GMRES: maxiter=%d tol=%g restart=%d",
                    gmres_maxiter,
                    gmres_tol,
                    gmres_restart,
                )
                # Seed among [rhs] and zero by measured residual (#858), solve,
                # and measure the true residual -- one compiled program, one
                # matvec call site (gmres_single_matvec).
                lam, _abs, _bn = _jit_gmres_one_matvec_solve(
                    params_data_tuple,
                    env_leaves,
                    rhs,
                    [rhs],
                    apply_Jt=_jit_apply_Jt,
                    chi=chi_post,
                    pb=pb,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
                _rel, _solved = _report_adjoint(
                    float(jax.device_get(_abs)),
                    float(jax.device_get(_bn)),
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                    warn=warn,
                )
                _F3_LAST_DIAGNOSTICS["adjoint_residual"] = _rel
```

In the `else:` (gmres) branch, replace everything from `warm = None` through
the `_measure_adjoint(...)` call and its
`_F3_LAST_DIAGNOSTICS["adjoint_residual"] = _rel` line with:

```python
            # Warm start first, then the Neumann term, then zero -- chosen by
            # measured residual inside the solve (#858).  A missing warm start
            # is passed as zeros, not None: zero ties the incumbent and loses,
            # and the candidates' pytree structure stays fixed so the compiled
            # solve is reused between cold and warm backwards (#1087).
            if prev_lam is not None:
                warm = jax.tree.unflatten(jax.tree.structure(rhs), prev_lam)
            else:
                warm = jax.tree.map(jnp.zeros_like, rhs)
            _GMRES_LOGGER.debug(
                "Eager GMRES: maxiter=%d tol=%g restart=%d warm=%s",
                gmres_maxiter,
                gmres_tol,
                gmres_restart,
                prev_lam is not None,
            )
            lam, _abs, _bn = _jit_gmres_one_matvec_solve(
                params_data_tuple,
                env_leaves,
                rhs,
                [warm, rhs],
                apply_Jt=_jit_apply_Jt,
                chi=chi_post,
                pb=pb,
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
            )
            _rel, _solved = _report_adjoint(
                float(jax.device_get(_abs)),
                float(jax.device_get(_bn)),
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
                warn=warn,
            )
            _F3_LAST_DIAGNOSTICS["adjoint_residual"] = _rel
```

Delete the now-unused `_eager_apply_I_minus_Jt` closure and the comment
block directly above it ("Both eager-GMRES paths ... share the same cached
_jit_apply_Jt matvec ..."). Keep `_eager_dE_denv`: `arnoldi_precheck` and
the gmres branch still use it. Update the `_eager_dE_denv` comment so it says
it is needed by "(a) the Arnoldi precheck, (b) adjoint_method='gmres'", and
drop the mention of the divergence fallback.

- [ ] **Step 7: Re-point the #858 and #1049 tests**

In `tests/test_vjp_cache_key_1049.py`, in `_BACKWARD_FNS`, replace
`"_jit_eager_gmres_solve",` with `"_jit_gmres_one_matvec_solve",`.

In `tests/test_adjoint_seed_858.py`:
- Change the import line to
  `from tenax.algorithms._gmres_lax import gmres_pytree, gmres_pytree_jax, gmres_single_matvec`.
- Replace the spy block in the call-site test (from
  `real = ad._jit_eager_gmres_solve` through the end of its `for` loop) with:

```python
    real = ad._jit_gmres_one_matvec_solve

    def spy(params, env, b_tree, candidates, *, apply_Jt, chi, pb, **kw):
        def matvec(v):
            return apply_Jt(params, env, v, chi=chi, pb=pb)

        seen.append((matvec, b_tree, list(candidates), kw["restart"]))
        return real(
            params, env, b_tree, candidates, apply_Jt=apply_Jt, chi=chi, pb=pb, **kw
        )

    monkeypatch.setattr(ad, "_jit_gmres_one_matvec_solve", spy)

    A0 = _random_D2_site()
    jax.grad(lambda a: _implicit_energy(a, chi=_HARMFUL_SEED_CHI, max_iter=80))(A0)

    assert seen, (
        "no eager GMRES solve ran, so this test observed nothing. The "
        "backward took the fused fixed-point branch instead; re-point the "
        "fixture at the branch that calls _jit_gmres_one_matvec_solve."
    )
    for i, (matvec, b_tree, candidates, restart) in enumerate(seen):
        b_norm = _l2(b_tree)
        neumann = _residual(matvec, b_tree, b_tree)
        assert neumann > b_norm, (
            f"solve {i}: the Neumann seed starts at {neumann:.4e} against "
            f"||b|| = {b_norm:.4e}, i.e. it is *not* harmful here, so this "
            f"test cannot tell a measured seed from an unconditional one. "
            f"Re-scan chi (measured 5.14x at chi={_HARMFUL_SEED_CHI})."
        )
        assert any(
            all(np.array_equal(np.asarray(x), np.asarray(y))
                for x, y in zip(jax.tree.leaves(c), jax.tree.leaves(b_tree)))
            for c in candidates
        ), f"solve {i}: rhs is not among the candidates handed to the solver"
        # The seed the solver itself picks (maxiter=0: seed phase only).
        seed, _, _ = gmres_single_matvec(
            matvec, b_tree, candidates, tol=1e-30, maxiter=0, restart=restart
        )
        seed_resid = _residual(matvec, seed, b_tree)
        assert seed_resid <= b_norm * (1 + 1e-9), (
            f"adjoint solve {i} was seeded at residual {seed_resid:.4e} "
            f"against ||b|| = {b_norm:.4e}. GMRES's residual bound is against "
            f"its *starting* point, so seeding above ||b|| lets it return a "
            f"lambda worse than lambda = 0 -- and it cannot recover if it "
            f"stagnates. See #858."
        )
```

- Rename the comment `# A module-level operator for the wrapper test below:
  ``_jit_eager_gmres_solve``` to name `_jit_gmres_one_matvec_solve`.
- Replace `test_the_compiled_solve_hands_its_seed_to_gmres` with:

```python
def test_the_compiled_solve_hands_its_seed_to_gmres():
    """``_jit_gmres_one_matvec_solve`` must use the caller's candidates.

    A wrapper that dropped them (or substituted ``rhs``) would bring #858
    back with the spy test above still green (Codex review of #1088).  So run
    it with a budget too small to converge, where the answer depends on the
    start: it must equal ``gmres_single_matvec`` from the same good seed and
    differ from the answer started at zero.
    """
    import tenax.algorithms._ctm_energy_ad as ad

    rng = np.random.default_rng(2)
    b = (jnp.asarray(rng.standard_normal(_WRAP_N)),)
    kw = {"tol": 1e-14, "maxiter": 1, "restart": 2}

    def matvec(v):
        return _wrap_apply_I_minus_Jt(None, None, v, chi=0, pb="auto")

    A = np.eye(_WRAP_N) - np.asarray(_WRAP_J).T
    exact = np.linalg.solve(A, np.asarray(b[0]))
    good = (jnp.asarray(exact + 1e-3 * rng.standard_normal(_WRAP_N)),)
    assert _residual(matvec, good, b) < _l2(b), "premise: the seed beats zero"

    got, _, _ = ad._jit_gmres_one_matvec_solve(
        (), (), b, [good], apply_Jt=_wrap_apply_I_minus_Jt, chi=0, pb="auto", **kw
    )
    want, _, _ = gmres_single_matvec(matvec, b, [good], **kw)
    from_zero, _, _ = gmres_single_matvec(matvec, b, [], **kw)

    gap = float(np.linalg.norm(np.asarray(want[0]) - np.asarray(from_zero[0])))
    assert gap > 1e-6, (
        f"premise: with this budget the start changes the answer by only "
        f"{gap:.2e}, so this test could not tell them apart"
    )
    np.testing.assert_allclose(
        np.asarray(got[0]),
        np.asarray(want[0]),
        rtol=1e-10,
        atol=1e-12,
        err_msg="the compiled solve did not start from the candidate it was given",
    )
```

`_residual(matvec, x, b)` and `_l2(tree)` already exist in this file and
are used by the spy test. If either takes different arguments, adapt the
calls to its existing signature. Do not add a new helper.

- [ ] **Step 8: Run the wiring tests and the regression set**

Run:
`cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider tests/test_gmres_single_matvec.py tests/test_gmres_single_matvec_wiring.py tests/test_adjoint_seed_858.py tests/test_vjp_cache_key_1049.py tests/test_gmres_adjoint_no_recompile_1087.py tests/test_shared_chain_rule_bwd.py tests/test_ctm_implicit_warm_start_adjoint.py`
Expected: all PASS.

Then run the wider adjoint set, not slow:
`cd ~/tenax-gmres1 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider -m "not slow" $(grep -l "ctm_energy_implicit\|_ctm_energy_ad" tests/*.py)`
Expected: no failure that does not also fail on `origin/main`. For any
failure, run the same test on a clean `origin/main` checkout before blaming
this change. See the memory note "main fast-other chronic red": six named
CTM/QR tests fail on main on every platform.

- [ ] **Step 9: Commit**

```bash
cd ~/tenax-gmres1 && PATH=$HOME/.local/bin:$PATH pre-commit run --files src/tenax/algorithms/_ctm_energy_ad.py tests/test_gmres_single_matvec_wiring.py tests/test_adjoint_seed_858.py tests/test_vjp_cache_key_1049.py tests/conftest.py
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_gmres_single_matvec_wiring.py tests/test_adjoint_seed_858.py tests/test_vjp_cache_key_1049.py tests/conftest.py
git commit -m "perf(ad): eager-GMRES adjoint compiles the J^T matvec once

_jit_gmres_one_matvec_solve replaces _jit_eager_gmres_solve: seed choice,
solve and true residual in one program with one matvec call site, so the
standalone _jit_apply_Jt no longer compiles on these paths.  The fused
Neumann module returns its dE/denv and seeds cold starts internally, so
the fixed-point path no longer compiles _jit_dE_denv.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg"
```

---

### Task 3: Acceptance measurement, docs, CHANGELOG

**Files:**
- Modify: `CHANGELOG.md` (under `## Unreleased` → `### Behavior Changes`)
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md` (insert before
  `#### Persistent compilation cache`, about line 342)
- Modify: `docs/superpowers/specs/2026-10-16-gmres-single-matvec-design.md`
  (append a "Measured" section)

**Interfaces:**
- Consumes: Tasks 1-2 merged on the branch.

- [ ] **Step 1: Run the acceptance probe (CPU, D=2 chi=8)**

The probe script is the spike's
`/tmp/claude-1007/-home-yjkao-tenax/9d5185cf-8f92-4cd5-aae3-5ba5b2d21977/scratchpad/gmres/bwd_modules.py`.
Its runner `run.sh` hard-codes `PYTHONPATH=$HOME/tenax-gmresprobe/src`. Use
a copy pointed at this worktree:

```bash
S=/tmp/claude-1007/-home-yjkao-tenax/9d5185cf-8f92-4cd5-aae3-5ba5b2d21977/scratchpad/gmres
sed 's#tenax-gmresprobe/src#tenax-gmres1/src#' $S/run.sh > $S/run_new.sh && chmod +x $S/run_new.sh
$S/run_new.sh new_gm_d2 2 8 gmres 48-63 & $S/run_new.sh new_fp_d2 2 8 fixed_point 32-47 & wait
for l in new_gm_d2 new_fp_d2; do echo "== $l"; grep -E "^\[$l|^exit" $S/$l.log; grep -E "Finished XLA compilation of jit\(" $S/$l.log | sed -E 's/.*compilation of jit\(([^)]*)\) in ([0-9.]+) sec/\2 \1/' | sort -rn | head -6; done
```

Expected, against the spike baseline (`gm_d2.log`, `fp_d2.log` in the same
directory):
- `gm`: `_jit_gmres_one_matvec_solve` at most about 800 s, against
  `_jit_eager_gmres_solve` 2,458 s + `_jit_apply_Jt` 435 s;
- `gm`: no `_jit_apply_Jt` compile;
- `fp`: no `_jit_dE_denv` compile;
- both: `gnorm` equals `0.18727973877...` to 1e-8 relative;
- warm call (`call2`) at most 1.0 s.

**NO-GO gate:** if the GMRES-solve compile is not at least 2x below
2,458 + 435 s, stop. Report the numbers to the controller and do not do the
remaining steps.

- [ ] **Step 2: Record the measurement in the spec**

Append to `docs/superpowers/specs/2026-10-16-gmres-single-matvec-design.md`:

```markdown
## Measured (CPU AMD EPYC 7742, D=2 chi=8 U(1), first gradient)

| module | before | after |
|---|---|---|
| GMRES solve | 2,458 s (`_jit_eager_gmres_solve`) | <X> s (`_jit_gmres_one_matvec_solve`) |
| standalone `_jit_apply_Jt` | 435 s | not compiled |
| `_jit_dE_denv` on `fixed_point` | 48 s | not compiled |
| first gradient, `gmres` | 3,729 s | <Y> s |
| first gradient, `fixed_point` | 1,349 s | <Z> s |
```

Replace `<X>`, `<Y>` and `<Z>` with the measured values from Step 1. The
`call1` line gives the first-gradient totals.

- [ ] **Step 3: CHANGELOG and guide**

Insert at the top of `### Behavior Changes` under `## Unreleased` in
`CHANGELOG.md`:

```markdown
- **The eager-GMRES adjoint compiles the J^T matvec once.** The implicit-AD
  backward's GMRES (`adjoint_method="gmres"`, and the fallback when the fused
  Neumann loop does not converge) used JAX's GMRES, whose program inlines the
  matvec at three call sites, and compiled a fourth standalone copy to choose
  the seed and measure the residual. A new single-matvec GMRES does the seed
  choice, the solve and the residual measurement in one program with one call
  site, and the fixed-point path no longer compiles a separate `dE/denv`.
  Gradients agree with the previous solver to the solve tolerance, not bit
  for bit. Measured on CPU, D=2 chi=8 U(1), first gradient with
  `adjoint_method="gmres"`: 3,729 s -> <Y> s.
```

Use the same `<Y>` value as in Step 2.

Insert before `#### Persistent compilation cache` in
`docs/guide/algorithms/ipeps_ad_paths.md`:

```markdown
The GMRES adjoint (`adjoint_method="gmres"`, and the fallback when the fused
Neumann loop does not converge) compiles the sweep VJP once, like the fused
module.  JAX's own GMRES applied it at three places in one program, and XLA
compiled each one; the solver tenax uses instead applies it at one.  At D=3
the fallback is the common case near a slowly converging forward fixed point,
so this is most of the backward compile there.

```

- [ ] **Step 4: Commit**

```bash
cd ~/tenax-gmres1 && PATH=$HOME/.local/bin:$PATH pre-commit run --files CHANGELOG.md docs/guide/algorithms/ipeps_ad_paths.md docs/superpowers/specs/2026-10-16-gmres-single-matvec-design.md
git add CHANGELOG.md docs/guide/algorithms/ipeps_ad_paths.md docs/superpowers/specs/2026-10-16-gmres-single-matvec-design.md
git commit -m "docs: single-matvec GMRES adjoint, measured compile drop

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg"
```
