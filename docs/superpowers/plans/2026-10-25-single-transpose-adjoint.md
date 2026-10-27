# One Transpose: Joint Adjoint Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile the transpose J^T v once per backward by running the Neumann loop as the first phase of the GMRES phase machine, in one program.

**Architecture:** `_gmres_lax.py` gets a private `_gmres_machine` (the body of today's `gmres_single_matvec`, which becomes a thin wrapper with an unchanged program) and a new entry point `neumann_then_gmres`, which adds a `_NEUMANN` phase in front of `_SEED/_RESID/_ARNOLDI` and shares their one matvec call site. `_ctm_energy_ad.py` replaces `_jit_neumann_bwd` + `_jit_gmres_joint_solve` with one `_jit_joint_adjoint_solve` and rewires the two joint branches of `_bwd_core`. Then mutants, then the CPU pre-gate G0 and the GPU A/B.

**Tech Stack:** JAX 0.10.2 (`lax.while_loop`, `lax.switch`, `lax.cond`), pytest, the tenax block-sparse jit (`block_sparse_jit`, `_fusion_variant`).

**Spec:** `docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md` (amends §4 of `docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md`; read that spec's §9 for the measurements that motivate this one).

## Global Constraints

- Worktree `~/tenax-linonce`, branch `perf/linearize-once-adjoint`. Never push, never open a PR, never touch `main`.
- Interpreter: `~/tenax-linonce/.venv/bin/python` only (never bare `python`, never `uv run` from another clone).
- λ must equal today's Neumann -> GMRES result to round-off; the gradient must equal `adjoint_chain_rule="separate"` to rtol=1e-10, atol=1e-12.
- Not changed: `adjoint_chain_rule="separate"`, `_jit_gmres_one_matvec_solve`, `_jit_fused_fixed_point_bwd`, `_neumann_loop`, the split CTM, the forward CTM, `_jit_linearize_joint`, `_jit_dE_denv`, `_jit_direct`.
- `gmres_single_matvec` keeps its signature, behaviour and lowered program.
- Tests: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest ... -p no:cacheprovider > LOG 2>&1`, then read LOG. One pytest at a time. Never pipe a run into `tail`/`head`.
- Count compiles with `/jax/core/compile/jaxpr_trace_duration` events, never `jit._cache_size()`.
- GPU: `CUDA_VISIBLE_DEVICES=1` only. GPU 0 runs another job (pid 869113): never touch it. Host RAM is tight: nothing heavy beside a GPU arm.
- Mutation runs: commit first; each anchor unique (`command grep -c -F` prints 1); `PYTHONDONTWRITEBYTECODE=1`; delete `__pycache__` under `src/tenax/algorithms` before each run; restore with `git checkout -- <file>`.
- Use `command grep` (plain `grep` is a ugrep wrapper here).
- Commit messages end with exactly:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
  ```
- Pre-commit hooks (ruff, ruff-format) run on commit. On a hook failure, re-stage and commit again with the full message; never `--reuse-message=HEAD`.
- Scratch: `S=/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/single_t` (create it with `mkdir -p $S`).

## Review Focus

1. **Complex adjoint data** (complex128 env / λ): the Neumann phase must use the same `jnp.linalg.norm` / `abs**2` arithmetic as `_neumann_loop`, with no silent cast to real. Pinned by Task 1 case `complex`.
2. **A non-finite operator** (NaN from J^T): the Neumann phase must hand over (NaN `diff` fails `diff > tol`), not loop to `maxiter`, and the result must fail closed (`abs_resid` NaN, `converged` False). Pinned by Task 1 case `nonfinite`.
3. **`maxiter=0`**: zero Neumann steps, `inf` residual, straight to `_SEED` with seeds `[0, b]`. Pinned by Task 1 case `zero_iter`.
4. **A zero right-hand side**: Neumann converges at once on λ=0; the solve must return zeros, not NaN from a `0/0`. Pinned by Task 1 case `zero_rhs`.
5. **Leaves of different shapes** (2-D and 1-D leaves in one pytree): the per-leaf ℓ1-of-ℓ2 `diff` must equal `_neumann_loop`'s, which sums `jnp.linalg.norm` over the leaves. Pinned by the Task 1 fixture (leaf shapes `(2, 2)` and `(2,)`).

---

### Task 1: `neumann_then_gmres` — the Neumann phase in the GMRES phase machine

**Files:**
- Modify: `src/tenax/algorithms/_gmres_lax.py:341-618` (phase constants, `gmres_single_matvec`)
- Create: `tests/test_neumann_then_gmres.py`
- Modify: `tests/conftest.py:282-283` (file-to-bucket map)

**Interfaces:**
- Consumes: `tenax.algorithms._ctm_energy_ad._neumann_loop(apply_Jt, dE_denv, init_lam, use_init, *, tol, maxiter) -> (diverged, converged, n_iter, lam_final, abs_resid, b_norm)` (oracle only, unchanged); `gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart, aux=False)` (unchanged).
- Produces:
  - `neumann_then_gmres(jt_aux, b_tree, init_lam, use_init, *, tol, maxiter, restart, neumann=True) -> (diverged, converged, n_iter, lam_neumann, abs_resid_neumann, lam, abs_resid, b_norm, aux_of_lam)`. `jt_aux(v)` returns `(J^T v, a(v))`. With `neumann=False` the first five are `(False, False, 0, init_lam, inf)` and the seeds are `[init_lam, b]`.
  - Test helpers in `tests/test_neumann_then_gmres.py` that Task 2 imports: `_oracle(jt_aux, b, init, use_init, *, tol, maxiter, restart)` (same return tuple) and `_assert_same(got, want, rtol=1e-12, atol=1e-14)`.

- [ ] **Step 1: Record the lowered program of `gmres_single_matvec` before any edit**

```bash
cd ~/tenax-linonce && S=/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/single_t && mkdir -p $S
cat > $S/gsm_lowered.py <<'EOF'
import sys

import jax
import jax.numpy as jnp

import tenax  # noqa: F401  (enables x64)
from tenax.algorithms._gmres_lax import gmres_single_matvec

A = jnp.arange(36.0).reshape(6, 6) / 100 + jnp.eye(6)
P = jnp.arange(18.0).reshape(3, 6) / 10
b = (jnp.ones(6),)
c = ((jnp.zeros(6),), (jnp.full(6, 0.5),))
plain = lambda bb, cc: gmres_single_matvec(
    lambda v: (A @ v[0],), bb, list(cc), tol=1e-10, maxiter=5, restart=4
)
with_aux = lambda bb, cc: gmres_single_matvec(
    lambda v: ((A @ v[0],), P @ v[0]), bb, list(cc), tol=1e-10, maxiter=5,
    restart=4, aux=True,
)
fn = plain if sys.argv[1] == "plain" else with_aux
print(jax.jit(fn).lower(b, c).as_text())
EOF
for k in plain aux; do .venv/bin/python $S/gsm_lowered.py $k > $S/gsm_before_$k.txt; done
wc -l $S/gsm_before_plain.txt $S/gsm_before_aux.txt
```

Expected: two non-empty files (hundreds of lines each).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_neumann_then_gmres.py`:

```python
"""``neumann_then_gmres``: the Neumann loop and GMRES at one matvec site.

The joint adjoint solve ran the Neumann loop and the joint GMRES solve as two
compiled programs, each holding a copy of the CTM-sweep transpose.  Here the
Neumann loop is the first phase of the GMRES phase machine, so one program
applies ``J^T`` at one call site.  The oracle is the composition the backward
ran before: ``_neumann_loop``, then ``gmres_single_matvec`` from the seeds
``[lam if solved else 0, b]``.  Spec:
docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._gmres_lax import gmres_single_matvec, neumann_then_gmres

jax.config.update("jax_enable_x64", True)

TOL = 1e-12
RESTART = 6


def _tree(vec):
    """A flat (6,) vector as two leaves of different shapes (Review Focus 5)."""
    return (vec[:4].reshape(2, 2), vec[4:])


def _flat(t):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(t)])


def _contractive(rho, n=6, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    if complex_:
        M = M + 1j * rng.standard_normal((n, n))
    M *= rho / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(n)
    if complex_:
        b = b + 1j * rng.standard_normal(n)
    return M, b


def _jt_aux(M, P):
    M = jnp.asarray(M)
    P = jnp.asarray(P)

    def jt_aux(v):
        f = _flat(v)
        return _tree(M.T @ f), P @ f

    return jt_aux


def _oracle(jt_aux, b, init, use_init, *, tol, maxiter, restart):
    """The two solvers as the backward composed them before this change."""
    diverged, converged, n_iter, lam_n, resid_n, _bn = cea._neumann_loop(
        lambda v: jt_aux(v)[0], b, init, use_init, tol=tol, maxiter=maxiter
    )
    solved = converged & ~diverged
    seed = jax.tree.map(lambda x: jnp.where(solved, x, jnp.zeros_like(x)), lam_n)

    def matvec(v):
        jt, a = jt_aux(v)
        return jax.tree.map(lambda vi, ji: vi - ji, v, jt), a

    lam, abs_resid, b_norm, a = gmres_single_matvec(
        matvec, b, [seed, b], tol=tol, maxiter=maxiter, restart=restart, aux=True
    )
    return diverged, converged, n_iter, lam_n, resid_n, lam, abs_resid, b_norm, a


def _assert_same(got, want, rtol=1e-12, atol=1e-14):
    gl, wl = jax.tree.leaves(got), jax.tree.leaves(want)
    assert len(gl) == len(wl) and gl
    for g, w in zip(gl, wl):
        g, w = np.asarray(g), np.asarray(w)
        assert g.shape == w.shape and g.dtype == w.dtype, (g.dtype, w.dtype)
        if np.issubdtype(g.dtype, np.inexact):
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol, equal_nan=True)
        else:
            np.testing.assert_array_equal(g, w)


_CASES = {
    "converges": dict(rho=0.5),
    "warm": dict(rho=0.5, warm=True),
    "diverges": dict(rho=1.5),
    "slow": dict(rho=0.97, maxiter=4),
    "zero_iter": dict(rho=0.5, maxiter=0),
    "complex": dict(rho=0.5, complex_=True),
    "zero_rhs": dict(rho=0.5, zero_b=True),
    "nonfinite": dict(rho=0.5, nan=True),
}


def _run(case):
    c = _CASES[case]
    M, b = _contractive(c["rho"], complex_=c.get("complex_", False))
    if c.get("nan"):
        M[1, 2] = np.nan
    if c.get("zero_b"):
        b = np.zeros_like(b)
    P = np.arange(18.0).reshape(3, 6) / 10
    jt_aux = _jt_aux(M, P)
    if c.get("warm"):
        exact = np.linalg.solve(np.eye(6) - M.T, b)
        init = _tree(jnp.asarray(exact + 1e-6))
        use_init = jnp.array(True)
    else:
        init = _tree(jnp.zeros(6, dtype=jnp.asarray(b).dtype))
        use_init = jnp.array(False)
    b_tree = _tree(jnp.asarray(b))
    kw = dict(tol=TOL, maxiter=c.get("maxiter", 200), restart=RESTART)
    got = jax.jit(lambda bb, i, u: neumann_then_gmres(jt_aux, bb, i, u, **kw))(
        b_tree, init, use_init
    )
    want = jax.jit(lambda bb, i, u: _oracle(jt_aux, bb, i, u, **kw))(
        b_tree, init, use_init
    )
    return got, want


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_one_program_matches_the_two_solvers_composed(case):
    got, want = _run(case)
    _assert_same(got, want)
    diverged, converged, n_iter, _ln, resid_n, lam, abs_resid, _bn, _a = got
    # The regime each case is for (fixtures go vacuous otherwise).
    if case in ("converges", "warm", "complex"):
        assert bool(converged) and not bool(diverged) and int(n_iter) > 1
    if case == "warm":
        cold, _ = _run("converges")
        assert int(n_iter) < int(cold[2])
    if case == "diverges":
        assert bool(diverged)
        assert float(abs_resid) <= TOL * 10 * max(1.0, float(got[7]))
    if case == "slow":
        assert not bool(converged) and not bool(diverged) and int(n_iter) == 4
    if case == "zero_iter":
        assert int(n_iter) == 0 and not bool(converged)
        assert np.isinf(float(resid_n))
    if case == "complex":
        assert jnp.iscomplexobj(lam[0])
    if case == "zero_rhs":
        assert bool(converged)
        assert all(np.all(np.asarray(x) == 0) for x in lam)
    if case == "nonfinite":
        assert not bool(converged) and np.isnan(float(abs_resid))


def test_the_neumann_phase_off_is_gmres_from_the_warm_start():
    M, b = _contractive(0.5)
    P = np.arange(18.0).reshape(3, 6) / 10
    jt_aux = _jt_aux(M, P)
    b_tree = _tree(jnp.asarray(b))
    warm = _tree(jnp.asarray(np.linalg.solve(np.eye(6) - M.T, b) + 1e-3))
    kw = dict(tol=TOL, maxiter=50, restart=RESTART)
    got = jax.jit(
        lambda bb, w: neumann_then_gmres(
            jt_aux, bb, w, jnp.array(True), neumann=False, **kw
        )
    )(b_tree, warm)

    def matvec(v):
        jt, a = jt_aux(v)
        return jax.tree.map(lambda vi, ji: vi - ji, v, jt), a

    want = jax.jit(
        lambda bb, w: gmres_single_matvec(matvec, bb, [w, bb], aux=True, **kw)
    )(b_tree, warm)
    _assert_same(got[5:], want, rtol=0, atol=0)
    assert not bool(got[0]) and not bool(got[1]) and int(got[2]) == 0
    assert np.isinf(float(got[4]))


_M_SITE, _B_SITE = _contractive(0.5, seed=9)


@jax.jit
def _named_jt(v):
    f = _flat(v)
    return _tree(jnp.asarray(_M_SITE).T @ f), f[:3]


def _call_sites(fn):
    b = _tree(jnp.asarray(_B_SITE))
    text = jax.jit(fn).lower(b, b, jnp.array(False)).as_text()
    return len(re.findall(r"call @_named_jt\w*\(", text))


def test_the_counter_sees_two_sites_in_the_two_solvers_composed():
    """Premise: the counter can see more than one site."""
    kw = dict(tol=1e-10, maxiter=5, restart=4)
    assert _call_sites(lambda b, i, u: _oracle(_named_jt, b, i, u, **kw)) == 2


@pytest.mark.parametrize("neumann", [True, False])
def test_the_operator_is_applied_at_exactly_one_site(neumann):
    kw = dict(tol=1e-10, maxiter=5, restart=4, neumann=neumann)
    assert (
        _call_sites(lambda b, i, u: neumann_then_gmres(_named_jt, b, i, u, **kw))
        == 1
    )
```

Add to the file-to-bucket map in `tests/conftest.py`, next to `"test_gmres_single_matvec.py": "core",`:

```python
    "test_neumann_then_gmres.py": "core",
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd ~/tenax-linonce && JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_neumann_then_gmres.py -q -p no:cacheprovider > $S/t1_red.log 2>&1; command grep -E "Error|passed|failed" $S/t1_red.log | tail -5`
Expected: collection error `ImportError: cannot import name 'neumann_then_gmres'`.

- [ ] **Step 4: Implement**

In `src/tenax/algorithms/_gmres_lax.py`:

(a) After `_SEED, _RESID, _ARNOLDI = 0, 1, 2` add:

```python
# The first phase of :func:`neumann_then_gmres`, numbered last so the three
# GMRES phases keep their numbers and ``gmres_single_matvec`` its program.
_NEUMANN = 3
```

(b) In `gmres_single_matvec`, keep the signature and the whole docstring. Replace everything after the docstring with one call, and move the old body into a new private function directly below it, with the extra parameter `neumann`:

```python
    return _gmres_machine(
        matvec,
        b_tree,
        candidates,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        aux=aux,
        neumann=None,
    )


def _gmres_machine(matvec, b_tree, candidates, *, tol, maxiter, restart, aux, neumann):
    """The loop behind :func:`gmres_single_matvec` and :func:`neumann_then_gmres`.

    ``neumann`` is ``None`` for plain GMRES: ``matvec`` is the operator ``A``.
    Otherwise it is ``{"init": init_lam, "use_init": use_init, "maxiter": int}``,
    ``matvec`` is ``v -> (J^T v, a(v))`` (``aux`` must be True), the GMRES
    phases see ``A v = v - J^T v``, and ``candidates`` serve only the dtype
    check: the seeds come from the Neumann phase's handoff.
    """
    # ... the old body of gmres_single_matvec follows, edited as (c)-(h) ...
```

(c) Right after the three lines that build `cands`, `n_cand` and `C`, add:

```python
    if neumann is not None:
        # Seeds [lam_neumann if solved else 0, b], written at the handoff.
        n_cand = 2
        C = jnp.zeros((2, n), dtype)
```

(d) Right after `if aux:\n        state0["best_aux"] = aux0`, add:

```python
    if neumann is not None:
        n_maxiter = int(neumann["maxiter"])
        n_maxiter_arr = jnp.array(n_maxiter, dtype=jnp.int32)
        # _neumann_loop: cold start from b, warm start from init_lam.
        lam0 = jnp.where(
            neumann["use_init"], flatten(neumann["init"]).astype(dtype), b
        )
        state0.update(
            {
                "C": C,
                "phase": jnp.int32(_NEUMANN),
                "v_in": lam0,
                "n_lam": lam0,
                "n_diff": jnp.array(jnp.inf, dtype=rdtype),
                "n_k": jnp.int32(0),
                "n_div": jnp.array(False),
                # Fails closed: with zero steps this stays inf.
                "n_resid": jnp.array(jnp.inf, dtype=rdtype),
            }
        )
```

(e) In `seed(s, y, a)`, read the candidates from the state in the Neumann mode. Replace the two lines

```python
        best_x = jnp.where(better, C[s["k"]], s["best_x"])
```
and
```python
        nxt = C[jnp.minimum(k1, max(n_cand - 1, 0))]
```
with

```python
        Cs = C if neumann is None else s["C"]
        best_x = jnp.where(better, Cs[s["k"]], s["best_x"])
```
and
```python
        nxt = Cs[jnp.minimum(k1, max(n_cand - 1, 0))]
```

(f) After the `arnoldi` function, add the Neumann phase and its handoff:

```python
    def handoff(s):
        """End of the Neumann phase: GMRES from [lam if solved else 0, b]."""
        solved = (s["n_diff"] <= tol_r) & ~s["n_div"]
        seed0 = jnp.where(solved, s["n_lam"], zeros_n)
        return {
            **s,
            "C": jnp.stack([seed0, b]),
            "k": jnp.int32(0),
            "v_in": seed0,
            "phase": jnp.int32(_SEED),
        }

    def neumann_step(s, y, a, jt):
        """``_neumann_loop``'s body: lam <- b + J^T lam, on ``jt = J^T v_in``."""
        del y, a
        new = b + jt
        delta = new - s["n_lam"]
        d_leaves = jax.tree.leaves(unflatten(delta))
        diff = sum(jnp.linalg.norm(d) for d in d_leaves).astype(rdtype)
        resid = jnp.sqrt(sum(jnp.sum(jnp.abs(d) ** 2) for d in d_leaves)).astype(
            rdtype
        )
        diverged = (diff > s["n_diff"]) & (s["n_k"] > jnp.array(5, jnp.int32))
        s = {
            **s,
            "n_lam": new,
            "n_diff": diff,
            "n_k": s["n_k"] + jnp.array(1, jnp.int32),
            "n_div": diverged,
            "n_resid": resid,
            "v_in": new,
        }
        # _neumann_loop's cond_fn; a NaN diff fails ``diff > tol`` and hands over.
        go_on = (s["n_k"] < n_maxiter_arr) & ~diverged & (diff > tol_r)
        return lax.cond(go_on, lambda t: t, handoff, s)
```

(g) Replace `body` with:

```python
    def body(s):
        y, a = apply(s["v_in"])  # the one and only matvec call site
        if neumann is None:
            return lax.switch(s["phase"], (seed, resid, arnoldi), s, y, a)
        jt = y
        y = s["v_in"] - jt  # A v = v - J^T v for the GMRES phases
        return lax.switch(
            s["phase"],
            (
                lambda t, y_, a_, _j: seed(t, y_, a_),
                lambda t, y_, a_, _j: resid(t, y_, a_),
                lambda t, y_, a_, _j: arnoldi(t, y_, a_),
                neumann_step,
            ),
            s,
            y,
            a,
            jt,
        )

    if neumann is not None and n_maxiter == 0:
        state0 = handoff(state0)  # zero Neumann steps: straight to the seeds
```

(h) Replace the return block after `out = lax.while_loop(...)` with:

```python
    abs_resid = jnp.where(out["nonfinite"], jnp.nan, out["best_res"])
    if neumann is not None:
        return (
            out["n_div"],
            out["n_diff"] <= tol_r,
            out["n_k"],
            unflatten(out["n_lam"]),
            out["n_resid"],
            unflatten(out["best_x"]),
            abs_resid,
            b_norm,
            out["best_aux"],
        )
    if aux:
        return unflatten(out["best_x"]), abs_resid, b_norm, out["best_aux"]
    return unflatten(out["best_x"]), abs_resid, b_norm
```

(i) After `_gmres_machine`, add the public entry point:

```python
def neumann_then_gmres(
    jt_aux, b_tree, init_lam, use_init, *, tol, maxiter, restart, neumann=True
):
    """The joint adjoint solve: the Neumann loop, then GMRES, at ONE matvec site.

    Solves ``(I - J^T) lam = b``.  ``jt_aux(v)`` returns ``(J^T v, a(v))`` with
    ``a`` linear (``J_params^T`` in the joint-VJP chain rule).

    With ``neumann`` (the default fixed-point method) a ``_NEUMANN`` phase runs
    first, with the arithmetic of ``_ctm_energy_ad._neumann_loop``:
    ``lam_0 = init_lam if use_init else b``, ``lam_{k+1} = b + J^T lam_k``, the
    divergence guard, at most ``maxiter`` steps.  It hands over to the GMRES
    phases with the seeds ``[lam if solved else 0, b]``: the list the host
    built between two compiled programs before, each with its own copy of
    ``J^T`` (spec 2026-10-25-single-transpose-adjoint-design.md).  Without
    ``neumann`` (``adjoint_method="gmres"``) this is :func:`gmres_single_matvec`
    on ``A v = v - J^T v`` from the seeds ``[init_lam, b]``; ``use_init`` is
    ignored.

    Returns ``(diverged, converged, n_iter, lam_neumann, abs_resid_neumann,
    lam, abs_resid, b_norm, a(lam))``.  Without ``neumann`` the first five are
    ``(False, False, 0, init_lam, inf)``.
    """
    if neumann:
        return _gmres_machine(
            jt_aux,
            b_tree,
            [init_lam],
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            aux=True,
            neumann={"init": init_lam, "use_init": use_init, "maxiter": maxiter},
        )

    def matvec(v):
        jt, a = jt_aux(v)
        return jax.tree.map(lambda vi, ji: vi - ji, v, jt), a

    lam, abs_resid, b_norm, a = _gmres_machine(
        matvec,
        b_tree,
        [init_lam, b_tree],
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        aux=True,
        neumann=None,
    )
    return (
        jnp.array(False),
        jnp.array(False),
        jnp.int32(0),
        init_lam,
        jnp.array(jnp.inf, dtype=abs_resid.dtype),
        lam,
        abs_resid,
        b_norm,
        a,
    )
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_neumann_then_gmres.py -q -p no:cacheprovider > $S/t1_green.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t1_green.log | tail -12`
Expected: `12 passed` (8 cases, 1 Neumann-off, 1 premise, 2 one-site), 0 failed.

- [ ] **Step 6: Confirm `gmres_single_matvec`'s program is unchanged**

```bash
for k in plain aux; do .venv/bin/python $S/gsm_lowered.py $k > $S/gsm_after_$k.txt; cmp $S/gsm_before_$k.txt $S/gsm_after_$k.txt && echo "$k identical"; done
```
Expected: `plain identical` and `aux identical`. (This is the spec's test 6 as a one-time check: the lowered text depends on the JAX version, so it cannot be a CI test.)

- [ ] **Step 7: Run the neighbouring suites**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_gmres_single_matvec.py tests/test_gmres_single_matvec_wiring.py tests/test_adjoint_seed_858.py -q -p no:cacheprovider > $S/t1_neigh.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t1_neigh.log | tail -6`
Expected: all pass, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add src/tenax/algorithms/_gmres_lax.py tests/test_neumann_then_gmres.py tests/conftest.py
git commit -m "feat(adjoint): neumann_then_gmres runs the Neumann loop as a GMRES phase

One while_loop, one matvec call site: the Neumann loop becomes the first
phase of the GMRES phase machine and hands over with the seeds
[lam if solved else 0, b]. gmres_single_matvec is now a wrapper over the
shared _gmres_machine; its lowered program is unchanged.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 2: `_jit_joint_adjoint_solve` — one program in the backward

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (import at line 50; delete `_jit_gmres_joint_solve` at ~239-269 and `_jit_neumann_bwd` at ~354-370; add `_jit_joint_adjoint_solve`; `_neumann_loop` docstring ~277; `_jit_linearize_joint` docstring ~2140; the two joint branches of `_bwd_core` ~2455-2600 and ~2711-2720)
- Modify: `tests/test_linearize_once_adjoint.py`
- Modify: `tests/test_joint_chain_rule.py:116,120,192,201,290`
- Modify: `tests/test_vjp_cache_key_1049.py:213-215`
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md:312-317,378-390`

**Interfaces:**
- Consumes: `neumann_then_gmres(jt_aux, b_tree, init_lam, use_init, *, tol, maxiter, restart, neumann=True)` (Task 1); test helpers `_oracle`, `_assert_same` from `tests/test_neumann_then_gmres.py` (Task 1).
- Produces: `_jit_joint_adjoint_solve(vjp_joint, rhs, init_lam, use_init, *, tol, maxiter, restart, neumann)` — a module-level `block_sparse_jit` program, called through `_fusion_variant(..., by_bond_dim=True)`. Returns the 9-tuple of `neumann_then_gmres`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_linearize_once_adjoint.py`:

Replace the module docstring with:

```python
"""Linearise once, jit only the transpose, for the implicit-AD adjoint.

The default backward (``adjoint_chain_rule="joint"``) builds the gauge-fixed
sweep's pullback once (``_jit_linearize_joint``) and hands it to one joint
adjoint solve (``_jit_joint_adjoint_solve``): the Neumann loop and GMRES as
phases of one program, with one copy of the transpose.  The oracle is
``"separate"``, whose programs are unchanged.  Specs:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md,
docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md
"""
```

Replace `_SOLVERS = ("_jit_neumann_bwd", "_jit_gmres_joint_solve")` with:

```python
_SOLVERS = ("_jit_joint_adjoint_solve",)
```

In `test_one_backward_linearises_once`, replace the two lines after the first assertion with:

```python
    assert names.count("_jit_joint_adjoint_solve") == 1, names
```

Add after `test_one_backward_linearises_once`:

```python
from tests.test_neumann_then_gmres import _assert_same, _oracle  # noqa: E402

_DOT = re.compile(r"\bdot_general\[")


def _dot_count(fn, *args):
    return len(_DOT.findall(str(jax.make_jaxpr(fn)(*args))))


def _joint_solve_call(spy):
    (call,) = [c for c in spy if c[0] == "_jit_joint_adjoint_solve"]
    return call


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_joint_solve_holds_one_transpose(spy):
    """Neumann and GMRES share one matvec call site, so the program's
    contractions are one pullback's plus the Krylov bookkeeping."""
    _grad(_site(_FUSED_SEED), "joint")
    _name, _kind, variant, args, kwargs = _joint_solve_call(spy)
    vjp_joint, rhs = args[0], args[1]
    one = _dot_count(lambda f, v: f(tuple(v)), vjp_joint, rhs)
    assert one > 16  # regime: a second copy would stand out
    assert _dot_count(lambda *a: variant(*a, **kwargs), *args) <= one + 16


def _check_against_oracle(call):
    _name, _kind, variant, args, kwargs = call
    got = variant(*args, **kwargs)
    kw = {k: kwargs[k] for k in ("tol", "maxiter", "restart")}

    def oracle(f, b, i, u):
        def jt_aux(v):
            params_ct, env_ct = f(tuple(v))
            return env_ct, params_ct

        return _oracle(jt_aux, b, i, u, **kw)

    want = jax.jit(oracle)(*args)
    _assert_same(got, want)
    return got


@pytest.mark.usefixtures("fresh_vjp_cache")
@pytest.mark.parametrize("case", ["cold", "warm", "unsolved", "complex"])
def test_the_joint_solve_matches_the_two_solvers_composed(spy, case):
    dtype = jnp.complex128 if case == "complex" else jnp.float64
    A0 = _site(_FUSED_SEED, dtype=dtype)
    kw = dict(gmres_maxiter=3) if case == "unsolved" else {}
    _grad(A0, "joint", **kw)
    if case == "warm":
        spy.clear()
        _grad(A0 * (1.0 + 1e-3), "joint")
    call = _joint_solve_call(spy)
    assert call[4]["neumann"] is True
    if case == "warm":
        assert bool(call[3][3])  # regime: the warm start is used
    got = _check_against_oracle(call)
    if case == "unsolved":
        assert not bool(got[1])  # regime: the Neumann phase did not solve
    else:
        assert bool(got[1]) and not bool(got[0])


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_u1_joint_solve_matches_the_two_solvers_composed(spy):
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
    got = _check_against_oracle(_joint_solve_call(spy))
    assert int(got[2]) > 0  # regime: the Neumann phase ran


@pytest.fixture
def rule_spy(monkeypatch):
    """``{program name: by_bond_dim}`` for each ``_fusion_variant`` call."""
    rules = {}
    real = cea._fusion_variant

    def spying(prog, params_data_tuple, *, by_bond_dim):
        rules[prog.__name__] = by_bond_dim
        return real(prog, params_data_tuple, by_bond_dim=by_bond_dim)

    monkeypatch.setattr(cea, "_fusion_variant", spying)
    return rules


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_joint_solve_takes_the_joint_fusion_rule(rule_spy):
    """It holds the params VJP, so the joint programs' rule applies."""
    _grad(_site(_FUSED_SEED), "joint")
    assert rule_spy == {
        "_jit_linearize_joint": False,
        "_jit_joint_adjoint_solve": True,
    }, rule_spy
```

Add at the end of the file:

```python
@pytest.mark.slow
@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_fermionic_joint_solve_matches_the_two_solvers_composed(spy):
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
    jax.grad(
        lambda p: jnp.real(
            ctm_energy_implicit(
                {(0, 0): p[0], (1, 0): p[1]},
                CHECKERBOARD_NEIGHBORS,
                H,
                chi=8,
                max_iter=200,
                conv_tol=1e-10,
                gmres_tol=1e-11,
                adjoint_chain_rule="joint",
            )
        )
    )((A, B))
    got = _check_against_oracle(_joint_solve_call(spy))
    assert int(got[2]) > 0  # regime: the Neumann phase ran
```

In `tests/test_joint_chain_rule.py`, replace every `"_jit_gmres_joint_solve"` with `"_jit_joint_adjoint_solve"` (lines 116, 120, 192, 201, 290).

In `tests/test_vjp_cache_key_1049.py`, in `_BACKWARD_FNS`, replace the three lines

```python
    "_jit_gmres_joint_solve",  # the default chain rule ("auto" -> "joint")
    "_jit_linearize_joint",  # its pullback feeds the two solvers below
    "_jit_neumann_bwd",
```
with
```python
    "_jit_joint_adjoint_solve",  # the default chain rule ("auto" -> "joint")
    "_jit_linearize_joint",  # its pullback feeds the joint solve
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -m "not slow" -q -p no:cacheprovider > $S/t2_red.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t2_red.log | tail -15`
Expected: FAILs on every test that looks for `_jit_joint_adjoint_solve` (for example `ValueError: not enough values to unpack` in `_joint_solve_call`, or the names assertions).

- [ ] **Step 3: Implement**

In `src/tenax/algorithms/_ctm_energy_ad.py`:

(a) Line 50: `from tenax.algorithms._gmres_lax import gmres_pytree, gmres_single_matvec` becomes

```python
from tenax.algorithms._gmres_lax import (
    gmres_pytree,
    gmres_single_matvec,
    neumann_then_gmres,
)
```

(b) Delete the whole `_jit_gmres_joint_solve` definition (its `@partial(block_sparse_jit, static_argnames=("tol", "maxiter", "restart"))` decorator, signature, docstring and body) and put this in its place:

```python
@partial(block_sparse_jit, static_argnames=("tol", "maxiter", "restart", "neumann"))
def _jit_joint_adjoint_solve(
    vjp_joint, rhs, init_lam, use_init, *, tol, maxiter, restart, neumann
):
    """The joint adjoint solve on the saved linearisation: one program, one J^T.

    ``vjp_joint`` is the gauge-fixed sweep's pullback over ``(params, env)``
    from ``_jit_linearize_joint``: ``vjp_joint(v) == (J_params^T v,
    J_env^T v)``.  :func:`neumann_then_gmres` runs the Neumann loop (with
    ``neumann``, the fixed-point method) and then GMRES on
    ``(I - J_env^T) lam = rhs`` at one matvec call site, and carries
    ``J_params^T`` of its best iterate, so the chain rule's indirect term comes
    out of the solve and ``_jit_chain_rule`` never compiles (#1123).  Before,
    the Neumann loop and the joint GMRES solve were two programs, each with its
    own copy of the transpose (spec
    2026-10-25-single-transpose-adjoint-design.md).  Call it through
    :func:`_fusion_variant` with ``by_bond_dim=True``: it holds the params VJP
    and its arguments carry no ``SymmetricTensor``.  ``init_lam`` must keep
    ``rhs``'s pytree structure from call to call or it recompiles (#1087).
    Returns :func:`neumann_then_gmres`'s 9-tuple.
    """

    def jt_aux(v):
        params_ct, env_ct = vjp_joint(tuple(v))
        return env_ct, params_ct

    return neumann_then_gmres(
        jt_aux,
        rhs,
        init_lam,
        use_init,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        neumann=neumann,
    )
```

(c) Delete the whole `_jit_neumann_bwd` definition (decorator, signature, docstring, body).

(d) In the `_neumann_loop` docstring, replace

```
    Shared by ``_jit_fused_fixed_point_bwd`` (``"separate"``) and
    ``_jit_neumann_bwd`` (``"joint"``), so both chain rules run the same loop.
```
with
```
    Used by ``_jit_fused_fixed_point_bwd`` (``"separate"``).  The ``"joint"``
    chain rule runs the same arithmetic as the first phase of
    :func:`neumann_then_gmres`.
```

(e) In the `_jit_linearize_joint` docstring, replace ```_jit_neumann_bwd`` and`` / ```_jit_gmres_joint_solve`` take it as an argument and compile only the`` (two lines) with ```_jit_joint_adjoint_solve`` takes it as an argument and compiles only the`` and reflow the paragraph.

(f) In `_bwd_core`, fixed-point branch, replace the `if adjoint_chain_rule == "joint":` block that calls `_jit_neumann_bwd` (from the comment `# Linearise once (spec 2026-10-25)` to the closing `)` of the `pick_variant(_jit_neumann_bwd, False)(...)` call) with:

```python
            if adjoint_chain_rule == "joint":
                # Linearise once (spec 2026-10-25), then ONE joint solve: the
                # Neumann loop and GMRES are phases of one program, so the
                # transpose compiles once (spec 2026-10-25-single-transpose).
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
                    joint_lam,
                    joint_abs,
                    _b_norm,
                    joint_indirect,
                ) = pick_variant(_jit_joint_adjoint_solve, True)(
                    vjp_joint,
                    fused_rhs,
                    init_lam,
                    use_init,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                    neumann=True,
                )
```

(g) Further down, in the `if adjoint_chain_rule == "joint":` block after `fused_solved`, replace

```python
                seed = (
                    tuple(lam_final)
                    if fused_solved
                    else tuple(jnp.zeros_like(r) for r in fused_rhs)
                )
                lam, _abs, _bn, indirect = pick_variant(_jit_gmres_joint_solve, True)(
                    vjp_joint,
                    fused_rhs,
                    [seed, fused_rhs],
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```
with
```python
                # The joint solve above already ran GMRES from these seeds:
                # [lam_final if fused_solved else 0, rhs].
                lam, _abs, _bn, indirect = (
                    joint_lam,
                    joint_abs,
                    _b_norm,
                    joint_indirect,
                )
```
and in the comment block above it, replace `-- and the joint GMRES solve ALWAYS follows,` with `-- and GMRES ALWAYS follows, in the same program,`.

(h) In the `adjoint_method == "gmres"` branch, replace

```python
                lam, _abs, _bn, indirect = pick_variant(_jit_gmres_joint_solve, True)(
                    vjp_joint,
                    rhs,
                    [warm, rhs],
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
```
with
```python
                *_neumann, lam, _abs, _bn, indirect = pick_variant(
                    _jit_joint_adjoint_solve, True
                )(
                    vjp_joint,
                    rhs,
                    warm,
                    jnp.array(True),
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                    neumann=False,
                )
```

(i) Check that no reference is left: `command grep -n "_jit_neumann_bwd\|_jit_gmres_joint_solve" src tests docs/guide` must print nothing except the docs lines fixed in Step 4.

- [ ] **Step 4: Update the AD-paths guide**

In `docs/guide/algorithms/ipeps_ad_paths.md`, replace

```
gauge-fixed sweep once (`_jit_linearize_joint`) and passes the saved
pullback to the Neumann loop (`_jit_neumann_bwd`) and the joint GMRES solve
(`_jit_gmres_joint_solve`), which compile only the transpose.  Those two
take no `SymmetricTensor` argument, so the backward picks their variant
from the site tensors by the rules above.
```
with
```
gauge-fixed sweep once (`_jit_linearize_joint`) and passes the saved
pullback to one joint adjoint solve (`_jit_joint_adjoint_solve`): the
Neumann loop and GMRES as phases of one program, which compiles the
transpose once.  It takes no `SymmetricTensor` argument, so the backward
picks its variant from the site tensors by the rules above.
```

and replace

```
keeps the pullback.  The Neumann loop (`_jit_neumann_bwd`) runs on the env
half of that pullback and compiles only the transpose.  It is not the
env-only program of `"separate"`.  A joint GMRES solve
(`_jit_gmres_joint_solve`) takes the same pullback and returns `J_params^T`
of its best iterate.  It always follows the loop, seeded with the loop's `λ`
(usually accepted after one matvec) or, when the loop gave up, started cold
as the fallback is.
```
with
```
keeps the pullback.  One joint solve (`_jit_joint_adjoint_solve`) runs the
Neumann loop on the env half of that pullback and then GMRES, at one matvec
call site, so the transpose compiles once.  It is not the env-only program
of `"separate"`.  GMRES returns `J_params^T` of its best iterate.  It always
follows the loop in the same program, seeded with the loop's `λ` (usually
accepted after one matvec) or, when the loop gave up, started cold as the
fallback is.
```

- [ ] **Step 5: Run the changed test files**

Run, one at a time, reading each log:

```bash
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -m "not slow" -q -p no:cacheprovider > $S/t2_lin.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t2_lin.log | tail -8
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_joint_chain_rule.py tests/test_vjp_cache_key_1049.py tests/test_neumann_then_gmres.py -m "not slow" -q -p no:cacheprovider > $S/t2_nb.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t2_nb.log | tail -8
JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_linearize_once_adjoint.py -m slow -q -p no:cacheprovider > $S/t2_slow.log 2>&1; command grep -E "^(FAILED|ERROR)|passed|failed" $S/t2_slow.log | tail -6
```
Expected: all pass, 0 failed. (`test_vjp_cache_key_1049.py::test_frozensets_are_keyed_by_value_and_sets_are_not` fails on `main` under `PYTHONHASHSEED=0`; if it fails here, rerun it alone once and record it — it is not this change.)

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_linearize_once_adjoint.py tests/test_joint_chain_rule.py tests/test_vjp_cache_key_1049.py docs/guide/algorithms/ipeps_ad_paths.md
git commit -m "perf(adjoint): one joint adjoint solve, one copy of the transpose

_jit_joint_adjoint_solve replaces _jit_neumann_bwd and
_jit_gmres_joint_solve: the Neumann loop and GMRES run as phases of one
program (neumann_then_gmres), so J^T v compiles once per backward. lambda
equals the two-program composition to round-off (tested against it as an
oracle).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

The controller runs the core suite (`-m core`) after this task; the implementer does not.

---

### Task 3: Mutation-test the new tests

**Files:**
- Modify (temporarily, restored after each mutant): `src/tenax/algorithms/_gmres_lax.py`, `src/tenax/algorithms/_ctm_energy_ad.py`
- Record: the report file (one line per mutant: `Task 3: mutant Nn <killed|SURVIVED> by <test names>`)

**Interfaces:**
- Consumes: Tasks 1-2's committed code and tests.
- Produces: the mutant verdicts; any test tightened to kill a survivor (tests only, never `src/`).

Mutant command (from `~/tenax-linonce`, after applying each mutant; `$M` is the mutant name):

```bash
find src/tenax/algorithms -name __pycache__ -type d -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_neumann_then_gmres.py tests/test_linearize_once_adjoint.py -m "not slow" -q -p no:cacheprovider > $S/mut_$M.log 2>&1
command grep -E "^(FAILED|ERROR)|passed|failed" $S/mut_$M.log | tail -15
git checkout -- src/tenax/algorithms/_gmres_lax.py src/tenax/algorithms/_ctm_energy_ad.py
```

Before each mutant, check that its anchor is unique: `command grep -c -F '<anchor>' <file>` must print `1`. First run the command once with no mutant and `M=base`: all must pass.

- [ ] **Step 1: N1 — the Neumann phase applies the operator at a second site**

File `_gmres_lax.py`. Anchor: `        new = b + jt`. Replace with:

```python
        new = b + flatten(matvec(unflatten(s["v_in"]))[0]).astype(dtype)
```
Run with `M=N1`. Expected: killed by `test_the_operator_is_applied_at_exactly_one_site[True]` and `test_the_joint_solve_holds_one_transpose`.

- [ ] **Step 2: N2 — the Neumann update uses `v - J^T v`**

File `_gmres_lax.py`. Anchor: `        new = b + jt`. Replace with `        new = b + y`, and replace the line `        del y, a` (anchor, unique) with `        del a`. Run with `M=N2`. Expected: killed by `test_the_one_program_matches_the_two_solvers_composed[converges]` (and the other cases) and `test_the_joint_solve_matches_the_two_solvers_composed[cold]`.

- [ ] **Step 3: N3 — the handoff seeds λ even when Neumann did not solve**

File `_gmres_lax.py`. Anchor: `        seed0 = jnp.where(solved, s["n_lam"], zeros_n)`. Replace with `        seed0 = s["n_lam"]`. Run with `M=N3`. Expected: killed by `test_the_one_program_matches_the_two_solvers_composed[diverges]`, `[slow]`, and `test_the_joint_solve_matches_the_two_solvers_composed[unsolved]`.

- [ ] **Step 4: N4 — the divergence guard is dropped**

File `_gmres_lax.py`. Anchor: `        diverged = (diff > s["n_diff"]) & (s["n_k"] > jnp.array(5, jnp.int32))`. Replace with `        diverged = jnp.array(False)`. Run with `M=N4`. Expected: killed by `test_the_one_program_matches_the_two_solvers_composed[diverges]`.

- [ ] **Step 5: N5 — the joint solve takes the sweep-only fusion rule**

File `_ctm_energy_ad.py`. Anchor: `                ) = pick_variant(_jit_joint_adjoint_solve, True)(` (the fixed-point branch; unique because the gmres branch spells it over two lines). Replace `True` with `False`. Run with `M=N5`. Expected: killed by `test_the_joint_solve_takes_the_joint_fusion_rule`.

- [ ] **Step 6: Restore and confirm a clean tree**

Run: `git status --porcelain`
Expected: no output. For any SURVIVED mutant: add or tighten a test (RED with the mutant, GREEN without), commit it with the trailer, rerun the mutant, and record the result.

---

### Task 4: Gates G0 and A1-A4, and the results section

**Files:**
- Modify: `docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md` (append §9 Results)
- Use (unchanged): `benchmarks/bench_linearize_once.py`; the op counter `/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/linonce/ops/count_ops.py`

**Interfaces:**
- Consumes: the branch head after Task 3; the base worktree `~/tenax-linonce-base` (detached at `b71f5a5`, main) and its base measurements: G0 base = 830,453 optimised HLO instructions (parent spec §9); A1 base = 2403.5 s step-1 backward compile (`$OLD/base_d3.log`, `OLD=/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/linonce`); A3 base = `$OLD/base_d3.grad.npy`; A4 base = `device_peak_bytes` 1,561,049,088 in `$OLD/base_d3.json`.
- Produces: the gate verdicts and §9 of the new spec.

- [ ] **Step 1: Freeze the head in its own worktree**

```bash
cd ~/tenax-linonce && H=$(git rev-parse HEAD)
git -C ~/tenax-linonce-head checkout -q --detach $H && git -C ~/tenax-linonce-head log --oneline -1
```
Expected: the head worktree prints the branch head commit. (Its `.venv` installs tenax from that worktree in editable mode, so no reinstall is needed.)

- [ ] **Step 2: G0 — the CPU pre-gate (head only; base is known)**

```bash
G=$S/g0; rm -rf $G; mkdir -p $G/head/ir $G/head/xla
cd ~/tenax-linonce-head && JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false JAX_LOG_COMPILES=1 JAX_DUMP_IR_TO=$G/head/ir XLA_FLAGS="--xla_dump_to=$G/head/xla --xla_dump_hlo_as_text" timeout 7200 taskset -c 0-31 .venv/bin/python benchmarks/bench_linearize_once.py --D 2 --chi 8 --steps 1 --out $G/head.json > $G/head.log 2>&1; echo rc=$?
```
Then count. The counter's `PROGRAMS` tuple must name the new program: copy `count_ops.py` to `$G/count_ops.py`, add `"_jit_joint_adjoint_solve",` to its `PROGRAMS` tuple, and run `~/tenax-linonce-base/.venv/bin/python -I $G/count_ops.py $G head`.
Grade: G0 = (optimised#0 of `_jit_linearize_joint` + `_jit_dE_denv` + `_jit_joint_adjoint_solve`) / 830,453. Expected (estimate, not a requirement): about 0.61. **If G0 > 0.8: stop, write §9 with the numbers, and report — no GPU run.** If 0.7 < G0 <= 0.8: record it, run the GPU A/B anyway (A1 decides), and flag it.

- [ ] **Step 3: The GPU A/B — head arm only (base is already measured)**

Only this job may run on the host; start the memory guard first (it kills the benchmark if MemAvailable < 12 GB):

```bash
W=$S/gpu; mkdir -p $W
cd ~/tenax-linonce-head && CUDA_VISIBLE_DEVICES=1 JAX_LOG_COMPILES=1 JAX_ENABLE_COMPILATION_CACHE=false timeout 21600 taskset -c 32-47 /usr/bin/time -v .venv/bin/python benchmarks/bench_linearize_once.py --D 3 --chi 16 --steps 4 --out $W/head_d3.json > $W/head_d3.log 2>&1; echo rc=$?
```
Expected: rc=0, `MARK json_written` in the log, `$W/head_d3.grad.npy` present.

- [ ] **Step 4: Grade A1-A4**

- A1: sum the step-1 `Finished XLA compilation of jit(_jit_...)` times of the backward programs (`_jit_linearize_joint`, `_jit_dE_denv`, `_jit_joint_adjoint_solve`, `_jit_direct`) before the first `step 1/4` line; divide by 2403.5. Pass <= 0.7; > 0.9 is a no-go. Also record step 2 (both arms recompile there; base step 2 = 5127.2 s).
- A2: median of all `step_times` in `head_d3.json` / median of base `step_times` (`[6.446, 49.911, 33.145]` -> 33.145). Pass <= 1.05.
- A3: `np.linalg.norm(g_head - g_base) / np.linalg.norm(g_base)` with the two `.grad.npy` files. Pass <= 1e-8. Also the step-1 E and GMRES resid lines equal base's (E=-0.5529995394, resid=1.329e-06).
- A4: `device_peak_bytes` head / 1,561,049,088. Pass <= 1.2 (above: a finding to report, not to tune).

- [ ] **Step 5: Write §9 and commit**

Append `## 9. Results (<date>)` to the new spec: the G0 table (per-program lowered/optimised counts, base vs head), the A1 table (per-program step-1 compile seconds), A2-A4 values with pass/fail, step-2 numbers, and the run conditions (GPU 1, D=3 chi=16, `--steps 4`, cache off, base `b71f5a5`, head `<sha>`). Label `device_peak_bytes` as process-lifetime (SU and forward included).

```bash
git add docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md
git commit -m "docs(spec): single-transpose adjoint results (G0, A1-A4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```
