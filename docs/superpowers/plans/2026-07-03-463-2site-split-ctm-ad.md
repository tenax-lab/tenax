# 2-site Split-CTM AD (Dense) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit + implicit autodiff through the coupled 2-site checkerboard split-CTM fixed point, lift the policy guard, and dispatch `optimize_gs_ad(fuse_virtual_legs=False, recipe='2x2', 2-site)` to it — closing #463 Phase 2 for the dense (bosonic) 2-site path.

**Architecture:** Phase 1 landed the dense joint 2-site *forward* (`ctm_split_tensor_2site` → coupled `(env_A, env_B)`, absorbs proven bit-identical to the fused 2×2 oracle). This plan wraps that forward in AD. The single-site split AD (`_split_ctm_energy_ad.py`) already differentiates a *single* `SplitCTMTensorEnv` via a plain Neumann series with a per-tensor Γ phase-fix gauge, validated implicit==explicit. Because that machinery is written entirely against `jax.tree` leaves, it generalizes to a `{coord: SplitCTMTensorEnv}` dict pytree by (a) swapping the fixed-point step from `_split_step` (single) to `_split_step_multisite` (uses `_split_ctm_sweep_multisite` + per-coord Γ phase-fix), and (b) making the `custom_vjp` differentiate w.r.t. a `{coord: Tensor}` site dict instead of a single `A`. No new linear-algebra primitives. Four independently-landable tasks, each gated on the prior's parity.

**Tech Stack:** Python 3.11/3.12, JAX (float64), Tenax `Tensor`/`DenseTensor`/`SplitCTMTensorEnv`, `jax.custom_vjp`, pytest (`-m core`).

---

## Critical validation lesson (read before writing any AD test)

From the single-site Phase-2 validation ([[feedback_ctm_parity_needs_convergent_input]] and `tests/test_split_ctm_fuse_flag.py::test_split_implicit_grad_matches_explicit` docstring):

1. **Never use random site tensors for AD/energy parity.** The fused 2-site CTM *oscillates* (never converges) on raw random input, so any split-vs-fused comparison on random tensors is meaningless numerology. Every parity test in this plan builds a **physical, convergent** Heisenberg Néel checkerboard state via 2-site simple update, using the existing `_build_su_neel(D=2)` helper in `tests/test_split_ctm_2site.py:859`.

2. **The trusted AD gate is `implicit == explicit`, NOT `implicit == finite-difference`.** The split `energy_fn` carries a *pre-existing Wirtinger gap* (a real/complex-derivative mismatch present on the single-site path too). AD-vs-FD inherits that gap; explicit AD shares it. So the primary Tier-3 assertions are:
   - split-implicit gradient **direction** matches split-explicit gradient: `cos > 1 - 1e-9`
   - split-implicit gradient **magnitude** matches split-explicit: `rel_err < 1e-6`
   - split energy **value** matches the fused-path energy: `< 1e-8` (lossless `chi_I=2*chi`) / `< 1e-6` (`chi_I=chi`)

   Finite difference is used only as a loose *directional* sanity check (`cos > 0.99`), never as a tight magnitude gate, and its docstring must state the Wirtinger-gap reason.

---

## File Structure

Four files, each with one clear responsibility. All changes are additive; no Phase-1 forward code is modified.

- **`src/tenax/algorithms/_split_ctm_energy_ad.py`** (MODIFY) — owns split-CTM AD energy. Currently single-site only. Add the multisite (2-site checkerboard) siblings: `_split_step_multisite`, `_converge_split_multisite_gauge_fixed`, `converge_split_env_2site` (forward-only), `_split_ctm_converge_multisite` (`custom_vjp` over the site dict), `ctm_energy_split_explicit_2site`, `ctm_energy_split_implicit_2site`. The single-site functions stay untouched; the 2-site ones mirror them over a `{coord: ...}` pytree.
- **`src/tenax/algorithms/ipeps_ad_policy.py`** (MODIFY) — owns AD dispatch policy. Relax `validate_split_ctm_config` to accept the 2-site checkerboard recipe under `fuse_virtual_legs=False`; add a 2-site branch to `_split_ctm_energy_fn` inside `make_ctm_energy_fn` that routes to the new 2-site entries and uses the split multisite energy (ignoring the fused default `energy_fn`).
- **`src/tenax/algorithms/ipeps_optimize.py`** (MODIFY) — owns the optimizer loops. Give `_optimize_gs_ad_tensor_2site` a `use_split` branch that swaps the fused `python_loop_ctm_converge` for the split `converge_split_env_2site` in the three forward-only spots (env-cache warm-start, line-search probe, final-env eval), mirroring the single-site `use_split` block at `ipeps_optimize.py:1305-1420`.
- **`tests/test_split_ctm_2site_ad.py`** (CREATE) — owns the Phase-2 2-site AD test suite: Tier-3 parity (implicit==explicit, split==fused, FD directional sanity), guard-lift regression, dispatch routing, and an end-to-end `optimize_gs_ad` smoke.

### Key reference signatures (already on `main`, do not modify)

```python
# _split_ctm_tensor_convergence.py
def _split_ctm_sweep_multisite(envs, site_tensors, bars, neighbors, chi, chi_I,
                               renormalize, recipe="2x2") -> dict[Coord, SplitCTMTensorEnv]
def _initialize_split_multisite_env(site_tensors, chi, chi_I) -> dict[Coord, SplitCTMTensorEnv]
def ctm_split_tensor_2site(A, B, chi, max_iter=100, conv_tol=1e-8, chi_I=None,
                           renormalize=True, recipe="2x2") -> tuple[SplitCTMTensorEnv, SplitCTMTensorEnv]

# _split_ctm_tensor_energy.py
def compute_energy_split_ctm_tensor_multisite(site_tensors, envs, neighbors, gate, d=None) -> jax.Array
def compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=None) -> jax.Array

# ad_utils.py
def _phase_fix_split_ctm_tensor(env) -> SplitCTMTensorEnv          # per-single-env Γ phase-fix

# _ctm_tensor_convergence.py
CHECKERBOARD_NEIGHBORS   # {(0,0): {...}, (1,0): {...}} bipartite neighbor map
```

The single-site AD template being mirrored lives in `_split_ctm_energy_ad.py`: `_split_step` (90-101), `_converge_split_gauge_fixed` (111-140), `_split_ctm_converge` + fwd/bwd (143-202), `ctm_energy_split_explicit` (34-76), `ctm_energy_split_implicit` (205-245), `converge_split_env` (248-276).

---

## Task 1: Multisite forward-only + explicit-AD split converge

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_energy_ad.py` (add imports + 3 functions after line 276)
- Test: `tests/test_split_ctm_2site_ad.py` (create)

Foundation task: the multisite fixed-point step, a forward-only converge (for warm-start/line-search), and an explicit-AD (unrolled) converge. No `custom_vjp` yet.

- [ ] **Step 1: Write the failing test**

Create `tests/test_split_ctm_2site_ad.py` with a shared convergent-input fixture and the first test. Reuse `_build_su_neel` / `_heisenberg_gate` by importing them from the existing Phase-1 test module.

```python
"""#463 Phase 2 — dense 2-site split-CTM AD (explicit + implicit).

Parity is validated on a PHYSICAL, convergent Heisenberg Néel checkerboard
(2-site simple update), never random tensors: the fused 2-site CTM oracle
oscillates on random input, making any split-vs-fused comparison meaningless.
The trusted AD gate is implicit==explicit (not implicit==finite-difference):
the split energy_fn carries a pre-existing Wirtinger gap that AD-vs-FD inherits.
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tests.test_split_ctm_2site import _build_su_neel, _heisenberg_gate


@pytest.fixture(scope="module")
def su_state():
    """Convergent (A, B) Heisenberg Néel checkerboard via 2-site simple update."""
    A, B = _build_su_neel(D=2)
    return A, B


def test_converge_split_env_2site_matches_forward(su_state):
    """Forward-only multisite converge lands on the same fixed-point energy as
    ctm_split_tensor_2site (both are the Γ-gauge-fixed coupled fixed point)."""
    from tenax.algorithms._split_ctm_energy_ad import converge_split_env_2site
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8

    envs_ref = ctm_split_tensor_2site(A, B, chi, max_iter=100, conv_tol=1e-12, chi_I=chi)
    E_ref = float(
        compute_energy_split_ctm_tensor_2site(A, B, envs_ref[0], envs_ref[1], gate, d=2)
    )

    envs = converge_split_env_2site(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS,
        chi=chi, chi_I=chi, max_iter=100, conv_tol=1e-12, min_iter=2,
    )
    E = float(
        compute_energy_split_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], gate, d=2)
    )
    assert abs(E - E_ref) < 1e-9, f"forward converge mismatch: {E} vs {E_ref}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_converge_split_env_2site_matches_forward -x -q`
Expected: FAIL with `ImportError: cannot import name 'converge_split_env_2site'`

- [ ] **Step 3: Write minimal implementation**

In `src/tenax/algorithms/_split_ctm_energy_ad.py`, extend `__all__` and add the multisite forward machinery. Insert after `converge_split_env` (line 276):

```python
# ---------------------------------------------------------------------------
# Multisite (2-site checkerboard) split-CTM AD (#463 Phase 2)
# ---------------------------------------------------------------------------
#
# The single-site machinery above is written entirely against jax.tree leaves,
# so it generalizes to a {coord: SplitCTMTensorEnv} dict pytree by swapping the
# fixed-point step for the multisite sweep and per-coord Γ phase-fix.  The
# custom_vjp now differentiates w.r.t. a {coord: Tensor} site dict.


def _split_step_multisite(site_tensors, envs, neighbors, chi, chi_I, renormalize):
    """One gauge-fixed multisite split-CTM sweep: ``Γ ∘ sweep`` per coord.

    Fixed-point map ``f`` for the coupled ``{coord: env}`` system.  The per-coord
    Γ phase-fix is what makes the joint env converge element-wise.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_sweep_multisite,
    )
    from tenax.algorithms.ad_utils import _phase_fix_split_ctm_tensor

    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _split_ctm_sweep_multisite(
        envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, recipe="2x2"
    )
    return {c: _phase_fix_split_ctm_tensor(e) for c, e in envs.items()}


def _converge_split_multisite_gauge_fixed(
    site_tensors, neighbors, chi, chi_I, max_iter, conv_tol, renormalize, min_iter,
    envs_init=None,
):
    """Run gauge-fixed multisite split-CTM to an element-wise fixed point.

    Convergence is measured element-wise on the Γ-phase-fixed ``{coord: env}``
    dict (all tensors, all coords), mirroring the single-site
    :func:`_converge_split_gauge_fixed`.
    """
    if envs_init is not None:
        envs = envs_init
    else:
        from tenax.algorithms._split_ctm_tensor_convergence import (
            _initialize_split_multisite_env,
        )

        envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    prev = None
    for it in range(max_iter):
        envs = _split_step_multisite(
            site_tensors, envs, neighbors, chi, chi_I, renormalize
        )
        if prev is not None and it + 1 >= min_iter:
            if _split_env_max_diff(envs, prev) < conv_tol:
                break
        prev = envs
    return envs


def converge_split_env_2site(
    site_tensors,
    neighbors,
    *,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
    envs_init=None,
):
    """Forward-only gauge-fixed 2-site split-CTM converge (no gradient).

    Returns the ``{coord: SplitCTMTensorEnv}`` dict at the same Γ-phase-fixed
    element-wise fixed point that :func:`ctm_energy_split_implicit_2site`
    differentiates.  Forward-only energy evaluations on the 2-site split path
    (optimizer warm-start, line-search probe, final-env eval) must use this so
    the line-search φ(α) and the gradient dφ/dα stay mutually consistent —
    the 2-site analogue of :func:`converge_split_env`.
    """
    if chi_I is None:
        chi_I = chi
    return _converge_split_multisite_gauge_fixed(
        site_tensors, neighbors, chi, chi_I, max_iter, conv_tol, renormalize,
        min_iter, envs_init=envs_init,
    )
```

Note `_split_env_max_diff` (line 104) already works on any pytree (it walks `jax.tree.leaves`), so a dict-of-envs is handled unchanged.

Update `__all__` (line 17):

```python
__all__ = [
    "converge_split_env",
    "converge_split_env_2site",
    "ctm_energy_split_explicit",
    "ctm_energy_split_explicit_2site",
    "ctm_energy_split_implicit",
    "ctm_energy_split_implicit_2site",
]
```

(The two `*_2site` energy entries are added in Task 2; listing them now is harmless — `__all__` is not import-checked — and avoids a second edit. If a linter objects, add them in Task 2 instead.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_converge_split_env_2site_matches_forward -x -q`
Expected: PASS

- [ ] **Step 5: Add the explicit-AD converge test**

Append to `tests/test_split_ctm_2site_ad.py`:

```python
def test_explicit_multisite_converge_grad_finite(su_state):
    """Unrolled explicit multisite converge yields a finite, non-zero gradient
    w.r.t. A on the convergent state."""
    from tenax.algorithms._split_ctm_energy_ad import (
        _explicit_split_multisite_converge,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4

    def loss(a):
        envs = _explicit_split_multisite_converge(
            {(0, 0): a, (1, 0): B}, CHECKERBOARD_NEIGHBORS,
            chi=chi, chi_I=chi, warmup_steps=10, backprop_steps=10,
        )
        return compute_energy_split_ctm_tensor_2site(
            a, B, envs[(0, 0)], envs[(1, 0)], gate, d=2
        ).real

    e, g = jax.value_and_grad(loss)(A)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.isfinite(e)
    assert jnp.all(jnp.isfinite(gs)) and float(jnp.sum(jnp.abs(gs))) > 0
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_explicit_multisite_converge_grad_finite -x -q`
Expected: FAIL with `ImportError: cannot import name '_explicit_split_multisite_converge'`

- [ ] **Step 7: Add the explicit-AD converge helper**

In `_split_ctm_energy_ad.py`, after `converge_split_env_2site`, add the unrolled (differentiable) converge — mirrors `ad_utils.ctm_split_tensor_converge_explicit` (single-site) over the dict:

```python
def _explicit_split_multisite_converge(
    site_tensors, neighbors, *, chi, chi_I=None, renormalize=True,
    warmup_steps=0, backprop_steps=20,
):
    """Explicit (unrolled) multisite split-CTM converge for warm-start AD.

    ``warmup_steps`` sweeps run under ``stop_gradient``; ``backprop_steps``
    sweeps are fully differentiable.  Uses the raw multisite sweep (no Γ
    phase-fix) — the energy is gauge-invariant, matching the single-site
    :func:`ad_utils.ctm_split_tensor_converge_explicit`.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _initialize_split_multisite_env,
        _split_ctm_sweep_multisite,
    )

    if chi_I is None:
        chi_I = chi
    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    for _ in range(warmup_steps):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, "2x2"
        )
    if warmup_steps > 0:
        envs = jax.tree.map(jax.lax.stop_gradient, envs)
    for _ in range(backprop_steps):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, "2x2"
        )
    return envs
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_explicit_multisite_converge_grad_finite -x -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_energy_ad.py tests/test_split_ctm_2site_ad.py
git commit -m "feat(#463): multisite forward-only + explicit split-CTM converge (Phase 2 Task 1)"
```

---

## Task 2: Implicit + explicit 2-site AD energy entries (Tier-3 parity)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_energy_ad.py`
- Test: `tests/test_split_ctm_2site_ad.py`

The core AD: a `custom_vjp` over the `{coord: Tensor}` site dict → coupled `{coord: env}` fixed point, plus the two public energy entries. This is the exact structural mirror of `_split_ctm_converge` (143-202) and `ctm_energy_split_{explicit,implicit}` (34-76, 205-245), lifted to the dict pytree.

- [ ] **Step 1: Write the failing Tier-3 parity test**

Append to `tests/test_split_ctm_2site_ad.py`:

```python
def test_2site_implicit_grad_matches_explicit(su_state):
    """PRIMARY Tier-3 gate: 2-site split implicit (Neumann) gradient matches the
    trusted explicit (unrolled) gradient on the convergent Néel state.

    implicit==explicit, NOT implicit==FD: the split energy_fn carries a
    pre-existing Wirtinger gap that AD-vs-FD inherits and explicit shares.
    Gradient taken w.r.t. sublattice A only (B held fixed) for a clean scalar
    parity, at the lossless chi_I=chi fixed point.
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4  # chi = D*D lossless on a physical low-interlayer-rank state

    def loss_imp(a):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=80, conv_tol=1e-13, min_iter=2,
        ).real

    def loss_exp(a):
        return ctm_energy_split_explicit_2site(
            {(0, 0): a, (1, 0): B}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, warmup_steps=40, backprop_steps=40,
        ).real

    e_i, g_i = jax.value_and_grad(loss_imp)(A)
    e_e, g_e = jax.value_and_grad(loss_exp)(A)
    gi = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i)])
    ge = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_e)])

    assert jnp.allclose(e_i, e_e, atol=1e-9), f"energy mismatch: {e_i} vs {e_e}"
    cos = float(jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge)))
    rel = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))
    assert cos > 1 - 1e-9, f"gradient direction mismatch: cos={cos}"
    assert rel < 1e-6, f"gradient magnitude mismatch: rel={rel}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_2site_implicit_grad_matches_explicit -x -q`
Expected: FAIL with `ImportError: cannot import name 'ctm_energy_split_explicit_2site'`

- [ ] **Step 3: Add the implicit `custom_vjp` and both energy entries**

In `_split_ctm_energy_ad.py`, after `_explicit_split_multisite_converge`, add the fixed-point `custom_vjp` over the site dict (mirrors `_split_ctm_converge`/fwd/bwd) and the public entries. `neighbors` and `static` are non-differentiable (`nondiff_argnums`).

```python
@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static):
    """Converge gauge-fixed multisite split-CTM; custom-VJP via implicit diff.

    ``static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)``.
    Differentiates w.r.t. the ``{coord: Tensor}`` site dict; ``envs_init`` is a
    gradient-free warm-start seed (fixed point is seed-independent).  Returns the
    converged ``{coord: SplitCTMTensorEnv}`` dict.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    return _converge_split_multisite_gauge_fixed(
        site_tensors, neighbors, chi, chi_I, max_iter, conv_tol, renormalize,
        min_iter, envs_init=envs_init,
    )


def _split_ctm_converge_multisite_fwd(site_tensors, envs_init, neighbors, static):
    envs = _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static)
    return envs, (site_tensors, envs, envs_init)


def _split_ctm_converge_multisite_bwd(neighbors, static, residuals, g):
    """Backward via Neumann series ``λ = Σ_n (J^T)^n g`` at the coupled fixed point.

    ``g`` is the cotangent on the converged ``{coord: env}`` dict.  Accumulate λ
    in env space with ``J^T = (∂f/∂envs)^T`` (site dict fixed), then project once
    to site space with ``(∂f/∂site_tensors)^T λ``.  Direct mirror of the
    single-site :func:`_split_ctm_converge_bwd`, now over the dict pytree.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    site_tensors, envs, envs_init = residuals

    _, vjp_env_fn = jax.vjp(
        lambda e: _split_step_multisite(
            site_tensors, e, neighbors, chi, chi_I, renormalize
        ),
        envs,
    )
    _, vjp_site_fn = jax.vjp(
        lambda s: _split_step_multisite(
            s, envs, neighbors, chi, chi_I, renormalize
        ),
        site_tensors,
    )

    max_fp_iter = min(max_iter, 50)
    grads = g
    lam = g
    for _ in range(max_fp_iter):
        grads = vjp_env_fn(grads)[0]
        grads_inf = max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(grads))
        if grads_inf < conv_tol:
            break
        lam = jax.tree.map(lambda li, gi: li + gi, lam, grads)
        lam_norm = sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(lam)) ** 0.5
        if not math.isfinite(lam_norm) or lam_norm > 1e15:
            lam = jax.tree.map(lambda li, gi: li - gi, lam, grads)
            break

    d_site = vjp_site_fn(lam)[0]
    d_envs_init = (
        None if envs_init is None else jax.tree.map(jnp.zeros_like, envs_init)
    )
    return (d_site, d_envs_init)


_split_ctm_converge_multisite.defvjp(
    _split_ctm_converge_multisite_fwd, _split_ctm_converge_multisite_bwd
)


def ctm_energy_split_explicit_2site(
    site_tensors, neighbors, gate, *, chi=20, warmup_steps=3, backprop_steps=20,
    chi_I=None, renormalize=True, energy_fn=None, **_ignored,
):
    """2-site checkerboard iPEPS energy with explicit (unrolled) split-CTM AD."""
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn is not supported on the split path; "
            "use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_multisite,
    )

    envs = _explicit_split_multisite_converge(
        site_tensors, neighbors, chi=chi, chi_I=chi_I, renormalize=renormalize,
        warmup_steps=warmup_steps, backprop_steps=backprop_steps,
    )
    return compute_energy_split_ctm_tensor_multisite(
        site_tensors, envs, neighbors, gate
    )


def ctm_energy_split_implicit_2site(
    site_tensors, neighbors, gate, *, chi=20, max_iter=100, conv_tol=1e-8,
    chi_I=None, renormalize=True, min_iter=2, energy_fn=None, envs_init=None,
    **_ignored,
):
    """2-site checkerboard iPEPS energy with implicit (fixed-point) split-CTM AD.

    The coupled ``(env_A, env_B)`` forward is run to a gauge-fixed element-wise
    fixed point; the gradient comes from implicit differentiation (Neumann
    series) over the joint ``{coord: env}`` pytree.  *envs_init* is an optional
    gradient-free ``{coord: SplitCTMTensorEnv}`` warm-start seed.
    """
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn is not supported on the split path; "
            "use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_multisite,
    )

    static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)
    envs = _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static)
    return compute_energy_split_ctm_tensor_multisite(
        site_tensors, envs, neighbors, gate
    )
```

- [ ] **Step 4: Run the parity test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_2site_implicit_grad_matches_explicit -x -q`
Expected: PASS

If `rel` or `cos` fails, the most likely cause is an under-converged explicit reference — bump `warmup_steps`/`backprop_steps` to 60, or tighten implicit `conv_tol` to `1e-14`, before suspecting the VJP. A genuine VJP bug shows as a *direction* failure (`cos` far below 1), not a small `rel`.

- [ ] **Step 5: Add the split-vs-fused energy parity test**

Append:

```python
def test_2site_split_energy_matches_fused_ad_path(su_state):
    """The AD-energy value (split implicit) matches the fused 2-site energy on
    the convergent state — energy correctness independent of the gradient."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8

    envA, envB = ctm_tensor_2site(A, B, chi, max_iter=100, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))

    E_split = float(
        ctm_energy_split_implicit_2site(
            {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=100, conv_tol=1e-12, min_iter=2,
        ).real
    )
    assert abs(E_split - E_fused) < 1e-6, f"split={E_split} fused={E_fused}"
```

- [ ] **Step 6: Add the FD directional-sanity test (loose, documents Wirtinger gap)**

Append:

```python
def test_2site_implicit_grad_fd_directional(su_state):
    """FD is a LOOSE directional sanity check only, never a tight magnitude gate.

    The split energy_fn carries a pre-existing Wirtinger (real/complex-derivative)
    gap, so AD-vs-FD magnitude does NOT match to 1e-6 — only the direction agrees.
    Trusted magnitude parity is implicit==explicit (see
    test_2site_implicit_grad_matches_explicit)."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4

    def loss(a):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=80, conv_tol=1e-13, min_iter=2,
        ).real

    _, g = jax.value_and_grad(loss)(A)
    g_ad = jax.tree.leaves(g)[0].ravel()

    A_data = A.todense()
    eps = 1e-5
    flat = A_data.ravel()
    idxs = list(range(0, flat.size, max(1, flat.size // 12)))[:12]  # sample 12 dirs
    g_fd = []
    from tenax.algorithms.ipeps import _wrap_as_dense_tensor
    for i in idxs:
        pert = flat.at[i].add(eps).reshape(A_data.shape)
        pert_m = flat.at[i].add(-eps).reshape(A_data.shape)
        ep = loss(_wrap_as_dense_tensor(pert))
        em = loss(_wrap_as_dense_tensor(pert_m))
        g_fd.append(float((ep - em) / (2 * eps)))
    g_fd = jnp.array(g_fd)
    g_ad_s = jnp.array([float(g_ad[i]) for i in idxs])
    cos = float(
        jnp.dot(g_ad_s, g_fd) / (jnp.linalg.norm(g_ad_s) * jnp.linalg.norm(g_fd) + 1e-30)
    )
    assert cos > 0.99, f"AD and FD gradients point in different directions: cos={cos}"
```

- [ ] **Step 7: Run all Task-2 tests**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py -x -q`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_energy_ad.py tests/test_split_ctm_2site_ad.py
git commit -m "feat(#463): implicit + explicit 2-site split-CTM AD energy; Tier-3 parity (Phase 2 Task 2)"
```

---

## Task 3: Policy guard-lift + `make_ctm_energy_fn` 2-site dispatch

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py` (`validate_split_ctm_config` 44-80, `_split_ctm_energy_fn` 260-300)
- Test: `tests/test_split_ctm_2site_ad.py`

Relax the recipe guard to allow the 2-site checkerboard, and route `make_ctm_energy_fn` to the new 2-site entries when the site dict has 2 coords.

- [ ] **Step 1: Write the failing dispatch + guard-lift test**

Append to `tests/test_split_ctm_2site_ad.py`:

```python
def test_validate_split_ctm_config_allows_2site():
    """The 2-site checkerboard recipe ('2x2') is allowed under fuse=False; the
    three chi-changing knobs are still rejected."""
    from tenax.algorithms.ipeps_config import CTMConfig
    from tenax.algorithms.ipeps_ad_policy import validate_split_ctm_config

    cfg = CTMConfig(chi=8, chi_I=8, fuse_virtual_legs=False)
    validate_split_ctm_config(cfg, "1x1")  # single-site still OK
    validate_split_ctm_config(cfg, "2x2")  # 2-site now OK — must not raise

    import pytest
    bump = CTMConfig(chi=8, chi_I=8, fuse_virtual_legs=False, chi_auto_bump=True)
    with pytest.raises(NotImplementedError):
        validate_split_ctm_config(bump, "2x2")


def test_make_ctm_energy_fn_dispatches_2site_split(su_state):
    """make_ctm_energy_fn routes a 2-coord site dict to the 2-site split path
    (fuse=False, recipe='2x2'), matching a direct implicit-2site call, with a
    finite gradient through the dispatch closure."""
    import jax
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
    from tenax.algorithms.ipeps_config import CTMConfig

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8
    cfg = CTMConfig(
        chi=chi, chi_I=chi, fuse_virtual_legs=False,
        max_iter=100, conv_tol=1e-12, min_iter=2,
    )
    fn = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS, gate=gate,
        get_ctm_cfg=lambda: cfg, env_cache={}, use_explicit=False,
        explicit_warmup=3, explicit_steps=20, explicit_backward_steps=None,
        energy_fn=None, recipe="2x2",
    )
    E_dispatch = float(fn({(0, 0): A, (1, 0): B}).real)
    E_direct = float(
        ctm_energy_split_implicit_2site(
            {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=100, conv_tol=1e-12, min_iter=2,
        ).real
    )
    assert abs(E_dispatch - E_direct) < 1e-10

    def loss(a):
        return fn({(0, 0): a, (1, 0): B}).real
    _, g = jax.value_and_grad(loss)(A)
    gs = jax.tree.leaves(g)[0]
    assert bool((abs(gs).sum() > 0)) and bool((gs == gs).all())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_validate_split_ctm_config_allows_2site tests/test_split_ctm_2site_ad.py::test_make_ctm_energy_fn_dispatches_2site_split -x -q`
Expected: FAIL — the guard raises `NotImplementedError` for recipe `"2x2"` (first test), and the dispatch calls `ctm_energy_split_implicit` which hits `_extract_single_site` → `NotImplementedError` (second test).

- [ ] **Step 3: Relax the recipe guard**

In `ipeps_ad_policy.py`, replace the `recipe != "1x1"` reject (lines 59-63) so both `"1x1"` and `"2x2"` are allowed:

```python
    if recipe not in ("1x1", "2x2"):
        raise NotImplementedError(
            "fuse_virtual_legs=False (split CTM) supports gs_recipe in "
            f"('1x1', '2x2'); got recipe={recipe!r}."
        )
```

Update the docstring line 47-48 to reflect that the split forward now supports the 2-site checkerboard (`"2x2"`) recipe in addition to single-site — replace "is fixed-χ single-site, so" with "is fixed-χ (single-site or 2-site checkerboard), so".

- [ ] **Step 4: Add the 2-site branch to `_split_ctm_energy_fn`**

In `ipeps_ad_policy.py`, extend the deferred import (lines 255-258) and branch in `_split_ctm_energy_fn` (260-300) on `len(site_tensors)`:

```python
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit,
        ctm_energy_split_implicit_2site,
    )

    def _split_ctm_energy_fn(site_tensors, ctm_cfg):
        """Dispatch to the split (``fuse_virtual_legs=False``) path.

        Single-site (``recipe='1x1'``) and 2-site checkerboard (``recipe='2x2'``)
        split forwards exist (#463 Phase 2); guard unsupported combinations up
        front.  The split path computes its own split-aware energy internally,
        so the fused default ``energy_fn`` (if any) is ignored here; only a
        genuinely custom callback would be rejected downstream.
        """
        validate_split_ctm_config(ctm_cfg, recipe)
        n_sites = len(site_tensors)
        if n_sites == 2:
            if use_explicit:
                return ctm_energy_split_explicit_2site(
                    site_tensors, neighbors, gate,
                    chi=ctm_cfg.chi, warmup_steps=explicit_warmup,
                    backprop_steps=explicit_steps, chi_I=ctm_cfg.chi_I,
                    renormalize=ctm_cfg.renormalize, energy_fn=None,
                )
            _cached = env_cache.get("envs", None)
            envs_init = _cached if _cached else None
            return ctm_energy_split_implicit_2site(
                site_tensors, neighbors, gate,
                chi=ctm_cfg.chi, max_iter=ctm_cfg.max_iter,
                conv_tol=ctm_cfg.conv_tol, chi_I=ctm_cfg.chi_I,
                renormalize=ctm_cfg.renormalize, min_iter=ctm_cfg.min_iter,
                energy_fn=None, envs_init=envs_init,
            )
        # single-site (existing path)
        if use_explicit:
            return ctm_energy_split_explicit(
                site_tensors, neighbors, gate,
                chi=ctm_cfg.chi, warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                backward_steps=explicit_backward_steps,
                chi_I=ctm_cfg.chi_I, renormalize=ctm_cfg.renormalize,
                energy_fn=energy_fn,
            )
        _cached = env_cache.get("envs", None)
        split_env_init = _cached.get((0, 0)) if _cached else None
        return ctm_energy_split_implicit(
            site_tensors, neighbors, gate,
            chi=ctm_cfg.chi, max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol, chi_I=ctm_cfg.chi_I,
            renormalize=ctm_cfg.renormalize, min_iter=ctm_cfg.min_iter,
            energy_fn=energy_fn, env_init=split_env_init,
        )
```

Note: on the 2-site path `env_cache["envs"]` is the whole `{coord: SplitCTMTensorEnv}` dict (Task 4 makes the optimizer store it that way), so it is passed directly as `envs_init`; on the single-site path it stays `.get((0, 0))`. The fused default `energy_fn` is deliberately dropped on the split path (the split entries compute `compute_energy_split_ctm_tensor_multisite` internally); a genuinely custom `energy_fn` is not reachable here because the 2-site optimizer only sets the default 2-site callback (Task 4 passes it through unchanged for the fused path).

- [ ] **Step 5: Run the Task-3 tests to verify they pass**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_validate_split_ctm_config_allows_2site tests/test_split_ctm_2site_ad.py::test_make_ctm_energy_fn_dispatches_2site_split -x -q`
Expected: PASS

- [ ] **Step 6: Guard against single-site regression**

Run the existing single-site split fuse-flag suite to confirm the dispatch refactor didn't break the 1-site path:

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -q`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps_ad_policy.py tests/test_split_ctm_2site_ad.py
git commit -m "feat(#463): lift 2-site split policy guard + make_ctm_energy_fn dispatch (Phase 2 Task 3)"
```

---

## Task 4: `_optimize_gs_ad_tensor_2site` split integration + end-to-end

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (`_optimize_gs_ad_tensor_2site`, 2737-3060+)
- Test: `tests/test_split_ctm_2site_ad.py`

Give the 2-site optimizer a `use_split` branch so the three forward-only CTMs (env-cache warm-start `_update_env_cache_2s`, line-search probe `loss_fn_fwd`, and the final-env eval) run through `converge_split_env_2site` instead of the fused `python_loop_ctm_converge`, mirroring the single-site block at `ipeps_optimize.py:1305-1420`. The AD loss already routes correctly via `make_ctm_energy_fn` (Task 3).

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_split_ctm_2site_ad.py`:

Config surface (verified against current code): `optimize_gs_ad(gate, A_init, config)`
takes 3 positional args — everything (`unit_cell`, `chi`, `recipe`) comes from
`config`. `fuse_virtual_legs` is a field on the nested `CTMConfig`
(`config.ctm.fuse_virtual_legs`, default `True`); `build_ad_ctm_config` passes it
through unchanged, so `ctm_cfg_2s.fuse_virtual_legs` reflects it with no plumbing.
2-site returns `((A_opt, B_opt), (env_A, env_B), E_gs)`. `optimize_gs_ad` imports
from `tenax` (top-level), NOT `tenax.algorithms.ipeps`. `su_init` defaults `True`
but only fires when `A_init is None`; we pass explicit `(A, B)`, so set it `False`
for clarity.

```python
def test_optimize_gs_ad_2site_split_runs(su_state):
    """optimize_gs_ad with config.ctm.fuse_virtual_legs=False + recipe='2x2' runs
    a bipartite Heisenberg optimization end-to-end (a few steps), producing a
    finite, physical (variational, above the spin-1/2 AFH floor) energy."""
    from tenax import optimize_gs_ad
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    A, B = su_state
    gate = _heisenberg_gate()
    ctm = CTMConfig(chi=8, chi_I=8, fuse_virtual_legs=False)
    cfg = iPEPSConfig(
        ctm=ctm, unit_cell="2site", gs_num_steps=3, gs_implicit_ad=True,
        gs_c4v=False, gs_recipe="2x2", gs_optimizer="lbfgs", su_init=False,
    )
    (A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, (A, B), cfg)
    E = float(E_gs)
    assert E == E  # finite (not NaN)
    assert E > -1.0, f"energy below spin-1/2 AFH floor (non-variational): {E}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_optimize_gs_ad_2site_split_runs -x -q`
Expected: FAIL — the fused `python_loop_ctm_converge` warm-start in `_update_env_cache_2s` runs (since `use_split_2s` does not exist yet), stores a fused `{coord: CTMTensorEnv}` in `_env_cache_2s["envs"]`, and the split AD loss (Task 3 dispatch) then feeds that fused dict as `envs_init` to `ctm_energy_split_implicit_2site`, which expects a `SplitCTMTensorEnv` dict → shape/type error.

- [ ] **Step 3: Add the `use_split` branch to the 2-site optimizer**

In `_optimize_gs_ad_tensor_2site` (`ipeps_optimize.py`), right after `ctm_cfg_2s = build_ad_ctm_config(config)` (line 2843), add — mirroring the single-site block at 1305-1336:

```python
    from tenax.algorithms._split_ctm_energy_ad import converge_split_env_2site
    from tenax.algorithms.ipeps_ad_policy import validate_split_ctm_config

    use_split_2s = not ctm_cfg_2s.fuse_virtual_legs
    if use_split_2s:
        validate_split_ctm_config(ctm_cfg_2s, config.gs_recipe)
        if (
            config.gs_ctm_conv_tol_schedule is not None
            or config.gs_ctm_max_iter_schedule is not None
            or config.gs_plateau_patience_schedule is not None
            or config.gs_chi_schedule_steps is not None
        ):
            raise NotImplementedError(
                "CTM/chi schedules are not supported on the split-CTM path; "
                "use fuse_virtual_legs=True."
            )

    def _split_forward_2s(site_tensors, envs_init=None):
        """Forward-only gauge-fixed 2-site split-CTM converge (warm-start/probe/
        final-env).  Lands on the same fixed point the implicit-AD loss
        differentiates so the line-search φ(α)/dφ(α) stay consistent."""
        return converge_split_env_2site(
            site_tensors, CHECKERBOARD_NEIGHBORS,
            chi=ctm_cfg_2s.chi, max_iter=ctm_cfg_2s.max_iter,
            conv_tol=ctm_cfg_2s.conv_tol, chi_I=ctm_cfg_2s.chi_I,
            renormalize=ctm_cfg_2s.renormalize, min_iter=ctm_cfg_2s.min_iter,
            envs_init=envs_init,
        )
```

Then in `_update_env_cache_2s` (line 2929-2953), branch the CTM converge on `use_split_2s`. Replace the `python_loop_ctm_converge` call (2938-2949) with:

```python
        if use_split_2s:
            envs = _split_forward_2s(site_tensors, _env_cache_2s.get("envs", None))
            _env_cache_2s["envs"] = envs
            # Split path has no in-CTM chi-bump; the reactive-bump metrics are
            # unused (schedules/bump are rejected above).
            _env_cache_2s["max_truncation_error"] = 0.0
            _env_cache_2s["max_smallest_S"] = 0.0
        else:
            envs, info = python_loop_ctm_converge(
                site_tensors,
                CHECKERBOARD_NEIGHBORS,
                **ctm_converge_kwargs(
                    ctm_cfg_2s, env_init=_env_cache_2s.get("envs", None)
                ),
            )
            _env_cache_2s["envs"] = envs
            _env_cache_2s["max_truncation_error"] = float(info.max_truncation_error)
            _env_cache_2s["max_smallest_S"] = float(info.max_smallest_S)
```

Apply the identical `use_split_2s` branch to `loss_fn_fwd` (the line-search probe, starting line 3044) and to the final-env evaluation. Find the final-env eval and any other `python_loop_ctm_converge` call sites in this function:

Run: `grep -n "python_loop_ctm_converge" src/tenax/algorithms/ipeps_optimize.py`

For each call inside `_optimize_gs_ad_tensor_2site`, wrap it with the same `if use_split_2s: envs = _split_forward_2s(...) else: <existing>` pattern. The probe and final-env cases only need `envs` (not `info`), so on the split branch just assign `envs = _split_forward_2s(site_tensors, _env_cache_2s.get("envs", None))`.

**Energy evaluation on the split forward-only envs:** the line-search probe and final-env code compute energy via the fused `compute_energy_ctm_tensor_2site` / `_energy_fn_2site`. On the split branch, that must become `compute_energy_split_ctm_tensor_2site`. Locate every `compute_energy_ctm_tensor_2site` / `_energy_fn_2site` call reachable from a forward-only (non-AD) path and branch it:

Run: `grep -n "compute_energy_ctm_tensor_2site\|_energy_fn_2site" src/tenax/algorithms/ipeps_optimize.py`

Add near the top of the function:

```python
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    def _forward_energy_2s(A_norm, B_norm, envs):
        if use_split_2s:
            return compute_energy_split_ctm_tensor_2site(
                A_norm, B_norm, envs[(0, 0)], envs[(1, 0)], gate, d_phys
            )
        return compute_energy_ctm_tensor_2site(
            A_norm, B_norm, envs[(0, 0)], envs[(1, 0)], gate, d_phys
        )
```

and use `_forward_energy_2s(...)` in the forward-only (line-search/final) energy spots. The AD `loss_fn` (2918) is unchanged — it already routes through `make_ctm_energy_fn`, which Task 3 dispatches to the split energy.

- [ ] **Step 4: Run the end-to-end test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py::test_optimize_gs_ad_2site_split_runs -x -q`
Expected: PASS

If the energy is below the AFH floor (non-variational), that is the coupled-fixed-point convergence risk flagged in design §10 — it means the split 2-site loss is not landing on the true fixed point during optimization, NOT a code bug. Diagnose by raising `chi`/`max_iter` and confirming `test_2site_split_energy_matches_fused_ad_path` still holds at the optimizer's chi; do not paper over it by loosening the assertion.

- [ ] **Step 5: Guard against 2-site fused-path regression**

Run the existing 2-site optimizer tests to confirm the `use_split_2s` branching left the fused (default) path untouched:

Run: `uv run pytest tests/ -q -k "2site and (optimize or ad)" -m core`
Expected: PASS (all pre-existing 2-site optimizer tests)

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_split_ctm_2site_ad.py
git commit -m "feat(#463): 2-site optimizer split-CTM forward integration + end-to-end (Phase 2 Task 4)"
```

---

## Final verification

- [ ] **Run the full new suite + affected neighbors:**

Run: `uv run pytest tests/test_split_ctm_2site_ad.py tests/test_split_ctm_fuse_flag.py tests/test_split_ctm_2site.py -q`
Expected: PASS (all)

- [ ] **Run the core marker (CI-required):**

Run: `uv run pytest -m core -q`
Expected: PASS (all)

- [ ] **Docs:** update the `#463` status memory and, if a public API name changed, `src/tenax/__init__.py` `__all__` + `README.md`. The new functions are all private/internal (`_split_ctm_*`) except the `*_2site` energy entries, which are internal AD helpers — no README/`__init__` change expected. Confirm with:

Run: `grep -rn "ctm_energy_split" README.md src/tenax/__init__.py`
Expected: no matches (these stay internal) — if there ARE matches for the single-site ones, add the 2-site siblings alongside them.

- [ ] **Open the PR** per CLAUDE.md workflow:

```bash
git push -u origin design/463-2site-split-ctm-ad
gh pr create --title "feat(#463): dense 2-site split-CTM AD (Phase 2)" --body "$(cat <<'EOF'
Closes #463 Phase 2 (dense 2-site AD). Adds explicit + implicit autodiff through
the coupled 2-site checkerboard split-CTM fixed point, lifts the policy guard,
and dispatches optimize_gs_ad(fuse_virtual_legs=False, recipe='2x2', 2-site).

Tier-3 parity validated on a convergent SU Néel state (implicit==explicit,
split==fused); FD is directional-only (documented Wirtinger gap).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (spec coverage)

- Design §7 explicit AD → Task 2 `ctm_energy_split_explicit_2site` + Task 1 `_explicit_split_multisite_converge`. ✅
- Design §7 implicit AD (coupled `(env_A, env_B)` pytree, same Neumann machinery) → Task 2 `_split_ctm_converge_multisite` custom_vjp. ✅
- Design §7 guards gain a 2-site branch → Task 3 (`_split_ctm_energy_fn` `len==2` branch). Deliberate deviation from literal "`_extract_single_site` gains a branch": parallel 2-site functions instead of overloading single-site ones, matching the existing `_optimize_gs_ad_tensor` vs `_optimize_gs_ad_tensor_2site` split. ✅
- Design §7 policy relax (allow 2-site checkerboard under fuse=False; keep chi-bump/ramp/heuristic/custom-energy/cg rejects) → Task 3 `validate_split_ctm_config`. ✅
- Design §7 `optimize_gs_ad` dispatch → Task 4. ✅
- Design §8 Tier-3 on direction-dependent convergent inputs, never uniform/random → every test uses `_build_su_neel`. ✅ (A≠B by construction: Néel bias puts A on phys-0, B on phys-1.)
- Acceptance "optimize_gs_ad(...2site) runs end-to-end" → Task 4 Step 1. ✅
- Out of scope (SymmetricTensor Tiers, fermionic Tier-4, chi-bump/schedule on split) → deferred to Phase 3/4, not in this plan. ✅

**Config surface (resolved during planning):** `fuse_virtual_legs` is `config.ctm.fuse_virtual_legs`; `build_ad_ctm_config` passes it through unchanged, so `ctm_cfg_2s.fuse_virtual_legs` already reflects the user's choice — no plumbing addition needed. `optimize_gs_ad` is a 3-positional-arg entry imported from `tenax` (top-level); 2-site returns `((A,B),(env_A,env_B),E_gs)`. Task 4 Step 1's test is written against this verified surface.

**Genuine execution-time risk (design §10, not a code gap):** the coupled `(env_A, env_B)` fixed point may converge differently than two independent single-site loops, and the implicit-AD Neumann adjoint on the larger joint pytree is unproven at χ≥16. If Task 4 Step 4 shows a sub-floor (non-variational) energy, that is this risk surfacing — diagnose via chi/max_iter, do not loosen the assertion. The GMRES-conditioning concern is mitigated by reusing the *validated* single-site Neumann machinery (same `conv_tol` early-exit + `1e15` blow-up guard), but larger-pytree conditioning should be watched at χ=16 in a follow-up bench, not gated here.
