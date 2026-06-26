# #463 Phase 2 — `fuse_virtual_legs` flag + single-site split-CTM AD — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user set `CTMConfig.fuse_virtual_legs=False` to run the single-site (`recipe="1x1"`) `optimize_gs_ad` path on the χ²·D⁴ split-CTM double layer instead of the χ²·D⁶ fused one, with both explicit and implicit AD producing gradients that match the fused path to 1e-8. Default stays `True` (zero behavior change).

**Architecture:** Add one config bool. Add a new module `_split_ctm_energy_ad.py` with single-site `ctm_energy_split_explicit` (unrolled AD via `ctm_split_tensor_converge_explicit`) and `ctm_energy_split_implicit` (a self-contained `jax.custom_vjp` fixed-point backward that differentiates `_split_ctm_tensor_sweep` directly). Branch once on the flag inside the single dispatcher `make_ctm_energy_fn`, gated on a single-site cell. Multisite/2-site/c4v/honeycomb/PESS raise `NotImplementedError` when the flag is off.

**Tech Stack:** Python, JAX (`jax.custom_vjp`, `jax.vjp`, pytrees), Tenax Tensor protocol, pytest.

**Spec:** `docs/superpowers/specs/2026-06-26-463-phase2-split-ctm-fuse-flag-design.md`

---

## File Structure

- **Modify** `src/tenax/algorithms/ipeps_config.py` — add `fuse_virtual_legs: bool = True` to `CTMConfig` + docstring.
- **Create** `src/tenax/algorithms/_split_ctm_energy_ad.py` — `ctm_energy_split_explicit`, `ctm_energy_split_implicit` (single-site only).
- **Modify** `src/tenax/algorithms/ipeps_ad_policy.py` — branch `make_ctm_energy_fn` on the flag + single-site/knob guards.
- **Modify** `src/tenax/algorithms/ipeps_optimize.py` — early `NotImplementedError` in the 2-site and multisite dispatchers when flag is off (clearer message).
- **Modify** `src/tenax/algorithms/_ctm_tensor_c4v.py` — guard c4v entry.
- **Create** `tests/test_split_ctm_fuse_flag.py` — all new tests (single-site).
- **Modify** `src/tenax/__init__.py` — export the two new energy functions.

> **Note on c4v/honeycomb/PESS guards:** c4v goes through `_ctm_tensor_c4v.py`; honeycomb and PESS are reached via their own `optimize_gs_ad` recipe branches in `ipeps_optimize.py`. Task 5 adds guards at the dispatcher entry points (where `config.ctm.fuse_virtual_legs` is first visible), not deep in the CTM loop.

---

## Task 1: Add `fuse_virtual_legs` config flag

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py` (class `CTMConfig`, after the `chi_I` field ~line 82 and add a one-line docstring entry)
- Test: `tests/test_split_ctm_fuse_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split_ctm_fuse_flag.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_config import CTMConfig

pytestmark = pytest.mark.core


def test_fuse_virtual_legs_defaults_true():
    cfg = CTMConfig(chi=8)
    assert cfg.fuse_virtual_legs is True


def test_fuse_virtual_legs_can_disable():
    cfg = CTMConfig(chi=8, fuse_virtual_legs=False)
    assert cfg.fuse_virtual_legs is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k fuse_virtual_legs -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'fuse_virtual_legs'`

- [ ] **Step 3: Add the field**

In `src/tenax/algorithms/ipeps_config.py`, add the field next to `chi_I` (keep dataclass field ordering — it has a default, so any position after the existing defaulted fields is fine):

```python
    fuse_virtual_legs: bool = True  # True: fused double-layer (χ²·D⁶, default).
    # False: single-site (recipe="1x1") split ket/bra double layer (χ²·D⁴).
    # #463 Phase 2 — multisite/2-site/c4v/honeycomb/PESS raise NotImplementedError
    # when False (no multisite split forward yet).
```

Also add a one-line entry to the class docstring `Attributes:` block:

```
        fuse_virtual_legs:  When ``True`` (default) the CTM uses the fused
                            double-layer tensor.  When ``False`` the single-site
                            (``recipe="1x1"``) AD path uses the split ket/bra
                            double layer (χ²·D⁴ memory).  Only the single-site
                            path is supported; other lattices raise
                            ``NotImplementedError`` (#463 Phase 2).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k fuse_virtual_legs -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_split_ctm_fuse_flag.py
git commit -m "feat(#463): add CTMConfig.fuse_virtual_legs flag (default True)"
```

---

## Task 2: `ctm_energy_split_explicit` (single-site, unrolled AD)

**Files:**
- Create: `src/tenax/algorithms/_split_ctm_energy_ad.py`
- Test: `tests/test_split_ctm_fuse_flag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_site(D, d, seed):
    """Single-site U(1)-trivial DenseTensor with labels (u,d,l,r,phys)."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(d, dtype=np.int32)
    idx = [
        TensorIndex("u", D, sym, zD, FlowDirection.OUT),
        TensorIndex("d", D, sym, zD, FlowDirection.OUT),
        TensorIndex("l", D, sym, zD, FlowDirection.OUT),
        TensorIndex("r", D, sym, zD, FlowDirection.OUT),
        TensorIndex("phys", d, sym, zd, FlowDirection.OUT),
    ]
    return DenseTensor(data, idx)


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
def test_split_explicit_grad_matches_fused(D, chi):
    from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_explicit

    A = _make_site(D, 2, seed=7)
    gate = _heisenberg_gate()
    st = {(0, 0): A}

    def fused(a):
        return ctm_energy_explicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate,
            chi=chi, warmup_steps=3, backprop_steps=12,
        )

    def split(a):
        return ctm_energy_split_explicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate,
            chi=chi, warmup_steps=3, backprop_steps=12, chi_I=chi,
        )

    e_fused, g_fused = jax.value_and_grad(lambda a: fused(a).real)(A)
    e_split, g_split = jax.value_and_grad(lambda a: split(a).real)(A)

    np.testing.assert_allclose(np.asarray(e_split), np.asarray(e_fused), atol=1e-8)
    gf = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_fused)])
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_split)])
    np.testing.assert_allclose(np.asarray(gs), np.asarray(gf), atol=1e-8, rtol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k split_explicit_grad -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.algorithms._split_ctm_energy_ad'`

- [ ] **Step 3: Create the module with the explicit function**

```python
# src/tenax/algorithms/_split_ctm_energy_ad.py
"""Single-site split-CTM AD energy entry points (#463 Phase 2).

Mirrors ``ctm_energy_explicit`` / ``ctm_energy_implicit`` from
``_ctm_energy_ad`` but on the split (ket/bra-separate) double layer.
Single-site only: the split forward (``ctm_split_tensor``) converges one
site as an isolated 1×1 iPEPS.  Multisite has no split forward yet.
"""

from __future__ import annotations

import jax

from tenax.algorithms._split_ctm_tensor_convergence import (
    ctm_split_tensor,
    _split_ctm_tensor_sweep,
)
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor
from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit

__all__ = ["ctm_energy_split_explicit", "ctm_energy_split_implicit"]


def _extract_single_site(site_tensors):
    if len(site_tensors) != 1:
        raise NotImplementedError(
            "split-CTM (fuse_virtual_legs=False) supports only the single-site "
            f"(recipe='1x1') path; got {len(site_tensors)} sites."
        )
    ((_coord, A),) = site_tensors.items()
    return A


def ctm_energy_split_explicit(
    site_tensors,
    neighbors,
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    backward_steps: int | None = None,
    chi_I: int | None = None,
    renormalize: bool = True,
    energy_fn=None,
    **_ignored,
):
    """Single-site iPEPS energy with explicit (unrolled) split-CTM AD."""
    A = _extract_single_site(site_tensors)
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn (e.g. coarse-grain) is not supported on the split "
            "path yet; use fuse_virtual_legs=True."
        )
    if backward_steps is not None:
        raise ValueError(
            "backward_steps (TBPTT) is not supported on the split explicit path; "
            "set gs_explicit_ad_backward_steps=None or use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi
    env = ctm_split_tensor_converge_explicit(
        A,
        chi=chi,
        chi_I=chi_I,
        renormalize=renormalize,
        num_steps=backprop_steps,
        warmup_steps=warmup_steps,
    )
    return compute_energy_split_ctm_tensor(A, env, gate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k split_explicit_grad -v`
Expected: PASS (2 passed). If energy matches but gradient is off by > 1e-8, increase `backprop_steps` to 20 in both branches of the test (CTM must reach the same fixed point on both paths before gradients align).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_energy_ad.py tests/test_split_ctm_fuse_flag.py
git commit -m "feat(#463): ctm_energy_split_explicit (single-site unrolled AD)"
```

---

## Task 3: `ctm_energy_split_implicit` (single-site `custom_vjp` fixed point)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_energy_ad.py`
- Test: `tests/test_split_ctm_fuse_flag.py`

**Math (implicit function theorem at the CTM fixed point):**
`env* = sweep(A, env*)`, `E = energy(A, env*)`. With cotangent `g` on `E`:
`dE/denv*` and direct `∂E/∂A` come from `jax.vjp(energy)`. Solve
`λ = (I − J_envᵀ)⁻¹ (dE/denv*)` by Neumann series, where `J_envᵀ·v = jax.vjp(λe. sweep(A, e))(v)`.
Then `dL/dA = ∂E/∂A·g + J_Aᵀ·λ`, with `J_Aᵀ·λ = jax.vjp(λa. sweep(a, env*))(λ)`.
This handles Dense and U(1) (`SymmetricTensor`) tensors because `_split_ctm_tensor_sweep`
operates on the full `A` pytree — no single-leaf assumption.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
def test_split_implicit_grad_matches_fused(D, chi):
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit

    A = _make_site(D, 2, seed=11)
    gate = _heisenberg_gate()

    def fused(a):
        return ctm_energy_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate,
            chi=chi, max_iter=80, conv_tol=1e-10,
        ).real

    def split(a):
        return ctm_energy_split_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate,
            chi=chi, max_iter=80, conv_tol=1e-10, chi_I=chi,
        ).real

    e_fused, g_fused = jax.value_and_grad(fused)(A)
    e_split, g_split = jax.value_and_grad(split)(A)

    np.testing.assert_allclose(np.asarray(e_split), np.asarray(e_fused), atol=1e-8)
    gf = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_fused)])
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_split)])
    np.testing.assert_allclose(np.asarray(gs), np.asarray(gf), atol=1e-8, rtol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k split_implicit_grad -v`
Expected: FAIL — `ImportError: cannot import name 'ctm_energy_split_implicit'`

- [ ] **Step 3: Implement the implicit function**

Append to `src/tenax/algorithms/_split_ctm_energy_ad.py`:

```python
def ctm_energy_split_implicit(
    site_tensors,
    neighbors,
    gate,
    *,
    chi: int = 20,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    energy_fn=None,
    adjoint_method: str = "vjp",
    **_ignored,
):
    """Single-site iPEPS energy with implicit-diff split-CTM backward.

    Forward: ``ctm_split_tensor`` to convergence.  Backward: solve
    ``(I - J_envᵀ) λ = dE/denv`` by Neumann series and chain to ``dE/dA``.
    """
    A = _extract_single_site(site_tensors)
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn (e.g. coarse-grain) is not supported on the split "
            "path yet; use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi
    max_fp_iter = max_iter

    def _sweep(a, e):
        return _split_ctm_tensor_sweep(e, a, chi, chi_I, renormalize)

    def _energy(a, e):
        return compute_energy_split_ctm_tensor(a, e, gate).real

    @jax.custom_vjp
    def _fixed_point_energy(a):
        env = ctm_split_tensor(a, chi, max_iter, conv_tol, chi_I, renormalize)
        return _energy(a, env)

    def _fwd(a):
        env = ctm_split_tensor(a, chi, max_iter, conv_tol, chi_I, renormalize)
        return _energy(a, env), (a, env)

    def _bwd(res, g):
        a, env = res
        # direct ∂E/∂a and dE/denv
        _, vjp_energy = jax.vjp(_energy, a, env)
        dE_da_direct, dE_denv = vjp_energy(g)
        # J_envᵀ and J_aᵀ via one linearization of the sweep at (a, env)
        _, vjp_sweep = jax.vjp(_sweep, a, env)
        # Neumann: λ = Σ_n (J_envᵀ)^n dE_denv
        lam = dE_denv
        term = dE_denv
        for _ in range(max_fp_iter):
            term = vjp_sweep(term)[1]  # J_envᵀ · term  (env cotangent only)
            lam = jax.tree.map(lambda x, y: x + y, lam, term)
            term_inf = max(
                float(jax.numpy.max(jax.numpy.abs(t)))
                for t in jax.tree.leaves(term)
            )
            if term_inf < conv_tol:
                break
        dE_da_env = vjp_sweep(lam)[0]  # J_aᵀ · λ
        d_a = jax.tree.map(lambda x, y: x + y, dE_da_direct, dE_da_env)
        return (d_a,)

    _fixed_point_energy.defvjp(_fwd, _bwd)
    return _fixed_point_energy(A)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k split_implicit_grad -v`
Expected: PASS (2 passed).

Troubleshooting if it fails:
- *Energy matches, gradient off:* the Neumann sum has not converged — raise `max_iter` (e.g. 120) or lower `conv_tol` to `1e-12` in the test. If `J_envᵀ` has spectral radius ≥ 1 (no gauge fix), the sum diverges; add a phase gauge fix to `_split_ctm_tensor_sweep`'s output (per `feedback_phase_gauge_default`) — see the spec's gauge risk note. If gauge work is needed, ship Task 2 (explicit) as the parity gate and open a follow-up issue for the implicit gauge.
- *`vjp_sweep(term)` shape error:* confirm `_sweep(a, e)` returns the same env pytree structure it takes; `_split_ctm_tensor_sweep(env, A, ...)` returns a `SplitCTMTensorEnv`, so cotangents match `env`.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_energy_ad.py tests/test_split_ctm_fuse_flag.py
git commit -m "feat(#463): ctm_energy_split_implicit (single-site custom_vjp fixed point)"
```

---

## Task 4: Dispatch on the flag in `make_ctm_energy_fn` + guards

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py` (`make_ctm_energy_fn`, the `_ctm_energy_fn` closure ~line 213; the deferred import block ~line 208)
- Test: `tests/test_split_ctm_fuse_flag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
def _make_energy_fn(fuse, use_explicit, chi):
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
    from tenax.algorithms.ipeps_config import CTMConfig

    gate = _heisenberg_gate()
    cfg = CTMConfig(chi=chi, max_iter=80, conv_tol=1e-10, fuse_virtual_legs=fuse)
    return make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=use_explicit,
        explicit_warmup=3,
        explicit_steps=12,
    )


@pytest.mark.parametrize("use_explicit", [True, False])
def test_dispatch_split_matches_fused(use_explicit):
    A = _make_site(2, 2, seed=13)
    fused_fn = _make_energy_fn(True, use_explicit, chi=8)
    split_fn = _make_energy_fn(False, use_explicit, chi=8)

    e_fused, g_fused = jax.value_and_grad(lambda a: fused_fn({(0, 0): a}).real)(A)
    e_split, g_split = jax.value_and_grad(lambda a: split_fn({(0, 0): a}).real)(A)

    np.testing.assert_allclose(np.asarray(e_split), np.asarray(e_fused), atol=1e-8)
    gf = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_fused)])
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_split)])
    np.testing.assert_allclose(np.asarray(gs), np.asarray(gf), atol=1e-8, rtol=1e-8)


def test_dispatch_multisite_split_raises():
    split_fn = _make_energy_fn(False, use_explicit=True, chi=8)
    A = _make_site(2, 2, seed=14)
    B = _make_site(2, 2, seed=15)
    with pytest.raises(NotImplementedError, match="single-site"):
        split_fn({(0, 0): A, (1, 0): B})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k "dispatch" -v`
Expected: FAIL — split path still routes to fused (energy/grad may match by luck for explicit, but `test_dispatch_multisite_split_raises` fails because no guard exists; and for D where fused≠split memory path the assertion holds only once split is wired). The decisive failure is `test_dispatch_multisite_split_raises` (no `NotImplementedError` raised).

- [ ] **Step 3: Add the import and the branch**

In `src/tenax/algorithms/ipeps_ad_policy.py`, extend the deferred import (~line 208):

```python
    from tenax.algorithms._ctm_energy_ad import (
        ctm_energy_explicit,
        ctm_energy_implicit,
    )
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_implicit,
    )
```

Inside `_ctm_energy_fn`, immediately after `ctm_cfg = get_ctm_cfg()` and `env_init = env_cache.get("envs", None)`, insert the split branch **before** the existing `if use_explicit:` fused block:

```python
        if not ctm_cfg.fuse_virtual_legs:
            if len(site_tensors) != 1:
                raise NotImplementedError(
                    "split-CTM (fuse_virtual_legs=False) supports only the "
                    f"single-site (recipe='1x1') path; got {len(site_tensors)} "
                    "sites. Use fuse_virtual_legs=True for multisite/2-site."
                )
            if getattr(ctm_cfg, "forward_gauge", "phase") == "sigma":
                raise ValueError(
                    "forward_gauge='sigma' is not supported with "
                    "fuse_virtual_legs=False; use 'phase' or fuse_virtual_legs=True."
                )
            if use_explicit:
                return ctm_energy_split_explicit(
                    site_tensors,
                    neighbors,
                    gate,
                    chi=ctm_cfg.chi,
                    warmup_steps=explicit_warmup,
                    backprop_steps=explicit_steps,
                    backward_steps=explicit_backward_steps,
                    chi_I=ctm_cfg.chi_I,
                    renormalize=ctm_cfg.renormalize,
                    energy_fn=energy_fn,
                )
            return ctm_energy_split_implicit(
                site_tensors,
                neighbors,
                gate,
                chi=ctm_cfg.chi,
                max_iter=ctm_cfg.max_iter,
                conv_tol=ctm_cfg.conv_tol,
                chi_I=ctm_cfg.chi_I,
                renormalize=ctm_cfg.renormalize,
                energy_fn=energy_fn,
                adjoint_method=ctm_cfg.adjoint_method,
            )
```

> `explicit_backward_steps` is a closure variable of `make_ctm_energy_fn` (default `None`); it is already in scope.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k "dispatch" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_ad_policy.py tests/test_split_ctm_fuse_flag.py
git commit -m "feat(#463): dispatch make_ctm_energy_fn on fuse_virtual_legs (single-site)"
```

---

## Task 5: Guards for 2-site / multisite / c4v / honeycomb / PESS dispatchers

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (the 2-site dispatcher `_optimize_gs_ad_tensor_2site` ~line 2461 and the multisite dispatcher ~line 3899; the c4v/honeycomb/PESS recipe branches in the top-level `optimize_gs_ad`)
- Modify: `src/tenax/algorithms/_ctm_tensor_c4v.py` (entry `ctm_tensor_c4v` ~line 216, if reached independently)
- Test: `tests/test_split_ctm_fuse_flag.py`

> The `len != 1` guard in Task 4 already covers any path that builds its energy fn through `make_ctm_energy_fn` with >1 site. Task 5 adds **earlier, clearer** guards at the dispatcher entry points so the user gets a path-specific message (c4v, honeycomb, PESS) before any CTM work begins, and covers c4v which uses an isometric projector incompatible with the split path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_ctm_fuse_flag.py`. (These call the public `optimize_gs_ad` with a tiny step budget; they assert the guard fires before convergence work.)

```python
def test_c4v_split_raises():
    import tenax
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    A = _make_site(2, 2, seed=21)
    ctm = CTMConfig(chi=8, fuse_virtual_legs=False)
    cfg = iPEPSConfig(ctm=ctm, gs_c4v=True, gs_max_iter=1)
    with pytest.raises(NotImplementedError, match="c4v"):
        tenax.optimize_gs_ad(A, _heisenberg_gate(), cfg)
```

> Adjust the `iPEPSConfig` construction to match the actual constructor (check `ipeps_config.py` for `iPEPSConfig` field names — `ctm`, `gs_c4v`/`gs_implicit_ad`, `gs_max_iter`). If the c4v switch is `adjoint_method`/`gs_c4v` per `feedback_implicit_ad_path_shared`, use whichever selects the c4v reference path. Mirror the existing c4v test setup in `tests/` for the exact wiring.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k c4v_split -v`
Expected: FAIL — no guard, so it proceeds (and likely errors elsewhere or runs), not `NotImplementedError("...c4v...")`.

- [ ] **Step 3: Add the guards**

At the top of each non-single-site dispatcher (where `config.ctm` is available), add:

```python
    if not config.ctm.fuse_virtual_legs:
        raise NotImplementedError(
            "fuse_virtual_legs=False (split-CTM) is not supported on the "
            "<PATH> path; only the single-site recipe='1x1' path supports it "
            "(#463 Phase 2). Use fuse_virtual_legs=True here."
        )
```

Replace `<PATH>` with `c4v`, `honeycomb`, `PESS`, `2-site`, or `multisite` at each site:
- `_optimize_gs_ad_tensor_2site` → `2-site`
- `_optimize_gs_ad_multisite` → `multisite`
- the c4v recipe branch (or `ctm_tensor_c4v`) → `c4v`
- the honeycomb recipe branch → `honeycomb`
- the PESS recipe branch → `PESS`

> Find the exact insertion points by grepping `def _optimize_gs_ad` and the recipe `if`/`elif` ladder in `optimize_gs_ad`. The guard must read the resolved `config.ctm.fuse_virtual_legs` (the same `config` object the dispatcher already uses).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k "c4v_split" -v`
Expected: PASS. Then run the whole new file: `uv run pytest tests/test_split_ctm_fuse_flag.py -v` — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py src/tenax/algorithms/_ctm_tensor_c4v.py tests/test_split_ctm_fuse_flag.py
git commit -m "feat(#463): guard non-single-site dispatchers against fuse_virtual_legs=False"
```

---

## Task 6: Exports + docs

**Files:**
- Modify: `src/tenax/__init__.py` (`__all__` + lazy export table, next to `compute_energy_split_ctm_tensor` ~line 119 / 502)
- Modify: `README.md` (split-CTM / iPEPS section, if it lists CTM AD entry points)
- Test: `tests/test_split_ctm_fuse_flag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_ctm_fuse_flag.py`:

```python
def test_split_energy_fns_exported():
    import tenax
    assert hasattr(tenax, "ctm_energy_split_explicit")
    assert hasattr(tenax, "ctm_energy_split_implicit")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k exported -v`
Expected: FAIL — `AssertionError` (attributes not on `tenax`).

- [ ] **Step 3: Add the exports**

In `src/tenax/__init__.py`, add to the lazy-import table (mirroring the existing `compute_energy_split_ctm_tensor` entries pointing at `tenax.algorithms._split_ctm_tensor`, but these live in `_split_ctm_energy_ad`):

```python
    "ctm_energy_split_explicit": (
        "tenax.algorithms._split_ctm_energy_ad",
        "ctm_energy_split_explicit",
    ),
    "ctm_energy_split_implicit": (
        "tenax.algorithms._split_ctm_energy_ad",
        "ctm_energy_split_implicit",
    ),
```

and add both names to `__all__`.

> Match the exact lazy-export mechanism used in this file (it uses a name→(module, attr) mapping consumed by `__getattr__`). Copy the surrounding pattern verbatim.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -k exported -v`
Expected: PASS.

- [ ] **Step 5: Update README (only if it enumerates CTM AD entry points)**

If `README.md` lists `ctm_energy_explicit`/`ctm_energy_implicit` or documents iPEPS memory profiles, add a sentence: "Set `CTMConfig.fuse_virtual_legs=False` on the single-site (`recipe='1x1'`) path to use the χ²·D⁴ split-CTM double layer." If the README does not enumerate these, skip (no placeholder edit).

- [ ] **Step 6: Full new-file run + commit**

Run: `uv run pytest tests/test_split_ctm_fuse_flag.py -v`
Expected: all pass.

```bash
git add src/tenax/__init__.py README.md tests/test_split_ctm_fuse_flag.py
git commit -m "docs(#463): export ctm_energy_split_{explicit,implicit}"
```

---

## Task 7: Regression — default path unchanged

**Files:** none (verification only)

- [ ] **Step 1: Run the existing split + CTM core suite**

Run: `uv run pytest tests/test_split_ctm_tensor.py tests/test_fermionic_ed_reference.py tests/test_ctm_env_pad_chi_schedule.py -m core -q`
Expected: all pass (no regression from the new module/imports).

- [ ] **Step 2: Run the full core marker**

Run: `uv run pytest -m core -q`
Expected: all pass. The default `fuse_virtual_legs=True` means every existing path is byte-for-byte unchanged; only the new opt-in branch is added.

- [ ] **Step 3: Final commit (if any doc/memory cleanup)**

```bash
git add -A
git commit -m "chore(#463): Phase 2 single-site split-CTM flag — regression green" --allow-empty
```

---

## Self-Review notes (addressed)

- **Spec coverage:** Component 1 → Task 1; Component 2 (explicit) → Task 2; Component 2 (implicit) → Task 3; Component 3 (dispatch + single-site/knob guards) → Task 4; other-path guards → Task 5; exports/docs → Task 6; "default unchanged" acceptance → Task 7.
- **Type/name consistency:** module `_split_ctm_energy_ad.py`; functions `ctm_energy_split_explicit` / `ctm_energy_split_implicit` used identically in Tasks 2–6; `_extract_single_site` shared; `compute_energy_split_ctm_tensor` (single-site) used in both energy fns.
- **Known open risk (carried from spec):** the implicit Neumann solve assumes `J_envᵀ` spectral radius < 1, which needs a gauge fix on the split sweep. Task 3 Step 4 gives the gauge-fix fallback and a follow-up-issue off-ramp so the explicit path (Task 2) still lands the parity gate if the implicit gauge needs separate work.
