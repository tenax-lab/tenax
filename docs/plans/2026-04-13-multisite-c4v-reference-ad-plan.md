# Multisite c4v_reference AD — Approach A Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the standard JAX eigh backward in the iPEPS CTM projector with the Lorentzian-regularized truncated-eigh backward (lifted from PR #304's `c4v_reference` path) on the explicit-AD path, and auto-promote it for both 1-site and 2-site unit cells, dense + U(1) SymmetricTensor. Targets the χ=8 convergence gap on 2-site shared-tensor C4v Heisenberg (issue #299, E=-0.558 → literature -0.6548).

**Architecture:** Hoist `_truncated_eigh_lorentzian_backward` + `truncated_eigh_regularized` out of `src/tenax/algorithms/_ctm_tensor_c4v_reference_ad.py` into a new shared module `_lorentzian_eigh.py` with a dispatch layer. Re-export from the reference file so PR #304's path is untouched. Add `CTMConfig.projector_backward: Literal["standard", "lorentzian"]` with a single dispatch hook inside `_ctm_projector.py`'s eigh branch. Auto-promote in `ipeps_optimize.py` mirroring the existing `forward_gauge="qr" → "phase"` block.

**Tech Stack:** JAX (custom_vjp, jax.vjp), existing Tenax tensor types (`DenseTensor`, `SymmetricTensor`), pytest with core/slow markers.

**Related docs:** `docs/plans/2026-04-13-multisite-c4v-reference-ad-design.md` (design), `docs/ipeps-code-paths.md` (architectural map), `docs/guide/algorithms/ipeps_ad_paths.md` (user recipe).

**Out of scope:** Implicit fixed-point adjoint (that's Approach B, separate plan), `projector_method="qr"` and `"svd"` backward improvements, FermionParity, honeycomb C3v, multi-state targeting.

---

## Preparation

Before starting:
1. Create a git worktree for isolation: `git worktree add ../tenax-lorentzian-ad -b feat/lorentzian-projector-backward main`
2. Verify `uv run pytest -m core` is green on `main` before touching anything.
3. Confirm PR #315 (the design doc) is open; this implementation PR will reference it.
4. Pre-commit hooks installed (`pre-commit install`) — memory flag `feedback_precommit.md`.

---

## Task 1: Survey the projector eigh call-site

**Files:**
- Read: `src/tenax/algorithms/_ctm_projector.py`
- Read: `src/tenax/algorithms/_ctm_tensor_c4v_reference_ad.py:53-180` (existing Lorentzian kernel)
- Read: `src/tenax/algorithms/ipeps_config.py` (for `CTMConfig` field layout)

**Goal:** Confirm that the `projector_method="eigh"` branch in `_ctm_projector.py` has a single clean eigh call-site for dense tensors and a single one for symmetric tensors. If not, a hoist-into-helper refactor must happen first as Task 1b.

**Actions:**
1. Grep for `eigh` calls inside `_ctm_projector.py` — identify each call-site, function name, and whether it takes a raw array or a `DenseTensor`/`SymmetricTensor`.
2. Record each call-site as `file:line` in a scratch note inside the worktree (e.g. `docs/plans/_scratch-lorentzian-ad.md`, committed with the final PR or deleted before merge).
3. Do **not** modify any code yet.

**Output:** a list of call-sites with their tensor types. If there is exactly one dense call-site and one symmetric call-site, skip Task 1b. Otherwise, create Task 1b to consolidate them.

**Commit:** none — this is pure reconnaissance.

---

## Task 1b (conditional): Consolidate projector eigh call-sites

Only execute if Task 1 found more than one eigh call per tensor type.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_projector.py`

**Step 1:** Write a failing test capturing the current dense projector output for a small fixed input, so the refactor is provably a no-op.

```python
# tests/test_ctm_projector_refactor_noop.py
import pytest
from tenax.algorithms._ctm_projector import build_projector_eigh  # current API
from tenax import DenseTensor
import jax.numpy as jnp

def test_build_projector_eigh_stable_under_refactor():
    rho = DenseTensor(jnp.array([[2.0, 0.5], [0.5, 1.0]]), [...])  # fill in indices
    P, _ = build_projector_eigh(rho, chi_target=2)
    # Record the numerical output from main; pin it in this test.
    expected = jnp.array([...])  # to be filled by the engineer from a main-branch run
    assert jnp.allclose(P.data, expected, rtol=1e-10)
```

**Step 2:** Run it on `main`, record `expected`, commit the test to the branch only after pinning the numbers.

**Step 3:** Refactor — extract every eigh call into one helper:

```python
def _projector_eigh_dense(rho_matrix, chi_target):
    return _lorentzian_eigh_dense(rho_matrix, chi_target, backward="standard")  # placeholder; swaps in Task 6
```

**Step 4:** Re-run the pinned test — must still pass.

**Step 5:** Commit: `refactor(ctm): consolidate projector eigh call-sites (no-op)`.

---

## Task 2: Create the shared Lorentzian kernel module

**Files:**
- Create: `src/tenax/algorithms/_lorentzian_eigh.py`
- Modify: `src/tenax/algorithms/_ctm_tensor_c4v_reference_ad.py` (re-export)
- Test: `tests/test_lorentzian_eigh_kernel.py`

**Step 1:** Write the failing test first.

```python
# tests/test_lorentzian_eigh_kernel.py
import pytest
import jax
import jax.numpy as jnp
from tenax.algorithms._lorentzian_eigh import (
    _lorentzian_eigh_dense,
    truncated_eigh_regularized,
)

def test_lorentzian_dense_matches_fd_on_symmetric_matrix():
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (6, 6))
    A = 0.5 * (A + A.T)
    chi = 4
    def loss(mat):
        _, U = truncated_eigh_regularized(mat, chi)
        return jnp.sum(U ** 2)
    grad_ad = jax.grad(loss)(A)
    eps = 1e-4
    grad_fd = jnp.zeros_like(A)
    for i in range(6):
        for j in range(6):
            Ap = A.at[i, j].add(eps)
            Am = A.at[i, j].add(-eps)
            grad_fd = grad_fd.at[i, j].set((loss(Ap) - loss(Am)) / (2 * eps))
    assert jnp.allclose(grad_ad, grad_fd, atol=1e-5)
```

**Step 2:** Run it — expect `ImportError: cannot import name '_lorentzian_eigh_dense'`.

**Step 3:** Create `_lorentzian_eigh.py`. Move `_truncated_eigh_lorentzian_backward` and `truncated_eigh_regularized` from `_ctm_tensor_c4v_reference_ad.py:53-180` **verbatim** into the new file. Rename `_truncated_eigh_lorentzian_backward` → `_lorentzian_eigh_dense` (the public dense-layer name). Leave `truncated_eigh_regularized` name unchanged — it is the custom_vjp and already well-named.

Add a module docstring pointing to the design doc and the Francuz et al. citation already present in the reference file.

**Step 4:** In `_ctm_tensor_c4v_reference_ad.py`, replace the original definitions with re-exports:

```python
from ._lorentzian_eigh import (
    _lorentzian_eigh_dense as _truncated_eigh_lorentzian_backward,  # alias for back-compat
    truncated_eigh_regularized,
)
```

Keep the alias only if it is imported elsewhere in the repo; otherwise drop it.

**Step 5:** Run the full test module for the reference path: `uv run pytest tests/test_c4v_reference_ad.py -v`. Must stay green — any failure means the move was not verbatim and needs fixing before proceeding.

**Step 6:** Run the new kernel test: `uv run pytest tests/test_lorentzian_eigh_kernel.py::test_lorentzian_dense_matches_fd_on_symmetric_matrix -v`. Must pass.

**Step 7:** Commit: `refactor(ipeps): hoist Lorentzian eigh kernel into shared module`.

---

## Task 3: Add the dense dispatch layer

**Files:**
- Modify: `src/tenax/algorithms/_lorentzian_eigh.py`
- Test: `tests/test_lorentzian_eigh_kernel.py` (extend)

**Step 1:** Write the failing test.

```python
def test_lorentzian_eigh_dispatch_dense_tensor():
    import jax.numpy as jnp
    from tenax import DenseTensor, TensorIndex, FlowDirection
    from tenax.algorithms._lorentzian_eigh import lorentzian_eigh

    idx_in = TensorIndex(dim=4, flow=FlowDirection.IN)
    idx_out = TensorIndex(dim=4, flow=FlowDirection.OUT)
    A = jnp.eye(4) * jnp.array([3.0, 2.0, 1.0, 0.5])
    rho = DenseTensor(A, [idx_in, idx_out])

    eigvals, U, meta = lorentzian_eigh(rho, chi_target=2)
    assert eigvals.shape == (2,)
    assert jnp.allclose(jnp.sort(eigvals)[::-1], jnp.array([3.0, 2.0]))
```

**Step 2:** Run it — expect `ImportError: cannot import name 'lorentzian_eigh'`.

**Step 3:** Add the public dispatch function:

```python
def lorentzian_eigh(tensor, chi_target, *, truncation_correction_enabled=True):
    """Lorentzian-regularized truncated eigendecomposition with type dispatch.

    Dense tensors go to the core kernel directly; SymmetricTensor input is
    handled per-block in Task 4.
    """
    from tenax import DenseTensor, SymmetricTensor
    if isinstance(tensor, DenseTensor):
        eigvals, U = truncated_eigh_regularized(
            tensor.data, chi_target,
            truncation_correction_enabled=truncation_correction_enabled,
        )
        meta = {"backend": "dense"}
        return eigvals, U, meta
    if isinstance(tensor, SymmetricTensor):
        raise NotImplementedError("SymmetricTensor dispatch added in Task 4")
    raise TypeError(f"lorentzian_eigh: unsupported tensor type {type(tensor)}")
```

**Step 4:** Run the test — must pass.

**Step 5:** Commit: `feat(ipeps): add Lorentzian eigh dispatch layer (dense)`.

---

## Task 4: Add the SymmetricTensor per-block dispatch

**Files:**
- Modify: `src/tenax/algorithms/_lorentzian_eigh.py`
- Test: `tests/test_lorentzian_eigh_kernel.py` (extend)

**Step 1:** Write the failing FD-AD test on a U(1) symmetric matrix.

```python
def test_lorentzian_eigh_symmetric_u1_fd_ad():
    import jax
    import jax.numpy as jnp
    from tenax import SymmetricTensor, TensorIndex, FlowDirection
    from tenax.algorithms._lorentzian_eigh import lorentzian_eigh

    # Two U(1) sectors of size 3 and 3, charges 0 and 1.
    idx_in = TensorIndex(charges=[0, 0, 0, 1, 1, 1], flow=FlowDirection.IN)
    idx_out = idx_in.dagger()
    block0 = jnp.array([[2.0, 0.1, 0.0], [0.1, 1.5, 0.2], [0.0, 0.2, 1.0]])
    block1 = jnp.array([[1.8, 0.05, 0.0], [0.05, 1.2, 0.1], [0.0, 0.1, 0.8]])
    rho = SymmetricTensor.from_blocks({0: block0, 1: block1}, [idx_in, idx_out])

    def loss(rho_):
        eigvals, U, _ = lorentzian_eigh(rho_, chi_target=4)
        return jnp.sum(eigvals ** 2)

    grad_ad = jax.grad(loss)(rho)
    # FD check: perturb one entry in each block, confirm agreement within 1e-4.
    ...  # fill in the standard FD loop from test_c4v_reference_ad.py
```

**Step 2:** Run — expect `NotImplementedError: SymmetricTensor dispatch added in Task 4`.

**Step 3:** Extend `lorentzian_eigh` — iterate over sectors, call `truncated_eigh_regularized` per block, reassemble:

```python
if isinstance(tensor, SymmetricTensor):
    from tenax._symmetry import FermionParity
    if any(isinstance(c, FermionParity) for c in tensor.charges):
        raise NotImplementedError(
            "Lorentzian backward for fermionic tensors is deferred; "
            "see docs/plans/2026-04-13-multisite-c4v-reference-ad-design.md"
        )
    blocks_eig = {}
    blocks_U = {}
    # Allocate chi per block according to a global top-chi selection across sectors.
    # Use the same block-allocation policy that tenax.linalg.eigh already uses.
    block_chis = _allocate_chi_per_block(tensor, chi_target)
    for sector, block in tensor.iter_blocks():
        chi_block = block_chis[sector]
        if chi_block == 0:
            continue
        ev, U = truncated_eigh_regularized(
            block, chi_block,
            truncation_correction_enabled=truncation_correction_enabled,
        )
        blocks_eig[sector] = ev
        blocks_U[sector] = U
    eigvals = _concat_block_eigvals(blocks_eig)
    U_out = SymmetricTensor.from_blocks(blocks_U, [...])
    meta = {"backend": "symmetric", "block_chis": block_chis}
    return eigvals, U_out, meta
```

Re-use the existing block-chi allocation policy from `tenax.linalg.eigh` — do **not** reinvent it. If that policy is inline, hoist it into a helper (`_allocate_chi_per_block`) in one commit and call it from both sites. That is a separate step 3b.

**Step 4:** Run the U(1) test — must pass.

**Step 5:** Commit: `feat(ipeps): add SymmetricTensor per-block Lorentzian dispatch`.

---

## Task 5: FermionParity rejection test

**Files:**
- Test: `tests/test_lorentzian_eigh_kernel.py` (extend)

**Step 1:** Write the test.

```python
def test_lorentzian_eigh_fermion_parity_raises():
    import pytest
    from tenax import SymmetricTensor
    from tenax._symmetry import FermionParity
    from tenax.algorithms._lorentzian_eigh import lorentzian_eigh
    rho = _make_fermion_parity_matrix()  # small helper; see fermionic_ipeps tests for idioms
    with pytest.raises(NotImplementedError, match="fermionic"):
        lorentzian_eigh(rho, chi_target=2)
```

**Step 2:** Run — must pass already from the guard added in Task 4.

**Step 3:** Commit: `test(ipeps): fermion parity rejection for Lorentzian eigh`.

---

## Task 6: Add `CTMConfig.projector_backward` field

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py`
- Test: `tests/test_ipeps_config.py` (or wherever `CTMConfig` is currently tested)

**Step 1:** Write a failing test asserting the new field exists and has the correct default.

```python
def test_ctm_config_projector_backward_default_is_standard():
    from tenax import CTMConfig
    config = CTMConfig(chi=8)
    assert config.projector_backward == "standard"
    config2 = CTMConfig(chi=8, projector_backward="lorentzian")
    assert config2.projector_backward == "lorentzian"
    with pytest.raises((ValueError, TypeError)):
        CTMConfig(chi=8, projector_backward="bogus")
```

**Step 2:** Run — expect attribute error.

**Step 3:** Add the field to the dataclass:

```python
projector_backward: Literal["standard", "lorentzian"] = "standard"
```

Add a `__post_init__` validation if `CTMConfig` already uses one; otherwise rely on the `Literal` type check enforced by whatever validation path the repo uses.

**Step 4:** Run — must pass.

**Step 5:** Commit: `feat(ipeps): add CTMConfig.projector_backward field`.

---

## Task 7: Wire `projector_backward` through the projector call-site

**Files:**
- Modify: `src/tenax/algorithms/_ctm_projector.py`
- Modify: call-sites that thread `CTMConfig` into `_ctm_projector` (search with `grep`)

**Step 1:** Write a failing integration test at the projector level.

```python
def test_projector_eigh_uses_lorentzian_backward_when_requested():
    import jax
    from tenax.algorithms._ctm_projector import build_projector_eigh
    rho = _make_small_dense_rho()
    def loss(r, backend):
        P, _ = build_projector_eigh(r, chi_target=2, projector_backward=backend)
        return (P.data ** 2).sum()
    g_std = jax.grad(lambda r: loss(r, "standard"))(rho)
    g_lor = jax.grad(lambda r: loss(r, "lorentzian"))(rho)
    # At non-degenerate spectra the two should agree to 1e-6; the Lorentzian
    # diverges only in the degenerate regime.
    assert jnp.allclose(g_std.data, g_lor.data, atol=1e-6)
```

**Step 2:** Run — expect TypeError on unknown `projector_backward` argument.

**Step 3:** Add the parameter to `build_projector_eigh` (and any wrapper functions between it and the eigh call). Inside the function:

```python
if projector_backward == "lorentzian":
    from ._lorentzian_eigh import lorentzian_eigh
    eigvals, U, _ = lorentzian_eigh(rho, chi_target=chi_target)
else:
    eigvals, U = _standard_eigh(rho, chi_target=chi_target)
```

**Step 4:** Thread `projector_backward` from `CTMConfig` through every function in the `_ctm_projector` call chain. Use `grep` on `projector_method=` in `src/tenax/algorithms/` to find them — any function that already passes `projector_method` must also pass `projector_backward`.

**Step 5:** Run the new integration test + the full reference-mode suite (`uv run pytest tests/test_c4v_reference_ad.py -v`) — must all pass.

**Step 6:** Commit: `feat(ipeps): plumb projector_backward into CTM projector path`.

---

## Task 8: Auto-promotion in `ipeps_optimize.py`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py`
- Test: `tests/test_lorentzian_projector_backward.py` (new file)

**Step 1:** Write three failing tests.

```python
# tests/test_lorentzian_projector_backward.py
import logging
import pytest
from tenax import iPEPSConfig, CTMConfig
from tenax.algorithms.ipeps_optimize import _resolve_projector_backward

def test_auto_promotes_when_explicit_ad_and_eigh():
    ctm = CTMConfig(chi=8, projector_method="eigh")
    config = iPEPSConfig(ctm=ctm, gs_explicit_ad=True)
    resolved = _resolve_projector_backward(config)
    assert resolved.ctm.projector_backward == "lorentzian"

def test_respects_explicit_user_standard():
    ctm = CTMConfig(chi=8, projector_method="eigh", projector_backward="standard")
    config = iPEPSConfig(ctm=ctm, gs_explicit_ad=True)
    resolved = _resolve_projector_backward(config)
    assert resolved.ctm.projector_backward == "standard"

def test_does_not_promote_when_projector_method_is_qr():
    ctm = CTMConfig(chi=8, projector_method="qr")
    config = iPEPSConfig(ctm=ctm, gs_explicit_ad=True)
    resolved = _resolve_projector_backward(config)
    assert resolved.ctm.projector_backward == "standard"

def test_auto_promotion_is_logged(caplog):
    ctm = CTMConfig(chi=8, projector_method="eigh")
    config = iPEPSConfig(ctm=ctm, gs_explicit_ad=True)
    with caplog.at_level(logging.INFO, logger="tenax"):
        _resolve_projector_backward(config)
    assert any("lorentzian" in rec.message.lower() for rec in caplog.records)
```

**Step 2:** Run — expect `ImportError: cannot import name '_resolve_projector_backward'`.

**Step 3:** Add the helper to `ipeps_optimize.py`. Place it next to the existing forward-gauge promotion block; reference that block in a comment so future maintainers see the pattern:

```python
def _resolve_projector_backward(config: iPEPSConfig) -> iPEPSConfig:
    """Auto-promote projector_backward to 'lorentzian' when beneficial.

    Mirrors the existing forward_gauge 'qr' → 'phase' auto-promotion.
    Only acts when:
      - gs_explicit_ad is True, and
      - ctm.projector_method == 'eigh', and
      - ctm.projector_backward was left at the default 'standard'.
    """
    if not config.gs_explicit_ad:
        return config
    if config.ctm.projector_method != "eigh":
        return config
    # "standard" is the sentinel for "user did not opt out"; respect explicit
    # standard values set by the user (they pass through unchanged since the
    # caller needs to know they were explicit).
    if _user_set_projector_backward(config):
        return config
    new_ctm = replace(config.ctm, projector_backward="lorentzian")
    logger.info(
        "projector_backward auto-promoted: standard → lorentzian "
        "(explicit AD + projector_method=eigh)"
    )
    return replace(config, ctm=new_ctm)
```

**Step 4:** Detecting "user explicitly passed standard" is tricky with dataclass defaults. Two options:
- **Option X (preferred):** add a sentinel `"auto"` as the dataclass default, and interpret `"auto"` as "promote if eligible, else use standard".
- **Option Y:** track a companion set `iPEPSConfig._user_set_fields` populated in a custom `__init__`.

Pick **Option X** — it is simpler and matches how `forward_gauge` handles the same problem. Update the test:

```python
def test_ctm_config_projector_backward_default_is_auto():
    assert CTMConfig(chi=8).projector_backward == "auto"
```

Update the `Literal` in Task 6 to `Literal["auto", "standard", "lorentzian"]` and change the default. Then the promotion test becomes: `"auto"` + `gs_explicit_ad=True` + `projector_method="eigh"` → resolves to `"lorentzian"`; `"standard"` stays `"standard"`.

**Step 5:** Wire `_resolve_projector_backward` into the optimizer entry point. Find the existing `forward_gauge` promotion in `ipeps_optimize.py` and add the new call immediately after it.

**Step 6:** Run all new tests + `uv run pytest -m core tests/test_c4v_reference_ad.py tests/test_ipeps_optimize.py` — must all pass.

**Step 7:** Commit: `feat(ipeps): auto-promote projector_backward to lorentzian`.

---

## Task 9: End-to-end FD-AD agreement on 2-site checkerboard (dense, core tier)

**Files:**
- Test: `tests/test_lorentzian_projector_backward.py` (extend)

**Step 1:** Write the test. D=2, χ=4, 2-site Heisenberg, single CTM convergence + energy gradient check.

```python
@pytest.mark.core
def test_2site_checkerboard_lorentzian_grad_matches_fd_dense():
    import jax
    import jax.numpy as jnp
    from tenax import iPEPSConfig, CTMConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad
    from tenax.models import heisenberg_gate
    # Build a small deterministic initial (A, B) pair at D=2.
    A0, B0 = _make_random_2site_state(D=2, d=2, seed=0)
    gate = heisenberg_gate()

    def loss(params, backend):
        A, B = params
        config = iPEPSConfig(
            unit_cell="2site", gs_c4v=True,
            gs_explicit_ad=True, gs_explicit_ad_steps=4, gs_explicit_ad_warmup=0,
            ctm=CTMConfig(chi=4, max_iter=20, projector_method="eigh",
                          projector_backward=backend),
        )
        _, _, E = optimize_gs_ad(gate, A_init=(A, B), config=config, num_steps=0)
        return E

    g_std = jax.grad(loss, argnums=0)((A0, B0), "standard")
    g_lor = jax.grad(loss, argnums=0)((A0, B0), "lorentzian")

    # In the non-degenerate regime at chi=4 the two backwards should still
    # agree to within 1e-3 on each tensor.
    assert jnp.allclose(g_std[0].data, g_lor[0].data, atol=1e-3)
    assert jnp.allclose(g_std[1].data, g_lor[1].data, atol=1e-3)
```

Note: `num_steps=0` should evaluate the loss and return gradient without updating; if the API is different adapt accordingly.

**Step 2:** Run — should pass if Tasks 2–8 landed correctly. If it fails on the standard path, the test is measuring a real divergence and needs a smaller case; if it fails on the lorentzian path, the plumbing is wrong.

**Step 3:** Commit: `test(ipeps): FD-AD agreement for 2-site lorentzian backward (dense)`.

---

## Task 10: End-to-end FD-AD agreement on 2-site checkerboard (U(1) symmetric, core tier)

**Files:**
- Test: `tests/test_lorentzian_projector_backward.py` (extend)

**Step 1:** Write the test — same shape as Task 9 but build `A0, B0` with a U(1) symmetry on the physical index and virtual bonds.

```python
@pytest.mark.core
def test_2site_checkerboard_lorentzian_grad_matches_fd_u1():
    # Same as Task 9 but SymmetricTensor(U1)
    ...
```

Use charges [+1/2, -1/2] on the physical leg and {-1, 0, +1} on each D=2 virtual bond (or whatever convention the rest of the U(1) iPEPS tests already use — **do not invent a new one**, grep `target_charge` and `U1` in `tests/` and copy).

**Step 2:** Run — must pass.

**Step 3:** Commit: `test(ipeps): FD-AD agreement for 2-site lorentzian backward (U1)`.

---

## Task 11: Cross-block degeneracy regression test (core)

**Files:**
- Test: `tests/test_lorentzian_eigh_kernel.py` (extend)

**Step 1:** Write the test — two U(1) sectors with numerically equal eigenvalues, assert gradient is finite and FD-matching.

```python
def test_cross_block_degeneracy_gradient_finite():
    # Two 2x2 blocks with matching eigenvalues {3.0, 1.0}.
    block_a = jnp.diag(jnp.array([3.0, 1.0]))
    block_b = jnp.diag(jnp.array([3.0, 1.0]))
    rho = SymmetricTensor.from_blocks({0: block_a, 1: block_b}, [...])
    def loss(r): ...
    g = jax.grad(loss)(rho)
    assert jnp.all(jnp.isfinite(g.to_jax_array()))
    # FD check at a few entries.
```

**Step 2:** Run — must pass (no code changes needed; the per-block Lorentzian does not mix sectors).

**Step 3:** Commit: `test(ipeps): cross-block degeneracy regression`.

---

## Task 12: Slow-tier convergence benchmark — 2-site χ=8 closes issue #299

**Files:**
- Test: `tests/test_ipeps_lorentzian_convergence.py` (new, `@pytest.mark.slow`)

**Step 1:** Write the test.

```python
import pytest
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad, sublattice_rotate_gate
from tenax.models import heisenberg_gate

@pytest.mark.slow
def test_2site_shared_c4v_chi8_closes_issue_299():
    H = heisenberg_gate()
    H_rot = sublattice_rotate_gate(H)
    config = iPEPSConfig(
        unit_cell="2site",
        gs_c4v=True,
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=False,  # 2-site path skips this (design doc PR #304)
        gs_explicit_ad=True,
        gs_explicit_ad_steps=20,
        gs_explicit_ad_warmup=2,
        gs_projector_method="eigh",
        gs_stall_recovery="reset",
        ctm=CTMConfig(chi=8, max_iter=80, projector_method="eigh"),
        su_init=True,
        num_sweeps=50,
    )
    _, _, E = optimize_gs_ad(H_rot, A_init=None, config=config)
    # Issue #299 baseline (standard backward) reaches E ≈ -0.558.
    # Target: Lorentzian backward reaches E < -0.62 in 50 steps.
    assert E < -0.62, f"Lorentzian backward did not close issue #299 (E={E})"
```

**Step 2:** Run — **expected to pass if the hypothesis is correct**. If it fails, the projector backward is not the sole cause of issue #299 and the plan needs revisiting (see "risks" in the design doc).

**Step 3:** If it passes: commit `test(ipeps): slow benchmark closing issue #299 at chi=8`. Update issue #299 with the new baseline in the PR description.

**Step 4:** If it fails: **stop the plan**. Do not mark issue #299 as fixed. Open a discussion issue with the measured energy trajectory and re-brainstorm — Approach A's hypothesis was wrong and Approach B may need to be elevated ahead of schedule.

---

## Task 13: Slow-tier 1-site parity regression

**Files:**
- Test: `tests/test_ipeps_lorentzian_convergence.py` (extend)

**Step 1:** Write the test — 1-site C4v + Lorentzian auto-promoted vs existing 1-site C4v + standard at D=2, χ=16. Energies should agree to 1e-3 within the first few sweeps.

```python
@pytest.mark.slow
def test_1site_chi16_no_regression_with_lorentzian():
    ...
    assert abs(E_lor - E_std) < 1e-3
```

**Step 2:** Run — must pass.

**Step 3:** Commit: `test(ipeps): 1-site chi=16 parity with lorentzian backward`.

---

## Task 14: Update `docs/ipeps-code-paths.md`

**Files:**
- Modify: `docs/ipeps-code-paths.md`

**Step 1:** Add a new Status Summary row:

```
| Lorentzian projector backward (explicit AD) | **Working** | Auto-default when `projector_method="eigh"` and `gs_explicit_ad=True`; 1-site + 2-site, dense + U(1) symmetric. |
```

**Step 2:** Add a new Config Cheat Sheet row:

```
| Projector backward     | `projector_backward`      | `"auto"`       | `"standard"` / `"lorentzian"` (auto-promoted to lorentzian on explicit AD + eigh) |
```

**Step 3:** Update the pipeline graph — add a branch under the `eigh` projector box marking the Lorentzian variant.

**Step 4:** Commit: `docs(ipeps): document lorentzian projector backward`.

---

## Task 15: Update `docs/guide/algorithms/ipeps_ad_paths.md`

**Files:**
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md`

**Step 1:** Add a note to the recommended-path section describing the new auto-promotion; update any benchmark table that compares explicit-AD paths to include the Lorentzian variant.

**Step 2:** Commit: `docs(ipeps): add lorentzian backward to recommended path guide`.

---

## Task 16: Open the PR

**Step 1:** Verify `uv run pytest -m core` is green.

**Step 2:** Push the branch: `git push -u origin feat/lorentzian-projector-backward`.

**Step 3:** Open the PR with `gh pr create` — title `feat(ipeps): lorentzian projector backward for explicit AD (closes #299)`. Body references design doc PR #315, issue #299, PR #304 / #311 (where the kernel originated), and lists the auto-promotion behavior change prominently in a "Migration" section.

**Step 4:** Add the `run-full-tests` label so the slow tier runs.

**Step 5:** Wait for CI. Do **not** merge until:
- Core CI is green.
- The `run-full-tests` label triggers the slow tier and issue #299's slow test passes.
- At least one human review.

---

## Success criteria (repeat from design doc)

- All core-tier FD-AD tests pass (Tasks 2, 3, 4, 5, 9, 10, 11).
- Issue #299 closed: slow test in Task 12 shows E < −0.62 at χ=8 on 2-site shared-tensor C4v Heisenberg with Lorentzian backward.
- 1-site parity regression in Task 13 shows no regression at D=2, χ=16.
- Reference mode (`c4v_reference`) behavior bit-identical — the full `tests/test_c4v_reference_ad.py` suite stays green throughout Tasks 2–8.

## Kill criteria (when to stop and re-brainstorm)

- Task 12 fails → projector backward is not the cause of issue #299; Approach A does not close the gap; re-open brainstorming with new data.
- Task 2 breaks `tests/test_c4v_reference_ad.py` → the verbatim hoist was not verbatim; fix before touching anything else.
- Task 7 plumbing balloons beyond a single-digit number of call-sites → the projector backward abstraction is wrong for the current CTM architecture; hoist into Task 1b helper first, revisit.
