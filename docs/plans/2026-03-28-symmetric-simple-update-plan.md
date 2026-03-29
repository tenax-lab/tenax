# 2-Site Symmetric Simple Update — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the 1-site SU code path and implement a unified 2-site Tensor-protocol SU that works with both DenseTensor and SymmetricTensor.

**Architecture:** Lift the existing dense 2-site SU algorithm to the Tensor protocol using `contract()` / `truncated_svd()` / `scale_bond_axis()`. Remove all 1-site SU code and the dense 2-site JAX-array code. The `ipeps()` entry point becomes a single 2-site Tensor-protocol path. A `sublattice_rotate` utility converts (A, B) to a single C4v tensor for downstream AD.

**Tech Stack:** JAX, Tenax Tensor protocol (DenseTensor, SymmetricTensor), `tenax.contraction.contractor` (contract, truncated_svd), `tenax.algorithms._tensor_utils` (scale_bond_axis)

---

### Task 1: Write 2-site Tensor-protocol SU functions

**Files:**
- Create functions in: `src/tenax/algorithms/ipeps_simple_update.py`
- Test: `tests/test_ipeps.py`

**Step 1: Write failing tests for 2-site Tensor-protocol SU**

Add to `tests/test_ipeps.py` — new test class after `TestTensorSimpleUpdate`:

```python
class TestTensor2SiteSimpleUpdate:
    """Tests for the 2-site Tensor-protocol simple update."""

    @staticmethod
    def _make_dense_ipeps(key, D=2, d=2):
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        data = jax.random.normal(key, (D, D, D, D, d))
        data = data / (jnp.linalg.norm(data) + 1e-10)
        indices = (
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
        )
        return DenseTensor(data, indices)

    @staticmethod
    def _make_symmetric_ipeps(key, D=2, d=2):
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        indices = (
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
        )
        return SymmetricTensor.random_normal(indices, key)

    @staticmethod
    def _heisenberg_gate():
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_horizontal_dense_tensor_runs(self):
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
        )

        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = self._make_dense_ipeps(key_A)
        B = self._make_dense_ipeps(key_B)
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, lam_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert A_new.labels() == ("u", "d", "l", "r", "phys")
        assert B_new.labels() == ("u", "d", "l", "r", "phys")
        assert np.isfinite(float(A_new.norm()))
        assert np.isfinite(float(B_new.norm()))

    def test_vertical_dense_tensor_runs(self):
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_vertical_tensor,
        )

        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = self._make_dense_ipeps(key_A)
        B = self._make_dense_ipeps(key_B)
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, lam_new = _simple_update_2site_vertical_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert A_new.labels() == ("u", "d", "l", "r", "phys")
        assert B_new.labels() == ("u", "d", "l", "r", "phys")

    def test_symmetric_tensor_2site_runs(self):
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
            _simple_update_2site_vertical_tensor,
        )
        from tenax.core.tensor import SymmetricTensor

        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = self._make_symmetric_ipeps(key_A)
        B = self._make_symmetric_ipeps(key_B)
        gate = _make_trotter_gate_tensor(
            self._heisenberg_gate(), dt=0.01, site_tensor=A
        )
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_h, B_h, lam_h_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert isinstance(A_h, SymmetricTensor)
        assert isinstance(B_h, SymmetricTensor)

        A_v, B_v, lam_v_new = _simple_update_2site_vertical_tensor(
            A_h, B_h, gate, lam_h_new, lam_v, D
        )
        assert isinstance(A_v, SymmetricTensor)
        assert isinstance(B_v, SymmetricTensor)
        assert np.isfinite(float(A_v.norm()))

    def test_returns_different_A_and_B(self):
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
        )

        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = self._make_dense_ipeps(key_A)
        B = self._make_dense_ipeps(key_B)
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, _ = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert not jnp.allclose(A_new.todense(), B_new.todense(), atol=1e-8)

    def test_lambda_normalized(self):
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
            _simple_update_2site_vertical_tensor,
        )

        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = self._make_dense_ipeps(key_A)
        B = self._make_dense_ipeps(key_B)
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        _, _, lam_h_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        _, _, lam_v_new = _simple_update_2site_vertical_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert jnp.allclose(jnp.max(lam_h_new), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.max(lam_v_new), 1.0, atol=1e-10)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ipeps.py::TestTensor2SiteSimpleUpdate -v`
Expected: FAIL — `_simple_update_2site_horizontal_tensor` does not exist

**Step 3: Implement the 2-site Tensor-protocol SU functions**

Add to `src/tenax/algorithms/ipeps_simple_update.py`, after `_make_trotter_gate_tensor`:

```python
def _simple_update_2site_horizontal_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the horizontal bond A.r <-> B.l.

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     Left site tensor with labels (u, d, l, r, phys).
        B:     Right site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_h_new)
    """
    # 1. Absorb outer lambdas onto A (u, d, l) + shared lambda on A.r
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)  # shared bond

    # 2. Absorb outer lambdas onto B (u, d, r) — NOT l (shared)
    B_abs = scale_bond_axis(B, "u", lam_v)
    B_abs = scale_bond_axis(B_abs, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.r with B.l
    A_left = A_abs.relabel("r", "shared")
    B_right = B_abs.relabels({
        "u": "u_B", "d": "d_B", "l": "shared",
        "r": "r_B", "phys": "phys_B",
    })
    theta = contract(A_left, B_right)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. SVD split: A legs = (u, d, l, si_out), B legs = (u_B, d_B, r_B, sj_out)
    U, sigma, Vh, _ = truncated_svd(
        theta,
        left_labels=["u", "d", "l", "si_out"],
        right_labels=["u_B", "d_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
    )

    # 6. New lambda
    lam_h_new = sigma / (jnp.max(sigma) + EPS)
    sqrt_sig = jnp.sqrt(sigma + EPS)

    # 7. Reconstruct A_new: absorb sqrt(sigma) into bond, relabel
    U = U.transpose((0, 1, 2, 4, 3))  # move bond_new to r position
    A_new = U.relabels({"bond_new": "r", "si_out": "phys"})
    A_new = scale_bond_axis(A_new, "r", sqrt_sig)

    # 8. Reconstruct B_new: absorb sqrt(sigma) into bond, relabel
    Vh = Vh.transpose((1, 2, 0, 3, 4))  # move bond_new to l position
    B_new = Vh.relabels({"bond_new": "l", "u_B": "u", "d_B": "d",
                          "r_B": "r", "sj_out": "phys"})
    B_new = scale_bond_axis(B_new, "l", sqrt_sig)

    # 9. Remove outer lambdas
    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)

    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "d", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)

    B_new = scale_bond_axis(B_new, "u", inv_lam_v)
    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize by max element
    a_max = float(A_new.max_abs())
    if a_max > EPS:
        A_new = A_new * (1.0 / a_max)
    b_max = float(B_new.max_abs())
    if b_max > EPS:
        B_new = B_new * (1.0 / b_max)

    return A_new, B_new, lam_h_new


def _simple_update_2site_vertical_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the vertical bond A.d <-> B.u.

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     Top site tensor with labels (u, d, l, r, phys).
        B:     Bottom site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_v_new)
    """
    # 1. Absorb outer lambdas onto A (u, l, r) + shared lambda on A.d
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)  # shared bond

    # 2. Absorb outer lambdas onto B (d, l, r) — NOT u (shared)
    B_abs = scale_bond_axis(B, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "l", lam_h)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.d with B.u
    A_top = A_abs.relabel("d", "shared")
    B_bottom = B_abs.relabels({
        "u": "shared", "d": "d_B", "l": "l_B",
        "r": "r_B", "phys": "phys_B",
    })
    theta = contract(A_top, B_bottom)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. SVD split: A legs = (u, l, r, si_out), B legs = (d_B, l_B, r_B, sj_out)
    U, sigma, Vh, _ = truncated_svd(
        theta,
        left_labels=["u", "l", "r", "si_out"],
        right_labels=["d_B", "l_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
    )

    # 6. New lambda
    lam_v_new = sigma / (jnp.max(sigma) + EPS)
    sqrt_sig = jnp.sqrt(sigma + EPS)

    # 7. Reconstruct A_new: bond_new -> d position
    U = U.transpose((0, 4, 1, 2, 3))  # (u, bond_new, l, r, si_out)
    A_new = U.relabels({"bond_new": "d", "si_out": "phys"})
    A_new = scale_bond_axis(A_new, "d", sqrt_sig)

    # 8. Reconstruct B_new: bond_new -> u position
    Vh = Vh.transpose((0, 1, 2, 3, 4))  # (bond_new, d_B, l_B, r_B, sj_out)
    B_new = Vh.relabels({"bond_new": "u", "d_B": "d", "l_B": "l",
                          "r_B": "r", "sj_out": "phys"})
    B_new = scale_bond_axis(B_new, "u", sqrt_sig)

    # 9. Remove outer lambdas
    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)

    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)
    A_new = scale_bond_axis(A_new, "r", inv_lam_h)

    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "l", inv_lam_h)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize by max element
    a_max = float(A_new.max_abs())
    if a_max > EPS:
        A_new = A_new * (1.0 / a_max)
    b_max = float(B_new.max_abs())
    if b_max > EPS:
        B_new = B_new * (1.0 / b_max)

    return A_new, B_new, lam_v_new
```

**Important:** Check whether `Tensor.max_abs()` exists. If not, use `float(T.norm())` as a fallback for normalization. The design doc says "normalize by max element" — verify which method the Tensor protocol supports and use it.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ipeps.py::TestTensor2SiteSimpleUpdate -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_simple_update.py tests/test_ipeps.py
git commit -m "feat: add 2-site Tensor-protocol simple update functions"
```

---

### Task 2: Remove 1-site SU code and dense 2-site SU code

**Files:**
- Modify: `src/tenax/algorithms/ipeps_simple_update.py`
- Modify: `tests/test_ipeps.py`

**Step 1: Remove 1-site functions from `ipeps_simple_update.py`**

Delete these functions entirely:
- `_simple_update_1x1` (lines 21-61)
- `_simple_update_3leg` (lines 64-99)
- `_simple_update_bond` (lines 102-216)
- `_simple_update_horizontal` (lines 219-230)
- `_simple_update_vertical` (lines 233-242)
- `_absorb_lambdas_tensor` (lines 250-265)
- `_simple_update_horizontal_tensor` (lines 268-334)
- `_simple_update_vertical_tensor` (lines 337-403)

**Step 2: Remove dense 2-site functions from `ipeps_simple_update.py`**

Delete these functions entirely:
- `_simple_update_2site_bond` (lines 461-585)
- `_simple_update_2site_horizontal` (lines 588-660)
- `_simple_update_2site_vertical` (lines 663-735)

After removal, the file should contain only:
- `_make_trotter_gate_tensor`
- `_simple_update_2site_horizontal_tensor` (new, from Task 1)
- `_simple_update_2site_vertical_tensor` (new, from Task 1)

**Step 3: Remove 1-site test classes from `test_ipeps.py`**

Delete:
- `TestSimpleUpdate1x1` class (lines 240-376)
- `test_1x1_backward_compatible` method inside `TestIPEPS2Site` (lines 696-707)
- `test_su_1site_d2_energy` method inside `TestHeisenbergBenchmark` (lines 1040-1051)

**Step 4: Remove dense 2-site test class**

Delete `TestSimpleUpdate2Site` class (lines 538-635) — this tested the dense JAX-array 2-site functions that were just removed. The new `TestTensor2SiteSimpleUpdate` from Task 1 replaces it.

**Step 5: Update imports in `test_ipeps.py`**

The top-level imports at line 32 reference removed functions:
```python
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_1x1,
    _simple_update_2site_horizontal,
    _simple_update_2site_vertical,
)
```
Remove `_simple_update_1x1`, `_simple_update_2site_horizontal`, `_simple_update_2site_vertical` from this import block.

**Step 6: Remove old `TestTensorSimpleUpdate` class**

Delete `TestTensorSimpleUpdate` class (lines 1289-1452) — this tested 1-site Tensor-protocol SU which is removed. The tests for 2-site Tensor-protocol are in the new `TestTensor2SiteSimpleUpdate`.

**Step 7: Run tests to verify removals don't break remaining tests**

Run: `uv run pytest tests/test_ipeps.py -v -x --ignore-glob='*slow*' -k "not ad_d2 and not ad_chi"`
Expected: Remaining tests PASS (some will fail because `ipeps()` still references removed functions — that's expected and fixed in Task 3)

**Step 8: Commit**

```bash
git add src/tenax/algorithms/ipeps_simple_update.py tests/test_ipeps.py
git commit -m "refactor: remove 1-site and dense 2-site SU code"
```

---

### Task 3: Rewrite `ipeps()` entry point (keep `unit_cell` for AD)

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py`
- Modify: `src/tenax/algorithms/ipeps_optimize.py`

**Note:** `unit_cell` stays in `iPEPSConfig` — it is still used by `optimize_gs_ad` to dispatch between 1-site and 2-site AD optimization. `ipeps()` ignores it since SU is always 2-site.

**Step 1: Rewrite `ipeps()` in `src/tenax/algorithms/ipeps.py`**

Replace the entire function body. The new `ipeps()`:

```python
def ipeps(
    hamiltonian_gate: Tensor | jax.Array,
    initial_peps: tuple[Tensor, Tensor] | tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, tuple[Tensor, Tensor], tuple[CTMEnvironment, CTMEnvironment]]:
    """Run iPEPS 2-site simple update + CTM for a 2D quantum lattice model.

    Always uses a 2-site checkerboard unit cell. The returned (A, B) tensors
    are Tensor-protocol objects (DenseTensor or SymmetricTensor).

    Args:
        hamiltonian_gate: 2-site Hamiltonian (d,d,d,d) as Tensor or JAX array.
        initial_peps:     (A, B) tuple of Tensor or JAX array, or None for
                          random initialization.
        config:           iPEPSConfig.

    Returns:
        (energy_per_site, (A, B), (env_A, env_B))
    """
    from tenax.algorithms.ipeps_simple_update import (
        _make_trotter_gate_tensor,
        _simple_update_2site_horizontal_tensor,
        _simple_update_2site_vertical_tensor,
    )

    D = config.max_bond_dim

    # Resolve initial A, B tensors
    if initial_peps is not None:
        A_init, B_init = initial_peps
        # Wrap raw JAX arrays as DenseTensor
        if not isinstance(A_init, Tensor):
            A_init = _wrap_as_dense_tensor(A_init)
        if not isinstance(B_init, Tensor):
            B_init = _wrap_as_dense_tensor(B_init)
    else:
        # Random initialization
        gate_arr = (
            hamiltonian_gate.todense()
            if isinstance(hamiltonian_gate, Tensor)
            else jnp.array(hamiltonian_gate)
        )
        d_phys = gate_arr.shape[0]
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A_init = _wrap_as_dense_tensor(jax.random.normal(key_A, (D, D, D, D, d_phys)))
        B_init = _wrap_as_dense_tensor(jax.random.normal(key_B, (D, D, D, D, d_phys)))

    # Normalize
    a_norm = float(A_init.norm())
    if a_norm > EPS:
        A_init = A_init * (1.0 / a_norm)
    b_norm = float(B_init.norm())
    if b_norm > EPS:
        B_init = B_init * (1.0 / b_norm)

    # Build Trotter gate
    gate = _make_trotter_gate_tensor(hamiltonian_gate, config.dt, site_tensor=A_init)

    # Initialize lambdas from actual tensor bond dimensions
    _labels = A_init.labels()
    D_h = A_init.indices[_labels.index("r")].dim
    D_v = A_init.indices[_labels.index("d")].dim
    lam_h = jnp.ones(D_h)
    lam_v = jnp.ones(D_v)

    # Simple update loop
    A, B = A_init, B_init
    for step in range(config.num_imaginary_steps):
        if step % 2 == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, D
            )
        else:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, D
            )

    # CTM environment (dense path for now)
    A_dense = A.todense()
    B_dense = B.todense()
    env_A, env_B = ctm_2site(A_dense, B_dense, config.ctm)

    # Compute energy
    gate_dense = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d = gate_dense.shape[0]
    energy = compute_energy_ctm_2site(A_dense, B_dense, env_A, env_B, gate_dense, d)

    return float(energy), (A, B), (env_A, env_B)
```

**Step 2: Add `_wrap_as_dense_tensor` helper** (if not already present in `ipeps.py`)

Check if this helper exists. If not, add it:

```python
def _wrap_as_dense_tensor(arr: jax.Array) -> DenseTensor:
    """Wrap a raw 5-leg JAX array as a DenseTensor with trivial U(1) charges."""
    D_u, D_d, D_l, D_r, d = arr.shape
    sym = U1Symmetry()
    indices = (
        TensorIndex(sym, np.zeros(D_u, dtype=np.int32), FlowDirection.OUT, label="u"),
        TensorIndex(sym, np.zeros(D_d, dtype=np.int32), FlowDirection.IN, label="d"),
        TensorIndex(sym, np.zeros(D_l, dtype=np.int32), FlowDirection.OUT, label="l"),
        TensorIndex(sym, np.zeros(D_r, dtype=np.int32), FlowDirection.IN, label="r"),
        TensorIndex(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(arr, indices)
```

**Step 4: Remove dead code from `ipeps.py`**

Delete:
- `_ipeps_tensor()` function
- `_build_1x1_peps()` function
- `_ipeps_2site()` function
- Old imports of removed SU functions (`_simple_update_1x1`, `_absorb_lambdas_tensor`, `_simple_update_horizontal_tensor`, `_simple_update_vertical_tensor`, `_simple_update_2site_horizontal`, `_simple_update_2site_vertical`)

**Step 5: Update `optimize_gs_ad` in `ipeps_optimize.py`**

Two `su_init` paths need updating:

1. **1-site AD path** (around line 114-126): When `su_init=True`, it calls `ipeps()` then does `su_peps.get_tensor((0,0))`. Change to:
```python
if config.su_init:
    _, (A_su, B_su), _ = ipeps(gate, None, config)
    A_init = A_su  # Use sublattice A as starting point for 1-site AD
```

2. **2-site AD path** (around line 291-303): Same pattern:
```python
if config.su_init:
    from tenax.algorithms.ipeps import ipeps
    su_config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=config.num_imaginary_steps,
        dt=config.dt,
        ctm=config.ctm,
    )
    _, (A_su, B_su), _ = ipeps(gate, None, su_config)
    A = A_su if isinstance(A_su, Tensor) else _wrap_as_dense_tensor(A_su)
    B = B_su if isinstance(B_su, Tensor) else _wrap_as_dense_tensor(B_su)
```

Keep all `unit_cell` references in `ipeps_optimize.py` — AD still uses them.

**Step 5: Update remaining test files**

In `test_ipeps.py`:
- `TestIPEPS2Site`: Keep `unit_cell="2site"` in configs (still valid, just ignored by SU). Update assertions for new return type (tuple instead of TensorNetwork).
- `TestIPEPSRun`: Update for new signature — `ipeps()` now always returns `(A, B)` tuple.
- `TestHeisenbergBenchmark`: Update `test_ad_d2_energy` — it currently does `peps_su.get_tensor((0,0))`, change to unpack tuple from `ipeps()`.
- `TestOptimizeGsAd2Site`: Keep as-is (AD path unchanged).

**Step 6: Run all core tests**

Run: `uv run pytest -m core tests/test_ipeps.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps.py \
        src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps.py
git commit -m "refactor: unify ipeps() to always-2site Tensor-protocol path"
```

---

### Task 4: Add `sublattice_rotate` utility

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py`
- Modify: `src/tenax/__init__.py`
- Test: `tests/test_ipeps.py`

**Step 1: Write failing test**

```python
class TestSublatticeRotate:
    def test_rotate_produces_c4v_tensor(self):
        from tenax.algorithms.ipeps import sublattice_rotate
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        sym = U1Symmetry()
        D, d = 2, 2
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)

        def make_tensor(key):
            data = jax.random.normal(key, (D, D, D, D, d))
            indices = (
                TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
                TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
                TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
                TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
                TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
            )
            return DenseTensor(data / (jnp.linalg.norm(data) + 1e-10), indices)

        key_A, key_B = jax.random.split(jax.random.PRNGKey(42))
        A = make_tensor(key_A)
        B = make_tensor(key_B)

        C = sublattice_rotate(A, B)
        assert C.labels() == ("u", "d", "l", "r", "phys")
        assert np.isfinite(float(C.norm()))

    def test_rotate_identical_tensors_returns_same(self):
        """If A == B (no sublattice breaking), rotation should return ~A."""
        from tenax.algorithms.ipeps import sublattice_rotate
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        sym = U1Symmetry()
        D, d = 2, 2
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        data = jax.random.normal(jax.random.PRNGKey(0), (D, D, D, D, d))
        # Make it C4v symmetric: A[u,d,l,r,s] = A[d,u,r,l,s]
        data = 0.5 * (data + data.transpose(1, 0, 3, 2, 4))
        indices = (
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
        )
        A = DenseTensor(data / (jnp.linalg.norm(data) + 1e-10), indices)
        C = sublattice_rotate(A, A)
        # Should be proportional to A
        assert jnp.allclose(C.todense(), A.todense(), atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps.py::TestSublatticeRotate -v`
Expected: FAIL — `sublattice_rotate` does not exist

**Step 3: Implement `sublattice_rotate`**

Add to `src/tenax/algorithms/ipeps.py`:

```python
def sublattice_rotate(A: Tensor, B: Tensor) -> Tensor:
    """Average A and pi-rotated B into a single C4v-symmetric tensor.

    Applies a 180-degree rotation to B (permute u<->d, l<->r), then
    averages with A. Useful for transitioning from 2-site SU to 1-site
    C4v AD optimization.

    Args:
        A: Sublattice-A tensor with labels (u, d, l, r, phys).
        B: Sublattice-B tensor with labels (u, d, l, r, phys).

    Returns:
        Single tensor (A + rot(B)) / 2 with labels (u, d, l, r, phys).
    """
    # Pi rotation: swap u<->d, l<->r
    B_rot = B.relabels({"u": "d", "d": "u", "l": "r", "r": "l"})
    # Reorder to match A's label order
    B_rot = B_rot.transpose_by_labels(A.labels())
    return (A + B_rot) * 0.5
```

**Important:** Verify that `Tensor.transpose_by_labels()` exists. If not, use `B_rot.transpose(...)` with the correct permutation to match A's label ordering `(u, d, l, r, phys)`.

**Step 4: Export from `src/tenax/__init__.py`**

Add `sublattice_rotate` to the import from `tenax.algorithms.ipeps` and to `__all__`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_ipeps.py::TestSublatticeRotate -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps.py src/tenax/__init__.py tests/test_ipeps.py
git commit -m "feat: add sublattice_rotate utility for 2-site to 1-site C4v"
```

---

### Task 5: Update example and final cleanup

**Files:**
- Modify: `examples/heisenberg_ipeps_su.py`
- Modify: `src/tenax/algorithms/ipeps.py` (docstring)
- Modify: `src/tenax/__init__.py` (if needed)

**Step 1: Update example**

Rewrite `examples/heisenberg_ipeps_su.py` to remove the 1x1 run. The example should only show the 2-site run, and update the config to not use `unit_cell`:

```python
def run_simple_update(gate, D, chi, num_steps, dt, label=""):
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=num_steps,
        dt=dt,
        ctm=CTMConfig(chi=chi, max_iter=100),
    )
    # ... same print/timing logic ...
    energy, (A, B), envs = ipeps(gate, initial_peps=None, config=config)
    # ...

def main():
    gate = heisenberg_gate()
    run_simple_update(gate, D=2, chi=16, num_steps=200, dt=0.3,
                      label="2-site checkerboard, D=2")
```

**Step 2: Update module docstring in `ipeps.py`**

Update the module docstring to reflect that SU is always 2-site.

**Step 3: Run full core test suite**

Run: `uv run pytest -m core -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add examples/heisenberg_ipeps_su.py src/tenax/algorithms/ipeps.py
git commit -m "docs: update example and docstrings for 2-site-only SU"
```

---

### Task 6: Regression test — DenseTensor matches old dense path

**Files:**
- Test: `tests/test_ipeps.py`

**Step 1: Write regression test**

```python
def test_dense_tensor_2site_heisenberg_energy(self):
    """DenseTensor 2-site SU should give E < -0.63 (same as old dense path)."""
    gate = self._heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=200,
        dt=0.3,
        ctm=CTMConfig(chi=10, max_iter=40),
    )
    energy, (A, B), _ = ipeps(gate, None, config)
    assert float(energy) < -0.63, (
        f"Energy {float(energy)} not low enough — should match old dense 2-site"
    )
```

Add this to the `TestIPEPS2Site` class (which should already have the `test_2site_heisenberg_D2_energy` test — verify it still passes with the new code path).

**Step 2: Run test**

Run: `uv run pytest tests/test_ipeps.py::TestIPEPS2Site::test_2site_heisenberg_D2_energy -v`
Expected: PASS with E < -0.63

**Step 3: Commit (if new test was added)**

```bash
git add tests/test_ipeps.py
git commit -m "test: verify DenseTensor 2-site SU energy matches old path"
```

---

### Task 7: Final verification

**Step 1: Run full core tests**

Run: `uv run pytest -m core -v`
Expected: All PASS

**Step 2: Run non-slow tests**

Run: `uv run pytest -m "not slow" -v`
Expected: All PASS

**Step 3: Check for stale imports/references**

Run: `grep -rn "unit_cell" src/tenax/ tests/test_ipeps.py`
Expected: No remaining references to `unit_cell` in production code or tests (except possibly comments/docs)

Run: `grep -rn "_simple_update_1x1\|_simple_update_3leg\|_build_1x1_peps\|_ipeps_tensor\b" src/tenax/ tests/`
Expected: No references

**Step 4: Commit any fixes**

If stale references found, fix and commit.
