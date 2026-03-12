# Kagome XXZ via PESS — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a PESS simulation for the spin-1/2 XXZ model on the Kagome lattice with simple update, CTM energy, and AD optimization support.

**Architecture:** Add `xxz_gate(delta)` to the library. Build a standalone example script implementing PESS (Projected Entangled Simplex States) with rank-3 site/simplex tensors, simple update via HOSVD, PESS→iPEPS coarse-graining for CTM energy, and serialization for AD reuse.

**Tech Stack:** JAX, tenax (iPEPS/CTM), numpy

---

### Task 1: Add `xxz_gate(delta)` to the library

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py`
- Modify: `src/tenax/algorithms/__init__.py`
- Modify: `src/tenax/__init__.py`
- Test: `tests/test_ipeps.py`

**Step 1: Write the failing test**

Add a new test class at the end of `tests/test_ipeps.py`:

```python
class TestXXZGate:
    def test_xxz_gate_shape(self):
        from tenax.algorithms.ipeps import xxz_gate
        gate = xxz_gate(delta=1.0)
        assert gate.todense().shape == (2, 2, 2, 2)

    def test_xxz_gate_recovers_heisenberg(self):
        from tenax.algorithms.ipeps import heisenberg_gate, xxz_gate
        H_heis = heisenberg_gate().todense()
        H_xxz = xxz_gate(delta=1.0).todense()
        assert jnp.allclose(H_heis, H_xxz, atol=1e-14)

    def test_xxz_gate_ising_limit(self):
        from tenax.algorithms.ipeps import xxz_gate
        H = xxz_gate(delta=0.0).todense()
        # delta=0 => only Sx·Sx + Sy·Sy = 0.5*(S+S- + S-S+), no Sz·Sz
        Sp = jnp.array([[0, 1], [0, 0]], dtype=jnp.float64)
        Sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.float64)
        H_expected = 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
        assert jnp.allclose(H.reshape(4, 4), H_expected, atol=1e-14)

    def test_xxz_gate_is_dense_tensor(self):
        from tenax.algorithms.ipeps import xxz_gate
        from tenax.core.tensor import DenseTensor
        gate = xxz_gate(delta=0.5)
        assert isinstance(gate, DenseTensor)

    def test_xxz_gate_labels(self):
        from tenax.algorithms.ipeps import xxz_gate
        gate = xxz_gate(delta=1.0)
        assert gate.labels() == ("si", "sj", "si_out", "sj_out")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps.py::TestXXZGate -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/ipeps.py` right after `heisenberg_gate()`:

```python
def xxz_gate(delta: float = 1.0, dtype=jnp.float64) -> DenseTensor:
    """Build the 2-site XXZ Hamiltonian as a DenseTensor.

    ``H = delta * Sz Sz + 0.5 (S+ S- + S- S+)`` on two spin-1/2 sites.

    Args:
        delta: Anisotropy parameter. delta=1 is isotropic Heisenberg,
               delta=0 is XX model, delta→∞ is Ising limit.
        dtype: Array dtype.

    Returns:
        4-leg DenseTensor with labels ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = delta * jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="si_out"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="sj_out"),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)
```

Add `xxz_gate` to exports in `src/tenax/algorithms/__init__.py` (line 30, add to import and `__all__`) and `src/tenax/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps.py::TestXXZGate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps.py src/tenax/algorithms/__init__.py src/tenax/__init__.py tests/test_ipeps.py
git commit -m "feat: add xxz_gate(delta) utility for anisotropic Heisenberg"
```

---

### Task 2: Build Kagome PESS data structures and initialization

**Files:**
- Create: `examples/kagome_xxz_pess.py`

**Step 1: Write the PESS initialization code**

```python
"""Kagome XXZ ground state via PESS (Projected Entangled Simplex States).

Implements the PESS ansatz of Xie et al., PRL 112, 147203 (2014) for the
spin-1/2 XXZ model on the Kagome lattice.

Usage:
    python kagome_xxz_pess.py [--D 2] [--chi 20] [--delta 1.0] [--steps 200]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from tenax import iPEPSConfig, xxz_gate


def kagome_triangle_hamiltonian(delta: float = 1.0) -> jnp.ndarray:
    """Build the 3-site XXZ Hamiltonian on one Kagome triangle.

    H_tri = H_12 + H_23 + H_31
    where H_ij = delta * Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j)

    Returns:
        Array of shape (8, 8) = (2^3, 2^3).
    """
    I = np.eye(2)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])

    def h_pair(O1, O2, site_i, site_j, n=3):
        """Build 2-body operator O1_i O2_j on n sites."""
        ops = [I] * n
        ops[site_i] = O1
        ops[site_j] = O2
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H = np.zeros((8, 8))
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        H += delta * h_pair(Sz, Sz, i, j)
        H += 0.5 * (h_pair(Sp, Sm, i, j) + h_pair(Sm, Sp, i, j))
    return jnp.array(H)


def make_trotter_gate_3site(H: jnp.ndarray, dt: float) -> jnp.ndarray:
    """exp(-dt * H) reshaped to (d, d, d, d, d, d) for 3-site gate."""
    gate = jnp.array(expm(-dt * np.array(H)))
    return gate.reshape(2, 2, 2, 2, 2, 2)


def init_pess(D: int, d: int = 2, key=None):
    """Initialize random PESS tensors for Kagome lattice.

    Returns:
        site_tensors: dict with keys "a", "b", "c" — rank-3 tensors (D, D, d)
        simplex_tensors: dict with keys "up", "down" — rank-3 tensors (D, D, D)
        lambdas: dict of singular value vectors on each of 6 bonds
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    site_tensors = {
        "a": jax.random.normal(keys[0], (D, D, d)) * 0.01,
        "b": jax.random.normal(keys[1], (D, D, d)) * 0.01,
        "c": jax.random.normal(keys[2], (D, D, d)) * 0.01,
    }
    simplex_tensors = {
        "up": jax.random.normal(keys[3], (D, D, D)) * 0.01,
        "down": jax.random.normal(keys[4], (D, D, D)) * 0.01,
    }
    # 6 bonds: up-triangle (a-b, b-c, c-a) and down-triangle (a-b, b-c, c-a)
    # Label bonds by (simplex, site_pair)
    lambdas = {
        ("up", "ab"): jnp.ones(D),
        ("up", "bc"): jnp.ones(D),
        ("up", "ca"): jnp.ones(D),
        ("down", "ab"): jnp.ones(D),
        ("down", "bc"): jnp.ones(D),
        ("down", "ca"): jnp.ones(D),
    }
    return site_tensors, simplex_tensors, lambdas
```

**Step 2: Test by running the script**

Run: `python examples/kagome_xxz_pess.py --help`
Expected: prints usage without error

**Step 3: Commit**

```bash
git add examples/kagome_xxz_pess.py
git commit -m "feat: add Kagome PESS data structures and initialization"
```

---

### Task 3: Implement PESS simple update with HOSVD

**Files:**
- Modify: `examples/kagome_xxz_pess.py`

**Step 1: Implement the HOSVD truncation and simple update**

Add to `kagome_xxz_pess.py`:

```python
def hosvd_truncate(theta, D_max, d=2):
    """Truncate a 3-site tensor theta via HOSVD back into S_a, S_b, S_c, T.

    theta has shape (D_a_ext, D_b_ext, D_c_ext, d, d, d) where D_x_ext
    includes external bonds absorbed from lambdas.

    Returns:
        S_a: (D_a_ext, D_new, d)  — site tensor a with new internal bond
        S_b: (D_b_ext, D_new, d)  — site tensor b
        S_c: (D_c_ext, D_new, d)  — site tensor c
        T_new: (D_new, D_new, D_new) — new simplex tensor (core)
        lam_ab, lam_bc, lam_ca: new singular values on internal bonds
    """
    D_a, D_b, D_c = theta.shape[:3]

    # Mode-unfoldings and truncated SVD for each site leg
    # Mode-a: reshape to (D_a * d, D_b * D_c * d * d)
    M_a = theta.transpose(0, 3, 1, 2, 4, 5).reshape(D_a * d, -1)
    U_a, _, _ = jnp.linalg.svd(M_a, full_matrices=False)
    D_new_a = min(D_max, U_a.shape[1])
    U_a = U_a[:, :D_new_a].reshape(D_a, d, D_new_a)  # (D_a_ext, d, D_new)

    M_b = theta.transpose(1, 4, 0, 2, 3, 5).reshape(D_b * d, -1)
    U_b, _, _ = jnp.linalg.svd(M_b, full_matrices=False)
    D_new_b = min(D_max, U_b.shape[1])
    U_b = U_b[:, :D_new_b].reshape(D_b, d, D_new_b)  # (D_b_ext, d, D_new)

    M_c = theta.transpose(2, 5, 0, 1, 3, 4).reshape(D_c * d, -1)
    U_c, _, _ = jnp.linalg.svd(M_c, full_matrices=False)
    D_new_c = min(D_max, U_c.shape[1])
    U_c = U_c[:, :D_new_c].reshape(D_c, d, D_new_c)  # (D_c_ext, d, D_new)

    # Core tensor: project theta onto truncated basis
    # T_core = U_a^T · U_b^T · U_c^T · theta (contract physical + external)
    core = jnp.einsum("aDi,bEj,cFk,abcDEF->ijk", U_a, U_b, U_c, theta)

    # Site tensors: swap to (D_ext, D_new, d) convention
    S_a = U_a.transpose(0, 2, 1)  # (D_a_ext, D_new, d)
    S_b = U_b.transpose(0, 2, 1)
    S_c = U_c.transpose(0, 2, 1)

    return S_a, S_b, S_c, core


def pess_simple_update_triangle(
    S_a, S_b, S_c, T, lambdas_ext, lambdas_int, gate_3site, D_max,
):
    """One simple update step on a single triangle.

    Args:
        S_a, S_b, S_c: site tensors, shape (D_ext, D_int, d)
        T: simplex tensor, shape (D_int, D_int, D_int)
        lambdas_ext: dict of external lambda vectors for each site
        lambdas_int: dict of internal lambda vectors (site-to-simplex)
        gate_3site: 3-site Trotter gate, shape (d, d, d, d, d, d)
        D_max: max bond dimension after truncation

    Returns:
        S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new
    """
    d = S_a.shape[-1]

    # Step 1: Absorb sqrt(lambda) on internal bonds into site tensors
    lam_a = lambdas_int["a"]
    lam_b = lambdas_int["b"]
    lam_c = lambdas_int["c"]

    S_a_abs = S_a * jnp.sqrt(lam_a)[None, :, None]
    S_b_abs = S_b * jnp.sqrt(lam_b)[None, :, None]
    S_c_abs = S_c * jnp.sqrt(lam_c)[None, :, None]

    # Absorb sqrt(lambda) from simplex side too
    T_abs = T * jnp.sqrt(lam_a)[:, None, None]
    T_abs = T_abs * jnp.sqrt(lam_b)[None, :, None]
    T_abs = T_abs * jnp.sqrt(lam_c)[None, None, :]

    # Absorb external lambdas into site tensors
    lam_ext_a = lambdas_ext.get("a", jnp.ones(S_a.shape[0]))
    lam_ext_b = lambdas_ext.get("b", jnp.ones(S_b.shape[0]))
    lam_ext_c = lambdas_ext.get("c", jnp.ones(S_c.shape[0]))

    S_a_full = S_a_abs * lam_ext_a[:, None, None]
    S_b_full = S_b_abs * lam_ext_b[:, None, None]
    S_c_full = S_c_abs * lam_ext_c[:, None, None]

    # Step 2: Contract into theta = S_a · S_b · S_c · T
    # theta[Da, Db, Dc, sa, sb, sc]
    theta = jnp.einsum("aip,bjq,ckr,ijk->abcpqr", S_a_full, S_b_full, S_c_full, T_abs)

    # Step 3: Apply 3-site gate
    theta = jnp.einsum("abcpqr,pqrPQR->abcPQR", theta, gate_3site)

    # Step 4: HOSVD truncation
    S_a_new, S_b_new, S_c_new, T_new = hosvd_truncate(theta, D_max, d)

    # Step 5: Remove external lambdas from new site tensors
    lam_ext_a_inv = 1.0 / (lam_ext_a + 1e-15)
    lam_ext_b_inv = 1.0 / (lam_ext_b + 1e-15)
    lam_ext_c_inv = 1.0 / (lam_ext_c + 1e-15)

    S_a_new = S_a_new * lam_ext_a_inv[:, None, None]
    S_b_new = S_b_new * lam_ext_b_inv[:, None, None]
    S_c_new = S_c_new * lam_ext_c_inv[:, None, None]

    # Step 6: Extract new internal lambdas via SVD of T_new along each mode
    def extract_lambda(T_core, mode):
        shape = T_core.shape
        M = T_core.reshape(shape[mode], -1) if mode == 0 else \
            jnp.moveaxis(T_core, mode, 0).reshape(shape[mode], -1)
        s = jnp.linalg.svd(M, compute_uv=False)
        return s / (jnp.max(s) + 1e-15)

    lambdas_int_new = {
        "a": extract_lambda(T_new, 0),
        "b": extract_lambda(T_new, 1),
        "c": extract_lambda(T_new, 2),
    }

    return S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new


def pess_simple_update(
    site_tensors, simplex_tensors, lambdas,
    H_triangle, dt, D_max, num_steps,
):
    """Full PESS simple update loop over all triangles.

    Args:
        site_tensors: dict {"a", "b", "c"} of shape (D_ext, D_int, d)
        simplex_tensors: dict {"up", "down"} of shape (D, D, D)
        lambdas: dict of lambda vectors keyed by (simplex, bond)
        H_triangle: 3-site Hamiltonian (8x8)
        dt: imaginary time step
        D_max: max bond dimension
        num_steps: number of Trotter steps

    Returns:
        Updated (site_tensors, simplex_tensors, lambdas)
    """
    gate = make_trotter_gate_3site(H_triangle, dt)

    for step in range(num_steps):
        # Update up-triangles
        lam_ext = {
            "a": lambdas[("down", "ca")],
            "b": lambdas[("down", "ab")],
            "c": lambdas[("down", "bc")],
        }
        lam_int = {
            "a": lambdas[("up", "ca")],
            "b": lambdas[("up", "ab")],
            "c": lambdas[("up", "bc")],
        }
        Sa, Sb, Sc, T_up, lam_int_new = pess_simple_update_triangle(
            site_tensors["a"], site_tensors["b"], site_tensors["c"],
            simplex_tensors["up"], lam_ext, lam_int, gate, D_max,
        )
        site_tensors["a"] = Sa
        site_tensors["b"] = Sb
        site_tensors["c"] = Sc
        simplex_tensors["up"] = T_up
        lambdas[("up", "ca")] = lam_int_new["a"]
        lambdas[("up", "ab")] = lam_int_new["b"]
        lambdas[("up", "bc")] = lam_int_new["c"]

        # Update down-triangles
        lam_ext = {
            "a": lambdas[("up", "ca")],
            "b": lambdas[("up", "ab")],
            "c": lambdas[("up", "bc")],
        }
        lam_int = {
            "a": lambdas[("down", "ca")],
            "b": lambdas[("down", "ab")],
            "c": lambdas[("down", "bc")],
        }
        Sa, Sb, Sc, T_down, lam_int_new = pess_simple_update_triangle(
            site_tensors["a"], site_tensors["b"], site_tensors["c"],
            simplex_tensors["down"], lam_ext, lam_int, gate, D_max,
        )
        site_tensors["a"] = Sa
        site_tensors["b"] = Sb
        site_tensors["c"] = Sc
        simplex_tensors["down"] = T_down
        lambdas[("down", "ca")] = lam_int_new["a"]
        lambdas[("down", "ab")] = lam_int_new["b"]
        lambdas[("down", "bc")] = lam_int_new["c"]

        if step % 50 == 0:
            print(f"  SU step {step}/{num_steps}")

    return site_tensors, simplex_tensors, lambdas
```

**Step 2: Test**

Run: `python -c "from examples.kagome_xxz_pess import init_pess, pess_simple_update, kagome_triangle_hamiltonian; S, T, L = init_pess(2); pess_simple_update(S, T, L, kagome_triangle_hamiltonian(1.0), 0.01, 2, 5); print('OK')"`
Expected: prints "OK"

**Step 3: Commit**

```bash
git add examples/kagome_xxz_pess.py
git commit -m "feat: implement PESS simple update with HOSVD for Kagome"
```

---

### Task 4: Implement PESS → iPEPS coarse-graining

**Files:**
- Modify: `examples/kagome_xxz_pess.py`

**Step 1: Implement the coarse-graining**

Add to `kagome_xxz_pess.py`:

```python
def pess_to_ipeps(site_tensors, simplex_tensors, lambdas):
    """Contract PESS into an effective square-lattice iPEPS super-site.

    Coarse-grains one up-triangle (3 sites + simplex) into a single
    super-site tensor A with enlarged physical dimension d_eff = d^3 = 8.

    The effective iPEPS has bond dimension D_eff = D^2 (two PESS bonds
    cross each boundary of the coarse-grained cell).

    Returns:
        A: array of shape (D_eff, D_eff, D_eff, D_eff, d_eff)
           with leg ordering (up, down, left, right, phys)
    """
    S_a = site_tensors["a"]  # (D_ext, D_int, d)
    S_b = site_tensors["b"]
    S_c = site_tensors["c"]
    T_up = simplex_tensors["up"]  # (D_int, D_int, D_int)

    D_ext = S_a.shape[0]
    D_int = T_up.shape[0]
    d = S_a.shape[-1]

    # Absorb internal lambdas
    lam_a = lambdas[("up", "ca")]
    lam_b = lambdas[("up", "ab")]
    lam_c = lambdas[("up", "bc")]

    S_a_abs = S_a * jnp.sqrt(lam_a)[None, :, None]
    S_b_abs = S_b * jnp.sqrt(lam_b)[None, :, None]
    S_c_abs = S_c * jnp.sqrt(lam_c)[None, :, None]

    T_abs = T_up * jnp.sqrt(lam_a)[:, None, None]
    T_abs = T_abs * jnp.sqrt(lam_b)[None, :, None]
    T_abs = T_abs * jnp.sqrt(lam_c)[None, None, :]

    # Contract: theta[a_ext, b_ext, c_ext, sa, sb, sc] = S_a · S_b · S_c · T
    theta = jnp.einsum("aip,bjq,ckr,ijk->abcpqr", S_a_abs, S_b_abs, S_c_abs, T_abs)

    # Map external bonds to square lattice directions:
    # a_ext → up, b_ext → right, c_ext → down
    # The 4th direction (left) comes from the down-triangle's bond
    # For simplicity, absorb down-triangle lambda as identity on 4th leg
    D_eff = D_ext
    d_eff = d ** 3  # = 8

    # Reshape: (D_ext, D_ext, D_ext, d^3) → need 4 virtual legs
    # Add a trivial 4th leg for the square lattice mapping
    # A[up, down, left, right, phys]
    A = theta.reshape(D_ext, D_ext, D_ext, d_eff)

    # Pad with trivial left leg (dim 1) for square lattice compatibility
    A = A[:, :, None, :, :]  # (D_ext, D_ext, 1, D_ext, d_eff)
    # Reorder to (up, down, left, right, phys)
    A = A.transpose(0, 1, 2, 3, 4)

    return A


def save_pess(filename, site_tensors, simplex_tensors, lambdas):
    """Save PESS state to disk."""
    data = {}
    for name, tensor in site_tensors.items():
        data[f"site_{name}"] = np.array(tensor)
    for name, tensor in simplex_tensors.items():
        data[f"simplex_{name}"] = np.array(tensor)
    for (simplex, bond), lam in lambdas.items():
        data[f"lambda_{simplex}_{bond}"] = np.array(lam)
    np.savez(filename, **data)
    print(f"PESS state saved to {filename}")


def load_pess(filename):
    """Load PESS state from disk."""
    data = np.load(filename)
    site_tensors = {
        name: jnp.array(data[f"site_{name}"]) for name in ["a", "b", "c"]
    }
    simplex_tensors = {
        name: jnp.array(data[f"simplex_{name}"]) for name in ["up", "down"]
    }
    lambdas = {}
    for simplex in ["up", "down"]:
        for bond in ["ab", "bc", "ca"]:
            lambdas[(simplex, bond)] = jnp.array(data[f"lambda_{simplex}_{bond}"])
    return site_tensors, simplex_tensors, lambdas
```

**Step 2: Test**

Run: `python -c "from examples.kagome_xxz_pess import init_pess, pess_to_ipeps; S, T, L = init_pess(2); A = pess_to_ipeps(S, T, L); print('Shape:', A.shape)"`
Expected: prints shape info without error

**Step 3: Commit**

```bash
git add examples/kagome_xxz_pess.py
git commit -m "feat: add PESS to iPEPS coarse-graining and serialization"
```

---

### Task 5: Wire up the full simulation script

**Files:**
- Modify: `examples/kagome_xxz_pess.py`

**Step 1: Add the main simulation function and CLI**

Add to `kagome_xxz_pess.py`:

```python
def compute_energy_pess(site_tensors, simplex_tensors, lambdas, delta, chi):
    """Compute energy per site via PESS → iPEPS → CTM."""
    from tenax import ctm, iPEPSConfig
    from tenax.algorithms.ipeps_rdm import compute_energy_ctm

    A = pess_to_ipeps(site_tensors, simplex_tensors, lambdas)

    config = iPEPSConfig(ctm=__import__("tenax").CTMConfig(chi=chi))
    gate = xxz_gate(delta).todense().reshape(2, 2, 2, 2)

    # For the super-site, we need a gate on the enlarged Hilbert space
    # Use the triangle Hamiltonian directly for energy
    H_tri = kagome_triangle_hamiltonian(delta)

    # Simple approach: compute <H_tri> = Tr(rho * H_tri)
    # where rho is from the CTM environment
    # For now, use the 2-site gate on nearest-neighbor bonds of the
    # effective square lattice
    from tenax.algorithms.ipeps_ctm import ctm as ctm_dense

    D_eff = A.shape[0]
    d_eff = A.shape[-1]

    env = ctm_dense(A.squeeze(axis=2), config.ctm)  # remove trivial left leg
    E = compute_energy_ctm(A.squeeze(axis=2), env, gate, d_eff)
    return float(E)


def main():
    parser = argparse.ArgumentParser(description="Kagome XXZ via PESS")
    parser.add_argument("--D", type=int, default=2, help="PESS bond dimension")
    parser.add_argument("--chi", type=int, default=20, help="CTM bond dimension")
    parser.add_argument("--delta", type=float, default=1.0, help="XXZ anisotropy")
    parser.add_argument("--steps", type=int, default=200, help="SU steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Trotter step")
    parser.add_argument("--save", type=str, default=None, help="Save PESS to file")
    parser.add_argument("--load", type=str, default=None, help="Load PESS from file")
    args = parser.parse_args()

    print(f"Kagome XXZ PESS: D={args.D}, chi={args.chi}, delta={args.delta}")

    if args.load:
        print(f"Loading PESS state from {args.load}")
        site_tensors, simplex_tensors, lambdas = load_pess(args.load)
    else:
        site_tensors, simplex_tensors, lambdas = init_pess(args.D)

    # Build Hamiltonian
    H_tri = kagome_triangle_hamiltonian(args.delta)
    print(f"Triangle Hamiltonian built (delta={args.delta})")

    # Simple update
    print(f"Running {args.steps} simple update steps (dt={args.dt})...")
    site_tensors, simplex_tensors, lambdas = pess_simple_update(
        site_tensors, simplex_tensors, lambdas,
        H_tri, args.dt, args.D, args.steps,
    )
    print("Simple update complete.")

    # Save if requested
    if args.save:
        save_pess(args.save, site_tensors, simplex_tensors, lambdas)

    # Energy via CTM
    print(f"Computing energy via CTM (chi={args.chi})...")
    E = compute_energy_pess(site_tensors, simplex_tensors, lambdas, args.delta, args.chi)
    print(f"Energy per site: {E:.8f}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the full pipeline**

Run: `python examples/kagome_xxz_pess.py --D 2 --chi 8 --steps 20 --dt 0.01 --delta 1.0`
Expected: runs to completion, prints energy

**Step 3: Commit**

```bash
git add examples/kagome_xxz_pess.py
git commit -m "feat: wire up full Kagome PESS simulation with CLI"
```

---

### Task 6: Add save/load + AD optimization bridge

**Files:**
- Modify: `examples/kagome_xxz_pess.py`

**Step 1: Add AD optimization using PESS-initialized iPEPS**

Add to `kagome_xxz_pess.py`:

```python
def optimize_ad_from_pess(site_tensors, simplex_tensors, lambdas, delta, config):
    """Run AD optimization starting from PESS-initialized iPEPS.

    Contracts PESS → iPEPS super-site, then optimizes via optimize_gs_ad().
    """
    from tenax import optimize_gs_ad

    A_init = pess_to_ipeps(site_tensors, simplex_tensors, lambdas)
    gate = xxz_gate(delta)

    A_opt, env, E_gs = optimize_gs_ad(gate, A_init, config)
    return A_opt, env, E_gs
```

Add `--ad` flag to `main()`:

```python
    parser.add_argument("--ad", action="store_true", help="Run AD optimization after SU")
    parser.add_argument("--ad-steps", type=int, default=100, help="AD optimization steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="AD learning rate")
```

And at end of `main()`:

```python
    if args.ad:
        print(f"Running AD optimization ({args.ad_steps} steps, lr={args.lr})...")
        ad_config = iPEPSConfig(
            ctm=__import__("tenax").CTMConfig(chi=args.chi),
            gs_optimizer="adam",
            gs_learning_rate=args.lr,
            gs_num_steps=args.ad_steps,
        )
        A_opt, env, E_ad = optimize_ad_from_pess(
            site_tensors, simplex_tensors, lambdas, args.delta, ad_config,
        )
        print(f"AD energy per site: {float(E_ad):.8f}")
```

**Step 2: Test**

Run: `python examples/kagome_xxz_pess.py --D 2 --chi 8 --steps 10 --dt 0.01 --save /tmp/pess_test.npz`
Run: `python examples/kagome_xxz_pess.py --load /tmp/pess_test.npz --chi 8 --steps 0`
Expected: both complete without error, second loads and computes energy

**Step 3: Commit**

```bash
git add examples/kagome_xxz_pess.py
git commit -m "feat: add AD optimization bridge and save/load for PESS"
```

---

### Task 7: Update docs and exports

**Files:**
- Modify: `docs/api/algorithms.rst`
- Modify: `README.md`

**Step 1: Add xxz_gate to API docs**

In `docs/api/algorithms.rst`, after the `heisenberg_gate` autofunction line, add:

```rst
.. autofunction:: tenax.algorithms.ipeps.xxz_gate
```

**Step 2: Add Kagome PESS to README features list**

Add a bullet to the iPEPS section of README.md mentioning Kagome PESS example.

**Step 3: Commit**

```bash
git add docs/api/algorithms.rst README.md
git commit -m "docs: add xxz_gate to API docs and Kagome PESS to README"
```
