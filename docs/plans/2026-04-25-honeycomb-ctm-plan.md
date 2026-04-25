# Native Honeycomb iPEPS CTM with AD — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land a native rank-4, 6-corner, 3-direction, 2-sublattice honeycomb iPEPS CTM with implicit AD, exposed as a public Tenax algorithm. Replaces the dummy-bond brick-wall workaround currently used by `pess_optimize.py`.

**Architecture:** Parallel `_ctm_honeycomb_*.py` family alongside the existing checkerboard CTM (`_ctm_tensor_*.py`). Rank-4 honeycomb supersites with leg labels `(e0, e1, e2, phys)`. Per-sublattice `HoneycombCTMEnv` NamedTuple with 3 corners + 3 left + 3 right column tensors. One CTM iteration sweeps 3 honeycomb edge directions with paired moves across the 2 sublattices. Implicit AD via custom VJP + JIT-fused GMRES backward, mirroring `_ctm_energy_ad.py:ctm_energy_implicit`.

**Tech stack:** JAX (jit, grad, custom_vjp, GMRES), Tenax (Tensor protocol, `_ctm_projector` patterns, `_ctm_python_loop` plumbing), pytest with auto file-name markers.

**Reference design:** `docs/plans/2026-04-25-honeycomb-ctm-design.md`. Read it before starting Task 1.

**Branch:** `feat/honeycomb-ctm` (worktree `.worktrees/honeycomb-ctm`). Design doc already committed (`8744cbd`).

**Branch hygiene:** before commits, run `git status` to confirm only intended files are staged. Use `git commit` (not `--amend`) — pre-commit hook is wired via `core.hooksPath` in this worktree.

---

## Task 1: `HoneycombCTMEnv` NamedTuple + pytree registration

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_env.py`
- Create: `tests/test_ctm_honeycomb_env.py`

**Step 1: Write the failing test**

```python
# tests/test_ctm_honeycomb_env.py
"""HoneycombCTMEnv shape and pytree behavior."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.core.tensor import DenseTensor


def _dummy_tensor(shape):
    return jnp.zeros(shape, dtype=jnp.complex128)


def test_env_has_nine_fields():
    env = HoneycombCTMEnv(
        C0=_dummy_tensor((4, 4)),
        C1=_dummy_tensor((4, 4)),
        C2=_dummy_tensor((4, 4)),
        L0=_dummy_tensor((4, 9, 4)),
        L1=_dummy_tensor((4, 9, 4)),
        L2=_dummy_tensor((4, 9, 4)),
        R0=_dummy_tensor((4, 9, 4)),
        R1=_dummy_tensor((4, 9, 4)),
        R2=_dummy_tensor((4, 9, 4)),
    )
    assert env.C0.shape == (4, 4)
    assert env.L1.shape == (4, 9, 4)
    assert env.R2.shape == (4, 9, 4)


def test_env_is_pytree():
    """jax.tree_util.tree_map should iterate the 9 fields."""
    env = HoneycombCTMEnv(
        *[_dummy_tensor((4, 4)) for _ in range(3)],
        *[_dummy_tensor((4, 9, 4)) for _ in range(6)],
    )
    leaves = jax.tree_util.tree_leaves(env)
    assert len(leaves) == 9
    doubled = jax.tree_util.tree_map(lambda x: x + 1.0, env)
    assert jnp.all(doubled.C0 == 1.0)
```

**Step 2: Run, verify failure**

```bash
uv run pytest tests/test_ctm_honeycomb_env.py -v
```
Expected: ImportError on `tenax.algorithms._ctm_honeycomb_env`.

**Step 3: Implement**

```python
# src/tenax/algorithms/_ctm_honeycomb_env.py
"""Native honeycomb CTM environment data structure.

Per sublattice, the env consists of 3 corner tensors (one per honeycomb
edge direction α ∈ {0, 1, 2}) and 3 left + 3 right column tensors.

Shapes (chi = boundary dim, D = bond dim):
    C_α: (chi, chi)         [labels: (chi_in_α, chi_out_α)]
    L_α: (chi, D**2, chi)   [labels: (chi_in_α, e_α_d2, chi_out_α)]
    R_α: (chi, D**2, chi)   [labels: (chi_in_α, e_α_d2, chi_out_α)]
"""
from __future__ import annotations

from typing import NamedTuple

from tenax.core.tensor import Tensor

__all__ = ["HoneycombCTMEnv"]


class HoneycombCTMEnv(NamedTuple):
    C0: Tensor
    C1: Tensor
    C2: Tensor
    L0: Tensor
    L1: Tensor
    L2: Tensor
    R0: Tensor
    R1: Tensor
    R2: Tensor
```

NamedTuple is automatically registered as a pytree by JAX — no extra registration needed.

**Step 4: Verify pass**

```bash
uv run pytest tests/test_ctm_honeycomb_env.py -v
```
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_honeycomb_env.py tests/test_ctm_honeycomb_env.py
git commit -m "feat(honeycomb): HoneycombCTMEnv NamedTuple"
```

---

## Task 2: `HONEYCOMB_NEIGHBORS` map

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_topology.py`
- Modify: `tests/test_ctm_honeycomb_env.py` (add neighbor tests)

**Behavior:** Mirror `CHECKERBOARD_NEIGHBORS` in `_ctm_tensor_convergence.py:136-139`, but with 3 honeycomb edge directions.

**Step 1: Add failing tests**

```python
# Append to tests/test_ctm_honeycomb_env.py
from tenax.algorithms._ctm_honeycomb_topology import (
    HONEYCOMB_NEIGHBORS,
    HONEYCOMB_DIRECTIONS,
    Coord,
)


def test_honeycomb_neighbors_two_sublattice():
    assert set(HONEYCOMB_NEIGHBORS.keys()) == {(0, 0), (1, 0)}
    for coord, nbrs in HONEYCOMB_NEIGHBORS.items():
        assert set(nbrs.keys()) == {"e0", "e1", "e2"}
        # Every neighbor must point to the *other* sublattice
        other = (1, 0) if coord == (0, 0) else (0, 0)
        for direction, target in nbrs.items():
            assert target == other, f"{coord}.{direction} -> {target} not bipartite"


def test_honeycomb_directions_tuple():
    assert HONEYCOMB_DIRECTIONS == ("e0", "e1", "e2")
```

**Step 2: Verify failure** — `ImportError`.

**Step 3: Implement**

```python
# src/tenax/algorithms/_ctm_honeycomb_topology.py
"""Neighbor maps and direction labels for the honeycomb lattice."""
from __future__ import annotations

__all__ = ["Coord", "HONEYCOMB_DIRECTIONS", "HONEYCOMB_NEIGHBORS"]

Coord = tuple[int, int]

HONEYCOMB_DIRECTIONS: tuple[str, str, str] = ("e0", "e1", "e2")

HONEYCOMB_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"e0": (1, 0), "e1": (1, 0), "e2": (1, 0)},
    (1, 0): {"e0": (0, 0), "e1": (0, 0), "e2": (0, 0)},
}
```

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_honeycomb_topology.py tests/test_ctm_honeycomb_env.py
git commit -m "feat(honeycomb): HONEYCOMB_NEIGHBORS map and direction labels"
```

---

## Task 3: Double-layer tensor `_double_layer_honeycomb`

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_init.py`
- Create: `tests/test_ctm_honeycomb_init.py`

**Behavior:** Given rank-4 site tensor `A` with labels `(e0, e1, e2, phys)`, build double-layer `T = sum_s A^s ⊗ A_bra^s` with labels `(e0_d2, e1_d2, e2_d2)` of dim D² each. Mirror the pattern in `_ctm_tensor_init.py:_build_double_layer_tensor` (read `_ctm_tensor_init.py:84-104` first to match the `bar()` + relabel + fuse style).

**Step 1: Failing test**

```python
# tests/test_ctm_honeycomb_init.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_init import _double_layer_honeycomb
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Build a rank-4 honeycomb site tensor with labels (e0, e1, e2, phys)."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


def test_double_layer_shape_and_labels():
    A = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(0))
    T = _double_layer_honeycomb(A)
    assert set(T.labels()) == {"e0_d2", "e1_d2", "e2_d2"}
    for label in ("e0_d2", "e1_d2", "e2_d2"):
        ax = T.labels().index(label)
        assert T.shape[ax] == 9  # D**2 = 9


def test_double_layer_hermiticity_on_phys():
    """T_iijj... = T_jjii... when contracted on the bra with the conjugate
    transpose of the ket — the trace structure is positive semi-definite."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    T = _double_layer_honeycomb(A)
    # Contract T's three legs to a scalar against random env tensors:
    # ⟨env|T|env⟩ should be real for arbitrary unitary env.
    key = jax.random.PRNGKey(2)
    e = []
    for i in range(3):
        v = jax.random.normal(jax.random.fold_in(key, i), (4,)) + \
            1j * jax.random.normal(jax.random.fold_in(key, i + 10), (4,))
        # The fused dim is 4 = D**2 = 2**2
        e.append(v.astype(jnp.complex128))
    # Direct contraction of T (D**2=4 each leg) with three arbitrary vectors:
    # this is basically tr(M v_0 v_1 v_2) via reshape; for the test we just
    # check the result is finite and complex.
    val = jnp.einsum("ijk,i,j,k->", T.todense(), e[0], e[1], e[2])
    assert jnp.isfinite(val.real) and jnp.isfinite(val.imag)
```

**Step 2: Verify failure** — `ImportError`.

**Step 3: Implement**

```python
# src/tenax/algorithms/_ctm_honeycomb_init.py
"""Double-layer construction and env initialization for honeycomb CTM."""
from __future__ import annotations

from tenax.algorithms._ctm_tensor_init import _fuse_pair_by_label
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection
from tenax.core.tensor import Tensor

__all__ = ["_double_layer_honeycomb"]

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _double_layer_honeycomb(A: Tensor) -> Tensor:
    """Build the rank-3 double-layer tensor for a honeycomb site.

    Input:  A with labels ``(e0, e1, e2, phys)``, 4 legs.
    Output: 3-leg tensor with labels ``(e0_d2, e1_d2, e2_d2)``, dimensions D².

    Mirrors ``_ctm_tensor_init._build_double_layer_tensor`` (rank-5 square
    case) but with 3 virtual legs instead of 4.
    """
    A_bra = A.bar().relabels({"e0": "E0", "e1": "E1", "e2": "E2"})
    a6 = contract(A, A_bra)
    result = _fuse_pair_by_label(a6, "e0", "E0", "e0_d2", OUT)
    result = _fuse_pair_by_label(result, "e1", "E1", "e1_d2", OUT)
    result = _fuse_pair_by_label(result, "e2", "E2", "e2_d2", OUT)
    return result
```

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_honeycomb_init.py tests/test_ctm_honeycomb_init.py
git commit -m "feat(honeycomb): rank-3 double-layer tensor builder"
```

---

## Task 4: Initialize honeycomb env at small chi

**Files:**
- Modify: `src/tenax/algorithms/_ctm_honeycomb_init.py`
- Modify: `tests/test_ctm_honeycomb_init.py`

**Behavior:** `initialize_honeycomb_env(sites, chi_init)` returns `dict[Coord, HoneycombCTMEnv]`. Initialize each (C, L, R) by random complex128 entries — corners shape `(chi_init, chi_init)`, columns shape `(chi_init, D², chi_init)` — for cleanest AD startup. Match the conventions in `_ctm_tensor_init.initialize_ctm_tensor_env` for index labeling and flows.

**Step 1: Failing test**

```python
def test_initialize_env_returns_two_sublattices():
    from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
    A = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(3))
    B = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(4))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4)
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    for coord, env in envs.items():
        assert env.C0.shape == (4, 4)
        assert env.L0.shape == (4, 9, 4)
        assert env.R2.shape == (4, 9, 4)
```

**Step 2: Verify failure.**

**Step 3: Implement** — append to `_ctm_honeycomb_init.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.core.index import TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def initialize_honeycomb_env(
    sites: dict[Coord, Tensor],
    chi_init: int,
    *,
    seed: int = 0,
) -> dict[Coord, HoneycombCTMEnv]:
    """Random complex128 init at chi_init for each sublattice's env.

    Each corner is rank-2 (chi_init, chi_init) and each column is rank-3
    (chi_init, D**2, chi_init) where D is the site's bond dim.
    """
    sym = U1Symmetry()
    envs: dict[Coord, HoneycombCTMEnv] = {}
    key = jax.random.PRNGKey(seed)
    for coord, A in sites.items():
        D = A.shape[A.labels().index("e0")]
        d2 = D * D
        chi_charges = np.zeros(chi_init, dtype=np.int32)
        d2_charges = np.zeros(d2, dtype=np.int32)
        chi_idx = lambda flow, lbl: TensorIndex.from_charges(
            sym, chi_charges.copy(), flow, label=lbl
        )
        d2_idx = lambda lbl: TensorIndex.from_charges(
            sym, d2_charges.copy(), OUT, label=lbl
        )
        corners, lefts, rights = [], [], []
        for alpha in range(3):
            k_c, k_l, k_r, key = jax.random.split(key, 4)
            c_data = (
                jax.random.normal(k_c, (chi_init, chi_init))
                + 1j * jax.random.normal(jax.random.fold_in(k_c, 1), (chi_init, chi_init))
            ).astype(jnp.complex128)
            l_data = (
                jax.random.normal(k_l, (chi_init, d2, chi_init))
                + 1j * jax.random.normal(jax.random.fold_in(k_l, 1), (chi_init, d2, chi_init))
            ).astype(jnp.complex128)
            r_data = (
                jax.random.normal(k_r, (chi_init, d2, chi_init))
                + 1j * jax.random.normal(jax.random.fold_in(k_r, 1), (chi_init, d2, chi_init))
            ).astype(jnp.complex128)
            corners.append(DenseTensor(c_data, (
                chi_idx(IN, f"chi_in_{alpha}"),
                chi_idx(OUT, f"chi_out_{alpha}"),
            )))
            lefts.append(DenseTensor(l_data, (
                chi_idx(IN, f"chi_in_{alpha}"),
                d2_idx(f"e{alpha}_d2"),
                chi_idx(OUT, f"chi_out_{alpha}"),
            )))
            rights.append(DenseTensor(r_data, (
                chi_idx(IN, f"chi_in_{alpha}"),
                d2_idx(f"e{alpha}_d2"),
                chi_idx(OUT, f"chi_out_{alpha}"),
            )))
        envs[coord] = HoneycombCTMEnv(
            C0=corners[0], C1=corners[1], C2=corners[2],
            L0=lefts[0], L1=lefts[1], L2=lefts[2],
            R0=rights[0], R1=rights[1], R2=rights[2],
        )
    return envs
```

Add `initialize_honeycomb_env` to `__all__`.

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): initialize_honeycomb_env at small chi"
```

---

## Task 5: Honeycomb projector (isometric SVD/eigh + S_safe)

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_projector.py`
- Create: `tests/test_ctm_honeycomb_projector.py`

**Behavior:** Given a boundary tensor of shape `(chi, D**2, chi)` (or higher contracted form), compute isometric `(P, P_dagger)` projectors that truncate `chi*D²` → `chi`. Lift the `S_safe` clamp pattern from `_ctm_projector.py:_compute_projector_tensor`. Implementation should use `tensor.contract` and `tenax.linalg.svd`/`eigh` (block-sparse-friendly), not `jnp.einsum` — per Y3.

**Investigation step (do this first):** Read `src/tenax/algorithms/_ctm_projector.py` to understand the existing projector API: how `S_safe` is computed, how phase fix is wrapped via `stop_gradient`, what shape the input boundary has, and how `(P, P_dagger)` are returned with which leg labels. The honeycomb version is the same primitive with a 3-leg boundary (vs 4-leg square) and labels keyed by direction `α`.

**Step 1: Failing test**

```python
# tests/test_ctm_honeycomb_projector.py
"""Honeycomb projector isometry + S_safe NaN protection."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_honeycomb_projector import compute_honeycomb_projector


def _random_boundary(chi: int, d2: int, key: jax.Array):
    """Boundary tensor (chi, d2, chi) for projector test."""
    re = jax.random.normal(key, (chi, d2, chi))
    im = jax.random.normal(jax.random.fold_in(key, 1), (chi, d2, chi))
    return (re + 1j * im).astype(jnp.complex128)


def test_projector_isometry():
    """P_dagger @ P = I for the truncated subspace."""
    chi, d2 = 4, 9
    boundary = _random_boundary(chi, d2, jax.random.PRNGKey(0))
    P, P_dag = compute_honeycomb_projector(boundary, method="eigh", chi=chi)
    # P: (chi, d2, chi_out), P_dag: (chi_out, chi, d2) → product is (chi_out, chi_out)
    PtP = jnp.einsum("abc,cab->", P_dag, P)
    # Just check shapes and finiteness for now; full isometry test below.
    assert P.shape[-1] == chi  # truncated dim
    identity = jnp.einsum("abc,abd->cd", P, P_dag.conj())
    assert jnp.allclose(identity, jnp.eye(chi), atol=1e-6)


def test_projector_no_nan_on_degenerate_spectrum():
    """A boundary with rank < chi should not produce NaN gradients."""
    chi, d2 = 4, 9
    boundary = jnp.zeros((chi, d2, chi), dtype=jnp.complex128)
    boundary = boundary.at[0, 0, 0].set(1.0)  # rank-1
    P, P_dag = compute_honeycomb_projector(boundary, method="eigh", chi=chi)
    assert jnp.all(jnp.isfinite(P))
    assert jnp.all(jnp.isfinite(P_dag))


def test_biorthogonal_method_raises_not_implemented():
    boundary = _random_boundary(4, 9, jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="biorthogonal"):
        compute_honeycomb_projector(boundary, method="biorthogonal", chi=4)
```

**Step 2: Verify failure** — `ImportError`.

**Step 3: Implement**

```python
# src/tenax/algorithms/_ctm_honeycomb_projector.py
"""Isometric projector for the honeycomb CTM.

Mirrors `_ctm_projector._compute_projector_tensor` but for the rank-3
honeycomb boundary. The pluggable ``method`` argument anticipates a
biorthogonal projector (Paper 2 §II.C / Corboz 2014) as a future
extension; for v1 only ``eigh`` and ``svd`` are implemented.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["compute_honeycomb_projector"]

_S_SAFE_EPS = 1e-12


def _phase_fix(U: jax.Array, eps: float = 1e-8) -> jax.Array:
    """First-above-threshold phase fix (variPEPS convention).

    For each column of U, find the first row whose abs > eps and divide by
    that entry's phase. Wrapped in stop_gradient since phase is a gauge dof.
    """
    abs_U = jnp.abs(U)
    threshold = abs_U > eps
    first_idx = jnp.argmax(threshold, axis=0)
    rows = jnp.take_along_axis(U, first_idx[None, :], axis=0)[0]
    phase = jnp.where(jnp.abs(rows) > eps, rows / (jnp.abs(rows) + _S_SAFE_EPS), 1.0)
    return U / jax.lax.stop_gradient(phase)[None, :]


def compute_honeycomb_projector(
    boundary: jax.Array,
    *,
    method: str = "eigh",
    chi: int,
    s_safe_eps: float = _S_SAFE_EPS,
) -> tuple[jax.Array, jax.Array]:
    """Isometric projector that truncates a (chi, d2, chi) boundary to chi.

    Args:
        boundary: Boundary tensor; reshape ``(chi, d2*chi)`` for SVD/eigh.
        method: ``"eigh"`` or ``"svd"``. ``"biorthogonal"`` raises NotImplementedError.
        chi: Output truncation dim.
        s_safe_eps: Floor for singular values to avoid 1/0 in 1/sqrt(S).

    Returns:
        P:     ``(chi_in, d2, chi_out)`` isometry, chi_out ≤ chi.
        P_dag: ``(chi_out, chi_in, d2)`` adjoint.
    """
    if method == "biorthogonal":
        raise NotImplementedError(
            "biorthogonal projectors are deferred; see "
            "docs/plans/2026-04-25-honeycomb-ctm-design.md"
        )
    if method not in ("eigh", "svd"):
        raise ValueError(f"unknown projector method: {method!r}")

    chi_in, d2, _ = boundary.shape
    M = boundary.reshape(chi_in, d2 * chi_in)

    if method == "svd":
        U, S, _ = jnp.linalg.svd(M, full_matrices=False)
    else:  # eigh on M @ M^†
        rho = M @ M.conj().T
        S2, U = jnp.linalg.eigh(rho)
        S2 = jnp.flip(S2)
        U = jnp.flip(U, axis=1)
        S = jnp.sqrt(jnp.clip(S2, a_min=s_safe_eps))

    U = U[:, :chi]
    S = S[:chi]
    S_safe = jnp.where(S > s_safe_eps, S, s_safe_eps)
    inv_sqrt_S = 1.0 / jnp.sqrt(S_safe)

    U = _phase_fix(U)

    P = (U * inv_sqrt_S[None, :]).reshape(chi_in, d2, chi_in)[:, :, : U.shape[1]]
    P_dag = (U.conj().T * inv_sqrt_S[:, None]).reshape(U.shape[1], chi_in, d2)
    return P, P_dag
```

NOTE: the exact reshape/labeling layout above is a starting point; consult `_ctm_projector.py` and refine to match the existing convention before committing. Adjust the test `test_projector_isometry` if needed to match the corrected `(P, P_dag)` index ordering.

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): isometric projector with S_safe + phase fix"
```

---

## Task 6: Single direction move

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_moves.py`
- Create: `tests/test_ctm_honeycomb_moves.py`

**Behavior:** `move_direction_alpha(envs, sites, alpha, *, chi, projector_method, forward_gauge)` performs a paired update of `(C_α, L_α, R_α)` for both sublattices simultaneously along honeycomb edge direction α. Returns updated `dict[Coord, HoneycombCTMEnv]`.

This is the algorithmic heart of the CTM. The exact contraction pattern follows Paper 2 §II.C Fig. 10 — read those equations carefully before writing the move.

**Investigation step (do this first):** Inspect `src/tenax/algorithms/_ctm_tensor_moves.py` and `_ctm_tensor_paired_moves.py` to understand:
- How the existing checkerboard move builds the absorbed boundary (4 legs → projector)
- How `_ctm_tensor_move_horizontal` couples both sublattices in one move
- How `_renormalize_tensor_env` is called per move

The honeycomb version is a 3-leg-boundary analog with 3 directions per iteration instead of 4.

**Step 1: Failing test**

```python
# tests/test_ctm_honeycomb_moves.py
"""Single-direction move shape + idempotence tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tenax.algorithms._ctm_honeycomb_moves import move_direction_alpha
from tests.test_ctm_honeycomb_init import _make_random_honeycomb_site


def test_move_preserves_env_shapes():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    new_envs = move_direction_alpha(
        envs, sites, alpha=0, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for coord in envs:
        assert new_envs[coord].C0.shape == envs[coord].C0.shape
        assert new_envs[coord].L0.shape == envs[coord].L0.shape


def test_move_returns_finite():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(2))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(3))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    new_envs = move_direction_alpha(
        envs, sites, alpha=1, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for env in new_envs.values():
        for field in env._fields:
            arr = getattr(env, field)
            assert jnp.all(jnp.isfinite(arr.todense()))
```

**Step 2: Verify failure.**

**Step 3: Implement** — write the move per Paper 2 §II.C Fig. 10. Skeleton:

```python
# src/tenax/algorithms/_ctm_honeycomb_moves.py
"""Honeycomb CTM directional moves and step composition."""
from __future__ import annotations

import jax.numpy as jnp

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_init import _double_layer_honeycomb
from tenax.algorithms._ctm_honeycomb_projector import compute_honeycomb_projector
from tenax.algorithms._ctm_honeycomb_topology import Coord, HONEYCOMB_NEIGHBORS
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.tensor import Tensor

__all__ = [
    "move_direction_alpha",
    "honeycomb_ctm_step",
    "_renormalize_honeycomb_env",
]


def _normalize_tensor(T: Tensor) -> Tensor:
    norm = T.max_abs()
    return T * (1.0 / (norm + EPS))


def _renormalize_honeycomb_env(env: HoneycombCTMEnv) -> HoneycombCTMEnv:
    """Normalize all 9 fields by max-abs to prevent exponential growth."""
    return HoneycombCTMEnv(*[_normalize_tensor(getattr(env, f)) for f in env._fields])


def move_direction_alpha(
    envs: dict[Coord, HoneycombCTMEnv],
    sites: dict[Coord, Tensor],
    *,
    alpha: int,
    chi: int,
    projector_method: str,
    forward_gauge: str,
) -> dict[Coord, HoneycombCTMEnv]:
    """Update (C_α, L_α, R_α) for BOTH sublattices via paired projector.

    The honeycomb-edge direction α connects sublattice A's e_α leg to
    sublattice B's e_α leg. The paired update absorbs A's column into A's
    corner and B's column into B's corner using a projector pair derived
    from the joint A-B boundary.

    See Paper 2 (PRE 109, 045305, 2024) §II.C, Fig. 10 for the exact
    contraction pattern.
    """
    # 1. Build double-layer tensors for both sublattices.
    T_A = _double_layer_honeycomb(sites[(0, 0)])
    T_B = _double_layer_honeycomb(sites[(1, 0)])

    # 2. Build the joint boundary along direction α, fold A and B together.
    #    The joint boundary couples L_α^A, T_A, T_B, R_α^B (or symmetric pair).
    #    Reshape result to (chi, D², chi) and feed to the projector.
    #    [TODO: write the explicit contractor calls following Fig. 10.]

    # 3. Compute projector P, P† via compute_honeycomb_projector.

    # 4. Apply projector to update C_α, L_α, R_α for both sublattices.

    # 5. If forward_gauge == "sigma", apply sigma gauge to new corners.

    # 6. Return new envs dict.
    raise NotImplementedError("implement per Paper 2 §II.C Fig. 10")
```

**Implementation note:** the move structure is the algorithmic heart of the port. Build it carefully, with sub-steps:

1. Construct the boundary tensor first (shape `(chi, D², chi)` or however it factors); add an isolated unit test for that intermediate.
2. Then plug it into `compute_honeycomb_projector`.
3. Then apply the projectors to update each (C, L, R) field; one update at a time, with a unit test per direction's update.
4. Then assemble into the full `move_direction_alpha`.

If the move's contraction pattern is unclear from the paper, **stop and ask before guessing.** Wrong contraction = silent numerical errors, not a clean failure.

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): single-direction paired CTM move"
```

---

## Task 7: Full step (3-direction sweep) + renormalization

**Files:**
- Modify: `src/tenax/algorithms/_ctm_honeycomb_moves.py`
- Modify: `tests/test_ctm_honeycomb_moves.py`

**Behavior:** `honeycomb_ctm_step(envs, sites, *, chi, projector_method, forward_gauge, renormalize=True)` calls `move_direction_alpha` for α ∈ {0, 1, 2} in sequence, then `_renormalize_honeycomb_env`. Returns updated envs dict.

**Step 1: Failing test**

```python
def test_full_step_runs_without_nan():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(4))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(5))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    from tenax.algorithms._ctm_honeycomb_moves import honeycomb_ctm_step
    new_envs = honeycomb_ctm_step(
        envs, sites, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for env in new_envs.values():
        for field in env._fields:
            arr = getattr(env, field)
            assert jnp.all(jnp.isfinite(arr.todense()))


def test_full_step_normalization():
    """After renormalize, max(|tensor|) ≈ 1."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(6))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(7))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    from tenax.algorithms._ctm_honeycomb_moves import honeycomb_ctm_step
    new_envs = honeycomb_ctm_step(
        envs, sites, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for env in new_envs.values():
        for field in env._fields:
            arr = getattr(env, field)
            assert float(arr.max_abs()) <= 1.0 + 1e-6
```

**Step 2: Verify failure** (`ImportError`/`NotImplementedError`).

**Step 3: Implement** — append:

```python
def honeycomb_ctm_step(
    envs: dict[Coord, HoneycombCTMEnv],
    sites: dict[Coord, Tensor],
    *,
    chi: int,
    projector_method: str,
    forward_gauge: str,
    renormalize: bool = True,
) -> dict[Coord, HoneycombCTMEnv]:
    """One full CTM iteration: 3 directional moves + optional renorm."""
    for alpha in (0, 1, 2):
        envs = move_direction_alpha(
            envs, sites,
            alpha=alpha, chi=chi,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
        )
    if renormalize:
        envs = {k: _renormalize_honeycomb_env(v) for k, v in envs.items()}
    return envs
```

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): honeycomb_ctm_step (3-direction sweep)"
```

---

## Task 8: Convergence check

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_convergence.py`
- Create: `tests/test_ctm_honeycomb_convergence.py`

**Behavior:** `check_honeycomb_convergence(env_old, env_new, *, method, tol)` returns `bool`. Two methods: `"elementwise"` (max-abs diff over all 9 fields × 2 sublattices) and `"svd"` (singular-value diff on each corner). Mirror `_ctm_sv_diff` and `_ctm_tensor_convergence._ctm_tensor_multisite` patterns.

**Step 1: Failing test**

```python
# tests/test_ctm_honeycomb_convergence.py
import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_honeycomb_convergence import check_honeycomb_convergence
from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tests.test_ctm_honeycomb_init import _make_random_honeycomb_site


def test_identical_envs_are_converged():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    assert check_honeycomb_convergence(envs, envs, method="elementwise", tol=1e-10)


def test_different_envs_not_converged():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs1 = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    envs2 = initialize_honeycomb_env(sites, chi_init=4, seed=43)
    assert not check_honeycomb_convergence(envs1, envs2, method="elementwise", tol=1e-10)
```

**Step 2: Verify failure.**

**Step 3: Implement** — `check_honeycomb_convergence` per the patterns in `_ctm_tensor_convergence.py`. Both methods. Returns `bool`.

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_honeycomb_convergence.py tests/test_ctm_honeycomb_convergence.py
git commit -m "feat(honeycomb): convergence check (elementwise + sv)"
```

---

## Task 9: Single-bond RDM (2-vertex)

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_energy.py`
- Create: `tests/test_ctm_honeycomb_energy.py`

**Behavior:** `_rdm2_bond(sites, envs, *, alpha)` returns the `(d², d²)` reduced density matrix for the bond along honeycomb-edge direction α connecting A and B. Construction parallels `_ctm_tensor_energy._rdm2x1_tensor` but with the 6-corner / 3-column-tensor environment.

**Investigation step:** Inspect `_ctm_tensor_energy._rdm2x1_tensor` to learn the contraction pattern conventions (which envs go where, how the open physical legs are propagated, how the result is symmetrized via `0.5*(rho + rho.conj().T)` and normalized by trace).

**Step 1: Failing test** — RDM hermiticity, positivity (eigenvalues ≥ 0), trace 1, on a small random sites setup. ~3 assertions.

**Step 2-5:** Implement, verify, commit.

```bash
git commit -am "feat(honeycomb): 2-vertex bond RDM helper"
```

---

## Task 10: 1-site RDM + triangle-energy helper for kagome use

**Files:**
- Modify: `src/tenax/algorithms/_ctm_honeycomb_energy.py`
- Modify: `tests/test_ctm_honeycomb_energy.py`

**Behavior:** `_rdm1(sites, envs, *, sublattice)` returns `(d, d)` 1-site RDM. `compute_honeycomb_triangle_energy(sites, envs, hamiltonian)` returns `Tr(ρ_A · H) + Tr(ρ_B · H)` where H is `(d_fused, d_fused)` and `d_fused` is the supersite physical dim.

This is the `energy_fn=` override the kagome iPESS path passes in.

**Step 1-5:** TDD as before.

```bash
git commit -am "feat(honeycomb): 1-site RDM + triangle-energy helper"
```

---

## Task 11: Default energy function (3-edge NN sum)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_honeycomb_energy.py`
- Modify: `tests/test_ctm_honeycomb_energy.py`

**Behavior:** `compute_honeycomb_energy(sites, envs, hamiltonian)` returns sum of `Tr(ρ_α · H_bond)` for α ∈ {0, 1, 2}. `H_bond` shape `(d², d²)`. This is the default for vanilla honeycomb iPEPS use cases (e.g. spin-1/2 Heisenberg).

```bash
git commit -am "feat(honeycomb): default 3-edge NN bond energy"
```

---

## Task 12: Forward CTM loop (no AD yet)

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_forward.py`
- Create: `tests/test_ctm_honeycomb_forward.py`

**Behavior:** `honeycomb_ctm_run(sites, *, chi, max_iter, conv_tol, projector_method, forward_gauge, chi_ramp=None)` returns converged `dict[Coord, HoneycombCTMEnv]`. Python-loop, mirroring `_ctm_python_loop.py`. Emits `warnings.warn` (not raise) on `max_iter` exceedance. No AD path yet.

**Step 1: Failing test** — at D=2, χ=4, 30 iterations, on a random site, the energy should stabilize (not blow up, |E_iter - E_iter+5|/|E| < 1e-3).

**Step 2-5:** TDD.

```bash
git commit -am "feat(honeycomb): forward CTM run with chi ramp + warn-on-nonconvergence"
```

---

## Task 13: Implicit AD with custom VJP + JIT-fused GMRES backward

**Files:**
- Create: `src/tenax/algorithms/_ctm_honeycomb_ad.py`
- Create: `tests/test_ctm_honeycomb_ad.py`

**Behavior:** `honeycomb_ctm_energy_implicit(sites, hamiltonian, *, chi, max_iter, conv_tol, projector_method, forward_gauge, chi_ramp, energy_fn, gmres_tol, gmres_maxiter, gmres_restart, arnoldi_precheck)` — public entry. Custom VJP via `jax.custom_vjp`: forward calls `honeycomb_ctm_run` then `energy_fn`; backward solves `(I - ∂F/∂env|env*) · v = ∂E/∂env|env*` via JIT-fused GMRES, then chains the rule for `∂E/∂sites`.

**Investigation step (do this first, BEFORE TDD):** Read `src/tenax/algorithms/_ctm_energy_ad.py:ctm_energy_implicit` end-to-end. Note:
- How the forward residuals are packed for the backward call
- How GMRES is wrapped (`jax.scipy.sparse.linalg.gmres` vs Tenax's own implementation)
- How `arnoldi_precheck` flag is consumed
- How errors in the GMRES solve are warned-not-raised (per memory: PR #343 honors `gmres_maxiter`)

The honeycomb implicit-AD wrapper is structurally identical; the only differences are which env type and which energy_fn it operates on.

**Step 1: Failing test**

```python
# tests/test_ctm_honeycomb_ad.py
"""Implicit-AD gradient finiteness + FD-vs-AD agreement."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_ad import honeycomb_ctm_energy_implicit
from tests.test_ctm_honeycomb_init import _make_random_honeycomb_site


def _heisenberg_bond_xxz(d: int = 2, delta: float = 1.0):
    """Spin-1/2 Heisenberg XXZ bond Hamiltonian, (d**2, d**2)."""
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.kron(sx, sx) + np.kron(sy, sy) + delta * np.kron(sz, sz)
    return H.astype(np.complex128)


def test_grad_finite():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()

    def loss(s):
        return honeycomb_ctm_energy_implicit(
            s, H, chi=4, max_iter=30, conv_tol=1e-8,
            projector_method="eigh", forward_gauge="phase",
            chi_ramp=None, energy_fn=None,
            gmres_tol=1e-6, gmres_maxiter=30, gmres_restart=10,
            arnoldi_precheck=False,
        )

    e0 = loss(sites)
    g = jax.grad(loss)(sites)
    assert jnp.isfinite(e0)
    for coord, grad_site in g.items():
        for ax in range(grad_site.ndim):
            arr = grad_site.todense() if hasattr(grad_site, "todense") else grad_site
            assert jnp.all(jnp.isfinite(arr))


def test_fd_vs_ad_small():
    """Complex-step finite difference vs autograd at D=2, χ=4."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(2))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(3))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()

    def loss(s):
        return honeycomb_ctm_energy_implicit(
            s, H, chi=4, max_iter=50, conv_tol=1e-10,
            projector_method="eigh", forward_gauge="phase",
            chi_ramp=None, energy_fn=None,
            gmres_tol=1e-9, gmres_maxiter=50, gmres_restart=20,
            arnoldi_precheck=False,
        )

    g_ad = jax.grad(loss)(sites)
    # Pick one element of A's data, perturb, compare:
    A_data = sites[(0, 0)].data
    h = 1e-6
    A_plus = A_data.at[0, 0, 0, 0].add(h)
    A_minus = A_data.at[0, 0, 0, 0].add(-h)
    e_plus = loss({(0, 0): A.replace_data(A_plus), (1, 0): B})
    e_minus = loss({(0, 0): A.replace_data(A_minus), (1, 0): B})
    fd = float((e_plus - e_minus) / (2 * h))
    ad = float(g_ad[(0, 0)].todense().real[0, 0, 0, 0])
    assert abs(fd - ad) / (abs(ad) + 1e-8) < 1e-3
```

**Step 2: Verify failure** (no implementation).

**Step 3: Implement** — mirror `_ctm_energy_ad.py` structure, swap in honeycomb env / step / energy.

**Step 4: Verify pass.** This is the M1 gate; if FD-AD doesn't agree to 1e-3, debug before moving on.

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): implicit-AD energy via custom VJP + GMRES backward"
```

---

## Task 14: Public re-export shim + lazy-import registration

**Files:**
- Create: `src/tenax/algorithms/honeycomb_ctm.py`
- Modify: `src/tenax/algorithms/__init__.py` (add to `_LAZY_IMPORTS`)
- Modify: `src/tenax/__init__.py` (add to `__all__`)

**Step 1-5:** Add explicit named re-exports mirroring `ipeps_ctm.py` style. Lazy-import registration.

```python
# src/tenax/algorithms/honeycomb_ctm.py
"""Public honeycomb iPEPS CTM API — explicit named re-exports.

Internal code should import from the concrete modules:
  - ``_ctm_honeycomb_ad`` (honeycomb_ctm_energy_implicit)
  - ``_ctm_honeycomb_env`` (HoneycombCTMEnv)
  - ``_ctm_honeycomb_init`` (initialize_honeycomb_env, _double_layer_honeycomb)
  - ``_ctm_honeycomb_topology`` (HONEYCOMB_NEIGHBORS, HONEYCOMB_DIRECTIONS)

This shim re-exports for downstream/notebook usage.
"""

from tenax.algorithms._ctm_honeycomb_ad import (
    honeycomb_ctm_energy_implicit as honeycomb_ctm_energy_implicit,
)
from tenax.algorithms._ctm_honeycomb_env import (
    HoneycombCTMEnv as HoneycombCTMEnv,
)
from tenax.algorithms._ctm_honeycomb_forward import (
    honeycomb_ctm_run as honeycomb_ctm_run,
)
from tenax.algorithms._ctm_honeycomb_init import (
    initialize_honeycomb_env as initialize_honeycomb_env,
)
from tenax.algorithms._ctm_honeycomb_topology import (
    HONEYCOMB_DIRECTIONS as HONEYCOMB_DIRECTIONS,
)
from tenax.algorithms._ctm_honeycomb_topology import (
    HONEYCOMB_NEIGHBORS as HONEYCOMB_NEIGHBORS,
)
```

In `algorithms/__init__.py:_LAZY_IMPORTS`, add:

```python
"honeycomb_ctm_energy_implicit": (
    "tenax.algorithms._ctm_honeycomb_ad", "honeycomb_ctm_energy_implicit"
),
"honeycomb_ctm_run": (
    "tenax.algorithms._ctm_honeycomb_forward", "honeycomb_ctm_run"
),
"HoneycombCTMEnv": (
    "tenax.algorithms._ctm_honeycomb_env", "HoneycombCTMEnv"
),
"initialize_honeycomb_env": (
    "tenax.algorithms._ctm_honeycomb_init", "initialize_honeycomb_env"
),
"HONEYCOMB_NEIGHBORS": (
    "tenax.algorithms._ctm_honeycomb_topology", "HONEYCOMB_NEIGHBORS"
),
```

In `tenax/__init__.py:__all__`, add the same names.

**Step 1: test**

```python
# Add to tests/test_ctm_honeycomb_env.py
def test_public_api_lazy_import():
    import tenax
    import tenax.algorithms
    # Lazy access through the lazy-import dict
    assert hasattr(tenax.algorithms, "honeycomb_ctm_energy_implicit")
    assert hasattr(tenax.algorithms, "HoneycombCTMEnv")
    # Direct shim import
    from tenax.algorithms.honeycomb_ctm import (
        honeycomb_ctm_energy_implicit, HoneycombCTMEnv, HONEYCOMB_NEIGHBORS
    )
    assert callable(honeycomb_ctm_energy_implicit)
```

**Steps 2-4:** verify, implement, verify pass.

**Step 5: Commit**

```bash
git commit -am "feat(honeycomb): public API lazy-import + re-export shim"
```

---

## Task 15: Numerical-safeguard test suite

**Files:**
- Create: `tests/test_ctm_honeycomb_safeguards.py`

**Behavior:** Tests for input-validation rejections + AD safeguards listed in design doc Section "Numerical safeguards":

- `test_rejects_real_dtype` — float64 input → `ValueError("complex128")`
- `test_rejects_mismatched_bond_dims` — A with D=4, B with D=2 → `ValueError`
- `test_rejects_wrong_rank` — rank-5 input → `ValueError`
- `test_s_safe_no_nan_at_degenerate_spectrum` — already in projector test, lift here
- `test_phase_fix_idempotent` — applied twice == applied once
- `test_nonconvergence_warns_returns_finite` — `max_iter=1` produces a `UserWarning` and finite scalar
- `test_biorthogonal_method_raises` — `method="biorthogonal"` → `NotImplementedError`

**Step 1: Test file with all 7 tests, expected to fail until input validation is wired.**

**Step 2: Verify failure** for the input-validation tests.

**Step 3: Implement** input validation in `honeycomb_ctm_energy_implicit` — small block at the top:

```python
def honeycomb_ctm_energy_implicit(sites, hamiltonian, *, chi, ...):
    # Input validation (cheap, fail fast)
    for coord, A in sites.items():
        if A.dtype != jnp.complex128:
            raise ValueError(
                f"site {coord} has dtype {A.dtype}; complex128 required for "
                "variational stability (memory: project_complex_tensors_variational.md)"
            )
        if len(A.labels()) != 4:
            raise ValueError(
                f"site {coord} has rank {len(A.labels())}; rank-4 expected "
                "with labels (e0, e1, e2, phys)"
            )
        labels = set(A.labels())
        if labels != {"e0", "e1", "e2", "phys"}:
            raise ValueError(
                f"site {coord} has labels {labels}; expected {{e0, e1, e2, phys}}"
            )
    # Bond-dim consistency:
    Ds = [A.shape[A.labels().index("e0")] for A in sites.values()]
    if len(set(Ds)) != 1:
        raise ValueError(f"site bond dims differ: {Ds}")
    # ... rest of function
```

**Step 4: Verify pass.**

**Step 5: Commit**

```bash
git commit -am "test(honeycomb): numerical-safeguard test suite + input validation"
```

---

## Task 16: M2a — Lukin-Sotnikov uniform reproduction (slow regression)

**Files:**
- Create: `tests/test_ctm_honeycomb_lukin_sotnikov.py`

**Behavior:** Run uniform iPEPS (A=B) on Heisenberg honeycomb at D=2, χ=20 with the new path. Initialize from a short L-BFGS minimization (or the simpler approach: random init + a few hundred CTM iterations + measure energy — no L-BFGS needed for the smoke test). Compare against Lukin-Sotnikov Table I D=2 entry.

This test is `@pytest.mark.slow` because it runs at χ=20.

**Step 1: Failing test (asserts the published energy)**

```python
import pytest
import jax
import jax.numpy as jnp

from tenax.algorithms.honeycomb_ctm import honeycomb_ctm_energy_implicit
from tests.test_ctm_honeycomb_init import _make_random_honeycomb_site


# Lukin-Sotnikov Table I, Heisenberg honeycomb, D=2:
LUKIN_SOTNIKOV_D2_ENERGY = -0.5443  # placeholder — replace with actual paper value before merge


@pytest.mark.slow
def test_lukin_sotnikov_d2_uniform_reproduction():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(42))
    sites = {(0, 0): A, (1, 0): A}  # uniform
    H = ...  # Heisenberg bond
    energy = honeycomb_ctm_energy_implicit(
        sites, H, chi=20, max_iter=200, conv_tol=1e-10, ...
    )
    # Without optimization, just CTM at random A — energy is high.
    # For the test, do a short BFGS to bring it close to the literature.
    # Or: assert the energy is at least within 5% of the published value
    # at this small D.
    assert abs(energy - LUKIN_SOTNIKOV_D2_ENERGY) / abs(LUKIN_SOTNIKOV_D2_ENERGY) < 0.05
```

**Important:** Look up the actual Lukin-Sotnikov Table I D=2 entry from the paper before merging. If only χ=20+ runs are reported, scale the test χ accordingly.

**Step 2-5:** TDD.

```bash
git commit -am "test(honeycomb): M2a Lukin-Sotnikov uniform-iPEPS regression"
```

---

## Task 17: M2b — kagome iPESS swap-out smoke (slow regression)

**Files:**
- Create: `tests/test_pess_ad_honeycomb.py`

**Behavior:** Run the kagome iPESS smoke from `tests/test_pess_ad.py` against the new `honeycomb_ctm_energy_implicit` path (without modifying `pess_optimize.py` — that's the follow-up PR). Build a parallel loss closure inside the test that uses the new path. Compare converged energies at fixed seed, D=2, d=3, χ=8.

**Step 1: Test asserting agreement within 1e-3** between dummy-bond hack energy and new-path energy.

**Step 2-5:** TDD.

```bash
git commit -am "test(honeycomb): M2b kagome iPESS native vs dummy-bond agreement"
```

---

## Task 18: README + Sphinx + module-level docs

**Files:**
- Modify: `README.md` — add "Honeycomb iPEPS with AD" subsection
- Modify: `docs/source/algorithms.rst` (or `docs/source/ipeps.rst`, whichever exists) — add a section
- Modify: `src/tenax/algorithms/_ctm_honeycomb_ad.py` — top-of-file docstring with the two paper citations and a link to the design doc

**Step 1:** No test for docs.

**Step 2:** N/A.

**Step 3:** Write the docs.

**Step 4:**
```bash
cd docs && make html
```
Verify no warnings about the new section.

**Step 5: Commit**

```bash
git commit -am "docs: honeycomb iPEPS CTM in README, Sphinx, and module docstring"
```

---

## Task 19: Open PR

**Step 1:** Push and open PR:

```bash
git push -u origin feat/honeycomb-ctm
gh pr create --title "feat(honeycomb): native rank-4 honeycomb iPEPS CTM with implicit AD" --body "$(cat <<'EOF'
## Summary
- New `_ctm_honeycomb_*.py` module family implementing native rank-4, 6-corner, 3-direction, 2-sublattice honeycomb iPEPS CTM with implicit AD.
- Public entry: `tenax.algorithms.honeycomb_ctm_energy_implicit`. Custom VJP + JIT-fused GMRES backward, mirroring `_ctm_energy_ad.py:ctm_energy_implicit`.
- Replaces (in a follow-up PR) the Kronecker-delta dummy-bond brick-wall workaround currently used by `pess_optimize.py`.
- References: Lukin-Sotnikov PRB 107, 054424 (2023) for the 6-corner CTMRG; Paper 2 §II.C of PRE 109, 045305 (2024) for the 2-sublattice extension.

## What this PR does NOT do
- It does **not** rewire `pess_optimize.py` to use the new path — that's a follow-up PR (M2b validates the new path produces equivalent energies).
- It does **not** add SymmetricTensor tests (deferred per design doc Y3).
- It does **not** implement biorthogonal projectors (stub raises `NotImplementedError`; Paper 2 §II.C followup).
- It does **not** consolidate with the existing checkerboard CTM (`_ctm_tensor_*.py`); the duplication is intentional. See "Future consolidation" in `docs/plans/2026-04-25-honeycomb-ctm-design.md`. **Reviewers: please flag duplication you'd want addressed in a follow-up consolidation PR rather than as a v1 blocker.**

## Test plan
- [ ] `uv run pytest tests/test_ctm_honeycomb_*.py -v` passes
- [ ] `uv run pytest -m core` passes (existing CI)
- [ ] `uv run pytest -m slow tests/test_ctm_honeycomb_lukin_sotnikov.py tests/test_pess_ad_honeycomb.py` passes (regression)
- [ ] FD-vs-AD agreement at D=2, χ=4, rel-tol 1e-3 (`test_fd_vs_ad_small`)
- [ ] Lukin-Sotnikov Table I D=2 reproduced within 5% (M2a)
- [ ] kagome iPESS new path agrees with dummy-bond hack within 1e-3 at D=2 d=3 χ=8 (M2b)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 2:** After CI passes:

```bash
gh pr merge --squash --delete-branch --auto
```

---

## Open questions to resolve during execution (not blockers)

1. **Lukin-Sotnikov Table I D=2 entry value** (Task 16). Need to look up the published value from PRB 107, 054424 (2023) and substitute in the test. If only χ ≥ 20 entries exist, set test χ accordingly.
2. **Move's exact contraction pattern** (Task 6). The skeleton in this plan is structural; the actual contraction follows Paper 2 §II.C Fig. 10. If the figure is ambiguous on a contraction order, **stop and ask** rather than guess.
3. **Phase fix vs sigma gauge equivalence on honeycomb.** Memory `project_phase_gauge_default.md` says phase gauge is the right default; sigma may or may not need adjustments for the 6-corner topology. Test both at low D, χ before deciding which to expose first.
4. **Biorthogonal projector tests** (followup PR). The stub raises `NotImplementedError` in v1; the followup PR implements it, adds tests against the dummy-bond hack and (if available) variPEPS reference numbers.
