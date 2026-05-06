# Multisite-CTM RDM Brute-Force Diagnostic — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the brute-force-vs-CTM RDM audit probe + structural-invariants pytest gates that localise which helper in the multisite-CTM consumer chain produces unphysical infinite-lattice RDMs at D=4 χ=16, blocking Phase C.3 of the parent plan.

**Architecture:** Two new files — an `examples/` audit probe (no asserts, JSON output) and a `tests/test_pess_3site_multisite_rdm_invariants.py` strict-gate file. Brute-force reference is exact contraction of the 3-site multisite tensors on a 3×3 PBC kagome-stripe torus (sublattice = `(x+y) mod 3 ∈ {u,v,w}`, 9 sites total, `d^9 = 512`-dim wavefunction at d=2). RDMs from the wavefunction are compared to the multisite-CTM RDMs that `compute_energy_pess_3site_multisite` consumes. **No `src/` changes.**

**Tech Stack:** JAX (`jnp.einsum` with `opt_einsum` ordering), pytest, existing Tenax algorithms (`pess_to_kagome_3site_multisite`, `compute_energy_pess_3site_multisite`, `ctm_energy_implicit`, `_rdm{2x1,1x2}_tensor_2site`, `_rdm_3site_marginal_vw_{row,col}`).

**Pre-flight check:** Ensure pre-commit hooks are installed (`pre-commit --version` should print 4.x). Pre-commit runs `ruff format` + `ruff check` on each commit.

**Sub-skills referenced:**
- @superpowers:test-driven-development for each task (failing test first → minimal impl → green → commit)
- @superpowers:verification-before-completion before claiming each task done
- @superpowers:executing-plans for the overall task-by-task execution flow

---

## Lattice constants (used by every task)

```python
# 3×3 PBC torus position → sublattice. Position index = 3*y + x (row-major).
# Sublattice = (x + y) mod 3 ↦ {0:"u", 1:"v", 2:"w"}.
_POS_TO_NAME: tuple[str, ...] = (
    "u", "v", "w",  # y=0: (0,0), (1,0), (2,0)
    "v", "w", "u",  # y=1: (0,1), (1,1), (2,1)
    "w", "u", "v",  # y=2: (0,2), (1,2), (2,2)
)

# 9 horizontal bonds (right of pos -> next x): position index pairs
_H_BONDS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 0),  # y=0 row, last wraps PBC
    (3, 4), (4, 5), (5, 3),  # y=1 row
    (6, 7), (7, 8), (8, 6),  # y=2 row
)
# 9 vertical bonds (bottom of pos -> next y): position index pairs
_V_BONDS: tuple[tuple[int, int], ...] = (
    (0, 3), (1, 4), (2, 5),  # x=0,1,2 columns y=0->1
    (3, 6), (4, 7), (5, 8),  # y=1->2
    (6, 0), (7, 1), (8, 2),  # y=2->0 wrap
)
```

The 6 named RDMs the energy formula consumes, with their brute-force position-index representatives (matching the energy formula's `nn_visits` order — see `_pess_multisite_energy.py:347-358`):

| Bond name | Visit | Brute-force positions | RDM ordering |
|---|---|---|---|
| `uv_h`  | `(u, right)` → v | `(0, 1)` (= u^(0,0), v^(1,0)) | `(s_u, s_v, s_u', s_v')` |
| `uv_v`  | `(u, bottom)` → v | `(0, 3)` (= u^(0,0), v^(0,1)) | `(s_u, s_v, s_u', s_v')` |
| `wu_h`  | `(w, right)` → u | `(2, 0)` (= w^(2,0), u^(0,0)) | `(s_w, s_u, s_w', s_u')` |
| `wu_v`  | `(w, bottom)` → u | `(2, 5)` (= w^(2,0), u^(2,1)) | `(s_w, s_u, s_w', s_u')` |
| `vw_row` | 3-site marginal, row | `(0, 1, 2)` → trace pos 0 → keep `(1, 2)` | `(s_v, s_w, s_v', s_w')` |
| `vw_col` | 3-site marginal, col | `(0, 3, 6)` → trace pos 0 → keep `(3, 6)` | `(s_v, s_w, s_v', s_w')` |

---

### Task 1: Multisite 3×3 PBC torus contraction

**Files:**
- Create: `tests/test_pess_3site_multisite_rdm_invariants.py`

**Step 1: Write the failing test** (translation-invariance under (1,−1) diagonal shift — leaves sublattice assignment invariant; strong correctness check that catches any bond-mislabelling).

```python
"""Brute-force vs multisite-CTM RDM diagnostic + structural-invariants gates.

See docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md for design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import IPESSState, pess_to_kagome_3site_multisite

# 3×3 PBC torus position → sublattice. See the plan's "Lattice constants".
_POS_TO_NAME: tuple[str, ...] = (
    "u", "v", "w",
    "v", "w", "u",
    "w", "u", "v",
)


def _contract_multisite_3x3_torus(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Exact contraction of the 3-site multisite encoding on a 3×3 PBC torus.

    Sublattice = (x + y) mod 3 ↦ {0:u, 1:v, 2:w}. 9 sites at row-major
    positions 0..8. Each site has 4 virtual legs (top, bot, lft, rgt) and
    1 physical leg. Output: rank-9 array indexed by physical legs in
    row-major position order.

    Bond labels (1 letter per bond):
      Horizontal (right→left): a..i for the 9 H_BONDS pairs.
      Vertical   (bot→top):    j..r for the 9 V_BONDS pairs.
      Physical:                A..I (for positions 0..8).

    See the plan for derivation.
    """
    # Per-site einsum strings: each is "top, bot, lft, rgt, phys".
    strings = (
        "pjcaA",  # pos 0 (u): top=p (V6 wrap), bot=j (V0), lft=c (H2 wrap), rgt=a (H0)
        "qkabB",  # pos 1 (v): top=q (V7 wrap), bot=k (V1), lft=a (H0),     rgt=b (H1)
        "rlbcC",  # pos 2 (w): top=r (V8 wrap), bot=l (V2), lft=b (H1),     rgt=c (H2)
        "jmfdD",  # pos 3 (v): top=j (V0),      bot=m (V3), lft=f (H5 wrap),rgt=d (H3)
        "kndeE",  # pos 4 (w): top=k (V1),      bot=n (V4), lft=d (H3),     rgt=e (H4)
        "loefF",  # pos 5 (u): top=l (V2),      bot=o (V5), lft=e (H4),     rgt=f (H5)
        "mpigG",  # pos 6 (w): top=m (V3),      bot=p (V6), lft=i (H8 wrap),rgt=g (H6)
        "nqghH",  # pos 7 (u): top=n (V4),      bot=q (V7), lft=g (H6),     rgt=h (H7)
        "orhiI",  # pos 8 (v): top=o (V5),      bot=r (V8), lft=h (H7),     rgt=i (H8)
    )
    spec = ",".join(strings) + "->ABCDEFGHI"
    args = [sites[_POS_TO_NAME[p]] for p in range(9)]
    return jnp.einsum(spec, *args, optimize="optimal")


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_multisite_3x3_torus_translation_invariant_diagonal(D):
    """ψ on the 3×3 PBC torus must be invariant under the (1,-1) diagonal
    shift: this preserves the (x+y) mod 3 sublattice assignment, so a
    correct contraction reproduces ψ exactly under the induced 9-position
    permutation.

    Permutation π: (x,y) → ((x+1) mod 3, (y-1) mod 3). For row-major
    pos = 3y+x, the new pos for old pos p with (x = p%3, y = p//3) is
    new_pos = 3*((y-1) % 3) + ((x+1) % 3).
    """
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)

    perm = tuple(
        3 * ((p // 3 - 1) % 3) + ((p % 3 + 1) % 3)
        for p in range(9)
    )
    # Permute the AXES of psi by perm (axis 0 should now hold what was at axis perm[0]).
    psi_shifted = jnp.transpose(psi, perm)

    np.testing.assert_allclose(
        np.asarray(psi),
        np.asarray(psi_shifted),
        rtol=1e-12,
        atol=1e-12,
        err_msg=(
            f"Multisite 3×3 torus wavefunction not invariant under (1,-1) "
            f"diagonal shift at D={D}. Bond labelling in "
            f"_contract_multisite_3x3_torus is wrong."
        ),
    )
```

**Step 2: Run test to verify it fails**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_multisite_3x3_torus_translation_invariant_diagonal -v`
Expected: FAIL with `NameError` or `ModuleNotFoundError` (file is partial — `_contract_multisite_3x3_torus` is in the test file but no other helpers yet are needed for this test). Actually expected: PASS at D=1 trivially (product state), and PASS at D=2,3 if the einsum string is correct. **If it fails at D=2,3, the einsum string has a bond-pairing bug.**

If PASS → great, the contraction is correct. If FAIL → debug the einsum string before proceeding.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): _contract_multisite_3x3_torus + translation-invariance gate

3×3 PBC torus brute-force for the multisite encoding. Sublattice =
(x+y) mod 3. Translation invariance under (1,-1) diagonal shift is the
correctness gate (preserves sublattice assignment).

First step toward localising the Phase C.3 multisite-CTM bug.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Brute-force RDM extraction from torus wavefunction

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append helpers + test)

**Step 1: Write the failing test** (RDM is Hermitian, trace 1, PSD when extracted from a normalised torus ψ).

Append to `tests/test_pess_3site_multisite_rdm_invariants.py`:

```python
def _brute_force_rdm_from_torus_psi(
    psi: jnp.ndarray, sites_to_keep: tuple[int, ...]
) -> jnp.ndarray:
    """Reduce ρ = Tr_{rest}(|ψ⟩⟨ψ|) / ⟨ψ|ψ⟩ on the kept sites.

    Args:
        psi: rank-9 wavefunction from `_contract_multisite_3x3_torus` or
             equivalent, indexed by physical legs at row-major positions 0..8.
        sites_to_keep: ordered tuple of position indices in {0..8}. The
            returned ρ has axes (s_keep[0], s_keep[1], ..., s_keep[0]', ...).

    Returns:
        rho with shape `(d,) * (2 * len(sites_to_keep))`. To get the
        matrix form, reshape to `(d**k, d**k)` where k = len(sites_to_keep).
    """
    n = psi.ndim
    assert n == 9, f"expected rank-9 torus ψ, got rank-{n}"
    sites_to_trace = tuple(i for i in range(n) if i not in sites_to_keep)
    # tensordot contracts the traced axes with their conjugates.
    rho = jnp.tensordot(psi, jnp.conj(psi), axes=(sites_to_trace, sites_to_trace))
    # Normalise.
    norm_sq = jnp.tensordot(psi, jnp.conj(psi), axes=(tuple(range(n)), tuple(range(n))))
    return rho / norm_sq


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize(
    "sites_to_keep",
    [(0, 1), (0, 3), (2, 0), (2, 5), (1, 2), (3, 6)],
)
def test_brute_force_rdm_is_physical(D, sites_to_keep):
    """Brute-force RDMs from the 3×3 torus ψ must be Hermitian, trace 1, PSD."""
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)
    rho_t = _brute_force_rdm_from_torus_psi(psi, sites_to_keep)
    d = 2
    k = len(sites_to_keep)
    rho = jnp.reshape(rho_t, (d**k, d**k))

    # Hermiticity.
    err_h = float(jnp.linalg.norm(rho - jnp.conj(rho.T)))
    assert err_h < 1e-10, f"RDM not Hermitian at D={D} sites={sites_to_keep}: ‖ρ-ρ†‖={err_h:.3e}"
    # Trace 1.
    tr = complex(jnp.trace(rho))
    assert abs(tr - 1.0) < 1e-10, f"RDM trace ≠ 1 at D={D} sites={sites_to_keep}: tr={tr}"
    # PSD.
    eig = np.linalg.eigvalsh(np.asarray(rho))
    assert eig.min() > -1e-10, f"RDM not PSD at D={D} sites={sites_to_keep}: λ_min={eig.min():.3e}"
```

**Step 2: Run tests**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_brute_force_rdm_is_physical -v`
Expected: PASS at all (D, sites) combos. If FAIL, investigate the trace-axis ordering in `tensordot`.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): brute-force RDM extraction from 3×3 torus ψ

Hermiticity / trace 1 / PSD sanity for ρ = Tr_rest(|ψ⟩⟨ψ|) / <ψ|ψ>
on the 6 RDM bonds the multisite energy formula consumes.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Multisite-CTM RDM extraction helper

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append `_collect_ctm_rdms`)

**Step 1: Write the failing test** (CTM RDMs are Hermitian and trace 1 — same physical-validity sanity for the production path).

Append to `tests/test_pess_3site_multisite_rdm_invariants.py`:

```python
import dataclasses

from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor_2site, _rdm2x1_tensor_2site
from tenax.algorithms._pess_multisite_energy import (
    _rdm_3site_marginal_vw_col,
    _rdm_3site_marginal_vw_row,
    kagome_xxz_pair_hamiltonian,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import kagome_triangle_xxz_hamiltonian, pess_simple_update
from tenax.algorithms.pess_optimize import _make_multisite_indices
from tenax.core.lattice import kagome
from tenax.core.tensor import DenseTensor


def _collect_ctm_rdms(
    state: IPESSState, chi: int, max_iter: int = 100, conv_tol: float = 1e-10
) -> dict[str, jnp.ndarray]:
    """Run multisite-CTM on the encoded state and extract the 6 RDMs the
    energy formula consumes.

    Returns dict with keys {"uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col"}.
    Each value has shape (d, d, d, d) in the same axis convention used by
    `compute_energy_pess_3site_multisite` (i.e. `(s_A, s_B, s_A', s_B')` for
    2-site bonds; for the marginalised v-w bonds it's `(s_v, s_w, s_v', s_w')`).
    """
    d = 2
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas,
    )
    D = sites["u"].shape[0]
    indices = _make_multisite_indices(D, d)
    # Per-site projective normalisation (matches `build_pess_loss_3site_multisite`).
    site_tensors_by_name = {}
    for name, A in sites.items():
        A_norm = A / (jnp.linalg.norm(A) + 1e-12)
        site_tensors_by_name[name] = DenseTensor(A_norm, indices)

    envs_by_name = ctm_multisite(
        site_tensors_by_name,
        kagome(),
        chi=chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    S_u = site_tensors_by_name["u"]
    S_v = site_tensors_by_name["v"]
    S_w = site_tensors_by_name["w"]
    env_u = envs_by_name["u"]
    env_v = envs_by_name["v"]
    env_w = envs_by_name["w"]

    return {
        # 4 NN bonds (matching nn_visits order in compute_energy_pess_3site_multisite).
        "uv_h": _rdm2x1_tensor_2site(S_u, S_v, env_u, env_v),
        "uv_v": _rdm1x2_tensor_2site(S_u, S_v, env_u, env_v),
        "wu_h": _rdm2x1_tensor_2site(S_w, S_u, env_w, env_u),
        "wu_v": _rdm1x2_tensor_2site(S_w, S_u, env_w, env_u),
        # 2 marginalised-3-site v-w bonds.
        "vw_row": _rdm_3site_marginal_vw_row(S_u, S_v, S_w, env_u, env_v, env_w),
        "vw_col": _rdm_3site_marginal_vw_col(S_u, S_v, S_w, env_u, env_v, env_w),
    }


@pytest.mark.algorithm
def test_collect_ctm_rdms_returns_six_physical_rdms():
    """Smoke test: at D=2 χ=8 SU-warmstart, all 6 RDMs are 4-tensor (d,d,d,d),
    each Hermitian when reshaped to (d²,d²) and trace 1."""
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(state, H, dt_schedule=[(0.1, 50)], D_max=2)

    rdms = _collect_ctm_rdms(state, chi=8, max_iter=30, conv_tol=1e-7)
    assert set(rdms.keys()) == {"uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col"}
    for name, rdm in rdms.items():
        assert rdm.shape == (2, 2, 2, 2), f"{name}: shape={rdm.shape}"
        m = jnp.reshape(rdm, (4, 4))
        # Hermitian (loose tol — χ=8 is unconverged at D=2 but should still be ~Herm).
        err_h = float(jnp.linalg.norm(m - jnp.conj(m.T)))
        assert err_h < 1e-6, f"{name}: ‖ρ-ρ†‖={err_h:.3e}"
        # Trace ~1.
        tr = complex(jnp.trace(m))
        assert abs(tr - 1.0) < 1e-6, f"{name}: tr={tr}"
```

**Step 2: Run test**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_collect_ctm_rdms_returns_six_physical_rdms -v`
Expected: PASS in ~30-60 s (CTM forward at χ=8 D=2 is fast, plus first-call JAX compile).

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): _collect_ctm_rdms — extract the 6 multisite-CTM RDMs

Mirrors the dispatch in compute_energy_pess_3site_multisite. Smoke test
asserts all 6 RDMs are (d,d,d,d) shape, Hermitian, trace 1 at χ=8 D=2 SU.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: D=1 brute-force vs CTM equality (gate #6)

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append test)

**Step 1: Write the failing test**

Append:

```python
# Position-index representatives for the 6 brute-force RDMs (see plan table).
_BF_BOND_POSITIONS: dict[str, tuple[int, ...]] = {
    "uv_h": (0, 1),     # (u^(0,0), v^(1,0))
    "uv_v": (0, 3),     # (u^(0,0), v^(0,1))
    "wu_h": (2, 0),     # (w^(2,0), u^(0,0))
    "wu_v": (2, 5),     # (w^(2,0), u^(2,1))
    "vw_row": (1, 2),   # marginalise (0,1,2) over u → keep (1,2) = (v,w)
    "vw_col": (3, 6),   # marginalise (0,3,6) over u → keep (3,6) = (v,w)
}


def _brute_force_rdms(state: IPESSState) -> dict[str, jnp.ndarray]:
    """Compute the 6 brute-force RDMs from the 3×3 torus wavefunction.

    Returns dict with the same keys + axis convention as `_collect_ctm_rdms`.
    """
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)
    out = {}
    for name, sites_to_keep in _BF_BOND_POSITIONS.items():
        rho = _brute_force_rdm_from_torus_psi(psi, sites_to_keep)
        # rho axes: (s_keep[0], s_keep[1], s_keep[0]', s_keep[1]') — already
        # matches the 4-tensor convention of `_collect_ctm_rdms`.
        out[name] = rho
    return out


@pytest.mark.core
def test_d1_brute_force_equals_ctm_rdms():
    """Gate #6: at D=1 the encoded state is a product state, so finite-torus
    brute-force = infinite-lattice CTM exactly. All 6 RDMs must agree to 1e-10
    at any χ ≥ 4."""
    state = IPESSState.random(D=1, d=2, key=jax.random.PRNGKey(0))
    rdms_bf = _brute_force_rdms(state)
    rdms_ctm = _collect_ctm_rdms(state, chi=4, max_iter=20, conv_tol=1e-10)
    for name in rdms_bf:
        diff = float(jnp.linalg.norm(rdms_bf[name] - rdms_ctm[name]))
        assert diff < 1e-10, (
            f"D=1 brute-force ≠ CTM for {name}: ‖Δρ‖_F = {diff:.3e}. "
            f"Either the multisite encoding has a leg-mapping bug at the "
            f"product-state level, or the CTM RDM extraction does."
        )
```

**Step 2: Run test**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_d1_brute_force_equals_ctm_rdms -v`
Expected: PASS to 1e-10. If FAIL, this is itself a bug-localising signal: at D=1 there's no torus≠infinite gap so any disagreement is a real bug in either the brute-force or CTM RDM helper. **Stop-and-investigate** if FAIL — that's exactly the localising witness Phase C.3 needs.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): D=1 brute-force vs multisite-CTM RDM equality (gate #6)

At D=1 the encoded state is product, so finite-torus brute-force =
infinite-lattice CTM exactly. Strict 1e-10 equality test on all 6 RDMs.

If this fails, it directly localises the Phase C.3 bug to a consumer
helper that's already wrong at the product-state level.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Structural invariants 1-3 (Hermiticity, PSD, trace) at D=2 χ=16

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append test)

**Step 1: Write the failing test**

Append:

```python
@pytest.mark.algorithm
def test_ctm_rdms_hermitian_psd_trace1_at_d2_chi16():
    """Gates #1-3: the 6 multisite-CTM RDMs at D=2 χ=16 SU-warmstart must be
    Hermitian (1e-10), PSD (λ_min ≥ -1e-10), and trace 1 (1e-8 — looser to
    absorb χ-conv noise).

    All gates assert on the same SU state; failure of any gate fires
    immediately and points at the offending RDM.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(
        state, H, dt_schedule=[(0.1, 100), (0.01, 100)], D_max=2,
    )

    rdms_ctm = _collect_ctm_rdms(state, chi=16, max_iter=100, conv_tol=1e-9)
    failures: list[str] = []
    for name, rdm in rdms_ctm.items():
        m = jnp.reshape(rdm, (4, 4))
        # Gate #1: Hermitian.
        err_h = float(jnp.linalg.norm(m - jnp.conj(m.T)))
        if err_h > 1e-10:
            failures.append(f"{name}: ‖ρ-ρ†‖_F={err_h:.3e} > 1e-10")
        # Gate #2: PSD. Hermitise first (real symmetric eigenvalues are
        # what physics demands; small Hermiticity violation is OK at χ=16).
        m_h = 0.5 * (m + jnp.conj(m.T))
        eig = np.linalg.eigvalsh(np.asarray(m_h))
        if eig.min() < -1e-10:
            failures.append(f"{name}: λ_min={eig.min():.3e} < -1e-10 (not PSD)")
        # Gate #3: trace 1 (looser tol).
        tr = complex(jnp.trace(m))
        if abs(tr - 1.0) > 1e-8:
            failures.append(f"{name}: |tr-1|={abs(tr - 1.0):.3e} > 1e-8")
    assert not failures, "Structural-invariants violations:\n  " + "\n  ".join(failures)
```

**Step 2: Run test**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_ctm_rdms_hermitian_psd_trace1_at_d2_chi16 -v -s`
Expected: PASS in ~60 s. **If FAIL on any RDM**, that's the bug — file the failure on a memory note and the C.3 follow-up PR fixes the offending helper.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): structural invariants 1-3 (Herm/PSD/trace) at D=2 χ=16

Strict gates on the 6 multisite-CTM RDMs at the production D=2 χ=16
SU-warmstart point. Failure on any RDM directly localises the Phase C.3
bug to that helper.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Marginalisation-consistency (gate #4) at D=2 χ=16

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append test)

**Step 1: Write the failing test**

Marginalisation consistency: if the multisite-CTM environments are correctly constructed, then two paths to the v-w 2-site RDM must agree on the infinite lattice:

(a) `_rdm_3site_marginal_vw_row` (1×3 horizontal closure, marginalise u) — gives ρ_vw via T_u-mediated path through S_u.
(b) `_rdm2x1_tensor_2site(S_v, S_w, env_v, env_w)` — DIRECT 2-site call across the dim-1 v-w iPEPS bond. Both inputs are the SAME envs as path (a). Note: v.right=w in the kagome neighbor map, so this is a valid 2-site h-bond call.

If the multisite-CTM envs are correct, both paths probe the same physical v-w correlator and must agree to 1e-8 at converged CTM. **This test fires if the dim-1 v-w bonds break the standard 2-site RDM helpers under-the-hood, OR if `_rdm_3site_marginal_vw_row` consumes envs incorrectly.**

Append:

```python
@pytest.mark.algorithm
def test_marginalisation_consistency_at_d2_chi16():
    """Gate #4: ρ_vw computed two ways from the SAME multisite-CTM envs must
    agree to 1e-8 on the infinite lattice.

    Path A: `_rdm_3site_marginal_vw_row` (1×3 horizontal block, marginalise u).
    Path B: `_rdm2x1_tensor_2site(S_v, S_w, env_v, env_w)` directly across the
            dim-1 v-w iPEPS bond (v.right = w in the kagome neighbour map).

    Disagreement ⇒ either the envs feed the two helpers inconsistently
    (env-construction bug) or one of the helpers handles the dim-1 v-w bond
    incorrectly.
    """
    d = 2
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=d)
    state = IPESSState.random(D=2, d=d, key=jax.random.PRNGKey(0))
    state = pess_simple_update(
        state, H, dt_schedule=[(0.1, 100), (0.01, 100)], D_max=2,
    )

    # Re-run CTM here so we control envs + RDM helpers explicitly.
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas,
    )
    D = sites["u"].shape[0]
    indices = _make_multisite_indices(D, d)
    site_tensors_by_name = {
        name: DenseTensor(A / (jnp.linalg.norm(A) + 1e-12), indices)
        for name, A in sites.items()
    }
    envs_by_name = ctm_multisite(
        site_tensors_by_name, kagome(), chi=16, max_iter=100, conv_tol=1e-9,
    )
    S_u = site_tensors_by_name["u"]
    S_v = site_tensors_by_name["v"]
    S_w = site_tensors_by_name["w"]
    env_u = envs_by_name["u"]
    env_v = envs_by_name["v"]
    env_w = envs_by_name["w"]

    # Path A: 3-site row-marginal.
    rho_vw_A = _rdm_3site_marginal_vw_row(S_u, S_v, S_w, env_u, env_v, env_w)
    # Path B: direct 2-site h-bond across the dim-1 v-w iPEPS bond.
    rho_vw_B = _rdm2x1_tensor_2site(S_v, S_w, env_v, env_w)
    # Renormalise both to trace 1 (in case env-only normalisation differs).
    rho_vw_A_m = jnp.reshape(rho_vw_A, (d * d, d * d))
    rho_vw_B_m = jnp.reshape(rho_vw_B, (d * d, d * d))
    rho_vw_A_m = rho_vw_A_m / jnp.trace(rho_vw_A_m)
    rho_vw_B_m = rho_vw_B_m / jnp.trace(rho_vw_B_m)
    diff = float(jnp.linalg.norm(rho_vw_A_m - rho_vw_B_m))
    assert diff < 1e-8, (
        f"Marginalisation-consistency FAIL at D=2 χ=16: "
        f"‖ρ_vw_3site_marginal − ρ_vw_2site_direct‖_F = {diff:.3e} > 1e-8. "
        f"The multisite-CTM envs feed the two RDM paths inconsistently, "
        f"or one of the helpers mishandles the dim-1 v-w iPEPS bond."
    )
```

**Step 2: Run test**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_marginalisation_consistency_at_d2_chi16 -v -s`
Expected: this is the **most likely test to fire** given the C.3 symptoms. If it does, that directly identifies Suspect 2 or 3 in the design's decision tree.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): marginalisation-consistency gate #4 at D=2 χ=16

ρ_vw via 3-site row-marginal vs ρ_vw via direct 2-site h-bond across the
dim-1 v-w iPEPS bond — both must agree at converged CTM. Most likely
test to fire given C.3 symptoms; localises bug to env consumption.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Per-bond ⟨H⟩ spectrum bound (gate #5)

**Files:**
- Modify: `tests/test_pess_3site_multisite_rdm_invariants.py` (append test)

**Step 1: Write the failing test**

Append:

```python
@pytest.mark.algorithm
def test_per_bond_energy_in_local_spectrum():
    """Gate #5: tr(ρ_bond · H_pair) for each of the 6 bonds must lie in
    [-3/4 - eps, 1/4 + eps]. Outside ⇒ ρ_bond is unphysical at the level
    of expectation values.

    For spin-½ isotropic XXZ at δ=1: eigvalsh(H_pair) = {-3/4, 1/4×3}.
    """
    eps = 1e-8
    spec_min, spec_max = -0.75, 0.25
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta=1.0, d=2))  # (d,d,d,d)

    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(
        state, H, dt_schedule=[(0.1, 100), (0.01, 100)], D_max=2,
    )
    rdms_ctm = _collect_ctm_rdms(state, chi=16, max_iter=100, conv_tol=1e-9)

    failures: list[str] = []
    for name, rdm in rdms_ctm.items():
        e = complex(jnp.einsum("ijkl,ijkl->", rdm, H_pair))
        e_real = e.real
        if not (spec_min - eps <= e_real <= spec_max + eps):
            failures.append(
                f"{name}: ⟨H⟩={e_real:+.4e} outside [{spec_min}, {spec_max}]"
            )
    assert not failures, "Spectrum-bound violations:\n  " + "\n  ".join(failures)
```

**Step 2: Run test**

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py::test_per_bond_energy_in_local_spectrum -v -s`
Expected: PASS at the SU state (E/site = -0.21 in the C.3 probe; per-bond ≈ -0.105, well inside the spectrum). This gate is an early-warning for stronger pathologies, not the test the C.3 blocker would currently fire.

**Step 3: Commit**

```bash
git add tests/test_pess_3site_multisite_rdm_invariants.py
git commit -m "test(pess): per-bond ⟨H⟩ ∈ spectrum bound gate #5

Each of the 6 bonds' tr(ρ·H_pair) must lie in [-3/4, 1/4] for spin-½
δ=1 XXZ. Early-warning gate for stronger pathologies than the current
C.3 blocker.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 7).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Audit probe over the (D, χ, state) ladder

**Files:**
- Create: `examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py`

**Step 1: Write the probe** (no asserts; pure diagnostic + JSON; matches the existing C.3 probe pattern):

```python
"""Phase C.3 RDM brute-force vs multisite-CTM diagnostic.

Audit-only probe — writes a JSON of per-bond Frobenius deltas and per-bond
energies across a (D, χ, state-kind) ladder. Stop-and-ask checkpoint #5
in ``docs/plans/2026-05-05-multisite-kagome-pess.md`` is reached by
inspecting the resulting JSON / printed table; this script does NOT assert
the variational floor.

See the design at
``docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic-design.md`` and the
implementation plan at
``docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md``.

Usage::

    python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py
    python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py \\
        --D-ladder 1,2,3,4 --chi-ladder 8,16,32 --state-kind both
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._pess_multisite_energy import kagome_xxz_pair_hamiltonian
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_simple_update,
)

# Reuse helpers from the test file (test files are importable as modules).
from tests.test_pess_3site_multisite_rdm_invariants import (
    _BF_BOND_POSITIONS,
    _brute_force_rdms,
    _collect_ctm_rdms,
)

DELTA = 1.0
D_PHYS = 2


def _su_warmstart(state: IPESSState, D: int) -> IPESSState:
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    return pess_simple_update(
        state, H, dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)], D_max=D,
    )


def _per_bond_metrics(rdms_bf, rdms_ctm, H_pair) -> dict[str, dict[str, float]]:
    out = {}
    for name in rdms_bf:
        bf = rdms_bf[name]
        ctm = rdms_ctm[name]
        diff = float(jnp.linalg.norm(bf - ctm))
        e_bf = float(complex(jnp.einsum("ijkl,ijkl->", bf, H_pair)).real)
        e_ctm = float(complex(jnp.einsum("ijkl,ijkl->", ctm, H_pair)).real)
        out[name] = {
            "frobenius_delta": diff,
            "energy_bf": e_bf,
            "energy_ctm": e_ctm,
            "energy_delta": e_ctm - e_bf,
        }
    return out


def _run_one(
    D: int, chi: int, state_kind: str, seed: int, H_pair
) -> dict:
    t0 = time.perf_counter()
    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))
    if state_kind == "su":
        state = _su_warmstart(state, D)
    t_state = time.perf_counter() - t0

    t0 = time.perf_counter()
    rdms_bf = _brute_force_rdms(state)
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    rdms_ctm = _collect_ctm_rdms(state, chi=chi, max_iter=100, conv_tol=1e-9)
    t_ctm = time.perf_counter() - t0

    metrics = _per_bond_metrics(rdms_bf, rdms_ctm, H_pair)
    return {
        "D": D,
        "chi": chi,
        "state_kind": state_kind,
        "seed": seed,
        "wall_seconds_state": t_state,
        "wall_seconds_brute_force": t_bf,
        "wall_seconds_ctm": t_ctm,
        "metrics": metrics,
    }


def _print_table(rows: list[dict]) -> None:
    bonds = list(_BF_BOND_POSITIONS.keys())
    header = f"{'D':>3} {'chi':>4} {'kind':>6}  " + "  ".join(
        f"{b:>13}" for b in bonds
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        row = f"{r['D']:>3} {r['chi']:>4} {r['state_kind']:>6}  " + "  ".join(
            f"{r['metrics'][b]['frobenius_delta']:>13.3e}" for b in bonds
        )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D-ladder", default="1,2,3,4")
    parser.add_argument("--chi-ladder", default="8,16,32")
    parser.add_argument("--state-kind", choices=["random", "su", "both"], default="su")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    Ds = [int(x) for x in args.D_ladder.split(",")]
    chis = [int(x) for x in args.chi_ladder.split(",")]
    kinds = ["random", "su"] if args.state_kind == "both" else [args.state_kind]

    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta=DELTA, d=D_PHYS))
    rows: list[dict] = []
    for D in Ds:
        for chi in chis:
            for kind in kinds:
                print(f"=== D={D} χ={chi} state={kind} ===", flush=True)
                row = _run_one(D, chi, kind, args.seed, H_pair)
                rows.append(row)

    print("\n=== Frobenius ‖ρ_brute − ρ_CTM‖_F per bond ===\n")
    _print_table(rows)

    args.output.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the probe**

Smoke-test the probe at the cheapest setting first:

Command: `uv run python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py --D-ladder 1,2 --chi-ladder 8 --state-kind su`
Expected: completes in < 90 s; prints a 2-row Frobenius table; writes JSON. At D=1 the row should show `~1e-12` deltas (encoded state is product). At D=2, deltas can be O(0.1) — that is the localising signal we want to read.

**Step 3: Commit**

```bash
git add examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py
git commit -m "audit(pess): RDM brute-force vs multisite-CTM probe (Phase C.3)

Audit-only probe over (D, χ, state) ladder writing JSON + printed table
of per-bond Frobenius deltas and energies. Reuses brute-force +
CTM-RDM helpers from tests/test_pess_3site_multisite_rdm_invariants.py.

Localising signal is the structure of the per-bond delta table — which
RDM diverges most strongly between brute-force (3×3 PBC torus) and
multisite-CTM (infinite lattice) at D≥2.

Plan: docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md (Task 8).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Run the full audit + record findings

**Step 1: Run the full ladder**

Command: `uv run python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py --D-ladder 1,2,3,4 --chi-ladder 8,16,32 --state-kind both`
Expected: ~10 min wall; writes the full JSON. Inspect the printed table — read the structure-of-deltas via the design's decision tree.

**Step 2: Run the strict gates** (full file, including the cheap `core` and slower `algorithm` markers):

Command: `uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py -v`
Expected: every test PASS, OR a specific gate fires with a localised error message. **Either outcome is informative** — passing gates rule out suspects; firing gates name the bug.

**Step 3: Update memory + commit findings**

If a gate fires, update `~/.claude/projects/-home-yjkao-tenax/memory/project_kagome_3site_multisite_pivot.md` with the localised finding (which RDM, which gate, what the delta table says). Commit a JSON snapshot of the audit run alongside if instructive:

```bash
git add examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.json
git commit -m "audit(pess): C.3 RDM diagnostic findings — <localised result>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If all gates pass: that itself rules out qualitative pathologies and points the C.3 fix toward quantitative-correlation magnitude — likely a deeper investigation of the multisite-CTM envs' long-range correlation accuracy at finite χ.

---

## Stop-and-ask checkpoints (active)

1. **After Task 1** — if `test_multisite_3x3_torus_translation_invariant_diagonal` fails at D≥2, the einsum bond labelling is wrong. Stop and debug with a smaller (D=2) hand-traced case before continuing.
2. **After Task 4** — if `test_d1_brute_force_equals_ctm_rdms` fails at D=1, that **is** the bug-localising witness. Stop the rest of the plan and pivot to fixing the helper that disagrees.
3. **After Task 6** — if `test_marginalisation_consistency_at_d2_chi16` fails at D=2 χ=16, that's the most likely localising path. Stop and pivot.
4. **After Task 9** — if all gates pass, the diagnostic is informative *negatively*: it rules out qualitative-pathology suspects and the C.3 follow-up shifts focus to quantitative env-accuracy at finite χ.

---

## Cross-references

- Design: `docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic-design.md`
- Parent plan: `docs/plans/2026-05-05-multisite-kagome-pess.md` (Phase C.3 BLOCKED, this resumes)
- Memory: `~/.claude/projects/-home-yjkao-tenax/memory/project_kagome_3site_multisite_pivot.md`
- C.3 audit-trail probes (precedent): `examples/kagome_pess_multisite_phase_c3_{probe,tight_ctm_probe,hz_diag}.py`
- WF-fidelity test (1-cell encoding gate): `tests/test_pess_3site_multisite_wavefunction.py`
- Multisite-CTM dispatch: `src/tenax/algorithms/_ctm_tensor_convergence.py::ctm_multisite`
- Energy chain under audit: `src/tenax/algorithms/_pess_multisite_energy.py::compute_energy_pess_3site_multisite`
- PR #398 (parent): https://github.com/tenax-lab/tenax/pull/398
