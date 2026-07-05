# 2-site Split-CTM SymmetricTensor (Phase 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the coupled 2-site checkerboard split-CTM path (forward + explicit/implicit AD) work for bosonic SymmetricTensor (trivial and nontrivial U(1)/Zₙ), mirroring the proven single-site symmetric path — closing #463 Phase 3. Fermionic is Phase 4.

**Architecture:** The 2plaq split path is already substantially polymorphic — enlarged corners are label-based `contract` + invertible `_fuse_ket_bra`, the projector already dispatches to the block-sparse symmetric kernel (`_compute_2x2_projector_symmetric`), and projectors are applied via the #605-safe `_apply_proj_unfused`. The one real gap is per-sector interlayer-SVD truncation: `_svd_split_edge_tensor` only takes the symmetric branch when it receives `base_charges`, which the 2plaq path currently never derives (driver hard-codes `None`; absorbs never derive it). Following the codebase's *un-plumbing* direction (`base_charges` is vestigial in `_apply_projector`), we derive it **locally at the two points of use** — each absorb reads the `A` it already holds; the driver reads `site_tensors[s]` — via a small local helper, adding **no** parameter to any sweep/absorb signature. The AD `custom_vjp` machinery is pure `jax.tree` over the env pytree and symmetry-agnostic; it is verified, not rewritten.

**Tech Stack:** Python 3.11/3.12, JAX (float64), Tenax `Tensor`/`DenseTensor`/`SymmetricTensor`/`U1Symmetry`/`TensorIndex`, `jax.custom_vjp`, pytest (`-m core` + `slow`).

---

## Critical validation lessons (read before writing any test)

1. **Trivial-U(1) parity is a *forward-correctness* guard, not a `base_charges` guard.** For trivial (all-zero) charges there is a single charge sector, so per-sector truncation is *identical* to global truncation — the trivial-U(1) energy/AD parity tests (Task 2/3) pass with or without the `base_charges` fix. They validate that the polymorphic symmetric path is *correct* (symmetric == dense). The `base_charges` fix is exercised only by **nontrivial** charges (multiple sectors competing for the `chi_I` budget), so the red→green test for the source fix is the **nontrivial-charge sector-preservation smoke** (Task 1) — the 2-site analogue of the single-site `test_fermionic_u1_charges_preserved_across_sweeps` regression.

2. **Never use random tensors for tight energy/AD parity.** The fused 2-site CTM oscillates on random input (see [[feedback_ctm_parity_needs_convergent_input]]), so every tight parity test uses a **physical convergent** Heisenberg Néel checkerboard from 2-site simple update via `_build_su_neel` (`tests/test_split_ctm_2site.py:859`), wrapped as a trivial-U(1) SymmetricTensor. The nontrivial-charge smoke (Task 1) is the *only* place a random SymmetricTensor is acceptable — it asserts structure (finite + sectors preserved), never an energy value.

3. **The trusted AD gate is `implicit == explicit`, not `implicit == finite-difference`.** The split energy_fn carries a pre-existing Wirtinger gap that AD-vs-FD inherits. Use the **XXZ Δ=0.3** clean-regime gate for the machine-exact assertion (`cos > 1−1e-9`, `rel < 1e-6`); the Heisenberg point is floored at `rel ~ 5e-4` by the degenerate-SV SVD backward (a known limitation, not a bug — see [[feedback_ad_parity_degenerate_svd_floor]]) and gets only a loose companion assertion.

---

## File Structure

- **`src/tenax/algorithms/_split_ctm_tensor_moves.py`** (MODIFY) — owns the 2plaq absorbs. Add one small module-level helper `_split_base_charges(A)`; call it at the top of each of the four `_split_ctm_absorb_*_2plaq` and pass the result to that function's `_svd_split_edge_tensor` call(s). No signature changes.
- **`src/tenax/algorithms/_split_ctm_tensor_convergence.py`** (MODIFY) — owns the 2-site sweep driver. Replace the hard-coded `base_charges=None` at the `_compute_split_plaquette_projector_pair` call with `_split_base_charges(site_tensors[s])`.
- **`tests/test_split_ctm_2site_symmetric.py`** (CREATE) — owns the Phase-3 symmetric suite: nontrivial-charge sector-preservation smoke (Task 1), trivial-U(1) energy parity (Task 2), symmetric AD parity (Task 3). Local helpers `_to_trivial_u1`, `_build_nontrivial_u1_pair`, `_xxz_gate`; reuses `_build_su_neel` / `_heisenberg_gate` from `tests/test_split_ctm_2site.py`.

### Key reference signatures (already on the branch; do not modify)

```python
# _split_ctm_tensor_moves.py
def _svd_split_edge_tensor(T, left_labels, right_labels, chi_I, ket_relabels,
                           bra_relabels, base_charges=None) -> tuple[Tensor, Tensor]
#   symmetric branch at line 852: `if isinstance(T, SymmetricTensor) and base_charges is not None:`
def _split_ctm_absorb_bottom_2plaq(env_src, A, A_bar, P_top_left, P_bot_left,
                                   P_top_curr, P_bot_curr, chi_I)   # + left/right/top twins

# _split_ctm_tensor_convergence.py
def _split_ctm_sweep_multisite_2x2(envs, site_tensors, bars, neighbors, chi, chi_I)
def _initialize_split_multisite_env(site_tensors, chi, chi_I) -> dict[Coord, SplitCTMTensorEnv]
def ctm_split_tensor_2site(A, B, chi, max_iter=100, conv_tol=1e-8, chi_I=None,
                           renormalize=True, recipe="2x2") -> tuple[SplitCTMTensorEnv, SplitCTMTensorEnv]

# _split_ctm_tensor_energy.py
def compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=None) -> jax.Array

# _split_ctm_energy_ad.py  (Phase-2, on this branch)
def ctm_energy_split_explicit_2site(site_tensors, neighbors, gate, *, chi, warmup_steps,
                                    backprop_steps, chi_I=None, renormalize=True, **_)
def ctm_energy_split_implicit_2site(site_tensors, neighbors, gate, *, chi, max_iter,
                                    conv_tol, chi_I=None, renormalize=True, min_iter=2, **_)

# _ctm_tensor_convergence.py
CHECKERBOARD_NEIGHBORS   # {(0,0): {...}, (1,0): {...}} bipartite neighbor map
```

The single-site precedent this mirrors: `_split_ctm_tensor_moves.py:1363/1451/1539/1627`
(each single-site move derives `A.indices[0].charges if isinstance(A, SymmetricTensor) else None`
locally); tests `tests/test_split_ctm_tensor.py::TestSplitCTMSymmetric` (lines 495-660).

---

## Task 1: `base_charges` local derivation (source fix) + nontrivial-charge sector-preservation

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py` (add `_split_base_charges`; edit 4 absorbs)
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py` (driver line ~272)
- Test: `tests/test_split_ctm_2site_symmetric.py` (create)

The core source change: engage per-sector interlayer-SVD truncation on the 2-site symmetric path so charge sectors survive across sweeps, driving it with the nontrivial-charge smoke.

- [ ] **Step 1: Write the failing sector-preservation smoke test**

Create `tests/test_split_ctm_2site_symmetric.py`:

```python
"""#463 Phase 3 — 2-site split-CTM SymmetricTensor support (bosonic U(1)/Zn).

Trivial-U(1) parity (Tasks 2/3) validates the polymorphic symmetric path is
correct (symmetric == dense). The nontrivial-charge sector-preservation smoke
(Task 1) is the red->green gate for per-sector interlayer-SVD truncation: without
base_charges the 2plaq path uses GLOBAL truncation, which starves a charge sector
across sweeps. Mirrors the single-site regression
test_fermionic_u1_charges_preserved_across_sweeps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
    _split_ctm_sweep_multisite_2x2,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tests.test_split_ctm_2site import _build_su_neel, _heisenberg_gate


def _bond_sector_count(env_tensor, interlayer_label):
    """Number of distinct charge sectors on a tensor's interlayer bond leg."""
    pos = env_tensor.labels().index(interlayer_label)
    return len(np.unique(np.asarray(env_tensor.indices[pos].charges)))


def _build_nontrivial_u1_pair(D=2, d=2):
    """Direction-dependent (A != B) nontrivial bosonic-U(1) checkerboard pair.

    Virtual/phys charges [0, 1] give two competing sectors, so global vs
    per-sector interlayer truncation differ. Requires D == 2 and d == 2.
    """
    assert D == 2 and d == 2, "helper hard-codes the [0,1] two-sector layout"
    sym = U1Symmetry()
    vc = np.array([0, 1], dtype=np.int32)
    pc = np.array([0, 1], dtype=np.int32)
    flows = (
        FlowDirection.OUT, FlowDirection.IN, FlowDirection.OUT,
        FlowDirection.IN, FlowDirection.IN,
    )
    labels = ("u", "d", "l", "r", "phys")
    charge_sets = (vc, vc, vc, vc, pc)
    indices = tuple(
        TensorIndex.from_charges(sym, c.copy(), f, label=lbl)
        for c, f, lbl in zip(charge_sets, flows, labels)
    )
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    A = SymmetricTensor.random_normal(indices, kA)
    B = SymmetricTensor.random_normal(indices, kB)
    return A, B


def test_2site_symmetric_charge_sectors_preserved():
    """Nontrivial-charge 2-site split sweeps stay finite, remain SymmetricTensor,
    and preserve both interlayer charge sectors (no sector starvation)."""
    A, B = _build_nontrivial_u1_pair(D=2, d=2)
    site_tensors = {(0, 0): A, (1, 0): B}
    bars = {c: t.bar() for c, t in site_tensors.items()}
    chi, chi_I = 6, 4

    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    for _ in range(3):
        envs = _split_ctm_sweep_multisite_2x2(
            envs, site_tensors, bars, CHECKERBOARD_NEIGHBORS, chi, chi_I
        )

    for coord, env in envs.items():
        for t in env:
            assert isinstance(t, SymmetricTensor), (
                f"{coord} env tensor collapsed to non-symmetric type"
            )
            assert jnp.all(jnp.isfinite(t.todense())), (
                f"{coord} env tensor non-finite after sweeps"
            )
        # The T1 ket interlayer bond must retain BOTH input sectors.
        assert _bond_sector_count(env.T1_ket, "t1k_I") >= 2, (
            f"{coord}: interlayer bond starved to a single charge sector "
            "(global truncation dropped a sector — base_charges not engaged)"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py::test_2site_symmetric_charge_sectors_preserved -x -q`

Expected: FAIL — the interlayer bond starves to a single sector (assertion on `_bond_sector_count`), because `_split_ctm_sweep_multisite_2x2` passes `base_charges=None`, so `_svd_split_edge_tensor` takes the global-truncation branch.

If instead it PASSES (this random state happened not to starve a sector), tighten the competition before implementing: raise `chi_I` toward `chi` and/or add a second `_build_nontrivial_u1_pair` seed, so the two sectors genuinely compete for the `chi_I` budget. Do not implement the fix until the test is red — a green-without-fix test proves nothing about `base_charges`.

- [ ] **Step 3: Add the local `base_charges` helper + engage it (source fix)**

In `src/tenax/algorithms/_split_ctm_tensor_moves.py`, add a module-level helper near the other private helpers (e.g. just above `_svd_split_edge_tensor`, ~line 826). `SymmetricTensor` is already imported in this module (used at line 852).

```python
def _split_base_charges(A: Tensor) -> np.ndarray | None:
    """Charges of a site tensor's first leg, for per-sector interlayer-SVD
    truncation on the symmetric split path.

    Returns ``None`` for a DenseTensor (global truncation). Derived locally at
    the point of use — mirrors the single-site moves and the fused
    ``_get_base_charges``; deliberately NOT a plumbed cross-call parameter
    (``base_charges`` is being un-plumbed — see ``_apply_projector``).
    """
    return A.indices[0].charges if isinstance(A, SymmetricTensor) else None
```

Then in each of the four `_split_ctm_absorb_{bottom,left,right,top}_2plaq`, derive
`base_charges` at the top of the function (right after the `def`/docstring) and pass
it to every `_svd_split_edge_tensor` call in that function. For
`_split_ctm_absorb_bottom_2plaq` the edit is:

```python
def _split_ctm_absorb_bottom_2plaq(
    env_src, A, A_bar, P_top_left, P_bot_left, P_top_curr, P_bot_curr, chi_I
):
    """..."""  # unchanged docstring
    base_charges = _split_base_charges(A)
    # ... unchanged corner/edge growth + projector application ...
    T3_ket_new, T3_bra_new = _svd_split_edge_tensor(
        T3g,
        left_labels=["chi_new", "u"],
        right_labels=["u_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t3k_r", "u": "d_ket", "_svd_bond": "t3k_I"},
        bra_relabels={"_svd_bond": "t3b_I", "u_bra": "d_bra", "chi_new_r": "t3b_l"},
        base_charges=base_charges,   # <-- add this line
    )
    # ... unchanged flow-fixups + return ...
```

Apply the identical two-line change (derive `base_charges = _split_base_charges(A)`
at the top; add `base_charges=base_charges` to the `_svd_split_edge_tensor` call) to
`_split_ctm_absorb_left_2plaq`, `_split_ctm_absorb_right_2plaq`, and
`_split_ctm_absorb_top_2plaq`. Confirm each absorb has exactly one
`_svd_split_edge_tensor` call:

Run: `grep -n "_svd_split_edge_tensor" src/tenax/algorithms/_split_ctm_tensor_moves.py`

Every call site *inside* an absorb gets `base_charges=base_charges`; the definition
(line ~827) is unchanged.

Then in `src/tenax/algorithms/_split_ctm_tensor_convergence.py`, import the helper
and replace the hard-coded `None` at the projector call (~line 272):

```python
from tenax.algorithms._split_ctm_tensor_moves import _split_base_charges
```

(add to the existing function-local import block in `_split_ctm_sweep_multisite_2x2`
that already pulls the absorb helpers — keep it function-local to avoid the module
import cycle noted at line 234), and:

```python
            Pt, Pb, _eps, _sS = _compute_split_plaquette_projector_pair(
                envs_old[s],
                envs_old[s_TR],
                envs_old[s_BL],
                envs_old[s_BR],
                site_tensors[s],
                bars[s],
                site_tensors[s_TR],
                bars[s_TR],
                site_tensors[s_BL],
                bars[s_BL],
                site_tensors[s_BR],
                bars[s_BR],
                chi,
                direction,
                base_charges=_split_base_charges(site_tensors[s]),  # <-- was None
            )
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py::test_2site_symmetric_charge_sectors_preserved -x -q`
Expected: PASS — per-sector truncation now preserves both interlayer sectors across sweeps.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_moves.py \
        src/tenax/algorithms/_split_ctm_tensor_convergence.py \
        tests/test_split_ctm_2site_symmetric.py
git commit -m "feat(#463): 2-site split-CTM per-sector interlayer truncation (Phase 3 Task 1)"
```

---

## Task 2: Trivial-U(1) forward + energy parity (Tier 1/2)

**Files:**
- Test: `tests/test_split_ctm_2site_symmetric.py` (append)

Validate the polymorphic symmetric 2-site forward is *correct*: on a physical
convergent Néel checkerboard wrapped as trivial-U(1), the symmetric energy matches
the dense energy. This is a regression guard (single-sector → sector-independent of
Task 1), across D∈{2,3}, χ∈{4,8}.

- [ ] **Step 1: Write the energy-parity test**

Append to `tests/test_split_ctm_2site_symmetric.py`:

```python
def _to_trivial_u1(A):
    """Wrap a dense iPEPS site (shape (D,D,D,D,d), labels u,d,l,r,phys) as a
    trivial-U(1) SymmetricTensor with the same data — robust to whatever indices
    the SU builder attached."""
    data = A.todense()
    sym = U1Symmetry()
    flows = (
        FlowDirection.OUT, FlowDirection.IN, FlowDirection.OUT,
        FlowDirection.IN, FlowDirection.IN,
    )
    labels = ("u", "d", "l", "r", "phys")
    indices = tuple(
        TensorIndex.from_charges(sym, np.zeros(n, dtype=np.int32), f, label=lbl)
        for n, f, lbl in zip(data.shape, flows, labels)
    )
    return SymmetricTensor.from_dense(data, indices)


@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 8)])
def test_2site_symmetric_energy_matches_dense(D, chi):
    """Trivial-U(1) symmetric split energy == dense split energy on a convergent
    Neel checkerboard. The D=3 case is also the design-§10 hard-fusion guard."""
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = _build_su_neel(D=D)
    gate = _heisenberg_gate()

    envA_d, envB_d = ctm_split_tensor_2site(
        A, B, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_dense = float(compute_energy_split_ctm_tensor_2site(A, B, envA_d, envB_d, gate, d=2))

    As, Bs = _to_trivial_u1(A), _to_trivial_u1(B)
    envA_s, envB_s = ctm_split_tensor_2site(
        As, Bs, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_sym = float(compute_energy_split_ctm_tensor_2site(As, Bs, envA_s, envB_s, gate, d=2))

    assert np.isfinite(E_sym), f"symmetric energy not finite: {E_sym}"
    assert abs(E_sym - E_dense) < 1e-6, f"sym={E_sym} dense={E_dense} (D={D}, chi={chi})"
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py::test_2site_symmetric_energy_matches_dense -q`
Expected: PASS (all 3 parametrizations).

This is a *regression guard*, so PASS on the first run is the expected outcome (the
polymorphic path already handles trivial-U(1); Task 1's fix is sector-independent
here). If it FAILS, the failure is a genuine dense-only assumption on the 2plaq
symmetric forward — diagnose the specific op (look for a `.todense()`, a raw
`reshape`, or a `DenseTensor`-typed constructor on the path the test exercises) and
fix it minimally in `_split_ctm_tensor_moves.py`, then re-run. A D=2 pass with a D=3
fail specifically implicates a hard-fusion charge-conjugation clash in the enlarged
corner (design §10) — inspect `_build_split_enlarged_corner`'s `_fuse_ket_bra` seams.

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_ctm_2site_symmetric.py
git commit -m "test(#463): 2-site trivial-U(1) split energy parity vs dense (Phase 3 Task 2)"
```

---

## Task 3: Symmetric AD parity (Tier 3)

**Files:**
- Test: `tests/test_split_ctm_2site_symmetric.py` (append)

Validate autodiff through the coupled 2-site symmetric fixed point: on the trivial-
U(1) wrapped convergent state, symmetric `implicit == explicit` and symmetric-grad
== dense-grad, using the XXZ Δ=0.3 clean-regime gate. These tests run the CTM to
convergence with AD → mark them `slow` (out of `-m core`, matching the Phase-2 AD
tests).

- [ ] **Step 1: Write the symmetric AD parity test (XXZ clean-regime gate)**

Append to `tests/test_split_ctm_2site_symmetric.py`:

```python
def _xxz_gate(delta=0.3, d=2):
    """XXZ 2-site gate H = delta*Sz.Sz + 0.5(Sp.Sm + Sm.Sp). delta=0.3 avoids the
    degenerate-SV SVD-backward floor that Heisenberg (delta=1) hits."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


@pytest.mark.slow
def test_2site_symmetric_ad_matches_explicit_and_dense():
    """PRIMARY Tier-3 gate: trivial-U(1) symmetric split AD gradient (w.r.t.
    sublattice A) matches BOTH the symmetric explicit (unrolled) gradient and the
    dense implicit gradient, on the convergent state with the XXZ Delta=0.3 gate.

    implicit==explicit / sym==dense, NOT implicit==FD: the split energy_fn carries a
    pre-existing Wirtinger gap (see feedback_ad_parity_degenerate_svd_floor).
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit_2site,
    )

    A_d, B_d = _build_su_neel(D=2)
    A_s, B_s = _to_trivial_u1(A_d), _to_trivial_u1(B_d)
    gate = _xxz_gate(0.3)
    chi = 4  # chi = D*D lossless on the physical low-interlayer-rank state

    def _flat_grad(g):
        return jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])

    def loss_imp(a, b):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): b}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=80, conv_tol=1e-13, min_iter=2,
        ).real

    def loss_exp(a, b):
        return ctm_energy_split_explicit_2site(
            {(0, 0): a, (1, 0): b}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, warmup_steps=40, backprop_steps=40,
        ).real

    # symmetric implicit vs symmetric explicit
    e_i, g_i = jax.value_and_grad(loss_imp)(A_s, B_s)
    e_e, g_e = jax.value_and_grad(loss_exp)(A_s, B_s)
    gi, ge = _flat_grad(g_i), _flat_grad(g_e)
    assert jnp.allclose(e_i, e_e, atol=1e-9), f"sym energy mismatch: {e_i} vs {e_e}"
    cos_ie = float(jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge)))
    rel_ie = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))
    assert cos_ie > 1 - 1e-9, f"sym implicit vs explicit direction: cos={cos_ie}"
    assert rel_ie < 1e-6, f"sym implicit vs explicit magnitude: rel={rel_ie}"

    # symmetric implicit vs dense implicit (same wrapped data)
    e_d, g_d = jax.value_and_grad(loss_imp)(A_d, B_d)
    gd = _flat_grad(g_d)
    assert jnp.allclose(e_i, e_d, atol=1e-8), f"sym vs dense energy: {e_i} vs {e_d}"
    cos_sd = float(jnp.real(jnp.vdot(gi, gd)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(gd)))
    rel_sd = float(jnp.linalg.norm(gi - gd) / jnp.linalg.norm(gd))
    assert cos_sd > 1 - 1e-9, f"sym vs dense direction: cos={cos_sd}"
    assert rel_sd < 1e-6, f"sym vs dense magnitude: rel={rel_sd}"
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py::test_2site_symmetric_ad_matches_explicit_and_dense -x -q`
Expected: PASS.

Expected outcome is PASS: the `custom_vjp` is `jax.tree`-generic, so SymmetricTensor
leaves flow through unchanged. If it FAILS on a *direction* mismatch (`cos` far below
1), there is a genuine dense-only assumption on the AD path — localize it (a
`.todense()` / dense reshape reachable from `_split_ctm_converge_multisite`'s fwd/bwd
or `_phase_fix_split_ctm_tensor`) and fix it minimally; do NOT restructure the Neumann
backward. If it fails only on `rel` (magnitude, direction fine), the symmetric
explicit reference is under-converged — bump `warmup_steps`/`backprop_steps` to 60 or
tighten implicit `conv_tol` to `1e-14`.

- [ ] **Step 3: Add the Heisenberg companion (documents the degenerate-SV floor)**

Append:

```python
@pytest.mark.slow
def test_2site_symmetric_ad_heisenberg_floor():
    """Heisenberg (Delta=1) companion: symmetric implicit vs explicit gradient agree
    in DIRECTION but the magnitude is floored at ~5e-4 by the degenerate-SV SVD
    backward (a known limitation, NOT a bug — the machine-exact gate is XXZ Delta=0.3,
    see test_2site_symmetric_ad_matches_explicit_and_dense)."""
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit_2site,
    )

    A_d, B_d = _build_su_neel(D=2)
    A_s, B_s = _to_trivial_u1(A_d), _to_trivial_u1(B_d)
    gate = _heisenberg_gate()
    chi = 4

    def loss_imp(a):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B_s}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, max_iter=80, conv_tol=1e-13, min_iter=2,
        ).real

    def loss_exp(a):
        return ctm_energy_split_explicit_2site(
            {(0, 0): a, (1, 0): B_s}, CHECKERBOARD_NEIGHBORS, gate,
            chi=chi, chi_I=chi, warmup_steps=40, backprop_steps=40,
        ).real

    _, g_i = jax.value_and_grad(loss_imp)(A_s)
    _, g_e = jax.value_and_grad(loss_exp)(A_s)
    gi = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i)])
    ge = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_e)])
    cos = float(jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge)))
    rel = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))
    assert cos > 1 - 1e-6, f"direction should still agree: cos={cos}"
    assert rel < 5e-3, f"magnitude should be near the ~5e-4 floor, got rel={rel}"
```

- [ ] **Step 4: Run both AD tests + commit**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py -q -m slow`
Expected: PASS (both AD tests).

```bash
git add tests/test_split_ctm_2site_symmetric.py
git commit -m "test(#463): 2-site symmetric split AD parity — XXZ machine-exact + Heisenberg floor (Phase 3 Task 3)"
```

---

## Final verification

- [ ] **Run the full Phase-3 suite:**

Run: `uv run pytest tests/test_split_ctm_2site_symmetric.py -q`
Expected: PASS (all — smoke, 3 energy-parity params, 2 AD tests).

- [ ] **Guard against dense/single-site regression** (the source fix touched shared
  files):

Run: `uv run pytest tests/test_split_ctm_2site.py tests/test_split_ctm_2site_ad.py tests/test_split_ctm_tensor.py -q`
Expected: PASS (all pre-existing dense 2-site + single-site symmetric suites — the
`base_charges=None`→`_split_base_charges(A)` change is a no-op for DenseTensor, which
returns `None`).

- [ ] **Run the core marker (CI-required):**

Run: `uv run pytest -m core -q`
Expected: PASS.

- [ ] **Docs:** the new functions are all private (`_split_base_charges`) or tests —
  no public API change. Confirm no README/`__init__` update is needed:

Run: `grep -rn "_split_base_charges\|ctm_energy_split" README.md src/tenax/__init__.py`
Expected: no matches (these stay internal).

  Update the `#463` status memory ([[project_463_2site_forward_phase2_handoff]] /
  [[project_463_split_ctm_canonical]]) to mark Phase 3 complete (bosonic symmetric)
  and Phase 4 (fermionic) as next.

- [ ] **Open / update the PR** per CLAUDE.md workflow. This branch already hosts
  Phase 2 (PR #685, open); Phase 3 stacks on it. Push and either add commits to #685
  or open a stacked PR retargeted to `main` after #685 merges:

```bash
git push origin design/463-2site-split-ctm-ad
```

  Apply the `run-full-tests` label so the `slow` AD tests run in CI.

---

## Self-review notes (spec coverage)

- Spec "Core change: derive `base_charges` locally, not plumbed" → Task 1
  (`_split_base_charges` helper; driver + 4 absorbs derive locally; zero signature
  changes). ✅
- Spec §2 Tier-1/2 energy parity (trivial-U(1) wrap, D∈{2,3}, χ∈{4,8}, direction-
  dependent A≠B) → Task 2 (`_build_su_neel` gives A≠B via Néel bias; parametrized). ✅
- Spec §2 Tier-3 AD parity (symmetric implicit==explicit + symmetric==dense; XXZ
  Δ=0.3 machine-exact + Heisenberg floor companion) → Task 3. ✅
- Spec §2 nontrivial-charge structural smoke (finite + SymmetricTensor + sectors
  preserved; no convergent-charged oracle) → Task 1 smoke. ✅
- Spec "AD verified, not rewritten" → Task 3 expects PASS with no AD-code change;
  contingency only fixes a genuine dense assumption minimally. ✅
- Spec risk "hard fusion at D≥3" → Task 2 D=3 energy-parity param is the guard, with
  a targeted diagnosis note. ✅
- Spec risk "trivial-charge short-circuit" → Task 1 validation lesson #1 documents
  that trivial-U(1) is sector-independent, so the smoke (nontrivial) is the real
  `base_charges` gate. ✅
- Out of scope (fermionic, convergent charged state, chi-bump/schedule) → not in this
  plan; deferred to Phase 4. ✅

**Genuine execution-time risk:** the nontrivial-charge smoke's red-first depends on
the random state actually starving a sector under global truncation. Task 1 Step 2
gives the concrete tightening (raise `chi_I`, add a seed) if it is green-without-fix —
never implement against a green test. The trivial-U(1) energy/AD tests are expected
PASS-first (regression guards); their contingency notes cover the low-probability case
of a real dense-only assumption surviving on the 2plaq symmetric path.
```
