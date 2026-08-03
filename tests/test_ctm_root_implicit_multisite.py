"""Multisite root implicit AD for asymmetric CTMRG (#715 Phase 2).

The index helpers below are the Appendix F shifted-cell assignment.  Getting
one wrong yields a silently wrong gradient — the #700 / #702 failure shape —
so each is pinned against a table transcribed from the authors' reference
implementation (``ImplicitDifferentiationPEPS.jl``,
``src/asymmetric/fixedpoints.jl``) rather than re-derived here.

Conventions: this port is 0-based in every index.  Julia's ``dir ∈ 1:4`` with
``mod1`` becomes ``k ∈ 0:3`` with ``%``; rows and columns likewise.
"""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_root_implicit_multisite import (
    above,
    above_left,
    left,
    left_projector,
    leftvec_invfroot_indices,
    next_coordinate,
    prev_coordinate,
    proj_sinv_indices,
    rightvec_invfroot_indices,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# Every (dir, row, col) on a 3x5 cell — big enough that a wrap-around and a
# two-step shift are distinguishable from each other and from a no-op.
NROWS, NCOLS = 3, 5
COORDS = list(itertools.product(range(4), range(NROWS), range(NCOLS)))


def _n(i, total):
    return (i + 1) % total


def _p(i, total):
    return (i - 1) % total


def _site_tensor(D=2, d=2, seed=42, eps=1.0):
    """Random 5-leg site tensor with trivial U(1) charges.

    Same construction as ``test_ctm_root_implicit_asym``; ``seed`` varies so a
    unit cell can be filled with genuinely different tensors, which is the
    only way a wrong cell shift shows up at all.
    """
    rng = np.random.RandomState(seed)
    data = eps * jax.numpy.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = data / (jax.numpy.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


# ------------------------------------------------------------------ #
# Ring walk                                                          #
# ------------------------------------------------------------------ #


def test_next_and_prev_coordinate_are_inverse():
    for co in COORDS:
        assert prev_coordinate(next_coordinate(co, NROWS, NCOLS), NROWS, NCOLS) == co
        assert next_coordinate(prev_coordinate(co, NROWS, NCOLS), NROWS, NCOLS) == co


def test_four_next_steps_return_to_the_start():
    """The ring closes: walking all four directions is the identity."""
    for co in COORDS:
        walked = co
        for _ in range(4):
            walked = next_coordinate(walked, NROWS, NCOLS)
        assert walked == co


def test_next_coordinate_matches_the_reference_table():
    r, c = 1, 2
    assert next_coordinate((0, r, c), NROWS, NCOLS) == (1, r, _n(c, NCOLS))
    assert next_coordinate((1, r, c), NROWS, NCOLS) == (2, _n(r, NROWS), c)
    assert next_coordinate((2, r, c), NROWS, NCOLS) == (3, r, _p(c, NCOLS))
    assert next_coordinate((3, r, c), NROWS, NCOLS) == (0, _p(r, NROWS), c)


# ------------------------------------------------------------------ #
# Shifted-cell assignment (Appendix F)                               #
# ------------------------------------------------------------------ #


def test_proj_sinv_indices_matches_the_reference_table():
    """S^-1 absorbed into the projectors: direction unchanged, cell steps
    *outward* along that direction."""
    r, c = 1, 2
    assert proj_sinv_indices((0, r, c), NROWS, NCOLS) == (0, _p(r, NROWS), c)
    assert proj_sinv_indices((1, r, c), NROWS, NCOLS) == (1, r, _n(c, NCOLS))
    assert proj_sinv_indices((2, r, c), NROWS, NCOLS) == (2, _n(r, NROWS), c)
    assert proj_sinv_indices((3, r, c), NROWS, NCOLS) == (3, r, _p(c, NCOLS))


def test_leftvec_invfroot_indices_matches_the_reference_table():
    """(s†s)^1/4 for the U isometry: direction steps *back* by one."""
    r, c = 1, 2
    assert leftvec_invfroot_indices((0, r, c), NROWS, NCOLS) == (
        3,
        _n(r, NROWS),
        _p(c, NCOLS),
    )
    assert leftvec_invfroot_indices((1, r, c), NROWS, NCOLS) == (
        0,
        _p(r, NROWS),
        _p(c, NCOLS),
    )
    assert leftvec_invfroot_indices((2, r, c), NROWS, NCOLS) == (
        1,
        _p(r, NROWS),
        _n(c, NCOLS),
    )
    assert leftvec_invfroot_indices((3, r, c), NROWS, NCOLS) == (
        2,
        _n(r, NROWS),
        _n(c, NCOLS),
    )


def test_rightvec_invfroot_indices_matches_the_reference_table():
    """(ss†)^1/4 for the V isometry: direction steps *forward* by one, and the
    cell by *two* — the only two-step shift in the whole assignment."""
    r, c = 1, 2
    assert rightvec_invfroot_indices((0, r, c), NROWS, NCOLS) == (
        1,
        r,
        _n(_n(c, NCOLS), NCOLS),
    )
    assert rightvec_invfroot_indices((1, r, c), NROWS, NCOLS) == (
        2,
        _n(_n(r, NROWS), NROWS),
        c,
    )
    assert rightvec_invfroot_indices((2, r, c), NROWS, NCOLS) == (
        3,
        r,
        _p(_p(c, NCOLS), NCOLS),
    )
    assert rightvec_invfroot_indices((3, r, c), NROWS, NCOLS) == (
        0,
        _p(_p(r, NROWS), NROWS),
        c,
    )


def test_enlarged_corner_neighbour_tables():
    """Corner / edge / projector positions relative to an enlarged corner."""
    r, c = 1, 2
    assert above_left((0, r, c), NROWS, NCOLS) == (0, _p(r, NROWS), _p(c, NCOLS))
    assert above_left((1, r, c), NROWS, NCOLS) == (1, _p(r, NROWS), _n(c, NCOLS))
    assert above_left((2, r, c), NROWS, NCOLS) == (2, _n(r, NROWS), _n(c, NCOLS))
    assert above_left((3, r, c), NROWS, NCOLS) == (3, _n(r, NROWS), _p(c, NCOLS))

    assert left((0, r, c), NROWS, NCOLS) == (3, r, _p(c, NCOLS))
    assert left((1, r, c), NROWS, NCOLS) == (0, _p(r, NROWS), c)
    assert left((2, r, c), NROWS, NCOLS) == (1, r, _n(c, NCOLS))
    assert left((3, r, c), NROWS, NCOLS) == (2, _n(r, NROWS), c)

    assert above((0, r, c), NROWS, NCOLS) == (0, _p(r, NROWS), c)
    assert above((1, r, c), NROWS, NCOLS) == (1, r, _n(c, NCOLS))
    assert above((2, r, c), NROWS, NCOLS) == (2, _n(r, NROWS), c)
    assert above((3, r, c), NROWS, NCOLS) == (3, r, _p(c, NCOLS))

    assert left_projector((0, r, c), NROWS, NCOLS) == (0, r, _p(c, NCOLS))
    assert left_projector((1, r, c), NROWS, NCOLS) == (1, _p(r, NROWS), c)
    assert left_projector((2, r, c), NROWS, NCOLS) == (2, r, _n(c, NCOLS))
    assert left_projector((3, r, c), NROWS, NCOLS) == (3, _n(r, NROWS), c)


# ------------------------------------------------------------------ #
# The link back to Phase 1                                           #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("k", range(4))
def test_at_one_by_one_the_cell_shifts_vanish(k):
    """A 1x1 unit cell has only one cell, so every (r, c) shift collapses and
    the assignment must reduce to the bare direction offsets Phase 1 uses."""
    assert proj_sinv_indices((k, 0, 0), 1, 1) == (k, 0, 0)
    assert leftvec_invfroot_indices((k, 0, 0), 1, 1) == ((k - 1) % 4, 0, 0)
    assert rightvec_invfroot_indices((k, 0, 0), 1, 1) == ((k + 1) % 4, 0, 0)


def test_at_one_by_one_the_quartic_roots_are_the_phase_one_neighbours():
    """``_covariant_pieces`` in the 1x1 module reads ``K_L[k-1]`` and
    ``K_R[k+1]``.  That is this table at ``nrows = ncols = 1`` — the two must
    not drift apart."""
    for k in range(4):
        assert leftvec_invfroot_indices((k, 0, 0), 1, 1)[0] == (k - 1) % 4
        assert rightvec_invfroot_indices((k, 0, 0), 1, 1)[0] == (k + 1) % 4


def test_shifts_are_genuinely_periodic():
    """Every helper must land inside the cell for every input."""
    helpers = (
        proj_sinv_indices,
        leftvec_invfroot_indices,
        rightvec_invfroot_indices,
        above_left,
        left,
        above,
        left_projector,
        next_coordinate,
        prev_coordinate,
    )
    for helper in helpers:
        for co in COORDS:
            k, r, c = helper(co, NROWS, NCOLS)
            assert 0 <= k < 4
            assert 0 <= r < NROWS
            assert 0 <= c < NCOLS


# ------------------------------------------------------------------ #
# Enlarged corners: the bridge to Phase 1                            #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_enlarged_corner_at_1x1_reproduces_the_phase1_quadrant():
    """A 1x1 unit cell must reproduce Phase 1's rotate-and-reuse quadrant.

    Phase 1 builds the upper-left quadrant from ``(C1, T1, T4)`` of an
    environment rotated ``k`` times.  This module reads the same three tensors
    by coordinate — ``above_left``, ``above``, ``left`` — from an unrotated
    cell.  If the tables are right the two agree to machine precision, and
    that equality is what licenses reusing every downstream Phase 1 formula.
    """
    import jax.numpy as jnp

    from tenax.algorithms._ctm_root_implicit_asym import (
        _init_env,
        _upper_left_quadrant,
        rotate_a,
        rotate_env,
        sweep,
    )
    from tenax.algorithms._ctm_root_implicit_multisite import (
        enlarged_corner,
        env_to_cell_maps,
    )

    A = _site_tensor()
    chi = 4
    env, a = _init_env(A, chi)
    # Give the environment genuine asymmetry, or the test passes on symmetry
    # alone and says nothing about the index tables.
    env, projs = sweep(env, a, chi)
    env, _ = sweep(env, a, chi, projs)

    corners, edges = env_to_cell_maps(env)
    env_k, a_k = env, a
    for k in range(4):
        want = _upper_left_quadrant(env_k, a_k)
        got = enlarged_corner(corners, edges, {(0, 0): a}, (k, 0, 0), 1, 1)
        assert got.shape == want.shape
        err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert err < 1e-13, f"direction {k}: relative error {err:.3e}"
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)


# ------------------------------------------------------------------ #
# Forward sweep                                                      #
# ------------------------------------------------------------------ #


def _gate(delta=1.0):
    import jax.numpy as jnp

    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


@pytest.mark.slow
def test_multisite_forward_at_1x1_matches_the_phase1_energy():
    """The 1x1 smoke test for the forward sweep.

    Phase 1 truncates with the left half-plane, this module with the upper
    half; ``docs/plans/2026-07-31-715-phase2-multisite-design.md`` shows those
    are the same truncation up to a shift and a transpose.  So the two
    environments are *not* tensor-equal, and only a gauge-invariant quantity
    compares — the energy.
    """
    import jax.numpy as jnp

    from tenax.algorithms._ctm_root_implicit_asym import asym_energy, converge
    from tenax.algorithms._ctm_root_implicit_multisite import (
        cell_maps_to_env,
        converge_multisite,
    )
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

    A = _site_tensor()
    chi, gate = 4, _gate()
    template = initialize_ctm_tensor_env(A, chi)

    env1, _a1, meta1 = converge(A, chi, max_iter=200, conv_tol=1e-12)
    E1 = float(asym_energy(A, env1, template, gate))

    corners, edges, meta2 = converge_multisite(
        {(0, 0): A}, chi, 1, 1, max_iter=200, conv_tol=1e-12
    )
    E2 = float(asym_energy(A, cell_maps_to_env(corners, edges), template, gate))

    assert meta2["converged"], f"multisite forward did not converge: {meta2}"
    assert jnp.isfinite(E2)
    rel = abs(E1 - E2) / abs(E1)
    assert rel < 1e-10, f"E_phase1={E1!r} E_multisite={E2!r} rel={rel:.3e}"


@pytest.mark.slow
def test_multisite_forward_runs_on_a_2x2_cell_of_different_tensors():
    """A 2x2 cell of *different* tensors — the configuration a wrong cell
    shift can actually be seen in.  Only asserts the forward is well formed;
    the gradient gate comes with the characteristic equations."""
    import jax.numpy as jnp

    from tenax.algorithms._ctm_root_implicit_multisite import converge_multisite

    chi = 4
    cell = {
        (0, 0): _site_tensor(seed=1),
        (0, 1): _site_tensor(seed=2),
        (1, 0): _site_tensor(seed=3),
        (1, 1): _site_tensor(seed=4),
    }
    corners, edges, meta = converge_multisite(
        cell, chi, 2, 2, max_iter=200, conv_tol=1e-12
    )

    assert len(corners) == 4 * 2 * 2
    assert len(edges) == 4 * 2 * 2
    for co, C in corners.items():
        assert C.shape == (chi, chi), co
        assert jnp.all(jnp.isfinite(C)), co
    for co, T in edges.items():
        assert T.shape[0] == chi and T.shape[2] == chi, co
        assert jnp.all(jnp.isfinite(T)), co
    assert meta["converged"], meta


@pytest.mark.slow
def test_the_unit_cell_is_not_secretly_uniform():
    """Guards the guard: if a 2x2 cell of different tensors converged to four
    identical environments, every cell-shift test built on it would be
    vacuous — a wrong shift would read the same tensor either way."""
    import jax.numpy as jnp

    from tenax.algorithms._ctm_root_implicit_multisite import converge_multisite

    cell = {
        (0, 0): _site_tensor(seed=1),
        (0, 1): _site_tensor(seed=2),
        (1, 0): _site_tensor(seed=3),
        (1, 1): _site_tensor(seed=4),
    }
    corners, _edges, _meta = converge_multisite(
        cell, 4, 2, 2, max_iter=200, conv_tol=1e-12
    )
    ref = corners[(0, 0, 0)]
    spread = max(
        float(jnp.linalg.norm(corners[(0, r, c)] - ref))
        for r in range(2)
        for c in range(2)
        if (r, c) != (0, 0)
    )
    assert spread > 1e-3, f"cells are effectively identical (spread {spread:.3e})"


@pytest.mark.slow
def test_the_1x1_energy_gate_is_load_bearing(monkeypatch):
    """Guards the gate: break the gluing partner and the 1x1 energy must move.

    The upper-half cut glues ``EC[co]`` to ``EC[next_coordinate(co)]``.  If the
    forward were somehow insensitive to that choice, the agreement above would
    be proving nothing.  Composing ``next`` with itself picks the wrong
    neighbour, and the energy shifts by ~6e-3 relative — three orders above the
    1e-10 gate and thirteen above the 1e-15 the correct wiring achieves.
    """
    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.algorithms._ctm_root_implicit_asym import asym_energy, converge
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

    A = _site_tensor()
    chi, gate = 4, _gate()
    template = initialize_ctm_tensor_env(A, chi)
    env1, _a1, _m1 = converge(A, chi, max_iter=200, conv_tol=1e-12)
    E1 = float(asym_energy(A, env1, template, gate))

    good = M.next_coordinate
    monkeypatch.setattr(
        M, "next_coordinate", lambda co, nr, nc: good(good(co, nr, nc), nr, nc)
    )
    corners, edges, _m = M.converge_multisite(
        {(0, 0): A}, chi, 1, 1, max_iter=60, conv_tol=1e-12
    )
    E_wrong = float(asym_energy(A, M.cell_maps_to_env(corners, edges), template, gate))

    assert abs(E1 - E_wrong) / abs(E1) > 1e-4, (
        "wrong gluing partner left the energy unchanged, so the 1x1 gate "
        f"cannot detect a miswiring (E={E_wrong!r})"
    )


# ------------------------------------------------------------------ #
# Characteristic equations                                           #
# ------------------------------------------------------------------ #


def _cell_2x2():
    """Four *different* site tensors — the only configuration in which a wrong
    cell shift is observable at all."""
    return {
        (0, 0): _site_tensor(seed=1),
        (0, 1): _site_tensor(seed=2),
        (1, 0): _site_tensor(seed=3),
        (1, 1): _site_tensor(seed=4),
    }


def _root_residual(cell, nrows, ncols, chi=4, polish_steps=3):
    import tenax.algorithms._ctm_root_implicit_multisite as M

    corners, edges, _meta, projs, a_by_cell = M.converge_multisite(
        cell, chi, nrows, ncols, max_iter=200, conv_tol=1e-12, return_projectors=True
    )
    _root, residual = M.root_parametrize_multisite(
        corners,
        edges,
        a_by_cell,
        chi,
        nrows,
        ncols,
        prev_projs=projs,
        polish_steps=polish_steps,
    )
    return residual


@pytest.mark.slow
def test_characteristic_equations_vanish_at_a_1x1_root():
    assert _root_residual({(0, 0): _site_tensor()}, 1, 1) < 1e-11


@pytest.mark.slow
def test_characteristic_equations_vanish_at_a_2x2_root():
    """The Phase 2 gate: all 20 equations per cell hold at the converged root
    of a 2x2 cell of different tensors."""
    assert _root_residual(_cell_2x2(), 2, 2) < 1e-11


@pytest.mark.slow
def test_every_block_of_F_vanishes_not_just_the_norm():
    """A single dominant block could hide four broken ones behind a small
    total, so check R_C, R_E, R_u, R_S and R_v separately."""
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M

    corners, edges, _m, projs, a_by = M.converge_multisite(
        _cell_2x2(), 4, 2, 2, max_iter=200, conv_tol=1e-12, return_projectors=True
    )
    root, _r = M.root_parametrize_multisite(
        corners, edges, a_by, 4, 2, 2, prev_projs=projs, polish_steps=3
    )
    blocks = M.characteristic_residual_multisite(root.y, a_by, root, 4)
    for name, blk in zip(("R_C", "R_E", "R_u", "R_S", "R_v"), blocks):
        norm = float(jnp.sqrt(sum(jnp.sum(jnp.abs(v) ** 2) for v in blk.values())))
        assert norm < 1e-11, f"{name} = {norm:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "table",
    ["proj_sinv_indices", "leftvec_invfroot_indices", "rightvec_invfroot_indices"],
)
def test_each_cell_shift_table_is_load_bearing(monkeypatch, table):
    """Perturb one table's *cell* component and the 2x2 root must stop being a
    root.  Without this, ``‖F(y*)‖ ~ 1e-13`` says only that the equations are
    self-consistent, not that Appendix F was transcribed correctly.

    The direction component is left alone so the failure is attributable to the
    ``(r, c)`` assignment specifically.  Measured: 4.7e-13 -> ~1, twelve orders.
    """
    import tenax.algorithms._ctm_root_implicit_multisite as M

    good = getattr(M, table)
    monkeypatch.setattr(
        M,
        table,
        lambda co, nr, nc: (
            good(co, nr, nc)[0],
            (good(co, nr, nc)[1] + 1) % nr,
            good(co, nr, nc)[2],
        ),
    )
    assert _root_residual(_cell_2x2(), 2, 2) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize(
    "table",
    ["proj_sinv_indices", "leftvec_invfroot_indices", "rightvec_invfroot_indices"],
)
def test_at_1x1_the_cell_shifts_are_invisible(monkeypatch, table):
    """The other half of the argument, and the reason the 2x2 gate exists.

    A 1x1 cell has one cell, so every ``(r, c)`` shift is the identity and the
    residual is *bit-identical* under the same perturbation that costs twelve
    orders at 2x2.  Any Phase 2 test written only at 1x1 verifies nothing about
    Appendix F.
    """
    import tenax.algorithms._ctm_root_implicit_multisite as M

    cell = {(0, 0): _site_tensor()}
    before = _root_residual(cell, 1, 1)

    good = getattr(M, table)
    monkeypatch.setattr(
        M,
        table,
        lambda co, nr, nc: (
            good(co, nr, nc)[0],
            (good(co, nr, nc)[1] + 1) % nr,
            good(co, nr, nc)[2],
        ),
    )
    assert _root_residual(cell, 1, 1) == before


# ------------------------------------------------------------------ #
# Adjoint: gradient vs finite differences                            #
# ------------------------------------------------------------------ #

_SZ = None


def _sz():
    import jax.numpy as jnp

    return jnp.array([[0.5, 0.0], [0.0, -0.5]])


def _fd_parity(cell, nrows, ncols, chi=4, h=1e-5, seed=0, on_root_residual="raise"):
    """(AD directional derivative, FD directional derivative).

    ``on_root_residual`` is threaded through because the mutation tests below
    deliberately construct a cell whose ``y*`` is *not* a root; they need the
    wrong gradient returned so they can measure how wrong it is, where a
    production caller wants the default hard failure.
    """
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.core.tensor import DenseTensor

    op = _sz()
    idx = {rc: A.indices for rc, A in cell.items()}
    _value, grad = M.cell_root_implicit_energy_and_grad(
        cell,
        op,
        chi=chi,
        nrows=nrows,
        ncols=ncols,
        on_root_residual=on_root_residual,
    )
    base = {rc: jnp.asarray(A.todense()) for rc, A in cell.items()}
    rng = np.random.RandomState(seed)
    dirs = {rc: jnp.asarray(rng.standard_normal(v.shape)) for rc, v in base.items()}

    def f(data):
        c = {rc: DenseTensor(v, idx[rc]) for rc, v in data.items()}
        return float(
            M.cell_observable_forward(
                c, op, chi, nrows, ncols, max_iter=300, conv_tol=1e-12
            )
        )

    ad = float(sum(jnp.real(jnp.sum(grad[rc] * dirs[rc])) for rc in grad))
    fd = (
        f({rc: base[rc] + h * dirs[rc] for rc in base})
        - f({rc: base[rc] - h * dirs[rc] for rc in base})
    ) / (2 * h)
    return ad, fd


@pytest.mark.slow
def test_gradient_matches_finite_differences_at_1x1():
    ad, fd = _fd_parity({(0, 0): _site_tensor()}, 1, 1)
    rel = abs(ad - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-8, f"AD={ad!r} FD={fd!r} rel={rel:.3e}"


@pytest.mark.slow
def test_gradient_matches_finite_differences_on_a_2x2_cell():
    """The Phase 2 gate (#715).

    Four different site tensors, so every Appendix F cell shift is live.
    Measured h-convergence at 1e-4 / 1e-5 / 1e-6: 6.5e-6, 6.5e-8, 5.6e-10 —
    clean second-order behaviour for a central difference, which is what says
    the gradient is right rather than merely close.
    """
    ad, fd = _fd_parity(_cell_2x2(), 2, 2)
    rel = abs(ad - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-6, f"AD={ad!r} FD={fd!r} rel={rel:.3e}"


def test_an_unconverged_root_raises_by_default(monkeypatch):
    """A non-vanishing ``‖F(y*)‖`` must be a hard failure, not a warning.

    The gradient solves the adjoint of equations ``y*`` does not satisfy, so it
    comes back finite, plausibly scaled and silently wrong (paper Fig. 1).  An
    unattended optimizer would keep stepping on it.  Nothing downstream can
    detect that, so the default has to be loud.

    Driven by making the *tolerance* impossible rather than by breaking the
    physics, so the test stays fast and independent of the cell-shift tables.
    """
    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.algorithms._ad_primitives import RootResidualError

    with pytest.raises(RootResidualError) as excinfo:
        M.cell_root_implicit_energy_and_grad(
            {(0, 0): _site_tensor()},
            _sz(),
            chi=4,
            nrows=1,
            ncols=1,
            root_residual_warn=0.0,
        )
    assert excinfo.value.residual >= 0.0
    assert excinfo.value.tolerance == 0.0


def test_the_residual_policy_is_validated_before_the_expensive_part():
    """A typo must fail immediately, not after a full CTM convergence."""
    import tenax.algorithms._ctm_root_implicit_multisite as M

    with pytest.raises(ValueError, match="on_root_residual"):
        M.cell_root_implicit_energy_and_grad(
            {(0, 0): _site_tensor()},
            _sz(),
            chi=4,
            nrows=1,
            ncols=1,
            on_root_residual="warm",  # not "warn"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "table",
    ["proj_sinv_indices", "leftvec_invfroot_indices", "rightvec_invfroot_indices"],
)
def test_a_wrong_cell_shift_breaks_the_2x2_gradient(monkeypatch, table):
    """The gate #715 asks for by name: the tables must be load-bearing *for the
    gradient*, not only for the root residual."""
    import tenax.algorithms._ctm_root_implicit_multisite as M

    good = getattr(M, table)
    monkeypatch.setattr(
        M,
        table,
        lambda co, nr, nc: (
            good(co, nr, nc)[0],
            (good(co, nr, nc)[1] + 1) % nr,
            good(co, nr, nc)[2],
        ),
    )
    ad, fd = _fd_parity(_cell_2x2(), 2, 2, on_root_residual="warn")
    rel = abs(ad - fd) / max(abs(fd), 1e-30)
    assert rel > 1e-3, f"wrong {table} still matched FD (rel={rel:.3e})"


@pytest.mark.slow
def test_the_two_site_energy_is_gauge_dependent_on_a_nonuniform_cell():
    """Records why the parity objective is a one-site observable.

    ``compute_energy_ctm_tensor`` builds two-site RDMs with one site's ring.
    On a cell of different tensors the halves meet on independently-gauged chi
    bonds, so the scalar jumps under arbitrarily small changes of ``A`` even
    though every fixed point converges to ~1e-13.  A gauge-dependent scalar has
    no derivative, which is why finite differences diverged as h -> 0 before
    the objective was changed.
    """
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.core.tensor import DenseTensor

    cell = _cell_2x2()
    idx = {rc: A.indices for rc, A in cell.items()}
    base = {rc: jnp.asarray(A.todense()) for rc, A in cell.items()}
    rng = np.random.RandomState(0)
    dirs = {rc: jnp.asarray(rng.standard_normal(v.shape)) for rc, v in base.items()}

    def energy(t):
        c = {rc: DenseTensor(base[rc] + t * dirs[rc], idx[rc]) for rc in base}
        return float(
            M.cell_energy_forward(c, _gate(), 4, 2, 2, max_iter=300, conv_tol=1e-12)
        )

    vals = [energy(t) for t in (-2e-5, -1e-5, 0.0, 1e-5, 2e-5)]
    spread = max(vals) - min(vals)
    assert spread > 1e-3, (
        "two-site energy looks smooth on a non-uniform cell; if that is now "
        f"true the one-site objective may no longer be necessary (spread {spread:.3e})"
    )

    # The one-site observable over the same span is smooth.
    def obs(t):
        c = {rc: DenseTensor(base[rc] + t * dirs[rc], idx[rc]) for rc in base}
        return float(
            M.cell_observable_forward(c, _sz(), 4, 2, 2, max_iter=300, conv_tol=1e-12)
        )

    o = [obs(t) for t in (-2e-5, -1e-5, 0.0, 1e-5, 2e-5)]
    second = max(abs(o[i + 2] - 2 * o[i + 1] + o[i]) for i in range(3))
    # A smooth f has second difference f''·h² — here h = 1e-5 and f'' ~ 75, so
    # ~7.5e-9 is the *expected* value, not a defect.  What distinguishes the
    # two objectives is scale: the one-site curvature term is seven orders
    # below the two-site energy's spread over the same span.
    assert second < 1e-4 * spread, (
        f"one-site observable is not smooth either (second difference {second:.3e} "
        f"vs two-site spread {spread:.3e})"
    )
