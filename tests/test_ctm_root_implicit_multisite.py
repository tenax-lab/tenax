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
# Forward sweep                                                      #
# ------------------------------------------------------------------ #
#
# Slow-bucket prune (#1149): the Phase-1 bridges (1x1 enlarged corner ==
# Phase 1 quadrant, 1x1 forward == Phase 1 energy, uniform 2x2 == 1x1, the
# assembled-cell observable, the 2x2 forward smoke run), the 1x1 root and FD
# cases, the cell-shift mutation cells (root residual, gradient, 1x1
# invisibility), the 1x1 gluing mutation and the gauge-dependence record were
# removed.  They were development scaffolding for #715 Phase 2: the kept 2x2
# FD tests contain the 1x1 cases, and a wrong Appendix F table fails them
# directly, which is what the mutation cells demonstrated.


def _gate(delta=1.0):
    import jax.numpy as jnp

    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


@pytest.mark.slow
def test_two_site_energy_is_smooth_on_a_non_uniform_cell():
    """#894, the load-bearing gauge check: the multisite two-site energy is a
    *smooth* function of the sites on a genuinely non-uniform 2x2, because its
    ring spans both adjacent cells' environments so the inter-cell bond gauge
    cancels.

    The old one-``A``-on-both-halves energy (``cell_energy_forward``) is NOT
    smooth there — it glues two chi bonds carrying independent gauges and jumps
    by ~1e-3 under an arbitrarily small change of ``A`` (measured on this issue).
    Asserting the old energy is rough is the regime assert: it proves the
    fixture actually exposes the inter-cell gauge, so a two-site energy that
    read smooth by accident could not pass.
    """
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.core.tensor import DenseTensor

    cell = _cell_2x2()
    gate, chi = _gate(delta=0.7), 4
    idx = {rc: A.indices for rc, A in cell.items()}
    base = {rc: jnp.asarray(A.todense()) for rc, A in cell.items()}
    rng = np.random.RandomState(0)
    dirs = {rc: jnp.asarray(rng.standard_normal(v.shape)) for rc, v in base.items()}
    kw = dict(max_iter=300, conv_tol=1e-12)

    def line(data):
        return {rc: DenseTensor(data[rc], idx[rc]) for rc in data}

    ts = (-2e-5, -1e-5, 0.0, 1e-5, 2e-5)

    def sample(energy_fn):
        vals = []
        for t in ts:
            c = line({rc: base[rc] + t * dirs[rc] for rc in base})
            vals.append(float(energy_fn(c, gate, chi, 2, 2, **kw)))
        second = max(abs(vals[i + 2] - 2 * vals[i + 1] + vals[i]) for i in range(3))
        spread = max(vals) - min(vals)
        return second, spread

    two_site_second, two_site_spread = sample(M.cell_two_site_energy_forward)
    old_second, _old_spread = sample(M.cell_energy_forward)

    # Regime assert: the old (gauge-dependent) energy really is rough on this
    # fixture, so the smoothness of the new one is meaningful and not vacuous.
    assert old_second > 1e-5, (
        f"the one-A-both-halves energy is smooth here (second diff "
        f"{old_second:.3e}) — the fixture does not expose the inter-cell gauge, "
        f"so this test cannot distinguish the two energies"
    )
    # The new energy is smooth: a C^2 f has second difference f''*h^2 ~ 1e-8
    # at h=1e-5, orders below the old energy's gauge jump.
    assert two_site_second < 1e-2 * old_second, (
        f"the multisite two-site energy is not smooth (second diff "
        f"{two_site_second:.3e}) against the old energy's {old_second:.3e} — the "
        f"inter-cell bond gauge is not cancelling"
    )
    # And it genuinely varies over the span (not a constant that is trivially
    # smooth): the directional derivative is resolved.
    assert two_site_spread > 1e-6, (
        f"the two-site energy is nearly constant over the span "
        f"({two_site_spread:.3e}); the smoothness assertion is then vacuous"
    )


@pytest.mark.slow
def test_two_site_energy_matches_the_production_multisite_energy():
    """#894: the multisite two-site energy is physically the right number, not
    merely smooth — it agrees with the production CTM path
    (``python_loop_ctm_converge`` + ``compute_energy_ctm_tensor_multisite``) on
    the same non-uniform 2x2 state.

    The two CTMs converge to gauge-equivalent but not bit-identical fixed points
    (different projector conventions truncate slightly differently at finite
    chi), so the agreement is at the ~few-e-3 finite-chi level the energy is
    known to differ across CTM implementations, not to machine precision.
    """
    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_energy import (
        compute_energy_ctm_tensor_multisite,
    )

    cell = _cell_2x2()
    gate, chi = _gate(delta=0.7), 4
    neighbors = M.cell_neighbors(2, 2)

    e_root = float(
        M.cell_two_site_energy_forward(
            cell, gate, chi, 2, 2, max_iter=300, conv_tol=1e-12
        )
    )

    prod_envs, _info = python_loop_ctm_converge(
        cell, neighbors, chi=chi, max_iter=300, conv_tol=1e-12
    )
    e_prod = float(
        compute_energy_ctm_tensor_multisite(cell, prod_envs, neighbors, gate)
    )
    assert abs(e_root - e_prod) < 5e-3, (
        f"multisite two-site energy {e_root:.8f} disagrees with the production "
        f"CTM energy {e_prod:.8f} beyond the cross-CTM finite-chi tolerance"
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


@pytest.fixture(scope="module")
def root_2x2():
    """One converge + root parametrisation of ``_cell_2x2()`` at chi=4, shared
    by the two characteristic-equation tests (slow-bucket prune, #1149: each
    used to converge it itself).

    Regime assert (was ``test_the_unit_cell_is_not_secretly_uniform``): if the
    four different tensors converged to four identical environments, every
    test built on ``_cell_2x2()`` -- these two and the 2x2 FD gates below --
    would be vacuous, since a wrong cell shift would read the same tensor
    either way.
    """
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M

    corners, edges, _meta, projs, a_by_cell = M.converge_multisite(
        _cell_2x2(), 4, 2, 2, max_iter=200, conv_tol=1e-12, return_projectors=True
    )
    ref = corners[(0, 0, 0)]
    spread = max(
        float(jnp.linalg.norm(corners[(0, r, c)] - ref))
        for r in range(2)
        for c in range(2)
        if (r, c) != (0, 0)
    )
    assert spread > 1e-3, f"cells are effectively identical (spread {spread:.3e})"
    root, residual = M.root_parametrize_multisite(
        corners, edges, a_by_cell, 4, 2, 2, prev_projs=projs, polish_steps=3
    )
    return root, residual, a_by_cell


@pytest.mark.slow
def test_characteristic_equations_vanish_at_a_2x2_root(root_2x2):
    """The Phase 2 gate: all 20 equations per cell hold at the converged root
    of a 2x2 cell of different tensors."""
    _root, residual, _a = root_2x2
    assert residual < 1e-11


@pytest.mark.slow
def test_every_block_of_F_vanishes_not_just_the_norm(root_2x2):
    """A single dominant block could hide four broken ones behind a small
    total, so check R_C, R_E, R_u, R_S and R_v separately."""
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M

    root, _r, a_by = root_2x2
    blocks = M.characteristic_residual_multisite(root.y, a_by, root, 4)
    for name, blk in zip(("R_C", "R_E", "R_u", "R_S", "R_v"), blocks):
        norm = float(jnp.sqrt(sum(jnp.sum(jnp.abs(v) ** 2) for v in blk.values())))
        assert norm < 1e-11, f"{name} = {norm:.3e}"


# ------------------------------------------------------------------ #
# Adjoint: gradient vs finite differences                            #
# ------------------------------------------------------------------ #

_SZ = None


def _sz():
    import jax.numpy as jnp

    return jnp.array([[0.5, 0.0], [0.0, -0.5]])


def _fd_parity(cell, nrows, ncols, chi=4, h=1e-5, seed=0, on_root_residual="raise"):
    """(AD directional derivative, FD directional derivative).

    ``on_root_residual`` is threaded through so a caller can deliberately
    construct a cell whose ``y*`` is *not* a root and still get the wrong
    gradient back to measure it (the #715 mutation cells did; they were
    removed in #1149), where a production caller wants the default hard
    failure.
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


def _fd_parity_energy(cell, nrows, ncols, chi=4, h=1e-5, seed=0):
    """(AD directional derivative, FD directional derivative) of the two-site
    *energy* objective (``gate=``), the ground-state objective the observable
    path (``_fd_parity``) cannot express off a 1x1 cell (#894)."""
    import jax.numpy as jnp

    import tenax.algorithms._ctm_root_implicit_multisite as M
    from tenax.core.tensor import DenseTensor

    gate = _gate(delta=0.7)
    idx = {rc: A.indices for rc, A in cell.items()}
    _value, grad = M.cell_root_implicit_energy_and_grad(
        cell, gate=gate, chi=chi, nrows=nrows, ncols=ncols
    )
    base = {rc: jnp.asarray(A.todense()) for rc, A in cell.items()}
    rng = np.random.RandomState(seed)
    dirs = {rc: jnp.asarray(rng.standard_normal(v.shape)) for rc, v in base.items()}

    def f(data):
        c = {rc: DenseTensor(v, idx[rc]) for rc, v in data.items()}
        return float(
            M.cell_two_site_energy_forward(
                c, gate, chi, nrows, ncols, max_iter=300, conv_tol=1e-12
            )
        )

    ad = float(sum(jnp.real(jnp.sum(grad[rc] * dirs[rc])) for rc in grad))
    fd = (
        f({rc: base[rc] + h * dirs[rc] for rc in base})
        - f({rc: base[rc] - h * dirs[rc] for rc in base})
    ) / (2 * h)
    return ad, fd


def test_op_and_gate_are_mutually_exclusive():
    """The objective is exactly one of a one-site ``op`` or a two-site ``gate``;
    passing both or neither is a caller error, not a silent default."""
    import tenax.algorithms._ctm_root_implicit_multisite as M

    cell = {(0, 0): _site_tensor()}
    with pytest.raises(ValueError):
        M.cell_root_implicit_energy_and_grad(cell, None, gate=None)
    with pytest.raises(ValueError):
        M.cell_root_implicit_energy_and_grad(cell, _sz(), gate=_gate())


@pytest.mark.slow
def test_two_site_energy_gradient_matches_fd_on_a_non_uniform_2x2():
    """#894, the ground-state gradient gate: ``dE/dA`` of the physical two-site
    energy FD-matches on a genuinely non-uniform 2x2 cell — the configuration
    the one-A-both-halves energy is gauge-dependent (non-differentiable) on, and
    the reason multisite root-implicit AD could not be wired before.  Four
    different tensors, so every inter-cell bond and Appendix F cell shift is
    live.  ~8.5 min: one 2x2 adjoint gradient plus two forward CTM converges.

    Measured: rel 3.5e-8 at h=1e-5 (E=0.0187, |g.v|=2.118) — the bar sits ~300x
    above it, robust to a different BLAS.
    """
    ad, fd = _fd_parity_energy(_cell_2x2(), 2, 2)
    rel = abs(ad - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-5, f"AD={ad!r} FD={fd!r} rel={rel:.3e}"


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
