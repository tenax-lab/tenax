"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry, ZnSymmetry
from tenax.algorithms._ctm_root_implicit_symmetric import (
    SymEnv,
    all_projectors_sym,
    converge_sym,
    half_infinite_sym,
    init_env_sym,
    lower_left_quadrant_sym,
    rotate_a_sym,
    rotate_env_sym,
    swap_env_convention_sym,
    sweep_sym,
    sym_energy,
    upper_left_quadrant_sym,
)


def _site_tensor(seed: int = 0) -> SymmetricTensor:
    """A U(1) iPEPS site tensor with non-trivial charges on every leg.

    Non-trivial deliberately: a trivial-charge tensor has one block, so every
    layout bug is invisible.  The fused ``D**2`` leg comes out with sector
    multiplicities ``[1, 2, 1]`` — *unequal*, which is the fragmenting case
    and the only one that exercises the layout arithmetic.

    **Axis order is ``(u, d, l, r, phys)`` — physical leg last.**  That is the
    library-wide iPEPS convention (``_build_double_layer_tensor``'s docstring,
    ``_ctm_utils`` line 62, ``ipeps_optimize``, ``pess_optimize``, ...) and it
    is load-bearing here, not cosmetic.  ``_build_double_layer_tensor``
    contracts by *label* and so is order-blind, but ``_CORNER_SPECS`` and
    ``_STD_EDGE_SPECS`` name their ``ref_axis`` *positionally* (0 = u, 1 = d).
    With ``phys`` first, ``ref_axis=0`` lands on the physical leg, and every
    corner/edge chi leg that quotes it inherits ``phys (x) phys-bar`` charges
    instead of virtual ones.  Measured consequences of getting this wrong:

    - The environment stops being uniform.  With ``phys`` first, ``T4``'s two
      chi legs come out with charges ``[-2, 0, 0, 2]`` and ``[-1, 0, 0, 1]``,
      so the two half-infinite quadrants cannot be glued along the cut — the
      contraction that ``half_infinite_sym`` has to do is not merely
      awkward, it has no charge-consistent form.  With ``phys`` last every
      chi leg carries ``[-1, 0, 0, 1]`` and the cut closes.
    - At D=3 (virtual sectors ``[-1, 0, 1]``) ``phys`` first makes
      ``initialize_ctm_tensor_env`` raise ``ValueError: data.shape (4, 4, 4)
      does not match index dims (4, 9, 4)`` — ``ref_axis=0`` is the d=2
      physical leg while ``ref_axis=1`` is a D=3 virtual one.  That was read
      as issue #667; it is really just this axis order.

    **D=2 with virtual sectors [0, 1], and the physical leg labelled
    ``phys``.** Both are load-bearing and were measured, not guessed:

    - Non-trivial charges deliberately: a trivial-charge tensor has one
      block, so every layout bug is invisible.
    - With any physical label other than ``phys``,
      ``compute_energy_ctm_tensor`` raises ``ValueError: output_labels
      contains 'phys' which is not a free label``.
    """
    sym = U1Symmetry()
    phys = TensorIndex(
        symmetry=sym,
        sectors=np.array([-1, 1]),
        multiplicities=np.array([1, 1]),
        flow=FlowDirection.OUT,
        label="phys",
    )

    def virt(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([0, 1]),
            multiplicities=np.array([1, 1]),
            flow=flow,
            label=lbl,
        )

    return SymmetricTensor.random_normal_np(
        (
            virt(FlowDirection.IN, "u"),
            virt(FlowDirection.OUT, "d"),
            virt(FlowDirection.IN, "l"),
            virt(FlowDirection.OUT, "r"),
            phys,
        ),
        np.random.RandomState(seed),
    )


def test_init_env_sym_keeps_every_tensor_symmetric():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    assert isinstance(env, SymEnv)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        t = getattr(env, name)
        assert isinstance(t, SymmetricTensor), f"{name} was densified"
        assert t.n_blocks > 0
    assert isinstance(a, SymmetricTensor)


def test_swap_env_convention_sym_is_an_involution():
    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    twice = swap_env_convention_sym(swap_env_convention_sym(env))
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        lhs = getattr(twice, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name


def test_swap_env_convention_sym_is_not_identity_on_asymmetric_env():
    """``init_env_sym``'s C4 is symmetric and T3/T4 palindromic by
    construction, so the involution test above would pass even if
    ``swap_env_convention_sym`` returned its input unchanged.  Replace C4,
    T3, T4 with fresh random tensors on the *same* indices — still valid
    ``SymmetricTensor``s, but generically neither symmetric nor palindromic —
    and check the swap actually transposes them rather than no-op'ing.
    """
    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    env = env._replace(
        C4=SymmetricTensor.random_normal_np(env.C4.indices, np.random.RandomState(1)),
        T3=SymmetricTensor.random_normal_np(env.T3.indices, np.random.RandomState(2)),
        T4=SymmetricTensor.random_normal_np(env.T4.indices, np.random.RandomState(3)),
    )
    swapped = swap_env_convention_sym(env)

    # Not the identity: the swapped C4/T3/T4 differ from the originals.
    assert float(jnp.max(jnp.abs(swapped.C4.todense() - env.C4.todense()))) > 1e-6
    assert float(jnp.max(jnp.abs(swapped.T3.todense() - env.T3.todense()))) > 1e-6
    assert float(jnp.max(jnp.abs(swapped.T4.todense() - env.T4.todense()))) > 1e-6

    # And it is exactly the documented transpose, not some other change.
    assert (
        float(
            jnp.max(jnp.abs(swapped.C4.todense() - env.C4.transpose((1, 0)).todense()))
        )
        < 1e-14
    )
    assert (
        float(
            jnp.max(
                jnp.abs(swapped.T3.todense() - env.T3.transpose((2, 1, 0)).todense())
            )
        )
        < 1e-14
    )
    assert (
        float(
            jnp.max(
                jnp.abs(swapped.T4.todense() - env.T4.transpose((2, 1, 0)).todense())
            )
        )
        < 1e-14
    )

    # The untouched tensors are unaffected.
    for name in ("C1", "C2", "C3", "T1", "T2"):
        lhs = getattr(swapped, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name


def _dense_env_of(env):
    from tenax.algorithms._ctm_root_implicit_asym import AsymEnv

    return AsymEnv(
        *[
            jnp.asarray(getattr(env, n).todense())
            for n in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
        ]
    )


def test_upper_left_quadrant_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import _upper_left_quadrant

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = _upper_left_quadrant(_dense_env_of(env), jnp.asarray(a.todense()))
    got = upper_left_quadrant_sym(env, a)
    assert got.labels() == ("chi_r", "a_r", "chi_d", "a_d")
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-12


def test_lower_left_quadrant_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import _lower_left_quadrant

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = _lower_left_quadrant(_dense_env_of(env), jnp.asarray(a.todense()))
    got = lower_left_quadrant_sym(env, a)
    assert got.labels() == ("chi_u", "a_u", "chi_r", "a_r")
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-12


def test_half_infinite_sym_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import half_infinite_environment

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = half_infinite_environment(_dense_env_of(env), jnp.asarray(a.todense()))
    got = half_infinite_sym(env, a)
    assert got.ndim == 2
    dense = got.todense()
    assert dense.shape == expected.shape
    assert float(jnp.max(jnp.abs(dense - expected))) < 1e-10


def test_rotate_a_sym_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import rotate_a

    A = _site_tensor()
    _env, a = init_env_sym(A, chi=4)
    expected = rotate_a(jnp.asarray(a.todense()))
    got = rotate_a_sym(a)
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-14


def test_rotate_env_sym_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import rotate_env

    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    expected = rotate_env(_dense_env_of(env))
    got = rotate_env_sym(env)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        lhs = getattr(got, name).todense()
        rhs = getattr(expected, name)
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name


def _retained_norm(p):
    """``norm(s_k)`` — the norm of the *unnormalised* retained spectrum.

    This is the scalar the dense module divides ``S_keep`` by, and therefore
    the scalar its projector closure comes out as; see
    :func:`test_projector_closure_is_a_multiple_of_the_identity_per_sector`.
    """
    return float(
        jnp.sqrt(
            sum(
                jnp.sum(p.sectors[q].S_keep_diag[: p.layout.dim_of(q)] ** 2)
                for q in p.layout.sectors
            )
        )
    )


def test_projector_closure_is_a_multiple_of_the_identity_per_sector():
    """``P_right @ P_left`` is ``norm(s_k)`` times the identity, not the identity.

    The dense docstring says the pair "reconstructs the identity on the
    retained subspace by construction", and for a bare ``S = diag(s_k)`` it
    does.  ``_ctm_root_implicit_asym.all_projectors`` hands
    ``_fishman_projectors`` the *normalised* ``S_keep = diag(s_k)/norm(s_k)``
    instead, so what actually comes out is

        P_bot @ P_top = S^-1/2 (U† M Vh†) S^-1/2
                      = norm(s_k) diag(s_k)^-1/2 diag(s_k) diag(s_k)^-1/2
                      = norm(s_k) * 1 ,

    which the dense module reproduces exactly — 60.30, 53.25, 59.78, 56.74 on
    the four directions of this fixture, not 1.  Harmless there because every
    renormalised corner and edge is rescaled straight afterwards, but it is
    the reference behaviour, and this port matches it rather than quietly
    dividing the factor out: ``S`` is a *variable* of the characteristic
    equations, and a differently normalised ``S`` is a different root.

    So the assertion is closure == c * 1 with ``c`` pinned two ways — computed
    from the retained spectrum, and read off the dense module — which is
    strictly more than "closure == 1" would have said.
    """
    from tenax.algorithms._ctm_root_implicit_asym import all_projectors

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = all_projectors_sym(env, a, chi=4)
    dense_projs = all_projectors(_dense_env_of(env), jnp.asarray(a.todense()), chi=4)
    assert len(projs) == 4
    worst = 0.0
    for k in range(4):
        p = projs[k]
        c = _retained_norm(p)
        dense_closure = dense_projs[k][1] @ dense_projs[k][0]
        assert float(jnp.max(jnp.abs(dense_closure - c * jnp.eye(4)))) < 1e-9, k
        for q in p.layout.sectors:
            k_q = p.layout.dim_of(q)
            closure = p.P_right[q] @ p.P_left[q]
            eye = c * jnp.eye(k_q, dtype=closure.dtype)
            diff = float(jnp.max(jnp.abs(closure - eye)))
            worst = max(worst, diff)
            assert diff < 1e-9, (k, q, diff)
    print(f"closure max|P_right @ P_left - norm(s_k) * 1|: {worst:.3e}")


def test_retained_dimension_totals_chi():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    for p in all_projectors_sym(env, a, chi=4):
        assert p.layout.total == 4


def _dense_retained_projector(p, cut_index, cut_charge_of_q):
    """Scatter the block-diagonal ``P_left @ P_right`` into the dense cut basis.

    ``P_left[q] @ P_right[q]`` lives on sector ``q`` of the *cut* leg, which is
    the fused ``(chi, d2)`` bond the two quadrants are glued along.  A
    ``SymmetricTensor``'s ``todense()`` writes the block with charge ``c`` at
    ``np.where(index.charges == c)[0]`` — ascending, not contiguous — and
    ``fuse_indices`` keeps the parent legs' row-major ordering inside that
    charges array, so those positions are exactly the rows of the dense
    module's plain ``.reshape(chi * d2, chi * d2)``.  Concatenating the sector
    blocks in charge order would land them somewhere else entirely.
    """
    charges = np.asarray(cut_index.charges)
    n = len(charges)
    out = np.zeros((n, n), dtype=complex)
    for q in p.layout.sectors:
        pos = np.where(charges == cut_charge_of_q[q])[0]
        block = np.asarray(p.P_left[q] @ p.P_right[q])
        assert block.shape == (len(pos), len(pos)), (q, block.shape, len(pos))
        out[np.ix_(pos, pos)] = block
    return jnp.asarray(out)


def test_retained_subspace_projector_matches_the_dense_module():
    """P_left @ P_right is gauge invariant, unlike P_left and P_right alone.

    An SVD leaves one phase per retained index free and the sector layer picks
    a different one from the dense layer, so ``P_left`` and ``P_right`` are
    *not* comparable to ``P_top`` and ``P_bot`` element-wise.  Their product is:
    it is the oblique projector onto the retained subspace, invariant under
    ``U -> U W``, ``Vh -> W† Vh``, ``S -> W† S W`` for any in-space ``W`` —
    including the permutation that relates the dense module's globally sorted
    retained spectrum to this module's sector-grouped one, and any mixing of
    the singular vectors of the two degenerate cross-sector values this fixture
    has at ``chi = 4``.
    """
    from tenax.algorithms._ctm_root_implicit_asym import all_projectors
    from tenax.algorithms._ctm_root_implicit_symmetric import (
        _as_matrix_sym,
        rotate_a_sym,
        rotate_env_sym,
    )
    from tenax.linalg import _group_blocks_by_bond_charge

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    dense_env = _dense_env_of(env)
    a_dense = jnp.asarray(a.todense())

    sym_projs = all_projectors_sym(env, a, chi=4)
    dense_projs = all_projectors(dense_env, a_dense, chi=4)

    env_k, a_k = env, a
    worst = 0.0
    for k in range(4):
        P_top, P_bot = dense_projs[k][0], dense_projs[k][1]
        expected = P_top @ P_bot  # (n, n), gauge invariant

        # Rebuild the cut leg independently of the implementation: the lower
        # quadrant's fused axes 0-1 are the bond the projectors act on.
        bot = _as_matrix_sym(lower_left_quadrant_sym(env_k, a_k))
        cut_index = bot.indices[0]
        grouped = _group_blocks_by_bond_charge(bot, [0], [1])
        cut_charge_of_q = {q: int(e[0][0][0]) for q, e in grouped.items()}

        got = _dense_retained_projector(sym_projs[k], cut_index, cut_charge_of_q)
        diff = float(jnp.max(jnp.abs(got - expected)))
        worst = max(worst, diff)
        assert diff < 1e-9, (k, diff)

        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    print(f"retained-subspace projector max|diff| over 4 directions: {worst:.3e}")


def test_projectors_are_idempotent_and_rank_chi():
    """A sanity check on the assembled projector itself, not on the reference.

    ``P_left @ P_right`` must be a projector onto a ``chi_q``-dimensional
    subspace — up to the ``norm(s_k)`` scale the dense module builds in, see
    :func:`test_projector_closure_is_a_multiple_of_the_identity_per_sector`.
    That is what "projector onto the retained subspace" means, and it fails
    loudly if a sector's blocks are built from mismatched pieces even when the
    dense reference happens to agree by accident.
    """
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    for p in all_projectors_sym(env, a, chi=4):
        # Carrying the same ``norm(s_k)`` the closure does, so idempotence
        # reads ``P² = c P`` and the trace is ``c * chi_q``.
        c = _retained_norm(p)
        for q in p.layout.sectors:
            proj = p.P_left[q] @ p.P_right[q]
            assert float(jnp.max(jnp.abs(proj @ proj - c * proj))) < 1e-8 * c
            trace = float(jnp.trace(proj).real)
            assert abs(trace - c * p.layout.dim_of(q)) < 1e-8 * c


def test_S_is_a_matrix_not_a_vector_and_is_globally_normalised():
    """``S`` is a chi_q x chi_q matrix, and its normalisation is global.

    Per-sector normalisation would leave each sector's ``S`` unit-norm; what
    the dense module does is normalise the retained spectrum of the *whole*
    cut once, so it is the sum over sectors that comes to one.
    """
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    for p in all_projectors_sym(env, a, chi=4):
        total = 0.0
        for q in p.layout.sectors:
            k_q = p.layout.dim_of(q)
            assert p.S[q].shape == (k_q, k_q)
            total += float(jnp.sum(jnp.abs(p.S[q]) ** 2))
        assert abs(total - 1.0) < 1e-12


def test_warm_pinning_reproduces_the_previous_gauge():
    """Re-pinning against its own output is a fixed point of the gauge pin."""
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    first = all_projectors_sym(env, a, chi=4)
    again = all_projectors_sym(env, a, chi=4, prev=first)
    for k in range(4):
        for q in first[k].layout.sectors:
            lhs = again[k].P_left[q]
            rhs = first[k].P_left[q]
            assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-12, (k, q)


# ``(seed, chi)`` pairs whose truncation boundary is *not* degenerate.
#
# Cross-sector ties break the dense-vs-symmetric comparison for a reason that
# is nobody's bug: with ``s[chi-1] == s[chi]`` in two different charge sectors,
# "the top chi singular values" is not a well-defined set.  The dense SVD sees
# one scattered matrix and may return any mixture of the tied pair, while the
# sector layer decomposes each sector separately and has to commit to one of
# them, so the two retain genuinely different subspaces.  This fixture is
# degenerate at chi=2 for every seed and at chi=3 for half of them; the pairs
# below are the tie-free ones, measured rather than guessed.
#
# chi=1 is kept deliberately: it retains one charge sector and leaves the other
# two empty, which is the ``chi_q == 0`` case ``BondLayout`` drops.
_TIE_FREE = [(0, 1), (0, 4), (1, 1), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


@pytest.mark.parametrize(("seed", "chi"), _TIE_FREE)
def test_retained_subspace_projector_matches_dense_across_layouts(seed, chi):
    """The same gate over several site tensors and several truncations.

    Varying ``chi`` is what varies the *layout*: at chi=1 one sector keeps
    everything and two keep nothing, at chi=3 the split is uneven, at chi=4 it
    is 1/2/1.  Each distributes the retained block over different offsets of
    the fused cut leg, so an assembly that merely concatenated the sector
    blocks in charge order would survive at most one of them.

    Note what is *not* varied: the environment itself, which is always
    ``initialize_ctm_tensor_env``'s.  A generic environment would be a stronger
    test, and ``SymmetricTensor.random_normal_np`` on the same ``TensorIndex``
    objects does not provide one — such a tensor is a legal
    ``SymmetricTensor`` but not a legal CTM environment tensor, and
    ``contract`` then silently drops the part of the network the charges cannot
    carry (3.09 out of 8.81 on the very first ``T1 x a`` contraction), so the
    symmetric quadrant stops agreeing with the dense einsum before the
    projectors are even reached.  Generic-environment coverage needs the
    symmetric sweep, i.e. a later task.
    """
    from tenax.algorithms._ctm_root_implicit_asym import all_projectors
    from tenax.algorithms._ctm_root_implicit_symmetric import (
        _as_matrix_sym,
        rotate_a_sym,
        rotate_env_sym,
    )
    from tenax.linalg import _group_blocks_by_bond_charge

    A = _site_tensor(seed)
    env, a = init_env_sym(A, chi=chi)
    dense_env = _dense_env_of(env)
    a_dense = jnp.asarray(a.todense())

    sym_projs = all_projectors_sym(env, a, chi=chi)
    dense_projs = all_projectors(dense_env, a_dense, chi=chi)

    env_k, a_k = env, a
    worst = 0.0
    for k in range(4):
        assert sym_projs[k].layout.total == chi, (k, sym_projs[k].layout.dims)
        expected = dense_projs[k][0] @ dense_projs[k][1]
        bot = _as_matrix_sym(lower_left_quadrant_sym(env_k, a_k))
        grouped = _group_blocks_by_bond_charge(bot, [0], [1])
        cut_charge_of_q = {q: int(e[0][0][0]) for q, e in grouped.items()}
        got = _dense_retained_projector(sym_projs[k], bot.indices[0], cut_charge_of_q)
        scale = float(jnp.max(jnp.abs(expected)))
        rel = float(jnp.max(jnp.abs(got - expected))) / scale
        worst = max(worst, rel)
        assert rel < 1e-9, (k, rel)
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    print(f"seed={seed} chi={chi}: retained-projector max rel diff {worst:.3e}")


def test_a_sector_that_retains_nothing_is_dropped_not_kept_at_width_zero():
    """chi=1 keeps one charge of the cut and drops the other two outright."""
    A = _site_tensor()
    env, a = init_env_sym(A, chi=1)
    for p in all_projectors_sym(env, a, chi=1):
        assert p.layout.total == 1
        assert len(p.layout.sectors) == 1
        assert set(p.P_left) == set(p.layout.sectors)
        assert set(p.P_right) == set(p.layout.sectors)
        assert set(p.S) == set(p.layout.sectors)
        # ``sectors`` still carries every decomposed sector, retained or not:
        # the characteristic equations need the null-space blocks of the ones
        # that kept nothing just as much.
        assert len(p.sectors) >= len(p.layout.sectors)


# ---------------------------------------------------------------------------
# Sweep, convergence and forward energy
# ---------------------------------------------------------------------------


def _convergent_site_tensor(seed: int = 2, eps: float = 0.3) -> SymmetricTensor:
    """A Z2 site tensor whose CTM *converges*, for the tests that need a limit.

    :func:`_site_tensor` cannot be used here, and the reason is a property of
    the state, not of this port.  Its U(1) structure — virtual sectors ``[0,1]``
    against physical ``[-1,1]`` — leaves exactly **8** allowed blocks, each a
    single scalar, because every block has to emit ``qp = +-1`` through virtual
    charges bounded by 1.  Run the *dense*, finite-difference-validated
    ``_ctm_root_implicit_asym.converge`` on that manifold and it never settles:
    the element-wise residual sits at 1.0-1.9 for 200 sweeps while the energy
    drifts (0.31 -> 1.17 -> 2.32 at 60/100/200 sweeps).  That was measured over
    112 variants — every single- and pair-block boost at three noise levels,
    two seeds — and over U(1) at D=3 with 32 blocks: **zero** converged.  The
    same dense routine converges in 23 sweeps on the trivial-charge fixture of
    ``test_ctm_root_implicit_asym``, so this is the state, not the algorithm.
    (This is the repo's standing rule that CTM parity needs a convergent input;
    a random tensor gives an oracle that oscillates and numbers that mean
    nothing.)

    Z2 (``ZnSymmetry(2)``) at the same leg dimensions has 16 allowed blocks
    instead of 8, and with a dominant block it converges in 36 sweeps —
    *the same 36* the dense module takes, which is the sharpest available
    statement that the two are running the same algebra.  The boost is needed:
    a plain random Z2 tensor converges densely but not per sector, because a
    retained dimension that moves between sweeps forces a cold gauge pin and
    the sign then oscillates with period two.

    Z2 gives **equal** sector sizes on the fused ``D**2`` leg, which is what
    the design note asks a Z2 fixture for; the *unequal*, fragmenting layout is
    covered by :func:`_site_tensor` in
    :func:`test_symmetric_sweep_tracks_the_dense_sweep_on_the_u1_fixture`, which
    does not need a limit.  Z2 with virtual multiplicities ``[2,1]`` (D=3, fused
    ``[5,4]``) is convergent *and* fragmenting — measured, 23 sweeps symmetric
    and dense alike — but costs ~170 s against this fixture's ~17 s, essentially
    all of it block-sparse compile, and this file runs in the ``core`` CI gate.
    """
    sym = ZnSymmetry(2)
    phys = TensorIndex(
        symmetry=sym,
        sectors=np.array([0, 1]),
        multiplicities=np.array([1, 1]),
        flow=FlowDirection.OUT,
        label="phys",
    )

    def virt(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([0, 1]),
            multiplicities=np.array([1, 1]),
            flow=flow,
            label=lbl,
        )

    A = SymmetricTensor.random_normal_np(
        (
            virt(FlowDirection.IN, "u"),
            virt(FlowDirection.OUT, "d"),
            virt(FlowDirection.IN, "l"),
            virt(FlowDirection.OUT, "r"),
            phys,
        ),
        np.random.RandomState(seed),
    )
    blocks = {k: eps * v for k, v in A.blocks.items()}
    blocks[(0, 1, 1, 0, 0)] = blocks[(0, 1, 1, 0, 0)] + 1.0
    return SymmetricTensor._from_blocks_unchecked(blocks, A.indices)


def _as_dense_site(A):
    """``A`` as a ``DenseTensor`` on the same indices.

    ``asym_energy`` builds the open double layer from whatever site tensor it is
    handed, and ``contract`` refuses to mix a ``SymmetricTensor`` double layer
    with the ``DenseTensor`` environment the dense module produces.  The dense
    reference path therefore has to be dense end to end.
    """
    from tenax.core.tensor import DenseTensor

    return DenseTensor(jnp.asarray(A.todense()), A.indices)


def test_unrotate_index_sym_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import _unrotate_index
    from tenax.algorithms._ctm_root_implicit_symmetric import _unrotate_index_sym

    for slot in (1, 2, 3, 4):
        for k in range(4):
            assert _unrotate_index_sym(slot, k) == _unrotate_index(slot, k)


def test_normalize_sym_is_max_abs_not_frobenius():
    """The dense module divides by ``max|x|``; Frobenius would be a different root.

    The two do not differ by a constant — the ratio depends on how the weight is
    spread — so a Frobenius normaliser would rescale ``S`` differently on every
    sweep and every tensor.
    """
    from tenax.algorithms._ctm_root_implicit_symmetric import _normalize_sym

    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    for name in ("C1", "T1"):
        t = getattr(env, name)
        got = _normalize_sym(t).todense()
        expected = t.todense() / (jnp.max(jnp.abs(t.todense())) + 1e-300)
        assert float(jnp.max(jnp.abs(got - expected))) < 1e-14, name
        assert abs(float(jnp.max(jnp.abs(got))) - 1.0) < 1e-14, name


def test_symmetric_sweep_keeps_every_tensor_block_sparse():
    """Nothing collapses to a single dense block, to zero, or out of the class.

    The failure this guards against is silent: ``contract`` drops the part of a
    network the charges cannot carry rather than raising, so a mis-assembled
    projector shows up as an environment that is still a ``SymmetricTensor`` but
    has lost its charge structure.  ``n_blocks > 1`` on every tensor says the
    block structure survived a full sweep; a densified port would fail the type
    assertion first.
    """
    A = _convergent_site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = None
    for _ in range(3):
        env, projs = sweep_sym(env, a, 4, projs)
        assert isinstance(env, SymEnv)
        for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
            t = getattr(env, name)
            assert isinstance(t, SymmetricTensor), f"{name} was densified"
            assert t.n_blocks > 1, (name, t.n_blocks)
    assert len(projs) == 4


def test_symmetric_forward_energy_matches_the_dense_module():
    """The gate for the whole forward: same number, block-sparse or not."""
    import tenax
    from tenax.algorithms._ctm_root_implicit_asym import asym_energy, converge
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

    A = _convergent_site_tensor()
    gate = tenax.heisenberg_gate()

    env_s, _a_s, meta_s = converge_sym(A, chi=4, max_iter=60)
    e_sym = float(sym_energy(A, env_s, gate))

    env_d, _a_d, meta_d = converge(A, chi=4, max_iter=60)
    e_dense = float(
        asym_energy(_as_dense_site(A), env_d, initialize_ctm_tensor_env(A, 4), gate)
    )

    assert meta_s["converged"], meta_s
    assert meta_s["iters"] < 60, meta_s
    assert abs(e_sym - e_dense) < 1e-10, (e_sym, e_dense)
    # Same algebra means the same iteration count, not merely the same limit.
    assert meta_s["iters"] == meta_d["iters"], (meta_s, meta_d)
    print(
        f"e_sym={e_sym!r} e_dense={e_dense!r} diff={abs(e_sym - e_dense):.3e} "
        f"iters sym={meta_s['iters']} dense={meta_d['iters']}"
    )


def test_converge_sym_returns_the_projector_chain_that_built_it():
    """``return_projectors`` is a requirement of the root, not a diagnostic.

    The converged environment sits in the bond gauge of the chain that produced
    it; a cold re-pin fixes a *different* gauge and leaves ``y*`` describing an
    environment it was not extracted from.  So the chain has to come back, and
    it has to be the *last* one — re-pinning the returned projectors against
    themselves is a fixed point, which is what this checks.
    """
    A = _convergent_site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=60, return_projectors=True)
    assert meta["converged"], meta
    assert len(projs) == 4
    again = all_projectors_sym(env, a, chi=4, prev=projs)
    for k in range(4):
        assert again[k].layout.dims == projs[k].layout.dims, k
        for q in projs[k].layout.sectors:
            lhs = again[k].P_left[q]
            rhs = projs[k].P_left[q]
            assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-8, (k, q)


def test_symmetric_sweep_tracks_the_dense_sweep_on_the_u1_fixture():
    """Sweep-by-sweep parity on the *fragmenting* U(1) layout.

    :func:`_convergent_site_tensor` is Z2 because U(1) at these leg charges has
    no CTM limit to compare (see its docstring), but the U(1) fixture is the one
    with unequal ``[1, 2, 1]`` sectors on the cut, so it must not go untested.
    Convergence is not needed for that: from the *same* seed environment, the
    symmetric and dense sweeps are the same map, so their corner spectra and
    their energies have to agree at *every* sweep, converged or not.  That is a
    stronger statement than agreeing on a limit — it fails on the first sweep
    that mis-glues, instead of being rescued by the fixed point.

    The tensors themselves are *not* compared: the per-sector gauge pin picks a
    different phase from the dense global one, so only gauge-invariant
    quantities are meaningful. Corner singular values and the energy are.

    Two tolerances, because two things are being said.  The *last* sweep has to
    agree to machine precision — that is the real assertion.  The looser bound
    over all sweeps covers one measured transient: at sweep 2 this fixture's cut
    has an **exact cross-sector tie**, ``s[3] == s[4] == 0.2801102`` in different
    charge sectors, so "the top chi singular values" is not a well-defined set
    and the global dense SVD and the per-sector one legitimately retain
    different subspaces (the same phenomenon the ``_TIE_FREE`` table above
    records).  The gap reopens and the discrepancy falls 1.5e-7 -> 1.3e-8 ->
    3.1e-10 -> 1e-16 as it does.
    """
    import tenax
    from tenax.algorithms import _ctm_root_implicit_asym as dense
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

    A = _site_tensor()
    A_dense = _as_dense_site(A)
    gate = tenax.heisenberg_gate()
    template = initialize_ctm_tensor_env(A, 4)

    env_s, a_s = init_env_sym(A, 4)
    env_d, a_d = dense._init_env(A, 4)
    projs_s = projs_d = None
    worst_sv = worst_e = 0.0
    last_e = float("inf")
    for _ in range(5):
        env_s, projs_s = sweep_sym(env_s, a_s, 4, projs_s)
        env_d, projs_d = dense.sweep(env_d, a_d, 4, projs_d)
        for name in ("C1", "C2", "C3", "C4"):
            sv_s = jnp.linalg.svd(getattr(env_s, name).todense(), compute_uv=False)
            sv_d = jnp.linalg.svd(getattr(env_d, name), compute_uv=False)
            worst_sv = max(
                worst_sv, float(jnp.max(jnp.abs(jnp.sort(sv_s) - jnp.sort(sv_d))))
            )
        e_s = float(sym_energy(A, env_s, gate))
        e_d = float(dense.asym_energy(A_dense, env_d, template, gate))
        last_e = abs(e_s - e_d)
        worst_e = max(worst_e, last_e)
    assert worst_sv < 1e-12, worst_sv
    assert worst_e < 1e-6, worst_e
    assert last_e < 1e-11, last_e
    print(
        f"u1 sweep parity: corner-sv {worst_sv:.3e}, energy worst {worst_e:.3e}, "
        f"last sweep {last_e:.3e}"
    )


def test_to_ctm_env_sym_applies_the_convention_swap():
    """The #718 boundary: the swap is load-bearing and must not be dropped.

    This module closes the ring uniformly; ``CTMTensorEnv`` closes it with
    ``C4`` transposed and ``T3``/``T4`` reversed.  Reinterpreting one as the
    other moved the energy by 1.5% and produced a +-2.121e-3 per-bond
    antisymmetry.  It is a *no-op* on the initialiser — symmetric ``C4``,
    palindromic ``T3``/``T4`` — so the check has to be made on a swept
    environment, where the asymmetry is real.
    """
    import tenax
    from tenax.algorithms._ctm_root_implicit_symmetric import (
        _CTM_LABELS,
        _to_ctm_env_sym,
    )
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_init import CTMTensorEnv

    A = _convergent_site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = None
    for _ in range(3):
        env, projs = sweep_sym(env, a, 4, projs)

    swapped = swap_env_convention_sym(env)
    ctm = _to_ctm_env_sym(env)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        lhs = getattr(ctm, name).todense()
        rhs = getattr(swapped, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name

    # C4/T3/T4 really moved: a swept environment is genuinely asymmetric, so
    # the swap is not silently the identity the way it is on the initialiser.
    for name in ("C4", "T3", "T4"):
        lhs = getattr(ctm, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) > 1e-6, name

    # And the swap changes the energy: relabelling without it is a real bug,
    # not a cosmetic one.
    unswapped = CTMTensorEnv(
        **{
            name: getattr(env, name).relabels(
                dict(zip(getattr(env, name).labels(), labels, strict=True))
            )
            for name, labels in _CTM_LABELS.items()
        }
    )
    gate = tenax.heisenberg_gate()
    e_ok = float(sym_energy(A, env, gate))
    e_bad = float(compute_energy_ctm_tensor(A, unswapped, gate))
    assert abs(e_ok - e_bad) > 1e-6, (e_ok, e_bad)
    print(f"swap matters: e={e_ok:.12f} vs unswapped e={e_bad:.12f}")
