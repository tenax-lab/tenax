"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry, ZnSymmetry
from tenax.algorithms._ctm_root_implicit_symmetric import (
    SymEnv,
    SymRoot,
    _double_layer_sym,
    _same_block_structure,
    absorb_inverse_roots_sym,
    all_projectors_sym,
    apply_bond_matrix,
    characteristic_residual_sym,
    converge_sym,
    half_infinite_sym,
    init_env_sym,
    lower_left_quadrant_sym,
    remove_inverse_roots_sym,
    root_parametrize_sym,
    rotate_a_sym,
    rotate_env_sym,
    swap_env_convention_sym,
    sweep_sym,
    sym_energy,
    sym_root_implicit_energy_and_grad,
    sym_root_to_covariant_convention,
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
    symmetric sweep, i.e. a later task — now
    :func:`test_the_projectors_match_the_dense_module_on_a_swept_u1_environment`.
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


def _truncation_gap(env_k: SymEnv, a_k: SymmetricTensor, chi: int) -> float | None:
    """The relative gap at the truncation boundary, computed the way it is used.

    ``sector_svd`` (``_ctm_root_implicit_sym_sectors.py``) ranks the *union*
    of every charge sector's full singular-value spectrum, descending, and
    keeps the top ``chi``.  This recomputes that same ranked union — not an
    approximation of it — and returns ``(s[chi-1] - s[chi]) / s[0]``: how far
    the last retained value sits above the first discarded one, relative to
    the cut's largest singular value.  ``None`` when the union has at most
    ``chi`` entries, i.e. the cut does not truncate anything and the retained
    set is unambiguous by construction (no ``s[chi]`` exists to be close to).
    """
    from tenax.algorithms._ctm_root_implicit_sym_sectors import sector_svd

    M = half_infinite_sym(env_k, a_k)
    svds, _layout = sector_svd(M, chi, row_axis=0, col_axis=1)
    merged = np.sort(np.concatenate([np.asarray(v.s) for v in svds.values()]))[::-1]
    if len(merged) <= chi:
        return None
    return float((merged[chi - 1] - merged[chi]) / merged[0])


# Below this, the truncation boundary is ambiguous rather than merely close:
# ``sector_svd`` decomposes each charge sector separately and then has to
# commit one of a near-tied pair straddling the cut to "retained" and the
# other to "discarded", while the dense module's one global SVD sees a single
# scattered matrix and can legitimately commit the other way.  Both are
# correct SVDs of the same matrix; "the top chi singular values" simply is
# not a well-defined set when two of them coincide to numerical precision, so
# the two layers can and do disagree on *which subspace* they retain — a
# LAPACK-build-dependent choice, not a bug in either layer.
#
# The CI failure this guards against measured a relative gap of 2.825e-4 at
# the truncation boundary of the failing (sweep, direction) pair; the
# smallest gap among the pairs that were NOT failing on that same run was
# 1.718e-3 — only 6x larger, i.e. within the same order of magnitude as the
# failure and not a safe place to draw the line, since this fixture's sweep
# is deliberately non-convergent (see ``_convergent_site_tensor``) and a
# near-tie a platform resolves differently shifts every later sweep's
# environment, not just that one comparison.
#
# 1e-2 sits roughly two orders of magnitude above the measured failure.
# Recomputing this same quantity locally for all 6 sweeps x 4 directions (24
# pairs total) puts 12 of them above 1e-2 — spanning every layout shape this
# trajectory visits (one-sector, three-sector, and every direction) — so the
# threshold buys a large margin over the failure without emptying the test.
_TRUNCATION_GAP_THRESHOLD = 1e-2

# However many (sweep, direction) pairs a given LAPACK build's near-ties
# leave above the threshold, the test must not be satisfied by checking one
# or two of them.  12/24 were measured locally; 8 leaves a third of that as
# slack for a platform whose chaotic trajectory happens to sit closer to the
# threshold more often, while still forcing the test to cover multiple
# sweeps and directions rather than degenerating into a no-op.
_MIN_COMPARISONS = 8


def test_the_projectors_match_the_dense_module_on_a_swept_u1_environment():
    """The fragmenting layout, checked on *generic* environments, sweep by sweep.

    :func:`test_retained_subspace_projector_matches_dense_across_layouts` pins
    the same quantity but only ever on ``initialize_ctm_tensor_env``'s
    environment, and says so: "Generic-environment coverage needs the symmetric
    sweep, i.e. a later task."  This is that check.  A swept U(1) environment is
    the case the initialiser cannot produce — its corner and edge chi legs carry
    whatever charges the *previous* truncation chose rather than the uniform
    ones the initialiser stamps — and it is where the layout arithmetic actually
    has to work rather than happening to be uniform.  The sweep is also the
    *only* way to get one: as that docstring records, a random
    ``SymmetricTensor`` on the same indices is a legal tensor but not a legal
    CTM environment, and the symmetric and dense quadrants stop agreeing before
    the projectors are reached.

    What makes it worth its 40 s is that the layout genuinely moves.  Measured
    over these six sweeps, direction 0 goes

        {-1:1, 0:2, 1:1} -> {0:3, 1:1} -> {-1:1, 0:3} -> {0:4}

    while directions 1 and 2 sit at ``{-1:1, 0:2, 1:1}`` throughout — so within
    one comparison the test covers three-sector and one-sector cuts, sectors
    that empty out, sectors that grow, and four directions disagreeing about all
    of it.  An assembly that mapped sector blocks to the wrong offsets of the
    fused cut leg survives none of that.

    Convergence is deliberately not required, and cannot be had: this fixture's
    CTM does not settle (see :func:`_convergent_site_tensor`).  It is not needed
    — the projectors of a *given* environment are a well-defined function of it,
    so the symmetric and dense layers must agree at every sweep, converged or
    not.  The dense reference is taken from the densified symmetric environment
    rather than from an independently swept dense chain, so the comparison
    isolates the projector layer: the two cannot drift apart upstream and then
    be blamed on the projectors.

    The compared quantity is ``P_left @ P_right``, the oblique projector onto
    the retained subspace, because it is the gauge-invariant one — see
    :func:`test_retained_subspace_projector_matches_the_dense_module`.

    Not every ``(sweep, direction)`` pair is actually comparable, for the same
    reason the ``_TIE_FREE`` table above excludes some ``(seed, chi)`` pairs:
    when ``s[chi-1]`` and ``s[chi]`` at the truncation boundary sit close
    enough, "the top chi singular values" is not a well-defined set and the
    dense module's single global SVD can legitimately retain a different
    subspace than the per-sector one — see :func:`_truncation_gap` and the
    ``_TRUNCATION_GAP_THRESHOLD`` comment above for the measured numbers and
    the margin behind the chosen cutoff.  Unlike ``_TIE_FREE``, the ambiguous
    pairs cannot be listed up front: which ``(sweep, direction)`` pairs land
    near a tie is itself part of this non-convergent trajectory's chaotic,
    platform-dependent behaviour, so the gap is measured and gated at
    comparison time instead.  Every comparison that *does* run is still held
    to the same 1e-10 tolerance as the rest of the suite — the gate only
    removes the pairs where that tolerance is not a meaningful thing to ask
    for, never loosens it for the pairs where it is.
    """
    from tenax.algorithms import _ctm_root_implicit_asym as dense
    from tenax.algorithms._ctm_root_implicit_symmetric import _as_matrix_sym
    from tenax.linalg import _group_blocks_by_bond_charge

    chi = 4
    A = _site_tensor()
    env, a = init_env_sym(A, chi)
    projs = None
    worst = 0.0
    seen = set()
    compared = 0
    skipped = 0
    for _sweep in range(6):
        sym_projs = all_projectors_sym(env, a, chi, projs)
        dense_projs = dense.all_projectors(
            _dense_env_of(env), jnp.asarray(a.todense()), chi
        )
        env_k, a_k = env, a
        for k in range(4):
            layout = sym_projs[k].layout
            assert layout.total == chi, layout.dims
            seen.add(layout.dims)
            gap = _truncation_gap(env_k, a_k, chi)
            if gap is not None and gap < _TRUNCATION_GAP_THRESHOLD:
                skipped += 1
                env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
                continue
            compared += 1
            bot = _as_matrix_sym(lower_left_quadrant_sym(env_k, a_k))
            grouped = _group_blocks_by_bond_charge(bot, [0], [1])
            cut_charge_of_q = {q: int(e[0][0][0]) for q, e in grouped.items()}
            got = _dense_retained_projector(
                sym_projs[k], bot.indices[0], cut_charge_of_q
            )
            expected = dense_projs[k][0] @ dense_projs[k][1]
            rel = float(jnp.max(jnp.abs(got - expected))) / float(
                jnp.max(jnp.abs(expected))
            )
            worst = max(worst, rel)
            assert rel < 1e-10, (k, rel, layout.dims)
            env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
        env, projs = sweep_sym(env, a, chi, projs)

    # Not vacuous: the environment really did leave the initialiser's uniform
    # layout, and the fragmenting three-sector cut really is among those
    # visited.  (Measured here: four distinct layouts, down to a single-sector
    # ``{0: 4}``.  Only the two statements below are asserted, because which
    # layouts a *non-convergent* trajectory visits after a few sweeps is a
    # weaker thing to pin than the parity itself.)  ``seen`` records every
    # layout the trajectory visits, whether or not its comparison was gated
    # out by ``_truncation_gap`` below, so this coverage claim does not depend
    # on the threshold at all.
    assert len(seen) >= 2, sorted(seen)
    assert any(len(d) == 3 for d in seen), sorted(seen)
    # Separately: not vacuous in the *comparison* sense either.  The gate
    # above is allowed to skip individual near-degenerate (sweep, direction)
    # pairs, but it must not be able to skip so many that the 1e-10 assertion
    # never actually runs — see ``_MIN_COMPARISONS`` above for the measured
    # count this leaves margin against.
    assert compared >= _MIN_COMPARISONS, (compared, skipped)
    print(
        f"swept-u1 projector parity {worst:.3e}; layouts visited {sorted(seen)}; "
        f"compared {compared}, skipped (near-degenerate boundary) {skipped}"
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


# ---------------------------------------------------------------------------
# Characteristic equations, covariant form (paper §V.3)
# ---------------------------------------------------------------------------


def _plain_z2_site_tensor(seed: int = 2) -> SymmetricTensor:
    """:func:`_convergent_site_tensor` *without* the dominant-block boost.

    Kept as its own fixture because it is the state whose sweep used to settle
    onto a residual **bond-sign orbit** rather than a fixed point — see
    :func:`test_the_unboosted_z2_state_settles_element_wise`, which is the only
    thing it is used for, and which records why #734 removed the orbit.
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


def _complex_site_tensor(imag: float = 0.25, seed: int = 5) -> SymmetricTensor:
    """:func:`_convergent_site_tensor` with an imaginary part on every block.

    Complex is not decoration: ``λ = ⟨X, X'⟩`` enters ``F`` as ``X'/λ - X``, so
    its phase is part of the equation and perturbing it is a silent, large
    error (#721).  Whether ``λ`` *itself* comes out complex is a bond-gauge
    question rather than a property of the state — on this fixture the four
    corner ``λ`` are ``0.4348, 0.4953, 0.4517, 0.6260``, real, where before
    #734's canonical ``Z2`` labels they were ``-0.3758+0.2185j``, ``0.4953``,
    ``0.4487-0.0522j``, ``0.6260``: the same moduli in a different gauge.  See
    :func:`test_the_root_survives_a_complex_state`.
    """
    A = _convergent_site_tensor()
    rng = np.random.RandomState(seed)
    blocks = {
        k: v + 1j * imag * jnp.asarray(rng.standard_normal(v.shape))
        for k, v in A.blocks.items()
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, A.indices)


def _residual_norm(R) -> float:
    """``‖F‖`` over a residual pytree.

    ``SymmetricTensor`` flattens to a *single* leaf — its packed ``_data``
    buffer — whose L2 norm is the Frobenius norm, so this needs no adapter and
    never materialises a dense array.
    """
    from tenax.algorithms._ctm_root_implicit_symmetric import _leaf_data

    return float(
        jnp.sqrt(sum(jnp.sum(jnp.abs(_leaf_data(x)) ** 2) for x in jax.tree.leaves(R)))
    )


@pytest.fixture(scope="module")
def converged_root():
    """One converged Z2 environment and its root, shared across the §V.3 tests.

    Module-scoped because the 36 sweeps plus the extraction are the expensive
    part and this file runs in the ``core`` CI gate; every test below reads the
    same root rather than rebuilding it.
    """
    A = _convergent_site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=60, return_projectors=True)
    assert meta["converged"], meta
    root, residual = root_parametrize_sym(env, a, chi=4, prev_projs=projs)
    return A, a, env, projs, root, residual


def test_the_converged_environment_is_a_root(converged_root):
    """The gate for the whole of §V.3: the sweep's fixed point solves ``F = 0``.

    ``F`` is assembled from the **upper** half of the plane while
    :func:`sweep_sym` truncates with the **left** half, so this is not a
    tautology — the two are related only through
    :func:`sym_root_to_covariant_convention`, and a sweep and an equation that
    disagree give a fixed point that is not a root.  That is how #718 started.

    ``prev_projs`` is passed deliberately: a cold re-pin fixes a *different*
    bond gauge and leaves ``y*`` describing an environment it was not extracted
    from.
    """
    _A, _a, _env, _projs, _root, residual = converged_root
    assert residual < 1e-10, residual
    print(f"norm(F(y*)) = {residual:.3e}")


def test_the_root_residual_is_not_vacuously_zero(converged_root):
    """``F`` really depends on ``y`` — 1e-13 at the root is a statement, not a bug.

    A residual that collapsed to zero because a contraction silently dropped
    the charge-conserving part of the network (``contract`` does that rather
    than raising) would pass the gate above and nothing else.  Perturbing the
    environment and the null-space coordinate ``u`` separately moves ``F`` by
    ten orders of magnitude, which says both blocks of the equation are live.
    """
    _A, a, _env, _projs, root, residual = converged_root

    rng = np.random.RandomState(0)
    env_p = []
    for t in root.env:
        blocks = {
            key: b + 1e-3 * jnp.asarray(rng.standard_normal(b.shape))
            for key, b in t.blocks.items()
        }
        env_p.append(SymmetricTensor._from_blocks_unchecked(blocks, t.indices))
    moved_env = _residual_norm(
        characteristic_residual_sym(
            (SymEnv(*env_p), root.u, root.s, root.v), a, root, 4
        )
    )

    u_p = tuple({q: m + 1e-3 for q, m in d.items()} for d in root.u)
    moved_u = _residual_norm(
        characteristic_residual_sym((root.env, u_p, root.s, root.v), a, root, 4)
    )

    assert moved_env > 1e-3, moved_env
    assert moved_u > 1e-4, moved_u
    print(f"|F| root {residual:.2e} / env+1e-3 {moved_env:.2e} / u+1e-3 {moved_u:.2e}")


def test_the_enlarged_corner_transpose_is_load_bearing(converged_root, monkeypatch):
    """§V.3 orders an enlarged corner ``(cut | outer)``; the helper returns the other.

    The dense module pins that transpose to 7.7e-16 against the reference
    implementation's dumped fixed point.  Dropping it is not a small error —
    it is a different network — and this test says so in the only currency that
    matters here: the root stops being a root.
    """
    from tenax.algorithms import _ctm_root_implicit_symmetric as mod
    from tenax.linalg import _group_blocks_by_bond_charge

    _A, a, _env, _projs, root, residual = converged_root

    def untransposed(matrix):
        grouped = _group_blocks_by_bond_charge(matrix, [0], [1])
        return {
            q: (e[0][2], int(e[0][0][0]), int(e[0][1][0])) for q, e in grouped.items()
        }

    monkeypatch.setattr(mod, "_cut_outer_blocks", untransposed)
    wrong = _residual_norm(characteristic_residual_sym(root.y, a, root, 4))
    # Two-part, scale-free: large in absolute terms AND enormous relative to
    # the true root.  A bare ``> 1.0`` is calibrated at the edge of a
    # platform-dependent quantity — macOS measured 0.924 here where Linux
    # measured 4.9, which failed CI while the counterfactual was in fact
    # firing across twelve orders of magnitude.  Same pattern as
    # ``test_a_wrong_bond_layout_breaks_the_root``.
    assert wrong > 1e-2, wrong
    assert wrong > 1e9 * residual, (wrong, residual)
    print(f"|F| with the (cut|outer) transpose {residual:.2e}, without it {wrong:.2e}")


def test_the_covariant_shift_is_load_bearing(converged_root):
    """§V.3's direction ``j`` is the forward sweep's rotation ``j+1``, not ``j``.

    Shifting once more takes every ``(u, S, v, U*, V*)`` to the wrong cut while
    leaving every shape intact, so nothing raises — the only symptom is that
    ``y*`` stops solving the equations.  That is exactly the failure mode the
    bridge exists to prevent, and it fails loudly: 19 against 3e-13.
    """
    _A, a, _env, _projs, root, residual = converged_root

    def shift(xs):
        return tuple(xs[(j + 1) % 4] for j in range(4))

    bad = root._replace(
        u=shift(root.u),
        s=shift(root.s),
        v=shift(root.v),
        U_star=shift(root.U_star),
        U_perp=shift(root.U_perp),
        Vh_star=shift(root.Vh_star),
        Vh_perp=shift(root.Vh_perp),
        s_star_inv=shift(root.s_star_inv),
        layouts=shift(root.layouts),
    )
    wrong = _residual_norm(
        characteristic_residual_sym((root.env, bad.u, bad.s, bad.v), a, bad, 4)
    )
    # Two-part, scale-free: large in absolute terms AND enormous relative to
    # the true root.  A bare ``> 1.0`` is calibrated at the edge of a
    # platform-dependent quantity — macOS measured 0.924 here where Linux
    # measured 4.9, which failed CI while the counterfactual was in fact
    # firing across twelve orders of magnitude.  Same pattern as
    # ``test_a_wrong_bond_layout_breaks_the_root``.
    assert wrong > 1e-2, wrong
    assert wrong > 1e9 * residual, (wrong, residual)
    print(f"|F| at the right shift {residual:.2e}, one shift further {wrong:.2e}")


def test_apply_bond_matrix_acts_only_on_the_chi_factor(converged_root):
    """The deleted kron, tested directly: per-sector identity is a no-op.

    And, less trivially, the operator it *does* apply is the one the dense
    module means by ``kron(root, eye_d2)``: a matrix on the ``chi`` factor of
    the cut leg and the identity on the ``D²`` factor.  The dense oracle is
    legitimate because ``fuse_indices`` keeps the parent legs' row-major
    ordering inside the fused charge array — see :func:`_as_matrix_sym` — so a
    dense ``kron`` on the *whole* leg is right even though the per-sector one
    is not (:func:`test_a_per_sector_kron_cannot_express_a_bond_matrix`).
    """
    _A, a, env, _projs, _root, _residual = converged_root
    quad = upper_left_quadrant_sym(env, a)
    chi_idx = quad.indices[0]
    dims = {
        int(c): int(m)
        for c, m in zip(np.asarray(chi_idx.sectors), np.asarray(chi_idx.multiplicities))
    }

    eye = {q: jnp.eye(d) for q, d in dims.items()}
    for side in ("left", "right"):
        same = apply_bond_matrix(quad, eye, axis=0, side=side)
        assert bool(jnp.all(same._data == quad._data)), side

    rng = np.random.RandomState(0)
    mats = {q: jnp.asarray(rng.standard_normal((d, d))) for q, d in dims.items()}
    charges = np.asarray(chi_idx.charges)
    M = np.zeros((len(charges), len(charges)))
    for q, m in mats.items():
        pos = np.where(charges == q)[0]
        M[np.ix_(pos, pos)] = np.asarray(m)
    M = jnp.asarray(M)
    dense_quad = quad.todense()

    got_l = apply_bond_matrix(quad, mats, axis=0, side="left").todense()
    want_l = jnp.einsum("ab,bjkl->ajkl", M, dense_quad)
    assert float(jnp.max(jnp.abs(got_l - want_l))) < 1e-12

    got_r = apply_bond_matrix(quad, mats, axis=0, side="right").todense()
    want_r = jnp.einsum("ajkl,ab->bjkl", dense_quad, M)
    assert float(jnp.max(jnp.abs(got_r - want_r))) < 1e-12

    # A charge the leg does not carry is a raise, not a silent skip.
    with pytest.raises(ValueError, match="no bond matrix for charge"):
        apply_bond_matrix(quad, {q: m for q, m in mats.items() if q != 0}, axis=0)


def test_a_per_sector_kron_cannot_express_a_bond_matrix():
    """Why the dense ``jnp.kron(root, eye_d2)`` is deleted rather than ported.

    Dense, the cut leg is the flat product ``n = chi * d2`` with ``chi`` slow,
    so a matrix on the ``chi`` factor is ``kron(root, I_d2)``.  Fused, sector
    ``q`` of the cut is a direct sum over every ``(q_chi, q_d2)`` summing to
    ``q``, and on the fragmenting U(1) fixture that is not a product at all:

    * the fused leg carries charges ``±2`` that the ``chi`` leg does not have,
      so there is no ``root`` block to kron with in the first place;
    * where the charge does exist the dimensions disagree — fused sector ``0``
      has 6 states while ``chi_0 * d2`` is ``2 * 4 = 8``.

    So the per-sector kron is not merely inaccurate, it is not shape-legal, and
    the fix is to keep the leg split and contract (:func:`apply_bond_matrix`).
    """
    from tenax.algorithms._ctm_root_implicit_symmetric import _as_matrix_sym

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = None
    for _ in range(3):
        env, projs = sweep_sym(env, a, 4, projs)

    quad = upper_left_quadrant_sym(env, a)
    chi_idx, a_idx = quad.indices[0], quad.indices[1]
    fused = _as_matrix_sym(quad).indices[0]
    chi_dims = {
        int(c): int(m)
        for c, m in zip(np.asarray(chi_idx.sectors), np.asarray(chi_idx.multiplicities))
    }
    fused_dims = {
        int(c): int(m)
        for c, m in zip(np.asarray(fused.sectors), np.asarray(fused.multiplicities))
    }
    d2 = a_idx.dim

    assert set(fused_dims) - set(chi_dims), (fused_dims, chi_dims)
    mismatched = {
        q: (n, chi_dims.get(q, 0) * d2)
        for q, n in fused_dims.items()
        if n != chi_dims.get(q, 0) * d2
    }
    assert mismatched, (fused_dims, chi_dims, d2)
    print(f"fused vs chi*d2 per sector: {mismatched}")


def test_sym_root_to_covariant_convention_shifts_transposes_and_swaps():
    """The bridge, on hand-built data where every block is identifiable.

    Three things at once and each is separately wrong-able: the ``+1``
    direction shift, the **plain** transpose of every block (not the adjoint —
    no conjugation enters), and the ``u <-> v`` / ``U <-> Vh`` swap.  Sector
    keys are carried across unchanged; see the function's docstring for the
    measurement that licenses that.
    """
    from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout

    def block(k, tag):
        return {0: jnp.asarray([[k + 0.5j, tag + 0.0j]])}

    layouts = tuple(BondLayout.from_dims({0: k + 1}) for k in range(4))
    root = SymRoot(
        env=None,
        u=tuple(block(k, 1) for k in range(4)),
        s=tuple(block(k, 2) for k in range(4)),
        v=tuple(block(k, 3) for k in range(4)),
        U_star=tuple(block(k, 4) for k in range(4)),
        U_perp=tuple(block(k, 5) for k in range(4)),
        Vh_star=tuple(block(k, 6) for k in range(4)),
        Vh_perp=tuple(block(k, 7) for k in range(4)),
        s_star_inv=tuple(block(k, 8) for k in range(4)),
        layouts=layouts,
    )
    cov = sym_root_to_covariant_convention(root)

    for j in range(4):
        src = (j + 1) % 4
        assert cov.layouts[j] is layouts[src]
        for got, want in (
            (cov.u[j], root.v[src]),
            (cov.s[j], root.s[src]),
            (cov.v[j], root.u[src]),
            (cov.U_star[j], root.Vh_star[src]),
            (cov.U_perp[j], root.Vh_perp[src]),
            (cov.Vh_star[j], root.U_star[src]),
            (cov.Vh_perp[j], root.U_perp[src]),
            (cov.s_star_inv[j], root.s_star_inv[src]),
        ):
            assert got[0].shape == want[0].T.shape
            assert bool(jnp.all(got[0] == want[0].T)), (j, "plain transpose")
            # Not the adjoint: the imaginary part survives unconjugated.
            assert bool(jnp.any(got[0] != want[0].conj().T))


def test_remove_and_absorb_inverse_roots_invert_each_other(converged_root):
    """Eq. 82 both ways, up to the per-tensor normalisation both maps apply.

    ``absorb_inverse_roots_sym`` is the *differentiable* direction — the energy
    is evaluated on the regular environment, so ``S`` reaches the energy only
    through it, and that is where ``S̆`` comes from.  A round trip that did not
    close would put a systematic error straight into the gradient.
    """
    _A, _a, _env, _projs, root, _residual = converged_root
    regular = absorb_inverse_roots_sym(root.env, root.s)
    back = remove_inverse_roots_sym(regular, root.s)
    for i, name in enumerate(("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")):
        x, y = root.env[i], back[i]
        xn = x._data / jnp.linalg.norm(x._data)
        yn = y._data / jnp.linalg.norm(y._data)
        assert float(jnp.max(jnp.abs(xn - yn))) < 1e-10, name
    # And the regular environment is genuinely different from the modified one.
    moved = max(
        float(jnp.max(jnp.abs(regular[i]._data - root.env[i]._data))) for i in range(8)
    )
    assert moved > 1e-6, moved


def test_the_covariant_layouts_are_the_forward_ones_shifted(converged_root):
    """``root.layouts[j]`` is the forward truncation at rotation ``j+1``.

    The shape bookkeeping of the whole covariant layer keys off this, and it is
    the one piece of the bridge that does not fail loudly if it is wrong at a
    *uniform* fixed point, where all four layouts happen to coincide.
    """
    _A, a, env, projs, root, _residual = converged_root
    forward = all_projectors_sym(env, a, 4, projs)
    for j in range(4):
        assert root.layouts[j].dims == forward[(j + 1) % 4].layout.dims, j


def test_the_root_survives_a_complex_state(monkeypatch):
    """A complex state, and the reason ``λ``'s phase is carried, not dropped (#721).

    ``F`` normalises as ``X'/λ - X``, so ``λ``'s phase is part of the equation:
    perturb it and the converged point stops being a root.  That is what #721
    is about, and the counterfactual below measures it directly.

    It used to be measured by real-projecting ``λ``, which on this fixture
    moved ``‖F(y*)‖`` from 3e-13 to 4.9.  That form went **vacuous** at #734:
    the four corner ``λ`` were ``-0.3758+0.2185j, 0.4953, 0.4487-0.0522j,
    0.6260`` only because the ``Z2`` bond was labelled with the non-canonical
    representative ``-1``, which put the odd sector first and picked a different
    SVD column gauge.  With canonical labels the same corners give ``0.4348,
    0.4953, 0.4517, 0.6260`` — *identical moduli*, zero phase — so ``.real`` is
    now a no-op here and would have quietly stopped testing anything.  Rotating
    the phase instead pins the same property without depending on which gauge
    the bond happens to land in.

    The ``.real`` counterfactual itself is not lost: the dense module keeps it,
    in a gauge chosen so that it cannot go vacuous, as
    ``test_the_dense_root_carries_lambdas_phase_rather_than_dropping_it``.
    """
    from tenax.algorithms import _ctm_root_implicit_symmetric as mod

    A = _complex_site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=60, return_projectors=True)
    assert meta["converged"], meta
    root, residual = root_parametrize_sym(env, a, chi=4, prev_projs=projs)
    assert residual < 1e-10, residual

    true_lambda = mod._vdot_sym

    def rotated(x, y):
        return jnp.exp(1j * 0.25) * true_lambda(x, y)

    monkeypatch.setattr(mod, "_vdot_sym", rotated)
    wrong = _residual_norm(characteristic_residual_sym(root.y, a, root, 4))
    # Two-part, scale-free: large in absolute terms AND enormous relative to
    # the true root.  A bare ``> 1.0`` is calibrated at the edge of a
    # platform-dependent quantity — macOS measured 0.924 for the old
    # real-projection form where Linux measured 4.9, which failed CI while the
    # counterfactual was in fact firing across twelve orders of magnitude.
    # Same pattern as ``test_a_wrong_bond_layout_breaks_the_root``.
    assert wrong > 1e-2, wrong
    assert wrong > 1e9 * residual, (wrong, residual)
    print(f"complex |F| = {residual:.3e}; with λ's phase rotated = {wrong:.3e}")


def test_the_root_survives_an_empty_and_uneven_sector_layout():
    """chi=2 on this fixture retains *different* charges in different directions.

    Two directions keep ``{0: 1, 1: 1}`` and two keep ``{0: 2}`` outright, so
    a sector is empty on half the ring.  That is the case where the bridge's
    per-direction layouts genuinely differ and where an implementation that
    assumed one global ``chi_q`` would break — at a uniform fixed point all four
    layouts coincide and nothing is tested.
    """
    A = _convergent_site_tensor()
    env, a, meta, projs = converge_sym(A, chi=2, max_iter=80, return_projectors=True)
    assert meta["converged"], meta
    root, residual = root_parametrize_sym(env, a, chi=2, prev_projs=projs)
    dims = [tuple(x.dims) for x in root.layouts]
    assert len({d for d in dims}) > 1, dims
    assert any(len(d) == 1 for d in dims), dims
    assert residual < 1e-10, residual
    print(f"chi=2 layouts {dims}, norm(F(y*)) = {residual:.3e}")


def test_a_moving_bond_layout_is_refused_rather_than_mis_extracted():
    """``S`` on charges the corners do not carry is a raise, not a wrong number.

    Away from a fixed point the truncation can retain a different charge
    distribution than the bonds the environment was built on — the U(1) fixture
    does exactly that, which is why it has no CTM limit (see
    :func:`_convergent_site_tensor`).  The polish loop sweeps instead of
    extracting when that happens, and says so plainly if it never resolves.
    """
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = None
    for _ in range(3):
        env, projs = sweep_sym(env, a, 4, projs)
    with pytest.raises(ValueError, match="never carried the charge distribution"):
        root_parametrize_sym(env, a, chi=4, prev_projs=projs, polish_steps=1)


def _move_one_dimension(layout):
    """``layout`` with one retained state moved from its first sector to its second.

    Total unchanged, so it is still a truncation to ``chi``; only *which*
    charges get to keep how many states differs.
    """
    from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout

    dims = dict(layout.dims)
    lo, hi = sorted(dims)[:2]
    dims[lo] -= 1
    dims[hi] += 1
    return BondLayout.from_dims(dims)


def test_a_wrong_bond_layout_breaks_the_root(converged_root):
    """The layout must be load-bearing.

    Take the converged root and move one retained dimension from one charge
    sector to another, keeping the total at chi.  Every shape still fits and
    every charge still conserves — only the *physics* is wrong.  If F is still
    a root after that, the sector bookkeeping is not doing anything and none
    of the other tests mean what they claim.

    Measured on this fixture, ``{0: 2, 1: 2}`` at chi=4 against
    ``{0: 1, 1: 3}``, and the answer comes in two parts because the layout is
    load-bearing in two different places.

    * The layout is recorded in the *environment*, not only in the equations:
      every corner and edge chi leg carries it as its index multiplicities.  So
      re-extracting a root of the **same** environment under a moved layout is
      refused outright — ``root_parametrize_sym`` will not put an ``S`` on a
      bond the corners do not have.  That is the "every shape still fits" claim
      failing: it does not, and the module says so rather than reshaping around
      it.
    * Give the wrong layout every chance instead — let it re-truncate the
      environment consistently, so that the bonds, ``S``, the isometries and
      the cut all agree with each other and nothing *can* raise — and it
      **does** find a root: ``‖F(y)‖`` falls by some nine orders over 3 → 40
      polish sweeps from the natural fixed point, and when the forced layout is
      converged from a fresh environment on its own terms, which is what this
      test does, it reaches the same order as the natural root.  Forcing a
      layout does not break the equations; it poses a *different* truncation
      problem, and that problem has its own fixed point.  What the moved
      dimension changes is the state you land on: the forced fixed point
      reports an energy a few times 1e-2 away from the one the truncation
      actually chose, next to the 1e-10 the natural layout reproduces against
      the dense module.  So the layout is load-bearing — it selects which fixed
      point you get — but ``‖F‖`` is not what detects it, and this test asserts
      the energy instead.

      The exact residual ladder and the two energies used to be quoted here and
      **had drifted** — the recorded 8.8e-2 → 2.4e-6 at 3 and 10 sweeps
      measures 2.8e-2 → 5.0e-5 today, and the forced energy recorded as
      ``-0.10873`` is ``-0.13176``.  Nothing failed, because the assertions are
      scale-free (``> 1e-3``, ``< 1e-8``) and deliberately so; but a docstring
      full of numbers no run reproduces is worse than one with none, so what is
      quoted now is only what the assertions actually constrain.  The test
      prints the live values.

      This corrects an earlier reading of this fixture.  Before the polish
      sweep forwarded ``layout_override`` (it was passed to
      ``all_projectors_sym`` but dropped on the sweep that advances the
      environment), the run alternated forced and natural truncations and
      ``‖F‖`` sat at 4.0e-1 — a floor produced by the alternation, not by the
      forced layout.  The claim "a legal truncation with no root nearby" was an
      artifact of that bug and is false.

    The control matters as much as the trap: pushing the *right* layout through
    the same override reproduces the fixture's residual (identically, in fact,
    on this machine — asserted to 1e-3 relative so a different BLAS cannot make
    it flaky), so what breaks the root above is the moved dimension and not the
    hook.
    """
    _A, a, env, projs, _root, residual = converged_root
    right = tuple(p.layout for p in projs)
    wrong = tuple(_move_one_dimension(x) for x in right)
    assert all(x.total == 4 for x in wrong), [x.dims for x in wrong]
    assert all(w.dims != r.dims for w, r in zip(wrong, right, strict=True))

    # The hook is inert: the right layout through it is the fixture's root.
    _same, res_control = root_parametrize_sym(
        env, a, chi=4, prev_projs=projs, layout_override=right, polish_steps=1
    )
    assert res_control < 1e-10, res_control
    assert abs(res_control - residual) <= 1e-3 * residual, (res_control, residual)

    # (a) The environment's own bonds record the layout, so a moved one is
    #     refused rather than mis-extracted.
    with pytest.raises(ValueError, match="never carried the charge distribution"):
        root_parametrize_sym(
            env, a, chi=4, prev_projs=projs, layout_override=wrong, polish_steps=1
        )

    # (b) Applied consistently — the polish sweeps use it too — the forced
    #     layout is shape-legal *and* reaches a root of its own.  Converge it
    #     on its own terms from a fresh environment, then ask what it costs.
    import tenax

    gate = tenax.heisenberg_gate()
    env_w, a_w = init_env_sym(_A, chi=4)
    prev_w = None
    for _ in range(40):
        env_w, prev_w = sweep_sym(env_w, a_w, 4, prev_w, layout_override=wrong)
    assert [p.layout.dims for p in prev_w] == [x.dims for x in wrong]

    forced_root, res_wrong = root_parametrize_sym(
        env_w, a_w, chi=4, prev_projs=prev_w, layout_override=wrong, polish_steps=3
    )
    assert [x.dims for x in forced_root.layouts] == [x.dims for x in wrong]
    # It is a root: forcing the layout poses a different problem, not a broken
    # one.  Asserting otherwise is what the dropped kwarg used to make look true.
    assert res_wrong < 1e-8, res_wrong

    # It is a *different* root.  That is what the layout buys, and the gap is
    # seven orders above the agreement the natural layout reaches.
    e_forced = float(sym_energy(_A, env_w, gate))
    e_true = float(sym_energy(_A, env, gate))
    assert abs(e_forced - e_true) > 1e-3, (e_forced, e_true)
    print(
        f"layout {right[0].dims} -> {wrong[0].dims}: "
        f"norm(F) {residual:.3e} -> {res_wrong:.3e} (still a root), "
        f"E {e_true:.8f} -> {e_forced:.8f}"
    )


def test_layout_override_refuses_a_layout_the_cut_cannot_supply():
    """The trap hook only forces truncations that exist, so the trap is a trap.

    ``layout_override`` is there to retain a *different* set of states from the
    same decomposition (:func:`test_a_wrong_bond_layout_breaks_the_root`).  If
    it also accepted charges the cut does not carry, or more states than a
    sector has, the trap test would be measuring an index error rather than a
    wrong truncation.
    """
    from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout

    A = _convergent_site_tensor()
    env, a = init_env_sym(A, chi=4)
    good = tuple(p.layout for p in all_projectors_sym(env, a, chi=4))

    absent = (BondLayout.from_dims({7: 4}),) + good[1:]
    with pytest.raises(ValueError, match="which the cut does not carry"):
        all_projectors_sym(env, a, chi=4, layout_override=absent)

    greedy = (BondLayout.from_dims({good[0].sectors[0]: 4096}),) + good[1:]
    with pytest.raises(ValueError, match="it only has"):
        all_projectors_sym(env, a, chi=4, layout_override=greedy)

    with pytest.raises(ValueError, match="one entry per direction"):
        all_projectors_sym(env, a, chi=4, layout_override=good[:3])

    # And passing the real layouts back in is a no-op, direction by direction.
    same = all_projectors_sym(env, a, chi=4, layout_override=good)
    assert [p.layout.dims for p in same] == [x.dims for x in good]


def test_a_wrong_sector_assignment_survives_every_shape_check(converged_root):
    """The companion trap, for the case no guard can catch.

    Moving a dimension changes shapes, so the module gets to refuse it.  This
    one does not: at chi=4 the Z2 cut retains ``{0: 2, 1: 2}`` and **every**
    per-sector block — ``u``, ``S``, ``v``, ``U*``, ``U_perp``, ``V*``,
    ``V_perp`` — has literally the same shape in both sectors, which the test
    asserts before doing anything else.  So exchanging the two sectors' blocks
    is invisible to every shape check, every charge check and the bond index
    itself; nothing raises, and the only thing that can notice is ``F``.

    It does: ``‖F‖`` goes from 2.7e-13 to ~1.2e3.  Swapping only ``S`` (27) or
    only the isometries (40) is enough on its own, so the sector assignment is
    live in both halves of the equations rather than in one place that happens
    to dominate.
    """
    _A, a, _env, _projs, root, residual = converged_root

    def swap(d):
        lo, hi = sorted(d)
        return {lo: d[hi], hi: d[lo]}

    def swap_all(xs):
        return tuple(swap(d) for d in xs)

    fields = ("u", "s", "v", "U_star", "U_perp", "Vh_star", "Vh_perp", "s_star_inv")
    for k in range(4):
        sectors = root.layouts[k].sectors
        assert len(sectors) == 2, (k, root.layouts[k].dims)
        for name in fields:
            shapes = {getattr(root, name)[k][q].shape for q in sectors}
            assert len(shapes) == 1, (k, name, shapes)

    swapped = root._replace(**{name: swap_all(getattr(root, name)) for name in fields})
    both = _residual_norm(
        characteristic_residual_sym(
            (root.env, swapped.u, swapped.s, swapped.v), a, swapped, 4
        )
    )
    s_only = _residual_norm(
        characteristic_residual_sym(
            (root.env, root.u, swap_all(root.s), root.v), a, root, 4
        )
    )
    iso_only = _residual_norm(
        characteristic_residual_sym(
            root.y,
            a,
            root._replace(
                **{
                    name: swap_all(getattr(root, name))
                    for name in ("U_star", "U_perp", "Vh_star", "Vh_perp")
                }
            ),
            4,
        )
    )
    # Scale-free, for the reason given in
    # ``test_the_root_survives_a_complex_state``: an absolute ``> 1.0`` sits at
    # the edge of a platform-dependent value.  These measured 1.19e3 / 27.0 /
    # 39.9 locally, so 1e-2 keeps three orders of margin at worst while still
    # asserting the swap is catastrophic rather than merely detectable.
    for name, value in (("both", both), ("S only", s_only), ("isometries", iso_only)):
        assert value > 1e-2, (name, value)
        assert value > 1e9 * residual, (name, value, residual)
    print(
        f"sector swap: |F| root {residual:.2e} / all {both:.2e} / S only "
        f"{s_only:.2e} / isometries only {iso_only:.2e}"
    )


def test_the_unboosted_z2_state_settles_element_wise():
    """The unboosted Z2 state reaches a genuine fixed point, not a sign orbit.

    This test used to pin the opposite, and the history is worth keeping.  With
    the ``Z2`` bond labelled by the non-canonical representative ``-1``, the
    odd sector sorted *first*, which picked a different SVD column gauge: the
    sweep then reproduced the environment exactly except that ``C2`` and ``C4``
    changed sign every iteration, so the raw element-wise residual sat at 1.88
    for 200 sweeps while every ``|·|`` agreed to 1e-13.  That ``σ`` was a bond
    gauge — per-direction signs ``s = (+, +, -, -)`` give corner ``C_j`` the
    factor ``s_{j-1} s_j`` and every edge ``s_j² = +1`` — so the orbit was a
    root, and the forward criterion had to be the phase-aligned one.

    #734's canonical labels put the odd sector second and the orbit is gone:
    every ``λ`` is ``+1``, and the raw and phase-aligned differences agree to
    1e-13.  The phase-aligned criterion is still the right one (the complex
    fixture and the general gauge argument have not changed), but it is no
    longer load-bearing *here*, so this pins the stronger element-wise
    statement instead.

    The retained layout is ``((0, 2), (1, 2))`` on all four directions at
    *every* sweep, so the warm gauge pin never cold-restarts and the projectors
    are stationary.  (It read ``((-1, 2), (0, 2))`` before #734 — the same two
    sectors, written with the non-canonical representative.)
    """
    from tenax.algorithms._ctm_root_implicit_symmetric import (
        _max_abs_diff_sym,
        _phase_aligned_max_diff,
    )

    A = _plain_z2_site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=80, return_projectors=True)
    assert meta["converged"], meta
    assert meta["residual"] < 1e-12, meta

    # The layout never moved, so the "cold re-pin" mechanism is ruled out.
    for p in projs:
        assert p.layout.dims == ((0, 2), (1, 2)), p.layout.dims

    env_next, _ = sweep_sym(env, a, 4, projs)
    names = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
    signs = {}
    for name in names:
        x, y = getattr(env, name), getattr(env_next, name)
        xn = x * (1.0 / jnp.linalg.norm(x._data))
        yn = y * (1.0 / jnp.linalg.norm(y._data))
        lam = complex(jnp.vdot(xn._data, yn._data))
        signs[name] = lam
        assert _phase_aligned_max_diff(xn, yn) < 1e-11, name
        # No orbit: the raw difference is as small as the phase-aligned one,
        # which is what distinguishes a fixed point from a gauge orbit.
        assert _max_abs_diff_sym(xn, yn) < 1e-11, (name, _max_abs_diff_sym(xn, yn))
        assert abs(lam - 1.0) < 1e-11, (name, lam)

    # Kept from the orbit era: the gauge-invariant consistency check.
    corner_product = np.prod([signs[f"C{i}"] for i in (1, 2, 3, 4)])
    assert abs(corner_product - 1.0) < 1e-10, corner_product

    _root, residual = root_parametrize_sym(env, a, chi=4, prev_projs=projs)
    assert residual < 1e-10, residual
    print(
        f"element-wise fixed point: λ = {[f'{signs[n].real:+.3f}' for n in names]}, "
        f"norm(F(y*)) = {residual:.3e}"
    )


# ---------------------------------------------------------------------------
# The adjoint and the gradient (paper Eq. 18)
#
# **Every test below that touches the ``sym_gradient`` fixture must carry
# ``@pytest.mark.slow``, and that is a CI requirement rather than a
# preference.**  Building the fixture peaks at 8.4 GB RSS — measured, staged:
# 0.69 GB after the root extraction, 2.33 after the energy VJP, 2.50 after the
# ``F`` VJP trace, 8.36 after the GMRES solve, flat thereafter.  The step that
# costs it is XLA compiling the ~15k-equation block-sparse VJP inside GMRES's
# ``lax.while_loop``; ``restart=10`` only reaches 7.19 GB and is both slower
# (93 s vs 49 s) and far less accurate (2.5e-10 vs 6.4e-15), so there is no
# cheap lever inside the solver.  GitHub's Linux runners have ~7 GB (see the
# ``jax.clear_caches`` threshold note in ``tests/conftest.py``), and ``-m
# core`` is the required gate, so an unmarked test here OOM-kills the runner
# — which is what a marker on only *some* consumers of a module-scoped
# fixture would still do, since the first consumer to run builds it.
# ``tests/conftest.py`` withholds this file's ``core`` marker from anything
# explicitly marked ``slow``; without the marker here that mechanism has
# nothing to act on.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sym_gradient():
    """``(A, gate, chi, energy, dE/dA, diagnostics)`` — computed once.

    Module scoped because this is the expensive object in the file: a 36-sweep
    convergence, the root extraction, three VJP traces and a GMRES solve
    through the block-sparse ``F``.  Every gradient test below reads the same
    result rather than recomputing it — and every one of them is ``slow``, for
    the 8.4 GB reason in the section header above.
    """
    import tenax

    A = _convergent_site_tensor()
    gate = tenax.heisenberg_gate()
    energy, grad, diag = sym_root_implicit_energy_and_grad(A, gate, chi=4)
    return A, gate, 4, energy, grad, diag


@pytest.mark.slow
def test_no_svd_or_eigh_primitive_in_the_backward(sym_gradient):
    """The claim the whole method rests on, asserted on the jaxpr.

    ``diagnostics["backward_jaxpr"]`` is the **primitive listing** of the six
    jaxprs the gradient traces — the energy, ``F(y)`` and ``F(p)``, and each of
    their pullbacks — rather than their pretty-printed text.  Two things about
    that are deliberate.

    The *forward* three are the half that actually pins the claim.  The VJP of
    a decomposition contains no decomposition: it is the stored ``U``, ``s``,
    ``Vh`` and a divided difference.  So an ``svd`` inside ``F`` would leave
    the pullbacks' primitive lists clean while putting the ``1/(s_i² - s_j²)``
    this method exists to avoid straight back into the gradient.

    And the listing replaces the text because ``jaxpr.__str__`` names variables
    ``a, b, ..., z, aa, ab, ...``; these programs run to ~40k equations, so the
    namer reaches a variable literally called ``svd`` — measured — which makes
    ``"svd" not in str(jaxpr)`` false against a jaxpr containing no such
    primitive.  The dense module's version of this test is safe only because
    its jaxpr is smaller.

    The forward decomposition still happens, in ``all_projectors_sym``; it
    enters all six as a *constant*.  That is the whole method: #566's
    block-sparse SVD/eigh VJP compile wall and #687's accuracy floor are
    avoided by never differentiating one.
    """
    _A, _gate, _chi, _e, _g, diag = sym_gradient
    text = diag["backward_jaxpr"]
    assert "svd" not in text, "an SVD survived into the backward"
    assert "eigh" not in text, "an eigh survived into the backward"
    # Not vacuous: the listing has to be a real listing.
    assert "dot_general" in text, text
    print(f"backward primitives: {sorted(text.split())}")


@pytest.mark.slow
def test_the_gradient_entry_point_reports_clean_diagnostics(sym_gradient):
    """The gradient is a ``SymmetricTensor`` on ``A``'s own indices, and finite.

    The charge structure is part of the answer, not scaffolding around it: a
    gradient that came back dense would have to be projected before it could
    be added to ``A``, and the projection is exactly the information the
    symmetric path is supposed to keep.
    """
    A, _gate, _chi, energy, grad, diag = sym_gradient

    assert isinstance(grad, SymmetricTensor)
    assert grad.indices == A.indices
    assert set(grad.blocks) == set(A.blocks)
    assert bool(jnp.all(jnp.isfinite(grad._data)))
    assert float(jnp.linalg.norm(grad._data)) > 1e-6

    assert diag["converged"], diag
    assert diag["iters"] < 60, diag
    assert diag["root_residual"] < 1e-10, diag
    assert diag["covariant_residual"] < 1e-10, diag
    assert diag["gmres_residual"] < 1e-8, diag
    assert diag["gauge_consistency"] < 1e-8, diag
    print(
        f"E = {float(energy):.12f}, |dE/dA| = "
        f"{float(jnp.linalg.norm(grad._data)):.6f}, "
        f"root {diag['root_residual']:.2e}, gmres {diag['gmres_residual']:.2e}, "
        f"gauge {diag['gauge_consistency']:.2e}"
    )


@pytest.mark.slow
def test_a_dense_site_tensor_is_refused(sym_gradient):
    """The dense 1x1 path is a different function, and says so."""
    A, gate, _chi, _e, _g, _diag = sym_gradient
    with pytest.raises(TypeError, match="SymmetricTensor"):
        sym_root_implicit_energy_and_grad(_as_dense_site(A), gate, chi=4)


@pytest.mark.slow
def test_the_singular_value_adjoint_is_not_negligible(sym_gradient):
    """``S̆`` is the same order as ``C̆`` — the #718 bug, as a number.

    The energy is evaluated on the *regular* environment, so ``S`` reaches it
    only through the Eq. 82 absorption of :func:`absorb_inverse_roots_sym`.
    Writing ``F`` in the regular variables instead — the obvious
    simplification — sets this ratio to exactly zero, and dense that was worth
    2.5e-2 of gradient error.  It is ~0.4 here, i.e. not a correction.
    """
    _A, _gate, _chi, _e, _g, diag = sym_gradient
    ratio = diag["singular_value_adjoint_ratio"]
    assert ratio > 1e-2, ratio
    print(f"|S̆| / |C̆, Ĕ| = {ratio:.4f}")


def test_the_energy_is_invariant_under_every_environment_phase():
    """What ``gauge_consistency`` measures, checked without differentiating.

    An independent phase on *each* environment tensor is an exact null
    direction of ``∂_y F``, so the adjoint system is singular and is solvable
    only because the right-hand side — the energy's cotangent — is orthogonal
    to every one of those directions.  That orthogonality is a property of the
    **energy boundary**, not of the characteristic equations, and #718 is the
    standing reminder that the boundary is where conventions go wrong.  Here it
    is checked directly: eight separate phases, energy unchanged.

    A phase rather than a scale, deliberately.  A real rescaling would also
    cancel in the ratio, but it is not the null direction at issue: the
    cotangent pairing is ``Re Σ g·δz`` with ``δz = i x``, and only a phase
    probes the direction the pairing is taken along.
    """
    import tenax

    A = _convergent_site_tensor()
    gate = tenax.heisenberg_gate()
    env, _a, meta = converge_sym(A, chi=4, max_iter=60)
    assert meta["converged"], meta
    e0 = complex(sym_energy(A, env, gate))

    names = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
    for i, name in enumerate(names):
        phase = jnp.exp(1j * (0.3 + 0.4 * i))
        phased = env._replace(**{name: getattr(env, name) * phase})
        e = complex(sym_energy(A, phased, gate))
        assert abs(e - e0) < 1e-10 * max(abs(e0), 1.0), (name, e, e0)
    print(f"energy invariant under all eight environment phases at E = {e0.real:.12f}")


def test_explicit_backprop_through_the_symmetric_sweep_does_not_even_trace(
    converged_root,
):
    """On this path the implicit route is not the faster one — it is the only one.

    The dense module's counterpart of this test
    (``test_the_gradient_survives_where_explicit_backprop_nans``) is a
    *numerical* statement: at D=3 the SVD backward divides by ``s_i² - s_j²``
    on a degenerate discarded spectrum and returns NaN.  Per sector the
    obstruction is one level harder, and it bites at D=2.  ``sector_svd``
    truncates **globally across sectors**, which means ranking every sector's
    singular values against each other in Python — ``float(sv)`` — so a
    tracer never gets past the first sweep.  There is no gradient to be
    inaccurate.

    Which is also why the finite-difference check below differences the map
    rather than calling ``jax.grad`` on it.
    """
    import tenax

    A, _a, env, _projs, _root, _residual = converged_root
    gate = tenax.heisenberg_gate()

    def energy_explicit(A_live):
        env_next, _projs = sweep_sym(env, _double_layer_sym(A_live), 4, None)
        return sym_energy(A_live, env_next, gate)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.grad(energy_explicit)(A)


@pytest.mark.slow
def test_the_gradient_matches_the_dense_root_implicit_gradient(sym_gradient):
    """Eq. 18 twice on the same state: block sparse against dense, 3e-15.

    The dense route (``_ctm_root_implicit_asym``) is the verified Phase 1
    implementation, so this is the parity gate for the whole symmetric layer —
    forward, root, adjoint and energy boundary at once.  It is a genuine
    comparison and not a rerun of the same code: the dense path takes one
    global SVD of the whole cut and orders the renormalised bond by singular
    value, this one takes a per-sector SVD and orders the bond by charge.  The
    two bonds differ by a permutation, which is a gauge — so the *gradient*
    agrees while almost nothing in between does.

    The comparison is unmasked.  The dense gradient is free to have components
    that break ``A``'s charge structure and the symmetric one cannot; they come
    out 4.2e-16, because the energy of a symmetric state is even under the
    group action and so stationary along every symmetry-breaking direction.
    """
    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_root_implicit_energy_and_grad,
    )

    A, gate, chi, energy, grad, _diag = sym_gradient
    e_dense, g_dense = asym_root_implicit_energy_and_grad(
        _as_dense_site(A), gate, chi=chi
    )

    assert abs(float(energy) - float(e_dense)) < 1e-12, (energy, e_dense)
    g_sym = jnp.asarray(grad.todense())
    g_dense = jnp.asarray(g_dense)
    rel = float(jnp.linalg.norm(g_sym - g_dense) / jnp.linalg.norm(g_dense))
    assert rel < 1e-9, rel
    print(f"symmetric vs dense root-implicit gradient: {rel:.3e} relative")


@pytest.mark.slow
def test_the_gradient_matches_a_directional_finite_difference(
    sym_gradient, converged_root
):
    """A finite difference of the **sweep map from a fixed start**, not the root.

    Three things about this test are load-bearing and each was expensive to
    learn:

    * The reference is the *truncated* map — ``n`` sweeps from the converged
      environment — and it is therefore wrong by however much ``n`` is short.
      Measured here: 4.3e-4 at 12 sweeps, 7.7e-7 at 20, 5.1e-9 at 30, i.e. it
      converges **onto** the implicit gradient.  A too-short reference
      disagrees with the implicit gradient and agrees with its own finite
      difference to four digits at two step sizes, which is indistinguishable
      by inspection from a wrong gradient.  So the ladder is asserted, not just
      the endpoint.
    * The differentiated map starts from a **fixed** environment.  Finite
      differencing the root, or the re-converged energy, does not work: the
      root parametrisation is gauge-*discontinuous* in ``p``, and ``|ẏ_fd|``
      diverges as ``h`` shrinks (dense measured 1.98e4 -> 1.19e5) even though
      every point is a valid root to 1e-16.
    * The perturbation is charge preserving — a ``SymmetricTensor`` on ``A``'s
      own indices — because that is the only kind of direction ``dE/dA`` is
      defined along here.

    The pairing is ``Re Σ g·δA``, unconjugated, which is how JAX pairs a
    cotangent with a tangent.
    """
    A, gate, chi, _e, grad, _diag = sym_gradient
    # The converged environment the other fixture already paid for; both come
    # from the same seeded ``_convergent_site_tensor()`` at the same ``chi``.
    _A0, _a0, env0, _projs, _root, _residual = converged_root

    def energy_explicit(A_live, nsweeps):
        a_live = _double_layer_sym(A_live)
        env, projs = env0, None
        for _ in range(nsweeps):
            env, projs = sweep_sym(env, a_live, chi, projs)
        return float(sym_energy(A_live, env, gate))

    rng = np.random.RandomState(7)
    V = SymmetricTensor.random_normal_np(A.indices, rng)
    V = V * (1.0 / float(jnp.linalg.norm(V._data)))
    assert set(V.blocks) == set(grad.blocks)
    # Block by block rather than on the flat buffer, so the pairing does not
    # depend on two tensors having packed their blocks in the same order.
    analytic = float(
        sum(jnp.real(jnp.sum(grad.blocks[k] * V.blocks[k])) for k in grad.blocks)
    )
    assert abs(analytic) > 1e-6, analytic

    h = 1e-6
    rel = {}
    for nsweeps in (12, 30):
        plus = energy_explicit(A + V * h, nsweeps)
        minus = energy_explicit(A - V * h, nsweeps)
        fd = (plus - minus) / (2 * h)
        rel[nsweeps] = abs(fd - analytic) / abs(analytic)

    assert rel[30] < 1e-6, rel
    # The 12-sweep reference is *supposed* to disagree; if it did not, the
    # test would be insensitive to the thing it is checking.
    assert rel[12] > 100 * rel[30], rel
    print(
        f"<dE/dA, V> = {analytic:.12e}; finite difference of the sweep map "
        f"relative error {rel[12]:.2e} at 12 sweeps, {rel[30]:.2e} at 30"
    )


def test_same_block_structure_rejects_a_permuted_layout():
    """Keys and buffer length are not enough to license a flat-buffer compare.

    A layout that merely *permutes* multiplicities between sectors —
    ``{0: 1, 1: 2}`` to ``{0: 2, 1: 1}`` — keeps the diagonal key set
    ``((0, 0), (1, 1))`` and keeps the packed buffer length (``1**2 + 2**2``
    equals ``2**2 + 1**2``) while every block's shape and offset moves.  The
    original predicate compared only those two things and so returned ``True``,
    which would let :func:`converge_sym` subtract unrelated flat-buffer entries
    and declare convergence on a meaningless residual.

    Reported by automated review on PR #729 and reproduced before fixing.
    """
    sym = U1Symmetry()

    def corner(m0, m1):
        def leg(flow):
            return TensorIndex(
                symmetry=sym,
                sectors=np.array([0, 1]),
                multiplicities=np.array([m0, m1]),
                flow=flow,
                label="c",
            )

        return SymmetricTensor.random_normal_np(
            (leg(FlowDirection.OUT), leg(FlowDirection.IN)),
            np.random.RandomState(0),
        )

    a, b = corner(1, 2), corner(2, 1)
    # The trap: identical on both things the old predicate looked at.
    assert a._block_keys == b._block_keys
    assert a._data.shape == b._data.shape
    # ... but the blocks genuinely differ in shape.
    assert [blk.shape for blk in a.blocks.values()] != [
        blk.shape for blk in b.blocks.values()
    ]
    assert not _same_block_structure(a, b)
    # Unchanged behaviour on a genuinely identical structure.
    assert _same_block_structure(a, corner(1, 2))
