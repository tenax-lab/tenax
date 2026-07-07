import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
)
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv


def _match_axes(src, ref):
    """Return a permutation of src's axes so its labels line up with ref's."""
    ref_labels = list(ref.labels())
    src_labels = list(src.labels())
    return tuple(src_labels.index(lbl) for lbl in ref_labels)


def _random_dense_A(D=2, d=2, seed=0):
    """5-leg (u,d,l,r,phys) DenseTensor iPEPS site tensor, trivial symmetry."""
    from tenax.algorithms._ctm_utils import _trivial_symmetry
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import DenseTensor

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((D, D, D, D, d)) + 0j
    sym = _trivial_symmetry()

    def idx(n, flow, label):
        return TensorIndex.from_charges(
            sym, np.zeros(n, dtype=np.int32), flow, label=label
        )

    return DenseTensor(
        data,
        (
            idx(D, FlowDirection.OUT, "u"),
            idx(D, FlowDirection.OUT, "d"),
            idx(D, FlowDirection.OUT, "l"),
            idx(D, FlowDirection.OUT, "r"),
            idx(d, FlowDirection.OUT, "phys"),
        ),
    )


def test_initialize_split_multisite_env_keys_and_type():
    A = _random_dense_A(seed=1)
    B = _random_dense_A(seed=2)
    envs = _initialize_split_multisite_env({(0, 0): A, (1, 0): B}, chi=6, chi_I=6)
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    assert isinstance(envs[(0, 0)], SplitCTMTensorEnv)
    assert isinstance(envs[(1, 0)], SplitCTMTensorEnv)
    assert envs[(0, 0)].C1 is not envs[(1, 0)].C1


def test_split_multisite_uniform_matches_single_site():
    """recipe='1x1' multisite sweep on a uniform cell == single-site forward."""
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_multisite,
        ctm_split_tensor,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm_1site_split_tensor,
    )

    A = _random_dense_A(seed=3)
    chi = 6
    single = ctm_split_tensor(A, chi, max_iter=20, conv_tol=0.0)
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): A},
        CHECKERBOARD_NEIGHBORS,
        chi,
        max_iter=20,
        conv_tol=0.0,
        recipe="1x1",
    )
    rho_single = _rdm_1site_split_tensor(A, single)
    rho_multi = _rdm_1site_split_tensor(A, envs[(0, 0)])
    assert np.allclose(rho_single, rho_multi, atol=1e-8)


def _split_env_and_fused_env(A, chi):
    """Matched pair: split env and fused env from the same converged single-site
    CTM, for enlarged-corner parity (uniform 1x1, so envs are directly comparable)."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor

    fused, _eps = ctm_tensor(A, chi, max_iter=30, conv_tol=0.0)
    split = ctm_split_tensor(A, chi, chi_I=chi, max_iter=30, conv_tol=0.0)
    return split, fused


@pytest.mark.parametrize(
    "position", ["top_left", "top_right", "bottom_left", "bottom_right"]
)
def test_split_enlarged_corner_matches_fused(position):
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
    from tenax.algorithms._split_ctm_tensor_moves import _build_split_enlarged_corner

    A = _random_dense_A(seed=5)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    if position == "top_left":
        Q_ref = _build_enlarged_corner(
            fused.C1, fused.T1, fused.T4, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C1,
            split.T1_ket,
            split.T1_bra,
            split.T4_ket,
            split.T4_bra,
            A,
            A_bar,
            position=position,
        )
    elif position == "top_right":
        Q_ref = _build_enlarged_corner(
            fused.C2, fused.T1, fused.T2, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C2,
            split.T1_ket,
            split.T1_bra,
            split.T2_ket,
            split.T2_bra,
            A,
            A_bar,
            position=position,
        )
    elif position == "bottom_left":
        Q_ref = _build_enlarged_corner(
            fused.C4, fused.T3, fused.T4, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C4,
            split.T3_ket,
            split.T3_bra,
            split.T4_ket,
            split.T4_bra,
            A,
            A_bar,
            position=position,
        )
    else:  # bottom_right
        Q_ref = _build_enlarged_corner(
            fused.C3, fused.T3, fused.T2, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C3,
            split.T3_ket,
            split.T3_bra,
            split.T2_ket,
            split.T2_bra,
            A,
            A_bar,
            position=position,
        )
    # Align split axes to ref label order, compare magnitudes up to gauge/normalization.
    Qs = Q_split.transpose(_match_axes(Q_split, Q_ref))
    dr = Q_ref.todense()
    ds = Qs.todense()
    dr = dr / np.max(np.abs(dr))
    ds = ds / np.max(np.abs(ds))
    assert dr.shape == ds.shape
    assert np.allclose(np.abs(dr), np.abs(ds), atol=1e-6)


@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_split_plaquette_projector_matches_fused(direction):
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import _compute_plaquette_projector_pair
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
    )

    A = _random_dense_A(seed=7)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt_ref, Pb_ref, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, direction
    )
    Pt_s, Pb_s, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        direction,
    )
    # Projectors match up to per-column sign/gauge; compare sorted magnitudes.
    assert np.allclose(
        np.sort(np.abs(Pt_ref.todense()).ravel()),
        np.sort(np.abs(Pt_s.todense()).ravel()),
        atol=1e-6,
    )
    assert np.allclose(
        np.sort(np.abs(Pb_ref.todense()).ravel()),
        np.sort(np.abs(Pb_s.todense()).ravel()),
        atol=1e-6,
    )


def _abs_sorted(t):
    # Scale-invariant magnitude signature: the split absorb defers per-corner
    # normalization to the sweep-level _renormalize_split_env, whereas the fused
    # absorb phase-fix-normalizes in place, so the two match only up to a global
    # scale (physically irrelevant -- CTM renormalizes every sweep).
    d = np.abs(t.todense())
    m = np.max(d)
    if m > 0:
        d = d / m
    return np.sort(d.ravel())


def _corner_sv_signature(t):
    # Gauge-invariant per-move corner check: under the #425 degenerate-subspace
    # projector-basis freedom, the split absorb's 2-leg corner equals the fused
    # one only up to a unitary on the new bond, so raw entries differ but the
    # singular values (basis-independent under a unitary on either leg) agree.
    # We SVD the max-normalized corner directly (no elementwise abs -- that would
    # break the unitary invariance and weaken the check).
    #
    # Tolerance note: this per-move check is oracle-limited to ~1e-3, NOT machine
    # precision, because _split_env_and_fused_env runs two INDEPENDENT CTM solvers
    # (fused ctm_tensor vs split ctm_split_tensor) that settle on slightly
    # different fixed points (leading corner SV differs at ~1e-4 for some seeds).
    # It localizes gross per-direction bugs; the definitive element-wise gate is
    # the fixed-point energy parity in Task 1.5 (same A drives both forwards).
    m = t.todense()
    m = m / np.max(np.abs(m))
    return np.sort(np.linalg.svd(m, compute_uv=False))


def test_split_absorb_bottom_corners_match_fused():
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_bottom_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_bottom_2plaq,
    )

    A = _random_dense_A(seed=11)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, "bottom"
    )
    C4f, T3f, C3f = _ctm_tensor_absorb_bottom_2plaq(fused, a, Pt, Pb, Pt, Pb)
    sPt, sPb, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        "bottom",
    )
    C4s, T3k, T3b, C3s = _split_ctm_absorb_bottom_2plaq(
        split, A, A_bar, sPt, sPb, sPt, sPb, chi_I=chi
    )
    assert np.allclose(_abs_sorted(C4s), _abs_sorted(C4f), atol=1e-6)
    assert np.allclose(_abs_sorted(C3s), _abs_sorted(C3f), atol=1e-6)


def test_split_absorb_left_corners_match_fused():
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_left_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_left_2plaq,
    )

    A = _random_dense_A(seed=13)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, "left"
    )
    C1f, T4f, C4f = _ctm_tensor_absorb_left_2plaq(fused, a, Pt, Pb, Pt, Pb)
    sPt, sPb, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        "left",
    )
    C1s, T4k, T4b, C4s = _split_ctm_absorb_left_2plaq(
        split, A, A_bar, sPt, sPb, sPt, sPb, chi_I=chi
    )
    assert np.allclose(_corner_sv_signature(C1s), _corner_sv_signature(C1f), atol=5e-3)
    assert np.allclose(_corner_sv_signature(C4s), _corner_sv_signature(C4f), atol=5e-3)


def test_split_absorb_right_corners_match_fused():
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_right_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_right_2plaq,
    )

    A = _random_dense_A(seed=15)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, "right"
    )
    C2f, T2f, C3f = _ctm_tensor_absorb_right_2plaq(fused, a, Pt, Pb, Pt, Pb)
    sPt, sPb, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        "right",
    )
    C2s, T2k, T2b, C3s = _split_ctm_absorb_right_2plaq(
        split, A, A_bar, sPt, sPb, sPt, sPb, chi_I=chi
    )
    assert np.allclose(_corner_sv_signature(C2s), _corner_sv_signature(C2f), atol=5e-3)
    assert np.allclose(_corner_sv_signature(C3s), _corner_sv_signature(C3f), atol=5e-3)


def test_split_absorb_top_corners_match_fused():
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_top_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_top_2plaq,
    )

    A = _random_dense_A(seed=17)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, "top"
    )
    C1f, T1f, C2f = _ctm_tensor_absorb_top_2plaq(fused, a, Pt, Pb, Pt, Pb)
    sPt, sPb, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        "top",
    )
    C1s, T1k, T1b, C2s = _split_ctm_absorb_top_2plaq(
        split, A, A_bar, sPt, sPb, sPt, sPb, chi_I=chi
    )
    assert np.allclose(_corner_sv_signature(C1s), _corner_sv_signature(C1f), atol=5e-3)
    assert np.allclose(_corner_sv_signature(C2s), _corner_sv_signature(C2f), atol=5e-3)


def test_split_2x2_sweep_runs_and_preserves_uniform():
    from tenax.algorithms._split_ctm_tensor_convergence import _split_ctm_multisite

    A = _random_dense_A(seed=12)
    chi = 6
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): A},
        CHECKERBOARD_NEIGHBORS,
        chi,
        max_iter=8,
        conv_tol=0.0,
        recipe="2x2",
    )
    C1 = envs[(0, 0)].C1.todense()
    assert np.all(np.isfinite(C1)) and np.max(np.abs(C1)) > 0


def test_ctm_split_tensor_2site_is_publicly_exported():
    """``ctm_split_tensor_2site`` is importable from the top-level ``tenax``.

    Its energy sibling ``compute_energy_split_ctm_tensor_2site`` is already a
    public export; a user who obtains the coupled 2-site split environment via
    ``ctm_split_tensor_2site`` must be able to import it the same way rather than
    reach into the private ``_split_ctm_tensor`` shim (#684).
    """
    import tenax
    from tenax.algorithms._split_ctm_tensor_convergence import (
        ctm_split_tensor_2site as _canonical,
    )

    assert "ctm_split_tensor_2site" in tenax.__all__
    assert hasattr(tenax, "ctm_split_tensor_2site")
    assert tenax.ctm_split_tensor_2site is _canonical


def test_ctm_split_tensor_2site_returns_two_envs():
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site

    A = _random_dense_A(seed=13)
    B = _random_dense_A(seed=14)
    env_A, env_B = ctm_split_tensor_2site(A, B, chi=6, max_iter=10, conv_tol=0.0)
    assert isinstance(env_A, SplitCTMTensorEnv)
    assert isinstance(env_B, SplitCTMTensorEnv)
    # Genuinely coupled: A's and B's envs must differ (distinct sublattices).
    assert not np.allclose(env_A.C1.todense(), env_B.C1.todense())


# ------------------------------------------------------------------ #
# Per-move EDGE parity (decoupled): localize the split-absorb edge bug #
# ------------------------------------------------------------------ #

# Fused-absorb edge output labels: (chi_left, D2, chi_right).
# Split-merge (via _merge_edge) reconstructs the SAME rank-3 object from the
# ket/bra halves. We compare gauge-invariant singular values matricizing as
# (chi_left | D2 . chi_right).
_EDGE_MERGE = {
    # direction: (ket_I, bra_I, d_ket, d_bra, fused, flow, ket_chi, bra_chi,
    #             std_chi_l, std_chi_r)
    "bottom": (
        "t3k_I",
        "t3b_I",
        "d_ket",
        "d_bra",
        "d2",
        "OUT",
        "t3k_r",
        "t3b_l",
        "t3_r",
        "t3_l",
    ),
    "left": (
        "t4k_I",
        "t4b_I",
        "l_ket",
        "l_bra",
        "l2",
        "IN",
        "t4k_d",
        "t4b_u",
        "t4_d",
        "t4_u",
    ),
    "right": (
        "t2k_I",
        "t2b_I",
        "r_ket",
        "r_bra",
        "r2",
        "OUT",
        "t2k_u",
        "t2b_d",
        "t2_u",
        "t2_d",
    ),
    "top": (
        "t1k_I",
        "t1b_I",
        "u_ket",
        "u_bra",
        "u2",
        "IN",
        "t1k_l",
        "t1b_r",
        "t1_l",
        "t1_r",
    ),
}
# Fused-absorb edge output rank-3 label order (chi_left, D2, chi_right).
_FUSED_EDGE_LABELS = {
    "bottom": ("t3_r", "d2", "t3_l"),
    "left": ("t4_d", "l2", "t4_u"),
    "right": ("t2_u", "r2", "t2_d"),
    "top": ("t1_l", "u2", "t1_r"),
}


def _merge_split_edge(T_ket, T_bra, direction):
    from tenax.algorithms._tensor_utils import fuse_indices
    from tenax.contraction.contractor import contract
    from tenax.core.index import FlowDirection

    (ket_I, bra_I, d_ket, d_bra, fused, flow, ket_chi, bra_chi, std_l, std_r) = (
        _EDGE_MERGE[direction]
    )
    fflow = FlowDirection.OUT if flow == "OUT" else FlowDirection.IN
    k = T_ket.relabel(ket_I, "_I")
    b = T_bra.relabel(bra_I, "_I")
    merged = contract(k, b)
    labels = merged.labels()
    merged = fuse_indices(
        merged, labels.index(d_ket), labels.index(d_bra), fused, fflow
    )
    merged = merged.relabels({ket_chi: std_l, bra_chi: std_r})
    return merged


def _edge_sv_signature(edge, labels):
    # Matricize as (chi_left | D2 . chi_right); compare sorted SVs normalized by
    # the leading SV. Invariant under independent unitaries on the two chi bonds
    # (the two envs come from independent solvers with chi in different gauges).
    m = edge.transpose(_match_axes(edge, edge)) if False else edge
    lbl = list(m.labels())
    perm = tuple(lbl.index(x) for x in labels)
    dense = m.todense().transpose(perm)
    cl, d2, cr = dense.shape
    mat = dense.reshape(cl, d2 * cr)
    sv = np.linalg.svd(mat, compute_uv=False)
    sv = sv / sv[0]
    return np.sort(sv)


def _split_absorb_edge_for_direction(direction):
    """Run fused + split absorb (uniform projectors) and return the two
    reconstructed rank-3 edges plus the fused-edge label order."""
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_bottom_2plaq,
        _ctm_tensor_absorb_left_2plaq,
        _ctm_tensor_absorb_right_2plaq,
        _ctm_tensor_absorb_top_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_bottom_2plaq,
        _split_ctm_absorb_left_2plaq,
        _split_ctm_absorb_right_2plaq,
        _split_ctm_absorb_top_2plaq,
    )

    fused_absorb = {
        "bottom": _ctm_tensor_absorb_bottom_2plaq,
        "left": _ctm_tensor_absorb_left_2plaq,
        "right": _ctm_tensor_absorb_right_2plaq,
        "top": _ctm_tensor_absorb_top_2plaq,
    }[direction]
    split_absorb = {
        "bottom": _split_ctm_absorb_bottom_2plaq,
        "left": _split_ctm_absorb_left_2plaq,
        "right": _split_ctm_absorb_right_2plaq,
        "top": _split_ctm_absorb_top_2plaq,
    }[direction]

    A = _random_dense_A(seed=11)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()

    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, direction
    )
    _c1, edge_fused, _c2 = fused_absorb(fused, a, Pt, Pb, Pt, Pb)

    sPt, sPb, _, _ = _compute_split_plaquette_projector_pair(
        split,
        split,
        split,
        split,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        A,
        A_bar,
        chi,
        direction,
    )
    # The 2x2 grown+projected edge has interlayer rank up to min(chi*D, D**3)
    # (the fused edge's ket|bra seam is genuinely full — verified rank 8 for
    # D=2,chi=6), so the ket/bra SVD split must retain at least that many
    # interlayer values to reproduce the fused edge losslessly.  A chi_I=chi
    # SVD split truncates this and is lossy (worst on the TOP edge); pass an
    # ample chi_I here so the check isolates absorb-logic errors, not the
    # (separately-tracked) interlayer-truncation limitation.
    chi_I = 2 * chi
    _s1, edge_ket, edge_bra, _s2 = split_absorb(
        split, A, A_bar, sPt, sPb, sPt, sPb, chi_I=chi_I
    )
    edge_split = _merge_split_edge(edge_ket, edge_bra, direction)
    return edge_fused, edge_split, _FUSED_EDGE_LABELS[direction]


@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_split_absorb_edge_matches_fused(direction):
    """Per-move edge parity (gauge-invariant SV signature).

    Decouples each absorb from the fixed-point loop: applies fused + split
    absorb once each on the SAME uniform single-site env, reconstructs the
    fused-form rank-3 edge from the split ket/bra pair, and compares sorted
    singular values (matricized chi_left | D^2.chi_right). Oracle-limited to
    ~1e-3 (two independent CTM solvers); a real bug shows as >10%.
    """
    edge_fused, edge_split, labels = _split_absorb_edge_for_direction(direction)
    sv_f = _edge_sv_signature(edge_fused, labels)
    sv_s = _edge_sv_signature(edge_split, labels)
    assert len(sv_f) == len(sv_s)
    assert np.allclose(sv_f, sv_s, atol=5e-3), (
        f"{direction}: max sv diff {np.max(np.abs(sv_f - sv_s)):.3e}\n"
        f"fused={sv_f}\nsplit={sv_s}"
    )


# ------------------------------------------------------------------ #
# Shared-env absorb BIT-PARITY (convergence-independent correctness). #
# ------------------------------------------------------------------ #

# Per-direction merge spec for the split ABSORB OUTPUT edge halves.  The split
# absorb output edges carry the standard split labels (see the _svd_split_edge_
# tensor relabels in _split_ctm_tensor_moves); this maps them to the FUSED
# absorb output edge form for the same direction.
_ABSORB_MERGE_SPEC = {
    "left": dict(
        ket_I="t4k_I",
        bra_I="t4b_I",
        d_ket="l_ket",
        d_bra="l_bra",
        fused_label="l2",
        fused_flow="IN",
        ket_chi="t4k_d",
        bra_chi="t4b_u",
        std_chi_l="t4_d",
        std_chi_r="t4_u",
    ),
    "right": dict(
        ket_I="t2k_I",
        bra_I="t2b_I",
        d_ket="r_ket",
        d_bra="r_bra",
        fused_label="r2",
        fused_flow="OUT",
        ket_chi="t2k_u",
        bra_chi="t2b_d",
        std_chi_l="t2_u",
        std_chi_r="t2_d",
    ),
    "top": dict(
        ket_I="t1k_I",
        bra_I="t1b_I",
        d_ket="u_ket",
        d_bra="u_bra",
        fused_label="u2",
        fused_flow="IN",
        ket_chi="t1k_l",
        bra_chi="t1b_r",
        std_chi_l="t1_l",
        std_chi_r="t1_r",
    ),
    "bottom": dict(
        ket_I="t3k_I",
        bra_I="t3b_I",
        d_ket="d_ket",
        d_bra="d_bra",
        fused_label="d2",
        fused_flow="OUT",
        ket_chi="t3k_r",
        bra_chi="t3b_l",
        std_chi_l="t3_r",
        std_chi_r="t3_l",
    ),
}


def _merge_absorb_edge(
    T_ket,
    T_bra,
    ket_I,
    bra_I,
    d_ket,
    d_bra,
    fused_label,
    fused_flow,
    ket_chi,
    bra_chi,
    std_chi_l,
    std_chi_r,
):
    """Replicate _split_env_to_tensor_standard._merge_edge on a single edge."""
    from tenax.algorithms._tensor_utils import fuse_indices
    from tenax.contraction.contractor import contract
    from tenax.core.index import FlowDirection

    fflow = FlowDirection.OUT if fused_flow == "OUT" else FlowDirection.IN
    k = T_ket.relabel(ket_I, "_I")
    b = T_bra.relabel(bra_I, "_I")
    merged = contract(k, b)
    labels = merged.labels()
    merged = fuse_indices(
        merged, labels.index(d_ket), labels.index(d_bra), fused_label, fflow
    )
    merged = merged.relabels({ket_chi: std_chi_l, bra_chi: std_chi_r})
    return merged


def _one_minus_fidelity(x, y):
    """1 - scale/phase-invariant overlap fidelity, aligning axes by label."""
    lx = list(x.labels())
    ly = list(y.labels())
    assert sorted(lx) == sorted(ly), f"label sets differ:\n {lx}\n {ly}"
    perm = tuple(lx.index(lbl) for lbl in ly)  # permute x -> y's order
    ax = np.asarray(x.transpose(perm).todense()).ravel()
    ay = np.asarray(y.todense()).ravel()
    nx = np.linalg.norm(ax)
    ny = np.linalg.norm(ay)
    assert nx > 0 and ny > 0
    return 1.0 - abs(np.vdot(ax, ay)) / (nx * ny)


@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_split_absorb_bitidentical_on_shared_env(direction):
    """Split 2x2 absorb == fused 2x2 absorb on ONE shared env (machine precision).

    This is the convergence-INDEPENDENT correctness proof for the split absorb
    kernels.  We build a generic split env, merge it to the fused representation
    via _split_env_to_tensor_standard, compute a single set of fused projectors,
    then apply the fused absorb and the split absorb with the SAME projectors and
    a lossless interlayer bond (chi_I=64).  Because both absorbs see the identical
    env and identical projectors, they must agree to machine precision for the two
    corners and the merged edge -- no fixed-point convergence is involved.
    """
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_moves import (
        _compute_plaquette_projector_pair,
        _ctm_tensor_absorb_bottom_2plaq,
        _ctm_tensor_absorb_left_2plaq,
        _ctm_tensor_absorb_right_2plaq,
        _ctm_tensor_absorb_top_2plaq,
    )
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_tensor_sweep,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_env_to_tensor_standard,
    )
    from tenax.algorithms._split_ctm_tensor_init import (
        initialize_split_ctm_tensor_env,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _split_ctm_absorb_bottom_2plaq,
        _split_ctm_absorb_left_2plaq,
        _split_ctm_absorb_right_2plaq,
        _split_ctm_absorb_top_2plaq,
    )

    fused_fn = {
        "left": _ctm_tensor_absorb_left_2plaq,
        "right": _ctm_tensor_absorb_right_2plaq,
        "top": _ctm_tensor_absorb_top_2plaq,
        "bottom": _ctm_tensor_absorb_bottom_2plaq,
    }[direction]
    split_fn = {
        "left": _split_ctm_absorb_left_2plaq,
        "right": _split_ctm_absorb_right_2plaq,
        "top": _split_ctm_absorb_top_2plaq,
        "bottom": _split_ctm_absorb_bottom_2plaq,
    }[direction]

    A = _random_dense_A(D=2, seed=11)
    A_bar = A.bar()
    a = _build_double_layer_tensor(A)
    chi = 6

    # Build a generic (non-uniform) split env via init + a few sweeps.
    E_s = initialize_split_ctm_tensor_env(A, chi, chi)
    for _ in range(5):
        E_s = _split_ctm_tensor_sweep(E_s, A, chi, chi, True)
    E_f = _split_env_to_tensor_standard(E_s)

    spec = _ABSORB_MERGE_SPEC[direction]

    # Self-guard: our single-edge merge must reproduce _split_env_to_tensor_
    # standard's edge for this direction to ~1e-12, else the comparison below
    # could silently pass on a broken merge.
    edge_ref = {
        "left": (E_s.T4_ket, E_s.T4_bra, E_f.T4),
        "right": (E_s.T2_ket, E_s.T2_bra, E_f.T2),
        "top": (E_s.T1_ket, E_s.T1_bra, E_f.T1),
        "bottom": (E_s.T3_ket, E_s.T3_bra, E_f.T3),
    }[direction]
    rt = _merge_absorb_edge(edge_ref[0], edge_ref[1], **spec)
    assert _one_minus_fidelity(rt, edge_ref[2]) < 1e-12

    # Single shared set of fused projectors drives BOTH absorbs.
    Pt, Pb, _, _ = _compute_plaquette_projector_pair(
        E_f, E_f, E_f, E_f, a, a, a, a, chi, direction
    )
    cf1, ef, cf2 = fused_fn(E_f, a, Pt, Pb, Pt, Pb)
    cs1, ek, eb, cs2 = split_fn(E_s, A, A_bar, Pt, Pb, Pt, Pb, 64)
    es_edge = _merge_absorb_edge(ek, eb, **spec)

    assert _one_minus_fidelity(cs1, cf1) < 1e-10
    assert _one_minus_fidelity(cs2, cf2) < 1e-10
    assert _one_minus_fidelity(es_edge, ef) < 1e-10


# ------------------------------------------------------------------ #
# Convergent-input 2-site energy parity (corrected Tier-2 gate).     #
# ------------------------------------------------------------------ #
#
# Random site tensors are deliberately NOT used here: the fused 2-site CTM
# oscillates (does not converge) on raw random input, so split-vs-fused energy
# parity on random tensors is meaningless.  A physical, convergent input is
# required.  We build a Heisenberg Neel iPEPS via 2-site simple update, confirm
# the fused CTM oracle plateaus, then compare split vs fused energy.


def _heisenberg_gate(d=2):
    import jax.numpy as jnp

    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def _build_su_neel(D=2, d=2, n_steps=80, dt=0.05):
    """Physical Heisenberg checkerboard (A,B) via 2-site simple update.

    ~80 imaginary-time steps is enough for the *fused CTM oracle* to plateau
    (that is what the parity test needs -- not fully-converged SU energy).
    """
    import jax
    import jax.numpy as jnp

    from tenax.algorithms.ipeps import (
        _make_trotter_gate_tensor,
        _wrap_as_dense_tensor,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_2site_horizontal_tensor,
        _simple_update_2site_vertical_tensor,
    )

    H = _heisenberg_gate(d)
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    A_data = 0.1 * jax.random.normal(kA, (D, D, D, D, d), dtype=jnp.float64)
    B_data = 0.1 * jax.random.normal(kB, (D, D, D, D, d), dtype=jnp.float64)
    # Neel bias: physical component 0 for A, 1 for B.
    A_data = A_data.at[0, 0, 0, 0, 0].add(1.0)
    B_data = B_data.at[0, 0, 0, 0, 1].add(1.0)
    A = _wrap_as_dense_tensor(A_data)
    B = _wrap_as_dense_tensor(B_data)
    A = A * (1.0 / float(A.norm()))
    B = B * (1.0 / float(B.norm()))

    gate = _make_trotter_gate_tensor(H, dt, site_tensor=A)
    lam_h = jnp.ones(D)
    lam_v = jnp.ones(D)
    for step in range(n_steps):
        if step % 2 == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, D
            )
        else:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, D
            )
        A = A * (1.0 / float(A.norm()))
        B = B * (1.0 / float(B.norm()))
    return A, B


def test_split_2site_energy_matches_fused_convergent():
    """Joint 2-site split forward matches fused energy on a convergent input.

    On a physical Heisenberg Neel SU state where the fused 2-site CTM oracle
    plateaus, the split forward reproduces the fused energy to ~1e-11 at a
    lossless interlayer bond (chi_I = 2*chi) and to ~1e-10 at chi_I = chi.
    Physical states have low interlayer rank, so chi_I = chi is already
    near-lossless (hence the tight 1e-6 tolerance is easily met).
    """
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import (
        compute_energy_ctm_tensor_2site,
    )
    from tenax.algorithms._split_ctm_tensor_convergence import (
        ctm_split_tensor_2site,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    gate = _heisenberg_gate()
    chi = 8
    A, B = _build_su_neel(D=2)

    def fused_energy(max_iter):
        envA, envB = ctm_tensor_2site(A, B, chi, max_iter=max_iter, conv_tol=1e-12)
        return float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))

    def split_energy(chi_I, max_iter):
        envA, envB = ctm_split_tensor_2site(
            A, B, chi, max_iter=max_iter, conv_tol=1e-12, chi_I=chi_I
        )
        return float(compute_energy_split_ctm_tensor_2site(A, B, envA, envB, gate, d=2))

    # (i) Fused oracle must plateau -- else the parity comparison is meaningless.
    E_fused_80 = fused_energy(80)
    E_fused = fused_energy(100)
    assert abs(E_fused_80 - E_fused) < 1e-8, (
        f"fused oracle did not plateau: |E(80)-E(100)|={abs(E_fused_80 - E_fused):.3e}"
    )

    # (ii) Lossless interlayer bond (chi_I = 2*chi): near machine precision.
    E_split_lossless = split_energy(2 * chi, 100)
    assert abs(E_split_lossless - E_fused) < 1e-8

    # (iii) chi_I = chi: physical states have low interlayer rank, so this is
    # near-lossless in practice (comfortably under 1e-6).
    E_split_chi = split_energy(chi, 100)
    assert abs(E_split_chi - E_fused) < 1e-6


def test_split_2site_energy_equals_multisite():
    """The dedicated 2-site energy path == the generic multisite N=2 path.

    Both ``compute_energy_split_ctm_tensor_2site`` and
    ``compute_energy_split_ctm_tensor_multisite`` exist; on a convergent coupled
    split env they must produce bit-identical energies (they count the same
    4 checkerboard NN bonds, normalised by 2 sites). This guards against future
    divergence of the two code paths.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        ctm_split_tensor_2site,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
        compute_energy_split_ctm_tensor_multisite,
    )

    gate = _heisenberg_gate()
    chi = 8
    A, B = _build_su_neel(D=2)

    env_A, env_B = ctm_split_tensor_2site(
        A, B, chi, max_iter=100, conv_tol=1e-12, chi_I=chi
    )

    E_2site = float(
        compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2)
    )

    site_tensors = {(0, 0): A, (1, 0): B}
    envs = {(0, 0): env_A, (1, 0): env_B}
    E_multi = float(
        compute_energy_split_ctm_tensor_multisite(
            site_tensors, envs, CHECKERBOARD_NEIGHBORS, gate, d=2
        )
    )

    assert abs(E_2site - E_multi) < 1e-10, (
        f"2-site vs multisite energy diverge: "
        f"E_2site={E_2site:.15f}, E_multi={E_multi:.15f}, "
        f"|diff|={abs(E_2site - E_multi):.3e}"
    )
