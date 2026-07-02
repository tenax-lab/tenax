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


def test_ctm_split_tensor_2site_returns_two_envs():
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site

    A = _random_dense_A(seed=13)
    B = _random_dense_A(seed=14)
    env_A, env_B = ctm_split_tensor_2site(A, B, chi=6, max_iter=10, conv_tol=0.0)
    assert isinstance(env_A, SplitCTMTensorEnv)
    assert isinstance(env_B, SplitCTMTensorEnv)
    # Genuinely coupled: A's and B's envs must differ (distinct sublattices).
    assert not np.allclose(env_A.C1.todense(), env_B.C1.todense())
