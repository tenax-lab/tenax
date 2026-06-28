import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard

pytestmark = pytest.mark.core


def _oracle():
    # import helper module robustly whether or not `tests` is a package
    try:
        from tests._split_ctm_oracle import (
            fused_env_to_split,
            heisenberg_gate,
            make_site,
        )
    except ModuleNotFoundError:
        from _split_ctm_oracle import fused_env_to_split, heisenberg_gate, make_site
    return make_site, heisenberg_gate, fused_env_to_split


@pytest.mark.parametrize("D", [2, 3])
def test_split_corner_init_is_rank1(D):
    """Split corner init must be rank-1 (variPEPS ``chi_init=1``).

    A rank-``min(chi,D)`` identity seed (``eye(min(chi,D))``) drove the split
    env onto an *artificially* degenerate corner fixed point (e.g.
    ``[0.5, 0.5, 0, 0]`` for D=2) whose degenerate subspace rotates each sweep,
    blocking element-wise convergence and implicit-AD fixed-point
    differentiation (#463).  A rank-1 seed matches the fused
    ``_make_rank1_dense_corner`` and converges element-wise.
    """
    from tenax.algorithms._split_ctm_tensor_init import (
        initialize_split_ctm_tensor_env,
    )

    make_site, _, _ = _oracle()
    A = make_site(D, 2, seed=7)
    chi = 4
    env = initialize_split_ctm_tensor_env(A, chi, chi)
    for name in ("C1", "C2", "C3", "C4"):
        C = np.asarray(getattr(env, name).todense())
        s = np.linalg.svd(C, compute_uv=False)
        rank = int(np.sum(s > 1e-10 * (s[0] + 1e-30)))
        assert rank == 1, f"{name} init rank={rank} (spectrum {np.round(s, 3)})"


@pytest.mark.parametrize("D", [2, 3])
def test_fused_to_split_roundtrip(D):
    make_site, heisenberg_gate, fused_env_to_split = _oracle()
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=8, max_iter=200, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = fused_env_to_split(fused_env, D, chi_I=8 * D)
    rt_env = _split_env_to_tensor_standard(split_env)
    E_rt = float(compute_energy_ctm_tensor(A, rt_env, gate))
    np.testing.assert_allclose(E_rt, E_fused, atol=1e-8)


from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor


@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 6)])
def test_split_matches_fused_lossless_chi_I(D, chi):
    # With the double-layer corner-pair projector and a lossless interlayer bond
    # (chi_I = chi*D), the split path is an exact factorization of the fused path
    # at the same chi. We force full convergence (conv_tol=0.0 -> run all
    # max_iter sweeps) on BOTH paths: the corner-singular-value early-break
    # criterion is unreliable for the low-rank corners of a random tensor (it
    # plateaus on a transient for the fused oracle too), so we compare the true
    # fixed points. A robust production convergence criterion is DL-Task 6.
    make_site, heisenberg_gate, fused_env_to_split = _oracle()
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=chi, max_iter=300, conv_tol=0.0)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = ctm_split_tensor(A, chi=chi, chi_I=chi * D, max_iter=300, conv_tol=0.0)
    E_split = float(compute_energy_split_ctm_tensor(A, split_env, gate))
    np.testing.assert_allclose(E_split, E_fused, atol=1e-8)


def test_factorize_projector_reconstructs():
    # P over (env, ketD, braD) -> chi factorizes exactly into P_first . P_second
    import jax.numpy as jnp  # noqa: F401

    from tenax.algorithms._split_ctm_tensor_moves import _factorize_projector
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    sym = U1Symmetry()
    env, Dk, Db, chi = 4, 2, 2, 5
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (env, Dk, Db, chi))
    z = lambda n: __import__("numpy").zeros(n, dtype="int32")  # noqa: E731
    idx = [
        TensorIndex.from_charges(sym, z(env), FlowDirection.IN, label="env"),
        TensorIndex.from_charges(sym, z(Dk), FlowDirection.IN, label="ketD"),
        TensorIndex.from_charges(sym, z(Db), FlowDirection.IN, label="braD"),
        TensorIndex.from_charges(sym, z(chi), FlowDirection.OUT, label="chi_new"),
    ]
    P = DenseTensor(data, idx)
    P_first, P_second, m = _factorize_projector(P, "env", "ketD", "braD", "chi_new")
    # contract P_first . P_second over the factorization bond -> reconstruct P
    from tenax.contraction.contractor import contract

    P_rec = contract(P_first, P_second)
    # compare dense values up to leg order
    a = np.asarray(P.todense())
    b = np.asarray(
        P_rec.transpose(
            tuple(
                P_rec.labels().index(lbl) for lbl in ["env", "ketD", "braD", "chi_new"]
            )
        ).todense()
    )
    np.testing.assert_allclose(b, a, atol=1e-10)


def test_split_production_chi_I_converges_to_lossless():
    """Production interlayer bond (chi_I=chi) is physical and tracks lossless.

    Spec oracle 2: with the production interlayer bond chi_I=chi the split
    energy must stay physical (<=0.75/bond) and match the lossless
    (chi_I=chi*D) fixed point.  With the rank-1 (variPEPS ``chi_init=1``)
    corner init (#463) the split env stays low-rank enough that chi_I=chi is
    already lossless even at D=3 — the interlayer-truncation error is ~0, not
    merely shrinking.  (The previous rank-``min(chi,D)`` corner seed left a
    nonzero interlayer error that decreased as chi grew; the rank-1 seed
    removes it entirely while reaching the *same* converged energy.)
    conv_tol=0.0 forces full sweeps so we compare true fixed points (the
    corner-SV criterion is blind to the degenerate corner; see DL-Task 6).
    """
    make_site, heisenberg_gate, _ = _oracle()
    D = 3
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()

    interlayer_err = []
    for chi in (4, 6):
        e_lossy = float(
            compute_energy_split_ctm_tensor(
                A,
                ctm_split_tensor(A, chi=chi, chi_I=chi, max_iter=200, conv_tol=0.0),
                gate,
            )
        )
        e_lossless = float(
            compute_energy_split_ctm_tensor(
                A,
                ctm_split_tensor(A, chi=chi, chi_I=chi * D, max_iter=200, conv_tol=0.0),
                gate,
            )
        )
        # E is the sum over the 2 NN bonds of the 1-site cell -> per-bond = E/2.
        assert abs(e_lossy / 2.0) <= 0.75 + 1e-6, (
            f"unphysical per-bond energy {e_lossy / 2.0}"
        )
        assert abs(e_lossless / 2.0) <= 0.75 + 1e-6
        interlayer_err.append(abs(e_lossy - e_lossless))

    # The interlayer-truncation error (chi_I=chi vs lossless at the SAME chi)
    # is non-increasing in chi and already converged (~0) under the rank-1
    # corner init: the production interlayer bond tracks the lossless fixed
    # point to machine precision.
    assert interlayer_err[1] <= interlayer_err[0] + 1e-12
    assert interlayer_err[-1] < 1e-6


@pytest.mark.parametrize("min_iter,expected_sweeps", [(2, 2), (5, 5), (8, 8)])
def test_split_min_iter_floor_blocks_early_break(
    monkeypatch, min_iter, expected_sweeps
):
    """The min_iter floor must defer the conv_tol break past the transient.

    This is a *mechanism* test (per the project's convergence-test guidance):
    with a huge ``conv_tol`` the corner singular-value criterion would break at
    the earliest opportunity (the 2nd sweep, once ``prev_sv`` exists). The
    ``min_iter`` floor must instead force exactly ``min_iter`` sweeps before the
    early break may fire. Actual physical convergence is validated by
    ``test_split_matches_fused_lossless_chi_I`` (conv_tol=0.0, forced sweeps).
    """
    make_site, _, _ = _oracle()
    import tenax.algorithms._split_ctm_tensor_convergence as C

    A = make_site(2, 2, seed=7)
    calls = {"n": 0}
    real_sweep = C._split_ctm_tensor_sweep

    def counting_sweep(*args, **kwargs):
        calls["n"] += 1
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(C, "_split_ctm_tensor_sweep", counting_sweep)
    # conv_tol=1e9 -> the SV diff is always below tol, so the loop breaks at the
    # first sweep allowed by the min_iter floor.
    C.ctm_split_tensor(A, chi=4, chi_I=8, max_iter=50, conv_tol=1e9, min_iter=min_iter)
    assert calls["n"] == expected_sweeps


@pytest.mark.parametrize("D,chi", [(2, 4), (3, 6)])
def test_split_bounded_equals_closed(D, chi):
    """The bounded chi^2*D^4 edge path must reproduce the closed chi^2*D^6 one.

    Two complementary gates, both to 1e-10:

    1. **Exact equality** of the 4-leg *projected edge* ``T*g`` (the direct
       output of the bounded vs closed edge application, before the interlayer
       SVD split) for the **left** move.  This is the precise
       ``_grow_and_project_bounded_lr == _project_grown_edge_tensor_lr`` claim;
       the other three moves are covered by
       ``tests/test_split_ctm_tensor.py::TestSplitCTMBoundedEdge::test_bounded_matches_closed_path_all_moves``.
    2. **Energy after a full sweep** (gauge-invariant).  The interlayer (``_I``)
       SVD that splits each projected edge into ket/bra halves is genuinely
       non-unique on the doubly-degenerate edge of a random seeded site (two
       SVDs whose inputs differ at 1e-16 pick a different rotation inside the
       degenerate subspace), so the raw ``T*_ket``/``T*_bra`` and the corners
       that subsequently absorb them are gauge-dependent.  The physical energy
       is gauge-invariant and equals to machine precision.
    """
    import jax.numpy as jnp

    import tenax.algorithms._split_ctm_tensor_moves as M
    from tenax.algorithms._ctm_projector import _compute_projector_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_tensor_sweep,
    )
    from tenax.algorithms._split_ctm_tensor_init import (
        initialize_split_ctm_tensor_env,
    )

    make_site, heisenberg_gate, _ = _oracle()
    A = make_site(D, 2, seed=7)
    A_bar = A.bar()
    env = initialize_split_ctm_tensor_env(A, chi, chi * D)

    # --- Gate 1: exact per-move projected-edge equality -----------------
    # Each tuple: (grow args for closed, left_fuse, right_fuse, corner specs)
    Dd = A.indices[0].dim

    def _bounded_vs_closed_edge(T_ket, T_bra, leg, kI, bI, grow_out, lf, rf, P_1, P_2):
        Tg = M._grow_edge_no_double_layer(T_ket, T_bra, A, A_bar, leg, kI, bI, grow_out)
        closed = M._project_grown_edge_tensor_lr(
            Tg, P_1, P_2, left_fuse=lf, right_fuse=rf
        )
        P_1u = M._unfuse_projector_fused(
            P_1, P_1.indices[0].dim // (Dd * Dd), Dd, "env", "ketD", "braD"
        )
        P_2u = M._unfuse_projector_fused(
            P_2, P_2.indices[0].dim // (Dd * Dd), Dd, "env", "ketD", "braD"
        )
        bounded = M._grow_and_project_bounded_lr(
            T_ket, T_bra, A, A_bar, P_1u, P_2u, leg, kI, bI, lf, rf
        )
        perm = tuple(bounded.labels().index(lbl) for lbl in closed.labels())
        a = closed.todense()
        b = bounded.transpose(perm).todense()
        scale = float(jnp.max(jnp.abs(a))) + 1e-30
        return float(jnp.max(jnp.abs(a - b))) / scale

    # Left move
    C1g, _ = M._doublelayer_grown_corner(
        env.C1,
        env.T1_ket,
        env.T1_bra,
        ("c1_r", "t1k_l"),
        "t1k_I",
        "t1b_I",
        ("c1_d", "u_ket", "u_bra"),
    )
    C4g, _ = M._doublelayer_grown_corner(
        env.C4,
        env.T3_ket,
        env.T3_bra,
        ("c4_r", "t3k_r"),
        "t3k_I",
        "t3b_I",
        ("c4_u", "d_ket", "d_bra"),
    )
    P_1, P_2, _ = _compute_projector_tensor(C1g, C4g, chi, base_charges=None)
    relerr = _bounded_vs_closed_edge(
        env.T4_ket,
        env.T4_bra,
        "l",
        "t4k_I",
        "t4b_I",
        ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
        ("t4k_d", "u", "U"),
        ("t4b_u", "d", "D"),
        P_1,
        P_2,
    )
    assert relerr < 1e-10, f"left edge relerr={relerr:.2e}"

    # --- Gate 2: gauge-invariant energy after a full sweep --------------
    gate = heisenberg_gate()
    saved = M._FORCE_CLOSED_EDGE
    try:
        M._FORCE_CLOSED_EDGE = True
        e_closed = env
        for _ in range(3):
            e_closed = _split_ctm_tensor_sweep(e_closed, A, chi, chi * D, True)
        M._FORCE_CLOSED_EDGE = False
        e_bounded = env
        for _ in range(3):
            e_bounded = _split_ctm_tensor_sweep(e_bounded, A, chi, chi * D, True)
    finally:
        M._FORCE_CLOSED_EDGE = saved

    E_closed = float(compute_energy_split_ctm_tensor(A, e_closed, gate))
    E_bounded = float(compute_energy_split_ctm_tensor(A, e_bounded, gate))
    np.testing.assert_allclose(E_bounded, E_closed, atol=1e-10)
