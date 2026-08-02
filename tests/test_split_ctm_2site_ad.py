"""#463 Phase 2 — dense 2-site split-CTM AD (explicit + implicit).

Parity is validated on PHYSICAL, convergent checkerboard states (2-site simple
update), never random tensors: the fused 2-site CTM oracle oscillates on random
input, making any split-vs-fused comparison meaningless.

The trusted AD gate is implicit==explicit (not implicit==finite-difference).
Historically that choice was justified by a "Wirtinger gap" in the split
energy_fn; that explanation was wrong -- the computation is entirely real, so
Wirtinger cannot apply, and the AD-vs-FD disagreement it described was the
#750 SVD-adjoint bug (the off-diagonal adjoint term came out at -0.5x, which
flips the gradient's sign).  With #750/#751 fixed, AD and FD agree.
implicit==explicit remains the primary gate here because it is far cheaper than
FD on a CTM fixed point, not because FD is untrustworthy.  The *tight*,
machine-exact implicit==explicit gate lives in the non-degenerate anisotropic
regime (``test_2site_implicit_grad_matches_explicit_clean_regime``, XXZ Δ=0.3):
there the projector singular values are well separated, the Lorentzian SVD
backward is exact, and rel~1e-15. At the SU(2)-symmetric Heisenberg point the
degenerate-SV SVD backward imposes a ~5e-4 regularization floor on the *explicit
reference*, so that test gates direction + energy + self-consistency instead
(see its docstring).
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tests.test_split_ctm_2site import _build_su_neel, _heisenberg_gate


@pytest.fixture(scope="module")
def su_state():
    """Convergent (A, B) Heisenberg Néel checkerboard via 2-site simple update."""
    A, B = _build_su_neel(D=2)
    return A, B


def _xxz_gate(delta, d=2):
    """XXZ 2-site gate H = Δ Sz⊗Sz + ½(S+⊗S- + S-⊗S+), reshaped (d,d,d,d).

    Δ != 1 breaks the SU(2) symmetry of the Heisenberg point, lifting the
    corner/projector singular-value degeneracy that otherwise triggers the
    Lorentzian-regularized SVD backward (see the clean-regime parity test)."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def _build_su_xxz(delta, D=2, d=2, n_steps=80, dt=0.05, seed=7):
    """Convergent anisotropic (A, B) checkerboard via 2-site XXZ simple update.

    Same construction as ``_build_su_neel`` but with an XXZ (Δ != 1) gate, so the
    resulting fixed point has NON-degenerate projector singular values."""
    from tenax.algorithms.ipeps import (
        _make_trotter_gate_tensor,
        _wrap_as_dense_tensor,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_2site_horizontal_tensor,
        _simple_update_2site_vertical_tensor,
    )

    H = _xxz_gate(delta, d)
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A_data = 0.1 * jax.random.normal(kA, (D, D, D, D, d), dtype=jnp.float64)
    B_data = 0.1 * jax.random.normal(kB, (D, D, D, D, d), dtype=jnp.float64)
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


def test_converge_split_env_2site_matches_forward(su_state):
    """Forward-only multisite converge lands on the same fixed-point energy as
    ctm_split_tensor_2site (both are the Γ-gauge-fixed coupled fixed point)."""
    from tenax.algorithms._split_ctm_energy_ad import converge_split_env_2site
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8

    envs_ref = ctm_split_tensor_2site(
        A, B, chi, max_iter=100, conv_tol=1e-12, chi_I=chi
    )
    E_ref = float(
        compute_energy_split_ctm_tensor_2site(A, B, envs_ref[0], envs_ref[1], gate, d=2)
    )

    envs = converge_split_env_2site(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=chi,
        chi_I=chi,
        max_iter=100,
        conv_tol=1e-12,
        min_iter=2,
    )
    E = float(
        compute_energy_split_ctm_tensor_2site(
            A, B, envs[(0, 0)], envs[(1, 0)], gate, d=2
        )
    )
    assert abs(E - E_ref) < 1e-9, f"forward converge mismatch: {E} vs {E_ref}"


def test_explicit_multisite_converge_grad_finite(su_state):
    """Unrolled explicit multisite converge yields a finite, non-zero gradient
    w.r.t. A on the convergent state."""
    from tenax.algorithms._split_ctm_energy_ad import (
        _explicit_split_multisite_converge,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4

    def loss(a):
        envs = _explicit_split_multisite_converge(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=chi,
            chi_I=chi,
            warmup_steps=10,
            backprop_steps=10,
        )
        return compute_energy_split_ctm_tensor_2site(
            a, B, envs[(0, 0)], envs[(1, 0)], gate, d=2
        ).real

    e, g = jax.value_and_grad(loss)(A)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.isfinite(e)
    assert jnp.all(jnp.isfinite(gs)) and float(jnp.sum(jnp.abs(gs))) > 0


def test_2site_implicit_grad_matches_explicit(su_state):
    """PRIMARY Tier-3 gate: 2-site split implicit (Neumann) gradient parity.

    implicit==explicit, NOT implicit==FD -- for cost, not for trust.  (The
    "Wirtinger gap" this docstring used to cite was the #750 SVD-adjoint bug,
    now fixed; see the module docstring.)
    Gradient taken w.r.t. sublattice A only (B held fixed) for a clean scalar
    parity, at the lossless chi_I=chi fixed point.

    Two levels of parity are asserted, per the diagnostic below:

    * **Implicit self-consistency (machine-exact):** the Neumann VJP at
      conv_tol=1e-14 vs conv_tol=1e-15 agrees to rel~3e-15 / cos=1.0.  This
      proves the fixed-point adjoint is FULLY CONVERGED and seed/depth-
      independent — it does NOT by itself prove the VJP is correct (a
      converged-but-wrong VJP would still be self-consistent).  The actual
      VJP-correctness gate is the machine-exact clean-regime companion,
      ``test_2site_implicit_grad_matches_explicit_clean_regime``, which lifts
      the projector SV degeneracy (XXZ Δ=0.3) so implicit==explicit becomes
      exact.  At the Heisenberg point here, we therefore assert only direction
      (cos) + energy, with the magnitude bound (rel<1e-3) reflecting the
      documented regularized-SVD floor rather than the VJP itself.
    * **Implicit vs explicit (direction + energy):** cos>1-1e-6 and energy to
      1e-9.  A tight rel<1e-6 magnitude match is NOT reachable here and is NOT a
      VJP bug: the 2×2 split projector uses the Lorentzian-regularized
      degenerate-SV SVD backward (``truncated_svd_symmetric_ad``), so the
      *explicit* reference — which differentiates through 60 stacked regularized
      SVDs — carries a fixed rel~4.7e-4 deviation from the implicit path's single
      SVD + Neumann series.  That deviation is INVARIANT to explicit depth
      (rel is identical to 13 digits at warmup/backprop 30, 60, 90), which rules
      out under-convergence and localises it to the regularized SVD adjoint, not
      the Task-2 VJP.  Direction (cos) still guards against a genuine
      sign/structure bug in the VJP.
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit_2site,
    )

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4  # chi = D*D lossless on a physical low-interlayer-rank state

    def loss_imp(a, conv_tol=1e-14):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=120,
            conv_tol=conv_tol,
            min_iter=2,
        ).real

    def loss_exp(a):
        return ctm_energy_split_explicit_2site(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            warmup_steps=60,
            backprop_steps=60,
        ).real

    e_i, g_i = jax.value_and_grad(loss_imp)(A)
    e_e, g_e = jax.value_and_grad(loss_exp)(A)
    # Independent implicit run at a tighter fixed-point tol: self-consistency.
    _, g_i2 = jax.value_and_grad(lambda a: loss_imp(a, conv_tol=1e-15))(A)

    gi = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i)])
    ge = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_e)])
    gi2 = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i2)])

    # (1) Machine-exact self-consistency of the Neumann VJP under test.
    rel_self = float(jnp.linalg.norm(gi - gi2) / jnp.linalg.norm(gi2))
    cos_self = float(
        jnp.real(jnp.vdot(gi, gi2)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(gi2))
    )
    # (2) Implicit-vs-explicit direction + energy.
    cos = float(
        jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge))
    )
    rel = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))

    assert rel_self < 1e-10, f"implicit VJP not self-consistent: rel={rel_self}"
    assert cos_self > 1 - 1e-12, f"implicit VJP self-direction: cos={cos_self}"
    assert jnp.allclose(e_i, e_e, atol=1e-9), f"energy mismatch: {e_i} vs {e_e}"
    assert cos > 1 - 1e-6, f"gradient direction mismatch: cos={cos}"
    # rel bound reflects the regularized-SVD reference floor (see docstring);
    # this is a ceiling on the explicit reference, not the VJP under test.
    assert rel < 1e-3, f"gradient magnitude mismatch: rel={rel}"


def test_2site_implicit_grad_matches_explicit_clean_regime():
    """PRIMARY VJP-correctness gate: on a NON-degenerate (anisotropic XXZ Δ=0.3)
    convergent state, the implicit (Neumann) gradient matches the explicit
    (unrolled) gradient to MACHINE PRECISION.

    This is the tight companion to test_2site_implicit_grad_matches_explicit
    (which runs at the SU(2)-symmetric Heisenberg point, where a degenerate-SV
    SVD-backward floor caps magnitude parity at rel~5e-4). Breaking the symmetry
    with Δ != 1 lifts the projector SV degeneracy, so the Lorentzian SVD-backward
    regularization becomes negligible and implicit==explicit becomes exact —
    proving the rel~5e-4 at Heisenberg is the regularized-SVD reference floor,
    NOT a bug in the implicit VJP under test. Measured here: cos=1.0, rel~2e-15."""
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit_2site,
    )

    A, B = _build_su_xxz(0.3)
    gate = _xxz_gate(0.3)
    chi = 4

    def loss_imp(a):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=120,
            conv_tol=1e-14,
            min_iter=2,
        ).real

    def loss_exp(a):
        return ctm_energy_split_explicit_2site(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            warmup_steps=60,
            backprop_steps=60,
        ).real

    e_i, g_i = jax.value_and_grad(loss_imp)(A)
    e_e, g_e = jax.value_and_grad(loss_exp)(A)
    gi = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i)])
    ge = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_e)])

    cos = float(
        jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge))
    )
    rel = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))
    assert jnp.allclose(e_i, e_e, atol=1e-9), f"energy mismatch: {e_i} vs {e_e}"
    assert cos > 1 - 1e-9, f"gradient direction mismatch: cos={cos}"
    assert rel < 1e-6, f"gradient magnitude mismatch (non-degenerate regime): rel={rel}"


def test_2site_split_energy_matches_fused_ad_path(su_state):
    """The AD-energy value (split implicit) matches the fused 2-site energy on
    the convergent state — energy correctness independent of the gradient."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8

    envA, envB = ctm_tensor_2site(A, B, chi, max_iter=100, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))

    E_split = float(
        ctm_energy_split_implicit_2site(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=100,
            conv_tol=1e-12,
            min_iter=2,
        ).real
    )
    assert abs(E_split - E_fused) < 1e-6, f"split={E_split} fused={E_fused}"


def test_2site_implicit_grad_fd_directional(su_state):
    """AD vs finite differences on the split energy — direction AND magnitude.

    This gate used to assert direction only, on the stated grounds that the
    split energy_fn carried a "Wirtinger (real/complex-derivative) gap".  That
    was a misdiagnosis: the computation is entirely real, so Wirtinger cannot
    apply.  The real cause was the #750 SVD-adjoint bug, which put the
    off-diagonal adjoint contribution at -0.5x and flipped the gradient's sign.
    With #750/#751 fixed, AD and FD agree in magnitude too, so the magnitude
    assertion below is now a live gate rather than a documented waiver."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site
    from tenax.algorithms.ipeps import _wrap_as_dense_tensor

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 4

    def loss(a):
        return ctm_energy_split_implicit_2site(
            {(0, 0): a, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=80,
            conv_tol=1e-13,
            min_iter=2,
        ).real

    _, g = jax.value_and_grad(loss)(A)
    g_ad = jax.tree.leaves(g)[0].ravel()

    A_data = A.todense()
    eps = 1e-5
    flat = A_data.ravel()
    idxs = list(range(0, flat.size, max(1, flat.size // 12)))[:12]  # sample ~12 dirs
    g_fd = []
    for i in idxs:
        pert = flat.at[i].add(eps).reshape(A_data.shape)
        pert_m = flat.at[i].add(-eps).reshape(A_data.shape)
        ep = loss(_wrap_as_dense_tensor(pert))
        em = loss(_wrap_as_dense_tensor(pert_m))
        g_fd.append(float((ep - em) / (2 * eps)))
    g_fd = jnp.array(g_fd)
    g_ad_s = jnp.array([float(g_ad[i]) for i in idxs])
    cos = float(
        jnp.dot(g_ad_s, g_fd)
        / (jnp.linalg.norm(g_ad_s) * jnp.linalg.norm(g_fd) + 1e-30)
    )
    assert cos > 0.99, f"AD and FD gradients point in different directions: cos={cos}"
    rel = float(jnp.linalg.norm(g_ad_s - g_fd) / (jnp.linalg.norm(g_fd) + 1e-30))
    # Threshold set by the FD step (eps=1e-5) on a CTM fixed point, not by any
    # remaining adjoint defect; pre-#750 this sat above 0.5 with a sign flip.
    assert rel < 1e-3, f"AD and FD gradient magnitudes disagree: rel={rel}"


def test_validate_split_ctm_config_allows_2site():
    """The 2-site checkerboard recipe ('2x2') is allowed under fuse=False; the
    three chi-changing knobs are still rejected."""
    from tenax.algorithms.ipeps_ad_policy import validate_split_ctm_config
    from tenax.algorithms.ipeps_config import CTMConfig

    cfg = CTMConfig(chi=8, chi_I=8, fuse_virtual_legs=False)
    validate_split_ctm_config(cfg, "1x1")  # single-site still OK
    validate_split_ctm_config(cfg, "2x2")  # 2-site now OK — must not raise

    bump = CTMConfig(chi=8, chi_I=8, fuse_virtual_legs=False, chi_auto_bump=True)
    with pytest.raises(NotImplementedError):
        validate_split_ctm_config(bump, "2x2")


def test_make_ctm_energy_fn_dispatches_2site_split(su_state):
    """make_ctm_energy_fn routes a 2-coord site dict to the 2-site split path
    (fuse=False, recipe='2x2'), matching a direct implicit-2site call, with a
    finite gradient through the dispatch closure."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit_2site
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
    from tenax.algorithms.ipeps_config import CTMConfig

    A, B = su_state
    gate = _heisenberg_gate()
    chi = 8
    cfg = CTMConfig(
        chi=chi,
        chi_I=chi,
        fuse_virtual_legs=False,
        max_iter=100,
        conv_tol=1e-12,
        min_iter=2,
    )
    fn = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=3,
        explicit_steps=20,
        explicit_backward_steps=None,
        energy_fn=None,
        recipe="2x2",
    )
    E_dispatch = float(fn({(0, 0): A, (1, 0): B}).real)
    E_direct = float(
        ctm_energy_split_implicit_2site(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=100,
            conv_tol=1e-12,
            min_iter=2,
        ).real
    )
    assert abs(E_dispatch - E_direct) < 1e-10

    def loss(a):
        return fn({(0, 0): a, (1, 0): B}).real

    _, g = jax.value_and_grad(loss)(A)
    gs = jax.tree.leaves(g)[0]
    assert bool(abs(gs).sum() > 0) and bool((gs == gs).all())


def test_optimize_gs_ad_2site_split_runs(su_state):
    """optimize_gs_ad with config.ctm.fuse_virtual_legs=False + recipe='2x2' runs
    a bipartite Heisenberg optimization end-to-end (a few steps), producing a
    finite, physical (variational, above the spin-1/2 AFH floor) energy."""
    from tenax import optimize_gs_ad
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    A, B = su_state
    gate = _heisenberg_gate()
    # chi=4=D^2 is lossless on this physical (low-interlayer-rank) state; a
    # capped CTM max_iter + 2 optimizer steps keep this end-to-end smoke
    # tractable while still exercising the full split path (warm-start +
    # line-search probe + implicit-AD gradient, all on split envs/energy).
    ctm = CTMConfig(chi=4, chi_I=4, fuse_virtual_legs=False, max_iter=40)
    cfg = iPEPSConfig(
        ctm=ctm,
        unit_cell="2site",
        gs_num_steps=2,
        gs_implicit_ad=True,
        gs_c4v=False,
        gs_recipe="2x2",
        gs_optimizer="lbfgs",
        su_init=False,
    )
    (A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, (A, B), cfg)
    E = float(E_gs)
    assert E == E  # finite (not NaN)
    assert E > -1.0, f"energy below spin-1/2 AFH floor (non-variational): {E}"
