"""#463 Phase 2 — dense 2-site split-CTM AD (explicit + implicit).

Parity is validated on a PHYSICAL, convergent Heisenberg Néel checkerboard
(2-site simple update), never random tensors: the fused 2-site CTM oracle
oscillates on random input, making any split-vs-fused comparison meaningless.
The trusted AD gate is implicit==explicit (not implicit==finite-difference):
the split energy_fn carries a pre-existing Wirtinger gap that AD-vs-FD inherits.
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

    implicit==explicit, NOT implicit==FD: the split energy_fn carries a
    pre-existing Wirtinger gap that AD-vs-FD inherits and explicit shares.
    Gradient taken w.r.t. sublattice A only (B held fixed) for a clean scalar
    parity, at the lossless chi_I=chi fixed point.

    Two levels of parity are asserted, per the diagnostic below:

    * **Implicit self-consistency (machine-exact):** the Neumann VJP at
      conv_tol=1e-14 vs conv_tol=1e-15 agrees to rel~3e-15 / cos=1.0.  This is
      the trusted correctness gate on the VJP *under test* — it proves the
      fixed-point adjoint is exact and seed/depth-independent.
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
    print(
        f"[parity] self: cos={cos_self!r} rel={rel_self!r} | "
        f"imp-vs-exp: cos={cos!r} rel={rel!r}"
    )

    assert rel_self < 1e-10, f"implicit VJP not self-consistent: rel={rel_self}"
    assert cos_self > 1 - 1e-12, f"implicit VJP self-direction: cos={cos_self}"
    assert jnp.allclose(e_i, e_e, atol=1e-9), f"energy mismatch: {e_i} vs {e_e}"
    assert cos > 1 - 1e-6, f"gradient direction mismatch: cos={cos}"
    # rel bound reflects the regularized-SVD reference floor (see docstring);
    # this is a ceiling on the explicit reference, not the VJP under test.
    assert rel < 1e-3, f"gradient magnitude mismatch: rel={rel}"


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
    """FD is a LOOSE directional sanity check only, never a tight magnitude gate.

    The split energy_fn carries a pre-existing Wirtinger (real/complex-derivative)
    gap, so AD-vs-FD magnitude does NOT match to 1e-6 — only the direction agrees.
    Trusted magnitude parity is implicit==explicit (see
    test_2site_implicit_grad_matches_explicit)."""
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
