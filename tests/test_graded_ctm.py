"""The graded Tensor CTM against an exact finite patch (#1035, design §5 step 4).

An untruncated CTM from the vacuum boundary is an exact contraction of a
finite open patch (``tests/_graded_ctm.py``), so its RDMs must equal Phase 2's
``double_layer_value`` on that patch element by element.  The Fock oracle
itself cannot be used here: it stores ``2**(sites + 2*bonds)`` amplitudes,
``2**33`` already for 3x3.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from _graded_cluster import double_layer_value, production_site
from _graded_ctm import centre_bond, patch, untruncated_env, vacuum_boundary_env

from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor,
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor,
    _rdm2x1_tensor_2site,
    _rdm_1site_tensor,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor


def _random_even_site(seed: int) -> np.ndarray:
    A = np.random.default_rng(seed).standard_normal((2, 2, 2, 2, 2))
    for k in itertools.product(*[range(n) for n in A.shape]):
        if sum(k) % 2:
            A[k] = 0.0
    return A


@pytest.fixture(scope="module")
def site_and_env():
    A_np = _random_even_site(11)
    A = production_site(A_np)
    env, _chi = untruncated_env(A, 1)
    return A_np, A, env


def _reference_rdm2(R, C, A_np, s, t, As=None) -> np.ndarray:
    """``ref[p_s, p_t, P_s, P_t] = <|P_s P_t><p_s p_t|>`` on the patch --
    the CTM RDM's ``(phys, phys_2, phys_bra, phys_bra_2)`` order.  ``As``
    overrides the uniform patch (e.g. a checkerboard)."""
    As = patch(R, C, A_np) if As is None else As
    norm = double_layer_value(R, C, As)
    ref = np.zeros((2, 2, 2, 2), dtype=complex)
    for k in itertools.product((0, 1), repeat=4):
        if sum(k) % 2:
            continue
        h2 = np.zeros((2, 2, 2, 2))
        h2[k[2], k[3], k[0], k[1]] = 1.0  # h2[P_s, P_t, p_s, p_t]
        ref[k] = double_layer_value(R, C, As, op=((s, t), h2)) / norm
    return ref


def _ctm_rdm(fn, A, env) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the non-PSD warning
        return np.asarray(fn(A, env))


@pytest.mark.parametrize(
    "fn,R,C,horizontal",
    [(_rdm2x1_tensor, 3, 4, True), (_rdm1x2_tensor, 4, 3, False)],
    ids=["2x1", "1x2"],
)
def test_two_site_rdm_equals_the_exact_patch(site_and_env, fn, R, C, horizontal):
    A_np, A, env = site_and_env
    s, t = centre_bond(R, C, horizontal)
    ref = _reference_rdm2(R, C, A_np, s, t)
    # Regime: the elements a wrong sign would corrupt are not trivially zero --
    # both occupied (P_s P_t term of the sign), hopping, and pairing (which
    # symmetrisation cancels when the sign is wrong).
    for k in [(1, 1, 1, 1), (0, 1, 1, 0), (0, 0, 1, 1)]:
        assert abs(ref[k]) > 5e-3, k
    got = _ctm_rdm(fn, A, env)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_one_site_rdm_equals_the_exact_patch(site_and_env):
    A_np, A, env = site_and_env
    R, C = 3, 3  # one sweep around a 1x1 centre
    rho2 = _reference_rdm2(R, C, A_np, (1, 1), (1, 2))
    ref = np.einsum("abcb->ac", rho2)  # trace out (1, 2): [p, P]
    assert ref[1, 1].real > 5e-3  # regime: the odd sector is populated
    got = _ctm_rdm(_rdm_1site_tensor, A, env)
    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert np.linalg.eigvalsh(got).min() > -1e-12


@pytest.mark.parametrize("sweep", ["single", "paired", "multisite"])
def test_the_sign_free_1x1_recipe_refuses_fermions(site_and_env, sweep):
    """The 1x1 recipe and the paired moves fuse chi and D² legs sign-free;
    fermionic input must not reach them silently."""
    from tenax.algorithms._ctm_tensor_convergence import (
        SINGLE_SITE_NEIGHBORS,
        _ctm_tensor_sweep,
        _ctm_tensor_sweep_multisite,
        _ctm_tensor_sweep_paired,
    )
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    _, A, env = site_and_env
    a = _build_double_layer_tensor(A)
    with pytest.raises(NotImplementedError, match="recipe='2x2'"):
        if sweep == "single":
            _ctm_tensor_sweep(env, a, 4, True)
        elif sweep == "paired":
            _ctm_tensor_sweep_paired(env, a, 4, True)
        else:
            _ctm_tensor_sweep_multisite(
                {(0, 0): env},
                {(0, 0): a},
                SINGLE_SITE_NEIGHBORS,
                4,
                True,
                "svd",
                recipe="1x1",
            )


@pytest.fixture(scope="module")
def checkerboard_and_envs():
    """A 2-site A/B checkerboard (what ``fpeps()`` runs), one untruncated
    sweep from the vacuum boundary."""
    A_np, B_np = _random_even_site(11), _random_even_site(12)
    A, B = production_site(A_np), production_site(B_np)
    chi = 4
    envs = {(0, 0): vacuum_boundary_env(A, chi), (1, 0): vacuum_boundary_env(B, chi)}
    dls = {(0, 0): _build_double_layer_tensor(A), (1, 0): _build_double_layer_tensor(B)}
    envs, _, _ = _ctm_tensor_sweep_multisite(
        envs, dls, CHECKERBOARD_NEIGHBORS, chi, True
    )
    return A_np, B_np, A, B, envs


@pytest.mark.parametrize(
    "fn,R,C,horizontal",
    [(_rdm2x1_tensor_2site, 3, 4, True), (_rdm1x2_tensor_2site, 4, 3, False)],
    ids=["2x1", "1x2"],
)
def test_checkerboard_rdm_equals_the_exact_patch(
    checkerboard_and_envs, fn, R, C, horizontal
):
    A_np, B_np, A, B, envs = checkerboard_and_envs
    s, t = centre_bond(R, C, horizontal)
    pa, pb = patch(R, C, A_np), patch(R, C, B_np)
    As = {c: pa[c] if (c[0] - s[0] + c[1] - s[1]) % 2 == 0 else pb[c] for c in pa}
    ref = _reference_rdm2(R, C, None, s, t, As=As)
    assert abs(ref[0, 0, 1, 1]) > 5e-3  # regime: pairing is populated
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = np.asarray(fn(A, B, envs[(0, 0)], envs[(1, 0)]))
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_the_fermionic_simple_update_keeps_the_site_flow_convention():
    """The shared SU sweep returns every leg reversed (the gate's output legs
    and the SVD's new bond); ``_fpeps_simple_update`` restores the input's
    flows, which the graded CTM is certified on -- by a plain dual, so the
    numbers are exactly the sweep's."""
    import jax

    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _fpeps_simple_update,
        _initialize_fpeps,
        _trotter_gate,
        spinless_fermion_gate,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_checkerboard_sweep,
    )

    cfg = FPEPSConfig(D=2, t=1.0, V=1.0)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    H = spinless_fermion_gate(cfg)
    flows = lambda t: [(i.label, i.flow) for i in t.indices]  # noqa: E731

    A_raw, B_raw, _ = _simple_update_checkerboard_sweep(
        A, A, _trotter_gate(H, 0.05), 2, 4
    )
    assert flows(A_raw) != flows(A)  # regime: the sweep does reverse them

    A_new, B_new, _ = _fpeps_simple_update(A, H, max_D=2, dt=0.05, steps=1)
    assert flows(A_new) == flows(A) and flows(B_new) == flows(A)
    np.testing.assert_array_equal(
        np.asarray(A_new.todense()), np.asarray(A_raw.todense())
    )
    np.testing.assert_array_equal(
        np.asarray(B_new.todense()), np.asarray(B_raw.todense())
    )
