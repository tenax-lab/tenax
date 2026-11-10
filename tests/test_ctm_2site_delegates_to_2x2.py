"""``ctm_2site()`` and ``ipeps()`` run the 2x2 recipe of ``ctm_tensor_2site``
since 0.9.3.

The legacy sweep built each sublattice's new boundary from that sublattice's
own corners and edges, with a projector of its own, so the two environments did
not share a basis at the A|B seams and the energy moved with the bond gauge of
the pair; ``tests/test_ipeps_2site_energy.py`` pins that.  This file pins the
delegation.  ``_ctm_2site_via_2x2`` is ``ctm_2site()`` without its warnings: it
maps the ``CTMConfig`` fields onto the multisite loop of ``ctm_tensor_2site`` and
returns two raw-array ``CTMEnvironment`` named tuples whose legs are in the order
the raw 2-site RDMs contract.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _window_oracle import ising_peps

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_config import CTMConfig, CTMEnvironment
from tenax.algorithms.ipeps_ctm_convergence import (
    CTMConvergenceInfo,
    _ctm_2site_via_2x2,
)
from tenax.algorithms.ipeps_rdm import _rdm1x2_2site, _rdm2x1_2site
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

# A 1x1 product state: its chi=1 corners are stationary from the second sweep,
# so the stopping rule is all that moves.
ONES = jnp.ones((1, 1, 1, 1, 2))

# Every entry point that must return the environments of the 2x2 recipe.
ENTRY_POINTS = [
    pytest.param(_ctm_2site_via_2x2, id="helper"),
]


def _random_site(seed):
    A = jax.random.normal(jax.random.PRNGKey(seed), (2, 2, 2, 2, 2))
    return A / jnp.linalg.norm(A)


def _symmetric_site():
    sym = U1Symmetry()
    zeros = np.zeros(2, dtype=np.int32)
    legs = [
        ("u", FlowDirection.OUT),
        ("d", FlowDirection.IN),
        ("l", FlowDirection.OUT),
        ("r", FlowDirection.IN),
        ("phys", FlowDirection.IN),
    ]
    indices = tuple(
        TensorIndex.from_charges(sym, zeros.copy(), flow, label=label)
        for label, flow in legs
    )
    return SymmetricTensor.random_normal(indices=indices, key=jax.random.PRNGKey(0))


def _asymmetric_site(seed, eps=0.3):
    """A D=2 product state plus ``eps`` x Gaussian noise: no lattice symmetry."""
    A = np.zeros((2, 2, 2, 2, 2))
    A[0, 0, 0, 0, 0] = 1.0
    A += eps * np.random.default_rng(seed).standard_normal(A.shape)
    return jnp.asarray(A)


def _asymmetric_hamiltonian(seed=7):
    """A random real-symmetric two-site operator.  It has no site-exchange
    symmetry: ``max|H - SWAP H SWAP|`` is 1.48 (measured)."""
    M = np.random.default_rng(seed).standard_normal((4, 4))
    return jnp.asarray((M + M.T) / 2)


def _raw_energy(A, B, env_A, env_B, H):
    """Half the sum of the four bond energies, from the raw 2-site RDMs.

    Written out here, not read from ``compute_energy_ctm_2site``, so that the
    orientation test does not depend on that function's bond count."""
    H = H.reshape(2, 2, 2, 2)
    rdms = (
        _rdm2x1_2site(A, B, env_A, env_B, 2),
        _rdm2x1_2site(B, A, env_B, env_A, 2),
        _rdm1x2_2site(A, B, env_A, env_B, 2),
        _rdm1x2_2site(B, A, env_B, env_A, 2),
    )
    return 0.5 * sum(float(jnp.einsum("ijkl,klij->", r, H).real) for r in rdms)


def _assert_same_envs(envs, ref):
    for env, ref_env in zip(envs, ref, strict=True):
        for name in CTMEnvironment._fields:
            np.testing.assert_array_equal(getattr(env, name), getattr(ref_env, name))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry(entry):
    """The raw environments are the 2x2 environments, in the order the raw
    2-site RDMs contract.

    The multisite sweep stores every corner with its two legs reversed (``C1``
    as ``(c1_r, c1_d)``), and the raw RDMs contract by position.  On a
    lattice-symmetric state each corner is a symmetric matrix, so a transposed
    corner changes nothing.  This pair has no lattice symmetry and ``H`` no
    site-exchange symmetry, so the raw energy must equal the energy that
    ``compute_energy_ctm_tensor_2site`` takes, by label, from the environments of
    ``ctm_tensor_2site``.  Measured on this pair at d6730510: the criterion
    reaches 8.1e-13; the weakest corner asymmetry is 0.057; with the corners as
    the sweep stores them the raw energy is 2.3e-2 off, with ``env_A`` and
    ``env_B`` swapped 0.22 off, and each corner alone moves it by at least 4.5e-4.
    """
    A, B, H = _asymmetric_site(5), _asymmetric_site(6), _asymmetric_hamiltonian()
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env_A, env_B, info = entry(A, B, cfg, return_meta=True)
    assert info.converged

    # The regime, so the test cannot go vacuous: a corner that is a symmetric
    # matrix is blind to a transposition.
    asymmetry = [
        float(jnp.max(jnp.abs(C - C.T)) / jnp.max(jnp.abs(C)))
        for env in (env_A, env_B)
        for C in (env.C1, env.C2, env.C3, env.C4)
    ]
    assert min(asymmetry) > 5e-3, asymmetry

    At, Bt = _wrap_as_dense_tensor(A), _wrap_as_dense_tensor(B)
    tensor_A, tensor_B = ctm_tensor_2site(
        At, Bt, cfg.chi, max_iter=cfg.max_iter, conv_tol=cfg.conv_tol
    )
    E_raw = _raw_energy(A, B, env_A, env_B, H)
    E_tensor = float(compute_energy_ctm_tensor_2site(At, Bt, tensor_A, tensor_B, H, 2))
    assert abs(E_raw - E_tensor) < 1e-9, (E_raw, E_tensor)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad", ["A", "B"])
@pytest.mark.parametrize(
    "shape",
    [(4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2, 2, 2), (2, 2, 3, 3, 2)],
    ids=["3-leg", "4-leg", "6-leg", "unequal-virtual"],
)
def test_rejects_a_bad_raw_shape(entry, bad, shape):
    """Only 5 legs ``(u, d, l, r, s)`` with one virtual dimension is a site
    tensor, on either sublattice: ``_wrap_as_dense_tensor`` reads ``shape[0]``
    for all four virtual legs and ``shape[4]`` for the physical one, unchecked."""
    good = _random_site(0)
    pair = (jnp.ones(shape), good) if bad == "A" else (good, jnp.ones(shape))
    with pytest.raises(ValueError, match=r"5 legs \(u, d, l, r, s\)"):
        entry(*pair, CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad", ["A", "B"])
def test_refuses_a_symmetric_tensor(entry, bad):
    """The raw-array result would need ``todense()`` on a block-sparse tensor;
    the message names ``ctm_tensor_2site``, which keeps the blocks."""
    good = _random_site(0)
    pair = (_symmetric_site(), good) if bad == "A" else (good, _symmetric_site())
    with pytest.raises(TypeError, match="ctm_tensor_2site"):
        entry(*pair, CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_numpy_and_dense_tensor_inputs_match_the_jax_array(entry):
    """NumPy arrays are converted and ``DenseTensor``s (what ``ipeps()``
    returns) pass through, each sublattice on its own: every mix gives the
    environments of the JAX arrays, bit for bit."""
    A, B = _random_site(0), _random_site(7)
    cfg = CTMConfig(chi=4, max_iter=5)
    ref = entry(A, B, cfg)
    for pair in (
        (np.asarray(A), np.asarray(B)),
        (_wrap_as_dense_tensor(A), _wrap_as_dense_tensor(B)),
        (A, _wrap_as_dense_tensor(B)),
        (np.asarray(A), B),
    ):
        _assert_same_envs(entry(*pair, cfg), ref)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_returns_raw_environments_and_python_scalars(entry):
    """Two raw environments with ``jax.Array`` fields of shape ``(chi, chi)`` and
    ``(chi, D**2, chi)``, as ``ctm_2site()`` always returned.  The info holds
    Python scalars, because the multisite loop is a Python loop."""
    A = jnp.asarray(ising_peps(0.2))
    env_A, env_B, info = entry(
        A, A, CTMConfig(chi=8, max_iter=500, conv_tol=1e-12), return_meta=True
    )
    for env in (env_A, env_B):
        assert isinstance(env, CTMEnvironment)
        assert isinstance(env.C1, jax.Array) and env.C1.shape == (8, 8)
        assert isinstance(env.T1, jax.Array) and env.T1.shape == (8, 4, 8)
    assert isinstance(info, CTMConvergenceInfo)
    assert type(info.converged) is bool
    assert type(info.n_iter) is int
    assert type(info.diff) is float


def test_helper_runs_no_qr_warmup():
    """``"svd"`` is passed, so a ``"qr"`` config starts no warm-up loop: the
    stationary state stops at sweep 2, not after six warm-up sweeps (8 sweeps,
    measured with ``"qr"`` passed on)."""
    _env_A, _env_B, info = _ctm_2site_via_2x2(
        ONES,
        ONES,
        CTMConfig(
            chi=1, max_iter=40, min_iter=1, projector_method="qr", qr_warmup_steps=6
        ),
        return_meta=True,
    )
    assert info.converged
    assert info.n_iter == 2


@pytest.mark.parametrize("renormalize", [True, False])
def test_helper_passes_renormalize(renormalize):
    """``renormalize`` reaches the loop: with it the largest entry of
    ``env_A.C1`` is 1; without it, 0.947 on this pair (measured)."""
    env_A, _env_B = _ctm_2site_via_2x2(
        _random_site(42),
        _random_site(43),
        CTMConfig(chi=4, max_iter=5, renormalize=renormalize),
    )
    peak = float(jnp.max(jnp.abs(env_A.C1)))
    if renormalize:
        assert peak == pytest.approx(1.0, abs=1e-12)
    else:
        assert peak < 0.99


def test_helper_emits_no_deprecation_warning():
    """``ipeps()`` and the fixture-only tests call the helper, so they do not
    trigger ``ctm_2site()``'s ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ctm_2site_via_2x2(ONES, ONES, CTMConfig(chi=1, max_iter=5))
