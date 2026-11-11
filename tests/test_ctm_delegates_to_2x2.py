"""``ctm()`` runs ``ctm_tensor(recipe="2x2")`` since 0.9.3.

The 1-site sweep that ``ctm()`` ran before often did not converge.  The oracle
here is the Ising PEPS: its norm is the classical 2D Ising partition function,
so C(1,0) has Onsager's exact value.  At beta=0.2 the old sweep stopped at
chi=12, 16 and 32 without converging, with C(1,0) errors of 7.6e-4, 4.3e-4 and
4.3e-4; at beta=0.3 it converged at no chi.  The 2x2 recipe converges at every
chi below in 14 to 57 sweeps, with errors of at most 1.6e-14 (measured
2026-11-09).

``_ctm_via_2x2`` is the delegation without ``ctm()``'s warnings.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import apply_op
from _window_oracle import closed_double_layer, correlators, ising_peps, onsager_c10

from tenax.algorithms._ctm_env_dense import _as_dense_env, _in_documented_leg_order
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv, initialize_ctm_tensor_env
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.fermionic_ipeps import FPEPSConfig, fermionic_ctm
from tenax.algorithms.ipeps_config import CTMConfig, CTMEnvironment
from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo, _ctm_via_2x2, ctm
from tenax.algorithms.ipeps_excitations import ExcitationConfig, compute_excitations
from tenax.algorithms.ipeps_rdm import compute_energy_ctm
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# A 1x1 product state: its chi=1 corners are stationary from the second sweep,
# so the stopping rule is all that moves.
ONES = jnp.ones((1, 1, 1, 1, 2))

ISING_CASES = [
    pytest.param(beta, chi, id=f"beta{beta}-chi{chi}")
    for beta in (0.2, 0.3)
    for chi in (8, 12, 16, 32)
]

# Every entry point that must return the environment of the 2x2 recipe.
ENTRY_POINTS = [
    pytest.param(_ctm_via_2x2, id="helper"),
    pytest.param(ctm, id="ctm"),
]


def _c10_error(env, beta):
    """|C(1,0) - Onsager| from the window oracle, which shares no code with CTM."""
    A = ising_peps(beta)
    B = apply_op(A, np.diag([1.0, -1.0]))
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        1,
    )
    return abs(C[(1, 0)].real - onsager_c10(beta))


def _random_site(seed=42):
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


def _asymmetric_site(eps=0.3, seed=5):
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


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("beta,chi", ISING_CASES)
def test_ising_c10_matches_onsager(entry, beta, chi):
    """Converged, and C(1,0) within 1e-12 of Onsager, at every chi."""
    env, info = entry(
        jnp.asarray(ising_peps(beta)),
        CTMConfig(chi=chi, max_iter=500, conv_tol=1e-12),
        return_meta=True,
    )
    assert info.converged
    # ``conv_tol`` reached ``ctm_tensor``: its default, 1e-8, stops earlier.
    assert info.diff < 1e-12
    assert _c10_error(env, beta) < 1e-12


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_corner_orientation_without_lattice_symmetry(entry):
    """The raw environment has its corners in the order the raw consumers contract.

    The 2x2 sweep of ``ctm_tensor`` stores each corner with its two legs in the
    reverse of the order ``CTMTensorEnv`` documents, and the raw consumers
    (``compute_energy_ctm``) contract by position.  On a lattice-symmetric state
    every corner is a symmetric matrix, so a transposed corner changes nothing and
    Onsager's C(1,0) cannot see it.  This state has no lattice symmetry and ``H``
    no site-exchange symmetry, so the energy of the raw environment must equal the
    energy that ``compute_energy_ctm_tensor`` takes, by label, from the environment
    of ``ctm_tensor`` itself.  Measured on this state: 22 sweeps; the weakest
    corner's asymmetry is 6.5e-2; with the corners left as the sweep stores them
    the raw energy is 5.3e-2 off, and each corner alone moves it by at least 3.7e-3.
    """
    A, H = _asymmetric_site(), _asymmetric_hamiltonian()
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env, info = entry(A, cfg, return_meta=True)
    assert info.converged

    # The regime, so the test cannot go vacuous: a corner that is a symmetric
    # matrix is blind to a transposition.  ``max|C - C.T| / max|C|`` is the same
    # in either orientation.
    asymmetry = [
        float(jnp.max(jnp.abs(C - C.T)) / jnp.max(jnp.abs(C)))
        for C in (env.C1, env.C2, env.C3, env.C4)
    ]
    assert min(asymmetry) > 5e-3, asymmetry

    A_tensor = _wrap_as_dense_tensor(A)
    tensor_env, _eps = ctm_tensor(
        A_tensor,
        cfg.chi,
        max_iter=cfg.max_iter,
        conv_tol=cfg.conv_tol,
        min_iter=cfg.min_iter,
    )
    E_raw = float(compute_energy_ctm(A, env, H, 2))
    E_tensor = float(compute_energy_ctm_tensor(A_tensor, tensor_env, H, 2))
    assert abs(E_raw - E_tensor) < 1e-9, (E_raw, E_tensor)


# The leg order that ``CTMTensorEnv`` documents, written out so that the checks
# below do not read the tables that the implementation reads.
DOCUMENTED_LEGS = {
    "C1": ("c1_d", "c1_r"),
    "C2": ("c2_l", "c2_d"),
    "C3": ("c3_u", "c3_l"),
    "C4": ("c4_r", "c4_u"),
    "T1": ("t1_l", "u2", "t1_r"),
    "T2": ("t2_u", "r2", "t2_d"),
    "T3": ("t3_r", "d2", "t3_l"),
    "T4": ("t4_d", "l2", "t4_u"),
}

# An axis order to store each field in: the sweep's reversed corners; the edges
# reversed, rotated, with two legs swapped, and one left as it is.
STORED_ORDER = {
    "C1": (1, 0),
    "C2": (1, 0),
    "C3": (1, 0),
    "C4": (1, 0),
    "T1": (2, 1, 0),
    "T2": (1, 2, 0),
    "T3": (0, 2, 1),
    "T4": (0, 1, 2),
}


def _documented_env():
    """A ``CTMTensorEnv`` in the documented order with random entries, so that no
    corner is a symmetric matrix and a transposition shows."""
    init = initialize_ctm_tensor_env(_wrap_as_dense_tensor(_random_site()), 4)
    keys = jax.random.split(jax.random.PRNGKey(3), len(init))
    return CTMTensorEnv(
        *(
            DenseTensor(jax.random.normal(key, f.todense().shape), f.indices)
            for key, f in zip(keys, init)
        )
    )


def _stored(env):
    """``env`` with each field stored in the axis order of ``STORED_ORDER``."""
    return CTMTensorEnv(
        *(getattr(env, name).permute_legs(axes) for name, axes in STORED_ORDER.items())
    )


def _raw_in_documented_order(env):
    """The raw environment of a ``CTMTensorEnv``, each field transposed by label
    into the order of ``DOCUMENTED_LEGS``."""
    return CTMEnvironment(
        *(
            jnp.transpose(
                field.todense(),
                [field.labels().index(label) for label in DOCUMENTED_LEGS[name]],
            )
            for name, field in env._asdict().items()
        )
    )


def test_in_documented_leg_order_permutes_by_label():
    """Whatever order a field is stored in, it comes back in the documented order
    with its entries in the right place; a field already in order comes back as
    the same object."""
    env = _documented_env()
    assert {name: t.labels() for name, t in env._asdict().items()} == DOCUMENTED_LEGS
    fixed = _in_documented_leg_order(_stored(env))
    for name, legs in DOCUMENTED_LEGS.items():
        assert getattr(fixed, name).labels() == legs
        np.testing.assert_array_equal(
            getattr(fixed, name).todense(), getattr(env, name).todense()
        )
    for kept, field in zip(_in_documented_leg_order(env), env):
        assert kept is field


def test_in_documented_leg_order_refuses_wrong_labels():
    """A field that lacks an expected label is an error, not a guess: the labels
    are all that says which leg is which."""
    env = _documented_env()
    wrong_label = env._replace(C2=env.C2.relabel("c2_d", "c2_x"))
    wrong_count = env._replace(C1=env.T1)
    for bad, name in ((wrong_label, "C2"), (wrong_count, "C1")):
        with pytest.raises(ValueError, match=f"CTMTensorEnv.{name}"):
            _in_documented_leg_order(bad)


def test_as_dense_env_reads_a_tensor_env_by_label():
    """``_as_dense_env`` drops the labels, so it puts each field of a
    ``CTMTensorEnv`` in the documented order first, whatever order the field is
    stored in."""
    env = _documented_env()
    raw = _as_dense_env(_stored(env))
    assert isinstance(raw, CTMEnvironment)
    for name in DOCUMENTED_LEGS:
        np.testing.assert_array_equal(getattr(raw, name), getattr(env, name).todense())


def test_compute_excitations_reads_a_tensor_env_by_label():
    """A ``CTMTensorEnv`` gives ``compute_excitations`` the spectrum of the raw
    environment in the documented order.

    ``ctm_tensor`` and ``optimize_gs_ad`` return the environment of the 2x2 sweep,
    which stores each corner with its two legs reversed, and the excitation
    contractions go by position.  The state has no lattice symmetry, and ``H`` no
    site-exchange symmetry, as in ``test_corner_orientation_without_lattice_symmetry``.
    Measured on this state: with the corners read by position, the spectrum is
    up to 0.29 off.
    """
    A, H = _asymmetric_site(), _asymmetric_hamiltonian()
    A_tensor = _wrap_as_dense_tensor(A)
    env, _eps, info = ctm_tensor(
        A_tensor, 8, max_iter=500, conv_tol=1e-12, return_meta=True
    )
    assert info.converged
    raw = _raw_in_documented_order(env)
    # The regime, so the test cannot go vacuous: a corner that is a symmetric
    # matrix is blind to a transposition.
    asymmetry = [
        float(jnp.max(jnp.abs(C - C.T)) / jnp.max(jnp.abs(C)))
        for C in (raw.C1, raw.C2, raw.C3, raw.C4)
    ]
    assert min(asymmetry) > 5e-3, asymmetry

    E_gs = float(compute_energy_ctm_tensor(A_tensor, env, H, 2))
    momenta = [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi / 3)]
    config = ExcitationConfig(num_excitations=3, null_space_tol=1e-2)
    expected = compute_excitations(A, raw, H, E_gs, momenta, config).energies
    got = compute_excitations(A_tensor, env, H, E_gs, momenta, config).energies
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "shape",
    [(4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2, 2, 2), (2, 2, 3, 3, 2)],
    ids=["3-leg", "4-leg", "6-leg", "unequal-virtual"],
)
def test_rejects_a_bad_raw_shape(entry, shape):
    """Only 5 legs ``(u, d, l, r, s)`` with one virtual dimension is a site
    tensor.  ``_wrap_as_dense_tensor`` reads ``shape[0]`` for all four virtual
    legs and ``shape[4]`` for the physical one, unchecked.  The old sweep ran
    the 3-leg array and failed on the others with einsum or reshape errors."""
    with pytest.raises(ValueError, match=r"5 legs \(u, d, l, r, s\)"):
        entry(jnp.ones(shape), CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_refuses_a_symmetric_tensor(entry):
    """The raw-array result would need ``todense()`` on a block-sparse tensor;
    the message names ``ctm_tensor``, which keeps the blocks."""
    with pytest.raises(TypeError, match="ctm_tensor"):
        entry(_symmetric_site(), CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_numpy_and_dense_tensor_inputs_match_the_jax_array(entry):
    """A NumPy array is converted and a ``DenseTensor`` (what ``ipeps()``
    returns) passes through: both give the environment of the JAX array, bit
    for bit."""
    A = _random_site()
    cfg = CTMConfig(chi=4, max_iter=5)
    ref = entry(A, cfg)
    for other in (np.asarray(A), _wrap_as_dense_tensor(A)):
        env = entry(other, cfg)
        for name in CTMEnvironment._fields:
            np.testing.assert_array_equal(getattr(env, name), getattr(ref, name))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_returns_a_raw_environment_and_python_scalars(entry):
    """Raw ``jax.Array`` fields of shape ``(chi, chi)`` and ``(chi, D**2,
    chi)``, as ``ctm()`` always returned.  The info holds Python scalars,
    because ``ctm_tensor`` is a Python loop."""
    env, info = entry(
        jnp.asarray(ising_peps(0.2)),
        CTMConfig(chi=8, max_iter=500, conv_tol=1e-12),
        return_meta=True,
    )
    assert isinstance(env, CTMEnvironment)
    assert isinstance(env.C1, jax.Array) and env.C1.shape == (8, 8)
    assert isinstance(env.T1, jax.Array) and env.T1.shape == (8, 4, 8)
    assert isinstance(info, CTMConvergenceInfo)
    assert type(info.converged) is bool
    assert type(info.n_iter) is int


def test_helper_runs_no_qr_warmup():
    """``"svd"`` is passed, so a ``"qr"`` config starts no warm-up loop: the
    stationary state stops at sweep 2, not after six warm-up sweeps."""
    _env, info = _ctm_via_2x2(
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
    """``renormalize`` reaches ``ctm_tensor``: with it the corner's largest
    entry is 1; without it, 0.748 on this state (measured)."""
    env = _ctm_via_2x2(
        _random_site(), CTMConfig(chi=4, max_iter=5, renormalize=renormalize)
    )
    peak = float(jnp.max(jnp.abs(env.C1)))
    if renormalize:
        assert peak == pytest.approx(1.0, abs=1e-12)
    else:
        assert peak < 0.9


def test_helper_emits_no_deprecation_warning():
    """``fermionic_ctm`` and the fixture-only tests call the helper so that
    they do not trigger ``ctm()``'s ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ctm_via_2x2(ONES, CTMConfig(chi=1, max_iter=5))


@pytest.mark.parametrize("chi", [12, 16])
def test_fermionic_ctm_raw_branch_reproduces_onsager(chi):
    """The raw branch returns the 2x2 environment as a ``CTMEnvironment``,
    without ``ctm()``'s ``DeprecationWarning``.  It ran the old 1-site sweep
    before, which missed C(1,0) by 7.6e-4 at chi=12 and 4.3e-4 at chi=16."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        env = fermionic_ctm(
            jnp.asarray(ising_peps(0.2)),
            FPEPSConfig(ctm_chi=chi, ctm_max_iter=500, ctm_conv_tol=1e-12),
        )
    assert isinstance(env, CTMEnvironment)
    assert _c10_error(env, 0.2) < 1e-12


def test_fermionic_ctm_raw_branch_keeps_ctmconfig_min_iter(monkeypatch):
    """The raw branch builds a ``CTMConfig``, so it keeps ``min_iter=10``; a
    direct ``ctm_tensor`` call would use ``min_iter=1``."""
    import tenax.algorithms.ipeps_ctm_convergence as conv

    seen = []

    def spy(A, config, *, return_meta=False):
        seen.append(config)
        return "env"

    monkeypatch.setattr(conv, "_ctm_via_2x2", spy)
    out = fermionic_ctm(
        ONES, FPEPSConfig(ctm_chi=1, ctm_max_iter=40, ctm_conv_tol=1e-8)
    )
    (cfg,) = seen
    assert out == "env"
    assert (cfg.chi, cfg.max_iter, cfg.conv_tol, cfg.min_iter) == (1, 40, 1e-8, 10)


def test_ctm_warns_that_it_is_deprecated():
    """Exactly one deprecation warning, naming the replacement and the lost
    ``jax.jit``."""
    with pytest.warns(DeprecationWarning, match="^ctm is deprecated") as record:
        ctm(ONES, CTMConfig(chi=1, max_iter=5))
    deprecations = [w for w in record if str(w.message).startswith("ctm is deprecated")]
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert "ctm_tensor" in message
    assert "jax.jit" in message
    # Python's default filters show a DeprecationWarning only when it is blamed
    # on __main__ (PEP 565).  With stacklevel=1 it would be blamed on ctm()'s own
    # module and hidden from every script, and the checks above would still pass.
    assert deprecations[0].filename == __file__


def test_ctm_rejects_initial_env():
    """``ctm_tensor`` has no warm start, and a silent cold start would look the
    same to a caller who asked for one."""
    env = _ctm_via_2x2(ONES, CTMConfig(chi=1, max_iter=5))
    with pytest.raises(ValueError, match="initial_env"):
        ctm(ONES, CTMConfig(chi=1, max_iter=5), initial_env=env)


@pytest.mark.parametrize("method", ["qr", "eigh"])
def test_ctm_warns_that_projector_method_is_ignored(method):
    """The 2x2 recipe always uses Fishman SVD: the environment is the one
    ``"svd"`` gives, and the caller is told."""
    A = _random_site()
    with pytest.warns(UserWarning, match="projector_method"):
        env = ctm(A, CTMConfig(chi=4, max_iter=5, projector_method=method))
    ref = _ctm_via_2x2(A, CTMConfig(chi=4, max_iter=5))
    for name in CTMEnvironment._fields:
        np.testing.assert_array_equal(getattr(env, name), getattr(ref, name))


def test_ctm_default_config_gives_no_user_warning():
    """The ``projector_method`` warning fires only when it applies."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctm(_random_site(), CTMConfig(chi=4, max_iter=5))
    assert not [w for w in caught if "projector_method" in str(w.message)]


def test_ctm_keeps_the_imaginary_part_of_a_complex_state():
    """#842: complex site tensors keep working.  The edges of this state have
    imaginary parts up to 0.57 (measured); a ``.real`` on the way would zero
    them."""
    key = jax.random.PRNGKey(0)
    real = jax.random.normal(key, (2, 2, 2, 2, 2))
    imag = jax.random.normal(jax.random.fold_in(key, 1), (2, 2, 2, 2, 2))
    A = (real + 1j * imag).astype(jnp.complex128)
    env = ctm(A, CTMConfig(chi=4, max_iter=6, conv_tol=1e-10))
    assert jnp.iscomplexobj(env.T1)
    edges = (env.T1, env.T2, env.T3, env.T4)
    assert max(float(jnp.max(jnp.abs(t.imag))) for t in edges) > 1e-3


def test_ctm_return_meta_matches_the_helper():
    """``ctm()`` is the helper plus the warnings: same environment, same info."""
    A = jnp.asarray(ising_peps(0.2))
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env, info = ctm(A, cfg, return_meta=True)
    ref_env, ref_info = _ctm_via_2x2(A, cfg, return_meta=True)
    assert info == ref_info
    for name in CTMEnvironment._fields:
        np.testing.assert_array_equal(getattr(env, name), getattr(ref_env, name))
