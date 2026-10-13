"""The CTM energy builds each open double layer once per call.

``compute_energy_ctm_tensor_2site`` used to call four RDM helpers that each
rebuilt the open double layer of both tensors: 8 builds for 2 distinct inputs.
The 1-site energy built it twice for one input.  The build is an eager
per-block Python loop on the symmetric path, so the repeats dominated
eager line-search energy probes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_tensor_energy as mod
from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor,
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor,
    _rdm2x1_tensor_2site,
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _indices(sym, virt, phys):
    F = FlowDirection
    mk = TensorIndex.from_charges
    v = np.asarray(virt, dtype=np.int32)
    p = np.asarray(phys, dtype=np.int32)
    return (
        mk(sym, v.copy(), F.OUT, label="u"),
        mk(sym, v.copy(), F.IN, label="d"),
        mk(sym, v.copy(), F.OUT, label="l"),
        mk(sym, v.copy(), F.IN, label="r"),
        mk(sym, p.copy(), F.IN, label="phys"),
    )


def _dense(seed, D=2, d=2):
    rng = np.random.RandomState(seed)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    return DenseTensor(data, _indices(U1Symmetry(), np.zeros(D), np.zeros(d)))


def _u1(seed, virt=(0, 1), phys=(0, 1)):
    return SymmetricTensor.random_normal(
        _indices(U1Symmetry(), virt, phys), jax.random.PRNGKey(seed)
    )


def _fermi(seed):
    return SymmetricTensor.random_normal(
        _indices(FermionParity(), (0, 1), (0, 1)), jax.random.PRNGKey(seed)
    )


CASES = {
    "dense": lambda s: _dense(s),
    "u1": lambda s: _u1(s),
    "fermionic": lambda s: _fermi(s),
}


def _gate(d=2):
    rng = np.random.RandomState(5)
    H = rng.standard_normal((d * d, d * d))
    return jnp.array((H + H.T) / 2).reshape(d, d, d, d)


@pytest.fixture
def count_builds(monkeypatch):
    calls = []
    real = mod._build_double_layer_open_tensor

    def counting(A, *a, **k):
        calls.append(A)
        return real(A, *a, **k)

    monkeypatch.setattr(mod, "_build_double_layer_open_tensor", counting)
    return calls


@pytest.mark.parametrize("kind", list(CASES))
def test_2site_builds_two_double_layers(kind, count_builds):
    A, B = CASES[kind](1), CASES[kind](2)
    env_A, env_B = initialize_ctm_tensor_env(A, 4), initialize_ctm_tensor_env(B, 4)
    count_builds.clear()
    compute_energy_ctm_tensor_2site(A, B, env_A, env_B, _gate(), d=2)
    assert len(count_builds) == 2
    assert {id(x) for x in count_builds} == {id(A), id(B)}


@pytest.mark.parametrize("kind", list(CASES))
def test_1site_builds_one_double_layer(kind, count_builds):
    A = CASES[kind](1)
    env = initialize_ctm_tensor_env(A, 4)
    count_builds.clear()
    compute_energy_ctm_tensor(A, env, _gate(), d=2)
    assert len(count_builds) == 1


@pytest.mark.parametrize("kind", list(CASES))
def test_2site_energy_equals_self_building_helpers(kind):
    """Bit-for-bit against the helpers' own (ao=None) build path.

    The reference lets every helper build its own double layers, i.e. the
    pre-change code path; it shares no prebuilt tensor with the function under
    test, and A != B, so an un-swapped (B, A) call changes the value.
    """
    A, B = CASES[kind](1), CASES[kind](2)
    env_A, env_B = initialize_ctm_tensor_env(A, 4), initialize_ctm_tensor_env(B, 4)
    H = _gate()
    rdms = [
        _rdm2x1_tensor_2site(A, B, env_A, env_B),
        _rdm2x1_tensor_2site(B, A, env_B, env_A),
        _rdm1x2_tensor_2site(A, B, env_A, env_B),
        _rdm1x2_tensor_2site(B, A, env_B, env_A),
    ]
    ref = 0.5 * sum(jnp.einsum("ijkl,klij->", r, H) for r in rdms).real
    got = compute_energy_ctm_tensor_2site(A, B, env_A, env_B, H, d=2)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # A != B in the fixture: the swapped bonds are not copies of the others.
    assert not np.array_equal(np.asarray(rdms[0]), np.asarray(rdms[1]))


@pytest.mark.parametrize("kind", list(CASES))
def test_1site_energy_equals_self_building_helpers(kind):
    A = CASES[kind](1)
    env = initialize_ctm_tensor_env(A, 4)
    H = _gate()
    ref = (
        jnp.einsum("ijkl,klij->", _rdm2x1_tensor(A, env), H)
        + jnp.einsum("ijkl,klij->", _rdm1x2_tensor(A, env), H)
    ).real
    got = compute_energy_ctm_tensor(A, env, H, d=2)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
