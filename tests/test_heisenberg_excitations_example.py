"""The Heisenberg excitations example reaches its spectrum (#1187).

``examples/heisenberg_ipeps_excitations.py`` did not reach its spectrum.  In
its ground-state stage, ``optimize_gs_ad`` with Adam on the plain
``heisenberg_gate()`` raised ``CTMNotConvergedError`` at optimization step 72.
A 1x1 cell also needs the sublattice-rotated gate, because one tensor sits on
both sublattices of the Neel state.  The example now optimises with the
recommended configuration of the AD guide (chi=8, then chi=16, on the rotated
gate) and takes its spectrum from ``compute_spectrum`` at X, (0, pi) and M.
The deprecated ``compute_excitations``, which truncates the momentum sum
(#1052), is gone from it.

The ground state runs for real.  ``compute_spectrum`` is replaced by a
recorder: it takes minutes at D=2, and the library's spectrum tests cover it.
The recorder checks what the example passes in, and returns a result of the
real type, so the printing runs too.
"""

import importlib.util
import pathlib
import re

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from _ipeps_spectrum_fixtures import E_D2, SY_COMPLEX  # noqa: E402

import tenax  # noqa: E402
from tenax import SpectrumResult, heisenberg_gate, sublattice_rotate_gate  # noqa: E402

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"

# E_D2 is the energy of Hasik's D=2 state.  The example's configuration reached
# -0.6602310935 on main at ab820d20 (2026-11-18), 2e-11 from it.
ENERGY_TOL = 1e-6
MOMENTA = [(np.pi, 0.0), (0.0, np.pi), (np.pi, np.pi)]


def _load_example(name):
    """Load an example script by its path, as the other example tests do."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array(x):
    return np.asarray(x.todense() if hasattr(x, "todense") else x)


def _same_momentum(k, target):
    """True when ``k`` and ``target`` agree modulo 2 pi."""
    diff = np.asarray(k, dtype=float) - np.asarray(target, dtype=float)
    return np.allclose(np.mod(diff + np.pi, 2 * np.pi) - np.pi, 0.0, atol=1e-12)


def _is_sy_up_to_phase(op):
    """True when ``op = c S^y`` with ``|c| = 1``, so it has the weights of S^y."""
    op = np.asarray(op)
    norm = np.linalg.norm(SY_COMPLEX)
    return (
        op.shape == (2, 2)
        and np.isclose(np.linalg.norm(op), norm)
        and np.isclose(abs(np.vdot(SY_COMPLEX, op)), norm**2)  # Cauchy-Schwarz
    )


def _fake_result(operators):
    return SpectrumResult(
        omega=np.array([0.65, 2.3, 3.0]),
        weights={name: np.array([0.7, 0.1, 0.0]) for name in operators},
        structure_factor={name: 0.8 for name in operators},
        rank=3,
        energy_density=E_D2,
        n_asymmetry=0.0,
        h_asymmetry=0.0,
        n_min_eig=1e-3,
        n_negative_modes=0,
        sum_rule_deficit={name: 0.0 for name in operators},
        gauge_residual=1e-12,
        null_dim=1,
    )


def test_example_reaches_its_spectrum(monkeypatch, capsys):
    example = _load_example("heisenberg_ipeps_excitations")
    # The library function, not a local wrapper around compute_excitations.
    assert example.compute_spectrum is tenax.compute_spectrum
    assert not hasattr(example, "compute_excitations")

    calls = []

    def recorder(A, gate, momentum, operators, config, **kwargs):
        calls.append((A, gate, momentum, operators, kwargs))
        return _fake_result(operators)

    monkeypatch.setattr(example, "compute_spectrum", recorder)
    example.main()
    out = capsys.readouterr().out

    energies = [float(e) for e in re.findall(r"E/site = (-?\d+\.\d+)", out)]
    assert energies, out
    assert abs(energies[0] - E_D2) < ENERGY_TOL, energies

    assert len(calls) == len(MOMENTA)
    for target in MOMENTA:
        assert sum(_same_momentum(call[2], target) for call in calls) == 1, target

    rotated = _array(sublattice_rotate_gate(heisenberg_gate()))
    for A, gate, _, operators, kwargs in calls:
        assert _array(A).shape == (2, 2, 2, 2, 2)
        assert np.allclose(_array(gate), rotated)
        assert any(_is_sy_up_to_phase(op) for op in operators.values())
        assert kwargs.get("gate_nnn") is None  # J2 = 0
