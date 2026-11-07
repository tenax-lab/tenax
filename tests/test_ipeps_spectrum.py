"""compute_spectrum and spectral_function: results, errors, warnings, checkpoints.

Most tests share the checkpoint ``"x"``, so only the first ``_run`` computes columns
and the others resume. A test that needs fresh columns takes its own ``name``.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    SY_COMPLEX,
    SY_REAL,
    SZ,
    apply_op,
    exact_u1_d2,
    hasik_d2,
    rotated_gate,
)

import tenax.algorithms._ad_primitives as ad_module
import tenax.algorithms._ipeps_excitation_ctm as ctm_module
import tenax.algorithms.ipeps_excitations as spectrum_module
from tenax import SpectrumConfig, SpectrumResult, compute_spectrum, spectral_function
from tenax.algorithms._ctm_tensor_convergence import _ctm_tensor_sweep_multisite
from tenax.algorithms._ctm_tensor_projector_2x2 import _SPLIT_BACKWARD_WIDTH
from tenax.algorithms._ipeps_excitation_ctm import CELL, NEIGHBORS, ExcitationCTM
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_excitations import (
    _gate_scale,
    _gauge_vectors,
    _reduce_spectrum,
    _spectrum_gate,
    _spectrum_momentum,
    _warn_spectrum,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

X_POINT = (np.pi, 0.0)
M_POINT = (np.pi, np.pi)
SCALE_TOL = 1e-6


def _basis(*vectors):
    """Orthonormal columns: the given vectors and one random direction."""
    A = hasik_d2()
    rng = np.random.default_rng(11)
    stack = [v.ravel() for v in vectors] + [rng.normal(size=A.size)]
    Q, _ = np.linalg.qr(np.stack(stack, axis=1))
    return Q


@pytest.fixture(scope="module")
def basis():
    A = hasik_d2()
    return _basis(apply_op(A, SY_REAL), A)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tmp_path_factory.mktemp("spectrum")


def _config(checkpoint, name, **kwargs):
    return SpectrumConfig(chi=10, checkpoint_dir=checkpoint / name, **kwargs)


def _run(checkpoint, basis, momentum=X_POINT, operators=None, name="x", **kwargs):
    operators = {"Sy": SY_REAL} if operators is None else operators
    return compute_spectrum(
        hasik_d2(),
        rotated_gate(),
        momentum,
        operators,
        _config(checkpoint, name, **kwargs),
        basis=basis,
    )


class _Stop(Exception):
    pass


@pytest.fixture
def no_ctm(monkeypatch):
    """Fail if a call gets as far as the expensive part."""

    def refuse(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(ctm_module, "_converged_env", refuse)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("momentum", "name", "expected"),
    [(X_POINT, "x", 0.203989), (M_POINT, "m", 0.808568)],
    ids=["X", "M"],
)
def test_structure_factor_and_sum_rule(checkpoint, basis, momentum, name, expected):
    result = _run(checkpoint, basis, momentum, name=name)
    assert result.structure_factor["Sy"] == pytest.approx(expected, abs=2e-5)
    assert result.weights["Sy"].sum() == pytest.approx(expected, abs=2e-5)
    assert abs(result.sum_rule_deficit["Sy"]) < 1e-6
    assert result.gauge_residual is None


def test_complex_operator_equals_its_real_form(checkpoint, basis):
    result = _run(checkpoint, basis, operators={"re": SY_REAL, "im": SY_COMPLEX})
    np.testing.assert_allclose(result.weights["im"], result.weights["re"], atol=1e-12)
    assert result.structure_factor["im"] == pytest.approx(
        result.structure_factor["re"], abs=1e-12
    )


def test_identity_operator_has_no_weight_and_no_warning(checkpoint, basis):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(checkpoint, basis, operators={"id": np.eye(2)})
    assert abs(result.structure_factor["id"]) < 1e-6
    assert result.sum_rule_deficit["id"] == 0.0
    assert np.isfinite(result.weights["id"]).all()


def test_operator_outside_the_basis_warns(checkpoint, basis):
    with pytest.warns(UserWarning, match="outside basis"):
        _run(checkpoint, basis, operators={"Sz": SZ})


def test_scaling_A_changes_nothing(checkpoint, basis):
    """Fresh columns. H and N scale together, so only a large factor shows a fault."""
    plain = _run(checkpoint, basis)
    scaled = compute_spectrum(
        1e-6 * hasik_d2(),
        rotated_gate(),
        X_POINT,
        {"Sy": SY_REAL},
        _config(checkpoint, "scaled"),
        basis=basis,
    )
    np.testing.assert_allclose(scaled.omega, plain.omega, atol=SCALE_TOL)
    np.testing.assert_allclose(
        scaled.weights["Sy"], plain.weights["Sy"], atol=SCALE_TOL
    )


def test_rescaled_A_resumes_from_the_checkpoint(checkpoint, basis):
    plain = _run(checkpoint, basis)
    scaled = compute_spectrum(
        3.0 * hasik_d2(),
        rotated_gate(),
        X_POINT,
        {"Sy": SY_REAL},
        _config(checkpoint, "x"),
        basis=basis,
    )
    np.testing.assert_allclose(scaled.omega, plain.omega, atol=1e-12)
    np.testing.assert_allclose(scaled.weights["Sy"], plain.weights["Sy"], atol=1e-12)


def test_dense_tensor_inputs_match_raw_arrays(checkpoint, basis):
    """optimize_gs_ad returns a DenseTensor site tensor and gate (#636)."""
    plain = _run(checkpoint, basis)
    symmetry = U1Symmetry()
    gate_indices = tuple(
        TensorIndex.from_charges(
            symmetry, np.zeros(2, dtype=np.int32), FlowDirection.OUT, label=f"g{i}"
        )
        for i in range(4)
    )
    wrapped = compute_spectrum(
        _wrap_as_dense_tensor(jnp.asarray(hasik_d2())),
        DenseTensor(jnp.asarray(rotated_gate()), gate_indices),
        X_POINT,
        {"Sy": SY_REAL},
        _config(checkpoint, "x"),
        basis=basis,
    )
    np.testing.assert_allclose(wrapped.omega, plain.omega, atol=1e-12)
    assert wrapped.energy_density == pytest.approx(plain.energy_density, abs=1e-12)


def test_basis_inside_the_null_space_keeps_nothing():
    A = hasik_d2()
    with pytest.warns(UserWarning, match="no direction survives"):
        result = compute_spectrum(
            A,
            rotated_gate(),
            X_POINT,
            {"id": np.eye(2)},
            SpectrumConfig(chi=10),
            basis=A.reshape(-1, 1),
        )
    assert result.rank == 0
    assert result.omega.shape == (0,)
    assert spectral_function(result, "id", np.linspace(0, 1, 3)).tolist() == [
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize(
    ("eigenvalues", "rank", "null_dim"),
    [
        ([1e-9], 0, 1),  # noise: a purely relative cut would keep it as a mode
        ([-1e-9], 0, 1),
        ([5e-3], 1, 0),  # real weight in a small basis: above the floor
        ([1.0, 2e-4, 5e-5], 2, 1),  # at full scale the cut is 1e-4 of the largest
        ([1.0, -5e-5], 1, 1),
        ([1.0, -1.0], 1, 0),  # a negative eigenvalue is dropped, not counted as null
    ],
)
def test_n_cut_is_relative_with_an_absolute_floor(eigenvalues, rank, null_dim):
    N = np.diag(eigenvalues)
    omega, modes, _, found_null_dim, _ = _reduce_spectrum(N, np.zeros_like(N), 1e-4)
    assert modes.shape[1] == rank
    assert omega.shape == (rank,)
    assert found_null_dim == null_dim


@pytest.mark.parametrize(("a", "b"), [(1, 0), (0, 1), (1, 1)])
def test_gauge_span_has_the_measured_rank(a, b):
    A = hasik_d2()
    assert np.linalg.matrix_rank(_gauge_vectors(A, a, b), tol=1e-10) == 2 * 2**2 - 1


def test_a_and_gauge_vectors_are_null_in_n():
    # Random combination: a symmetric direction alone hides a wrong contraction.
    A = hasik_d2()
    rng = np.random.default_rng(5)
    ctm = ExcitationCTM(A, rotated_gate(), 10, 30)
    phases = ctm_module.cell_phases(1, 0)
    vectors = _gauge_vectors(A, 1, 0)
    combined = vectors[:, 1:] @ rng.normal(size=vectors.shape[1] - 1)
    for v in (A.ravel(), combined / np.linalg.norm(combined)):
        _, g_n = ctm.ket_derivative(phases, np.reshape(v, A.shape), 1e-4)
        assert float(np.linalg.norm(np.asarray(g_n))) < 1e-3


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("momentum", "expected"),
    [
        ((np.pi, 0.0), (1, 0)),
        ((0.0, np.pi), (0, 1)),
        ((np.pi, np.pi), (1, 1)),
        ((-np.pi, 0.0), (1, 0)),
        ((3 * np.pi, np.pi), (1, 1)),
        ([np.pi, 2 * np.pi], (1, 0)),
    ],
)
def test_momentum_is_taken_modulo_two_pi(momentum, expected):
    assert _spectrum_momentum(momentum) == expected


@pytest.mark.parametrize(
    "momentum",
    [(0.0, 0.0), (2 * np.pi, 0.0), (np.pi / 2, np.pi / 2), (np.pi,), (np.pi, 0.0, 0.0)],
)
def test_momentum_outside_the_three_names_them(momentum):
    with pytest.raises(ValueError, match=r"\(pi, 0\), \(0, pi\), \(pi, pi\)"):
        _spectrum_momentum(momentum)


# ---------------------------------------------------------------------------
# Errors: each raises before any CTM work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"A": 1j * hasik_d2()}, ValueError, "real"),
        ({"A": hasik_d2().astype(np.float32)}, ValueError, "float64"),
        ({"A": hasik_d2()[0]}, ValueError, "shape"),
        ({"gate": np.eye(3)}, ValueError, "gate"),
        ({"gate": 1j * rotated_gate()}, ValueError, "real"),
        ({"operators": {"bad": np.eye(3)}}, ValueError, "bad"),
        ({"basis": np.ones((5, 2))}, ValueError, "shape"),
        ({"basis": np.ones((32, 2))}, ValueError, "orthonormal"),
        ({"basis": 1j * np.eye(32)[:, :2]}, ValueError, "real"),
        (
            {"momentum": (0.0, 0.0)},
            ValueError,
            r"Gamma for S\^x and S\^z, not for S\^y",
        ),
    ],
)
def test_bad_input_raises_before_ctm_work(no_ctm, change, error, message):
    arguments = {
        "A": hasik_d2(),
        "gate": rotated_gate(),
        "momentum": X_POINT,
        "operators": {"Sy": SY_REAL},
        "basis": None,
    } | change
    basis = arguments.pop("basis")
    with pytest.raises(error, match=message):
        compute_spectrum(
            arguments["A"],
            arguments["gate"],
            arguments["momentum"],
            arguments["operators"],
            SpectrumConfig(chi=10),
            basis=basis,
        )


@pytest.mark.parametrize("fd_step", [0.0, -1e-4, float("nan")])
def test_bad_fd_step_raises_before_ctm_work(no_ctm, fd_step):
    with pytest.raises(ValueError, match="fd_step"):
        compute_spectrum(
            hasik_d2(),
            rotated_gate(),
            X_POINT,
            {"Sy": SY_REAL},
            SpectrumConfig(chi=10, fd_step=fd_step),
        )


@pytest.mark.parametrize(
    "gate",
    [
        rotated_gate().astype(np.complex128),
        rotated_gate() + 1e-16j,
        rotated_gate().astype(np.float32),
    ],
    ids=["no imaginary part", "imaginary roundoff", "float32"],
)
def test_real_gate_in_another_dtype_becomes_float64(gate):
    """S^y as a complex matrix gives a complex gate with no imaginary part."""
    found = _spectrum_gate(gate, 2, "gate")
    assert found.dtype == np.float64
    np.testing.assert_allclose(found, rotated_gate(), atol=1e-7)


@pytest.mark.parametrize("bad", ["h", "n"])
def test_non_finite_column_raises_and_is_not_saved(monkeypatch, tmp_path, basis, bad):
    """A saved NaN fails every rerun at eigh, with a message that names no file."""

    class NanInColumnOne:
        ctm_converged = True

        def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
            self.A = jnp.asarray(A)
            self.calls = 0

        def densities(self, phases, eps, X, Y):
            return jnp.zeros(()), jnp.zeros(())

        def ket_derivative(self, phases, X, h):
            self.calls += 1
            good = jnp.ones_like(self.A)
            nan = jnp.nan * good if self.calls == 2 else good
            return (nan, good) if bad == "h" else (good, nan)

    monkeypatch.setattr(spectrum_module, "ExcitationCTM", NanInColumnOne)
    with pytest.raises(RuntimeError, match="column 1 "):
        compute_spectrum(
            hasik_d2(),
            rotated_gate(),
            X_POINT,
            {"Sy": SY_REAL},
            SpectrumConfig(chi=10, checkpoint_dir=tmp_path),
            basis=basis,
        )
    assert [path.name for path in tmp_path.glob("column_*.npz")] == ["column_00000.npz"]


@pytest.mark.parametrize("bad", ["h", "n"])
@pytest.mark.parametrize("units", [1.0, 1e6], ids=["J=1", "J=1e6"])
def test_oversized_column_raises_and_is_not_saved(
    monkeypatch, tmp_path, basis, bad, units
):
    """A finite blow-up reaches eigh and gives a wrong spectrum, with no error at all.

    H is linear in the gate and N has no units, so the bound must follow only H.
    """

    class HugeInColumnOne:
        ctm_converged = True

        def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
            self.A = jnp.asarray(A)
            self.calls = 0

        def densities(self, phases, eps, X, Y):
            return jnp.zeros(()), jnp.zeros(())

        def ket_derivative(self, phases, X, h):
            self.calls += 1
            good = jnp.ones_like(self.A)
            huge = 1e4 * good if self.calls == 2 else good
            return (units * huge, good) if bad == "h" else (units * good, huge)

    monkeypatch.setattr(spectrum_module, "ExcitationCTM", HugeInColumnOne)
    with pytest.raises(RuntimeError, match="column 1 .*fd_step=1e-3"):
        compute_spectrum(
            hasik_d2(),
            units * rotated_gate(),
            X_POINT,
            {"Sy": SY_REAL},
            SpectrumConfig(chi=10, checkpoint_dir=tmp_path),
            basis=basis,
        )
    assert [path.name for path in tmp_path.glob("column_*.npz")] == ["column_00000.npz"]


def test_a_zero_gate_passes_the_column_check(monkeypatch, basis):
    """Only the norm is wanted; H = 0 in units of a zero gate must not read as 0 / 0."""

    class NormOnly:
        ctm_converged = True

        def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
            self.A = jnp.asarray(A)

        def densities(self, phases, eps, X, Y):
            return jnp.zeros(()), jnp.zeros(())

        def ket_derivative(self, phases, X, h):
            return jnp.zeros_like(self.A), X

    monkeypatch.setattr(spectrum_module, "ExcitationCTM", NormOnly)
    result = compute_spectrum(
        hasik_d2(),
        np.zeros((2, 2, 2, 2)),
        X_POINT,
        {"Sy": SY_REAL},
        SpectrumConfig(chi=10),
        basis=basis,
    )
    np.testing.assert_allclose(result.omega, 0.0)


def test_symmetric_tensor_is_refused(no_ctm):
    A = object.__new__(SymmetricTensor)  # the refusal must not touch its contents
    with pytest.raises(NotImplementedError, match="SymmetricTensor"):
        compute_spectrum(A, rotated_gate(), X_POINT, {}, SpectrumConfig(chi=10))


def test_many_columns_without_checkpoints_warns(no_ctm):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3, 3, 3, 2))
    with pytest.warns(UserWarning, match="162 columns"), pytest.raises(_Stop):
        compute_spectrum(A, rotated_gate(), X_POINT, {}, SpectrumConfig(chi=10))


# ---------------------------------------------------------------------------
# Exact U(1): the SVD backward of the sweep
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exact_u1_ctm():
    ctm = ExcitationCTM(exact_u1_d2(), rotated_gate(), 6, 30)
    corner = np.linalg.svd(np.asarray(ctm.envs0[(0, 0)].C1.todense()), compute_uv=False)
    assert abs(corner[1] - corner[2]) < 1e-10 * corner[0], "no exact pair: vacuous"
    assert corner[1] > 1e-2 * corner[0], "no exact pair: vacuous"
    return ctm


@pytest.mark.parametrize(
    ("column", "rows", "tolerance"),
    [(23, (10, 23), 1e-4), (0, (0, 5), 1e-5)],
    ids=["the pair splits", "the pair stays exact"],
)
def test_exact_u1_column_matches_finite_differences(
    exact_u1_ctm, column, rows, tolerance
):
    """Column 23: a pair of the projector's matrices splits as ``h^2`` at the FD points.
    With the Lorentzian of the ground-state path, its row 10 is -3155 where FD-FD gives
    -0.007, and other chi and fd_step move the blow-up to other columns.

    Column 0 keeps the symmetry, so the pair stays exact. A diagonal that depends on the
    numerical basis of the pair put 2e-4 to 5e-3 into its row 5, where FD-FD gives 0: the
    size depends on the basis that roundoff picks.
    """
    ctm = exact_u1_ctm
    A = ctm.A

    def unit(column):
        X = np.zeros(A.size)
        X[column] = 1.0
        return jnp.asarray(X.reshape(A.shape))

    phases = ctm_module.cell_phases(1, 0)
    g_h, g_n = (
        np.asarray(g).ravel() for g in ctm.ket_derivative(phases, unit(column), 1e-4)
    )
    h = 1e-3
    for row in rows:
        fd_h, fd_n = 0.0, 0.0
        for sign_ket in (1.0, -1.0):
            for sign_bra in (1.0, -1.0):
                e, n = ctm.densities(
                    phases, sign_ket * h, unit(column), sign_bra * h * unit(row)
                )
                fd_h += sign_ket * sign_bra * float(e) / (4 * h * h)
                fd_n += sign_ket * sign_bra * float(n) / (4 * h * h)
        assert g_h[row] == pytest.approx(fd_h, abs=tolerance)
        assert g_n[row] == pytest.approx(fd_n, abs=1e-5)
    assert np.abs(g_h).max() < 1.0


def test_only_the_spectrum_sweep_takes_the_split_backward(monkeypatch):
    seen = []
    backward = ad_module._svd_sector_backward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("split_eps"))
        return backward(*args, **kwargs)

    monkeypatch.setattr(ad_module, "_svd_sector_backward", spy)
    ctm = ExcitationCTM(hasik_d2(), rotated_gate(), 4, 1)
    phases = ctm_module.cell_phases(1, 0)
    ctm.ket_derivative(phases, jnp.zeros_like(ctm.A), 1e-4)
    assert seen, "the spectrum sweep reached no SVD backward"
    assert set(seen) == {_SPLIT_BACKWARD_WIDTH}

    # What the ground-state path passes: its gradients must not move.
    seen.clear()

    def ground_state_sweep(tensor):
        dl = {c: ctm._double_layer(tensor, tensor) for c in CELL}
        envs = _ctm_tensor_sweep_multisite(
            ctm.envs0, dl, NEIGHBORS, 4, True, "svd", "flow", "2x2"
        )[0]
        return jnp.real(jnp.sum(envs[(0, 0)].C1.todense()))

    jax.grad(ground_state_sweep)(ctm.A)
    assert seen, "the ground-state sweep reached no SVD backward"
    assert set(seen) == {None}


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _clean_result(**changes):
    fields = {
        "omega": np.array([1.0]),
        "weights": {"Sy": np.array([0.2])},
        "structure_factor": {"Sy": 0.2},
        "rank": 1,
        "energy_density": -0.66,
        "n_asymmetry": 0.0,
        "h_asymmetry": 0.0,
        "n_min_eig": 0.0,
        "n_negative_modes": 0,
        "sum_rule_deficit": {"Sy": 0.0},
        "gauge_residual": 0.0,
        "null_dim": 7,
    }
    return SpectrumResult(**(fields | changes))


def test_clean_result_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_spectrum(_clean_result(), 0.5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rank": 0}, "no direction survives"),
        ({"n_asymmetry": 2e-4}, "N asymmetry"),
        ({"h_asymmetry": 2e-3}, "H asymmetry"),
        ({"n_min_eig": -2e-4}, "lowest N eigenvalue"),
        ({"n_negative_modes": 2}, "raise chi"),
        ({"sum_rule_deficit": {"Sy": 2e-3}}, "sum rule deficit"),
        ({"sum_rule_deficit": {"Sy": -2e-3}}, "sum rule deficit"),
        ({"gauge_residual": 2e-3}, "gauge residual"),
    ],
)
def test_each_diagnostic_warns_above_its_threshold(change, message):
    with pytest.warns(UserWarning, match=message):
        _warn_spectrum(_clean_result(**change), 0.5)


def test_h_asymmetry_limit_follows_the_gate():
    """H is linear in the gate, so the limit must move with it, in both directions."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_spectrum(_clean_result(h_asymmetry=9e-4), 0.5)
        _warn_spectrum(_clean_result(h_asymmetry=2e-3), 10.0)
    for asymmetry, gate_scale in [(1.1e-3, 0.5), (2e-6, 1e-4)]:
        with pytest.warns(UserWarning, match="H asymmetry"):
            _warn_spectrum(_clean_result(h_asymmetry=asymmetry), gate_scale)


@pytest.mark.filterwarnings("error:compute_spectrum")
def test_the_driver_gives_the_h_limit_the_scale_of_its_gates(monkeypatch, basis):
    """An asymmetry far above 1e-3, and small against a gate of 1e6."""

    class SlightlyAsymmetric:
        ctm_converged = True

        def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
            self.A = jnp.asarray(A)
            self.calls = 0

        def densities(self, phases, eps, X, Y):
            return jnp.zeros(()), jnp.zeros(())

        def ket_derivative(self, phases, X, h):
            self.calls += 1
            return 1e3 * X + 10.0 * self.calls * jnp.ones_like(self.A), X

    monkeypatch.setattr(spectrum_module, "ExcitationCTM", SlightlyAsymmetric)
    result = compute_spectrum(
        hasik_d2(),
        1e6 * rotated_gate(),
        X_POINT,
        {"Sy": SY_REAL},
        SpectrumConfig(chi=10),
        basis=basis,
    )
    assert 1e-3 < result.h_asymmetry < 1e-3 * 1e6


def test_gate_scale_is_the_largest_entry_of_both_gates():
    assert _gate_scale(np.asarray(rotated_gate()), None) == 0.5
    gate, gate_nnn = np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2))
    gate[0, 1, 1, 0], gate_nnn[0, 1, 1, 0] = -0.5, -3.0
    assert _gate_scale(gate, None) == 0.5
    assert _gate_scale(gate, gate_nnn) == 3.0


def test_unconverged_ctm_warns(checkpoint, basis, monkeypatch):
    real = ctm_module._converged_env
    monkeypatch.setattr(ctm_module, "_converged_env", lambda *a: (real(*a)[0], False))
    with pytest.warns(UserWarning, match="did not converge"):
        _run(checkpoint, basis)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_finished_run_resumes_without_derivatives(checkpoint, basis, monkeypatch):
    first = _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise AssertionError("resumed run recomputed a column")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", refuse)
    second = _run(checkpoint, basis)
    np.testing.assert_array_equal(second.omega, first.omega)


def test_interrupted_run_equals_an_uninterrupted_run(
    tmp_path, checkpoint, basis, monkeypatch
):
    uninterrupted = _run(checkpoint, basis)
    real = ExcitationCTM.ket_derivative
    calls = []

    def interrupt(self, *args):
        if len(calls) == 1:
            raise KeyboardInterrupt
        calls.append(1)
        return real(self, *args)

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, basis)
    assert len(list((tmp_path / "x").glob("column_*.npz"))) == 1
    (tmp_path / "x" / "leftover.tmp").write_bytes(b"partial")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", real)
    resumed = _run(tmp_path, basis)
    np.testing.assert_allclose(resumed.omega, uninterrupted.omega, atol=1e-10)


def test_checkpoint_keeps_columns_when_operators_change(checkpoint, basis, monkeypatch):
    _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise AssertionError("changing operators must not recompute a column")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", refuse)
    result = _run(checkpoint, basis, operators={"id": np.eye(2)})
    assert "id" in result.weights


def test_checkpoint_from_another_run_raises_before_ctm_work(
    checkpoint, basis, monkeypatch
):
    _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(ctm_module, "_converged_env", refuse)
    with pytest.raises(ValueError, match="different run"):
        _run(checkpoint, basis, fd_step=2e-4)


def test_damaged_checkpoint_names_the_file(tmp_path, basis):
    directory = tmp_path / "x"
    directory.mkdir()
    (directory / "column_00000.npz").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="column_00000.npz"):
        _run(tmp_path, basis)


# ---------------------------------------------------------------------------
# spectral_function
# ---------------------------------------------------------------------------


def test_spectral_function_is_a_lorentzian_sum():
    result = _clean_result(
        omega=np.array([1.0, 2.0]), weights={"Sy": np.array([0.2, 0.1])}
    )
    assert spectral_function(result, "Sy", 1.0, eta=0.5) == pytest.approx(
        0.2 / (0.5 * np.pi) + 0.1 * (0.5 / np.pi) / 1.25
    )
    grid = np.linspace(-2000.0, 2000.0, 800001)
    S = spectral_function(result, "Sy", grid, eta=0.1)
    assert S.shape == grid.shape
    assert np.trapezoid(S, grid) == pytest.approx(0.3, abs=1e-3)


def test_spectral_function_refuses_bad_arguments():
    result = _clean_result()
    with pytest.raises(KeyError, match="Sy"):
        spectral_function(result, "Sq", np.zeros(1))
    with pytest.raises(ValueError, match="eta"):
        spectral_function(result, "Sy", np.zeros(1), eta=0.0)


def test_negative_modes_stay_in_the_spectral_function():
    result = _clean_result(omega=np.array([-1.0]), n_negative_modes=1)
    assert spectral_function(result, "Sy", -1.0, eta=0.1) > 0.0
