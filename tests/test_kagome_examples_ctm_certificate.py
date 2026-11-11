"""The kagome PESS examples certify their CTM energy (#1183 follow-up 3).

``examples/kagome_xxz_pess.py`` and ``examples/kagome_xxz_spin1_pess.py`` put
each up-triangle on one square-lattice site whose ``r`` leg is a dummy:
``pess_to_ipeps`` fills only ``A[:, :, :, 0, :]``.  Every horizontal bond then
carries only index 0, and every CTM corner is exactly rank 1 at any ``chi``.
The corner-spectrum criterion of ``ctm_tensor`` cannot see such a CTM converge
-- the spectrum is ``[1, 0, ...]`` at every sweep while the edge tensors still
move -- so for D > 1 it reports ``converged=False`` with ``diff=inf`` at every
budget (#898).  The examples dropped that report and printed the energy
without a word.

``compute_pess_energy`` now returns ``(E/site, certified, notice)``.  When the
corner criterion is blind, it reruns the CTM with ``max_iter + 1`` and with
``2 * max_iter`` sweeps, and certifies the energy only if neither rerun moves
it by more than ``energy_tol``.  The first rerun catches a CTM cycle, whose
period can divide ``max_iter`` and hide it from the second; the second catches
a slow convergence, which a one-sweep change underestimates.

The oracle shares no code with the CTM.  With only index 0 on every
horizontal bond, each column is an independent infinite chain of
``A[:, :, 0, 0, :]``.  The dominant left and right eigenvectors of the column
transfer matrix give the single-site density matrix, and
E/site = 2 tr(rho H_tri) / 3: the examples measure ``H_tri (x) I`` on the
horizontal and on the vertical bond.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import examples.kagome_xxz_pess as spin_half  # noqa: E402
import examples.kagome_xxz_spin1_pess as spin_one  # noqa: E402
from tenax import compute_energy_ctm_tensor, ctm_tensor  # noqa: E402
from tenax.algorithms._ctm_diagnostics import env_is_collapsed  # noqa: E402
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor  # noqa: E402

CHI = 4

# (module, triangle Hamiltonian, sweeps that leave the energy wrong in the
# printed digits).  The last value is measured on the states of ``states``:
# |E - exact| is 2.3e-5 after 4 sweeps (spin-1/2) and 1.6e-4 after 2 (spin-1).
EXAMPLES = {
    "spin-1/2": (spin_half, spin_half.kagome_triangle_hamiltonian(1.0), 4),
    "spin-1": (spin_one, spin_one.kagome_triangle_hamiltonian_spin1(1.0), 2),
}
PHYS_DIM = {"spin-1/2": 2, "spin-1": spin_one.D_PHYS}


def _column_transfer(A):
    """A column site ``a[u, d, p]`` and its transfer matrix, top to bottom."""
    a = np.asarray(A)[:, :, 0, 0, :]  # l = r = 0: all that a horizontal bond keeps
    D = a.shape[0]
    # T[(u, u'), (d, d')] = sum_p a[u, d, p] conj(a[u', d', p]).
    return a, np.einsum("udp,vep->uvde", a, a.conj()).reshape(D * D, D * D)


def _exact(A, H_tri):
    """E/site of the decoupled-column state and the gap of its transfer matrix."""
    a, T = _column_transfer(A)
    D = a.shape[0]
    w, vr = np.linalg.eig(T)
    wl, vl = np.linalg.eig(T.T)
    R = vr[:, np.argmax(np.abs(w))].reshape(D, D)  # from below, on (d, d')
    L = vl[:, np.argmax(np.abs(wl))].reshape(D, D)  # from above, on (u, u')
    rho = np.einsum("uv,udp,veq,de->pq", L, a, a.conj(), R)
    rho = rho / np.trace(rho)
    mags = np.sort(np.abs(w))[::-1]
    gap_ratio = mags[1] / mags[0] if D > 1 else 0.0
    return float(np.real(2 * np.trace(rho @ H_tri) / 3)), gap_ratio


def _ctm_energy(A, H_tri, sweeps):
    """The examples' E/site after ``sweeps`` CTM sweeps, with no certificate."""
    d_eff = A.shape[4]
    H = jnp.array(np.kron(H_tri, np.eye(d_eff)).reshape((d_eff,) * 4))
    A_t = _wrap_as_dense_tensor(A)
    env, _ = ctm_tensor(A_t, CHI, max_iter=sweeps)
    return float(compute_energy_ctm_tensor(A_t, env, H, d_eff)) / 3.0


def _column_state(d, stay, flip):
    """A D=2 PESS state whose column transfer matrix is known in closed form.

    Going down a column, the vertical index ``u`` stays (amplitude ``stay``)
    or flips (amplitude ``flip``), and the physical triple records which:
    ``(u, 1 - u, 0)`` or ``(u, u, 0)``.  The column transfer matrix then has
    the eigenvalues ``stay**2 + flip**2`` and ``stay**2 - flip**2``, and zeros.
    A gauge ``G = diag(2, 1)`` on the vertical bond leaves the state unchanged,
    but not the cold-start boundary: for ``stay=0, flip=1`` the CTM energy
    alternates with the parity of the sweep count, and with ``G = I`` it is
    1/6 after 20, 21 and 40 sweeps.
    """
    D = 2
    G = np.diag([2.0, 1.0])
    S_a, S_b, S_c = (np.zeros((D, D, d)) for _ in range(3))
    for k in range(2):
        S_a[k, k, k] = 1.0  # ext = int = phys
        S_b[1 - k, k, k] = flip  # ext = 1 - int, phys = int
        S_b[k, k, 1 - k] = stay  # ext = int, phys = 1 - int
    S_c[0, 0, 0] = 1.0
    T_up = np.zeros((D, D, D))
    T_up[0, 0, 0] = T_up[1, 1, 0] = 1.0
    S_a = np.einsum("ux,xap->uap", G, S_a)
    S_b = np.einsum("yd,ybp->dbp", np.linalg.inv(G), S_b)
    return (
        [jnp.array(S_a), jnp.array(S_b), jnp.array(S_c)],
        [jnp.array(T_up), jnp.ones((D, D, D))],  # T_down: pess_to_ipeps ignores it
        [jnp.ones(D) for _ in range(6)],
    )


@pytest.fixture(scope="module")
def states():
    """One D=2 PESS state per example, each with a gapped column transfer matrix.

    The random spin-1/2 start has |lambda2/lambda1| = 0.78 and needs ~100
    sweeps; 20 simple-update steps bring it to 0.10.  The random spin-1 start
    already has 0.035.
    """
    H_half = EXAMPLES["spin-1/2"][1]
    return {
        "spin-1/2": spin_half.pess_simple_update(
            *spin_half.init_pess(2), H_half, 0.01, 2, 20
        ),
        "spin-1": spin_one.init_pess(2),
    }


@pytest.fixture
def ctm_budgets(monkeypatch):
    """Record the ``max_iter`` of every ``ctm_tensor`` call an example makes."""

    def install(module):
        calls = []
        real = module.ctm_tensor
        signature = inspect.signature(real)

        def spy(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append(bound.arguments["max_iter"])
            return real(*args, **kwargs)

        monkeypatch.setattr(module, "ctm_tensor", spy)
        return calls

    return install


def _blind_corner_regime(A, max_iter):
    """The dummy ``r`` leg makes the corner criterion unable to certify."""
    assert np.all(np.asarray(A)[:, :, :, 1:, :] == 0), "r leg is not a dummy"
    env, _, info = ctm_tensor(
        _wrap_as_dense_tensor(A), CHI, max_iter=max_iter, return_meta=True
    )
    assert env_is_collapsed(env)
    assert not info.converged and info.diff == float("inf"), info


@pytest.mark.parametrize("name", EXAMPLES)
def test_blind_corner_criterion_certified_by_energy(name, states, ctm_budgets):
    module, H_tri, _ = EXAMPLES[name]
    state = states[name]
    A = module.pess_to_ipeps(*state)
    E_exact, gap_ratio = _exact(A, H_tri)
    assert gap_ratio < 0.2, gap_ratio
    _blind_corner_regime(A, max_iter=20)

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(*state, H_tri, CHI, max_iter=20)

    assert certified, notice
    assert "NOT" not in notice, notice
    assert budgets == [20, 21, 40]
    assert abs(E - E_exact) < 1e-10, (E, E_exact)


@pytest.mark.parametrize("name", EXAMPLES)
def test_unconverged_energy_is_not_certified(name, states, ctm_budgets):
    module, H_tri, few = EXAMPLES[name]
    state = states[name]
    A = module.pess_to_ipeps(*state)
    E_exact, _ = _exact(A, H_tri)

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(*state, H_tri, CHI, max_iter=few)

    # The regime: after ``few`` sweeps the printed energy is still wrong.
    assert abs(E - E_exact) > 1e-6, (E, E_exact)
    assert not certified, notice
    assert "NOT" in notice, notice
    assert budgets == [few, few + 1, 2 * few]


@pytest.mark.parametrize("name", EXAMPLES)
def test_cycle_is_not_certified(name, ctm_budgets):
    """A CTM energy with period 2 is not certified, even after an even budget."""
    module, H_tri, _ = EXAMPLES[name]
    state = _column_state(PHYS_DIM[name], stay=0.0, flip=1.0)
    A = module.pess_to_ipeps(*state)
    # The regime: two transfer eigenvalues of equal size and opposite sign, and
    # an energy that alternates, so E after 20 and after 40 sweeps agree.
    top = sorted(np.linalg.eigvals(_column_transfer(A)[1]), key=abs)[-2:]
    assert np.isclose(top[0], -top[1]) and abs(top[1]) > 0.5, top
    E20, E21, E40 = (_ctm_energy(A, H_tri, n) for n in (20, 21, 40))
    assert abs(E20 - E40) < 1e-12, (E20, E40)
    assert abs(E20 - E21) > 1e-3, (E20, E21)

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(*state, H_tri, CHI, max_iter=20)

    assert not certified, notice
    assert "NOT" in notice, notice
    assert budgets == [20, 21, 40]


@pytest.mark.parametrize("name", EXAMPLES)
def test_slow_convergence_is_not_certified(name, ctm_budgets):
    """A slow, monotone CTM convergence is not certified by its small last step."""
    module, H_tri, _ = EXAMPLES[name]
    state = _column_state(PHYS_DIM[name], stay=1.0, flip=0.2)
    A = module.pess_to_ipeps(*state)
    E_exact, gap_ratio = _exact(A, H_tri)
    # The regime: |lambda2/lambda1| = 0.96/1.04, so after 20 sweeps one more
    # sweep moves the energy by ~1/12 of its error.  A tolerance between the
    # one-sweep and the doubled-budget change passes the first and fails the
    # second; the energy is wrong by more than that tolerance.
    assert gap_ratio > 0.9, gap_ratio
    E20, E21, E40 = (_ctm_energy(A, H_tri, n) for n in (20, 21, 40))
    step, drift = abs(E21 - E20), abs(E40 - E20)
    assert drift > 5 * step, (step, drift)
    tol = float(np.sqrt(step * drift))
    assert abs(E20 - E_exact) > tol, (E20, E_exact, tol)

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(
        *state, H_tri, CHI, max_iter=20, energy_tol=tol
    )

    assert not certified, notice
    assert "NOT" in notice, notice
    assert budgets == [20, 21, 40]


@pytest.mark.parametrize("name", EXAMPLES)
@pytest.mark.parametrize("max_iter", [0, -1])
def test_budget_below_one_is_rejected(name, max_iter, states, ctm_budgets):
    """No sweep runs, so nothing can be certified; ``ctm_tensor`` accepts it."""
    module, H_tri, _ = EXAMPLES[name]
    budgets = ctm_budgets(module)

    with pytest.raises(ValueError, match="max_iter"):
        module.compute_pess_energy(*states[name], H_tri, CHI, max_iter=max_iter)

    assert budgets == []


@pytest.mark.parametrize("name", EXAMPLES)
def test_corner_criterion_certifies_d1(name, ctm_budgets):
    """At D=1 the corner rank 1 is full rank, so ``ctm_tensor`` certifies it."""
    module, H_tri, _ = EXAMPLES[name]
    state = module.init_pess(1)
    A = module.pess_to_ipeps(*state)
    E_exact, _ = _exact(A, H_tri)
    _, _, info = ctm_tensor(
        _wrap_as_dense_tensor(A), CHI, max_iter=10, return_meta=True
    )
    assert info.converged, info

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(*state, H_tri, CHI, max_iter=10)

    assert certified, notice
    assert "NOT" not in notice, notice
    assert budgets == [10], "the corner criterion certified it; no energy check"
    assert abs(E - E_exact) < 1e-10, (E, E_exact)


@pytest.mark.parametrize("name", EXAMPLES)
def test_informative_unconverged_corner_is_not_certified(
    name, states, ctm_budgets, monkeypatch
):
    """A corner of rank > 1 that did not converge gets no energy check.

    ``pess_to_ipeps`` always gives a rank-1 corner, so the test makes
    ``env_is_collapsed`` say otherwise.
    """
    module, H_tri, _ = EXAMPLES[name]
    monkeypatch.setattr(module, "env_is_collapsed", lambda env, *a, **k: False)

    budgets = ctm_budgets(module)
    E, certified, notice = module.compute_pess_energy(
        *states[name], H_tri, CHI, max_iter=20
    )

    assert not certified, notice
    assert "NOT" in notice, notice
    assert budgets == [20]
