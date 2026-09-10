"""The plain symmetric CTM's reduced density matrices are density matrices.

This is #853's severity-attribution control, relocated from
``test_u1sz_defrag_prototype_610.py`` when that suite was deleted with the
closed #610 prototype (review of #959): it never touched the prototype's
``sector_dropping_truncation`` and is the only test that checks the plain
U(1)-Sz symmetric CTM path for positive-semidefinite RDMs and a physically
attainable energy.  The other PSD coverage checks the *detector* on synthetic
matrices (``test_rdm_validity_guard.py``) or runs the dense path, so without
this file a regression reproducing #853's indefinite RDM on the symmetric
path would fail nowhere.
"""

import jax
import numpy as np
import pytest

from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

# ``compute_energy_ctm_tensor`` returns ``E_h + E_v`` with ``H = S_i . S_j``,
# whose eigenvalues are -3/4 (singlet) and +1/4 (triplet).  Two bonds, so any
# genuine expectation value lies in [-1.5, +0.5] whatever the state and however
# poorly converged the environment: an inaccurate RDM is still a density
# matrix.  Leaving this interval means the RDM is not one, which is a different
# and worse failure than an inaccurate energy.
E_MIN, E_MAX = -1.5, 0.5

#: How far below zero the RDM spectrum may reach, **relative to its spectral
#: radius**.  Relative because the RDM is trace-normalised but its radius is
#: not fixed, so an absolute floor would mean different things at different
#: sweep counts.  Roundoff here is ~1e-16 of the radius; the case this guards
#: sits at -0.8 of it, so the exact value is not delicate.
RDM_PSD_TOL = 1e-8


def _rdm_min_eig_rel(rdm) -> float:
    """Smallest eigenvalue of a trace-normalised RDM, over its spectral radius."""
    M = np.asarray(rdm)
    M = M.reshape(M.shape[0] * M.shape[1], -1)
    M = M / np.trace(M)
    w = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    return float(w.min()) / max(float(np.max(np.abs(w))), 1e-300)


# One budget, not the original [20, 50] pair: the docstring's table shows every
# structural signal (positivity, rank, attainable energy) is budget-invariant
# from 20 to 100 sweeps, and a single symmetric convergence at this size costs
# ~44 min cold (measured 2026-09-10, a box ~2x faster than CI) -- the second
# budget was an hour of bucket time proving the same property (#960).
@pytest.mark.parametrize("max_iter", [20])
def test_the_plain_symmetric_ctm_keeps_the_rdm_positive(max_iter):
    """The control for #853: on the plain symmetric path, the RDM is a state.

    #853 reported ``E = +0.759`` from this fixture -- outside the ``[-1.5,
    +0.5]`` attainable for ``E_h + E_v`` with ``H = S_i . S_j`` -- and traced
    it to a reduced density matrix whose spectrum was ``[-1.2997, -0.0736,
    0.7446, 1.6287]`` at a trace of exactly 1.  It left open whether that came
    from the #610 sector-dropping prototype or from the symmetric 2x2 CTM
    underneath it, and said so explicitly, because the two have very different
    severities.

    It was the prototype.  Measured on the same tensor, same D=3, same chi=12,
    with dropping **off**::

        max_iter   criterion   corner rank   min eig / radius   energy
        20         1.9135e-02  4             +7.3027e-02        +0.166785
        30         2.4203e-02  4             +7.1104e-02        +0.113021
        40         1.3546e-02  4             +8.4792e-02        +0.089022
        50         2.2158e-02  4             +6.6507e-02        +0.080313
        60         2.4612e-02  4             +5.2411e-02        -0.096037
        100        2.6816e-02  4             +5.2767e-02        +0.183652

    Positive at every budget, corner rank 4 throughout, every energy inside
    the attainable interval -- against ``-0.798`` of the radius and ``+0.759``
    with dropping on at 50 sweeps.  The prototype deleted whole *charge*
    sectors from the environment's chi bonds; a CTM reduced density matrix is
    positive because ket and bra layers are truncated *consistently*, and a
    hard projection selected by charge rather than by weight is not that kind
    of truncation.

    The prototype is gone (#610 closed), but the attribution above stays
    falsifiable only while this control runs: if the plain path ever starts
    producing an indefinite RDM on this input, the defect class moves into the
    symmetric CTM itself and #853's severity jumps.  Convergence is
    deliberately not asserted -- under ``2x2`` the criterion does not reach
    ``conv_tol`` even at 400 sweeps on this input (see #853's remaining half).
    """
    from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor, _rdm2x1_tensor

    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _diff = ctm_tensor(A, chi=12, max_iter=max_iter, conv_tol=1e-7)

    rank = ctm_corner_rank(env)
    assert rank > 1, (
        f"the control collapsed to a rank-{rank} corner, so it is not a "
        f"control -- a chi_eff=1 environment cannot distinguish anything "
        f"about RDM positivity (#723, #726, #747)."
    )

    for name, rdm in (
        ("2x1 horizontal", _rdm2x1_tensor(A, env)),
        ("1x2 vertical", _rdm1x2_tensor(A, env)),
    ):
        min_eig = _rdm_min_eig_rel(rdm)
        assert min_eig >= -RDM_PSD_TOL, (
            f"the {name} RDM from the plain symmetric CTM reaches "
            f"{min_eig:.4e} of its spectral radius below zero, so it is not a "
            f"density matrix. That moves #853's defect class into the "
            f"symmetric 2x2 CTM itself, which is considerably more serious "
            f"than the retired prototype. Measured +6.7e-02 .. +8.5e-02 "
            f"across 20-100 sweeps when this was written."
        )

    e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert E_MIN <= e <= E_MAX, (
        f"control energy {e} is outside [{E_MIN}, {E_MAX}] with no sector "
        f"dropping anywhere -- see the positivity assertion above for why an "
        f"out-of-range energy means the RDM is not a state rather than that "
        f"the environment is merely unconverged."
    )
