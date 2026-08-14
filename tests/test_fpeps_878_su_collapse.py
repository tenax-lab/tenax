"""#878: the fermionic simple update drove the state to exactly zero.

`fpeps()` returned a state whose every bond weight, site tensor and physical
tensor were **exactly 0.0**, by step 10 at D=2 with a 200-step default.  Root
cause is #667 in a module #844 never touched: ``sigma`` was stored as the new
lambda *and* ``sqrt(sigma)`` was absorbed into ``Gamma``, so the next step's
``_absorb_lambdas`` scaled that bond by lambda again and it carried
``lambda**1.5``.  #667's own title says "fermionic -> zero norm"; only the
bosonic half was fixed.

Measured before the fix, D=2, dt=0.05, ``min(lam_h)`` per step::

    1  2.9258e-01     3  2.8988e-03    10  0.0
    2  1.9897e-02     5  1.0936e-07    ...  0.0

**These assert on the spectrum, never on the norm.**  ``_normalize_tensor`` runs
last in the update, so ``|A|`` reads a healthy 1.0 at every step above until the
one where it is exactly 0 -- a norm check cannot see this coming, and an
``isfinite`` check passes on the corpse.  That is why the defect survived.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    spinless_fermion_gate,
)

jax.config.update("jax_enable_x64", True)


def _run(D=2, steps=40, dt=0.05, V=0.0, seed=0):
    """Returns ``(A, lam_h, lam_v)`` -- ``B`` dropped, these assert on bonds."""
    cfg = FPEPSConfig(D=D, t=1.0, V=V, dt=dt, num_imaginary_steps=steps)
    H = spinless_fermion_gate(cfg)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(seed))
    A_opt, _B_opt, lam_h, lam_v = _fpeps_simple_update(
        A, H, max_D=D, dt=dt, steps=steps
    )
    return A_opt, lam_h, lam_v


@pytest.mark.parametrize("steps", [10, 40])
def test_the_bond_spectrum_does_not_collapse_to_zero(steps):
    """#878's symptom, asserted on the spectrum rather than the norm.

    ``> 0`` rather than ``isfinite``: the failure produced a finite, entirely
    zero answer, so every finiteness check passed on it.
    """
    A, lam_h, lam_v = _run(steps=steps)

    for name, lam in (("lam_h", lam_h), ("lam_v", lam_v)):
        arr = np.asarray(lam)
        assert float(np.max(arr)) > 0.0, (
            f"{name} collapsed to all-zero after {steps} steps -- #878"
        )
        assert float(np.min(arr)) > 0.0, (
            f"{name} = {arr} after {steps} steps: a dead bond direction is how "
            f"the lambda**1.5 runaway starts (#878)"
        )
    assert float(A.norm()) > 0.0, "the site tensor collapsed to zero"


def test_the_lambda_normalisation_is_relative_not_additive():
    """Pinned at the scale where the additive epsilon inverts, not at O(1).

    ``sigma / (max(sigma) + EPS)`` is a no-op while ``max(sigma) >> EPS``: at
    O(1) it returns ``1 - 1e-15``, which any sane tolerance accepts.  It only
    misbehaves once ``max(sigma)`` approaches ``EPS = 1e-15``, where it returns
    a spectrum whose maximum is well below 1 -- the next absorb shrinks the
    state further and the runaway closes in two steps (#748, #865).

    So this asserts on the *normalisation the fermionic module uses*, fed a
    spectrum at that scale.  Asserting ``max(lam) == 1`` on an ordinary run
    cannot distinguish the two implementations and passes on the defect.
    """
    from tenax.algorithms.fermionic_ipeps import _normalise_lambda

    tiny = jnp.array([1e-15, 1e-16])
    out = _normalise_lambda(tiny)
    assert float(jnp.max(out)) == pytest.approx(1.0, rel=1e-12), (
        f"max = {float(jnp.max(out)):.6f} on a spectrum at EPS scale; the "
        f"additive epsilon halves it (#878)"
    )
    # and it must not be a no-op that merely dodges the small case
    ordinary = _normalise_lambda(jnp.array([4.0, 1.0]))
    np.testing.assert_allclose(np.asarray(ordinary), [1.0, 0.25], rtol=1e-15)


def test_the_physical_tensor_carries_each_bond_weight_once():
    """A bond must pick up ``lambda``, not ``lambda**2``.

    Both ends of every bond contribute, so the physical tensor takes
    ``sqrt(lambda)`` on each leg -- exactly what the bosonic
    ``_to_physical_tensor`` does.  Absorbing the full ``lambda`` squares every
    bond weight of the lattice.
    """
    from tenax.algorithms.fermionic_ipeps import _to_physical_fpeps_tensor

    D = 2
    A, lam_h, lam_v = _run(D=D, steps=20)
    phys = _to_physical_fpeps_tensor(A, lam_h, lam_v)

    assert float(phys.norm()) > 0.0, "physical tensor collapsed to zero"

    # sqrt, not full lambda: scaling a leg by lam instead of sqrt(lam) changes
    # the tensor unless lam is all ones, so compare against the explicit build.
    from tenax.core._tensor_utils import scale_bond_axis

    want = scale_bond_axis(A, "u", jnp.sqrt(lam_v))
    want = scale_bond_axis(want, "d", jnp.sqrt(lam_v))
    want = scale_bond_axis(want, "l", jnp.sqrt(lam_h))
    want = scale_bond_axis(want, "r", jnp.sqrt(lam_h))
    want = want * (1.0 / float(want.norm()))

    got = np.asarray(phys.todense())
    ref = np.asarray(want.todense())
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "NOT a D=3 defect -- the fermionic sweep is seed-dependent at every "
        "bond dimension.  Surviving seeds out of 5, 600 steps, dt=0.05: D=2 "
        "4/5, D=3 2/5, D=4 4/5, D=6 4/5.  Seed 0 (used here, and throughout "
        "the original investigation) happens to die at D=3 and D=6 and live at "
        "D=2 and D=4, which is what made this look like a bond-dimension bug.  "
        "Same basin behaviour as #869 on the bosonic path, so it is fixed by "
        "the no-stored-lambda rewrite rather than by anything local.  Strict, "
        "so it flags the moment that lands."
    ),
)
def test_a_nominally_D3_state_uses_its_third_bond_direction():
    """Surviving is not enough -- the state has to be genuinely D=3.

    #667's other guard, which caught a "D=3" result that was really D=2 wearing
    a D=3 shape (lam_3 ~ 2e-6).

    Pinned at seed 0 deliberately: it is a *known-dying* seed, so this is a
    regression guard on the worst case rather than a coin flip.  Do not "fix"
    it by choosing a luckier seed -- that is precisely the mistake that made
    #869's diagnosis narrower than its title.
    """
    _A, lam_h, _lam_v = _run(D=3, steps=40)

    arr = np.sort(np.asarray(lam_h))[::-1]
    arr = arr / arr[0]
    assert arr[2] > 1e-3, (
        f"D=3 bond spectrum {arr} has a negligible third direction -- the "
        f"state is effectively D=2 (#667/#878)"
    )
