"""#878: the fermionic simple update drove the state to exactly zero.

`fpeps()` returned a state whose every bond weight, site tensor and physical
tensor were **exactly 0.0**, by step 10 at D=2 with a 200-step default.  Root
cause is #667 in a module #844 never touched: ``sigma`` was stored as the new
lambda *and* ``sqrt(sigma)`` was absorbed into ``Gamma``, so the next step's
lambda-absorption scaled that bond by lambda again and it carried
``lambda**1.5``.  #667's own title says "fermionic -> zero norm"; only the
bosonic half was fixed.  (That 1-site path, and the ``_absorb_lambdas`` helper
it ran on, are deleted -- ``fpeps()`` goes through the shared checkerboard
sweep.  These guards remain because the failure mode is a property of storing
lambda at all, which #882 is what actually removes.)

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
    _trotter_gate,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_simple_update import _simple_update_checkerboard_sweep

jax.config.update("jax_enable_x64", True)


def _run(D=2, steps=40, dt=0.05, V=0.0, seed=0, independent=False):
    """Returns ``(A, B, lambdas)`` -- the pair and all **four** bond spectra.

    ``independent=True`` drives the shared sweep directly instead of going
    through :func:`_fpeps_simple_update`, which ships the shared-spectrum
    default on purpose (the CDW does not need four free bonds -- see that
    function's docstring).

    Only the leg-to-bond mapping test asks for it, and it needs it in order to
    *be* a test.  With shared spectra ``lam.h_AB is lam.h_BA`` -- the same array
    object, not merely equal values -- so any reference built out of the four
    fields collapses to two, and a transposed or duplicated mapping satisfies it
    just as well as the correct one.
    """
    cfg = FPEPSConfig(D=D, t=1.0, V=V, dt=dt, num_imaginary_steps=steps)
    H = spinless_fermion_gate(cfg)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(seed))
    if not independent:
        return _fpeps_simple_update(A, H, max_D=D, dt=dt, steps=steps)
    return _simple_update_checkerboard_sweep(
        A, A, _trotter_gate(H, dt), D, 4 * steps, None, True
    )


@pytest.mark.parametrize("steps", [10, 40])
def test_the_bond_spectrum_does_not_collapse_to_zero(steps):
    """#878's symptom, asserted on the spectrum rather than the norm.

    ``> 0`` rather than ``isfinite``: the failure produced a finite, entirely
    zero answer, so every finiteness check passed on it.

    All four checkerboard bonds are checked, not one horizontal and one
    vertical: with two spectra the sweep's phases 2 and 3 overwrote the ones
    phases 0 and 1 produced, so a check on ``lam_h`` alone never saw the
    ``B.r<->A.l`` bond at all (#851).
    """
    A, B, lambdas = _run(steps=steps)

    for name, lam in zip(lambdas._fields, lambdas, strict=True):
        arr = np.asarray(lam)
        assert float(np.max(arr)) > 0.0, (
            f"{name} collapsed to all-zero after {steps} steps -- #878"
        )
        assert float(np.min(arr)) > 0.0, (
            f"{name} = {arr} after {steps} steps: a dead bond direction is how "
            f"the lambda**1.5 runaway starts (#878)"
        )
    for name, site in (("A", A), ("B", B)):
        assert float(site.norm()) > 0.0, f"site tensor {name} collapsed to zero"


def test_the_lambda_normalisation_is_relative_not_additive():
    """Pinned at the scale where the additive epsilon inverts, not at O(1).

    ``sigma / (max(sigma) + EPS)`` is a no-op while ``max(sigma) >> EPS``: at
    O(1) it returns ``1 - 1e-15``, which any sane tolerance accepts.  It only
    misbehaves once ``max(sigma)`` approaches ``EPS = 1e-15``, where it returns
    a spectrum whose maximum is well below 1 -- the next absorb shrinks the
    state further and the runaway closes in two steps (#748, #865).

    So this asserts on the *normalisation the fermionic path uses*, fed a
    spectrum at that scale.  Asserting ``max(lam) == 1`` on an ordinary run
    cannot distinguish the two implementations and passes on the defect.

    Imported from ``ipeps_simple_update``, which is where it lives and now the
    only place the fermionic sweep can reach it: the 1-site fermionic update
    that used to re-import it was deleted with the rest of the dead two-lambda
    path.
    """
    from tenax.algorithms.ipeps_simple_update import _normalise_lambda

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
    """A bond must pick up ``lambda``, not ``lambda**2``, on *its own* leg.

    Both ends of every bond contribute, so the physical tensor takes
    ``sqrt(lambda)`` on each leg.  Absorbing the full ``lambda`` squares every
    bond weight of the lattice.

    **Run with four independent spectra on purpose.**  The shipped default
    shares them, and under it ``lam.h_AB is lam.h_BA`` -- the same array object.
    The reference below would then collapse to ``{u: v, d: v, l: h, r: h}`` for
    both sites and could not tell the correct leg-to-bond map from a transposed
    or duplicated one: it would pass on the very defect it names.  With four
    distinct spectra it discriminates, which the guard at the top asserts rather
    than assumes.

    This is a claim about ``_to_physical_pair``'s mapping, not about the
    fermionic default.  ``test_su_851_four_bond_lambdas.py`` pins the same
    mapping from the bosonic side.
    """
    from tenax.algorithms.ipeps_simple_update import _to_physical_pair

    D = 2
    A, B, lam = _run(D=D, steps=20, independent=True)

    # The premise: four spectra that actually differ. Without this the rest of
    # the test is vacuous, so assert it rather than trusting the flag.
    for a, b in (("h_AB", "h_BA"), ("v_AB", "v_BA")):
        split = float(
            np.linalg.norm(np.asarray(getattr(lam, a)) - np.asarray(getattr(lam, b)))
        )
        assert split > 1e-6, (
            f"{a} and {b} agree to {split:.2e} -- the four spectra are not "
            f"distinguishable, so the leg-to-bond check below proves nothing"
        )

    A_phys, B_phys = _to_physical_pair(A, B, lam)

    for name, t in (("A", A_phys), ("B", B_phys)):
        assert float(t.norm()) > 0.0, f"physical tensor {name} collapsed to zero"

    # sqrt, not full lambda: scaling a leg by lam instead of sqrt(lam) changes
    # the tensor unless lam is all ones, so compare against the explicit build.
    # A and B are mirror images -- A.r and B.l are the *same* bond -- so the leg
    # to bond map is not the same for the two sites.
    from tenax.core._tensor_utils import scale_bond_axis

    for got_t, legs in (
        (A_phys, {"u": lam.v_BA, "d": lam.v_AB, "l": lam.h_BA, "r": lam.h_AB}),
        (B_phys, {"u": lam.v_AB, "d": lam.v_BA, "l": lam.h_AB, "r": lam.h_BA}),
    ):
        want = A if got_t is A_phys else B
        for leg, weight in legs.items():
            want = scale_bond_axis(want, leg, jnp.sqrt(weight))
        want = want * (1.0 / float(want.norm()))
        np.testing.assert_allclose(
            np.asarray(got_t.todense()),
            np.asarray(want.todense()),
            rtol=1e-12,
            atol=1e-14,
        )


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
    _A, _B, lam = _run(D=3, steps=40)

    arr = np.sort(np.asarray(lam.h_AB))[::-1]
    arr = arr / arr[0]
    assert arr[2] > 1e-3, (
        f"D=3 bond spectrum {arr} has a negligible third direction -- the "
        f"state is effectively D=2 (#667/#878)"
    )
