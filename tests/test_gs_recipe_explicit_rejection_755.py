"""The fused explicit-AD path rejects a recipe it cannot honour (#755).

``ctm_energy_explicit`` takes no ``recipe`` and always runs the 2x2 step.
``make_ctm_energy_fn`` used to accept ``gs_recipe="1x1"`` on that branch and
thread it nowhere, so the run produced the ``2x2`` answer under a ``"1x1"``
label -- measured agreeing with ``2x2``+explicit to all 12 digits, with no
diagnostic.  The cost of that is not a wrong number; it is a *correctly
labelled config attached to the wrong experiment*, which survives into notes,
plots and papers.

The repo had already ruled on this one branch away: the 2-site split path
raises rather than accept-then-substitute, on the stated grounds that it
"would mislabel the experiment".  These tests pin the same rule for the fused
explicit path, and -- importantly -- pin that it stops there.  The *split*
explicit path does implement ``"1x1"`` and must keep accepting it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
from tenax.algorithms.ipeps_config import CTMConfig


def _energy_fn(*, recipe, fuse_virtual_legs=True, use_explicit=True):
    ctm_cfg = CTMConfig(chi=4, max_iter=4, fuse_virtual_legs=fuse_virtual_legs)
    return make_ctm_energy_fn(
        neighbors={(0, 0): {}},
        gate=None,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=use_explicit,
        explicit_warmup=1,
        explicit_steps=1,
        recipe=recipe,
    )


@pytest.mark.core
def test_fused_explicit_rejects_1x1():
    """The defect: accepted, ignored, and silently run as 2x2."""
    fn = _energy_fn(recipe="1x1")
    with pytest.raises(NotImplementedError, match="only supports gs_recipe='2x2'"):
        fn({(0, 0): None})


@pytest.mark.core
def test_the_rejection_points_at_2x2_not_at_implicit_ad():
    """A raise that names a broken alternative is worse than one that names none.

    The first version of this message said "Use gs_implicit_ad=True for the
    '1x1' recipe".  That is not a consistent way to get a 1x1 run:
    ``ctm_converge_kwargs`` does not forward ``recipe``, so the implicit
    path's line-search and final-evaluation forwards fall back to
    ``python_loop_ctm_converge``'s ``"2x2"`` default while only the loss sees
    ``"1x1"`` (#938).  It would have sent callers from one silent
    substitution into a subtler one.

    The honest recommendation is ``2x2``, because per #911 the ``1x1`` recipe
    reaches no fixed point for any state with ``D > 1`` under any projector
    method — so it is not a result the caller wants under *any* AD mode.
    """
    fn = _energy_fn(recipe="1x1")
    with pytest.raises(NotImplementedError) as exc:
        fn({(0, 0): None})
    msg = str(exc.value)

    assert "gs_recipe='2x2'" in msg, "the message must name the recipe to migrate to"
    assert "#911" in msg, "and say why 1x1 is not worth rescuing"
    # The split path is the one place 1x1 is genuinely wired end to end.
    assert "fuse_virtual_legs=False" in msg
    # Implicit AD may be mentioned, but only as a caveat carrying #938 —
    # never as the bare recommendation it used to be.
    if "gs_implicit_ad=True" in msg:
        assert "#938" in msg, (
            "the message names gs_implicit_ad=True without flagging that it "
            "threads recipe into the loss only (#938)"
        )


@pytest.mark.core
def test_fused_explicit_still_accepts_2x2():
    """The negative half: the guard must not reject the supported recipe.

    Constructing the closure and getting *past* the recipe check is the
    assertion; the call then fails downstream on the deliberately-empty
    neighbour map, which is a different error than the one under test.
    """
    fn = _energy_fn(recipe="2x2")
    with pytest.raises(Exception) as exc:  # noqa: B017 - any *other* failure
        fn({(0, 0): None})
    assert "only supports gs_recipe='2x2'" not in str(exc.value)


@pytest.mark.core
def test_the_rejection_is_scoped_to_the_fused_path():
    """``fuse_virtual_legs=False`` + ``"1x1"`` is supported and must stay so.

    ``ctm_energy_split_explicit`` does take ``recipe`` and does implement the
    single-site 1x1 forward, so a blanket raise in ``make_ctm_energy_fn`` --
    which is what #755 literally suggested -- would have broken a working
    combination (``test_split_ctm_fuse_flag.py`` exercises it).  This pins the
    scope, not just the behaviour.
    """
    fn = _energy_fn(recipe="1x1", fuse_virtual_legs=False)
    with pytest.raises(Exception) as exc:  # noqa: B017
        fn({(0, 0): None})
    assert "only supports gs_recipe='2x2'" not in str(exc.value), (
        "the fused-path guard leaked onto the split path, which implements 1x1"
    )


@pytest.mark.core
def test_the_implicit_branch_is_untouched():
    """Implicit AD threads ``"1x1"`` properly and must not be rejected."""
    fn = _energy_fn(recipe="1x1", use_explicit=False)
    with pytest.raises(Exception) as exc:  # noqa: B017
        fn({(0, 0): None})
    assert "only supports gs_recipe='2x2'" not in str(exc.value)
