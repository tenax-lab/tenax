"""What actually keeps the 2x2 enlarged corner contraction-correct (#834, #762).

``_build_enlarged_corner`` contracts environment legs that are **not** duals of
each other.  ``C1.c1_d`` meets ``T4.t4_d`` with both legs flowing IN, and each
edge's D^2 leg meets the double layer's matching face with both flows equal.
``_contract_symmetric`` pairs blocks by charge *value* and drops products that
land outside the output legs' conservation law, so on a same-flow bond it can
silently discard weight -- the #834 mechanism.

It does not discard any here, and the reason is worth pinning, because it is
not the reason one would guess:

* It is **not** that the contraction is convention-correct.  Feed the same call
  full-support random tensors carrying the *identical* indices and it is ~100%
  wrong (``test_the_agreement_is_a_property_of_the_swept_environment``).
* It is that the CTM sweep's renormalisation **dualizes the corners**, so from
  the first move onward every chi bond in the environment is a proper dual and
  no same-flow contraction happens at all
  (``test_the_swept_environment_has_no_same_flow_chi_bonds``).  Only the
  ``_STD_EDGE_SPECS`` *initial* environment carries same-flow chi bonds, and
  there the rank-1 corners have no weight in the sectors that would be dropped.

That is an invariant of the sweep, not of the type system, and nothing else
enforces it -- which is exactly why it is tested.  ``_ctm_tensor_c4v`` is the
same code path with that invariant broken: ``_c4v_to_full_env`` hands consumers
the *initial* convention rather than the swept one, its C1 x T4 bond drops 68%
of its weight, and that is #762.

If a future change makes the flow convention consistent up front, the negative
test below will start failing.  That failure is good news, not a regression:
delete it and keep the positive ones.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.core.tensor import DenseTensor, SymmetricTensor

# Chi bonds of the 8-tensor environment: (corner, corner leg, edge, edge leg).
_CHI_BONDS = (
    ("C1", "c1_r", "T1", "t1_l"),
    ("C1", "c1_d", "T4", "t4_d"),
    ("C2", "c2_l", "T1", "t1_r"),
    ("C2", "c2_d", "T2", "t2_u"),
    ("C3", "c3_u", "T2", "t2_d"),
    ("C3", "c3_l", "T3", "t3_l"),
    ("C4", "c4_u", "T3", "t3_r"),
    ("C4", "c4_r", "T4", "t4_u"),
)


def _densify(t):
    """Flow-insensitive twin of a SymmetricTensor: the ground-truth contraction."""
    return (
        DenseTensor(t.todense(), tuple(t.indices))
        if isinstance(t, SymmetricTensor)
        else t
    )


def _rel_gap(sym_out, den_out) -> float:
    """Relative difference between a symmetric result and its densified twin."""
    s = np.asarray(sym_out.todense())
    d = np.asarray(den_out.todense())
    # todense() leg order follows each tensor's own index order; align by label.
    d = np.transpose(d, [den_out.labels().index(lab) for lab in sym_out.labels()])
    return float(np.linalg.norm(s - d) / max(np.linalg.norm(d), 1e-30))


def _leg(env: CTMTensorEnv, tensor_name: str, label: str):
    return next(i for i in getattr(env, tensor_name).indices if i.label == label)


def _record_call_site_gaps(monkeypatch) -> list[tuple[str, float]]:
    """Patch ``_build_enlarged_corner`` to score every call against dense.

    Patches the name in ``_ctm_tensor_moves`` (where it is imported and called),
    not in its defining module, so the sweep actually sees the wrapper.
    """
    import tenax.algorithms._ctm_tensor_moves as moves
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner

    gaps: list[tuple[str, float]] = []

    def scored(C, T_h, T_v, a, *, position):
        out = _build_enlarged_corner(C, T_h, T_v, a, position=position)
        if isinstance(out, SymmetricTensor):
            reference = _build_enlarged_corner(
                _densify(C),
                _densify(T_h),
                _densify(T_v),
                _densify(a),
                position=position,
            )
            gaps.append((position, _rel_gap(out, reference)))
        return out

    monkeypatch.setattr(moves, "_build_enlarged_corner", scored)
    return gaps


def _run_charged_sweep(D: int, chi: int, max_iter: int):
    """A few CTM sweeps on a charged U(1)-Sz state (trivial charges prove nothing)."""
    from tenax.algorithms._ctm_tensor import ctm_tensor_2site
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    return ctm_tensor_2site(A, B, chi=chi, max_iter=max_iter, conv_tol=1e-8)


@pytest.mark.parametrize("D", [2])
def test_enlarged_corner_matches_dense_at_every_sweep_call_site(monkeypatch, D):
    """Every ``_build_enlarged_corner`` call in a charged sweep equals dense.

    The whole point of running a *sweep* rather than a single call is that the
    initial environment is rank-1 and cannot discriminate: its corners have no
    weight in the sectors a same-flow contraction drops, so it agrees for a
    reason that says nothing about the renormalised environment.
    """
    gaps = _record_call_site_gaps(monkeypatch)
    _run_charged_sweep(D=D, chi=4, max_iter=2)

    assert gaps, "no symmetric enlarged-corner calls were observed"
    worst = max(gaps, key=lambda g: g[1])
    assert worst[1] < 1e-12, (
        f"D={D}: enlarged corner diverged from the dense contraction at "
        f"{worst[0]} (rel={worst[1]:.3e}) over {len(gaps)} call sites. The "
        "symmetric CTM is dropping block weight on a non-dual bond -- see the "
        "module docstring and #834."
    )


@pytest.mark.slow
def test_enlarged_corner_matches_dense_at_every_sweep_call_site_d3(monkeypatch):
    """Same guard at D=3, where the Sz seam charge set stops being self-dual.

    D=2 masks charge-convention defects because the Sz seam is self-dual there
    (the #605 lesson); D=3 is the first bond dimension that can see them.
    """
    gaps = _record_call_site_gaps(monkeypatch)
    _run_charged_sweep(D=3, chi=8, max_iter=3)

    assert gaps, "no symmetric enlarged-corner calls were observed"
    worst = max(gaps, key=lambda g: g[1])
    assert worst[1] < 1e-12, (
        f"D=3: enlarged corner diverged from dense at {worst[0]} "
        f"(rel={worst[1]:.3e}) over {len(gaps)} call sites"
    )


@pytest.fixture(scope="module")
def swept_env_d2():
    """One charged D=2 sweep, shared: the sweep dominates this file's runtime."""
    return _run_charged_sweep(D=2, chi=4, max_iter=2)


def test_the_swept_environment_has_no_same_flow_chi_bonds(swept_env_d2):
    """The sweep repairs the flow convention the initial environment ships with.

    ``_STD_EDGE_SPECS``/``_CORNER_SPECS`` leave four of the eight chi bonds
    same-flow; the renormalisation dualizes the corners so the swept
    environment has none.  This is the invariant that makes the enlarged corner
    correct, so it is stated directly rather than left implicit in the test
    above.
    """
    envA, _envB = swept_env_d2

    offenders = []
    for corner, corner_leg, edge, edge_leg in _CHI_BONDS:
        a, b = _leg(envA, corner, corner_leg), _leg(envA, edge, edge_leg)
        if a.flow == b.flow:
            offenders.append(
                f"{corner}.{corner_leg} ~ {edge}.{edge_leg} (both {a.flow.name})"
            )
        elif not np.array_equal(np.asarray(a.charges), np.asarray(b.charges)):
            offenders.append(
                f"{corner}.{corner_leg} ~ {edge}.{edge_leg} (charges differ)"
            )

    assert not offenders, (
        "swept environment has non-dual chi bonds: "
        + "; ".join(offenders)
        + ". _contract_symmetric pairs blocks by charge value, so a consumer "
        "contracting these drops weight silently (#834, and #762 for what that "
        "costs in energy)."
    )


def test_the_agreement_is_a_property_of_the_swept_environment(monkeypatch):
    """Same indices, full support -> the same call is ~100% wrong.

    This is the discriminator.  Without it, the tests above read as "the
    symmetric contraction is convention-correct", which is false: swap in
    tensors that populate every conservation-allowed block of the *initial*
    (same-flow) convention and the enlarged corner loses most of its weight.

    If this ever starts failing, the flow convention was made consistent up
    front.  Delete this test -- do not loosen it.
    """
    from tenax.algorithms._ctm_tensor_init import (
        _build_double_layer_tensor,
        initialize_ctm_tensor_env,
    )
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env = initialize_ctm_tensor_env(A, chi=4)
    a = _build_double_layer_tensor(A)

    key = jax.random.PRNGKey(11)
    full_support = {
        name: SymmetricTensor.random_normal(
            getattr(env, name).indices, jax.random.fold_in(key, i)
        )
        for i, name in enumerate(("C1", "T1", "T4"))
    }
    C, T_h, T_v = full_support["C1"], full_support["T1"], full_support["T4"]

    gap = _rel_gap(
        _build_enlarged_corner(C, T_h, T_v, a, position="top_left"),
        _build_enlarged_corner(
            _densify(C), _densify(T_h), _densify(T_v), _densify(a), position="top_left"
        ),
    )
    assert gap > 1e-2, (
        f"expected the initial same-flow convention to lose weight on "
        f"full-support operands, got rel={gap:.3e}. If the flow convention was "
        "fixed, delete this test and tighten the docstring above it."
    )
