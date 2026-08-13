"""#834: block-sparse ``contract()`` silently disagreeing with the dense one.

``_contract_symmetric`` pairs blocks by **equal charge value**; dense einsum
pairs **position p with position p**.  Where those differ the contractor
discarded the mismatched products with a bare ``continue`` -- no error, no
warning.  Measured before this change: 22 of 64 flow/charge configurations
disagreed by 4.2e-01 to 1.5e+00 relative, with ``|sym|/|den|`` from 0.458 to
1.514.  A result *larger* than the true one is not discarded weight; blocks were
being mis-paired.

Two mechanisms produce a disagreement, and they need different handling:

1. **mis-pairing** -- the value-pairing and the position-pairing select
   different slots.  Decidable from leg charges and block keys.
2. **discarded products** -- a compatible block product whose output key falls
   outside the output legs' valid set.  Whether that changes the answer depends
   on the product's *value*, which is traced.

## Why this is a diagnostic and not a default-on guard

Because a structural check gets the default CTM path exactly backwards.
Scoring every ``_contract_symmetric`` call of a D=2, chi=8 charged U(1)-Sz sweep
against the densified contraction of the same operands:

===========================================  =====  ==============
call site                                    calls  max rel. gap
===========================================  =====  ==============
*refused* by the leg-pairing check
  ``_apply_proj_unfused``                       40  1.7e-16
  ``_build_enlarged_corner`` (4 frames)         82  0.0
  ``_ctm_tensor_absorb_*_2plaq`` (4)            16  0.0
*allowed* by it
  ``_apply_proj_unfused``                       56  **8.3e-01**
===========================================  =====  ==============

Every refusal is a false alarm, and the one genuinely wrong site is allowed.
Turning the check on by default would break the default path *and* still miss
the defect on it.  So both checks are armed by ``TENAX_STRICT_CONTRACT=1``,
under which they are sound and complete on the grid below, and
``test_strict_mode_reports_real_weight_loss_in_the_ctm_sweep`` pins the
production defect that remains.

## Why not ``TensorIndex.is_dual_of``

That is this tree's *other* duality convention (opposite flow + **negated**
charges).  Negation preserves the charge set but permutes the position->charge
map, so it blesses contractions that mis-pair: measured 28 of 256 wrong.
``test_is_dual_of_would_be_unsound_as_a_guard`` pins it, so nobody "simplifies"
the check into the predicate that looks like it means this.
"""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

from tenax.contraction import contractor
from tenax.contraction.contractor import contract, contract_with_subscripts
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import (
    FermionicU1,
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor

FLOWS = (FlowDirection.OUT, FlowDirection.IN)
U1 = U1Symmetry()
STRICT = "TENAX_STRICT_CONTRACT"

# (symmetry, contracted-leg charges, free_a charges, free_b charges)
CASES = [
    (U1, [0, 1], [0, 1], [0, 1]),
    (U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1]),
    (U1, [-1, 0, 1], [0, 1], [0, 1]),
    (U1, [0, 1, 1, 2], [0, 1], [0, 1]),
    (U1, [0, 1, 1, 2], [0, 1, 2], [-1, 0, 1]),
    (U1, [0, 1, 2], [0, 1, 2], [-1, 0, 1]),
    (U1, [0, 0, 0], [0, 0], [0, 0]),  # control: trivial charges
    (U1, [-2, -1, 0, 1, 2], [-1, 0, 1], [-1, 0, 1]),
    (ZnSymmetry(2), [0, 1], [0, 1], [0, 1]),
    (ZnSymmetry(3), [0, 1, 2], [0, 1, 2], [0, 1, 2]),
    (ZnSymmetry(4), [0, 1, 2, 3], [0, 1, 2], [0, 1, 3]),
    (FermionParity(), [0, 1], [0, 1], [0, 1]),
    (FermionParity(), [0, 1, 1], [0, 1], [0, 1, 1]),
    (FermionicU1(), [0, 1], [0, 1], [0, 1]),
    (FermionicU1(), [-1, 0, 1], [0, 1, 2], [-1, 0, 1]),
    # Bit-packed charges: the group inverse is not integer negation here, so a
    # predicate that reached for `-q` instead of `sym.dual` would go wrong.
    (
        ProductSymmetry(FermionParity(), U1Symmetry()),
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
    ),
]

COMBOS = list(itertools.product(FLOWS, FLOWS, (+1, -1)))


def _pair(sym, ck, free_a, free_b, flow_a, flow_b, sign_b, seed=0):
    """Build ``A(i,k)`` and ``B(k,j)`` with a controllable pairing on ``k``.

    ``sign_b < 0`` builds B's shared leg from ``sym.dual(ck)`` -- the
    ``is_dual_of`` convention -- rather than from ``ck`` itself.
    """
    ck = np.asarray(ck, np.int32)
    cb_k = sym.dual(ck) if sign_b < 0 else ck.copy()
    idx_a = (
        TensorIndex.from_charges(
            sym, np.asarray(free_a, np.int32), FlowDirection.OUT, label="i"
        ),
        TensorIndex.from_charges(sym, ck.copy(), flow_a, label="k"),
    )
    idx_b = (
        TensorIndex.from_charges(sym, np.asarray(cb_k, np.int32), flow_b, label="k"),
        TensorIndex.from_charges(
            sym, np.asarray(free_b, np.int32), FlowDirection.IN, label="j"
        ),
    )
    A = SymmetricTensor.random_normal(idx_a, jax.random.PRNGKey(seed))
    B = SymmetricTensor.random_normal(idx_b, jax.random.PRNGKey(seed + 1))
    return A, B


def _dense_reference(A, B):
    """Ground truth: the same contraction with both operands densified."""
    return np.asarray(
        contract(
            DenseTensor(A.todense(), A.indices), DenseTensor(B.todense(), B.indices)
        ).todense()
    )


def _informative(A, B):
    """Skip configurations that carry no information either way."""
    if not A.blocks or not B.blocks:
        return False
    return np.linalg.norm(_dense_reference(A, B)) >= 1e-30


# --------------------------------------------------------------- the property
@pytest.mark.parametrize("sym,ck,free_a,free_b", CASES)
@pytest.mark.parametrize("flow_a,flow_b,sign_b", COMBOS)
def test_agrees_with_dense_or_refuses_under_strict(
    monkeypatch, sym, ck, free_a, free_b, flow_a, flow_b, sign_b
):
    """Under ``TENAX_STRICT_CONTRACT=1``: agree with dense, or refuse.

    The third option -- returning a different tensor with no error and no
    warning -- is the defect, and this forbids it.  No predicate appears in the
    assertion: dense is the reference, so the test cannot pass by agreeing with
    the checks' own reasoning.
    """
    monkeypatch.setenv(STRICT, "1")
    A, B = _pair(sym, ck, free_a, free_b, flow_a, flow_b, sign_b)
    if not _informative(A, B):
        pytest.skip("no blocks, or an identically-zero dense reference")

    try:
        result = np.asarray(contract(A, B).todense())
    except ValueError as exc:
        assert "#834" in str(exc), f"refused, but without the diagnostic: {exc}"
        return

    dense = _dense_reference(A, B)
    rel = np.linalg.norm(result - dense) / np.linalg.norm(dense)
    assert rel < 1e-12, (
        f"block-sparse and dense disagree by {rel:.3e} and the contraction was "
        f"allowed (flows {flow_a.name}/{flow_b.name}, sign_b={sign_b:+d}, "
        f"charges={ck}, symmetry={type(sym).__name__})"
    )


def _scan(monkeypatch):
    """Score every grid configuration four ways.

    Yields ``(label, default_refuses, pairing_refuses, strict_refuses,
    disagrees, is_dual_of, structural_ok)``.  ``disagrees`` is measured with
    strict off, so it records what the library actually returns rather than the
    checks' opinion of it.
    """
    for sym, ck, free_a, free_b in CASES:
        for flow_a, flow_b, sign_b in COMBOS:
            for seed in (0, 7):
                A, B = _pair(sym, ck, free_a, free_b, flow_a, flow_b, sign_b, seed)
                if not _informative(A, B):
                    continue

                monkeypatch.setenv(STRICT, "0")
                try:
                    result = np.asarray(contract(A, B).todense())
                    default_refuses = False
                except ValueError:
                    result, default_refuses = None, True

                monkeypatch.setenv(STRICT, "1")
                try:
                    contractor._validate_contracted_legs([A, B], "ik,kj->ij")
                    pairing_refuses = False
                except ValueError:
                    pairing_refuses = True
                try:
                    contract(A, B)
                    strict_refuses = False
                except ValueError:
                    strict_refuses = True
                monkeypatch.setenv(STRICT, "0")

                dense = _dense_reference(A, B)
                if result is None:
                    disagrees = True
                else:
                    denom = max(np.linalg.norm(dense), np.linalg.norm(result))
                    disagrees = np.linalg.norm(result - dense) / denom >= 1e-12

                ka, kb = A.indices[1], B.indices[0]
                ca, cb = np.asarray(ka.charges), np.asarray(kb.charges)
                yield (
                    f"{type(sym).__name__} k={ck} {flow_a.name}/{flow_b.name} "
                    f"sign={sign_b:+d} seed={seed}",
                    default_refuses,
                    pairing_refuses,
                    strict_refuses,
                    disagrees,
                    ka.is_dual_of(kb),
                    ka.flow != kb.flow and np.array_equal(ca, cb),
                )


def test_strict_mode_fires_exactly_when_the_answer_would_be_wrong(monkeypatch):
    """Sound *and* complete under strict mode: refuses <=> the answer differs.

    ``->`` soundness: nothing wrong is admitted.  ``<-`` completeness: nothing
    correct is refused.  Completeness holds *on this grid* because
    ``random_normal`` fills every allowed block, so a discarded product is always
    non-zero here.  It does not hold in production -- the CTM initial
    environment discards ~2000 products per sweep whose value is exactly zero --
    which is why strict mode is opt-in rather than the default.
    """
    unsound, incomplete, n = [], [], 0
    for label, _default, _pairing, strict, disagrees, _dual, _struct in _scan(
        monkeypatch
    ):
        n += 1
        if disagrees and not strict:
            unsound.append(label)
        if strict and not disagrees:
            incomplete.append(label)

    assert n >= 100, f"grid collapsed to {n} configurations; it is not testing much"
    assert not unsound, (
        f"strict mode admits {len(unsound)} wrong contractions: {unsound[:5]}"
    )
    assert not incomplete, (
        f"strict mode refuses {len(incomplete)} contractions that are exact: "
        f"{incomplete[:5]}"
    )


def test_the_default_mode_refuses_nothing(monkeypatch):
    """Off by default means off: no configuration changes behaviour.

    Deliberate, and measured -- see the module docstring. A structural check
    turned on by default refuses only correct work on the default CTM path.
    """
    refused = [
        label
        for label, default, _pairing, _strict, _dis, _dual, _struct in _scan(
            monkeypatch
        )
        if default
    ]
    assert not refused, f"default mode refused {len(refused)}: {refused[:5]}"


def test_the_leg_pairing_check_alone_would_miss_the_discard_class(monkeypatch):
    """The two checks are not redundant: each catches cases the other does not.

    If this ever comes back empty, one of them has become dead code -- but find
    out which before deleting it, because the likelier cause is the grid losing
    a configuration class than the library gaining a unified test.
    """
    pairing_misses = [
        label
        for label, _default, pairing, strict, disagrees, _dual, _struct in _scan(
            monkeypatch
        )
        if disagrees and strict and not pairing
    ]
    assert pairing_misses, (
        "every wrong configuration is now caught by the leg-pairing check alone, "
        "so the discard check is pinning nothing on this grid"
    )


def test_the_leg_pairing_check_never_refuses_an_exact_contraction(monkeypatch):
    """Soundness of the half that reads only structure.

    It is *not* complete -- that is the discard class above -- but a check that
    refused correct work would be worse than none, and this one is why the
    predicate consults populated blocks instead of leg metadata alone.
    """
    false_alarms = [
        label
        for label, _default, pairing, _strict, disagrees, _dual, _struct in _scan(
            monkeypatch
        )
        if pairing and not disagrees
    ]
    assert not false_alarms, (
        f"leg-pairing check refuses {len(false_alarms)} exact contractions: "
        f"{false_alarms[:5]}"
    )


def test_the_grid_exercises_both_outcomes(monkeypatch):
    """Anti-vacuity: an equivalence is trivially true on an empty side."""
    refused = exact = 0
    for _label, _default, _pairing, strict, _dis, _dual, _struct in _scan(monkeypatch):
        refused += strict
        exact += not strict
    assert refused >= 10 and exact >= 10, (
        f"grid is one-sided ({refused} refused, {exact} allowed); the soundness "
        f"or completeness half of the equivalence proves nothing"
    )


def test_is_dual_of_would_be_unsound_as_a_guard(monkeypatch):
    """Pin why the check is not ``is_dual_of``, which is what it looks like.

    ``is_dual_of`` means opposite flow + *negated* charges.  Negation preserves
    the charge set but permutes the position->charge map, so the two pairings
    diverge and it admits contractions that are wrong.
    """
    admitted_wrong = [
        label
        for label, _default, _pairing, _strict, disagrees, dual, _struct in _scan(
            monkeypatch
        )
        if dual and disagrees
    ]
    assert admitted_wrong, (
        "is_dual_of no longer admits a wrong contraction on this grid, so this "
        "test no longer pins anything -- re-derive before deleting it"
    )


def test_a_purely_structural_check_would_refuse_correct_contractions(monkeypatch):
    """Pin why the predicate consults the blocks and not just the leg metadata.

    The structural condition (opposite flows + element-wise equal charges) is
    sound but over-refuses, and what it over-refuses is not exotic -- the CTM
    initial environment is one.  A future simplification back to leg metadata
    alone would refuse work that is exact.
    """
    over_refused = [
        label
        for label, _default, _pairing, _strict, disagrees, _dual, struct in _scan(
            monkeypatch
        )
        if not struct and not disagrees
    ]
    assert over_refused, (
        "the structural predicate no longer over-refuses on this grid; if that "
        "is genuinely true the check could be simplified, but measure first"
    )


# ------------------------------------------------------- specific known cases
def test_the_issue_reproduction_is_refused(monkeypatch):
    """The headline configuration from #834: OUT/IN with negated charges."""
    monkeypatch.setenv(STRICT, "1")
    A, B = _pair(U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1], *FLOWS, -1)
    with pytest.raises(ValueError, match="#834"):
        contract(A, B)


def test_the_dense_answer_is_not_representable_so_refusing_is_correct(monkeypatch):
    """Refusing is the only available answer, not the conservative one.

    ``A.k`` OUT ``[-1,0,1]`` against ``B.k`` IN ``[1,0,-1]``: dense pairs
    position 0 of each -- charge -1 against charge +1 -- and puts weight on
    output block ``(i=1, j=-1)``, whose charge under ``i`` OUT / ``j`` IN is
    ``-(1) + (-1) = -2``.  That violates the output legs' own conservation law,
    so no ``SymmetricTensor`` over those indices can represent the dense result.

    Pinned because "make block-sparse agree with dense" is the natural reading
    of #834 and it is impossible, not merely unimplemented.
    """
    A, B = _pair(
        U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1], FlowDirection.OUT, FlowDirection.IN, -1
    )
    ka, kb = A.indices[1], B.indices[0]
    assert ka.is_dual_of(kb), "fixture no longer exercises the dual convention"

    dense = _dense_reference(A, B)
    out_i, out_j = A.indices[0], B.indices[1]
    ci, cj = np.asarray(out_i.charges), np.asarray(out_j.charges)
    offending = [
        (int(ci[p]), int(cj[q]))
        for p in range(dense.shape[0])
        for q in range(dense.shape[1])
        if abs(dense[p, q]) > 1e-12
        and int(out_i.flow) * int(ci[p]) + int(out_j.flow) * int(cj[q]) != 0
    ]
    assert offending, (
        "fixture no longer produces a non-conserving dense block, so it no "
        "longer demonstrates why refusing is correct"
    )
    monkeypatch.setenv(STRICT, "1")
    with pytest.raises(ValueError, match="#834"):
        contract(A, B)


def test_flip_flow_is_the_accepted_convention(monkeypatch):
    """``bar()``/``flip_flow()`` pairs contract even under strict, and match dense."""
    monkeypatch.setenv(STRICT, "1")
    shared = TensorIndex.from_charges(
        U1, np.array([-1, 0, 1], np.int32), FlowDirection.IN, label="k"
    )
    idx_a = (
        TensorIndex.from_charges(
            U1, np.array([0, 1, 2], np.int32), FlowDirection.OUT, label="i"
        ),
        shared.flip_flow(),
    )
    idx_b = (
        shared,
        TensorIndex.from_charges(
            U1, np.array([-1, 0, 1], np.int32), FlowDirection.IN, label="j"
        ),
    )
    A = SymmetricTensor.random_normal(idx_a, jax.random.PRNGKey(0))
    B = SymmetricTensor.random_normal(idx_b, jax.random.PRNGKey(1))
    assert len(A.blocks) > 1 and len(B.blocks) > 1, "degenerate fixture"

    result = np.asarray(contract(A, B).todense())
    assert np.allclose(result, _dense_reference(A, B), atol=1e-13)


def test_trivial_charges_are_unaffected(monkeypatch):
    """The control: at all-zero charges every flow combination is exact.

    Negation is the identity there, so the position->charge map cannot be
    permuted.  This is why #834 stayed hidden.
    """
    monkeypatch.setenv(STRICT, "1")
    for flow_a, flow_b in itertools.product(FLOWS, FLOWS):
        A, B = _pair(U1, [0, 0, 0], [0, 0], [0, 0], flow_a, flow_b, +1)
        result = np.asarray(contract(A, B).todense())
        assert np.allclose(result, _dense_reference(A, B), atol=1e-13)


def test_same_flow_is_allowed_when_only_the_identity_sector_is_populated(monkeypatch):
    """The escape that keeps the predicate block-aware rather than structural.

    ``_STD_EDGE_SPECS`` builds the CTM initial environment with same-flow chi
    bonds.  It is exact because the rank-1 corners carry weight only in the
    sector that would otherwise be discarded.  A predicate reading leg metadata
    alone cannot see that.
    """
    monkeypatch.setenv(STRICT, "1")
    charges = np.array([0, 1, -1], np.int32)
    idx_a = (
        TensorIndex.from_charges(U1, np.array([0], np.int32), FlowDirection.OUT, "i"),
        TensorIndex.from_charges(U1, charges, FlowDirection.IN, "k"),
    )
    idx_b = (
        TensorIndex.from_charges(U1, charges.copy(), FlowDirection.IN, "k"),
        TensorIndex.from_charges(U1, np.array([0], np.int32), FlowDirection.IN, "j"),
    )
    A = SymmetricTensor({(0, 0): jax.numpy.ones((1, 1))}, idx_a)
    B = SymmetricTensor({(0, 0): jax.numpy.ones((1, 1))}, idx_b)
    assert A.indices[1].flow == B.indices[0].flow, "fixture must be same-flow"

    result = np.asarray(contract(A, B).todense())
    assert np.allclose(result, _dense_reference(A, B), atol=1e-13)


# ------------------------------------------------------------- the diagnostic
def test_the_refusal_names_both_legs_and_the_issue(monkeypatch):
    monkeypatch.setenv(STRICT, "1")
    A, B = _pair(U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1], *FLOWS, -1)
    with pytest.raises(ValueError) as exc:
        contract(A, B)
    message = str(exc.value)
    assert "'k'" in message, f"does not name the offending legs: {message}"
    assert "#834" in message
    assert "flip_flow" in message or "bar" in message, (
        f"does not say what to do instead: {message}"
    )


def test_contract_with_subscripts_is_checked_too(monkeypatch):
    """The other public entry point into ``_contract_symmetric``."""
    monkeypatch.setenv(STRICT, "1")
    A, B = _pair(U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1], *FLOWS, -1)
    out = (A.indices[0], B.indices[1])
    with pytest.raises(ValueError, match="#834"):
        contract_with_subscripts([A, B], "ik,kj->ij", out)


def test_the_check_survives_jit(monkeypatch):
    """Block keys are static metadata, so the check still fires under tracing."""
    monkeypatch.setenv(STRICT, "1")
    A, B = _pair(U1, [-1, 0, 1], [0, 1, 2], [-1, 0, 1], *FLOWS, -1)
    with pytest.raises(ValueError, match="#834"):
        jax.jit(lambda a, b: contract(a, b).todense())(A, B)


def test_a_trace_within_one_tensor_is_not_treated_as_a_pairing(monkeypatch):
    """A repeated label on one tensor is a trace, not two legs meeting.

    Both legs then belong to the same operand, so there is no pairing between
    representations to disagree about.
    """
    monkeypatch.setenv(STRICT, "1")
    idx = (
        TensorIndex.from_charges(
            U1, np.array([-1, 0, 1], np.int32), FlowDirection.OUT, "k"
        ),
        TensorIndex.from_charges(
            U1, np.array([-1, 0, 1], np.int32), FlowDirection.OUT, "k2"
        ),
    )
    A = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(0))
    contractor._validate_contracted_legs([A], "kk->")


# ------------------------------------------- what the default mode must not do
def _charged_u1_sweep(max_iter: int = 3):
    from tenax.algorithms._ctm_tensor import ctm_tensor_2site
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    return ctm_tensor_2site(A, B, chi=8, max_iter=max_iter, conv_tol=1e-8)


def test_the_default_ctm_path_still_contracts():
    """A charged U(1)-Sz sweep must not be refused.

    This is the constraint that shaped the whole change.  The default path
    contracts same-flow chi bonds carrying non-cancelling charges and discards
    ~2000 products per sweep, and is *exact* anyway -- the discarded products are
    all zero.  Both checks refuse it, which is why both are opt-in.
    """
    _charged_u1_sweep()


def test_strict_mode_reports_real_weight_loss_in_the_ctm_sweep(monkeypatch):
    """The rest of #834 is live on the default CTM path -- pinned, not fixed.

    Measured with the checks disarmed, scoring every call against the densified
    contraction of the same operands: ``_apply_proj_unfused`` is **8.3e-01
    relative wrong** on 56 of 96 calls, and discards 192 non-zero products (max
    ``|product|`` 8.15).  Every *other* refused site in the sweep is exact to
    1.7e-16, so this is not the same false alarm as the enlarged corner.

    It is filed separately because the mechanism is target inference over
    mixed-charge operands (``_parse_contraction_prelude`` credits a tensor's
    target only when all of its blocks agree), not the leg pairing above.

    When a fix lands this test starts failing. That is the intended signal:
    re-measure and convert it to the positive form rather than deleting it.
    """
    monkeypatch.setenv(STRICT, "1")
    with pytest.raises(ValueError, match="#834"):
        _charged_u1_sweep()


# The accelerated execution paths, each of which returns from
# ``_contract_symmetric`` before the per-block loop.  ``_validate_contracted_legs``
# runs first and so covers all of them; the *discard* check does not, which is
# what these pin.
_ACCEL_FLAGS = (
    "TENAX_BATCH_BLOCKSPARSE",
    "TENAX_STACK_BLOCKSPARSE",
    "TENAX_USE_CUTENSOR_BLOCKSPARSE",
)


def _discard_class_pair(monkeypatch):
    """A grid configuration strict refuses via the *discard* check alone.

    Leg pairing must accept it, so that what the accelerated paths would skip is
    the only thing standing between the caller and a silent disagreement.
    """
    for sym, ck, free_a, free_b in CASES:
        for flow_a, flow_b, sign_b in COMBOS:
            A, B = _pair(sym, ck, free_a, free_b, flow_a, flow_b, sign_b, 0)
            if not _informative(A, B):
                continue
            monkeypatch.setenv(STRICT, "1")
            try:
                contractor._validate_contracted_legs([A, B], "ik,kj->ij")
                pairing_refuses = False
            except ValueError:
                pairing_refuses = True
            try:
                contract(A, B)
                strict_refuses = False
            except ValueError:
                strict_refuses = True
            monkeypatch.setenv(STRICT, "0")
            if strict_refuses and not pairing_refuses:
                return A, B
    return None, None


@pytest.mark.parametrize("flag", _ACCEL_FLAGS)
def test_strict_mode_is_not_bypassed_by_an_accelerated_backend(monkeypatch, flag):
    """An armed audit must be complete on every execution path, or it lies.

    ``_contract_symmetric`` returns early for the cuTENSOR, stacked and batched
    backends, all of which drop out-of-set output keys with the same bare
    ``continue`` as the per-block loop.  Before this was fixed,
    ``TENAX_STRICT_CONTRACT=1`` combined with ``TENAX_BATCH_BLOCKSPARSE=1``
    returned a silently truncated result and raised nothing.

    That is worse than not having the flag: the whole contract of a diagnostic
    is that "no raise" means "no disagreement", and a partial audit reports
    clean on exactly the configurations it did not inspect.
    """
    A, B = _discard_class_pair(monkeypatch)
    assert A is not None, (
        "the grid no longer contains a discard-class configuration, so this "
        "test is pinning nothing -- re-derive it before deleting"
    )
    monkeypatch.setenv(STRICT, "1")
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="#834"):
        contract(A, B)
