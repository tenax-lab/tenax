"""Normalising a complex number to a pure phase must not NaN its VJP (#789).

Four sites normalise a reference element to a unit phase like this::

    phases = jnp.where(jnp.abs(signs) > 0, signs / jnp.abs(signs), 1.0)

The ``jnp.where`` makes the **value** correct at ``signs = 0``.  It does not
make the **gradient** correct: ``jnp.where`` does not short-circuit a VJP, so
the unselected branch still evaluates ``0/0 = NaN`` and contributes
``0 * NaN = NaN``.

That distinction is the whole reason this survived, and it is what the tests
here are built around.  Measured on the bare idiom::

    value = [0.707+0.707j, 1+0j, 1+0j]      finite -> True
    grad  = [2.2e-16-2.2e-16j, nan+nanj, 0j] finite -> False

The values are *identical* between the broken and fixed forms, so any test
that checks outputs passes against the defect.  Only the gradient sees it, and
only in the zero column -- the gradient looks healthy everywhere else, so it
does not announce itself; it silently poisons one column of whatever consumes
it.

The fix is the fenced-argument idiom already used correctly elsewhere in the
tree (``_ctm_projector.py``'s ``S_rsqrt``, ``_normalise_rdm``): keep the
non-differentiable op from ever *seeing* the bad value, rather than selecting
against its result afterwards.
"""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ad_primitives import _fix_svd_signs, _unit_phase
from tenax.algorithms._ctm_root_implicit_asym import _pin_bond_gauge
from tenax.algorithms._ctm_root_implicit_symmetric import _pin_bond_gauge_sector
from tenax.algorithms._ctm_tensor_projector_2x2 import _gauge_fixed_svd


def _unfenced(z):
    """The original idiom, kept so the tests can show what changed."""
    return jnp.where(jnp.abs(z) > 0, z / jnp.abs(z), 1.0)


# ---------------------------------------------------------------------------
# The helper the four sites now share.
# ---------------------------------------------------------------------------


def test_the_value_is_identical_to_the_unfenced_form():
    """The fix must be a pure gradient change -- no physics may move.

    Bit-identical, not ``approx``: the fenced form performs the same division
    on the same operands, and anything looser would hide a real change.
    """
    z = jnp.array([1.0 + 1j, 3.0 - 4j, -2.0 + 0j, 0.0 + 0j, 1e-30 + 0j])

    assert jnp.array_equal(_unit_phase(z), _unfenced(z))


def test_the_gradient_is_finite_at_zero():
    """The defect itself, at the smallest scale that shows it."""
    z = jnp.array([1.0 + 1j, 0.0 + 0j, 2.0 + 0j])

    grad = jax.grad(lambda x: jnp.sum(jnp.abs(_unit_phase(x)) ** 2).real)(z)

    assert bool(jnp.isfinite(grad).all()), f"NaN in the VJP at zero: {grad}"


def test_the_unfenced_form_really_does_NaN():
    """Pins that the tests above are exercising a real defect, not a tautology.

    Without this the suite could pass against a no-op "fix": every other test
    here would still be green if ``_unit_phase`` were simply the old idiom and
    JAX had changed to short-circuit ``where``.  This asserts the failure mode
    is still present in the form we removed, so the guard keeps meaning
    something.
    """
    z = jnp.array([1.0 + 1j, 0.0 + 0j, 2.0 + 0j])

    grad = jax.grad(lambda x: jnp.sum(jnp.abs(_unfenced(x)) ** 2).real)(z)

    assert not bool(jnp.isfinite(grad).all()), (
        "the unfenced idiom no longer NaNs -- if JAX changed, this whole guard "
        "needs re-deriving rather than deleting"
    )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_a_wholly_zero_input_is_finite_in_both_value_and_gradient(dtype):
    """The degenerate extreme: nothing to take a phase from at all."""
    z = jnp.zeros(4, dtype=dtype)

    value = _unit_phase(z)
    grad = jax.grad(lambda x: jnp.sum(jnp.abs(_unit_phase(x)) ** 2).real)(z)

    assert bool(jnp.isfinite(value).all())
    assert bool(jnp.isfinite(grad).all())
    assert jnp.allclose(value, 1.0)


# ---------------------------------------------------------------------------
# Per site. A shared helper is only worth having if every caller uses it.
# ---------------------------------------------------------------------------


def test_fix_svd_signs_is_finite_on_a_zero_column():
    """``U`` arrives as an argument, so a zero column is an ordinary input.

    On the block-sparse path an empty or rank-deficient charge sector produces
    exactly this: a column whose largest-magnitude entry is 0.
    """
    U = jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]])
    s = jnp.ones(2)
    Vh = jnp.eye(2, dtype=U.dtype)

    grad = jax.grad(lambda u: jnp.sum(jnp.abs(_fix_svd_signs(u, s, Vh)[0]) ** 2).real)(
        U
    )

    assert bool(jnp.isfinite(grad).all()), f"NaN in the VJP: {grad.ravel()}"


def test_pin_bond_gauge_is_finite_when_the_warm_reference_vanishes():
    """``ref = sum(conj(prev_P) * P)`` is zero whenever the columns are orthogonal.

    That is not exotic on a near-degenerate retained subspace, which is the
    regime this function exists to stabilise in the first place.
    """
    chi = 2
    P_top = jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    prev = jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])  # orthogonal
    P_bot = jnp.eye(chi, dtype=P_top.dtype)
    U = jnp.eye(chi, dtype=P_top.dtype)
    Vh = jnp.eye(chi, dtype=P_top.dtype)

    ref = jnp.sum(jnp.conj(prev) * P_top, axis=0)
    assert jnp.allclose(ref, 0.0), "precondition: the warm reference must vanish"

    grad = jax.grad(
        lambda p: (
            jnp.sum(
                jnp.abs(_pin_bond_gauge(U, Vh, p, P_bot, chi, prev_P_top=prev)[2]) ** 2
            ).real
        )
    )(P_top)

    assert bool(jnp.isfinite(grad).all()), f"NaN in the VJP: {grad.ravel()}"


def test_pin_bond_gauge_sector_is_finite_when_the_warm_reference_vanishes():
    """Same defect, symmetric sector-wise copy."""
    k_q = 2
    P_left = jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    prev = jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
    P_right = jnp.eye(k_q, dtype=P_left.dtype)
    U = jnp.eye(k_q, dtype=P_left.dtype)
    Vh = jnp.eye(k_q, dtype=P_left.dtype)

    grad = jax.grad(
        lambda p: (
            jnp.sum(
                jnp.abs(
                    _pin_bond_gauge_sector(p, P_right, U, Vh, k_q, prev_P_left=prev)[0]
                )
                ** 2
            ).real
        )
    )(P_left)

    assert bool(jnp.isfinite(grad).all()), f"NaN in the VJP: {grad.ravel()}"


def test_the_svd_entry_point_cannot_itself_reach_a_zero_phase_reference():
    """Scope, stated honestly: site 4's fence is defensive, not reachable here.

    ``_gauge_fixed_svd`` takes a matrix and reads its phase reference from
    LAPACK's ``U``, whose columns are orthonormal -- so ``max|U|`` per column
    is 1 even for the all-zero matrix, and the zero branch is never selected
    through this entry point.  Fixing it is still right (defensive code should
    be correct, and it is one line of a shared helper), but the issue's claim
    that all four sites are reachable does not survive measurement.

    The all-zero matrix *does* produce a NaN gradient here -- from the SVD
    backward on a fully degenerate spectrum, which is the #406/#750 cluster
    and a different defect.  Asserted so the two are not conflated later.
    """
    for M in (jnp.zeros((3, 3)), jnp.eye(3) * 2.0):
        U, _s, _Vh = _gauge_fixed_svd(M)
        assert float(jnp.abs(U).max(axis=0).min()) > 0.5, (
            "a column of U had no dominant entry; the phase reference would be "
            "reachable after all and this test needs re-deriving"
        )

    # Non-degenerate spectrum: the phase fix is on the gradient path and clean.
    M = jnp.array([[3.0, 0.0], [0.0, 1.0]])
    grad = jax.grad(lambda m: jnp.sum(jnp.abs(_gauge_fixed_svd(m)[0]) ** 2).real)(M)
    assert bool(jnp.isfinite(grad).all())


# ---------------------------------------------------------------------------
# The fix has to stay applied everywhere.
# ---------------------------------------------------------------------------


def test_no_site_reintroduces_the_unfenced_idiom():
    """Four copies of one idiom is how this became four bugs.

    The same "fix landed on one of N copies" pattern as #828, #829 and #842.
    Asserting on the source keeps a new copy from being written by hand
    instead of calling the shared helper.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "tenax"

    def _divides_by_abs(node) -> bool:
        """True if ``node`` contains ``... / <something>.abs(...)``."""
        return any(
            isinstance(sub, ast.BinOp)
            and isinstance(sub.op, ast.Div)
            and isinstance(sub.right, ast.Call)
            and isinstance(sub.right.func, ast.Attribute)
            and sub.right.func.attr == "abs"
            for sub in ast.walk(node)
        )

    class Finder(ast.NodeVisitor):
        """Match the *unfenced* idiom, and only that.

        Parsed rather than grepped so the docstring in ``_unit_phase``, which
        quotes the broken form in order to explain it, is not a false positive
        -- and so a comment cannot hide a real one.

        The discriminator is **where the division sits**.  The correct fence
        selects the *operand*::

            safe = jnp.where(jnp.abs(w) > cutoff, w, 1.0)
            inv  = jnp.where(jnp.abs(w) > cutoff, 1.0 / safe, 0.0)

        and the tree has four of those already (``_ad_primitives`` QR floor,
        ``_ctm_c4v_root_implicit`` pseudo-inverse, ``_normalise_rdm``).  The
        bug selects the *result*, so the division by ``abs`` appears inside the
        true branch, where its VJP is still evaluated.  Keying on the guard
        alone would flag every correct fence in the tree.
        """

        def __init__(self):
            self.hits: list[int] = []

        def visit_Call(self, node):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "where"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Compare)
            ):
                cmp = node.args[0]
                if (
                    isinstance(cmp.left, ast.Call)
                    and isinstance(cmp.left.func, ast.Attribute)
                    and cmp.left.func.attr == "abs"
                    and len(cmp.ops) == 1
                    and isinstance(cmp.ops[0], ast.Gt)
                    and _divides_by_abs(node.args[1])
                ):
                    self.hits.append(node.lineno)
            self.generic_visit(node)

    offenders = []
    for path in sorted(root.rglob("*.py")):
        finder = Finder()
        finder.visit(ast.parse(path.read_text()))
        offenders += [f"{path.relative_to(root)}:{n}" for n in finder.hits]

    assert not offenders, (
        f"these select against a division by zero instead of fencing its "
        f"argument, so their VJP is NaN at zero: {offenders}. Use "
        f"tenax.algorithms._ad_primitives._unit_phase."
    )
