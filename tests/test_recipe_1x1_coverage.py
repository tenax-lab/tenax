"""Every public function that accepts ``recipe`` must warn on ``"1x1"`` (#911).

Three consecutive review rounds on the #911 deprecation each found entry points
that had been missed -- first the split forward pair, then
``ctm_energy_implicit``, then ``python_loop_ctm_converge`` and the split AD
family.  Each round the response was a longer hand-written list, and each time
the next round found more.  There are **24** functions taking a ``recipe``
parameter across 8 modules; a list maintained by hand is not going to converge.

So this file replaces the list with a rule, derived from source at test time:

    every *public* function that accepts ``recipe`` either calls
    ``_warn_recipe_1x1_deprecated`` in its own body, or is on an explicit
    allowlist that says why it does not need to.

The point is what happens to the *next* one.  A new public CTM entry point with
a ``recipe`` parameter fails this test the moment it is written, instead of
being found by a reviewer two rounds later -- or not at all.

Static, deliberately.  It cannot prove the warning actually fires; that is what
the per-entry-point ``pytest.warns`` tests in
``test_recipe_1x1_deprecation.py`` are for.  What it proves is *coverage*, which
is the thing that kept being wrong, and it proves it about code that no test
happens to call.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "tenax"
WARN_FN = "_warn_recipe_1x1_deprecated"

# Functions that take ``recipe`` but must NOT warn, each with the reason.
# Adding a name here is a deliberate act that a reviewer can see; leaving one
# out is a test failure.  "It is private" is not on its own sufficient -- a
# private function reachable only through a covered public one is exempt
# *because* of that reachability, which is what the reason has to state.
EXEMPT: dict[str, str] = {
    # --- per-sweep, called inside a convergence loop that already warned ---
    "_ctm_tensor_sweep_multisite": (
        "one sweep, not a run: called max_iter times by ctm_tensor and "
        "_ctm_tensor_multisite, both of which warn once at entry"
    ),
    "_split_ctm_sweep_multisite": (
        "one sweep; called by _split_ctm_multisite, which warns once at entry"
    ),
    "_split_step": (
        "one sweep; called by ctm_energy_split_explicit, which warns at entry"
    ),
    "_make_jit_ctm_step": (
        "builds a single jitted sweep; the recipe is closed over, and the "
        "callers (python_loop_ctm_converge, _python_loop_chi_ramp) warn"
    ),
    # --- private, reachable only through a public function that warns ---
    "_sigma_gauged_ctm_converge": (
        "inner convergence for ctm_energy_implicit, which warns at entry; "
        "warning here would fire once per chi-ramp stage"
    ),
    "_ctm_energy_implicit_dispatch": "internal dispatch under ctm_energy_implicit",
    "_make_implicit_vjp_fn": "backward-pass factory under ctm_energy_implicit",
    "_converge_split_gauge_fixed": (
        "inner convergence shared by ctm_energy_split_implicit and "
        "converge_split_env; both warn at entry"
    ),
    "_python_loop_chi_ramp": (
        "per-stage helper under python_loop_ctm_converge, which warns; "
        "warning here would fire once per ramp stage"
    ),
    "_ctm_tensor_multisite": (
        "private, but warns anyway -- it is the single point every multisite "
        "caller passes through.  Listed here only so the rule below does not "
        "have to special-case private-functions-that-do-warn"
    ),
    "_split_ctm_multisite": (
        "private, but warns anyway -- the single point ctm_split_tensor_2site "
        "and every other split multisite caller passes through"
    ),
    # --- public, but delegate to a covered function in the same call ---
    # These pass ``_deprecation_stacklevel=4`` so the warning the delegate
    # raises still names the *caller's* line rather than the delegating one --
    # which is what makes delegation acceptable here instead of a second warn.
    "ctm_tensor_2site": (
        "delegates to _ctm_tensor_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    "ctm_multisite": (
        "delegates to _ctm_tensor_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    "ctm_split_tensor_2site": (
        "delegates to _split_ctm_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    # --- takes the recipe but never runs a sweep ---
    "validate_split_ctm_config": (
        "validator: inspects the recipe and raises, never converges anything. "
        "Warning here would fire on configs that are about to be rejected"
    ),
    "make_ctm_energy_fn": (
        "factory: returns an energy_fn closing over the recipe.  The function "
        "it returns runs the CTM through an entry point that warns, so warning "
        "at construction would fire for callers who never evaluate it"
    ),
}


def _public_recipe_functions() -> dict[str, tuple[pathlib.Path, ast.FunctionDef]]:
    """Every function in ``src/tenax`` with a ``recipe`` parameter."""
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in node.args.args] + [
                a.arg for a in node.args.kwonlyargs
            ]
            if "recipe" in names:
                found[node.name] = (path, node)
    return found


def _calls_the_warning(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == WARN_FN)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == WARN_FN)
        )
        for n in ast.walk(node)
    )


def test_the_scan_finds_something():
    """Non-vacuity: if the AST walk silently found nothing, everything below passes."""
    found = _public_recipe_functions()
    assert len(found) >= 20, f"expected ~24 recipe-taking functions, found {len(found)}"
    assert "ctm_tensor" in found


def test_every_recipe_taking_function_warns_or_is_explicitly_exempt():
    """The rule. A new uncovered entry point fails here, not two rounds later."""
    uncovered = [
        f"{path.name}::{name}"
        for name, (path, node) in sorted(_public_recipe_functions().items())
        if name not in EXEMPT and not _calls_the_warning(node)
    ]
    assert not uncovered, (
        "these functions accept recipe='1x1' but neither warn nor appear in "
        "EXEMPT, so callers reach the non-converging recipe silently:\n  "
        + "\n  ".join(uncovered)
        + f"\n\nEither call {WARN_FN}() at entry, or add the name to EXEMPT in "
        "this file with the reason it does not need to (see #911)."
    )


def test_the_exempt_list_has_no_stale_entries():
    """An allowlist that outlives its subject silently weakens the rule above."""
    found = _public_recipe_functions()
    stale = sorted(set(EXEMPT) - set(found))
    assert not stale, (
        "EXEMPT names functions that no longer take a `recipe` parameter; "
        "remove them so the allowlist keeps meaning what it says:\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_exempt_entries_carry_a_reason(name):
    """'It is private' is not a reason; reachability through a warner is."""
    reason = EXEMPT[name]
    assert len(reason) > 30, f"{name}: give a real reason, got {reason!r}"
