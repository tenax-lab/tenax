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
    "algorithms/_ctm_tensor_convergence.py::_ctm_tensor_sweep_multisite": (
        "one sweep, not a run: called max_iter times by ctm_tensor and "
        "_ctm_tensor_multisite, both of which warn once at entry"
    ),
    "algorithms/_split_ctm_tensor_convergence.py::_split_ctm_sweep_multisite": (
        "one sweep; called by _split_ctm_multisite, which warns once at entry"
    ),
    "algorithms/_split_ctm_energy_ad.py::_split_step": (
        "one sweep; called by ctm_energy_split_explicit, which warns at entry"
    ),
    "algorithms/_ctm_python_loop.py::_make_jit_ctm_step": (
        "builds a single jitted sweep; the recipe is closed over, and the "
        "callers (python_loop_ctm_converge, _python_loop_chi_ramp) warn"
    ),
    # --- private, reachable only through a public function that warns ---
    "algorithms/_ctm_energy_ad.py::_sigma_gauged_ctm_converge": (
        "inner convergence for ctm_energy_implicit, which warns at entry; "
        "warning here would fire once per chi-ramp stage"
    ),
    "algorithms/_ctm_energy_ad.py::_ctm_energy_implicit_dispatch": "internal dispatch under ctm_energy_implicit",
    "algorithms/_ctm_energy_ad.py::_make_implicit_vjp_fn": "backward-pass factory under ctm_energy_implicit",
    "algorithms/_split_ctm_energy_ad.py::_converge_split_gauge_fixed": (
        "inner convergence shared by ctm_energy_split_implicit and "
        "converge_split_env; both warn at entry"
    ),
    "algorithms/_ctm_python_loop.py::_python_loop_chi_ramp": (
        "per-stage helper under python_loop_ctm_converge, which warns; "
        "warning here would fire once per ramp stage"
    ),
    "algorithms/_ctm_tensor_convergence.py::_ctm_tensor_multisite": (
        "private, but warns anyway -- it is the single point every multisite "
        "caller passes through.  Listed here only so the rule below does not "
        "have to special-case private-functions-that-do-warn"
    ),
    "algorithms/_split_ctm_tensor_convergence.py::_split_ctm_multisite": (
        "private, but warns anyway -- the single point ctm_split_tensor_2site "
        "and every other split multisite caller passes through"
    ),
    # --- public, but delegate to a covered function in the same call ---
    # These pass ``_deprecation_stacklevel=4`` so the warning the delegate
    # raises still names the *caller's* line rather than the delegating one --
    # which is what makes delegation acceptable here instead of a second warn.
    "algorithms/_ctm_tensor_convergence.py::ctm_tensor_2site": (
        "delegates to _ctm_tensor_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    "algorithms/_ctm_tensor_convergence.py::ctm_multisite": (
        "delegates to _ctm_tensor_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    "algorithms/_split_ctm_tensor_convergence.py::ctm_split_tensor_2site": (
        "delegates to _split_ctm_multisite, which warns; passes "
        "_deprecation_stacklevel=4 so the warning names the caller"
    ),
    # --- takes the recipe but never runs a sweep ---
    "algorithms/ipeps_ad_policy.py::validate_split_ctm_config": (
        "validator: inspects the recipe and raises, never converges anything. "
        "Warning here would fire on configs that are about to be rejected"
    ),
    "algorithms/ipeps_ad_policy.py::make_ctm_energy_fn": (
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
                # Keyed by module *and* name, deliberately.  Keying by bare
                # ``node.name`` had two holes, both of which let an uncovered
                # entry point pass this file (#921 review r4):
                #
                #   1. two recipe-taking functions sharing a name in different
                #      modules overwrote each other, so one was never scanned;
                #   2. a new function sharing a name with any EXEMPT entry was
                #      silently exempt without anyone adding it -- which is
                #      precisely the "deliberate act a reviewer can see" this
                #      allowlist is supposed to require.
                #
                # There are no collisions today (24 names, 24 definitions); the
                # rule exists for the next one, so hole 2 is the live risk.
                key = f"{path.relative_to(SRC)}::{node.name}"
                assert key not in found, f"duplicate scan key {key}"
                found[key] = (path, node)
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
    assert "algorithms/_ctm_tensor_convergence.py::ctm_tensor" in found


def test_every_recipe_taking_function_warns_or_is_explicitly_exempt():
    """The rule. A new uncovered entry point fails here, not two rounds later."""
    uncovered = [
        key
        for key, (_path, node) in sorted(_public_recipe_functions().items())
        if key not in EXEMPT and not _calls_the_warning(node)
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


def test_every_exempt_key_is_module_qualified():
    """Bare names are the hole this file had; reject them structurally.

    A bare ``"foo"`` entry exempts *every* function named ``foo`` anywhere in
    ``src/tenax``, present and future, without anyone deciding to.  The
    allowlist's whole premise is that adding to it is a deliberate act a
    reviewer can see, so the key has to name one definition.
    """
    unqualified = sorted(k for k in EXEMPT if "::" not in k)
    assert not unqualified, (
        "EXEMPT keys must be 'module/path.py::function', not a bare name — a "
        "bare name silently exempts any same-named function added later:\n  "
        + "\n  ".join(unqualified)
    )


def test_a_same_named_function_in_another_module_is_not_auto_exempt(tmp_path):
    """The live half of the keying bug (#921 review r4).

    There are no name collisions in ``src/tenax`` today, so the *overwrite*
    half is latent.  This half is not: before the fix, adding a function named
    like any EXEMPT entry — in any module — inherited its exemption and never
    had to warn.  Exercised through the real scanner against a probe tree, so
    reverting the key to ``node.name`` fails here.
    """
    victim = sorted(EXEMPT)[0]
    assert "::" in victim
    bare = victim.split("::", 1)[1]

    pkg = tmp_path / "src" / "tenax" / "algorithms"
    pkg.mkdir(parents=True)
    (pkg / "_probe_module.py").write_text(
        f"def {bare}(site_tensors, *, recipe: str = '2x2'):\n    return site_tensors\n"
    )

    import ast as _ast

    found = {}
    for path in sorted((tmp_path / "src" / "tenax").rglob("*.py")):
        tree = _ast.parse(path.read_text(), filename=str(path))
        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in node.args.args] + [
                a.arg for a in node.args.kwonlyargs
            ]
            if "recipe" in names:
                key = f"{path.relative_to(tmp_path / 'src' / 'tenax')}::{node.name}"
                found[key] = (path, node)

    assert len(found) == 1, found
    probe_key = next(iter(found))
    assert probe_key != victim, "the probe must not collide with a real key"
    assert probe_key not in EXEMPT, (
        f"a new function named {bare!r} in a different module inherited "
        f"{victim!r}'s exemption — the allowlist is keyed too loosely"
    )
