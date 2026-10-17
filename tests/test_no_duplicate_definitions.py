"""Guard against duplicate definitions (#926).

A second ``class Foo`` / ``def foo`` at the same scope silently shadows the
first: the earlier one is dead code, and a caller that imported it before the
rebinding keeps an object that differs from the one every later import sees.
``test_ctm_tensor_return_info`` pins one such class by import identity; this
test scans the whole package so the next one is caught without anyone knowing
to look.

Scope: top-level and class-level ``class`` / ``def`` / ``async def`` names per
module.  A property's getter, setter and deleter legitimately share one name:
a plain ``def``, ``@property`` and ``@x.getter`` share the getter slot, and
``@x.setter`` / ``@x.deleter`` on ``def x`` each have their own (a setter
decorated from another property's name rebinds ``x`` and counts as a getter),
so a second definition in any slot is flagged.  ``typing.overload``
declarations are allowed before their implementation (one after it replaces
the implementation and is flagged), and the throwaway name ``_`` is skipped.
Definitions inside ``if`` / ``try`` blocks or nested in functions are not
scanned (deliberate: conditional fallback definitions are legitimate).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tenax"


def _is_overload(node) -> bool:
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
    return False


def _accessor_kind(node) -> str:
    """The name slot a definition occupies: ``"setter"``, ``"deleter"`` or
    ``"get"``.

    A property's getter, setter and deleter share one name.  A plain ``def``,
    ``@property`` and ``@x.getter`` all occupy the getter slot (each replaces
    the attribute's getter), so a second definition of any of them shadows
    the first; ``@x.setter`` and ``@x.deleter`` each get a slot of their own.
    """
    for dec in getattr(node, "decorator_list", []):
        if (
            isinstance(dec, ast.Attribute)
            and dec.attr in ("setter", "deleter")
            and isinstance(dec.value, ast.Name)
            and dec.value.id == node.name
        ):
            return dec.attr
    # Anything else rebinds the name: a plain def, @property, @x.getter, and
    # also @y.setter on ``def x`` (which replaces x with a property derived
    # from y, discarding x's getter -- a mistyped property name).
    return "get"


def _scope_duplicates(body, scope):
    seen: dict[tuple[str, str], int] = {}
    out = []
    for node in body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # ``_`` is the conventional throwaway name (e.g. repeated
        # ``@func.register def _(...)`` with functools.singledispatch).
        if node.name != "_":
            key = (node.name, _accessor_kind(node))
            label = node.name if key[1] == "get" else f"{node.name}.{key[1]}"
            if _is_overload(node):
                # Overload declarations legitimately precede their
                # implementation; one AFTER it replaces the implementation
                # with typing's overload dummy (calls raise
                # NotImplementedError), so it is flagged.
                if key in seen:
                    out.append(
                        f"{scope}{label} (lines {seen[key]} and {node.lineno}, "
                        "@overload after the implementation)"
                    )
            elif key in seen:
                out.append(f"{scope}{label} (lines {seen[key]} and {node.lineno})")
            else:
                seen[key] = node.lineno
        if isinstance(node, ast.ClassDef):
            out.extend(_scope_duplicates(node.body, f"{scope}{node.name}."))
    return out


def find_duplicate_definitions(root: Path) -> list[str]:
    """Return ``path: name (lines a and b)`` for every repeated definition."""
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        found.extend(
            f"{path.relative_to(root)}: {d}" for d in _scope_duplicates(tree.body, "")
        )
    return found


def test_no_duplicate_definitions_in_src():
    dups = find_duplicate_definitions(SRC)
    assert not dups, "duplicate definitions shadow each other:\n" + "\n".join(dups)


def test_scanner_catches_a_duplicated_class_and_function(tmp_path):
    """Mutation check: the scanner must flag what it exists to flag."""
    (tmp_path / "clean.py").write_text(
        "class A:\n"
        "    @property\n    def x(self): return 1\n"
        "    @x.setter\n    def x(self, v): pass\n"
        "from typing import overload\n"
        "@overload\ndef f(a: int) -> int: ...\n"
        "@overload\ndef f(a: str) -> str: ...\n"
        "def f(a): return a\n"
    )
    assert find_duplicate_definitions(tmp_path) == []

    (tmp_path / "bad.py").write_text(
        "class Foo:\n    def m(self): pass\n    def m(self): pass\n"
        "class Foo: pass\n"
        "def g(): pass\ndef g(): pass\n"
        "class P:\n"
        "    @property\n    def x(self): return 1\n"
        "    @x.setter\n    def x(self, v): pass\n"
        "    @x.setter\n    def x(self, v): pass\n"
        "class Q:\n"
        "    @property\n    def y(self): return 1\n"
        "    @y.getter\n    def y(self): return 2\n"
        "class R:\n"
        "    def z(self): return 1\n"
        "    @property\n    def z(self): return 2\n"
        "class S:\n"
        "    @property\n    def x(self): return 1\n"
        "    @property\n    def y(self): return 2\n"
        "    @y.setter\n    def x(self, v): pass\n"
        "@overload\ndef h(a: int) -> int: ...\n"
        "def h(a): return a\n"
        "@overload\ndef h(a: str) -> str: ...\n"
    )
    dups = find_duplicate_definitions(tmp_path)
    assert any("Foo (lines 1 and 4)" in d for d in dups), dups
    assert any("g (lines 5 and 6)" in d for d in dups), dups
    assert any("Foo.m" in d for d in dups), dups
    assert any("P.x.setter (lines 11 and 13)" in d for d in dups), dups
    assert any(d.endswith("Q.y (lines 16 and 18)") for d in dups), dups
    assert any(d.endswith("R.z (lines 20 and 22)") for d in dups), dups
    assert any(d.endswith("S.x (lines 25 and 29)") for d in dups), dups
    assert any(
        "h (lines 32 and 34, @overload after the implementation)" in d for d in dups
    ), dups
