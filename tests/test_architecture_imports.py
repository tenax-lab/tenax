"""Import graph guardrails for maintainability.

These tests keep known legacy cycles allowlisted while failing when new
algorithm modules join an import cycle.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ALGO_ROOT = Path(__file__).resolve().parents[1] / "src" / "tenax" / "algorithms"
ALGO_PREFIX = "tenax.algorithms"

ALLOWED_CYCLIC_GROUPS = [
    {
        "tenax.algorithms._ctm_projector",
        "tenax.algorithms._ctm_tensor",
        "tenax.algorithms._ctm_tensor_c4v",
        "tenax.algorithms._ctm_tensor_convergence",
        "tenax.algorithms._ctm_tensor_moves",
        "tenax.algorithms._ctm_tensor_paired_moves",
        "tenax.algorithms._split_ctm_tensor",
        "tenax.algorithms._split_ctm_tensor_convergence",
        "tenax.algorithms._split_ctm_tensor_moves",
        "tenax.algorithms.ad_utils",
    },
    {
        "tenax.algorithms._jit_sweep",
        "tenax.algorithms.dmrg",
        "tenax.algorithms.dmrg3s",
    },
]


def _module_name(path: Path) -> str:
    rel = path.relative_to(ALGO_ROOT)
    suffix = rel.with_suffix("")
    parts = [ALGO_PREFIX, *suffix.parts]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _collect_algo_modules() -> dict[str, Path]:
    return {_module_name(p): p for p in sorted(ALGO_ROOT.rglob("*.py"))}


def _resolve_from_module(source: str, level: int, module: str | None) -> str:
    parts = source.split(".")
    base = parts[:-1]
    if level > 1:
        base = base[: -(level - 1)]
    if module:
        return ".".join([*base, module])
    return ".".join(base)


def _candidate_targets(base: str, names: Iterable[str], modules: set[str]) -> set[str]:
    targets: set[str] = set()
    for name in names:
        if name == "*":
            if base in modules:
                targets.add(base)
            continue
        candidate = f"{base}.{name}"
        if candidate in modules:
            targets.add(candidate)
        elif base in modules:
            targets.add(base)
    return targets


def _build_import_graph() -> dict[str, set[str]]:
    modules = _collect_algo_modules()
    module_names = set(modules)
    graph: dict[str, set[str]] = defaultdict(set)

    for source_mod, path in modules.items():
        graph[source_mod]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in module_names:
                        graph[source_mod].add(name)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_from_module(source_mod, node.level, node.module)
                    if node.level > 0
                    else node.module
                )
                if not base or not base.startswith(ALGO_PREFIX):
                    continue
                targets = _candidate_targets(
                    base,
                    [a.name for a in node.names],
                    module_names,
                )
                graph[source_mod].update(targets)
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: set[str] = set()
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.add(popped)
                if popped == node:
                    break
            components.append(component)

    for node in graph:
        if node not in indices:
            visit(node)

    return components


def test_algorithm_import_cycles_are_allowlisted_only():
    graph = _build_import_graph()
    cyclic_components = [
        component
        for component in _strongly_connected_components(graph)
        if len(component) > 1
    ]

    unexpected = [
        component
        for component in cyclic_components
        if not any(component <= allowed for allowed in ALLOWED_CYCLIC_GROUPS)
    ]

    assert not unexpected, (
        "Detected new import cycles outside allowlisted legacy SCCs:\n"
        + "\n".join(" - " + ", ".join(sorted(component)) for component in unexpected)
    )
