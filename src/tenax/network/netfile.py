"""Cytnx-style .net file parser and NetworkBlueprint.

A ``.net`` file declares tensor network topology (tensor names, leg labels,
output ordering, optional contraction order) separately from tensor data.
This enables a "template" pattern: parse once, load different tensors,
contract repeatedly — ideal for DMRG-style inner loops.

File format::

    # comment
    TensorA: i, j, k
    TensorB: k, l, m
    TOUT: i, j, l, m
    ORDER: (TensorA, TensorB)

Line types:
- ``Name: label1, label2, ...`` — tensor declaration
- ``TOUT: label1, label2``      — output label ordering (empty = scalar)
- ``ORDER: ((A,B),C)``          — optional pairwise contraction order
- Lines starting with ``#``     — comments (ignored)

Example usage::

    bp = NetworkBlueprint(\"\"\"
    A: i, j
    B: j, k
    TOUT: i, k
    \"\"\")
    bp.put_tensor("A", tensor_a)
    bp.put_tensor("B", tensor_b)
    result = bp.launch()
"""

from __future__ import annotations

import string
from collections import Counter
from pathlib import Path
from typing import Any

from tenax.contraction.contractor import contract, contract_with_subscripts
from tenax.core.index import Label
from tenax.core.tensor import Tensor

# ---------- .net file parser ----------


def parse_netfile(
    source: str | Path | list[str],
) -> dict[str, Any]:
    """Parse a ``.net`` file or string into a topology dictionary.

    Args:
        source: One of:
            - A ``Path`` to a ``.net`` file on disk.
            - A multi-line string with the file contents.
            - A list of already-split lines.

    Returns:
        Dictionary with keys:

        - ``"tensors"``: ``dict[str, list[str]]`` — tensor name → ordered labels.
        - ``"tout"``: ``list[str] | None`` — output labels (``None`` if absent,
          ``[]`` if ``TOUT:`` line is empty).
        - ``"order"``: ``str | None`` — raw ORDER string (``None`` if absent).

    Raises:
        ValueError: On duplicate tensor names, invalid syntax, or a label
            appearing on more than two tensors.
    """
    lines = _read_lines(source)

    tensors: dict[str, list[str]] = {}
    tout: list[str] | None = None
    order: str | None = None
    tout_seen = False

    for lineno, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            raise ValueError(
                f"Line {lineno}: expected 'Name: labels' or 'TOUT:/ORDER:', "
                f"got {line!r}"
            )

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if key == "TOUT":
            if tout_seen:
                raise ValueError(f"Line {lineno}: duplicate TOUT declaration")
            tout_seen = True
            tout = (
                [lbl.strip() for lbl in value.split(",") if lbl.strip()]
                if value
                else []
            )
        elif key == "ORDER":
            if order is not None:
                raise ValueError(f"Line {lineno}: duplicate ORDER declaration")
            order = value if value else None
        else:
            # Tensor declaration
            name = key
            if name in tensors:
                raise ValueError(f"Line {lineno}: duplicate tensor name {name!r}")
            labels = [lbl.strip() for lbl in value.split(",") if lbl.strip()]
            if not labels:
                raise ValueError(f"Line {lineno}: tensor {name!r} has no labels")
            tensors[name] = labels

    if not tensors:
        raise ValueError("No tensor declarations found")

    # Validate: no label on more than 2 tensors
    label_counts: Counter[str] = Counter()
    for labels in tensors.values():
        for lbl in labels:
            label_counts[lbl] += 1
    for lbl, count in label_counts.items():
        if count > 2:
            raise ValueError(
                f"Label {lbl!r} appears on {count} tensors (max 2 allowed)"
            )

    return {"tensors": tensors, "tout": tout, "order": order}


def _read_lines(source: str | Path | list[str]) -> list[str]:
    """Normalise *source* into a list of lines."""
    if isinstance(source, list):
        return source
    if isinstance(source, Path):
        return source.read_text().splitlines()
    # str — could be a file path or inline content.
    # Prefer inline content if it clearly looks like one (multiline), and
    # guard path checks because very long/invalid pseudo-paths may raise.
    if "\n" in source:
        return source.splitlines()
    p = Path(source)
    try:
        if p.is_file():
            return p.read_text().splitlines()
    except (OSError, ValueError):
        pass
    return source.splitlines()


# ---------- ORDER parser ----------


def _parse_order(
    order_str: str,
    tensor_names: set[str],
) -> list[tuple[str, str]]:
    """Parse a nested-parenthesis ORDER string into pairwise contraction steps.

    ``((A,B),C)`` → ``[("A","B"), ("_0","C")]``

    Args:
        order_str:    Raw ORDER value, e.g. ``"((A,B),C)"``.
        tensor_names: Valid tensor names (for validation).

    Returns:
        List of ``(left, right)`` pairs executed sequentially.  Intermediate
        results are named ``_0``, ``_1``, …

    Raises:
        ValueError: On syntax errors or unknown tensor names.
    """
    tokens = _tokenize_order(order_str)
    tree, pos = _parse_order_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(
            f"ORDER: unexpected tokens after position {pos}: {tokens[pos:]}"
        )

    steps: list[tuple[str, str]] = []
    _counter = [0]

    def _flatten(node: Any) -> str:
        if isinstance(node, str):
            if node not in tensor_names:
                raise ValueError(
                    f"ORDER: unknown tensor name {node!r}. "
                    f"Known: {sorted(tensor_names)}"
                )
            return node
        # node is a tuple (left_tree, right_tree)
        left_name = _flatten(node[0])
        right_name = _flatten(node[1])
        result_name = f"_{_counter[0]}"
        _counter[0] += 1
        steps.append((left_name, right_name))
        return result_name

    _flatten(tree)
    return steps


def _tokenize_order(s: str) -> list[str]:
    """Tokenize an ORDER string into ``(``, ``)``, ``,``, and name tokens."""
    tokens: list[str] = []
    i = 0
    s = s.strip()
    while i < len(s):
        c = s[i]
        if c in "(),":
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in "(), \t\n":
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_order_expr(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Recursive descent parser for ORDER expressions.

    Returns ``(tree, next_pos)`` where *tree* is either a string (tensor
    name) or a ``(left_tree, right_tree)`` tuple.
    """
    if pos >= len(tokens):
        raise ValueError("ORDER: unexpected end of expression")

    if tokens[pos] == "(":
        pos += 1  # consume '('
        left, pos = _parse_order_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ",":
            raise ValueError(
                f"ORDER: expected ',' after first operand at position {pos}"
            )
        pos += 1  # consume ','
        right, pos = _parse_order_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError(f"ORDER: expected ')' at position {pos}")
        pos += 1  # consume ')'
        return (left, right), pos

    # Must be a tensor name
    name = tokens[pos]
    if name in "(),":
        raise ValueError(f"ORDER: unexpected token {name!r} at position {pos}")
    return name, pos + 1


# ---------- Subscript pre-computation ----------


def _labels_to_subscripts_from_names(
    tensor_labels: dict[str, list[str]],
    node_order: list[str],
    output_labels: list[str] | None,
) -> str:
    """Build an einsum subscript string from label names only (no Tensor objects).

    Mirrors :func:`tenax.contraction.contractor._labels_to_subscripts` but
    operates on label names alone, enabling pre-computation at parse time.

    Args:
        tensor_labels: ``{tensor_name: [label, ...]}`` in declaration order.
        node_order:    Order of tensors in the subscript LHS.
        output_labels: Explicit output label ordering, or ``None`` for natural
            order (free labels in order of first appearance).

    Returns:
        Einsum subscript string, e.g. ``"ij,jk->ik"``.

    Raises:
        ValueError: If a label appears > 2 times or output_labels contains
            a non-free label.
    """
    # Count label occurrences
    label_counts: Counter[str] = Counter()
    for name in node_order:
        for lbl in tensor_labels[name]:
            label_counts[lbl] += 1

    for lbl, count in label_counts.items():
        if count > 2:
            raise ValueError(f"Label {lbl!r} appears {count} times (max 2)")

    free_labels = {lbl for lbl, cnt in label_counts.items() if cnt == 1}

    # Assign characters
    all_labels = sorted(label_counts.keys())
    if len(all_labels) > 52:
        raise ValueError(f"Too many unique labels ({len(all_labels)}); max 52")
    chars = string.ascii_lowercase + string.ascii_uppercase
    label_to_char = {lbl: chars[i] for i, lbl in enumerate(all_labels)}

    # Build per-tensor subscript strings
    tensor_subs = []
    for name in node_order:
        subs = "".join(label_to_char[lbl] for lbl in tensor_labels[name])
        tensor_subs.append(subs)

    # Determine output ordering
    if output_labels is None:
        seen: set[str] = set()
        ordered_free: list[str] = []
        for name in node_order:
            for lbl in tensor_labels[name]:
                if lbl in free_labels and lbl not in seen:
                    ordered_free.append(lbl)
                    seen.add(lbl)
        output_labels = ordered_free
    else:
        for lbl in output_labels:
            if lbl not in free_labels:
                raise ValueError(
                    f"output_labels contains {lbl!r} which is not a free label. "
                    f"Free labels are: {sorted(free_labels)}"
                )

    output_subs = "".join(label_to_char[lbl] for lbl in output_labels)
    return ",".join(tensor_subs) + "->" + output_subs


# ---------- NetworkBlueprint ----------


class NetworkBlueprint:
    """Reusable tensor network template parsed from a ``.net`` specification.

    Lifecycle: **parse → fill slots → execute** (repeat fill+execute as needed).

    Args:
        source: A ``.net`` file path (``str`` or ``Path``), a multi-line
            string, or a list of lines.

    Example::

        bp = NetworkBlueprint(\"\"\"
        A: i, j
        B: j, k
        TOUT: i, k
        \"\"\")
        bp.put_tensor("A", tensor_a)
        bp.put_tensor("B", tensor_b)
        result = bp.launch()
    """

    def __init__(self, source: str | Path | list[str]) -> None:
        spec = parse_netfile(source)
        self._tensor_labels: dict[str, list[str]] = spec["tensors"]
        self._node_order: list[str] = list(spec["tensors"].keys())
        self._tout: list[str] | None = spec["tout"]
        self._order_str: str | None = spec["order"]

        # Pre-compute ORDER steps if provided
        self._order_steps: list[tuple[str, str]] | None = None
        if self._order_str is not None:
            self._order_steps = _parse_order(self._order_str, set(self._node_order))

        # Pre-compute einsum subscripts (for the no-ORDER path)
        self._subscripts: str = _labels_to_subscripts_from_names(
            self._tensor_labels, self._node_order, self._tout
        )

        # Tensor slots
        self._tensors: dict[str, Tensor] = {}

    # ---- Properties ----

    @property
    def tensor_names(self) -> list[str]:
        """Tensor names in declaration order."""
        return list(self._node_order)

    @property
    def subscripts(self) -> str:
        """Pre-computed einsum subscript string."""
        return self._subscripts

    @property
    def output_labels(self) -> list[str] | None:
        """Output label ordering from the TOUT line (``None`` if absent)."""
        return self._tout

    # ---- Load tensors ----

    def put_tensor(
        self,
        name: str,
        tensor: Tensor,
        label_order: list[Label] | None = None,
    ) -> None:
        """Assign a tensor to a named slot.

        Args:
            name:        Tensor name (must match a name in the ``.net`` file).
            tensor:      The tensor to place.
            label_order: If given, specifies the current label ordering of
                *tensor* so it can be relabelled to match the blueprint.
                Must be the same length and set as ``tensor.labels()``.

        Raises:
            KeyError: If *name* is not declared in the blueprint.
            ValueError: If the tensor's rank doesn't match the blueprint.
        """
        if name not in self._tensor_labels:
            raise KeyError(f"Unknown tensor name {name!r}. Known: {self._node_order}")

        blueprint_labels = self._tensor_labels[name]
        if tensor.ndim != len(blueprint_labels):
            raise ValueError(
                f"Tensor {name!r} has rank {tensor.ndim} but blueprint "
                f"expects rank {len(blueprint_labels)}"
            )

        if label_order is not None:
            # label_order tells us the meaning of each axis on the incoming
            # tensor.  We relabel those axes to the blueprint labels.
            if len(label_order) != len(blueprint_labels):
                raise ValueError(
                    f"label_order has {len(label_order)} entries but "
                    f"blueprint expects {len(blueprint_labels)}"
                )
            current = [str(lb) for lb in tensor.labels()]
            provided = [str(lb) for lb in label_order]
            if sorted(provided) != sorted(current):
                raise ValueError(
                    f"label_order {label_order} doesn't match tensor labels "
                    f"{list(tensor.labels())} (as sets)."
                )
            mapping = {
                old: new
                for old, new in zip(label_order, blueprint_labels)
                if old != new
            }
            if mapping:
                tensor = tensor.relabels(mapping)
        else:
            # No label_order — check that the tensor already carries the
            # correct label *set*, then reorder if necessary.
            current = list(tensor.labels())
            if sorted(str(lb) for lb in current) != sorted(blueprint_labels):
                raise ValueError(
                    f"Tensor {name!r} labels {current} don't match blueprint "
                    f"labels {blueprint_labels} (as sets). Pass label_order "
                    f"to map them explicitly."
                )
            # If order differs, transpose to match blueprint ordering
            if [str(lb) for lb in current] != blueprint_labels:
                perm = [
                    next(i for i, cl in enumerate(current) if str(cl) == bl)
                    for bl in blueprint_labels
                ]
                tensor = tensor.transpose(tuple(perm))

        self._tensors[name] = tensor

    def put_tensors(self, mapping: dict[str, Tensor]) -> None:
        """Assign multiple tensors at once.

        Args:
            mapping: ``{name: tensor}`` pairs.
        """
        for name, tensor in mapping.items():
            self.put_tensor(name, tensor)

    def clear_tensors(self) -> None:
        """Remove all loaded tensors (keeps the blueprint topology)."""
        self._tensors.clear()

    def is_ready(self) -> bool:
        """Return ``True`` if every slot has a tensor."""
        return set(self._tensors.keys()) == set(self._node_order)

    # ---- Execute ----

    def launch(self, optimize: str = "auto") -> Tensor:
        """Contract the network and return the result tensor.

        Args:
            optimize: opt_einsum path optimizer (ignored when ORDER is given).

        Returns:
            Contracted tensor with labels matching the TOUT ordering.

        Raises:
            RuntimeError: If not all tensor slots have been filled.
        """
        if not self.is_ready():
            missing = set(self._node_order) - set(self._tensors.keys())
            raise RuntimeError(f"Cannot launch: missing tensors {sorted(missing)}")

        if self._order_steps is not None:
            return self._launch_ordered()
        return self._launch_einsum(optimize)

    def _launch_einsum(self, optimize: str) -> Tensor:
        """Contract via a single einsum call using pre-computed subscripts."""
        tensors = [self._tensors[name] for name in self._node_order]

        # Build output_indices from the actual tensors
        from tenax.contraction.contractor import _labels_to_subscripts

        _, output_indices = _labels_to_subscripts(tensors, self._tout)
        return contract_with_subscripts(
            tensors, self._subscripts, output_indices, optimize
        )

    def _launch_ordered(self) -> Tensor:
        """Contract following explicit ORDER steps (pairwise)."""
        assert self._order_steps is not None

        intermediates: dict[str, Tensor] = dict(self._tensors)
        for i, (left, right) in enumerate(self._order_steps):
            t_left = intermediates.pop(left)
            t_right = intermediates.pop(right)
            result = contract(t_left, t_right)
            intermediates[f"_{i}"] = result

        # Should be exactly one tensor left
        assert len(intermediates) == 1, (
            f"ORDER did not reduce to a single tensor: {list(intermediates)}"
        )
        result = next(iter(intermediates.values()))

        # Apply TOUT permutation if needed
        if self._tout is not None and self._tout:
            current = [str(lb) for lb in result.labels()]
            if current != self._tout:
                perm = [current.index(lbl) for lbl in self._tout]
                result = result.transpose(tuple(perm))
        elif self._tout is not None and not self._tout:
            # TOUT is empty → scalar, nothing to permute
            pass

        return result

    # ---- Interop ----

    def to_tensor_network(self) -> Any:
        """Convert loaded tensors into a :class:`TensorNetwork`.

        All loaded tensors are added as nodes.  Legs with matching labels
        across tensors are auto-connected.

        Returns:
            :class:`~tenax.network.network.TensorNetwork`

        Raises:
            RuntimeError: If not all tensor slots are filled.
        """
        if not self.is_ready():
            missing = set(self._node_order) - set(self._tensors.keys())
            raise RuntimeError(f"Cannot convert: missing tensors {sorted(missing)}")

        from tenax.network.network import TensorNetwork

        tn = TensorNetwork(name="from_netfile")
        for name in self._node_order:
            tn.add_node(name, self._tensors[name])

        # Connect by shared labels
        names = self._node_order
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                labels_i = set(self._tensor_labels[names[i]])
                labels_j = set(self._tensor_labels[names[j]])
                shared = labels_i & labels_j
                for lbl in sorted(shared):
                    try:
                        tn.connect(names[i], lbl, names[j], lbl)
                    except (ValueError, KeyError):
                        pass

        return tn


# ---------- Convenience ----------


def from_netfile(source: str | Path | list[str]) -> NetworkBlueprint:
    """Create a :class:`NetworkBlueprint` from a ``.net`` source.

    Equivalent to ``NetworkBlueprint(source)``.

    Args:
        source: File path, multi-line string, or list of lines.

    Returns:
        A new :class:`NetworkBlueprint`.
    """
    return NetworkBlueprint(source)
