"""Graph-based tensor network container with label-based contraction.

TensorNetwork represents a collection of tensors connected by their shared
leg labels. The graph structure (networkx.MultiGraph) tracks which legs are
connected, and the contraction engine (contractor.py) handles the actual
computation.

Key design choices:
- Edges are identified by (node_a, label_a, node_b, label_b) — no positional indexing
- connect_by_shared_label() auto-connects nodes that share a label name
- Contraction cache keyed by tuple[NodeId] for O(1) lookup (order-sensitive)
- Cache invalidated on any graph structure change (add/remove/replace/connect)
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx

from tenax.contraction.contractor import _labels_to_subscripts, contract_with_subscripts
from tenax.core.index import Label, TensorIndex
from tenax.core.tensor import Tensor

NodeId = Hashable


class TensorNetwork:
    """Graph-based container for a tensor network.

    The internal representation is an nx.MultiGraph where:
    - Nodes store the Tensor and its node_id.
    - Edges store which leg labels are connected: (label_a, label_b).
    - "Open" edges (no counterpart) represent physical/free indices.

    The contraction cache maps (tuple[NodeId], output_labels, optimize) ->
    Tensor, and is invalidated whenever the graph structure changes.

    Args:
        name: Optional human-readable name for this network.

    Example:
        >>> tn = TensorNetwork()
        >>> tn.add_node("A", tensor_A)
        >>> tn.add_node("B", tensor_B)
        >>> tn.connect_by_shared_label("A", "B")
        >>> result = tn.contract()
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._graph: nx.MultiGraph = nx.MultiGraph()
        self._tensors: dict[NodeId, Tensor] = {}
        # Edge data stored as list of dicts: {"label_a": ..., "label_b": ...}
        self._edge_connections: dict[tuple[NodeId, NodeId, int], dict] = {}
        self._cache: dict[Any, Tensor] = {}

    # ------------------------------------------------------------------ #
    # Node management                                                      #
    # ------------------------------------------------------------------ #

    def add_node(self, node_id: NodeId, tensor: Tensor) -> None:
        """Add a tensor as a node in the network.

        Args:
            node_id: Unique identifier for this node.
            tensor:  The tensor to store at this node.

        Raises:
            ValueError: If node_id already exists.
            ValueError: If tensor has duplicate labels.
        """
        if node_id in self._tensors:
            raise ValueError(
                f"Node {node_id!r} already exists. Use replace_tensor() to update."
            )

        labels = tensor.labels()
        if len(labels) != len(set(labels)):
            dupes = [lbl for lbl in labels if labels.count(lbl) > 1]
            raise ValueError(
                f"Tensor for node {node_id!r} has duplicate labels: {dupes}"
            )

        self._graph.add_node(node_id)
        self._tensors[node_id] = tensor
        self._invalidate_cache()

    def remove_node(self, node_id: NodeId) -> Tensor:
        """Remove a node and all edges connected to it.

        Args:
            node_id: The node to remove.

        Returns:
            The tensor that was stored at this node.

        Raises:
            KeyError: If node_id not found.
        """
        if node_id not in self._tensors:
            raise KeyError(f"Node {node_id!r} not found")

        tensor = self._tensors.pop(node_id)
        self._graph.remove_node(node_id)
        self._invalidate_cache()
        return tensor

    def replace_tensor(self, node_id: NodeId, tensor: Tensor) -> None:
        """Replace the tensor at an existing node.

        The new tensor must have the same set of labels as the old one,
        since labels define the connectivity in the graph. Dimensions and
        flows on connected legs are also validated.

        Args:
            node_id: The node to update.
            tensor:  The replacement tensor.

        Raises:
            KeyError:   If node_id not found.
            ValueError: If labels differ, or if a connected leg has a
                        different dimension or flow from the original.
        """
        if node_id not in self._tensors:
            raise KeyError(f"Node {node_id!r} not found")

        old_tensor = self._tensors[node_id]
        old_labels = set(old_tensor.labels())
        new_labels = set(tensor.labels())
        if old_labels != new_labels:
            raise ValueError(
                f"Replacement tensor has different labels. "
                f"Old: {sorted(old_labels)}, New: {sorted(new_labels)}"
            )

        # Validate dimensions and flows on connected legs
        old_idx_map = {idx.label: idx for idx in old_tensor.indices}
        new_idx_map = {idx.label: idx for idx in tensor.indices}
        for u, v, data in self._graph.edges(node_id, data=True):
            for label in self._labels_for_node(data, node_id):
                if label in old_idx_map and label in new_idx_map:
                    old_idx = old_idx_map[label]
                    new_idx = new_idx_map[label]
                    if old_idx.dim != new_idx.dim:
                        raise ValueError(
                            f"Replacement tensor changes dimension of connected "
                            f"leg {label!r}: {old_idx.dim} -> {new_idx.dim}."
                        )
                    if old_idx.flow != new_idx.flow:
                        raise ValueError(
                            f"Replacement tensor changes flow of connected "
                            f"leg {label!r}: {old_idx.flow.name} -> {new_idx.flow.name}."
                        )

        self._tensors[node_id] = tensor
        self._invalidate_cache()

    def get_tensor(self, node_id: NodeId) -> Tensor:
        """Return the tensor stored at a node.

        Args:
            node_id: Identifier of the node to look up.

        Returns:
            The Tensor (DenseTensor or SymmetricTensor) at that node.

        Raises:
            KeyError: If *node_id* is not in the network.
        """
        if node_id not in self._tensors:
            raise KeyError(f"Node {node_id!r} not found")
        return self._tensors[node_id]

    # ------------------------------------------------------------------ #
    # Edge management                                                      #
    # ------------------------------------------------------------------ #

    def connect(
        self,
        node_a: NodeId,
        label_a: Label,
        node_b: NodeId,
        label_b: Label,
    ) -> None:
        """Connect a specific leg of node_a to a specific leg of node_b.

        After connection, these two legs are treated as contracted when
        contract() is called on a subgraph containing both nodes.

        The leg labels on the two tensors do NOT need to match — the graph
        records which labels are paired. However, the TensorIndex objects
        must be compatible (same symmetry type, same bond dimension, opposite
        flows).

        Args:
            node_a:  First node.
            label_a: Label of the leg on node_a to connect.
            node_b:  Second node.
            label_b: Label of the leg on node_b to connect.

        Raises:
            KeyError:   If either node not found.
            KeyError:   If label not found on the corresponding tensor.
            ValueError: If the two TensorIndex objects are incompatible.
        """
        idx_a = self._get_index(node_a, label_a)
        idx_b = self._get_index(node_b, label_b)

        if not idx_a.compatible_with(idx_b):
            raise ValueError(
                f"Incompatible indices: "
                f"{node_a!r}[{label_a!r}] (dim={idx_a.dim}, flow={idx_a.flow.name}) "
                f"and {node_b!r}[{label_b!r}] (dim={idx_b.dim}, flow={idx_b.flow.name})"
            )

        self._graph.add_edge(
            node_a,
            node_b,
            label_a=label_a,
            label_b=label_b,
            owner_a=node_a,
            owner_b=node_b,
        )
        self._invalidate_cache()

    def connect_by_shared_label(self, node_a: NodeId, node_b: NodeId) -> int:
        """Auto-connect all legs sharing the same label between two nodes.

        Finds labels that appear on both node_a and node_b and connects them.
        This is the most natural API for networks where shared label names
        already encode the connectivity.

        Args:
            node_a: First node.
            node_b: Second node.

        Returns:
            Number of connections made.

        Raises:
            KeyError: If either node not found.
            ValueError: If no shared labels exist.
            ValueError: If shared labels have incompatible index objects.
        """
        labels_a = set(self._tensors[node_a].labels())
        labels_b = set(self._tensors[node_b].labels())
        shared = labels_a & labels_b

        if not shared:
            raise ValueError(
                f"No shared labels between {node_a!r} "
                f"(labels={sorted(labels_a)}) and {node_b!r} "
                f"(labels={sorted(labels_b)})"
            )

        # Validate all shared labels before mutating, so the network is
        # not left partially connected on failure.
        sorted_shared = sorted(shared, key=str)
        for label in sorted_shared:
            idx_a = self._get_index(node_a, label)
            idx_b = self._get_index(node_b, label)
            if not idx_a.compatible_with(idx_b):
                raise ValueError(
                    f"Incompatible indices on shared label {label!r}: "
                    f"{node_a!r} (dim={idx_a.dim}, flow={idx_a.flow.name}) "
                    f"vs {node_b!r} (dim={idx_b.dim}, flow={idx_b.flow.name})"
                )

        # Already validated; bypass connect() to avoid re-checking.
        for label in sorted_shared:
            self._graph.add_edge(
                node_a,
                node_b,
                label_a=label,
                label_b=label,
                owner_a=node_a,
                owner_b=node_b,
            )
        self._invalidate_cache()

        return len(sorted_shared)

    def disconnect(
        self,
        node_a: NodeId,
        label_a: Label,
        node_b: NodeId,
        label_b: Label,
    ) -> None:
        """Remove the edge connecting these two labeled legs.

        Args:
            node_a:  First node.
            label_a: Label of the leg on node_a.
            node_b:  Second node.
            label_b: Label of the leg on node_b.

        Raises:
            KeyError: If no such edge exists.
        """
        edges = list(self._graph.edges(node_a, data=True, keys=True))
        for u, v, key, data in edges:
            owner_a = data.get("owner_a")
            owner_b = data.get("owner_b")
            la = data.get("label_a")
            lb = data.get("label_b")
            # Match regardless of iteration order: check owner-based labels
            if (
                owner_a == node_a
                and la == label_a
                and owner_b == node_b
                and lb == label_b
            ) or (
                owner_a == node_b
                and la == label_b
                and owner_b == node_a
                and lb == label_a
            ):
                self._graph.remove_edge(u, v, key)
                self._invalidate_cache()
                return

        raise KeyError(
            f"No edge found connecting {node_a!r}[{label_a!r}] to "
            f"{node_b!r}[{label_b!r}]"
        )

    def relabel_bond(
        self,
        node_id: NodeId,
        old_label: Label,
        new_label: Label,
    ) -> None:
        """Rename a leg's label on a node and update all connected edges.

        Args:
            node_id:   The node whose leg label to rename.
            old_label: The current label.
            new_label: The new label.

        Raises:
            KeyError:   If node not found or old_label not in tensor.
            ValueError: If new_label already exists on the tensor (and
                        differs from old_label).
        """
        tensor = self._tensors[node_id]
        if new_label != old_label and new_label in tensor.labels():
            raise ValueError(
                f"Cannot relabel {old_label!r} -> {new_label!r} on node "
                f"{node_id!r}: label {new_label!r} already exists on this tensor."
            )
        self._tensors[node_id] = tensor.relabel(old_label, new_label)

        # Update any edges that reference this label
        for u, v, key, data in list(self._graph.edges(node_id, data=True, keys=True)):
            if data.get("owner_a") == node_id and data.get("label_a") == old_label:
                self._graph[u][v][key]["label_a"] = new_label
            elif data.get("owner_b") == node_id and data.get("label_b") == old_label:
                self._graph[u][v][key]["label_b"] = new_label

        self._invalidate_cache()

    def open_legs(self, node_id: NodeId) -> list[Label]:
        """Return labels of legs on node_id not connected to any other node.

        Args:
            node_id: The node to query.

        Returns:
            List of free (open) leg labels.
        """
        tensor = self._tensors[node_id]
        all_labels = set(tensor.labels())

        # Collect all connected labels for this node
        connected_labels: set[Label] = set()
        for u, v, data in self._graph.edges(node_id, data=True):
            connected_labels.update(self._labels_for_node(data, node_id))

        return sorted(all_labels - connected_labels, key=str)

    # ------------------------------------------------------------------ #
    # Contraction                                                          #
    # ------------------------------------------------------------------ #

    def contract(
        self,
        nodes: list[NodeId] | None = None,
        output_labels: Sequence[Label] | None = None,
        optimize: str = "auto",
        cache: bool = True,
    ) -> Tensor:
        """Contract a subset of nodes (or all nodes if nodes is None).

        Internally the method checks the cache, builds an einsum subscript
        string from the graph edge connectivity (contracting shared legs,
        keeping free legs), calls ``contract_with_subscripts()`` via
        opt_einsum, and caches the result.

        The output tensor's leg labels are the free labels in output_labels
        order (or natural order if not specified).

        Args:
            nodes:         List of node IDs to contract. None = all nodes.
            output_labels: Explicit output leg ordering by label.
            optimize:      opt_einsum optimizer.
            cache:         Whether to use/populate the cache.

        Returns:
            Contracted Tensor with all open/free legs remaining.
        """
        if nodes is None:
            nodes = list(self._tensors.keys())

        cache_key = (tuple(nodes), tuple(output_labels or ()), optimize)
        if cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = self._contract_nodes(nodes, output_labels, optimize)

        if cache:
            self._cache[cache_key] = result

        return result

    def _contract_nodes(
        self,
        nodes: list[NodeId],
        output_labels: Sequence[Label] | None,
        optimize: str,
    ) -> Tensor:
        """Build subscripts from graph connectivity and execute contraction."""
        node_set = set(nodes)

        # Build the subscript string from graph edges.
        # Legs connected within the subset get the same character (contracted).
        # Legs connected outside or unconnected get unique characters (free).

        # Collect all edges within the subset, mapping them to shared labels
        internal_edges: list[tuple[NodeId, Label, NodeId, Label]] = []
        for u, v, data in self._graph.edges(data=True):
            owner_a = data.get("owner_a", u)
            owner_b = data.get("owner_b", v)
            if owner_a in node_set and owner_b in node_set:
                la = data.get("label_a")
                lb = data.get("label_b")
                internal_edges.append((owner_a, la, owner_b, lb))

        # We need to build a new set of tensors with relabeled legs so that
        # internally-connected legs share a common label for the subscript builder.
        # Strategy: rename label_b to label_a for each internal edge pair
        # (make them share one label) using a copy of each tensor with new labels.

        # Build a mapping: for each node, which labels should be renamed and to what
        relabel_map: dict[NodeId, dict[Label, Label]] = {n: {} for n in nodes}

        # Collect all labels currently in use across the subset so we can
        # generate fresh intermediate labels that don't collide with anything.
        all_used_labels: set[Label] = set()
        for n in nodes:
            all_used_labels.update(self._tensors[n].labels())

        _fresh_counter = 0

        def _fresh_label() -> str:
            """Generate a label not present on any tensor in the subset."""
            nonlocal _fresh_counter
            while True:
                candidate = f"__internal_bond_{_fresh_counter}__"
                _fresh_counter += 1
                if candidate not in all_used_labels:
                    all_used_labels.add(candidate)
                    return candidate

        for node_a, label_a, node_b, label_b in internal_edges:
            # Make the two legs share a common label so the subscript
            # builder contracts them.
            if label_a != label_b:
                # Determine the effective labels after any prior relabeling
                effective_b_labels = set(self._tensors[node_b].labels())
                # Account for labels already scheduled for renaming
                for old, new in relabel_map[node_b].items():
                    effective_b_labels.discard(old)
                    effective_b_labels.add(new)

                if label_a not in effective_b_labels:
                    # Safe to rename label_b → label_a on node_b
                    relabel_map[node_b][label_b] = label_a
                else:
                    # label_a already exists on node_b for a *different* leg.
                    # Use a fresh intermediate label for both sides.
                    fresh = _fresh_label()
                    relabel_map[node_a][label_a] = fresh
                    relabel_map[node_b][label_b] = fresh

        # Apply relabeling
        relabeled_tensors = []
        for node in nodes:
            tensor = self._tensors[node]
            if relabel_map[node]:
                tensor = tensor.relabels(relabel_map[node])
            relabeled_tensors.append(tensor)

        # Disambiguate duplicate labels on disconnected legs.
        # After edge-relabeling, any label appearing on multiple tensors that
        # is NOT an intentionally-shared bond label must be renamed so that
        # _labels_to_subscripts treats them as independent free legs.
        bonded_label_pairs: set[tuple[int, Label]] = set()
        node_index = {n: i for i, n in enumerate(nodes)}
        for node_a, label_a, node_b, label_b in internal_edges:
            # After relabeling, both sides share the same label.
            effective_a = relabel_map[node_a].get(label_a, label_a)
            bonded_label_pairs.add((node_index[node_a], effective_a))
            bonded_label_pairs.add((node_index[node_b], effective_a))

        # Find labels that appear on multiple tensors but aren't bonded
        label_occurrences: dict[Label, list[int]] = {}
        for ti, tensor in enumerate(relabeled_tensors):
            for lbl in tensor.labels():
                label_occurrences.setdefault(lbl, []).append(ti)

        for lbl, tensor_idxs in label_occurrences.items():
            if len(tensor_idxs) <= 1:
                continue
            # Check if ALL occurrences are bonded (intentional)
            all_bonded = all((ti, lbl) in bonded_label_pairs for ti in tensor_idxs)
            if all_bonded:
                continue
            # Disambiguate: rename on all tensors except the first occurrence
            for ti in tensor_idxs[1:]:
                if (ti, lbl) not in bonded_label_pairs:
                    fresh = _fresh_label()
                    relabeled_tensors[ti] = relabeled_tensors[ti].relabel(lbl, fresh)

        # Now use the label-based subscript builder
        subscripts, auto_output_indices = _labels_to_subscripts(
            relabeled_tensors, output_labels
        )

        return contract_with_subscripts(
            relabeled_tensors, subscripts, auto_output_indices, optimize
        )

    # ------------------------------------------------------------------ #
    # Cache management                                                     #
    # ------------------------------------------------------------------ #

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    def clear_cache(self) -> None:
        """Manually clear the contraction cache."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def node_ids(self) -> list[NodeId]:
        """Return list of all node IDs."""
        return list(self._tensors.keys())

    def neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Return list of nodes connected to node_id."""
        return list(self._graph.neighbors(node_id))

    def is_connected(self) -> bool:
        """Return True if the network graph is connected."""
        if len(self._graph) == 0:
            return True
        return nx.is_connected(self._graph)

    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self._tensors)

    def n_edges(self) -> int:
        """Number of edges (connected leg pairs) in the network."""
        return self._graph.number_of_edges()

    @staticmethod
    def _labels_for_node(data: dict, node_id: NodeId) -> list[Label]:
        """Return the label(s) that belong to *node_id* in an edge's data dict.

        Edge data stores ``owner_a``/``owner_b`` to track which node owns
        ``label_a``/``label_b``, since ``nx.MultiGraph.edges(node)`` does
        not guarantee a stable ``(u, v)`` ordering.

        For self-loops (``owner_a == owner_b == node_id``), both labels are
        returned.
        """
        result: list[Label] = []
        if data.get("owner_a") == node_id:
            result.append(data.get("label_a"))
        if data.get("owner_b") == node_id:
            result.append(data.get("label_b"))
        return result

    def _get_index(self, node_id: NodeId, label: Label) -> TensorIndex:
        """Retrieve TensorIndex for a specific labeled leg."""
        tensor = self._tensors[node_id]
        for idx in tensor.indices:
            if idx.label == label:
                return idx
        raise KeyError(
            f"Label {label!r} not found on node {node_id!r}. "
            f"Available labels: {list(tensor.labels())}"
        )

    def to_mermaid(self) -> str:
        """Return a Mermaid graph diagram of the network.

        Contracted edges are shown as solid lines with bond labels.
        Open (free) legs are shown as dangling circle nodes.
        """
        lines = ["graph LR"]

        # Node definitions with shapes
        for nid in sorted(self._tensors, key=str):
            tensor = self._tensors[nid]
            shape = ",".join(str(idx.dim) for idx in tensor.indices)
            safe_id = str(nid).replace(" ", "_")
            lines.append(f'  {safe_id}["{nid} ({shape})"]')

        # Contracted edges
        seen_edges: set[tuple] = set()
        for u, v, data in self._graph.edges(data=True):
            label_a = data.get("label_a", "")
            label_b = data.get("label_b", "")
            edge_key = (
                min(str(u), str(v)),
                max(str(u), str(v)),
                str(label_a),
                str(label_b),
            )
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            safe_u = str(u).replace(" ", "_")
            safe_v = str(v).replace(" ", "_")
            bond_label = label_a if label_a == label_b else f"{label_a}/{label_b}"
            lines.append(f"  {safe_u} ---|{bond_label}| {safe_v}")

        # Open legs as dangling nodes
        for nid in sorted(self._tensors, key=str):
            safe_id = str(nid).replace(" ", "_")
            for label in self.open_legs(nid):
                safe_label = str(label).replace(" ", "_")
                open_id = f"{safe_id}_{safe_label}"
                lines.append(f'  {safe_id} -.- {open_id}(("{label}"))')

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TensorNetwork(name={self.name!r}, "
            f"nodes={self.n_nodes()}, edges={self.n_edges()})"
        )


# ------------------------------------------------------------------ #
# MPS / PEPS convenience constructors                                #
# ------------------------------------------------------------------ #


def build_mps(
    tensors: list[Tensor],
) -> TensorNetwork:
    """Build a Matrix Product State as a TensorNetwork.

    Adjacent tensors are connected wherever they share a common label
    with compatible dimensions.  Labels that appear on only one tensor
    remain as open (dangling) legs.

    Args:
        tensors: List of site tensors [A_0, A_1, ..., A_{L-1}].

    Returns:
        TensorNetwork with shared virtual bonds connected.
    """
    L = len(tensors)
    tn = TensorNetwork(name="MPS")

    for i, tensor in enumerate(tensors):
        tn.add_node(i, tensor)

    # Connect adjacent sites by shared labels
    for i in range(L - 1):
        labels_i = set(tensors[i].labels())
        labels_next = set(tensors[i + 1].labels())
        shared = labels_i & labels_next

        for label in sorted(shared, key=str):
            tn.connect(i, label, i + 1, label)

    return tn


def build_peps(
    tensors: list[list[Tensor]],
    Lx: int,
    Ly: int,
) -> TensorNetwork:
    """Build a PEPS (2D tensor network) as a TensorNetwork.

    Tensors are organized in a 2D grid ``tensors[i][j]`` for row *i*,
    column *j*.  Horizontal and vertical neighbours are connected wherever
    they share a common label with compatible dimensions.

    Args:
        tensors: 2D list [Lx][Ly] of site tensors.
        Lx:      Number of rows.
        Ly:      Number of columns.

    Returns:
        TensorNetwork with virtual bonds connected.
    """
    if len(tensors) != Lx:
        raise ValueError(f"tensors has {len(tensors)} rows but Lx={Lx}")
    for i, row in enumerate(tensors):
        if len(row) != Ly:
            raise ValueError(f"tensors[{i}] has {len(row)} columns but Ly={Ly}")

    tn = TensorNetwork(name="PEPS")

    # Add all nodes
    for i in range(Lx):
        for j in range(Ly):
            tn.add_node((i, j), tensors[i][j])

    # Connect horizontal neighbors
    for i in range(Lx):
        for j in range(Ly - 1):
            labels_ij = set(tensors[i][j].labels())
            labels_next = set(tensors[i][j + 1].labels())
            shared = labels_ij & labels_next
            for label in sorted(shared, key=str):
                tn.connect((i, j), label, (i, j + 1), label)

    # Connect vertical neighbors
    for i in range(Lx - 1):
        for j in range(Ly):
            labels_ij = set(tensors[i][j].labels())
            labels_next = set(tensors[i + 1][j].labels())
            shared = labels_ij & labels_next
            for label in sorted(shared, key=str):
                tn.connect((i, j), label, (i + 1, j), label)

    return tn
