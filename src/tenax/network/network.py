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
        since labels define the connectivity in the graph.

        Args:
            node_id: The node to update.
            tensor:  The replacement tensor.

        Raises:
            KeyError:   If node_id not found.
            ValueError: If labels differ from the original tensor.
        """
        if node_id not in self._tensors:
            raise KeyError(f"Node {node_id!r} not found")

        old_labels = set(self._tensors[node_id].labels())
        new_labels = set(tensor.labels())
        if old_labels != new_labels:
            raise ValueError(
                f"Replacement tensor has different labels. "
                f"Old: {sorted(old_labels)}, New: {sorted(new_labels)}"
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

        self._graph.add_edge(node_a, node_b, label_a=label_a, label_b=label_b)
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

        count = 0
        for label in sorted(shared, key=str):
            self.connect(node_a, label, node_b, label)
            count += 1

        return count

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
            if (
                v == node_b
                and data.get("label_a") == label_a
                and data.get("label_b") == label_b
            ):
                self._graph.remove_edge(u, v, key)
                self._invalidate_cache()
                return
            # Also check reversed direction
            if (
                v == node_a
                and u == node_b
                and data.get("label_a") == label_b
                and data.get("label_b") == label_a
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
            KeyError: If node not found or old_label not in tensor.
        """
        tensor = self._tensors[node_id]
        self._tensors[node_id] = tensor.relabel(old_label, new_label)

        # Update any edges that reference this label
        for u, v, key, data in list(self._graph.edges(node_id, data=True, keys=True)):
            if data.get("label_a") == old_label and u == node_id:
                self._graph[u][v][key]["label_a"] = new_label
            if data.get("label_b") == old_label and v == node_id:
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
            if u == node_id:
                connected_labels.add(data.get("label_a"))
            if v == node_id:
                connected_labels.add(data.get("label_b"))

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
            if u in node_set and v in node_set:
                la = data.get("label_a")
                lb = data.get("label_b")
                internal_edges.append((u, la, v, lb))

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
    open_boundary: bool = True,
) -> TensorNetwork:
    """Build a Matrix Product State as a TensorNetwork.

    Tensors are expected to have legs labeled with a convention that
    allows adjacent tensors to share virtual bond labels. If the tensors
    already have matching virtual bond labels (e.g., right label of site i
    matches left label of site i+1), they will be auto-connected.

    If virtual bond labels don't match, default virtual bonds are created:
    - Site i: left="v{i-1}_{i}", phys="p{i}", right="v{i}_{i+1}"

    Physical legs are not connected (they remain open).

    Args:
        tensors:        List of site tensors [A_0, A_1, ..., A_{L-1}].
        open_boundary:  If True, boundary virtual legs remain open.

    Returns:
        TensorNetwork with virtual bonds connected.
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

        if shared:
            for label in sorted(shared, key=str):
                try:
                    tn.connect(i, label, i + 1, label)
                except ValueError:
                    pass  # incompatible dimensions, skip

    return tn


def build_peps(
    tensors: list[list[Tensor]],
    Lx: int,
    Ly: int,
    open_boundary: bool = True,
) -> TensorNetwork:
    """Build a PEPS (2D tensor network) as a TensorNetwork.

    Tensors are organized in a 2D grid tensors[i][j] for row i, column j.
    Adjacent tensors are connected by shared virtual bond labels.

    Convention for virtual bond labels (if not already matching):
    - Horizontal bond (j, j+1) in row i: "h{i}_{j}_{j+1}"
    - Vertical bond row (i, i+1) in column j: "v{i}_{i+1}_{j}"
    - Physical leg at (i,j): "p{i}_{j}"

    Args:
        tensors:       2D list [Lx][Ly] of site tensors.
        Lx:            Number of rows.
        Ly:            Number of columns.
        open_boundary: If True, boundary virtual legs remain open.

    Returns:
        TensorNetwork with virtual bonds connected.
    """
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
                try:
                    tn.connect((i, j), label, (i, j + 1), label)
                except ValueError:
                    pass

    # Connect vertical neighbors
    for i in range(Lx - 1):
        for j in range(Ly):
            labels_ij = set(tensors[i][j].labels())
            labels_next = set(tensors[i + 1][j].labels())
            shared = labels_ij & labels_next
            for label in sorted(shared, key=str):
                try:
                    tn.connect((i, j), label, (i + 1, j), label)
                except ValueError:
                    pass

    return tn
