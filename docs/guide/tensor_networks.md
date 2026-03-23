# Tensor Networks

The `TensorNetwork` class is a graph-based container where **nodes** hold
tensors and **edges** record which leg labels are connected. It provides
caching, automatic label-based connectivity, and convenience builders for
MPS and PEPS.

## Creating a network

```python
from tenax import TensorNetwork

tn = TensorNetwork(name="my_network")

# Add tensors as nodes
tn.add_node("A", tensor_A)
tn.add_node("B", tensor_B)

# Connect legs with the same label automatically
tn.connect_by_shared_label("A", "B")

# Or connect specific legs explicitly
tn.connect("A", "right", "B", "left")
```

## Querying the network

```python
tn.node_ids()          # ["A", "B"]
tn.n_nodes()           # 2
tn.n_edges()           # number of connected leg pairs

tn.get_tensor("A")     # retrieve the stored tensor
tn.open_legs("A")      # labels of legs not connected to anything
tn.neighbors("A")      # ["B"]
tn.is_connected()      # True if the graph is connected
```

## Contraction

Contract all (or a subset of) nodes:

```python
# Contract the entire network
result = tn.contract()

# Contract a subset
result = tn.contract(nodes=["A", "B"])

# Specify output label order
result = tn.contract(output_labels=("phys_A", "phys_B"))
```

Results are cached by the set of contracted nodes. The cache is
automatically invalidated when the network structure changes.

## Modifying the network

```python
# Replace a tensor (must keep the same labels)
tn.replace_tensor("A", new_tensor_A)

# Remove a node and all its edges
old_tensor = tn.remove_node("A")

# Rename a leg and update all connected edges
tn.relabel_bond("B", "left", "input")

# Disconnect specific legs
tn.disconnect("A", "right", "B", "left")

# Manually clear the contraction cache
tn.clear_cache()
```

## Building an MPS

`build_mps` creates a `TensorNetwork` from a list of site tensors and
auto-connects adjacent sites by their shared virtual bond labels.

```python
from tenax import build_mps

mps = build_mps(site_tensors)
# Nodes are labelled 0, 1, ..., L-1
# Virtual bonds v{i}_{i+1} are connected automatically
```

Label convention for MPS site tensors (all sites are 3-leg):

| Site | Left bond | Physical | Right bond |
|------|-----------|----------|------------|
| 0 | `v_-1_0` (dim=1) | `p0` | `v0_1` |
| i | `v{i-1}_{i}` | `p{i}` | `v{i}_{i+1}` |
| L-1 | `v{L-2}_{L-1}` | `p{L-1}` | `v{L-1}_{L}` (dim=1) |

Boundary sites have explicit trivial dimension-1 bonds rather than omitting
the virtual leg. This makes all MPS tensors uniformly 3-leg.

## Building a PEPS

`build_peps` creates a 2D tensor network from a grid of site tensors:

```python
from tenax import build_peps

peps = build_peps(tensor_grid, Lx=3, Ly=3)
# Nodes are labelled (i, j)
# Horizontal and vertical bonds are connected automatically
```

Label convention for PEPS site tensors:

- Horizontal bond (column j to j+1 in row i): `h{i}_{j}_{j+1}`
- Vertical bond (row i to i+1 in column j): `v{i}_{i+1}_{j}`
- Physical leg at (i, j): `p{i}_{j}`

## Mermaid diagram export

`TensorNetwork.to_mermaid()` returns a [Mermaid](https://mermaid.js.org/)
graph string for visualization. Contracted edges are shown as solid lines
with bond labels; open (free) legs are shown as dangling circle nodes.

```python
tn = TensorNetwork(name="MPS")
tn.add_node("A", tensor_A)
tn.add_node("B", tensor_B)
tn.connect("A", "bond", "B", "bond")

print(tn.to_mermaid())
```

```
graph LR
  A["A (2,3)"]
  B["B (3,4)"]
  A ---|bond| B
  A -.- A_phys(("phys"))
  B -.- B_phys(("phys"))
```

Paste the output into any Mermaid renderer (GitHub markdown, Mermaid Live
Editor, or VS Code preview) to get an interactive diagram.
