---
name: tenax-blueprint
description: >
  Teach users to design tensor network contractions using Tenax's
  NetworkBlueprint and TensorNetwork classes. Translates tensor network diagrams
  (described verbally or visually) into .net topology strings, then shows how to
  load tensors and contract efficiently. Use this skill when the user asks about
  NetworkBlueprint, TensorNetwork, .net files, tensor network contraction,
  contraction order, opt_einsum, "how to contract a network", "define a tensor
  network topology", or wants to build custom algorithms beyond DMRG/TRG/iPEPS.
  Also trigger for DMRG environment contractions, transfer matrices, expectation
  values, or any multi-tensor contraction pattern.
---

# Network Blueprint Designer — Custom Tensor Network Contractions in Tenax

This skill teaches users to design and execute custom tensor network contractions
using Tenax's `NetworkBlueprint` (template pattern) and `TensorNetwork` (graph-
based) interfaces.

## When to Use This

Tenax's built-in algorithms (DMRG, TRG, iPEPS) handle standard calculations.
Use NetworkBlueprint or TensorNetwork when you need:

- Custom contraction patterns (e.g., DMRG effective Hamiltonian, transfer matrix)
- Reusable contraction templates (same topology, different tensors)
- Explicit control over contraction order
- Multi-tensor contractions beyond simple pairwise `contract(A, B)`

## The Two Interfaces

### TensorNetwork — graph-based, interactive

Build the network node by node, connect edges, contract.

```python
from tenax import TensorNetwork

tn = TensorNetwork()
tn.add_node("A", A)  # A is a SymmetricTensor or DenseTensor
tn.add_node("B", B)
tn.connect_by_shared_label("A", "B")  # Contracts legs with matching labels
result = tn.contract()
```

Best for: exploratory work, one-off contractions.

### NetworkBlueprint — template pattern, reusable

Define the topology once (as a string or .net file), then load different tensors
and contract repeatedly. This is the cytnx-style approach.

```python
from tenax import NetworkBlueprint, from_netfile

bp = NetworkBlueprint("""
L: a, b, c
M: a, p, q, d
A: b, p, s, e
M2: e, q, t, f
R: d, f, g
TOUT: c, s, t, g
""")

bp.put_tensors({"L": L, "M": M, "A": A, "M2": M2, "R": R})
result = bp.launch()

# Reuse with different tensors (e.g., in a DMRG sweep)
bp.put_tensor("A", new_A)
result2 = bp.launch()
```

Each tensor's labels must equal its node's leg names (as a set); otherwise
`put_tensor` raises `ValueError`. Pass `label_order=[...]` to
`put_tensor(name, tensor, label_order=...)` to map a tensor's own labels onto
the node's legs positionally.

Best for: inner loops (DMRG sweeps, CTM iterations), where the topology is fixed
but tensors change every step.

You can also load a `.net` file from disk:
```python
bp = from_netfile("path/to/network.net")
```

## .net File Format

The topology string is a simple declarative format:

```
NodeName: leg1, leg2, leg3, ...
```

- Each line defines a tensor node and its legs.
- Legs with the **same name across different nodes** are contracted.
- Legs that appear in only one node are **open** (uncontracted).
- `TOUT:` is a special line that specifies the **output tensor's leg order**.

### Rules

1. **Every leg name that appears twice** is contracted (summed over).
2. **Leg names appearing once** are open and must appear in `TOUT:`.
3. **`TOUT:` is required** — it defines the output tensor shape and leg ordering.
4. Leg names are arbitrary strings (no spaces, typically single letters).

### Example: MPS-MPO-MPS contraction (expectation value)

```
#  bra_A ---[bra]--- bra_B
#    |         |        |
#   [M_left]--[W]--[M_right]
#    |         |        |
#  ket_A ---[ket]--- ket_B
#

ket_A: vL, p, a          # MPS ket tensor A
W: wL, p, q, wR          # MPO tensor
bra_A: vL_bar, q, a_bar  # MPS bra tensor A (conjugate)
L: a, wL, a_bar          # Left environment
R: b, wR, b_bar          # Right environment
ket_B: a, s, b           # MPS ket tensor B
bra_B: a_bar, s_bar, b_bar  # MPS bra tensor B
TOUT: vL, vL_bar, s, s_bar  # Open legs
```

### Example: Simple two-tensor contraction

```
A: i, j, k
B: k, l, m
TOUT: i, j, l, m
```

Here `k` appears in both A and B, so it's contracted. The output has legs
`(i, j, l, m)` in that order.

## Design Process

When the user describes a contraction:

1. **Identify all tensors** and their ranks (number of legs).
2. **Name each leg** — use descriptive names (physical, bond_L, bond_R, etc.)
   or short labels (a, b, c) for complex networks.
3. **Identify shared legs** — which pairs of legs should be contracted?
   Give them the same name.
4. **Identify open legs** — these appear in `TOUT:`.
5. **Write the topology string.**
6. **Verify:** count legs per node = tensor rank. Count shared names =
   number of contractions. Open legs in TOUT = output rank.

## Common Patterns

### DMRG effective Hamiltonian (two-site)

The effective Hamiltonian for two-site DMRG contracts the left environment,
two MPO tensors, and the right environment around the two-site block:

```
L: a, wL, a_bar
W1: wL, p, q, wM
W2: wM, s, t, wR
R: b, wR, b_bar
TOUT: a, p, s, b, a_bar, q, t, b_bar
```

This gives the effective Hamiltonian as a tensor with 8 legs (4 ket + 4 bra).

### Transfer matrix (for correlation functions)

```
ket: vL, p, vR
bra: vL_bar, p, vR_bar
TOUT: vL, vL_bar, vR, vR_bar
```

Here `p` is the physical leg, contracted between ket and bra (trace over
physical index). The result is a transfer matrix in the bond space.

### CTM environment update

```
C: a, b
T: b, c, d
TOUT: a, c, d
```

Contract a corner tensor C with an edge tensor T to grow the environment.

## Optimization

`NetworkBlueprint` uses opt_einsum integration for optimal contraction path
finding. For very large networks, the path optimizer runs once at `launch()`
time and caches the result. Subsequent calls with different tensors (via
`put_tensor`) reuse the same contraction path.

This makes `NetworkBlueprint` ideal for inner loops where the same topology
is contracted thousands of times with different tensor data.

## Pedagogical Notes

- Drawing the tensor network diagram first (on paper or whiteboard) before
  writing the .net string is always a good idea.
- The `TOUT:` line is like specifying the "external legs" of a Feynman diagram
  — it defines what you're computing.
- Connect to physics: every contraction is a sum over internal degrees of
  freedom (like integrating out virtual particles in QFT, or tracing over
  bath degrees of freedom in open quantum systems).
