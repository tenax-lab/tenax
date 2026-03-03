# Tenax

**JAX-based tensor network library with symmetry-aware block-sparse tensors.**

Tenax provides label-based tensor contraction (Cytnx-style), abelian symmetry
support (U(1), Z_n), and implementations of DMRG, TRG, HOTRG,
and iPEPS algorithms.

```{admonition} Experimental project
:class: warning

This library is under active development and largely written with the assistance
of Claude Code (AI). While we test extensively, AI-generated code can contain
subtle bugs. Please verify results against known benchmarks before using them in
research. Bug reports and contributions are welcome.
```

## Key Features

- **Label-based contraction** -- shared labels between tensors are automatically contracted; no manual einsum subscripts needed.
- **Symmetry-aware tensors** -- `SymmetricTensor` stores only symmetry-allowed charge sectors, reducing memory and FLOPs.
- **JAX integration** -- all tensor types are registered as JAX pytrees for seamless `jit`, `grad`, and `vmap`.
- **Optimised contraction paths** -- `opt_einsum` finds the best contraction order before JAX executes.
- **Batteries-included algorithms** -- DMRG, TRG, HOTRG, iPEPS with simple configuration dataclasses.

## Getting Started

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/installation
guide/quickstart
guide/core_concepts
guide/contraction
guide/tensor_networks
guide/gotchas
```

## Algorithm Tutorials

```{toctree}
:maxdepth: 2
:caption: Algorithms

guide/algorithms/dmrg
guide/algorithms/idmrg
guide/algorithms/trg
guide/algorithms/hotrg
guide/algorithms/ipeps
guide/algorithms/ad_excitations
guide/algorithms/auto_mpo
```

## Tools

```{toctree}
:maxdepth: 2
:caption: Tools

guide/benchmarks
```

## Reference

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

## Additional Resources

```{toctree}
:maxdepth: 1
:caption: Notes

contraction_semantics
```
