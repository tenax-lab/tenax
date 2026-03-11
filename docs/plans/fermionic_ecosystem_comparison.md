# Fermionic Tensor-Network Implementation Comparison

Date: 2026-03-11

## Scope

This document compares Tenax's fermionic tensor-network implementation with:

- TeNPy
- ITensor
- TensorKit.jl
- MPToolkit
- peps-torch
- YASTN
- quimb

The focus is implementation behavior and user-facing workflow, not raw
performance benchmarking.

## Quick Matrix

| Library | Fermionic Core Model | Jordan-Wigner Handling | Symmetry / Block-Sparse | 2D / PEPS Fermions | Overall Maturity |
|---|---|---|---|---|---|
| **Tenax** | Graded signs in core tensor ops + fermionic symmetries (`FermionParity`, `FermionicU1`) | AutoMPO supports explicit fermionic op sets and auto JW string insertion | Built-in `SymmetricTensor` + fermionic grading | fPEPS path exists; split-CTM still has dense fallback points | Strong and coherent, but still evolving |
| **TeNPy** | Site-driven fermionic model (`FermionSite`, `SpinHalfFermionSite`) | Explicit metadata for JW needs (`op_needs_JW`, `charge_to_JW_signs`) | Mature U(1)/symmetry MPS ecosystem | Primarily 1D-first; JW mapping caveats for higher-D orderings are documented | Very mature in 1D workflows |
| **ITensor** | Rich fermionic site types (`Fermion`, `Electron`) + `OpSum`/MPO workflow | AutoMPO/OpSum handles fermionic Hamiltonian construction with JW conventions | Strong QN-based tensor infrastructure | Strong MPS/MPO core; 2D via mappings/algorithms rather than native PEPS focus | Highly mature and battle-tested |
| **TensorKit.jl** | Categorical tensor formalism with fermionic braiding/twist semantics | Not centered on AutoMPO-style JW convenience; more algebra-first | Very expressive sector/braiding/twist formalism | Depends on higher-level packages; strongest at formal tensor layer | Highest mathematical expressiveness |
| **MPToolkit** | Long-running MPS/DMRG toolkit with symmetry-heavy focus | Explicit string/product-operator workflows used in JW-style contexts | Strong internal + non-Abelian symmetry support | 1D/iDMRG-centric design | Very mature for symmetry-heavy 1D |
| **peps-torch** | PEPS optimization framework (CTM/optimization-centric) | Public docs do not prominently expose first-class fermionic operator/JW API | Supports abelian-symmetric iPEPS workflows | Strong 2D PEPS tooling; explicit fermionic API less visible in docs | Mature for PEPS numerics, less explicit on fermion API surface |
| **YASTN** | Symmetric tensor core + dedicated `fpeps` module with fermionic rules | Handles fermionic PEPS ordering with parity-preserving tensors and swap gates | Strong abelian-symmetric block-sparse tensor design | Native fPEPS stack with CTMRG/NTU/boundary environments | Mature and technically deep for symmetric TNs |
| **quimb** | Flexible TN framework; fermionic capabilities via ecosystem (`symmray`, fermionic arrays/builders) | Fermionic model/operator support exists, including experimental builders | Broad ecosystem flexibility; symmetry/fermion support via modular layers | Supports PEPS and fermionic construction utilities | Very broad, composable ecosystem |

## Tenax: Current Implementation Snapshot

### Implemented strengths

- Spinless fermion operator set is built in (`C`, `Cd`, `N`, `F`, `Id`) in
  [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py#L70).
- AutoMPO inserts Jordan-Wigner `F` strings when `fermionic_ops` is provided in
  [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py#L562).
- Core graded symmetries are first class (`FermionParity`, `FermionicU1`) in
  [symmetry.py](/home/yjkao/tenax/src/tenax/core/symmetry.py#L311).
- Fermionic sign handling is active in contraction logic in
  [contractor.py](/home/yjkao/tenax/src/tenax/contraction/contractor.py#L362).
- `SymmetricTensor.dagger()` uses the super-algebra sign convention in
  [tensor.py](/home/yjkao/tenax/src/tenax/core/tensor.py#L954).
- Fermionic fPEPS + tensor-protocol CTM path is available in
  [fermionic_ipeps.py](/home/yjkao/tenax/src/tenax/algorithms/fermionic_ipeps.py#L408).

### Confirmed gap

- Split-CTM energy still converts through dense environment reconstruction in
  [\_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L1292)
  and
  [\_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L1316).

## Library-by-Library Notes

### Tenax vs TeNPy

- **Where TeNPy leads:** site-model ergonomics and explicit JW metadata.
- **Where Tenax leads:** integrated graded tensor semantics in the core tensor
  object model, not only at the site/model layer.

### Tenax vs ITensor

- **Where ITensor leads:** mature, end-to-end fermionic MPS workflow depth.
- **Where Tenax is competitive:** fermionic AutoMPO construction, JW insertion,
  and block-sparse graded core are now present and tested.

### Tenax vs TensorKit.jl

- **Where TensorKit leads:** formal fermionic braiding/twist/category
  expressiveness.
- **Where Tenax leads:** practical Python/JAX integration and simpler API for
  common workflow tasks.

### Tenax vs MPToolkit

- **Where MPToolkit leads:** long-horizon maturity in symmetry-heavy 1D/iDMRG
  and explicit string-operator workflows.
- **Where Tenax leads:** modern single-stack implementation coherence (JAX +
  tensor/symmetry APIs in one codebase).

### Tenax vs peps-torch

- **Where peps-torch leads:** 2D PEPS optimization pipelines and CTM-centric
  numerics.
- **Where Tenax leads:** clearly documented in-core fermionic primitives
  (operator/JW/symmetry/sign semantics) in the primary library API.
- **Evidence caveat:** public peps-torch docs/README emphasize spin/Abelian
  iPEPS workflows; explicit first-class fermionic API is less visible there.

### Tenax vs YASTN

- **Where YASTN leads:** explicit, documented fPEPS formalism for fermionic
  anticommutation (parity-preserving tensors + swap gates) and a mature
  symmetric tensor core tuned for abelian charges.
- **Where Tenax leads:** a simpler single-library API surface for integrating
  fermionic symmetry objects, contraction rules, and AutoMPO-style JW term
  building in one codebase.

### Tenax vs quimb

- **Where quimb leads:** breadth and composability of ecosystem-level tools.
- **Where Tenax leads:** tighter, built-in graded block-sparse semantics in the
  core tensor implementation.

## Practical Bottom Line

1. Tenax is now solid on core fermionic tensor semantics and Jordan-Wigner-aware
   MPO construction.
2. TeNPy/ITensor/MPToolkit still set the bar for polished 1D production
   workflows.
3. TensorKit.jl remains ahead on mathematical fermionic tensor expressiveness.
4. YASTN is the closest reference for a deeply documented fermionic PEPS
   formalism in a symmetry-first tensor framework.
5. Tenax's most concrete internal technical gap is still removing dense fallback
   from split-CTM fermionic/symmetric paths.

## Sources

Tenax local code:

- [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py)
- [symmetry.py](/home/yjkao/tenax/src/tenax/core/symmetry.py)
- [tensor.py](/home/yjkao/tenax/src/tenax/core/tensor.py)
- [contractor.py](/home/yjkao/tenax/src/tenax/contraction/contractor.py)
- [fermionic_ipeps.py](/home/yjkao/tenax/src/tenax/algorithms/fermionic_ipeps.py)
- [\_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py)
- [test_fermionic.py](/home/yjkao/tenax/tests/test_fermionic.py)
- [test_fermionic_ipeps.py](/home/yjkao/tenax/tests/test_fermionic_ipeps.py)
- [test_ctm_tensor.py](/home/yjkao/tenax/tests/test_ctm_tensor.py)
- [test_split_ctm_tensor.py](/home/yjkao/tenax/tests/test_split_ctm_tensor.py)

External references:

- TeNPy `FermionSite`:
  https://tenpy.readthedocs.io/en/v1.0.6/reference/tenpy.networks.site.FermionSite.html
- TeNPy Jordan-Wigner notes:
  https://tenpy.readthedocs.io/en/v1.0.6/intro/JordanWigner.html
- ITensor included site types:
  https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html
- ITensor `OpSum`:
  https://docs.itensor.org/ITensorMPS/stable/OpSum.html
- TensorKit sectors / braiding / twist:
  https://jutho.github.io/TensorKit.jl/stable/lib/sectors/
- MPToolkit About:
  https://mptoolkit.qusim.net/Main/About
- MPToolkit iDMRG overview:
  https://mptoolkit.qusim.net/IDMRG/Overview
- MPToolkit basis/symmetry example:
  https://mptoolkit.qusim.net/Tools/MpReorderBasis
- peps-torch repository:
  https://github.com/jurajHasik/peps-torch
- peps-torch docs intro:
  https://jurajhasik.github.io/peps-torch/intro.html
- YASTN docs home:
  https://yastn.github.io/yastn/
- YASTN fPEPS module:
  https://yastn.github.io/yastn/yastn.fpeps.html
- quimb repository:
  https://github.com/jcmgray/quimb
- quimb tensor array ops (`isfermionic`):
  https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/array_ops/index.html
- quimb experimental operator builder:
  https://quimb.readthedocs.io/en/latest/autoapi/quimb/experimental/operatorbuilder/index.html
- YASTN fPEPS basics:
  https://yastn.github.io/yastn/theory/fpeps/basics.html
