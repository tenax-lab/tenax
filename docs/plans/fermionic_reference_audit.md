# Fermionic Reference Audit

Date: 2026-03-10

## Scope

This audit checks the current Tenax fermion implementation against four
reference libraries:

- ITensor
- quimb
- TeNPy
- TensorKit

The goal is not to rank the projects globally. The narrower question is:
how complete and internally consistent is Tenax's current fermion stack,
and where does it still lag behind established reference implementations?

## Current Tenax Status

The older internal fermion audit is no longer a good description of the
current tree. The following pieces are implemented today:

- Spinless fermion single-site operators exist in
  [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py#L70).
- Automatic Jordan-Wigner string insertion for fermionic AutoMPO terms
  exists in [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py#L570).
- `SymmetricTensor.dagger()` uses the super-algebra pairwise parity rule in
  [tensor.py](/home/yjkao/tenax/src/tenax/core/tensor.py#L954).
- fPEPS comments now describe graded/Koszul-sign handling rather than swap
  gates in
  [fermionic_ipeps.py](/home/yjkao/tenax/src/tenax/algorithms/fermionic_ipeps.py#L401).
- Dense cross-validation tests for fermionic block-sparse contraction exist in
  [test_fermionic.py](/home/yjkao/tenax/tests/test_fermionic.py#L869).
- Fermionic coverage also exists for CTM, split-CTM, TRG, HOTRG, and fPEPS
  in:
  [test_ctm_tensor.py](/home/yjkao/tenax/tests/test_ctm_tensor.py#L320),
  [test_split_ctm_tensor.py](/home/yjkao/tenax/tests/test_split_ctm_tensor.py#L344),
  [test_trg.py](/home/yjkao/tenax/tests/test_trg.py#L443),
  [test_hotrg.py](/home/yjkao/tenax/tests/test_hotrg.py#L347),
  [test_fermionic_ipeps.py](/home/yjkao/tenax/tests/test_fermionic_ipeps.py#L1).

Local verification:

- `uv run pytest -m core tests/test_fermionic.py -q` passed locally on
  2026-03-10.

## Main Comparison

### 1. Versus ITensor

Assessment:

- Tenax is now much closer to ITensor on the specific point of building
  fermionic MPOs from operator strings.
- Tenax is still behind ITensor in overall 1D fermion workflow maturity.

Why:

- ITensor has built-in fermionic site types such as `"Fermion"` and
  `"Electron"` and its AutoMPO / OpSum pipeline recognizes those operator
  families directly.
- ITensor documents that AutoMPO handles Jordan-Wigner rewriting
  automatically for fermionic Hamiltonian terms, but fermionic strings still
  need to be inserted manually in post-DMRG measurements of nonlocal
  correlators.
- Tenax now matches the AutoMPO-side expectation for spinless fermions via
  `fermion_site_ops()` and `_insert_jw_strings()`, but it does not yet have
  an equally mature site/model abstraction layer or the same breadth of
  standard fermionic measurement workflows.

Verdict:

- Tenax is no longer missing the core ITensor-style AutoMPO fermion feature.
- Tenax still trails ITensor in ecosystem depth, especially around
  ready-made site families, mature measurement patterns, and long-polished
  1D workflows.

### 2. Versus TeNPy

Assessment:

- Tenax now overlaps with TeNPy's basic 1D fermion handling, but TeNPy still
  has a more explicit and mature fermion-site API.

Why:

- TeNPy exposes `FermionSite`, `SpinHalfFermionSite`, and helpers such as
  `spin_half_species(...)`.
- TeNPy explicitly tracks whether an operator needs a Jordan-Wigner string
  (`op_needs_JW`) and how charges map to Jordan-Wigner parity
  (`charge_to_JW_signs`).
- Tenax has the lower-level ingredients for fermionic MPO construction and
  graded tensor algebra, but it does not yet expose an equally rich
  site-centric fermion model layer.

Verdict:

- Tenax is competitive on the narrow point of fermionic MPO construction.
- TeNPy remains ahead in ergonomic 1D model-building abstractions.

### 3. Versus quimb

Assessment:

- Tenax is more opinionated and more built-in on symmetry-aware tensor
  objects.
- quimb is broader and more composable at the array/backend layer.

Why:

- quimb documents support for symmetries and fermions through `symmray`, and
  its tensor array layer exposes `isfermionic(...)`.
- quimb also has experimental operator builders for fermionic lattice models
  such as Fermi-Hubbard and spinless Fermi-Hubbard.
- By contrast, Tenax's fermion support is implemented directly in its own
  `SymmetricTensor` / symmetry stack rather than delegated to a separate
  external array package.

Verdict:

- Tenax currently looks stronger if the question is "does the library itself
  own a coherent graded block-sparse tensor abstraction?"
- quimb looks stronger if the question is "how flexible is the surrounding
  tensor-network and backend ecosystem?"

### 4. Versus TensorKit

Assessment:

- TensorKit is substantially more expressive than Tenax on the mathematical
  side of fermionic tensor categories.

Why:

- TensorKit documents explicit braiding styles, fermionic twists, and
  braiding/permutation operations at the sector and fusion-tree level.
- Tenax implements graded signs inside tensor operations and a fermionic
  twist in `dagger()`, but it does not currently expose TensorKit-level
  categorical primitives such as explicit braiding of fusion structures.

Verdict:

- Tenax should not be described as TensorKit-level fermionic support.
- Tenax is better characterized as a practical graded block-sparse tensor
  implementation, not a full categorical fermionic tensor framework.

## Confirmed Gaps In Tenax

These are the current high-confidence gaps after comparing the codebase and
tests against the reference libraries.

1. **Split-CTM is not yet fully symmetric end to end.**
   The symmetric split-CTM path still falls back to dense conversions in
   several places, including initialization, projector construction, and
   energy evaluation. Verified dense fallback sites include
   [src/tenax/algorithms/_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L172),
   [src/tenax/algorithms/_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L528),
   [src/tenax/algorithms/_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L807),
   and
   [src/tenax/algorithms/_split_ctm_tensor.py](/home/yjkao/tenax/src/tenax/algorithms/_split_ctm_tensor.py#L1292).

2. **The `dagger()` docstring is still stale.**
   The implementation in
   [tensor.py](/home/yjkao/tenax/src/tenax/core/tensor.py#L954)
   uses the corrected pairwise parity formula, but the docstring still says
   it multiplies by the product of twist phases for all charges in the block
   key. That is a documentation inconsistency in a central fermionic API.

3. **Tenax lacks a higher-level fermion site/model layer comparable to
   ITensor and TeNPy.**
   The core tensor algebra is stronger than before, but user-facing fermion
   ergonomics are still relatively low-level.

4. **The strongest no-dense-fallback guarantee exists for standard CTM, not
   split-CTM.**
   There is already a standard CTM regression test exercising fermionic CTM
   on `SymmetricTensor` in
   [test_ctm_tensor.py](/home/yjkao/tenax/tests/test_ctm_tensor.py#L320),
   but there is no equivalent behavioral test proving that split-CTM avoids
   `todense()` / `from_dense()` on the symmetric path.

## Overall Verdict

The current Tenax fermion implementation is materially better than the old
internal audit implies.

- Compared with ITensor and TeNPy, Tenax has caught up on the core
  fermionic-AutoMPO/Jordan-Wigner feature set for simple Hamiltonian
  construction.
- Compared with quimb, Tenax has a more self-contained graded
  block-sparse tensor story, but a narrower surrounding ecosystem.
- Compared with TensorKit, Tenax is still far less expressive on explicit
  braiding/twist/category-theoretic structure.

So the right current description is:

- **Tenax fermions are implemented and tested at the practical tensor and
  Hamiltonian-construction level.**
- **Tenax fermions are not yet as mature as ITensor/TeNPy in end-user 1D
  workflow ergonomics.**
- **Tenax fermions are not yet as mathematically complete as TensorKit.**
- **The main internal technical gap is the partially dense symmetric
  split-CTM path, not missing Jordan-Wigner support.**

## Sources

Tenax local sources:

- [auto_mpo.py](/home/yjkao/tenax/src/tenax/algorithms/auto_mpo.py)
- [tensor.py](/home/yjkao/tenax/src/tenax/core/tensor.py)
- [fermionic_ipeps.py](/home/yjkao/tenax/src/tenax/algorithms/fermionic_ipeps.py)
- [tests/test_fermionic.py](/home/yjkao/tenax/tests/test_fermionic.py)
- [tests/test_ctm_tensor.py](/home/yjkao/tenax/tests/test_ctm_tensor.py)
- [tests/test_split_ctm_tensor.py](/home/yjkao/tenax/tests/test_split_ctm_tensor.py)

External references:

- ITensor AutoMPO:
  https://docs.itensor.org/ITensors/v0.1/AutoMPO.html
- ITensor support note on automatic Jordan-Wigner handling in AutoMPO:
  https://itensor.org/support/2422/jordan-wigner-fermi-string-in-self-writing-autompo
- ITensor site-type examples:
  https://docs.itensor.org/ITensorMPS/stable/examples/Physics.html
- TeNPy `FermionSite`:
  https://tenpy.readthedocs.io/en/v1.0.4/reference/tenpy.networks.site.FermionSite.html
- TeNPy `spin_half_species`:
  https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.site.spin_half_species.html
- quimb docs:
  https://quimb.readthedocs.io/en/latest/
- quimb array ops:
  https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/array_ops/index.html
- TensorKit sectors / braiding / twist:
  https://jutho.github.io/TensorKit.jl/stable/lib/sectors/
