"""Tenax: JAX-based tensor network library with symmetry-aware block-sparse tensors.

Label-based contraction (Cytnx-style):
    Tensor legs carry string or integer labels. Two legs with the same label
    across different tensors are automatically contracted when contract() is called.

.. note::
    Importing ``tenax`` enables JAX 64-bit mode (``jax_enable_x64``).
    All tensors and algorithms default to ``float64``.

    Algorithm modules (dmrg, ipeps, trg, etc.) are loaded lazily on first
    access to keep ``import tenax`` fast.  Core data structures (Tensor,
    TensorIndex, symmetry types, MPS, contraction, linalg, lattice, network)
    are imported eagerly.

Quick start::

    import jax
    import numpy as np
    from tenax import (
        U1Symmetry, TensorIndex, FlowDirection,
        SymmetricTensor, TensorNetwork, contract
    )

    u1 = U1Symmetry()
    key = jax.random.PRNGKey(0)

    A = SymmetricTensor.random_normal(
        indices=(
            TensorIndex.from_charges(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN,  label="phys"),
            TensorIndex.from_charges(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN,  label="left"),
            TensorIndex.from_charges(u1, np.array([1, 0, -1], dtype=np.int32), FlowDirection.OUT, label="bond"),
        ),
        key=key,
    )
    B = A.relabel("bond", "right").relabel("left", "bond")
    result = contract(A, B)   # contracts over shared label "bond"
    print(result.labels())    # ('phys', 'left', 'phys', 'right')
"""

import jax

# Enable 64-bit mode globally so all Tenax tensors default to float64.
# This must run at import time before any JAX computation is triggered.
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Eager imports: core data structures, contraction, linalg, lattice, network
# ---------------------------------------------------------------------------

from tenax.contraction.contractor import (
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)
from tenax.core.index import FlowDirection, FuseInfo, Label, TensorIndex
from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)
from tenax.core.mps import FiniteMPS, InfiniteMPS
from tenax.core.symmetry import (
    BaseNonAbelianSymmetry,
    BaseSymmetry,
    BraidingStyle,
    FermionicU1,
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import BlockKey, DenseTensor, SymmetricTensor, Tensor, inner
from tenax.linalg import eigh, qr, rsvd, svd
from tenax.network.netfile import NetworkBlueprint, from_netfile
from tenax.network.network import TensorNetwork, build_mps, build_peps

__version__ = "0.5.0"

# ---------------------------------------------------------------------------
# Lazy imports: algorithm modules are loaded on first access
# ---------------------------------------------------------------------------

# Maps attribute name -> (module_path, attribute_name_in_module)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # _ctm_tensor
    "CTMTensorEnv": ("tenax.algorithms._ctm_tensor", "CTMTensorEnv"),
    "compute_energy_ctm_tensor": (
        "tenax.algorithms._ctm_tensor",
        "compute_energy_ctm_tensor",
    ),
    "compute_energy_ctm_tensor_2site": (
        "tenax.algorithms._ctm_tensor",
        "compute_energy_ctm_tensor_2site",
    ),
    "ctm_tensor": ("tenax.algorithms._ctm_tensor", "ctm_tensor"),
    "ctm_tensor_2site": ("tenax.algorithms._ctm_tensor", "ctm_tensor_2site"),
    "ctm_tensor_c4v": ("tenax.algorithms._ctm_tensor", "ctm_tensor_c4v"),
    # _ctm_tensor_convergence
    "ctm_multisite": ("tenax.algorithms._ctm_tensor_convergence", "ctm_multisite"),
    # _krylov
    "krylov_expm": ("tenax.algorithms._krylov", "krylov_expm"),
    # _split_ctm_tensor
    "SplitCTMTensorEnv": ("tenax.algorithms._split_ctm_tensor", "SplitCTMTensorEnv"),
    "compute_energy_split_ctm_tensor": (
        "tenax.algorithms._split_ctm_tensor",
        "compute_energy_split_ctm_tensor",
    ),
    "ctm_split_tensor": ("tenax.algorithms._split_ctm_tensor", "ctm_split_tensor"),
    # _tensor_utils
    "fuse_indices": ("tenax.algorithms._tensor_utils", "fuse_indices"),
    "split_index": ("tenax.algorithms._tensor_utils", "split_index"),
    # coarse_grain
    "CGGates": ("tenax.algorithms.coarse_grain", "CGGates"),
    "honeycomb_cg_gates": ("tenax.algorithms.coarse_grain", "honeycomb_cg_gates"),
    "kagome_cg_gates": ("tenax.algorithms.coarse_grain", "kagome_cg_gates"),
    "compute_energy_cg": ("tenax.algorithms.coarse_grain", "compute_energy_cg"),
    # auto_mpo
    "AutoMPO": ("tenax.algorithms.auto_mpo", "AutoMPO"),
    "HamiltonianTerm": ("tenax.algorithms.auto_mpo", "HamiltonianTerm"),
    "build_auto_mpo": ("tenax.algorithms.auto_mpo", "build_auto_mpo"),
    "fermion_site_ops": ("tenax.algorithms.auto_mpo", "fermion_site_ops"),
    "spin_half_ops": ("tenax.algorithms.auto_mpo", "spin_half_ops"),
    "spin_one_ops": ("tenax.algorithms.auto_mpo", "spin_one_ops"),
    # dmrg
    "DMRGConfig": ("tenax.algorithms.dmrg", "DMRGConfig"),
    "DMRGResult": ("tenax.algorithms.dmrg", "DMRGResult"),
    "build_mpo_heisenberg": ("tenax.algorithms.dmrg", "build_mpo_heisenberg"),
    "build_random_mps": ("tenax.algorithms.dmrg", "build_random_mps"),
    "build_random_symmetric_mps": (
        "tenax.algorithms.dmrg",
        "build_random_symmetric_mps",
    ),
    "compute_mps_sector": ("tenax.algorithms.dmrg", "compute_mps_sector"),
    "dmrg": ("tenax.algorithms.dmrg", "dmrg"),
    "validate_mps_sector": ("tenax.algorithms.dmrg", "validate_mps_sector"),
    # fermionic_ipeps
    "FPEPSConfig": ("tenax.algorithms.fermionic_ipeps", "FPEPSConfig"),
    "fpeps": ("tenax.algorithms.fermionic_ipeps", "fpeps"),
    "spinless_fermion_gate": (
        "tenax.algorithms.fermionic_ipeps",
        "spinless_fermion_gate",
    ),
    # hotrg
    "HOTRGConfig": ("tenax.algorithms.hotrg", "HOTRGConfig"),
    "hotrg": ("tenax.algorithms.hotrg", "hotrg"),
    # idmrg
    "build_bulk_mpo_heisenberg": (
        "tenax.algorithms.idmrg",
        "build_bulk_mpo_heisenberg",
    ),
    "build_bulk_mpo_heisenberg_cylinder": (
        "tenax.algorithms.idmrg",
        "build_bulk_mpo_heisenberg_cylinder",
    ),
    "build_bulk_mpo_heisenberg_symmetric": (
        "tenax.algorithms.idmrg",
        "build_bulk_mpo_heisenberg_symmetric",
    ),
    "idmrg": ("tenax.algorithms.idmrg", "idmrg"),
    "iDMRGConfig": ("tenax.algorithms.idmrg", "iDMRGConfig"),
    "iDMRGResult": ("tenax.algorithms.idmrg", "iDMRGResult"),
    # ipeps
    "build_c4v_basis": ("tenax.algorithms.ipeps", "build_c4v_basis"),
    "c4v_coeffs_from_tensor": ("tenax.algorithms.ipeps", "c4v_coeffs_from_tensor"),
    "c4v_tensor_from_coeffs": ("tenax.algorithms.ipeps", "c4v_tensor_from_coeffs"),
    "heisenberg_gate": ("tenax.algorithms.ipeps", "heisenberg_gate"),
    "ipeps": ("tenax.algorithms.ipeps", "ipeps"),
    "sublattice_rotate_gate": ("tenax.algorithms.ipeps", "sublattice_rotate_gate"),
    "symmetrize_c4v": ("tenax.algorithms.ipeps", "symmetrize_c4v"),
    "xxz_gate": ("tenax.algorithms.ipeps", "xxz_gate"),
    # ipeps_config
    "CTMConfig": ("tenax.algorithms.ipeps_config", "CTMConfig"),
    "CTMEnvironment": ("tenax.algorithms.ipeps_config", "CTMEnvironment"),
    "SplitCTMEnvironment": ("tenax.algorithms.ipeps_config", "SplitCTMEnvironment"),
    "iPEPSConfig": ("tenax.algorithms.ipeps_config", "iPEPSConfig"),
    # ipeps_ctm
    "ctm": ("tenax.algorithms.ipeps_ctm", "ctm"),
    "ctm_2site": ("tenax.algorithms.ipeps_ctm", "ctm_2site"),
    "ctm_split": ("tenax.algorithms.ipeps_ctm", "ctm_split"),
    # ipeps_excitations
    "ExcitationConfig": ("tenax.algorithms.ipeps_excitations", "ExcitationConfig"),
    "ExcitationResult": ("tenax.algorithms.ipeps_excitations", "ExcitationResult"),
    "compute_excitations": (
        "tenax.algorithms.ipeps_excitations",
        "compute_excitations",
    ),
    "make_momentum_path": ("tenax.algorithms.ipeps_excitations", "make_momentum_path"),
    # ipeps_optimize
    "optimize_fpeps_ad": ("tenax.algorithms.ipeps_optimize", "optimize_fpeps_ad"),
    "optimize_gs_ad": ("tenax.algorithms.ipeps_optimize", "optimize_gs_ad"),
    "optimize_gs_ad_chi_schedule": (
        "tenax.algorithms.ipeps_optimize",
        "optimize_gs_ad_chi_schedule",
    ),
    # ipeps_rdm
    "compute_energy_ctm_2site": (
        "tenax.algorithms.ipeps_rdm",
        "compute_energy_ctm_2site",
    ),
    "compute_energy_split_ctm": (
        "tenax.algorithms.ipeps_rdm",
        "compute_energy_split_ctm",
    ),
    # observables
    "correlation": ("tenax.algorithms.observables", "correlation"),
    "expectation_value": ("tenax.algorithms.observables", "expectation_value"),
    "operator_charge": ("tenax.algorithms.observables", "operator_charge"),
    # tdvp
    "TDVPConfig": ("tenax.algorithms.tdvp", "TDVPConfig"),
    "TDVPResult": ("tenax.algorithms.tdvp", "TDVPResult"),
    "tdvp": ("tenax.algorithms.tdvp", "tdvp"),
    "tdvp_step": ("tenax.algorithms.tdvp", "tdvp_step"),
    # trg
    "TRGConfig": ("tenax.algorithms.trg", "TRGConfig"),
    "compute_free_wilson_fermion_tensor": (
        "tenax.algorithms.trg",
        "compute_free_wilson_fermion_tensor",
    ),
    "compute_ising_tensor": ("tenax.algorithms.trg", "compute_ising_tensor"),
    "ising_free_energy_exact": ("tenax.algorithms.trg", "ising_free_energy_exact"),
    "trg": ("tenax.algorithms.trg", "trg"),
    "wilson_fermion_free_energy_exact": (
        "tenax.algorithms.trg",
        "wilson_fermion_free_energy_exact",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module namespace so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tenax' has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # Version
    "__version__",
    # Symmetries
    "BaseSymmetry",
    "U1Symmetry",
    "ZnSymmetry",
    "BaseNonAbelianSymmetry",
    "BraidingStyle",
    "FermionParity",
    "FermionicU1",
    "ProductSymmetry",
    # Index
    "FlowDirection",
    "FuseInfo",
    "Label",
    "TensorIndex",
    # MPS
    "FiniteMPS",
    "InfiniteMPS",
    # Tensors
    "Tensor",
    "DenseTensor",
    "SymmetricTensor",
    "BlockKey",
    "inner",
    # Tensor utilities
    "fuse_indices",
    "split_index",
    # Contraction
    "contract",
    "contract_with_subscripts",
    "truncated_svd",
    "qr_decompose",
    # Linear algebra (tenax.linalg)
    "svd",
    "qr",
    "eigh",
    "rsvd",
    # AutoMPO
    "AutoMPO",
    "HamiltonianTerm",
    "build_auto_mpo",
    "spin_half_ops",
    "spin_one_ops",
    "fermion_site_ops",
    # DMRG
    "DMRGConfig",
    "DMRGResult",
    "dmrg",
    "build_mpo_heisenberg",
    "build_random_mps",
    "build_random_symmetric_mps",
    "compute_mps_sector",
    "validate_mps_sector",
    # Observables
    "expectation_value",
    "correlation",
    "operator_charge",
    # TRG
    "TRGConfig",
    "trg",
    "compute_ising_tensor",
    "compute_free_wilson_fermion_tensor",
    "ising_free_energy_exact",
    "wilson_fermion_free_energy_exact",
    # HOTRG
    "HOTRGConfig",
    "hotrg",
    # iDMRG
    "iDMRGConfig",
    "iDMRGResult",
    "idmrg",
    "build_bulk_mpo_heisenberg",
    "build_bulk_mpo_heisenberg_cylinder",
    "build_bulk_mpo_heisenberg_symmetric",
    # TDVP
    "TDVPConfig",
    "TDVPResult",
    "tdvp",
    "tdvp_step",
    "krylov_expm",
    # iPEPS
    "iPEPSConfig",
    "CTMConfig",
    "CTMEnvironment",
    "SplitCTMEnvironment",
    "heisenberg_gate",
    "ipeps",
    "sublattice_rotate_gate",
    "symmetrize_c4v",
    "build_c4v_basis",
    "c4v_coeffs_from_tensor",
    "c4v_tensor_from_coeffs",
    "xxz_gate",
    "ctm",
    "ctm_2site",
    "ctm_split",
    "compute_energy_ctm_2site",
    "compute_energy_split_ctm",
    "optimize_gs_ad",
    "optimize_gs_ad_chi_schedule",
    # Coarse-grained iPEPS
    "CGGates",
    "honeycomb_cg_gates",
    "kagome_cg_gates",
    "compute_energy_cg",
    # Standard CTM (Tensor protocol)
    "CTMTensorEnv",
    "ctm_tensor",
    "ctm_tensor_c4v",
    "ctm_tensor_2site",
    "ctm_multisite",
    "compute_energy_ctm_tensor",
    "compute_energy_ctm_tensor_2site",
    # Split CTM (Tensor protocol)
    "SplitCTMTensorEnv",
    "ctm_split_tensor",
    "compute_energy_split_ctm_tensor",
    # fPEPS (fermionic iPEPS)
    "FPEPSConfig",
    "fpeps",
    "optimize_fpeps_ad",
    "spinless_fermion_gate",
    # iPEPS Excitations
    "ExcitationConfig",
    "ExcitationResult",
    "compute_excitations",
    "make_momentum_path",
    # Lattice
    "Bond",
    "Lattice",
    "square",
    "checkerboard",
    "honeycomb",
    "triangular",
    "kagome",
    # Network
    "TensorNetwork",
    "build_mps",
    "build_peps",
    "NetworkBlueprint",
    "from_netfile",
]
