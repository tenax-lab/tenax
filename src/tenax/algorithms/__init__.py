"""Tensor network algorithms: DMRG, iDMRG, TRG, HOTRG, iPEPS."""

from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
from tenax.algorithms.auto_mpo import (
    AutoMPO,
    HamiltonianTerm,
    build_auto_mpo,
    spin_half_ops,
    spin_one_ops,
)
from tenax.algorithms.dmrg import (
    DMRGConfig,
    DMRGResult,
    build_mpo_heisenberg,
    build_random_mps,
    dmrg,
)
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    fpeps,
    spinless_fermion_gate,
)
from tenax.algorithms.hotrg import HOTRGConfig, hotrg
from tenax.algorithms.idmrg import (
    build_bulk_mpo_heisenberg,
    build_bulk_mpo_heisenberg_cylinder,
    idmrg,
    iDMRGConfig,
    iDMRGResult,
)
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, xxz_gate
from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_ctm_convergence import ctm, ctm_2site, ctm_split
from tenax.algorithms.ipeps_excitations import (
    ExcitationConfig,
    ExcitationResult,
    compute_excitations,
    make_momentum_path,
)
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.algorithms.ipeps_rdm import (
    compute_energy_ctm_2site,
    compute_energy_split_ctm,
)
from tenax.algorithms.trg import (
    TRGConfig,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)

__all__ = [
    # AutoMPO
    "AutoMPO",
    "HamiltonianTerm",
    "build_auto_mpo",
    "spin_half_ops",
    "spin_one_ops",
    # DMRG
    "DMRGConfig",
    "DMRGResult",
    "dmrg",
    "build_mpo_heisenberg",
    "build_random_mps",
    # iDMRG
    "iDMRGConfig",
    "iDMRGResult",
    "idmrg",
    "build_bulk_mpo_heisenberg",
    "build_bulk_mpo_heisenberg_cylinder",
    # TRG
    "TRGConfig",
    "trg",
    "compute_ising_tensor",
    "ising_free_energy_exact",
    # HOTRG
    "HOTRGConfig",
    "hotrg",
    # iPEPS
    "iPEPSConfig",
    "CTMConfig",
    "CTMEnvironment",
    "heisenberg_gate",
    "ipeps",
    "xxz_gate",
    "ctm",
    "ctm_2site",
    "SplitCTMEnvironment",
    "compute_energy_ctm_2site",
    "compute_energy_split_ctm",
    "ctm_split",
    "optimize_gs_ad",
    # CTM multisite
    "ctm_multisite",
    # fPEPS (fermionic iPEPS)
    "FPEPSConfig",
    "fpeps",
    "spinless_fermion_gate",
    # iPEPS Excitations
    "ExcitationConfig",
    "ExcitationResult",
    "compute_excitations",
    "make_momentum_path",
]
