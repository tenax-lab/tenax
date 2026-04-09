"""Tensor network algorithms: DMRG, iDMRG, TRG, HOTRG, iPEPS.

Algorithm modules are loaded lazily on first access.
"""

# Mapping from public name to (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ctm_multisite": ("tenax.algorithms._ctm_tensor_convergence", "ctm_multisite"),
    "AutoMPO": ("tenax.algorithms.auto_mpo", "AutoMPO"),
    "HamiltonianTerm": ("tenax.algorithms.auto_mpo", "HamiltonianTerm"),
    "build_auto_mpo": ("tenax.algorithms.auto_mpo", "build_auto_mpo"),
    "spin_half_ops": ("tenax.algorithms.auto_mpo", "spin_half_ops"),
    "spin_one_ops": ("tenax.algorithms.auto_mpo", "spin_one_ops"),
    "DMRGConfig": ("tenax.algorithms.dmrg", "DMRGConfig"),
    "DMRGResult": ("tenax.algorithms.dmrg", "DMRGResult"),
    "dmrg": ("tenax.algorithms.dmrg", "dmrg"),
    "build_mpo_heisenberg": ("tenax.algorithms.dmrg", "build_mpo_heisenberg"),
    "build_random_mps": ("tenax.algorithms.dmrg", "build_random_mps"),
    "FPEPSConfig": ("tenax.algorithms.fermionic_ipeps", "FPEPSConfig"),
    "fpeps": ("tenax.algorithms.fermionic_ipeps", "fpeps"),
    "spinless_fermion_gate": (
        "tenax.algorithms.fermionic_ipeps",
        "spinless_fermion_gate",
    ),
    "HOTRGConfig": ("tenax.algorithms.hotrg", "HOTRGConfig"),
    "hotrg": ("tenax.algorithms.hotrg", "hotrg"),
    "build_bulk_mpo_heisenberg": (
        "tenax.algorithms.idmrg",
        "build_bulk_mpo_heisenberg",
    ),
    "build_bulk_mpo_heisenberg_cylinder": (
        "tenax.algorithms.idmrg",
        "build_bulk_mpo_heisenberg_cylinder",
    ),
    "idmrg": ("tenax.algorithms.idmrg", "idmrg"),
    "iDMRGConfig": ("tenax.algorithms.idmrg", "iDMRGConfig"),
    "iDMRGResult": ("tenax.algorithms.idmrg", "iDMRGResult"),
    "heisenberg_gate": ("tenax.algorithms.ipeps", "heisenberg_gate"),
    "ipeps": ("tenax.algorithms.ipeps", "ipeps"),
    "xxz_gate": ("tenax.algorithms.ipeps", "xxz_gate"),
    "iPEPSConfig": ("tenax.algorithms.ipeps_config", "iPEPSConfig"),
    "CTMConfig": ("tenax.algorithms.ipeps_config", "CTMConfig"),
    "CTMEnvironment": ("tenax.algorithms.ipeps_config", "CTMEnvironment"),
    "SplitCTMEnvironment": ("tenax.algorithms.ipeps_config", "SplitCTMEnvironment"),
    "ctm": ("tenax.algorithms.ipeps_ctm", "ctm"),
    "ctm_2site": ("tenax.algorithms.ipeps_ctm", "ctm_2site"),
    "ctm_split": ("tenax.algorithms.ipeps_ctm", "ctm_split"),
    "ExcitationConfig": ("tenax.algorithms.ipeps_excitations", "ExcitationConfig"),
    "ExcitationResult": ("tenax.algorithms.ipeps_excitations", "ExcitationResult"),
    "compute_excitations": (
        "tenax.algorithms.ipeps_excitations",
        "compute_excitations",
    ),
    "make_momentum_path": ("tenax.algorithms.ipeps_excitations", "make_momentum_path"),
    "optimize_gs_ad": ("tenax.algorithms.ipeps_optimize", "optimize_gs_ad"),
    "compute_energy_ctm_2site": (
        "tenax.algorithms.ipeps_rdm",
        "compute_energy_ctm_2site",
    ),
    "compute_energy_split_ctm": (
        "tenax.algorithms.ipeps_rdm",
        "compute_energy_split_ctm",
    ),
    "TRGConfig": ("tenax.algorithms.trg", "TRGConfig"),
    "compute_ising_tensor": ("tenax.algorithms.trg", "compute_ising_tensor"),
    "ising_free_energy_exact": ("tenax.algorithms.trg", "ising_free_energy_exact"),
    "trg": ("tenax.algorithms.trg", "trg"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tenax.algorithms' has no attribute {name!r}")


def __dir__():
    return list(__all__)


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
