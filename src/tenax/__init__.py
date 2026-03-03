"""Tenax: JAX-based tensor network library with symmetry-aware block-sparse tensors.

Label-based contraction (Cytnx-style):
    Tensor legs carry string or integer labels. Two legs with the same label
    across different tensors are automatically contracted when contract() is called.

.. note::
    Importing ``tenax`` enables JAX 64-bit mode (``jax_enable_x64``).
    All tensors and algorithms default to ``float64``.

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
            TensorIndex(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN,  label="phys"),
            TensorIndex(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN,  label="left"),
            TensorIndex(u1, np.array([1, 0, -1], dtype=np.int32), FlowDirection.OUT, label="bond"),
        ),
        key=key,
    )
    B = A.relabel("bond", "right").relabel("left", "bond")
    result = contract(A, B)   # contracts over shared label "bond"
    print(result.labels())    # ('phys', 'left', 'phys', 'right')
"""

import jax

jax.config.update("jax_enable_x64", True)

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
from tenax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    compute_energy_ctm_2site,
    compute_energy_split_ctm,
    ctm,
    ctm_2site,
    ctm_split,
    ipeps,
    iPEPSConfig,
    optimize_gs_ad,
)
from tenax.algorithms.ipeps_excitations import (
    ExcitationConfig,
    ExcitationResult,
    compute_excitations,
    make_momentum_path,
)
from tenax.algorithms.trg import (
    TRGConfig,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)
from tenax.contraction.contractor import (
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)
from tenax.core.index import FlowDirection, Label, TensorIndex
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
from tenax.network.netfile import NetworkBlueprint, from_netfile
from tenax.network.network import TensorNetwork, build_mps, build_peps

__version__ = "0.1.0"

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
    "Label",
    "TensorIndex",
    # Tensors
    "Tensor",
    "DenseTensor",
    "SymmetricTensor",
    "BlockKey",
    "inner",
    # Contraction
    "contract",
    "contract_with_subscripts",
    "truncated_svd",
    "qr_decompose",
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
    # TRG
    "TRGConfig",
    "trg",
    "compute_ising_tensor",
    "ising_free_energy_exact",
    # HOTRG
    "HOTRGConfig",
    "hotrg",
    # iDMRG
    "iDMRGConfig",
    "iDMRGResult",
    "idmrg",
    "build_bulk_mpo_heisenberg",
    "build_bulk_mpo_heisenberg_cylinder",
    # iPEPS
    "iPEPSConfig",
    "CTMConfig",
    "CTMEnvironment",
    "SplitCTMEnvironment",
    "ipeps",
    "ctm",
    "ctm_2site",
    "ctm_split",
    "compute_energy_ctm_2site",
    "compute_energy_split_ctm",
    "optimize_gs_ad",
    # fPEPS (fermionic iPEPS)
    "FPEPSConfig",
    "fpeps",
    "spinless_fermion_gate",
    # iPEPS Excitations
    "ExcitationConfig",
    "ExcitationResult",
    "compute_excitations",
    "make_momentum_path",
    # Network
    "TensorNetwork",
    "build_mps",
    "build_peps",
    "NetworkBlueprint",
    "from_netfile",
]
