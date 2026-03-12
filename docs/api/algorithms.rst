Algorithms
==========

DMRG
----

.. autoclass:: tenax.algorithms.dmrg.DMRGConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.dmrg.DMRGResult
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.dmrg.dmrg

.. autofunction:: tenax.algorithms.dmrg.build_mpo_heisenberg

.. autofunction:: tenax.algorithms.dmrg.build_random_mps

iDMRG
-----

.. autoclass:: tenax.algorithms.idmrg.iDMRGConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.idmrg.iDMRGResult
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.idmrg.idmrg

.. autofunction:: tenax.algorithms.idmrg.build_bulk_mpo_heisenberg

.. autofunction:: tenax.algorithms.idmrg.build_bulk_mpo_heisenberg_cylinder

TRG
---

.. autoclass:: tenax.algorithms.trg.TRGConfig
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.trg.trg

.. autofunction:: tenax.algorithms.trg.compute_ising_tensor

.. autofunction:: tenax.algorithms.trg.ising_free_energy_exact

HOTRG
-----

.. autoclass:: tenax.algorithms.hotrg.HOTRGConfig
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.hotrg.hotrg

iPEPS
-----

.. autoclass:: tenax.algorithms.ipeps_config.iPEPSConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.ipeps_config.CTMConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.ipeps_config.CTMEnvironment
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.ipeps.xxz_gate

.. autofunction:: tenax.algorithms.ipeps.ipeps

.. autofunction:: tenax.algorithms.ipeps_ctm.ctm

.. autofunction:: tenax.algorithms.ipeps_ctm.ctm_2site

.. autofunction:: tenax.algorithms.ipeps_rdm.compute_energy_ctm_2site

.. autofunction:: tenax.algorithms.ipeps_optimize.optimize_gs_ad

.. autofunction:: tenax.algorithms._ctm_tensor_convergence.ctm_multisite

Lattice
-------

.. autoclass:: tenax.core.lattice.Bond
   :members:
   :no-index:

.. autoclass:: tenax.core.lattice.Lattice
   :members:
   :no-index:

.. autofunction:: tenax.core.lattice.square

.. autofunction:: tenax.core.lattice.checkerboard

.. autofunction:: tenax.core.lattice.honeycomb

.. autofunction:: tenax.core.lattice.triangular

.. autofunction:: tenax.core.lattice.kagome

AD Utilities
------------

.. autofunction:: tenax.algorithms.ad_utils.truncated_svd_ad

.. autofunction:: tenax.algorithms.ad_utils.ctm_converge

iPEPS Excitations
-----------------

.. autoclass:: tenax.algorithms.ipeps_excitations.ExcitationConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.ipeps_excitations.ExcitationResult
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.ipeps_excitations.compute_excitations

.. autofunction:: tenax.algorithms.ipeps_excitations.make_momentum_path

Fermionic iPEPS (fPEPS)
-----------------------

.. autoclass:: tenax.algorithms.fermionic_ipeps.FPEPSConfig
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.fermionic_ipeps.spinless_fermion_gate

.. autofunction:: tenax.algorithms.fermionic_ipeps.fpeps

AutoMPO
-------

.. autoclass:: tenax.algorithms.auto_mpo.AutoMPO
   :members:

.. autoclass:: tenax.algorithms.auto_mpo.HamiltonianTerm
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.auto_mpo.build_auto_mpo

.. autofunction:: tenax.algorithms.auto_mpo.spin_half_ops

.. autofunction:: tenax.algorithms.auto_mpo.spin_one_ops
