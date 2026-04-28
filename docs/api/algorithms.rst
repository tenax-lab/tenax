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

Honeycomb iPEPS CTM
-------------------

Native rank-4, 6-corner, 3-direction, 2-sublattice CTMRG for honeycomb
iPEPS with implicit-AD energy via custom VJP + JIT-fused GMRES backward.
References: Lukin & Sotnikov, PRB 107, 054424 (2023) for the 6-corner
CTMRG; PRE 109, 045305 (2024) §II.C for the 2-sublattice extension.
Design and implementation plan in
``docs/plans/2026-04-25-honeycomb-ctm-design.md`` and
``docs/plans/2026-04-25-honeycomb-ctm-plan.md``.

.. autofunction:: tenax.algorithms.honeycomb_ctm.honeycomb_ctm_energy_implicit

.. autofunction:: tenax.algorithms.honeycomb_ctm.honeycomb_ctm_run

.. autoclass:: tenax.algorithms.honeycomb_ctm.HoneycombCTMEnv
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.honeycomb_ctm.HoneycombConvergeInfo
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.honeycomb_ctm.initialize_honeycomb_env

.. autofunction:: tenax.algorithms.honeycomb_ctm.compute_honeycomb_energy

.. autofunction:: tenax.algorithms.honeycomb_ctm.compute_honeycomb_triangle_energy

In addition, two module-level constants describe the topology and are
available via the same shim:

* ``tenax.algorithms.honeycomb_ctm.HONEYCOMB_NEIGHBORS`` — bipartite
  neighbor map ``{coord: {direction: neighbor_coord}}``.
* ``tenax.algorithms.honeycomb_ctm.HONEYCOMB_DIRECTIONS`` — the canonical
  direction tuple ``("e0", "e1", "e2")``.

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

.. autofunction:: tenax.algorithms.ad_utils.ctm_tensor_converge

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
