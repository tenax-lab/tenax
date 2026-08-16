"""Fermionic infinite Projected Entangled Pair States (fPEPS) algorithm.

This module implements iPEPS for fermionic systems using FermionParity symmetry.
The state is represented as a PEPS with fermionic tensor structure, where
Koszul signs are automatically handled by SymmetricTensor operations.

Currently supports:
- Spinless fermion Hamiltonian: H = -t(c†c + h.c.) + V(n_i n_j)
- Trotter decomposition for imaginary time evolution
- fPEPS site tensor initialization with FermionParity
- Simple update (horizontal and vertical bonds)
- Full fPEPS optimization with CTM energy evaluation

Reference:
- Corboz et al., PRB 81, 165104 (2010)
- Barthel, Pineda, Eisert, PRA 80, 042333 (2009)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_simple_update import (
    BondWeights,
    _simple_update_checkerboard_sweep,
    _to_physical_pair,
)
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

if TYPE_CHECKING:
    from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv


@dataclass
class FPEPSConfig:
    """Configuration for fermionic iPEPS.

    Attributes:
        D:                    Virtual bond dimension.
        t:                    Hopping amplitude.
        V:                    Nearest-neighbour interaction strength.
        dt:                   Imaginary time step size for Trotter decomposition.
        num_imaginary_steps:  Number of imaginary time evolution steps.
        ctm_chi:              Bond dimension for CTM environment.
        ctm_max_iter:         Maximum CTM iterations.
        ctm_conv_tol:         CTM convergence tolerance.
    """

    D: int = 2
    t: float = 1.0
    V: float = 0.0
    dt: float = 0.01
    num_imaginary_steps: int = 200
    ctm_chi: int = 8
    ctm_max_iter: int = 50
    ctm_conv_tol: float = 1e-6


def spinless_fermion_gate(config: FPEPSConfig) -> SymmetricTensor:
    """Build the 2-site Hamiltonian H = -t(c†c + h.c.) + V(n_i n_j).

    The Hamiltonian acts on two spinless fermion sites with local
    Hilbert space ``{|0>, |1>}`` (empty, occupied). The fermionic
    anti-commutation relations are encoded via FermionParity symmetry.

    Args:
        config: FPEPSConfig with hopping t and interaction V.

    Returns:
        SymmetricTensor with 4 legs (si, sj, si_out, sj_out),
        shape (2, 2, 2, 2), using FermionParity symmetry.
    """
    t = config.t
    V = config.V

    # Build the dense 4x4 Hamiltonian matrix in the basis
    # |00>, |01>, |10>, |11> (site i tensor site j)
    #
    # c†_i c_j: |10><01| (with fermionic sign from Jordan-Wigner = +1 here)
    # c†_j c_i: |01><10|
    # n_i n_j:  |11><11|
    H = np.zeros((4, 4), dtype=np.float64)

    # Hopping: -t (c†_i c_j + c†_j c_i)
    # |01> -> |10>: c†_i c_j |01> = c†_i |00> = |10>, sign = +1
    # |10> -> |01>: c†_j c_i |10> = c†_j |00> = |01>, sign = +1
    H[2, 1] = -t  # <10|H|01>
    H[1, 2] = -t  # <01|H|10>

    # Interaction: V * n_i * n_j
    H[3, 3] = V  # <11|H|11>

    # Reshape to (2, 2, 2, 2): (si, sj, si_out, sj_out)
    H_4leg = H.reshape(2, 2, 2, 2)

    # Create TensorIndex objects with FermionParity
    sym = FermionParity()
    charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="si_out"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="sj_out"),
    )

    return SymmetricTensor.from_dense(jnp.array(H_4leg), indices)


def _trotter_gate(H: SymmetricTensor, dt: float) -> SymmetricTensor:
    """Compute the Trotter gate exp(-dt * H).

    Uses dense eigendecomposition: H = U diag(E) U†, then
    exp(-dt * H) = U diag(exp(-dt * E)) U†.

    Args:
        H:  2-site Hamiltonian as SymmetricTensor with 4 legs.
        dt: Imaginary time step (real-valued).

    Returns:
        SymmetricTensor with same indices as H, representing exp(-dt * H).
    """
    dense = H.todense().reshape(4, 4)
    dense_np = np.array(dense)

    # Eigendecomposition of the Hermitian matrix
    eigvals, eigvecs = np.linalg.eigh(dense_np)

    # Compute exp(-dt * H)
    exp_eigvals = np.exp(-dt * eigvals)
    gate = eigvecs @ np.diag(exp_eigvals) @ eigvecs.conj().T

    # Reshape back to (2, 2, 2, 2)
    gate_4leg = gate.reshape(2, 2, 2, 2)

    return SymmetricTensor.from_dense(jnp.array(gate_4leg), H.indices)


def _build_initial_fpeps_tensor(
    config: FPEPSConfig,
    key: jax.Array | None = None,
) -> SymmetricTensor:
    """Build a random initial fPEPS site tensor with FermionParity symmetry.

    The tensor has FermionParity symmetry on all legs. Virtual bond
    charges alternate 0, 1, 0, 1, ... for bond dimension D.
    Physical charges are [0, 1] (empty, occupied).

    Flows:
        u = OUT, d = IN, l = OUT, r = IN, phys = IN

    Args:
        config: FPEPSConfig with bond dimension D.
        key:    JAX random key. If None, uses PRNGKey(0).

    Returns:
        SymmetricTensor with 5 legs (u, d, l, r, phys).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    D = config.D
    sym = FermionParity()

    # Virtual charges: [i % 2 for i in range(D)]
    virt_charges = np.array([i % 2 for i in range(D)], dtype=np.int32)

    # Physical charges: [0, 1]
    phys_charges = np.array([0, 1], dtype=np.int32)

    indices = (
        TensorIndex.from_charges(sym, virt_charges, FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt_charges, FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt_charges, FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt_charges, FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys_charges, FlowDirection.IN, label="phys"),
    )

    return SymmetricTensor.random_normal(indices, key)


def _initialize_fpeps(config: FPEPSConfig, key: jax.Array) -> SymmetricTensor:
    """Create a random fPEPS site tensor A[u, d, l, r, phys].

    Thin wrapper around :func:`_build_initial_fpeps_tensor` for
    backward compatibility.

    Args:
        config: FPEPSConfig with bond dimension D.
        key:    JAX random key.

    Returns:
        SymmetricTensor with 5 legs (u, d, l, r, phys).
    """
    return _build_initial_fpeps_tensor(config, key)


def _normalize_tensor(T: SymmetricTensor) -> SymmetricTensor:
    """Normalize a SymmetricTensor to unit norm."""
    norm_val = float(T.norm())
    if norm_val <= EPS:
        return T
    return T * (1.0 / norm_val)


def _fpeps_simple_update(
    A: SymmetricTensor,
    hamiltonian_gate: SymmetricTensor,
    max_D: int,
    dt: float,
    steps: int,
    B: SymmetricTensor | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor, BondWeights]:
    """Run simple update for a given number of steps, on a **2-site** cell.

    Runs the shared checkerboard sweep rather than the 1-site routines above,
    which cannot be correct (#878).  On a 1-site ansatz ``A`` is *both* ends of
    every bond, and the 1-site update kept only ``U`` from the SVD and discarded
    ``Vh``, so ``A`` received the left/top half of every gate and never the
    right/bottom half.  That is the asymmetry #667 identified, and it drove the
    state to a product state independently of ``dt`` -- measured at equal
    imaginary time, ``dt`` = 0.05 / 0.01 / 0.005 all gave ``lam_h = [1, 0]``.

    The shared sweep is already fermion-capable: ``_truncation_base_charges``
    gates the layout constraint on ``is_fermionic``, so it keeps the per-sector
    keep counts the 1-site path needs (#558/#559/#563) while the bosonic
    checkerboard gets the unconstrained truncation #865 restored.

    **The two sublattices are returned separately, and that is not cosmetic.**
    The t-V ground state is a charge-density wave at finite ``V`` -- charge
    order on the checkerboard -- which no single tensor can represent.  Ask
    :func:`sublattice_gap` of the result to see whether a given run actually
    produced one; the measured numbers are quoted on :func:`fpeps`, along with
    the standing caveat that neither this nor #878 certifies the energy (#392).
    Collapsing the pair back to one tensor would hand the CTM a state the sweep
    never produced.

    All **four** bond spectra are carried through as a
    :class:`~tenax.algorithms.ipeps_simple_update.BondWeights`, not two.  The
    two-lambda return this replaced made the answer depend on ``steps % 4``:
    phases 0 and 2 both wrote ``lam_h``, so whichever ran last was stamped onto
    both horizontal bonds by ``_to_physical_pair`` (#851).

    ``independent_bonds`` is left at its default ``False``, so ``h_AB`` and
    ``h_BA`` share a spectrum (and likewise the vertical pair).  That is a
    deliberate choice for this Hamiltonian, not an oversight: the checkerboard
    charge-density wave is symmetric under reflection through a site, which maps
    ``A.r<->B.l`` onto ``B.r<->A.l``, so the two horizontal bonds carry the same
    Schmidt spectrum in the state this sweep is trying to reach -- the CDW
    breaks the *site* symmetry, which is what ``A != B`` expresses, and not the
    bond symmetry.  Freeing the two bonds buys nothing there and costs the
    robustness measured in
    :attr:`~tenax.algorithms.ipeps_config.iPEPSConfig.su_independent_bond_lambdas`
    (a dimerising direction that four free bonds can follow), which this path
    can least afford: its survival is already seed-dependent at every ``D``.

    Args:
        A:                Initial fPEPS site tensor for sublattice A.
        hamiltonian_gate: 2-site Hamiltonian (SymmetricTensor).
        max_D:            Maximum bond dimension.
        dt:               Imaginary time step.
        steps:            Number of simple update steps.  The sweep runs four
                          phases per step, so each of the four bonds is evolved
                          once per step.
        B:                Initial tensor for sublattice B.  Defaults to ``A``,
                          which starts both sublattices from the same tensor;
                          the sweep breaks the symmetry itself.

    Returns:
        ``(A_opt, B_opt, lambdas)`` after all steps, in Vidal form -- call
        :func:`~tenax.algorithms.ipeps_simple_update._to_physical_pair` for the
        tensors a CTM should contract.
    """
    gate = _trotter_gate(hamiltonian_gate, dt)
    # Four phases per step, so each of the four checkerboard bonds is evolved
    # once per step -- the same imaginary time per bond as the old two-phase
    # loop delivered for its two.
    return _simple_update_checkerboard_sweep(
        A, A if B is None else B, gate, max_D, 4 * steps
    )


def sublattice_gap(
    A: SymmetricTensor,
    B: SymmetricTensor,
    env_A: SplitCTMTensorEnv,
    env_B: SplitCTMTensorEnv,
) -> float:
    """Is there **charge order** between the two checkerboard sublattices?

    The trace distance between the two sublattices' one-site reduced density
    matrices, obtained by tracing out one site of the two-site RDM the energy is
    already computed from::

        rho_A = Tr_B rho_AB     rho_B = Tr_A rho_AB
        gap   = 0.5 * ||rho_A - rho_B||_1

    For spinless fermions ``FermionParity`` forbids the off-diagonal
    ``<c>``-type entries, so each ``rho`` is diagonal in the occupation basis
    and this is exactly ``|<n_A> - <n_B>|`` -- the charge-density-wave order
    parameter, which is what makes it the useful probe for the t-V model whose
    CDW is the reason :func:`fpeps` returns a pair at all.  It runs 0 at ``V=0``
    (free fermions, no charge order) to 1 for the fully polarised
    occupied/empty checkerboard.

    .. warning::
        **It is a one-body probe, and a zero does not mean one tensor would
        do.**  ``gap == 0`` says the two *one-site* RDMs coincide; it says
        nothing about two-site structure.  A columnar-dimer or bond-ordered
        state has identical on-site densities on both sublattices and reads
        ``0`` here while being genuinely two-site, so reading a zero as "the
        sweep collapsed, a 1-site ansatz would be faithful" is wrong on exactly
        the states where it matters.  A nonzero value *is* positive evidence of
        charge order; only the converse fails.  To rule out two-site order in
        general, compare a two-site observable -- e.g. the horizontal and
        vertical bond energies of the pair against each other.

    **This replaces a Gram-matrix fingerprint that did not measure the state.**
    The singular values of ``M = T T†`` on one virtual leg are *not* invariant
    under a PEPS bond gauge: under ``T -> G T`` the matrix goes to ``G M G†``,
    whose spectrum moves unless ``G`` is unitary, and simple update's gauge is
    not.  The same trap is why ``||A - B||`` is worthless here -- it stays ~1.7
    when the two tensors are provably the same physical state.  A reduced
    density matrix has no such freedom: the bond gauge cancels between the ket
    and bra layers of the environment contraction.

    It is also block-sparse throughout.  The Gram version called ``todense()``
    on the full rank-5 site tensor -- a ``D**4 * d`` array -- once per leg on
    each of the two sites, so eight of them per call, for an answer that is a
    ``d``-by-``d`` matrix.  Nothing here densifies anything larger than the
    ``d**4`` RDM.

    Args:
        A, B:           The two checkerboard site tensors, in **physical**
                        (CTM-contractable) form -- the pair :func:`fpeps`
                        returns, not the bare Vidal ``Gamma``.
        env_A, env_B:   Their converged ``SplitCTMTensorEnv`` environments, the
                        pair :func:`fpeps` returns alongside the state.

    Returns:
        The trace distance.  It is in ``[0, 1]`` by construction *for genuine
        density matrices*, and a value outside that range is the environment
        telling you its RDM is not PSD (#854 warns about the same thing on the
        energy path) -- measured up to **1.07** at chi=4 on a deliberately
        under-converged environment, against a few ``1e-4`` once the CTM has
        settled.  It is deliberately **not** clipped: the excess is a usable
        signal that the environment is too small or too few sweeps, and
        clipping would hide it inside a plausible-looking 1.0.
    """
    from tenax.algorithms._split_ctm_tensor_energy import _rdm2x1_split_tensor_2site

    # (s1_A_ket, s2_B_ket, s1_A_bra, s2_B_bra), already trace-normalised.
    rho = np.asarray(_rdm2x1_split_tensor_2site(A, B, env_A, env_B))
    rho_A = np.einsum("abcb->ac", rho)
    rho_B = np.einsum("abad->bd", rho)
    # The two-site RDM is trace-normalised, so both traces are 1 already; divide
    # anyway so a caller handing in an unnormalised environment gets a distance
    # between density matrices rather than a number scaled by the norm.
    #
    # The **signed** trace, not ``abs(trace)``.  These RDMs are only
    # approximately PSD -- #854's warning fires on this very path at small chi
    # -- and a negative trace means the environment has produced something that
    # is not a density matrix at all.  Dividing by its magnitude would flip the
    # sign of the whole matrix and hand back a plausible-looking distance in
    # [0, 1]; dividing by the signed value leaves the failure visible.  A trace
    # at exactly 0 is floored rather than divided, to keep this from raising in
    # place of the caller's own diagnostics.
    tr_A, tr_B = np.trace(rho_A).real, np.trace(rho_B).real
    rho_A = rho_A / (tr_A if abs(tr_A) > 1e-300 else 1e-300)
    rho_B = rho_B / (tr_B if abs(tr_B) > 1e-300 else 1e-300)
    diff = rho_A - rho_B
    diff = 0.5 * (diff + diff.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))


# ------------------------------------------------------------------ #
# CTM energy evaluation                                                #
# ------------------------------------------------------------------ #
#
# Fermionic signs are handled by the graded tensor formalism (Koszul
# signs in transpose, contraction, and SVD).  After simple update,
# the resulting SymmetricTensor site tensor can be contracted with
# a standard CTM — no explicit swap gates are needed.
# ------------------------------------------------------------------ #


def fermionic_ctm(A, config):
    """Run CTM to convergence for a fermionic PEPS site tensor.

    Fermionic signs are handled by the graded tensor formalism (Koszul
    signs), so CTM contraction uses the standard double-layer procedure.

    When *A* is a ``Tensor`` (DenseTensor or SymmetricTensor), uses the
    Tensor-protocol CTM (``ctm_tensor``) which avoids densification.
    Otherwise falls back to the dense bosonic CTM.

    Args:
        A:      fPEPS site tensor (SymmetricTensor with lambdas absorbed).
        config: FPEPSConfig.

    Returns:
        Converged ``CTMTensorEnv`` (if Tensor input) or ``CTMEnvironment``.
    """
    if isinstance(A, Tensor):
        from tenax.algorithms._ctm_tensor_convergence import ctm_tensor

        env, _ = ctm_tensor(
            A,
            chi=config.ctm_chi,
            max_iter=config.ctm_max_iter,
            conv_tol=config.ctm_conv_tol,
        )
        return env

    from tenax.algorithms.ipeps_config import CTMConfig
    from tenax.algorithms.ipeps_ctm_convergence import ctm

    ctm_cfg = CTMConfig(
        chi=config.ctm_chi,
        max_iter=config.ctm_max_iter,
        conv_tol=config.ctm_conv_tol,
    )
    return ctm(A, ctm_cfg)


def compute_energy_fermionic_ctm(A, env, hamiltonian_gate):
    """Compute energy per site using a CTM environment.

    Supports ``CTMTensorEnv`` (from ``ctm_tensor``) and legacy
    ``CTMEnvironment`` (from dense CTM).

    Args:
        A:                fPEPS site tensor (SymmetricTensor or dense).
        env:              Converged environment from :func:`fermionic_ctm`.
        hamiltonian_gate: 2-site Hamiltonian (SymmetricTensor or dense array).

    Returns:
        Energy per site (float).
    """
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_init import CTMTensorEnv

    if isinstance(env, CTMTensorEnv):
        # If env tensors are DenseTensor (fermionic fallback), densify A and gate
        if isinstance(env.C1, DenseTensor) and isinstance(A, SymmetricTensor):
            A = DenseTensor(A.todense(), A.indices)
        if isinstance(env.C1, DenseTensor) and isinstance(
            hamiltonian_gate, SymmetricTensor
        ):
            hamiltonian_gate = DenseTensor(
                hamiltonian_gate.todense(), hamiltonian_gate.indices
            )
        return float(compute_energy_ctm_tensor(A, env, hamiltonian_gate))

    from tenax.algorithms.ipeps_rdm import compute_energy_ctm

    A_dense = A.todense() if isinstance(A, Tensor) else A
    d = A_dense.shape[-1]
    if isinstance(hamiltonian_gate, SymmetricTensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)
    return float(compute_energy_ctm(A_dense, env, H, d))


def fpeps(
    hamiltonian_gate: SymmetricTensor,
    config: FPEPSConfig,
    initial_tensor: SymmetricTensor
    | tuple[SymmetricTensor, SymmetricTensor]
    | None = None,
    key: jax.Array | None = None,
) -> tuple[
    float,
    tuple[SymmetricTensor, SymmetricTensor],
    tuple[SplitCTMTensorEnv, SplitCTMTensorEnv],
]:
    """Run fPEPS: simple update optimization + CTM energy evaluation.

    **Two-site checkerboard**, end to end (#878).  The previous 1-site ansatz
    could not be right: ``A`` was both ends of every bond, the update discarded
    half of every gate, and the state collapsed to a product state regardless of
    ``dt``.  It also cannot represent the charge-density wave that is the t-V
    ground state at finite ``V``.

    .. warning::
        The **absolute energy is not certified** (#392), and this fix does not
        change that.  ``H`` carries no chemical potential, so both the empty
        state and the fully polarised checkerboard are exact ``E = 0``
        eigenstates -- fixed points of imaginary time that are not the ground
        state -- and the sweep is observed to settle on them: at 200 steps,
        D=2, ``V=0``, it reports ``E = -6e-05`` where the half-filled answer is
        about ``-1.6 t``.  :func:`sublattice_gap` tells you *which* state you
        landed on; it does not tell you it is the ground state.

    Args:
        hamiltonian_gate: 2-site Hamiltonian as SymmetricTensor.
        config:           FPEPSConfig.
        initial_tensor:   Either an ``(A, B)`` pair -- the form this function
                          returns, so its own output restarts it -- or a single
                          tensor, which starts both sublattices from the same
                          place and lets the sweep break the symmetry itself.
                          ``None`` for a random start.

                          **A restart is not a continuation.** The sweep always
                          begins from ``BondWeights.ones``, so its first cycle
                          treats the outer legs as unweighted while the tensors
                          handed in already carry ``sqrt(lambda)`` baked in.
                          ``fpeps(N)`` is therefore *not* ``fpeps(N/2)`` fed
                          back for another ``N/2``: the restart resumes a
                          differently-gauged state. Use it to continue
                          annealing, not to reproduce a longer single run.
        key:              JAX random key (used if initial_tensor is None).

    Returns:
        ``(energy, (A, B), (env_A, env_B))``.  **Changed in #878**: the state
        and environment are now pairs.

        The pair is in **physical** form -- the tensors the returned
        environments were converged on, each virtual leg already carrying
        ``sqrt(lambda)`` -- so it round-trips: hand it straight back as
        ``initial_tensor`` and the restart begins on the state that was
        returned.  Returning the bare Vidal ``Gamma`` instead would not: the
        bond weights live outside it, and a restart that reset them to ones
        would silently continue from a different state.

        :func:`sublattice_gap` on ``(A, B, env_A, env_B)`` measures the **charge
        order** between the two sublattices.  Measured on an 8-step D=2 sweep at
        chi=4 it tracks the interaction that drives that order, as it must:
        **0.037** at ``V=0`` (free fermions, no charge order at all), **0.270**
        at ``V=1``, **0.900** at ``V=2`` and **1.000** at ``V=4`` -- the fully
        polarised occupied/empty checkerboard.

        Read it in one direction only.  A nonzero value is evidence the pair
        carries real charge order; a zero is **not** evidence that one tensor
        would have done, because it is a one-body probe and a dimerised or
        bond-ordered state has identical on-site densities on both sublattices.
        See :func:`sublattice_gap` for the full statement.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    if isinstance(initial_tensor, tuple):
        A, B = initial_tensor
    elif initial_tensor is not None:
        A, B = initial_tensor, initial_tensor
    else:
        if key is None:
            key = jax.random.PRNGKey(0)
        A = B = _initialize_fpeps(config, key)

    A_opt, B_opt, lambdas = _fpeps_simple_update(
        A,
        hamiltonian_gate,
        max_D=config.D,
        dt=config.dt,
        steps=config.num_imaginary_steps,
        B=B,
    )

    # sqrt(lambda) on each leg, so every bond of the lattice picks it up exactly
    # once.  The old code applied the *full* lambda to each leg, which squared
    # every bond weight of the returned state (#878).
    #
    # All four spectra are handed over rather than two, and ``_to_physical_pair``
    # maps each leg to its own bond.  Be precise about what that buys *here*:
    # with ``independent_bonds=False`` -- the default this path ships, see
    # ``_fpeps_simple_update`` -- ``lambdas.h_AB is lambdas.h_BA``, so the
    # numbers are identical to the two-lambda code this replaced.  What changes
    # is that the mapping is now expressed once, in one place, instead of being
    # re-derived at each call site; it stops being possible to write the #851
    # mix-up here.  The numerical fix arrives only if the four bonds are ever
    # freed, or when #882 removes stored lambdas altogether.
    A_phys, B_phys = _to_physical_pair(A_opt, B_opt, lambdas)

    env_A, env_B = ctm_split_tensor_2site(
        A_phys,
        B_phys,
        config.ctm_chi,
        max_iter=config.ctm_max_iter,
        conv_tol=config.ctm_conv_tol,
    )
    d = A_phys.indices[A_phys.labels().index("phys")].dim
    energy = compute_energy_split_ctm_tensor_2site(
        A_phys, B_phys, env_A, env_B, hamiltonian_gate, d=d
    )

    return float(energy), (A_phys, B_phys), (env_A, env_B)
