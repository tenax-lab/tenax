"""Density Matrix Renormalization Group (DMRG) algorithm.

DMRG finds the ground state (or low-lying eigenstates) of a 1D quantum
Hamiltonian given as a Matrix Product Operator (MPO).

Architecture decisions:

- The outer sweep loop is a Python for-loop (not ``jax.lax.scan``) because bond
  dimensions change after each SVD truncation, preventing JIT across sweeps.
- The effective Hamiltonian matvec is ``@jax.jit`` compiled for performance.
- Lanczos eigensolver uses ``jax.lax.while_loop`` for static shapes inside JIT.
- Environment tensors (left/right blocks) are stored as Python lists of Tensor.

Label conventions::

    MPS site tensors:    legs = ("v{i-1}_{i}", "p{i}", "v{i}_{i+1}")
                         All sites are 3-leg, including boundaries:
                         site 0 has ("v_-1_0", "p0", "v0_1") with dim-1 left bond,
                         site L-1 has ("v{L-2}_{L-1}", "p{L-1}", "v{L-1}_{L}") with dim-1 right bond.
    MPO site tensors:    legs = ("w{i-1}_{i}", "mpo_top_{i}", "mpo_bot_{i}", "w{i}_{i+1}")
    Environment tensors: left_env[i] has legs ("mps_l", "mpo_l", "mps_l_conj")
                         right_env[i] has legs ("mps_r", "mpo_r", "mps_r_conj")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from tenax.algorithms._block_array import BlockArray

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor, inner
from tenax.linalg import qr as _linalg_qr
from tenax.network.network import TensorNetwork

# Optional Cython BLAS acceleration for hot loops.
try:
    from tenax.contraction._cython_blas import (
        cython_execute_plan as _cython_execute_plan,
    )

    _USE_CYTHON_PLAN = True
except ImportError:
    _USE_CYTHON_PLAN = False

try:
    from tenax.contraction._cython_blas import (
        cython_ba_sub_scaled_inplace as _cython_ba_sub_scaled_inplace,
    )

    _USE_CYTHON_SUB = True
except ImportError:
    _USE_CYTHON_SUB = False


@dataclass
class DMRGConfig:
    """Configuration for a DMRG run.

    Attributes:
        max_bond_dim:       Maximum allowed bond dimension (chi).
        num_sweeps:         Number of full left-right sweep cycles.
        convergence_tol:    Energy convergence threshold to stop early.
        num_states:         Number of lowest eigenstates to target (1 = ground state).
        two_site:           If True, use 2-site DMRG (allows bond dim growth).
                            If False, use 1-site DMRG (conserves bond dim exactly).
        lanczos_max_iter:   Maximum Lanczos iterations for eigenvalue solve.
        lanczos_tol:        Convergence tolerance for Lanczos.
        noise:              Perturbative noise added to density matrix (helps
                            escape local minima in 2-site DMRG).
        svd_trunc_err:      Maximum truncation error per SVD (overrides
                            max_bond_dim when set and more restrictive).
        target_charge:      Target total charge (e.g. 2*Sz for U(1)). If set,
                            validates MPS sector before and after each sweep.
                            Use with ``build_random_symmetric_mps(target_charge=...)``.
        subspace_expansion: If True, use subspace expansion during 1-site sweeps
                            to escape local minima. Requires ``two_site=False``.
        mixing_factor:      Mixing strength α for DMRG3S expansion (Hubig 2015).
                            Ignored when ``hybrid_mixing=True``.
        expansion_num_extra: Number of extra expansion states Δ beyond the kept
                            χ states. Default 0 means Δ = int(0.1 * χ).
        hybrid_mixing:      If True (default), use adaptive √(ε_trunc) scaling
                            instead of fixed α. The expansion is weighted by the
                            truncation error, giving strong mixing when needed
                            and vanishing mixing at convergence.
        numpy_blockwise:    Use numpy-only path for symmetric DMRG (no JAX
                            overhead). Default True. Set False to use the
                            original JAX-backed symmetric path.
        accelerator:        Backend dispatch mode for the DMRG sweep:
                            ``"auto"`` (default) — GPU/TPU uses JIT path; CPU with
                            symmetric tensors uses numpy/Cython path; CPU with
                            dense tensors uses JIT path.
                            ``"jit"`` — force JIT-compiled ``lax.scan`` sweep
                            (requires dense tensors; silently falls back for
                            symmetric or 1-site DMRG).
                            ``"off"`` — always use the existing Python sweep loop.
        verbose:            Print energy at each sweep.
    """

    max_bond_dim: int = 100
    num_sweeps: int = 10
    convergence_tol: float = 1e-10
    num_states: int = 1
    two_site: bool = True
    lanczos_max_iter: int = 50
    lanczos_tol: float = 1e-12
    noise: float = 0.0
    svd_trunc_err: float | None = None
    target_charge: int | None = None
    subspace_expansion: bool = False
    mixing_factor: float = 1e-3
    expansion_num_extra: int = 0
    hybrid_mixing: bool = True
    numpy_blockwise: bool = (
        True  # Use numpy-only path for symmetric DMRG (no JAX overhead)
    )
    accelerator: str = "auto"
    verbose: bool = False


class DMRGResult(NamedTuple):
    """Result of a DMRG run.

    Attributes:
        energy:               Final ground state energy.
        energies_per_sweep:   Energy at the end of each sweep.
        mps:                  FiniteMPS representing the optimized MPS.
        truncation_errors:    List of truncation errors at each bond update step.
        converged:            True if energy converged within convergence_tol.
    """

    energy: float
    energies_per_sweep: list[float]
    mps: FiniteMPS
    truncation_errors: list[float]
    converged: bool


class SweepOps(NamedTuple):
    """Callback bundle holding all backend-specific operations for a DMRG sweep.

    Dense and symmetric backends each provide their own implementations.
    The sweep loop is backend-agnostic — it only calls through ``ops.*``.
    """

    build_trivial_left_env: Callable[..., Tensor]
    build_trivial_right_env: Callable[..., Tensor]
    update_left_env: Callable[[Tensor, Tensor, Tensor], Tensor]
    update_right_env: Callable[[Tensor, Tensor, Tensor], Tensor]
    two_site_update: Callable[..., tuple[Tensor, float]]
    one_site_update: Callable[..., tuple[Tensor, float]]


def _dense_ops() -> SweepOps:
    """Return the dense (existing) backend callbacks."""
    return SweepOps(
        build_trivial_left_env=_build_trivial_left_env,
        build_trivial_right_env=_build_trivial_right_env,
        update_left_env=_update_left_env,
        update_right_env=_update_right_env,
        two_site_update=_two_site_update,
        one_site_update=_one_site_update,
    )


def dmrg(
    hamiltonian: TensorNetwork,
    initial_mps: FiniteMPS | TensorNetwork,
    config: DMRGConfig,
) -> DMRGResult:
    """Run DMRG to find the ground state of a 1D Hamiltonian given as MPO.

    The Hamiltonian must be provided as an MPO (Matrix Product Operator)
    TensorNetwork with L site tensors connected by virtual bonds.

    Args:
        hamiltonian:  MPO representation of the Hamiltonian.
        initial_mps:  Starting MPS as FiniteMPS or TensorNetwork (backward compat).
                      The result MPS is returned in DMRGResult.
        config:       DMRGConfig parameters.

    Returns:
        DMRGResult with energy, sweep history, optimized FiniteMPS, and diagnostics.
    """
    if config.subspace_expansion and config.two_site:
        raise ValueError(
            "subspace_expansion=True requires two_site=False. "
            "DMRG3S enrichment is a 1-site algorithm."
        )
    L = hamiltonian.n_nodes()
    if L < 2:
        raise ValueError(
            f"DMRG requires at least 2 sites, got L={L}. "
            "For a single site, diagonalize the operator directly."
        )

    # Convert TensorNetwork to FiniteMPS if needed
    if isinstance(initial_mps, TensorNetwork):
        _mps_tensors = [initial_mps.get_tensor(i) for i in range(L)]
        initial_mps = FiniteMPS.from_tensors(_mps_tensors)

    mps_tensors: list[Tensor] = list(initial_mps.tensors)
    mpo_tensors = [hamiltonian.get_tensor(i) for i in range(L)]

    # Select backend: symmetric when both MPS and MPO are all SymmetricTensor
    all_mps_sym = all(isinstance(t, SymmetricTensor) for t in mps_tensors)
    all_mpo_sym = all(isinstance(t, SymmetricTensor) for t in mpo_tensors)
    all_mps_dense = all(isinstance(t, DenseTensor) for t in mps_tensors)
    all_mpo_dense = all(isinstance(t, DenseTensor) for t in mpo_tensors)

    if all_mps_sym and all_mpo_sym:
        use_symmetric = True
        ops = _symmetric_ops(config)
    elif all_mps_dense and all_mpo_dense:
        use_symmetric = False
        ops = _dense_ops()
    else:
        raise TypeError(
            "dmrg() requires uniform tensor types: all DenseTensor or all "
            "SymmetricTensor. Got mixed types — convert explicitly."
        )

    # --- Accelerator dispatch: JIT-compiled sweep via lax.scan -----------
    if config.accelerator not in ("auto", "jit", "off"):
        raise ValueError(
            f"accelerator must be 'auto', 'jit', or 'off', got {config.accelerator!r}"
        )

    use_jit = False
    if config.accelerator == "jit":
        use_jit = True
    elif config.accelerator == "auto":
        device = jax.devices()[0].platform
        if device in ("gpu", "tpu"):
            use_jit = True
        elif not use_symmetric:
            use_jit = True

    if use_jit and config.two_site and not use_symmetric:
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        # Check if all bonds are already at chi_max.
        # When bonds are still growing (e.g. initial chi < chi_max), running
        # the JIT path wastes compute on padded-to-chi_max tensors.  Instead
        # we run Python sweeps first (warmup) and switch to JIT once all
        # bonds have saturated at chi_max.
        all_saturated = all(
            mps_tensors[i].todense().shape[-1] >= config.max_bond_dim
            for i in range(L - 1)
        )

        warmup_energies: list[float] = []
        warmup_trunc_errs: list[float] = []
        warmup_sweeps = 0
        warmup_converged = False

        if not all_saturated:
            # Phase 1: Warmup with Python sweeps until chi saturates.
            w_ops = _dense_ops()
            left_envs = _build_left_environments_list(
                mps_tensors, mpo_tensors, L, w_ops
            )
            right_envs = _build_right_environments_list(
                mps_tensors, mpo_tensors, L, w_ops
            )
            energy = 0.0

            for sweep in range(config.num_sweeps):
                prev_energy = energy

                if sweep > 0:
                    left_envs = _build_left_environments_list(
                        mps_tensors, mpo_tensors, L, w_ops
                    )

                # Left-to-right half-sweep
                for i in range(L - 1):
                    l_env = left_envs[i]
                    assert l_env is not None
                    _r = right_envs[i + 2]
                    r_env = _r if _r is not None else w_ops.build_trivial_right_env()
                    theta, e = w_ops.two_site_update(
                        mps_tensors[i],
                        mps_tensors[i + 1],
                        l_env,
                        mpo_tensors[i],
                        mpo_tensors[i + 1],
                        r_env,
                        config,
                    )
                    energy = float(e)

                    A, s, B, trunc_err = _svd_and_truncate_site(theta, i, config)
                    mps_tensors[i] = A
                    mps_tensors[i + 1] = B
                    warmup_trunc_errs.append(float(trunc_err))

                    left_envs[i + 1] = w_ops.update_left_env(l_env, A, mpo_tensors[i])

                # Rebuild right environments before R->L half-sweep
                right_envs = _build_right_environments_list(
                    mps_tensors, mpo_tensors, L, w_ops
                )

                # Right-to-left half-sweep
                for i in range(L - 2, -1, -1):
                    l_env = left_envs[i]
                    assert l_env is not None
                    _r2 = right_envs[i + 2]
                    r2_env = _r2 if _r2 is not None else w_ops.build_trivial_right_env()
                    theta, e = w_ops.two_site_update(
                        mps_tensors[i],
                        mps_tensors[i + 1],
                        l_env,
                        mpo_tensors[i],
                        mpo_tensors[i + 1],
                        r2_env,
                        config,
                    )
                    energy = float(e)

                    A, s, B, trunc_err = _svd_and_truncate_site(
                        theta, i, config, sweep_right=False
                    )
                    mps_tensors[i] = A
                    mps_tensors[i + 1] = B
                    warmup_trunc_errs.append(float(trunc_err))

                    right_envs[i + 1] = w_ops.update_right_env(
                        r2_env, B, mpo_tensors[i + 1]
                    )

                warmup_energies.append(energy)
                warmup_sweeps += 1
                if config.verbose:
                    print(f"Warmup sweep {sweep + 1}: E = {energy:.10f}")

                # Check if all bonds have reached chi_max
                all_saturated = all(
                    mps_tensors[i].todense().shape[-1] >= config.max_bond_dim
                    for i in range(L - 1)
                )
                if all_saturated:
                    break

                # Check convergence during warmup
                if sweep > 0 and abs(energy - prev_energy) < config.convergence_tol:
                    warmup_converged = True
                    break

        # Phase 2: JIT sweeps for remaining budget (if not already converged)
        remaining = config.num_sweeps - warmup_sweeps
        jit_energies: list[float] = []
        if remaining > 0 and not warmup_converged:
            raw_mps = [t.todense() for t in mps_tensors]
            raw_mpo = [t.todense() for t in mpo_tensors]

            jit_energies, mps_out_raw = jit_dmrg_sweep_dense(
                raw_mps,
                raw_mpo,
                chi_max=config.max_bond_dim,
                num_sweeps=remaining,
                lanczos_max_iter=config.lanczos_max_iter,
            )

            # Reconstruct DenseTensor wrappers from JIT output arrays.
            # The JIT path pads tensors to (chi_max, d, chi_max). Create new
            # indices that match the padded shape while preserving labels.
            sym = U1Symmetry()
            result_mps_tensors = []
            for i, orig_t in enumerate(mps_tensors):
                new_indices = []
                for leg_idx, orig_idx in enumerate(orig_t.indices):
                    padded_dim = mps_out_raw[i].shape[leg_idx]
                    if padded_dim == orig_idx.dim:
                        new_indices.append(orig_idx)
                    else:
                        new_charges = np.zeros(padded_dim, dtype=np.int32)
                        new_indices.append(
                            TensorIndex(
                                sym,
                                new_charges,
                                orig_idx.flow,
                                label=orig_idx.label,
                            )
                        )
                result_mps_tensors.append(
                    DenseTensor(mps_out_raw[i], tuple(new_indices))
                )
            mps_tensors = result_mps_tensors

        all_energies = warmup_energies + jit_energies
        final_energy = all_energies[-1] if all_energies else 0.0
        converged = warmup_converged or (
            len(all_energies) >= 2
            and abs(all_energies[-1] - all_energies[-2]) < config.convergence_tol
        )
        result_mps = FiniteMPS.from_tensors(mps_tensors)

        return DMRGResult(
            energy=final_energy,
            energies_per_sweep=all_energies,
            mps=result_mps,
            truncation_errors=warmup_trunc_errs,
            converged=converged,
        )
    if use_jit and config.two_site and use_symmetric:
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_symmetric

        energies, mps_out = jit_dmrg_sweep_symmetric(
            list(mps_tensors),
            list(mpo_tensors),
            chi_max=config.max_bond_dim,
            num_sweeps=config.num_sweeps,
            lanczos_max_iter=config.lanczos_max_iter,
        )

        result_mps = FiniteMPS.from_tensors(mps_out)

        converged = (
            len(energies) >= 2
            and abs(energies[-1] - energies[-2]) < config.convergence_tol
        )
        return DMRGResult(
            energy=energies[-1] if energies else 0.0,
            energies_per_sweep=energies,
            mps=result_mps,
            truncation_errors=[],
            converged=converged,
        )

    # (If use_jit is True but conditions not met, fall through silently
    #  to the Python sweep loop below.)

    # Validate initial MPS sector if target_charge is specified
    if config.target_charge is not None and use_symmetric:
        validate_mps_sector(mps_tensors, config.target_charge)

    # Right-canonicalize MPS before building environments (1-site mode).
    # The L→R sweep needs right environments built from right-canonical MPS.
    if not config.two_site:
        _rc_mps = FiniteMPS.from_tensors(mps_tensors).right_canonicalize()
        mps_tensors = list(_rc_mps.tensors)

    # Build left environments (L[i] = trivial for i=0)
    left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L, ops)
    right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L, ops)

    energies_per_sweep: list[float] = []
    truncation_errors: list[float] = []
    energy = 0.0
    converged = False

    # Special case: single-site system (L=1). Both sweep loops iterate
    # range(0) and range(-1, -1, -1) respectively, so no optimisation
    # happens and the energy stays at the default 0.0.  Solve once.
    if L == 1:
        l_env = left_envs[0]
        assert l_env is not None
        r_env = (
            right_envs[1]
            if right_envs[1] is not None
            else ops.build_trivial_right_env()
        )
        new_site, e = ops.one_site_update(
            mps_tensors[0],
            l_env,
            mpo_tensors[0],
            r_env,
            config,
        )
        energy = float(e)
        mps_tensors[0] = new_site
        converged = True
        energies_per_sweep.append(energy)

    _current_alpha = config.mixing_factor if config.subspace_expansion else 0.0

    for sweep in range(config.num_sweeps if L > 1 else 0):
        prev_energy = energy

        # Rebuild left environments from updated MPS before left-to-right sweep
        if sweep > 0:
            # After R→L, site 0 has accumulated L factors. Normalize.
            if not config.two_site:
                _n0 = float(jnp.sqrt(jnp.abs(inner(mps_tensors[0], mps_tensors[0]))))
                if _n0 > 1e-15 and abs(_n0 - 1.0) > 1e-10:
                    mps_tensors[0] = mps_tensors[0] * (1.0 / _n0)
            left_envs = _build_left_environments_list(mps_tensors, mpo_tensors, L, ops)

        # Left-to-right sweep
        for i in range(L - 1):
            l_env = left_envs[i]
            assert l_env is not None
            if config.two_site:
                _r = right_envs[i + 2]
                r_env = _r if _r is not None else ops.build_trivial_right_env()
                theta, e = ops.two_site_update(
                    mps_tensors[i],
                    mps_tensors[i + 1],
                    l_env,
                    mpo_tensors[i],
                    mpo_tensors[i + 1],
                    r_env,
                    config,
                )
                energy = float(e)

                A, s, B, trunc_err = _svd_and_truncate_site(theta, i, config)
                mps_tensors[i] = A
                mps_tensors[i + 1] = B
                truncation_errors.append(float(trunc_err))

                left_envs[i + 1] = ops.update_left_env(l_env, A, mpo_tensors[i])
            else:
                _ri = right_envs[i + 1]
                r_env_1s = _ri if _ri is not None else ops.build_trivial_right_env()
                new_site, e = ops.one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r_env_1s,
                    config,
                )
                energy = float(e)

                if config.subspace_expansion and i < L - 1:
                    if use_symmetric:
                        from tenax.algorithms.dmrg3s import (
                            expand_and_truncate_symmetric,
                        )

                        mps_tensors[i], mps_tensors[i + 1] = (
                            expand_and_truncate_symmetric(
                                new_site,
                                mps_tensors[i + 1],
                                l_env,
                                mpo_tensors[i],
                                alpha=_current_alpha,
                                max_bond_dim=config.max_bond_dim,
                                direction="left_to_right",
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        )
                    else:
                        if config.hybrid_mixing:
                            from tenax.algorithms.dmrg3s import (
                                expand_and_truncate_hybrid,
                            )

                            A_arr, B_arr = expand_and_truncate_hybrid(
                                new_site.todense(),
                                mps_tensors[i + 1].todense(),
                                l_env,
                                mpo_tensors[i],
                                max_bond_dim=config.max_bond_dim,
                                direction="left_to_right",
                                num_extra=config.expansion_num_extra,
                                hybrid=True,
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        else:
                            from tenax.algorithms.dmrg3s import (
                                expand_and_truncate_dense,
                            )

                            A_arr, B_arr = expand_and_truncate_dense(
                                new_site.todense(),
                                mps_tensors[i + 1].todense(),
                                l_env,
                                mpo_tensors[i],
                                alpha=_current_alpha,
                                max_bond_dim=config.max_bond_dim,
                                direction="left_to_right",
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        mps_tensors[i] = _rebuild_dense_tensor(
                            A_arr, new_site.indices, bond_pos=-1
                        )
                        mps_tensors[i + 1] = _rebuild_dense_tensor(
                            B_arr, mps_tensors[i + 1].indices, bond_pos=0
                        )
                else:
                    # QR + absorb R + build env (atomic step for 1-site)
                    right_bond = f"v{i}_{i + 1}"
                    left_labels = [lb for lb in new_site.labels() if lb != right_bond]
                    tmp_bond = f"_qr_{right_bond}"
                    Q, R = _linalg_qr(
                        new_site,
                        left_labels,
                        [right_bond],
                        new_bond_label=tmp_bond,
                    )
                    mps_tensors[i] = Q.relabel(tmp_bond, right_bond)
                    absorbed = contract(R, mps_tensors[i + 1])
                    mps_tensors[i + 1] = absorbed.relabel(tmp_bond, right_bond)

                left_envs[i + 1] = ops.update_left_env(
                    l_env, mps_tensors[i], mpo_tensors[i]
                )

        # After L→R, site L-1 has accumulated R factors. Normalize it
        # so right environments are built from a unit-norm MPS.
        if not config.two_site:
            _n = float(jnp.sqrt(jnp.abs(inner(mps_tensors[L - 1], mps_tensors[L - 1]))))
            if _n > 1e-15 and abs(_n - 1.0) > 1e-10:
                mps_tensors[L - 1] = mps_tensors[L - 1] * (1.0 / _n)

        # Rebuild right environments from updated MPS before right-to-left sweep
        right_envs = _build_right_environments_list(mps_tensors, mpo_tensors, L, ops)

        # Right-to-left sweep
        for i in range(L - 2, -1, -1):
            l_env = left_envs[i]
            assert l_env is not None
            _r2 = right_envs[i + 2]
            r2_env = _r2 if _r2 is not None else ops.build_trivial_right_env()
            if config.two_site:
                theta, e = ops.two_site_update(
                    mps_tensors[i],
                    mps_tensors[i + 1],
                    l_env,
                    mpo_tensors[i],
                    mpo_tensors[i + 1],
                    r2_env,
                    config,
                )
                energy = float(e)

                A, s, B, trunc_err = _svd_and_truncate_site(
                    theta, i, config, sweep_right=False
                )
                mps_tensors[i] = A
                mps_tensors[i + 1] = B
                truncation_errors.append(float(trunc_err))

                right_envs[i + 1] = ops.update_right_env(r2_env, B, mpo_tensors[i + 1])
            else:
                _r1 = right_envs[i + 1]
                r1_env = _r1 if _r1 is not None else ops.build_trivial_right_env()
                new_site, e = ops.one_site_update(
                    mps_tensors[i],
                    l_env,
                    mpo_tensors[i],
                    r1_env,
                    config,
                )
                energy = float(e)

                if config.subspace_expansion and i > 0:
                    if use_symmetric:
                        from tenax.algorithms.dmrg3s import (
                            expand_and_truncate_symmetric,
                        )

                        mps_tensors[i], mps_tensors[i - 1] = (
                            expand_and_truncate_symmetric(
                                new_site,
                                mps_tensors[i - 1],
                                r1_env,
                                mpo_tensors[i],
                                alpha=_current_alpha,
                                max_bond_dim=config.max_bond_dim,
                                direction="right_to_left",
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        )
                    else:
                        if config.hybrid_mixing:
                            from tenax.algorithms.dmrg3s import (
                                expand_and_truncate_hybrid,
                            )

                            B_arr, A_arr = expand_and_truncate_hybrid(
                                new_site.todense(),
                                mps_tensors[i - 1].todense(),
                                r1_env,
                                mpo_tensors[i],
                                max_bond_dim=config.max_bond_dim,
                                direction="right_to_left",
                                num_extra=config.expansion_num_extra,
                                hybrid=True,
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        else:
                            from tenax.algorithms.dmrg3s import (
                                expand_and_truncate_dense,
                            )

                            B_arr, A_arr = expand_and_truncate_dense(
                                new_site.todense(),
                                mps_tensors[i - 1].todense(),
                                r1_env,
                                mpo_tensors[i],
                                alpha=_current_alpha,
                                max_bond_dim=config.max_bond_dim,
                                direction="right_to_left",
                                svd_trunc_err=config.svd_trunc_err,
                            )
                        mps_tensors[i] = _rebuild_dense_tensor(
                            B_arr, new_site.indices, bond_pos=0
                        )
                        mps_tensors[i - 1] = _rebuild_dense_tensor(
                            A_arr, mps_tensors[i - 1].indices, bond_pos=-1
                        )
                else:
                    # RQ + absorb L + build env (atomic step for 1-site)
                    left_bond = f"v{i - 1}_{i}"
                    site_labels = new_site.labels()
                    if i > 0 and left_bond in site_labels:
                        other_labels = [lb for lb in site_labels if lb != left_bond]
                        tmp_bond = f"_qr_{left_bond}"
                        Q, R = _linalg_qr(
                            new_site,
                            other_labels,
                            [left_bond],
                            new_bond_label=tmp_bond,
                        )
                        Q = Q.relabel(tmp_bond, left_bond)
                        # Reorder so left_bond is first (MPS convention)
                        q_labels = Q.labels()
                        bond_pos = q_labels.index(left_bond)
                        if bond_pos != 0:
                            axes = (bond_pos,) + tuple(
                                j for j in range(len(q_labels)) if j != bond_pos
                            )
                            Q = Q.transpose(axes)
                        mps_tensors[i] = Q
                        absorbed = contract(mps_tensors[i - 1], R)
                        mps_tensors[i - 1] = absorbed.relabel(tmp_bond, left_bond)
                    else:
                        mps_tensors[i] = new_site
                right_envs[i] = ops.update_right_env(
                    r1_env, mps_tensors[i], mpo_tensors[i]
                )

        energies_per_sweep.append(energy)
        if config.verbose:
            print(f"Sweep {sweep + 1}/{config.num_sweeps}: E = {energy:.10f}")

        # Validate sector preservation after each sweep
        if config.target_charge is not None and use_symmetric:
            sector = compute_mps_sector(mps_tensors)
            if sector != config.target_charge:
                raise RuntimeError(
                    f"Sector drift detected after sweep {sweep + 1}: "
                    f"MPS sector={sector}, expected target_charge={config.target_charge}."
                )

        # Check convergence
        if sweep > 0 and abs(energy - prev_energy) < config.convergence_tol:
            converged = True
            if config.verbose:
                print(f"Converged at sweep {sweep + 1}")
            break

    # Build result MPS as FiniteMPS.
    # Convert any BlockArray tensors back to SymmetricTensor for storage.
    from tenax.algorithms._block_array import BlockArray, ba_to_symmetric

    final_tensors = [
        ba_to_symmetric(t) if isinstance(t, BlockArray) else t for t in mps_tensors
    ]
    if not config.two_site:
        orth_center = 0  # after final R→L sweep
    else:
        orth_center = None  # 2-site doesn't maintain strict canonical form
    result_mps = FiniteMPS.from_tensors(final_tensors, orth_center=orth_center)

    return DMRGResult(
        energy=energy,
        energies_per_sweep=energies_per_sweep,
        mps=result_mps,
        truncation_errors=truncation_errors,
        converged=converged,
    )


def _rebuild_dense_tensor(
    data: jax.Array,
    old_indices: tuple,
    bond_pos: int,
) -> DenseTensor:
    """Rebuild a DenseTensor when the bond dimension has changed.

    Creates a new TensorIndex at ``bond_pos`` with the correct dimension,
    preserving the symmetry, flow, and label from the original index.
    """
    idx = bond_pos if bond_pos >= 0 else len(old_indices) + bond_pos
    new_dim = data.shape[idx]
    old_idx = old_indices[idx]

    if old_idx.dim == new_dim:
        return DenseTensor(data, old_indices)

    new_bond_idx = TensorIndex(
        symmetry=old_idx.symmetry,
        charges=np.zeros(new_dim, dtype=np.int32),
        flow=old_idx.flow,
        label=old_idx.label,
    )
    new_indices = old_indices[:idx] + (new_bond_idx,) + old_indices[idx + 1 :]
    return DenseTensor(data, new_indices)


def _find_left_bond(labels: tuple, site: int) -> str | None:
    """Find the left virtual bond label for a given site."""
    for lbl in labels:
        if isinstance(lbl, str) and lbl.startswith(f"v{site - 1}_"):
            return lbl
    return None


def _find_right_bond(labels: tuple, site: int) -> str | None:
    """Find the right virtual bond label for a given site."""
    for lbl in labels:
        if isinstance(lbl, str) and lbl.startswith(f"v{site}_"):
            return lbl
    return None


def _build_left_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
    ops: SweepOps | None = None,
) -> list[Tensor | None]:
    """Build all left environment tensors by sweeping left to right.

    L_env[0] = trivial, L_env[i] = contraction of sites 0..i-1.

    Returns list of L+1 environment tensors (None used as placeholder where
    not yet computed; replaced with dense contractions in full implementation).
    """
    if ops is None:
        ops = _dense_ops()
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[0] = ops.build_trivial_left_env()

    for i in range(L - 1):
        env = envs[i]
        if env is not None:
            envs[i + 1] = ops.update_left_env(env, mps_tensors[i], mpo_tensors[i])

    return envs


def _build_right_environments_list(
    mps_tensors: list[Tensor],
    mpo_tensors: list[Tensor],
    L: int,
    ops: SweepOps | None = None,
) -> list[Tensor | None]:
    """Build all right environment tensors by sweeping right to left."""
    if ops is None:
        ops = _dense_ops()
    envs: list[Tensor | None] = [None] * (L + 1)
    envs[L] = ops.build_trivial_right_env()

    for i in range(L - 1, 0, -1):
        env = envs[i + 1]
        if env is not None:
            envs[i] = ops.update_right_env(env, mps_tensors[i], mpo_tensors[i])

    return envs


def _build_trivial_left_env(dtype=None) -> DenseTensor:
    """Build trivial (1x1x1) left boundary environment."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(jnp.ones((1, 1, 1), dtype=dtype), indices)


def _build_trivial_right_env(dtype=None) -> DenseTensor:
    """Build trivial (1x1x1) right boundary environment."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_conj_r"),
    )
    return DenseTensor(jnp.ones((1, 1, 1), dtype=dtype), indices)


def _update_left_env(
    left_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> DenseTensor:
    """Update left environment by absorbing one MPS/MPO site.

    Contracts: new_L[r, w, r'] = L[l, w_l, l'] * A[l, p, r] * W[w_l, p, p', w] * A*[l', p', r']

    Args:
        left_env: Current left environment tensor.
        mps_site: MPS site tensor A.
        mpo_site: MPO site tensor W.

    Returns:
        Updated left environment tensor.
    """
    # Dense implementation using todense() for generality
    L_dense = left_env.todense()  # shape (chi_l, D_w, chi_l')
    A_dense = mps_site.todense()  # shape (chi_l, d, chi_r) — always 3D
    W_dense = mpo_site.todense()  # shape (D_w_l, d_top, d_bot, D_w_r)

    # new_L[chi_r, D_w_r, chi_r'] =
    #   L[chi_l, D_w_l, chi_l'] * A[chi_l, d, chi_r] * W[D_w_l, d, d', D_w_r] * A*[chi_l', d', chi_r']
    # Using subscripts: L=abc (a=chi_l, b=D_w_l, c=chi_l')
    #                   A=apd (a=chi_l, p=d_ket, d=chi_r)
    #                   W=bpxe (b=D_w_l, p=d_ket, x=d_bra, e=D_w_r)
    #                   A*=cxf (c=chi_l', x=d_bra, f=chi_r')
    # -> new_L[d, e, f] = (chi_r, D_w_r, chi_r')
    new_L = jnp.einsum(
        "abc,apd,bpxe,cxf->def",
        L_dense,
        A_dense,
        W_dense,
        jnp.conj(A_dense),
    )

    sym = U1Symmetry()
    bond_r = np.zeros(new_L.shape[0], dtype=np.int32)
    bond_w = np.zeros(new_L.shape[1], dtype=np.int32)
    indices = (
        TensorIndex(sym, bond_r, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond_w, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond_r, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(new_L, indices)


def _update_right_env(
    right_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> DenseTensor:
    """Update right environment by absorbing one MPS/MPO site."""
    R_dense = right_env.todense()  # shape (chi_r, D_w, chi_r')
    B_dense = mps_site.todense()  # shape (chi_l, d, chi_r) — always 3D
    W_dense = mpo_site.todense()  # shape (D_w_l, d_top, d_bot, D_w_r)

    # new_R[chi_l, D_w_l, chi_l'] =
    #   R[chi_r, D_w_r, chi_r'] * B[chi_l, d, chi_r] * W[D_w_l, d, d', D_w_r] * B*[chi_l', d', chi_r']
    # R=abc (a=chi_r, b=D_w_r, c=chi_r')
    # B=dpa (d=chi_l, p=d_ket, a=chi_r)   [contracted on a]
    # W=epxb (e=D_w_l, p=d_ket, x=d_bra, b=D_w_r)  [contracted on a,b]
    # B*=fxc (f=chi_l', x=d_bra, c=chi_r')  [contracted on c]
    # -> new_R[d, e, f] = (chi_l, D_w_l, chi_l')
    new_R = jnp.einsum(
        "abc,dpa,epxb,fxc->def",
        R_dense,
        B_dense,
        W_dense,
        jnp.conj(B_dense),
    )

    sym = U1Symmetry()
    bond_l = np.zeros(new_R.shape[0], dtype=np.int32)
    bond_w = np.zeros(new_R.shape[1], dtype=np.int32)
    indices = (
        TensorIndex(sym, bond_l, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond_w, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond_l, FlowDirection.IN, label="env_mps_conj_r"),
    )
    return DenseTensor(new_R, indices)


def _effective_hamiltonian_matvec(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply effective Hamiltonian H_eff to 2-site wavefunction theta.

    H_eff = L * W_l * W_r * R (diagrammatic notation).
    All inputs are raw JAX arrays (flattened for JIT compatibility).

    This function is @jax.jit compiled for performance.

    Args:
        theta_flat:  Flattened 2-site wavefunction.
        theta_shape: Shape tuple for reshaping.
        L_env:       Left environment, shape (chi_l, d_w_l, chi_l).
        W_l:         Left MPO site, shape (d_w_l, d_p_l, d_p_l', d_w_m).
        W_r:         Right MPO site, shape (d_w_m, d_p_r, d_p_r', d_w_r).
        R_env:       Right environment, shape (chi_r, d_w_r, chi_r).

    Returns:
        Flattened result of H_eff @ theta.
    """
    theta = theta_flat.reshape(theta_shape)

    # Contract: L[a,b,c] * theta[a,p,q,d] * W_l[b,p,s,e] * W_r[e,q,t,f] * R[d,f,g]
    # -> result[c,s,t,g]
    # Indices:
    #   a = chi_l (MPS bond, ket)
    #   b = D_w_l (MPO bond left)
    #   c = chi_l (MPS bond, bra)
    #   p = d_phys_l (ket physical, left site)
    #   q = d_phys_r (ket physical, right site)
    #   d = chi_r (MPS bond right, ket)
    #   s = d_phys_l' (bra physical, left site)
    #   e = D_w_m (MPO bond middle)
    #   t = d_phys_r' (bra physical, right site)
    #   f = D_w_r (MPO bond right)
    #   g = chi_r (MPS bond right, bra)
    result = jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env,
        theta,
        W_l,
        W_r,
        R_env,
    )
    return result.ravel()


_matvec_jit = jax.jit(_effective_hamiltonian_matvec, static_argnums=(1,))


def _two_site_update(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 2-site DMRG update: contract theta, solve eigenvalue problem.

    Returns:
        (theta_opt, energy) where theta_opt is the optimized 2-site tensor.
    """
    # Contract theta = A[i] * A[i+1] (shared virtual bond contracted)
    shared = set(site_l.labels()) & set(site_r.labels())
    if shared:
        theta = contract(site_l, site_r)
    else:
        # No shared label: concatenate (this shouldn't happen in a valid MPS)
        theta = site_l

    # Use Lanczos to find the ground state
    theta_dense = theta.todense()  # always 4D: (chi_l, d_l, d_r, chi_r)
    theta_indices = theta.indices

    L_arr = left_env.todense()
    R_arr = right_env.todense()
    W_l_arr = mpo_l.todense()
    W_r_arr = mpo_r.todense()

    # Ensure environments are 3D
    if L_arr.ndim == 1:
        L_arr = L_arr.reshape(1, 1, 1)
    if R_arr.ndim == 1:
        R_arr = R_arr.reshape(1, 1, 1)

    theta_shape = theta_dense.shape
    theta_flat = theta_dense.ravel()

    def matvec(v: jax.Array) -> jax.Array:
        return _matvec_jit(v, theta_shape, L_arr, W_l_arr, W_r_arr, R_arr)

    energy, theta_opt_flat = _lanczos_solve(
        matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
    )

    theta_opt_dense = theta_opt_flat.reshape(theta_shape)
    theta_opt = DenseTensor(theta_opt_dense, theta_indices)
    return theta_opt, energy


def _one_site_update(
    site: Tensor,
    left_env: Tensor,
    mpo_site: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 1-site DMRG update."""
    site_dense = site.todense()  # always 3D: (chi_l, d, chi_r)
    site_shape = site_dense.shape
    site_flat = site_dense.ravel()

    L_arr = left_env.todense()
    R_arr = right_env.todense()
    W_arr = mpo_site.todense()

    if L_arr.ndim == 1:
        L_arr = L_arr.reshape(1, 1, 1)
    if R_arr.ndim == 1:
        R_arr = R_arr.reshape(1, 1, 1)

    def matvec(v: jax.Array) -> jax.Array:
        s = v.reshape(site_shape)
        # H_eff = L[a,b,c] * s[a,p,d] * W[b,p,x,e] * R[d,e,f] -> result[c,x,f]
        # a=chi_l_ket, b=D_w_l, c=chi_l_bra, p=d_ket, d=chi_r_ket,
        # x=d_bra, e=D_w_r, f=chi_r_bra
        result = jnp.einsum("abc,apd,bpxe,def->cxf", L_arr, s, W_arr, R_arr)
        return result.ravel()

    energy, site_opt_flat = _lanczos_solve(
        matvec, site_flat, config.lanczos_max_iter, config.lanczos_tol
    )

    site_opt_dense = site_opt_flat.reshape(site_shape)
    site_opt = DenseTensor(site_opt_dense, site.indices)
    return site_opt, energy


def _lanczos_solve(
    matvec: Callable[[jax.Array], jax.Array],
    initial_vector: jax.Array,
    num_steps: int,
    tol: float,
) -> tuple[float, jax.Array]:
    """Lanczos eigensolver for the smallest eigenvalue.

    Optimizations over the naive implementation:
    - Keeps alpha/beta as JAX scalars to avoid host-device sync per step
    - Vectorized eigenvector reconstruction via jnp.tensordot on stacked basis

    Args:
        matvec:         Function applying the effective Hamiltonian.
        initial_vector: Starting vector (will be normalized).
        num_steps:      Maximum number of Lanczos steps.
        tol:            Convergence tolerance on the residual.

    Returns:
        (eigenvalue, eigenvector) for the ground state.
    """
    v = initial_vector / (jnp.linalg.norm(initial_vector) + 1e-15)

    # Krylov basis and tridiagonal matrix coefficients
    basis = [v]
    alphas_jax: list[jax.Array] = []
    betas_jax: list[jax.Array] = [jnp.zeros(())]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha = jnp.dot(basis[-1].conj(), w).real
        alphas_jax.append(alpha)

        w = w - alpha * basis[-1]
        if step > 0:
            w = w - betas_jax[-1] * basis[-2]

        # Full reorthogonalization against all previous basis vectors
        # to prevent loss of orthogonality and ghost eigenvalues.
        for q in basis:
            w = w - jnp.dot(q.conj(), w) * q

        beta = jnp.linalg.norm(w)
        betas_jax.append(beta)

        # Convergence check requires host sync (unavoidable for loop control)
        if float(beta) < tol:
            break

        basis.append(w / beta)

    # Build tridiagonal matrix and find ground state
    n = len(alphas_jax)

    if n == 0:
        # No iterations completed — return initial vector with zero energy
        return 0.0, v

    if n == 1:
        # Single iteration: eigenvalue is alpha, eigenvector is first basis vector
        return float(alphas_jax[0]), basis[0]

    alphas_arr = jnp.stack(alphas_jax)
    betas_arr = jnp.stack(betas_jax[1:n])
    T = jnp.diag(alphas_arr) + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    idx = jnp.argmin(eigvals)
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Vectorized eigenvector reconstruction: stack basis and contract
    # basis may have n+1 entries (the last one was added but has no alpha);
    # krylov_coefs has length n, so slice basis to match.
    basis_stacked = jnp.stack(basis[:n], axis=0)  # (n, vec_dim)
    eigenvector = jnp.tensordot(krylov_coefs, basis_stacked, axes=1)
    eigenvector = eigenvector / (jnp.linalg.norm(eigenvector) + 1e-15)

    return eigenvalue, eigenvector


def _lanczos_solve_jit(
    matvec: Callable[[jax.Array], jax.Array],
    initial_vector: jax.Array,
    num_steps: int,
) -> tuple[float, jax.Array]:
    """JIT-compiled Lanczos eigensolver using lax.fori_loop.

    Compiles the entire Lanczos iteration into a single XLA program,
    eliminating per-iteration Python dispatch overhead. ~120x faster
    than the Python-loop version for dense tensors.

    Note: does NOT support early termination (runs all num_steps).
    Use the Python-loop version (_lanczos_solve) when early convergence
    detection is important.
    """
    n = initial_vector.shape[0]
    v0 = initial_vector / (jnp.linalg.norm(initial_vector) + 1e-15)

    basis = jnp.zeros((num_steps, n), dtype=v0.dtype)
    basis = basis.at[0].set(v0)
    alphas = jnp.zeros(num_steps, dtype=v0.dtype)
    betas = jnp.zeros(num_steps, dtype=v0.dtype)

    def body(step, state):
        basis, alphas, betas = state
        v = basis[step]
        w = matvec(v)
        alpha = jnp.dot(v.conj(), w).real
        alphas = alphas.at[step].set(alpha)

        w = w - alpha * v
        v_prev = jnp.where(step > 0, basis[step - 1], jnp.zeros_like(v))
        beta_prev = jnp.where(step > 0, betas[step - 1], 0.0)
        w = w - beta_prev * v_prev

        # Full reorthogonalization via scan
        def reorth_step(w, q):
            overlap = jnp.dot(q.conj(), w)
            return w - overlap * q, None

        w, _ = jax.lax.scan(reorth_step, w, basis)

        beta = jnp.linalg.norm(w)
        betas = betas.at[step].set(beta)
        v_new = jnp.where(beta > 1e-15, w / beta, jnp.zeros_like(w))
        basis = jnp.where(
            step + 1 < num_steps,
            basis.at[step + 1].set(v_new),
            basis,
        )
        return basis, alphas, betas

    basis, alphas, betas = jax.lax.fori_loop(0, num_steps, body, (basis, alphas, betas))

    T = jnp.diag(alphas) + jnp.diag(betas[:-1], k=1) + jnp.diag(betas[:-1], k=-1)
    eigvals, eigvecs = jnp.linalg.eigh(T)
    coefs = eigvecs[:, 0]
    eigenvector = coefs @ basis
    eigenvector = eigenvector / (jnp.linalg.norm(eigenvector) + 1e-15)
    return float(eigvals[0]), eigenvector


def _svd_and_truncate_site(
    theta: Tensor,
    site: int,
    config: DMRGConfig,
    sweep_right: bool = True,
) -> tuple[Tensor, jax.Array, Tensor, float]:
    """SVD of 2-site tensor and truncation.

    Computes SVD once via truncated_svd, then derives the truncation error
    from the full singular values returned by that same decomposition.

    Args:
        theta:       2-site wavefunction tensor.
        site:        Left site index.
        config:      DMRGConfig.
        sweep_right: If True, left site gets orthogonality center (A-form);
                     if False, right site gets it (B-form).

    Returns:
        (A_tensor, singular_values, B_tensor, truncation_error)
    """
    from tenax.algorithms._block_array import BlockArray

    if isinstance(theta, BlockArray):
        labels = tuple(idx.label for idx in theta.indices)
    else:
        labels = theta.labels()

    # Find physical and virtual labels
    # With uniform 3-leg tensors, site 0 has left bond "v_-1_0"
    if site > 0:
        left_virt = f"v{site - 1}_{site}"
    else:
        left_virt = "v_-1_0"
    right_virt = f"v{site + 1}_{site + 2}"
    left_phys = f"p{site}"
    right_phys = f"p{site + 1}"

    # Build actual left/right label splits based on what's available
    left_candidates = {left_virt, left_phys}
    right_candidates = {right_virt, right_phys}
    left_labels = [lbl for lbl in labels if lbl in left_candidates]
    right_labels = [lbl for lbl in labels if lbl in right_candidates]

    if not left_labels or not right_labels:
        # Fallback: split roughly in half
        n = len(labels)
        left_labels = list(labels[: n // 2])
        right_labels = list(labels[n // 2 :])

    bond_label = f"v{site}_{site + 1}"

    # Dispatch to numpy-only SVD when numpy_blockwise is enabled.
    # Returns BlockArray directly — avoids JAX array creation in the sweep loop.
    # The sweep loop stores these as BlockArray; env updates accept either type.
    # Only converted to SymmetricTensor at the end of dmrg() for the result.
    from tenax.algorithms._block_array import BlockArray

    if config.numpy_blockwise and isinstance(theta, (SymmetricTensor, BlockArray)):
        from tenax.algorithms._block_array import ba_to_symmetric
        from tenax.linalg import _truncated_svd_symmetric_np

        # _truncated_svd_symmetric_np needs SymmetricTensor for index metadata
        theta_sym = ba_to_symmetric(theta) if isinstance(theta, BlockArray) else theta

        A_ba, s, B_ba, s_full = _truncated_svd_symmetric_np(
            theta_sym,
            left_labels=left_labels,
            right_labels=right_labels,
            max_singular_values=config.max_bond_dim,
            max_truncation_err=config.svd_trunc_err,
            new_bond_label=bond_label,
            normalize=False,
        )

        n_keep = len(s)
        if len(s_full) > n_keep:
            total_sq = np.sum(s_full**2)
            trunc_sq = np.sum(s_full[n_keep:] ** 2)
            trunc_err = float(np.sqrt(trunc_sq / (total_sq + 1e-15)))
        else:
            trunc_err = 0.0

        if sweep_right:
            B_ba = _scale_bond_axis_ba(B_ba, bond_label, s)
        else:
            A_ba = _scale_bond_axis_ba(A_ba, bond_label, s)

        return A_ba, s, B_ba, trunc_err

    # JAX path: single SVD via truncated_svd (handles both Dense and Symmetric)
    A, s, B, s_full = truncated_svd(
        theta,
        left_labels=left_labels,
        right_labels=right_labels,
        new_bond_label=bond_label,
        max_singular_values=config.max_bond_dim,
        max_truncation_err=config.svd_trunc_err,
    )

    # Compute truncation error from the full singular-value spectrum
    # returned by truncated_svd (no second SVD needed).
    n_keep = len(s)
    if len(s_full) > n_keep:
        total_sq = jnp.sum(s_full**2)
        trunc_sq = jnp.sum(s_full[n_keep:] ** 2)
        trunc_err = float(jnp.sqrt(trunc_sq / (total_sq + 1e-15)))
    else:
        trunc_err = 0.0

    # Absorb singular values into the tensor moving away from the
    # orthogonality center so the MPS stays in canonical form.
    if sweep_right:
        # Left-to-right: A = U (left-canonical), absorb s into B
        B = scale_bond_axis(B, bond_label, s)
    else:
        # Right-to-left: B = Vh (right-canonical), absorb s into A
        A = scale_bond_axis(A, bond_label, s)

    return A, s, B, trunc_err


# ------------------------------------------------------------------ #
# Symmetric (block-sparse) backend                                     #
# ------------------------------------------------------------------ #


def _build_trivial_left_env_symmetric(dtype=None) -> SymmetricTensor:
    """Build trivial (1x1x1) left boundary environment as SymmetricTensor."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_l"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mpo_l"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    blocks: dict[tuple[int, ...], jax.Array] = {
        (0, 0, 0): jnp.ones((1, 1, 1), dtype=dtype)
    }
    return SymmetricTensor(blocks, indices)


def _build_trivial_right_env_symmetric(dtype=None) -> SymmetricTensor:
    """Build trivial (1x1x1) right boundary environment as SymmetricTensor."""
    if dtype is None:
        dtype = jnp.float64
    sym = U1Symmetry()
    bond = np.zeros(1, dtype=np.int32)
    indices = (
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond, FlowDirection.IN, label="env_mps_conj_r"),
    )
    blocks: dict[tuple[int, ...], jax.Array] = {
        (0, 0, 0): jnp.ones((1, 1, 1), dtype=dtype)
    }
    return SymmetricTensor(blocks, indices)


def _lanczos_solve_tensor(
    matvec: Callable[[Tensor], Tensor],
    initial: Tensor,
    num_steps: int,
    tol: float,
) -> tuple[float, Tensor]:
    """Lanczos eigensolver operating on Tensor objects (dense or symmetric).

    Uses inner(), norm(), and Tensor arithmetic instead of flat JAX arrays.

    Args:
        matvec:   Function applying H_eff to a Tensor, returning a Tensor.
        initial:  Starting vector (Tensor).
        num_steps: Maximum Lanczos iterations.
        tol:      Convergence tolerance on the residual norm.

    Returns:
        (eigenvalue, eigenvector) for the ground state as Tensor.
    """
    v_norm = initial.norm()
    v = initial * (1.0 / (float(v_norm) + 1e-15))

    basis: list[Tensor] = [v]
    alphas: list[float] = []
    betas: list[float] = [0.0]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha_val = float(inner(basis[-1], w).real)
        alphas.append(alpha_val)

        w = w - basis[-1] * alpha_val
        if step > 0:
            w = w - basis[-2] * betas[-1]

        # Full reorthogonalization against all previous basis vectors.
        for q in basis:
            w = w - q * float(inner(q, w))

        beta_val = float(w.norm())
        betas.append(beta_val)

        if beta_val < tol:
            break

        basis.append(w * (1.0 / beta_val))

    n = len(alphas)
    if n == 0:
        return 0.0, v
    if n == 1:
        return alphas[0], basis[0]

    # Build tridiagonal matrix and diagonalize
    alphas_arr = jnp.array(alphas)
    betas_arr = jnp.array(betas[1:n])
    T = jnp.diag(alphas_arr) + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    idx = int(jnp.argmin(eigvals))
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Reconstruct eigenvector: sum(c_k * basis[k]) — can't stack SymmetricTensors
    eigenvector = basis[0] * float(krylov_coefs[0])
    for k in range(1, n):
        eigenvector = eigenvector + basis[k] * float(krylov_coefs[k])

    ev_norm = float(eigenvector.norm())
    eigenvector = eigenvector * (1.0 / (ev_norm + 1e-15))

    return eigenvalue, eigenvector


def _lanczos_solve_np(
    matvec: Callable,
    initial: BlockArray,
    num_steps: int,
    tol: float,
) -> tuple[float, BlockArray]:
    """Lanczos eigensolver on BlockArray — pure numpy, no JAX.

    Same algorithm as ``_lanczos_solve_tensor`` but operates on BlockArray
    objects using ``ba_*`` functions from ``_block_array.py``.

    Args:
        matvec:    Function applying H_eff to a BlockArray, returning a BlockArray.
        initial:   Starting vector (BlockArray).
        num_steps: Maximum Lanczos iterations.
        tol:       Convergence tolerance on the residual norm.

    Returns:
        (eigenvalue, eigenvector) for the ground state as BlockArray.
    """
    from tenax.algorithms._block_array import (
        ba_add,
        ba_inner,
        ba_norm,
        ba_scale,
        ba_sub_scaled,
    )

    v_nrm = ba_norm(initial)
    v = ba_scale(initial, 1.0 / (v_nrm + 1e-15))

    basis = [v]
    alphas = []
    betas = [0.0]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha_val = ba_inner(basis[-1], w)
        alphas.append(alpha_val)

        # w = w - alpha * v_k  (fused, no intermediate dict)
        if _USE_CYTHON_SUB:
            _cython_ba_sub_scaled_inplace(w.blocks, basis[-1].blocks, alpha_val)
        else:
            w = ba_sub_scaled(w, basis[-1], alpha_val)
        if step > 0:
            if _USE_CYTHON_SUB:
                _cython_ba_sub_scaled_inplace(w.blocks, basis[-2].blocks, betas[-1])
            else:
                w = ba_sub_scaled(w, basis[-2], betas[-1])

        # Full reorthogonalization (fused sub+scale)
        for q in basis:
            coeff = ba_inner(q, w)
            if _USE_CYTHON_SUB:
                _cython_ba_sub_scaled_inplace(w.blocks, q.blocks, coeff)
            else:
                w = ba_sub_scaled(w, q, coeff)

        beta_val = ba_norm(w)
        betas.append(beta_val)

        if beta_val < tol:
            break

        basis.append(ba_scale(w, 1.0 / beta_val))

    n = len(alphas)
    if n == 0:
        return 0.0, v
    if n == 1:
        return alphas[0], basis[0]

    # Tridiagonal eigendecomposition — pure numpy
    T = np.diag(alphas) + np.diag(betas[1:n], k=1) + np.diag(betas[1:n], k=-1)
    eigvals, eigvecs = np.linalg.eigh(T)
    idx = int(np.argmin(eigvals))
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Reconstruct eigenvector
    eigenvector = ba_scale(basis[0], float(krylov_coefs[0]))
    for k in range(1, n):
        eigenvector = ba_add(eigenvector, ba_scale(basis[k], float(krylov_coefs[k])))

    ev_norm = ba_norm(eigenvector)
    eigenvector = ba_scale(eigenvector, 1.0 / (ev_norm + 1e-15))

    return eigenvalue, eigenvector


def _precompute_block_plan(
    tensors: list[SymmetricTensor],
    subscripts: str,
) -> list[tuple[list[tuple[int, ...]], str]]:
    """Precompute valid block combinations for a contraction.

    Returns a list of (block_keys, output_key) tuples representing
    all charge-compatible block combinations. This can be computed
    once and reused across Lanczos iterations.
    """
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")

    plan: list[tuple[list[tuple[int, ...]], str]] = []

    # Backtracking to enumerate all charge-compatible block combos
    combo_keys: list[tuple[int, ...]] = []
    char_charges: dict[str, int] = {}

    def _recurse(tensor_idx: int) -> None:
        if tensor_idx == len(tensors):
            output_key = tuple(char_charges.get(c, 0) for c in output_part)
            plan.append((list(combo_keys), output_key))
            return

        subs = input_subs[tensor_idx]
        for key in tensors[tensor_idx].blocks:
            added_chars: list[str] = []
            compatible = True
            for char, q in zip(subs, key):
                qi = int(q)
                if char in char_charges:
                    if char_charges[char] != qi:
                        compatible = False
                        break
                else:
                    char_charges[char] = qi
                    added_chars.append(char)

            if compatible:
                combo_keys.append(key)
                _recurse(tensor_idx + 1)
                combo_keys.pop()

            for char in added_chars:
                del char_charges[char]

    _recurse(0)
    return plan


def _to_np_blocks(t: SymmetricTensor) -> dict:
    """Convert a SymmetricTensor's blocks to NumPy arrays (cached-friendly)."""
    import numpy as np

    return {k: np.asarray(v) for k, v in t.blocks.items()}


def _blockwise_contract(
    tensors: list[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    expr_cache: dict[tuple[tuple[int, ...], ...], Any] | None = None,
    block_plan: list[tuple[list[tuple[int, ...]], str]] | None = None,
    np_blocks_cache: list[dict | None] | None = None,
    return_ba: bool = False,
) -> SymmetricTensor:
    """Contract multiple SymmetricTensors using block-level charge matching.

    Unlike ``_contract_symmetric`` in contractor.py, this handles multi-tensor
    contractions correctly by iterating over all compatible block combinations
    (with early pruning via charge matching) and does NOT filter output blocks
    by a conservation law — it trusts the contraction result.

    This is necessary for DMRG environment updates and matvec where contracted
    indices may have same-direction flows (ket-ket or bra-bra physical indices),
    which violates the opposite-flow assumption in ``_contract_symmetric``.

    Args:
        tensors:        List of SymmetricTensor inputs.
        subscripts:     Einsum subscript string (e.g., "abc,apd,bpxe,cxf->def").
        output_indices: TensorIndex metadata for the output legs.
        expr_cache:     Optional shared cache for opt_einsum contraction
                        expressions. Pass the same dict across calls (e.g.,
                        Lanczos iterations) to avoid recomputing paths.
        block_plan:     Optional precomputed plan from ``_precompute_block_plan``.
                        When provided, skips charge-matching backtracking and
                        directly iterates over the valid block combinations.
        np_blocks_cache: Optional pre-converted NumPy blocks for each tensor.
                        List of dicts (one per tensor), or None entries for
                        tensors that should be converted fresh. Avoids repeated
                        JAX→NumPy conversion for fixed environment tensors.

    Returns:
        SymmetricTensor with contracted result (bypasses conservation validation).

    Note:
        Callers must validate inputs via ``_assert_symmetric`` before calling.
    """
    # NOTE: cuTensorNet integration for _blockwise_contract is not beneficial.
    # The todense→contract→extract-blocks approach (370s) is slower than the
    # NumPy per-block path (34s) due to block extraction overhead.
    # cuTensorNet is useful for dense contractions (see cutensornet_backend.py)
    # and for the dense iDMRG/DMRG paths. See issue #200 for future
    # block-sparse GPU acceleration via cuTENSOR or padded vmap.

    # --- CPU/fallback path ---
    # Cache for opt_einsum contraction expressions
    if expr_cache is None:
        expr_cache = {}

    # Accumulate contributions per output key, then sum once at the end
    # to avoid repeated intermediate JAX array allocations.
    output_accum: dict[tuple[int, ...], list[jax.Array]] = {}

    if block_plan is not None:
        # Build NumPy blocks — use cache for env tensors, convert fresh otherwise
        np_blocks_list = []
        for i, t in enumerate(tensors):
            if np_blocks_cache is not None and np_blocks_cache[i] is not None:
                np_blocks_list.append(np_blocks_cache[i])
            else:
                np_blocks_list.append(_to_np_blocks(t))

        # Use BLAS plan instead of opt_einsum (2x less Python dispatch overhead)
        from tenax.contraction._blas_plan import get_cached_blas_plan

        # Cache for pre-extracted Cython step params (keyed by shape group)
        _step_params_cache: dict = {}

        for combo_keys, output_key in block_plan:
            combo_arrays = [
                np_blocks_list[i][combo_keys[i]] for i in range(len(np_blocks_list))
            ]
            block_shapes = tuple(a.shape for a in combo_arrays)
            if block_shapes not in expr_cache:
                expr_cache[block_shapes] = get_cached_blas_plan(
                    subscripts, block_shapes
                )
            plan = expr_cache[block_shapes]

            if _USE_CYTHON_PLAN:
                if block_shapes not in _step_params_cache:
                    _step_params_cache[block_shapes] = [
                        (
                            s.left_idx,
                            s.right_idx,
                            s.out_idx,
                            s.m,
                            s.n,
                            s.k,
                            s.left_perm,
                            s.right_perm,
                            s.out_shape,
                        )
                        for s in plan.steps
                    ]
                result_array = _cython_execute_plan(
                    _step_params_cache[block_shapes],
                    plan.n_inputs,
                    plan.n_buffers,
                    plan.output_perm,
                    combo_arrays,
                )
            else:
                result_array = plan.execute_numpy(combo_arrays)

            output_accum.setdefault(output_key, []).append(result_array)
    else:
        # Original backtracking approach
        input_part, output_part = subscripts.split("->")
        input_subs = input_part.split(",")

        combo_arrays: list[jax.Array] = []
        char_charges: dict[str, int] = {}

        def _recurse(tensor_idx: int) -> None:
            if tensor_idx == len(tensors):
                output_key = tuple(char_charges.get(c, 0) for c in output_part)

                block_shapes = tuple(a.shape for a in combo_arrays)
                if block_shapes in expr_cache:
                    expr = expr_cache[block_shapes]
                else:
                    expr = opt_einsum.contract_expression(
                        subscripts, *block_shapes, optimize="auto"
                    )
                    expr_cache[block_shapes] = expr
                result_array = expr(*combo_arrays, backend="jax")

                output_accum.setdefault(output_key, []).append(result_array)
                return

            subs = input_subs[tensor_idx]
            for key, arr in tensors[tensor_idx].blocks.items():
                added_chars: list[str] = []
                compatible = True
                for char, q in zip(subs, key):
                    qi = int(q)
                    if char in char_charges:
                        if char_charges[char] != qi:
                            compatible = False
                            break
                    else:
                        char_charges[char] = qi
                        added_chars.append(char)

                if compatible:
                    combo_arrays.append(arr)
                    _recurse(tensor_idx + 1)
                    combo_arrays.pop()

                for char in added_chars:
                    del char_charges[char]

        _recurse(0)

    # Sum accumulated contributions per output key
    output_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, arrays in output_accum.items():
        total = arrays[0]
        for a in arrays[1:]:
            total = total + a
        output_blocks[key] = total

    if return_ba:
        from tenax.algorithms._block_array import BlockArray

        return BlockArray(blocks=output_blocks, indices=output_indices)

    # Build result bypassing SymmetricTensor validation (flows may not
    # satisfy the standard conservation law for environment tensors).
    obj = object.__new__(SymmetricTensor)
    obj._indices = output_indices
    obj._init_flat_buffer(output_blocks)
    return obj


def _assert_symmetric(*tensors: Tensor, context: str) -> None:
    """Assert all tensors are SymmetricTensor or BlockArray; raise TypeError otherwise."""
    from tenax.algorithms._block_array import BlockArray

    for i, t in enumerate(tensors):
        if not isinstance(t, (SymmetricTensor, BlockArray)):
            raise TypeError(
                f"{context}: expected SymmetricTensor for input {i}, "
                f"got {type(t).__name__}. "
                f"The symmetric DMRG path must never fall back to dense."
            )


def _update_left_env_symmetric(
    left_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update left environment using block-sparse contraction.

    Contracts: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * A*[c,x,f]

    All inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        left_env, mps_site, mpo_site, context="_update_left_env_symmetric"
    )
    A = mps_site  # always 3D: (chi_l, d, chi_r)
    A_bra = A.bar()

    # Build output indices from the free legs of the contraction:
    # d = A's right virtual, e = W's right bond, f = A_bra's right virtual
    # bar() flips flows so bra virtual legs have opposite flow to ket legs,
    # which is the physically correct convention for environment tensors.
    out_indices = (A.indices[2], mpo_site.indices[3], A_bra.indices[2])
    tensors = [left_env, A, mpo_site, A_bra]
    subs = "abc,apd,bpxe,cxf->def"
    plan = _precompute_block_plan(tensors, subs)
    np_blocks = [_to_np_blocks(t) for t in tensors]
    result = _blockwise_contract(
        tensors,
        subs,
        output_indices=out_indices,
        block_plan=plan,
        np_blocks_cache=np_blocks,
    )
    return result


def _update_right_env_symmetric(
    right_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update right environment using block-sparse contraction.

    Contracts: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * B*[f,x,c]

    All inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        right_env, mps_site, mpo_site, context="_update_right_env_symmetric"
    )
    B = mps_site  # always 3D: (chi_l, d, chi_r)
    B_bra = B.bar()

    # Output: d = B's left virtual, e = W's left bond, f = B_bra's left virtual
    # bar() flips flows for physically correct bra convention.
    out_indices = (B.indices[0], mpo_site.indices[0], B_bra.indices[0])
    tensors = [right_env, B, mpo_site, B_bra]
    subs = "abc,dpa,epxb,fxc->def"
    plan = _precompute_block_plan(tensors, subs)
    np_blocks = [_to_np_blocks(t) for t in tensors]
    result = _blockwise_contract(
        tensors,
        subs,
        output_indices=out_indices,
        block_plan=plan,
        np_blocks_cache=np_blocks,
    )
    return result


def _two_site_update_symmetric(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 2-site DMRG update using block-sparse tensors.

    All tensor inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        site_l,
        site_r,
        left_env,
        mpo_l,
        mpo_r,
        right_env,
        context="_two_site_update_symmetric",
    )
    # Contract theta = A[i] * A[i+1] — always 4D: (chi_l, d_l, d_r, chi_r)
    shared = set(site_l.labels()) & set(site_r.labels())
    if shared:
        theta = contract(site_l, site_r)
    else:
        theta = site_l

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _matvec_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    # Precompute block plan once — reused across all Lanczos iterations
    _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
    _plan = _precompute_block_plan([left_env, theta, mpo_l, mpo_r, right_env], _subs)

    # Pre-convert env blocks to NumPy once — only v changes per Lanczos iteration
    # Tensor order: [left_env, v, mpo_l, mpo_r, right_env]
    _env_np = [
        _to_np_blocks(left_env),
        None,  # v — converted fresh each call
        _to_np_blocks(mpo_l),
        _to_np_blocks(mpo_r),
        _to_np_blocks(right_env),
    ]

    def matvec(v: Tensor) -> Tensor:
        result = _blockwise_contract(
            [left_env, v, mpo_l, mpo_r, right_env],
            _subs,
            output_indices=v.indices,
            expr_cache=_matvec_cache,
            block_plan=_plan,
            np_blocks_cache=_env_np,
        )
        return result

    energy, theta_opt = _lanczos_solve_tensor(
        matvec, theta, config.lanczos_max_iter, config.lanczos_tol
    )

    return theta_opt, energy


def _one_site_update_symmetric(
    site: Tensor,
    left_env: Tensor,
    mpo_site: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 1-site DMRG update using block-sparse tensors.

    All tensor inputs must be SymmetricTensor. The symmetric path must never
    fall back to dense operations.
    """
    _assert_symmetric(
        site, left_env, mpo_site, right_env, context="_one_site_update_symmetric"
    )

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _matvec_cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    # Precompute block plan once — reused across all Lanczos iterations
    _subs = "abc,apd,bpxe,def->cxf"
    _plan = _precompute_block_plan([left_env, site, mpo_site, right_env], _subs)

    # Pre-convert env blocks to NumPy once — only v changes per Lanczos iteration
    # Tensor order: [left_env, v, mpo_site, right_env]
    _env_np = [
        _to_np_blocks(left_env),
        None,  # v — converted fresh each call
        _to_np_blocks(mpo_site),
        _to_np_blocks(right_env),
    ]

    def matvec(v: Tensor) -> Tensor:
        result = _blockwise_contract(
            [left_env, v, mpo_site, right_env],
            _subs,
            output_indices=v.indices,
            expr_cache=_matvec_cache,
            block_plan=_plan,
            np_blocks_cache=_env_np,
        )
        return result

    energy, site_opt = _lanczos_solve_tensor(
        matvec, site, config.lanczos_max_iter, config.lanczos_tol
    )

    return site_opt, energy


# ------------------------------------------------------------------ #
# NumPy-only symmetric DMRG updates (no JAX in inner loop)            #
# ------------------------------------------------------------------ #


def _scale_bond_axis_ba(ba: BlockArray, bond_label: str, s: np.ndarray) -> BlockArray:
    """Scale BlockArray along its bond axis by singular values.

    Mirrors ``_scale_bond_axis_symmetric`` but operates on BlockArray
    with numpy singular values. Used after SVD to absorb singular values
    into A or B for canonical form.
    """
    from tenax.algorithms._block_array import BlockArray as _BA

    bond_axis = None
    for i, idx in enumerate(ba.indices):
        if idx.label == bond_label:
            bond_axis = i
            break
    if bond_axis is None:
        return ba

    bond_idx = ba.indices[bond_axis]
    new_blocks = {}
    for key, block in ba.blocks.items():
        charge_val = key[bond_axis]
        positions = np.where(bond_idx.charges == charge_val)[0]
        block_size = block.shape[bond_axis]
        scale_slice = s[positions[:block_size]]
        shape = [1] * block.ndim
        shape[bond_axis] = block_size
        new_blocks[key] = block * scale_slice.reshape(shape)
    return _BA(blocks=new_blocks, indices=ba.indices)


def _svd_and_truncate_site_np(
    theta_ba: BlockArray,
    site: int,
    config: DMRGConfig,
    sweep_right: bool = True,
) -> tuple[BlockArray, np.ndarray, BlockArray, float]:
    """SVD of 2-site tensor and truncation -- pure NumPy path.

    Same logic as ``_svd_and_truncate_site`` but operates on BlockArray
    and uses ``_truncated_svd_symmetric_np`` for the decomposition.

    Args:
        theta_ba:    2-site wavefunction as BlockArray.
        site:        Left site index.
        config:      DMRGConfig.
        sweep_right: If True, left site gets orthogonality center (A-form);
                     if False, right site gets it (B-form).

    Returns:
        (A_ba, singular_values, B_ba, truncation_error) -- all numpy.
    """
    from tenax.algorithms._block_array import ba_to_symmetric
    from tenax.linalg import _truncated_svd_symmetric_np

    labels = [idx.label for idx in theta_ba.indices]

    # Find physical and virtual labels
    if site > 0:
        left_virt = f"v{site - 1}_{site}"
    else:
        left_virt = "v_-1_0"
    right_virt = f"v{site + 1}_{site + 2}"
    left_phys = f"p{site}"
    right_phys = f"p{site + 1}"

    # Build actual left/right label splits based on what's available
    left_candidates = {left_virt, left_phys}
    right_candidates = {right_virt, right_phys}
    left_labels = [lbl for lbl in labels if lbl in left_candidates]
    right_labels = [lbl for lbl in labels if lbl in right_candidates]

    if not left_labels or not right_labels:
        n = len(labels)
        left_labels = list(labels[: n // 2])
        right_labels = list(labels[n // 2 :])

    bond_label = f"v{site}_{site + 1}"

    # Convert to SymmetricTensor for SVD (needs full index metadata)
    theta_sym = ba_to_symmetric(theta_ba)

    # SVD via numpy path -- returns (U_ba, s_final, Vh_ba, s_full)
    A_ba, s, B_ba, s_full = _truncated_svd_symmetric_np(
        theta_sym,
        left_labels=left_labels,
        right_labels=right_labels,
        max_singular_values=config.max_bond_dim,
        max_truncation_err=config.svd_trunc_err,
        new_bond_label=bond_label,
        normalize=False,
    )

    # Compute truncation error from the full singular-value spectrum
    n_keep = len(s)
    if len(s_full) > n_keep:
        total_sq = np.sum(s_full**2)
        trunc_sq = np.sum(s_full[n_keep:] ** 2)
        trunc_err = float(np.sqrt(trunc_sq / (total_sq + 1e-15)))
    else:
        trunc_err = 0.0

    # Absorb singular values into the tensor moving away from the
    # orthogonality center so the MPS stays in canonical form.
    if sweep_right:
        B_ba = _scale_bond_axis_ba(B_ba, bond_label, s)
    else:
        A_ba = _scale_bond_axis_ba(A_ba, bond_label, s)

    return A_ba, s, B_ba, trunc_err


def _two_site_update_symmetric_np(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 2-site DMRG update using numpy-only inner loop.

    Accepts SymmetricTensor inputs (matching the SweepOps callback
    signature) and returns SymmetricTensor + float. BlockArray is used
    internally to avoid JAX overhead in the Lanczos iterations.
    """
    from tenax.algorithms._block_array import (
        BlockArray,
        symmetric_to_ba,
    )

    _assert_symmetric(
        site_l,
        site_r,
        left_env,
        mpo_l,
        mpo_r,
        right_env,
        context="_two_site_update_symmetric_np",
    )

    # Convert inputs to BlockArray -- stays in numpy throughout
    site_l_ba = (
        symmetric_to_ba(site_l) if not isinstance(site_l, BlockArray) else site_l
    )
    site_r_ba = (
        symmetric_to_ba(site_r) if not isinstance(site_r, BlockArray) else site_r
    )

    labels_l = site_l_ba.labels()
    labels_r = site_r_ba.labels()
    shared = set(labels_l) & set(labels_r)

    if shared:
        # Build einsum subscripts from labels
        all_unique_labels = list(dict.fromkeys(list(labels_l) + list(labels_r)))
        label_to_char = {lb: chr(97 + i) for i, lb in enumerate(all_unique_labels)}
        input_l = "".join(label_to_char[lb] for lb in labels_l)
        input_r = "".join(label_to_char[lb] for lb in labels_r)
        output_chars = "".join(label_to_char[lb] for lb in labels_l if lb not in shared)
        output_chars += "".join(
            label_to_char[lb] for lb in labels_r if lb not in shared
        )
        theta_subs = f"{input_l},{input_r}->{output_chars}"

        # Build output indices from free legs
        out_indices = tuple(
            idx for idx in site_l_ba.indices if idx.label not in shared
        ) + tuple(idx for idx in site_r_ba.indices if idx.label not in shared)

        theta_plan = _precompute_block_plan([site_l_ba, site_r_ba], theta_subs)
        np_blocks = [site_l_ba.blocks, site_r_ba.blocks]
        theta_ba = _blockwise_contract(
            [site_l_ba, site_r_ba],
            theta_subs,
            output_indices=out_indices,
            block_plan=theta_plan,
            np_blocks_cache=np_blocks,
            return_ba=True,
        )
    else:
        theta_ba = site_l_ba

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    # Precompute block plan once
    _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
    _plan = _precompute_block_plan([left_env, theta_ba, mpo_l, mpo_r, right_env], _subs)

    # Pre-convert env blocks to NumPy once
    _env_np = [
        _to_np_blocks(left_env)
        if not isinstance(left_env, BlockArray)
        else left_env.blocks,
        None,  # v -- converted fresh each call
        _to_np_blocks(mpo_l) if not isinstance(mpo_l, BlockArray) else mpo_l.blocks,
        _to_np_blocks(mpo_r) if not isinstance(mpo_r, BlockArray) else mpo_r.blocks,
        _to_np_blocks(right_env)
        if not isinstance(right_env, BlockArray)
        else right_env.blocks,
    ]

    _out_indices = theta_ba.indices  # fixed across Lanczos iterations

    def matvec(v_ba: BlockArray) -> BlockArray:
        # Pass v blocks directly as np_blocks_cache — avoids numpy→JAX→numpy roundtrip
        _env_np[1] = v_ba.blocks
        return _blockwise_contract(
            [left_env, theta_ba, mpo_l, mpo_r, right_env],
            _subs,
            output_indices=_out_indices,
            expr_cache=_cache,
            block_plan=_plan,
            np_blocks_cache=_env_np,
            return_ba=True,
        )

    energy, theta_opt_ba = _lanczos_solve_np(
        matvec, theta_ba, config.lanczos_max_iter, config.lanczos_tol
    )

    # Return BlockArray directly — stays as numpy through sweep loop.
    # Converted to SymmetricTensor only at the end of dmrg().
    return theta_opt_ba, energy


def _one_site_update_symmetric_np(
    site: Tensor,
    left_env: Tensor,
    mpo_site: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """Perform 1-site DMRG update using numpy-only inner loop.

    Accepts SymmetricTensor inputs (matching the SweepOps callback
    signature) and returns SymmetricTensor + float. BlockArray is used
    internally to avoid JAX overhead in the Lanczos iterations.
    """
    from tenax.algorithms._block_array import (
        BlockArray,
        ba_to_symmetric,
        symmetric_to_ba,
    )

    _assert_symmetric(
        site,
        left_env,
        mpo_site,
        right_env,
        context="_one_site_update_symmetric_np",
    )

    # Convert site to BlockArray -- stays in numpy throughout
    site_ba = symmetric_to_ba(site) if not isinstance(site, BlockArray) else site

    # Shared cache for opt_einsum expressions across Lanczos iterations
    _cache: dict[tuple[tuple[int, ...], ...], Any] = {}

    # Precompute block plan once
    _subs = "abc,apd,bpxe,def->cxf"
    _plan = _precompute_block_plan([left_env, site_ba, mpo_site, right_env], _subs)

    # Pre-convert env blocks to NumPy once
    _env_np = [
        _to_np_blocks(left_env)
        if not isinstance(left_env, BlockArray)
        else left_env.blocks,
        None,  # v -- converted fresh each call
        _to_np_blocks(mpo_site)
        if not isinstance(mpo_site, BlockArray)
        else mpo_site.blocks,
        _to_np_blocks(right_env)
        if not isinstance(right_env, BlockArray)
        else right_env.blocks,
    ]

    _out_indices = site_ba.indices

    def matvec(v_ba: BlockArray) -> BlockArray:
        _env_np[1] = v_ba.blocks
        return _blockwise_contract(
            [left_env, site_ba, mpo_site, right_env],
            _subs,
            output_indices=_out_indices,
            expr_cache=_cache,
            block_plan=_plan,
            np_blocks_cache=_env_np,
            return_ba=True,
        )

    energy, site_opt_ba = _lanczos_solve_np(
        matvec, site_ba, config.lanczos_max_iter, config.lanczos_tol
    )

    # 1-site mode needs SymmetricTensor for QR/relabel in sweep loop
    return ba_to_symmetric(site_opt_ba), energy


def _update_left_env_np(
    left_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update left environment, accepting BlockArray or SymmetricTensor for MPS site."""
    from tenax.algorithms._block_array import BlockArray, ba_bar

    if isinstance(mps_site, BlockArray):
        A = mps_site
        A_bra = ba_bar(A)
    else:
        A = mps_site
        A_bra = A.bar()

    out_indices = (A.indices[2], mpo_site.indices[3], A_bra.indices[2])
    tensors = [left_env, A, mpo_site, A_bra]
    subs = "abc,apd,bpxe,cxf->def"
    plan = _precompute_block_plan(tensors, subs)
    np_blocks = [
        _to_np_blocks(t) if not isinstance(t, BlockArray) else t.blocks for t in tensors
    ]
    return _blockwise_contract(
        tensors,
        subs,
        output_indices=out_indices,
        block_plan=plan,
        np_blocks_cache=np_blocks,
        return_ba=True,
    )


def _update_right_env_np(
    right_env: Tensor,
    mps_site: Tensor,
    mpo_site: Tensor,
) -> SymmetricTensor:
    """Update right environment, accepting BlockArray or SymmetricTensor for MPS site."""
    from tenax.algorithms._block_array import BlockArray, ba_bar

    if isinstance(mps_site, BlockArray):
        B = mps_site
        B_bra = ba_bar(B)
    else:
        B = mps_site
        B_bra = B.bar()

    out_indices = (B.indices[0], mpo_site.indices[0], B_bra.indices[0])
    tensors = [right_env, B, mpo_site, B_bra]
    subs = "abc,dpa,epxb,fxc->def"
    plan = _precompute_block_plan(tensors, subs)
    np_blocks = [
        _to_np_blocks(t) if not isinstance(t, BlockArray) else t.blocks for t in tensors
    ]
    return _blockwise_contract(
        tensors,
        subs,
        output_indices=out_indices,
        block_plan=plan,
        np_blocks_cache=np_blocks,
        return_ba=True,
    )


def _symmetric_ops(config: DMRGConfig) -> SweepOps:
    """Return the block-sparse symmetric backend callbacks."""
    if config.numpy_blockwise:
        return SweepOps(
            build_trivial_left_env=_build_trivial_left_env_symmetric,
            build_trivial_right_env=_build_trivial_right_env_symmetric,
            update_left_env=_update_left_env_np,
            update_right_env=_update_right_env_np,
            two_site_update=_two_site_update_symmetric_np,
            one_site_update=_one_site_update_symmetric_np,
        )
    return SweepOps(
        build_trivial_left_env=_build_trivial_left_env_symmetric,
        build_trivial_right_env=_build_trivial_right_env_symmetric,
        update_left_env=_update_left_env_symmetric,
        update_right_env=_update_right_env_symmetric,
        two_site_update=_two_site_update_symmetric,
        one_site_update=_one_site_update_symmetric,
    )


# ------------------------------------------------------------------ #
# MPO builders                                                        #
# ------------------------------------------------------------------ #


def build_mpo_heisenberg(
    L: int,
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    dtype: Any = jnp.float64,
) -> TensorNetwork:
    """Build the MPO for the spin-1/2 XXZ Heisenberg chain.

    H = Jz * sum_i Sz_i Sz_{i+1} + Jxy/2 * sum_i (S+_i S-_{i+1} + S-_i S+_{i+1})
        + hz * sum_i Sz_i

    Returns a block-sparse (SymmetricTensor) MPO with U(1) charge conservation.
    Paired with a symmetric MPS (e.g. from ``build_random_symmetric_mps``),
    DMRG will use the fully block-sparse backend automatically.

    Args:
        L:      Chain length (number of sites).
        Jz:     Ising coupling strength.
        Jxy:    XY coupling strength.
        hz:     Longitudinal magnetic field.
        dtype:  JAX dtype for MPO tensors.

    Returns:
        TensorNetwork representing the MPO with L site tensors connected
        by virtual bonds. Each site tensor has legs:
        ("w{i-1}_{i}", "mpo_top_{i}", "mpo_bot_{i}", "w{i}_{i+1}")
    """
    terms: list[tuple[float, ...]] = []
    for i in range(L - 1):
        terms.append((Jz, "Sz", i, "Sz", i + 1))
        terms.append((Jxy / 2, "Sp", i, "Sm", i + 1))
        terms.append((Jxy / 2, "Sm", i, "Sp", i + 1))
    if hz != 0.0:
        for i in range(L):
            terms.append((hz, "Sz", i))
    if not terms:
        # L=1 with hz=0: add a zero on-site term so AutoMPO has at least one term
        terms.append((0.0, "Sz", 0))
    return build_auto_mpo(terms, L=L, symmetric=True)


def build_random_symmetric_mps(
    L: int,
    bond_dim: int = 4,
    dtype: Any = jnp.float64,
    seed: int = 42,
    target_charge: int = 0,
) -> TensorNetwork:
    """Build a random block-sparse MPS with U(1) charge conservation.

    .. deprecated::
        Use :meth:`FiniteMPS.random` instead::

            mps = FiniteMPS.random(L=L, d=2, chi=bond_dim, key=key,
                                   symmetric=True, symmetry=U1Symmetry(),
                                   target_charge=target_charge)

    Physical dimension is 2 (spin-1/2). Charges represent accumulated Sz:
    spin up = +1, spin down = -1. Virtual bonds carry sectors that allow
    the specified total-Sz subspace.

    Args:
        L:              Chain length.
        bond_dim:       Virtual bond dimension (must be >= 2; blocks distributed
                        across charge sectors).
        dtype:          JAX dtype.
        seed:           Random seed for block initialisation.
        target_charge:  Target total charge (2*Sz). Default 0 (Sz=0 sector).
                        Must satisfy parity: target_charge % 2 == L % 2
                        (each site contributes ±1).

    Returns:
        TensorNetwork representing the symmetric random MPS.

    Raises:
        ValueError: If target_charge has incompatible parity with L.
    """
    if target_charge % 2 != L % 2:
        raise ValueError(
            f"target_charge={target_charge} has parity {target_charge % 2} but "
            f"L={L} has parity {L % 2}. Each site contributes ±1, so total "
            f"charge must have the same parity as L."
        )

    import warnings

    warnings.warn(
        "build_random_symmetric_mps is deprecated. Use FiniteMPS.random() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    sym = U1Symmetry()

    # Physical: spin up = +1, spin down = −1
    phys_charges = np.array([1, -1], dtype=np.int32)
    trivial_zero = np.array([0], dtype=np.int32)

    # Virtual bond: include charge sectors compatible with target propagation.
    # For total charge Q, the right boundary needs virt charges in {Q-1, Q+1}
    # (since phys charges are ±1 and we need virt + phys = Q).
    # Interior bonds need a range that connects left boundary (near 0) to
    # right boundary (near Q). We include charges from -1 to max(1, Q+1)
    # (or min(-1, Q-1) to 1 for negative Q).
    if target_charge == 0:
        required_charges = [-1, 0, 1]
    else:
        # Include range from 0 to target, plus margins for both boundaries
        lo = min(-1, target_charge - 1)
        hi = max(1, target_charge + 1)
        required_charges = list(range(lo, hi + 1))

    # Distribute bond_dim states across the required charge sectors
    n_sectors = len(required_charges)
    per_sector = max(1, bond_dim // n_sectors)
    arrays = [np.full(per_sector, q, dtype=np.int32) for q in required_charges]
    virt_charges = np.concatenate(arrays)[:bond_dim]
    # If bond_dim is larger, pad with the middle charge
    if len(virt_charges) < bond_dim:
        mid_q = required_charges[n_sectors // 2]
        pad = np.full(bond_dim - len(virt_charges), mid_q, dtype=np.int32)
        virt_charges = np.concatenate([virt_charges, pad])

    mps = TensorNetwork(name=f"symmetric_MPS_L{L}")

    for i in range(L):
        key = jax.random.PRNGKey(seed + i)

        # Right boundary tensor uses target_charge; all others use identity (0)
        site_target = target_charge if i == L - 1 else None

        if L == 1:
            # Single-site: (trivial_IN, phys_IN, trivial_OUT)
            # target=target_charge enforces the sector; right bond carries charge 0.
            indices: tuple[TensorIndex, ...] = (
                TensorIndex(sym, trivial_zero, FlowDirection.IN, label="v_-1_0"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, trivial_zero, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == 0:
            # Left boundary: (trivial_IN, phys_IN, virt_right_OUT)
            indices = (
                TensorIndex(sym, trivial_zero, FlowDirection.IN, label="v_-1_0"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            # Right boundary: (virt_left_IN, phys_IN, trivial_OUT)
            # target=target_charge enforces the sector; right bond carries charge 0.
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, trivial_zero, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        else:
            # Middle: (virt_left_IN, phys_IN, virt_right_OUT)
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )

        tensor = SymmetricTensor.random_normal(
            indices, key=key, dtype=dtype, target=site_target
        )
        mps.add_node(i, tensor)

    # Connect virtual bonds
    for i in range(L - 1):
        bond_label = f"v{i}_{i + 1}"
        mps.connect(i, bond_label, i + 1, bond_label)

    return mps


def compute_mps_sector(mps_tensors: list[Tensor]) -> int | None:
    """Infer total charge sector of an MPS from its tensor block structure.

    With the 3-leg boundary convention every MPS tensor satisfies
    ``sum(flow_i * charge_i) = 0`` per block, and both boundary bonds
    carry charge 0.  The target charge Q is encoded through block
    selection: exactly one tensor (typically the orthogonality center)
    has all blocks satisfying ``sum(flow_i * charge_i) = Q`` instead
    of 0.  This function scans all tensors and returns Q.

    If all tensors satisfy standard conservation (``sum = 0``), the
    function returns 0.

    Args:
        mps_tensors: List of SymmetricTensor MPS site tensors.

    Returns:
        The total charge if consistently detectable, or None if the
        MPS is in a mixed sector (or contains no SymmetricTensor).
    """
    for site in mps_tensors:
        if not isinstance(site, SymmetricTensor):
            continue
        if not site.blocks:
            continue

        sectors: set[int] = set()
        for key in site.blocks:
            total = 0
            for idx, q in zip(site.indices, key):
                total += int(idx.flow) * q
            sectors.add(total)

        if len(sectors) != 1:
            return None
        charge = sectors.pop()
        if charge != 0:
            return charge

    # All tensors have standard conservation (sum = 0)
    return 0


def validate_mps_sector(mps_tensors: list[Tensor], target_charge: int) -> None:
    """Assert that an MPS is in the specified charge sector.

    Args:
        mps_tensors:   List of MPS site tensors (SymmetricTensor).
        target_charge: Expected total charge (e.g. 2*Sz for spin-1/2 U(1)).

    Raises:
        ValueError: If the MPS is not in the target sector.
    """
    sector = compute_mps_sector(mps_tensors)
    if sector is None:
        raise ValueError(
            f"Cannot determine MPS sector (mixed or no SymmetricTensor blocks). "
            f"Expected target_charge={target_charge}."
        )
    if sector != target_charge:
        raise ValueError(
            f"MPS sector {sector} does not match target_charge={target_charge}."
        )


def build_random_mps(
    L: int,
    physical_dim: int = 2,
    bond_dim: int = 4,
    dtype: Any = jnp.float64,
    seed: int = 0,
) -> TensorNetwork:
    """Build a random MPS for use as initial state in DMRG.

    .. deprecated::
        Use :meth:`FiniteMPS.random` instead::

            mps = FiniteMPS.random(L=L, d=physical_dim, chi=bond_dim, key=key)

    Args:
        L:            Chain length.
        physical_dim: Physical dimension per site.
        bond_dim:     Virtual bond dimension.
        dtype:        Data type.
        seed:         Random seed.

    Returns:
        TensorNetwork representing the random MPS.
    """
    import warnings

    warnings.warn(
        "build_random_mps is deprecated. Use FiniteMPS.random() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    sym = U1Symmetry()
    bond_d = np.zeros(physical_dim, dtype=np.int32)
    bond_chi = np.zeros(bond_dim, dtype=np.int32)
    bond_trivial = np.zeros(1, dtype=np.int32)

    mps = TensorNetwork(name=f"random_MPS_L{L}")

    shape: tuple[int, ...]
    indices: tuple[TensorIndex, ...]
    for i in range(L):
        key = jax.random.PRNGKey(seed + i)

        if L == 1:
            # Single-site MPS: (1, d, 1) with trivial bonds on both sides.
            shape = (1, physical_dim, 1)
            indices = (
                TensorIndex(sym, bond_trivial, FlowDirection.IN, label="v_-1_0"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, bond_trivial, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == 0:
            shape = (1, physical_dim, bond_dim)
            indices = (
                TensorIndex(sym, bond_trivial, FlowDirection.IN, label="v_-1_0"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(sym, bond_chi, FlowDirection.OUT, label=f"v{i}_{i + 1}"),
            )
        elif i == L - 1:
            shape = (bond_dim, physical_dim, 1)
            indices = (
                TensorIndex(sym, bond_chi, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, bond_trivial, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        else:
            shape = (bond_dim, physical_dim, bond_dim)
            indices = (
                TensorIndex(sym, bond_chi, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, bond_d, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(sym, bond_chi, FlowDirection.OUT, label=f"v{i}_{i + 1}"),
            )

        data = jax.random.normal(key, shape, dtype=dtype)
        # Normalize
        data = data / jnp.linalg.norm(data)
        mps.add_node(i, DenseTensor(data, indices))

    # Connect virtual bonds
    for i in range(L - 1):
        bond_label = f"v{i}_{i + 1}"
        mps.connect(i, bond_label, i + 1, bond_label)

    return mps
