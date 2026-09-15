"""AD-based iPEPS quasiparticle excitations.

Implements the method from Ponsioen, Assaad & Corboz, SciPost Phys. 12, 006
(2022): construct the effective Hamiltonian and norm matrices for iPEPS
excitations using JAX automatic differentiation, then solve the generalized
eigenvalue problem for the excitation spectrum.

Key components:
1. Mixed double-layer tensors (A/B substitutions in ket/bra)
2. Mixed RDM contractions for excitation energy/norm functionals
3. H_eff(k) and N(k) matrix construction via AD
4. Generalized eigenvalue solver with null-space projection
5. High-level ``compute_excitations()`` for the dispersion relation
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._einsum_compat import einsum_promoted
from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.core.tensor import SymmetricTensor, Tensor

# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------
#
# ``optimize_gs_ad`` returns Tensor-protocol objects (``DenseTensor`` site
# tensor + a ``CTMTensorEnv`` whose 8 fields are ``DenseTensor``s), while the
# excitation contractions below operate on raw ``jax.Array``s and a raw-array
# ``CTMEnvironment``.  The Tensor-based ``CTMTensorEnv`` fields share the exact
# leg/flow convention of the legacy ``CTMEnvironment`` (verified to machine
# precision against ``compute_energy_ctm``), so ``.todense()`` on each field is
# both correct and preserves the converged environment.
#
# The excitation contraction path is dense-only (raw ``einsum``).  For a
# ``DenseTensor`` ``.todense()`` just returns the already-materialized buffer,
# so it is free.  A ``SymmetricTensor``, however, would be densified here —
# violating the project rule against ``todense()`` on the symmetric path
# (CLAUDE.md / AGENTS.md): CTM edges scale as ``χ·D²·χ`` and a block-sparse
# production run could OOM or silently bypass the symmetry machinery.  Until a
# symmetric-aware excitation implementation exists we reject it explicitly.


def _as_dense_array(x: jax.Array | Tensor) -> jax.Array:
    """Return a raw ``jax.Array`` from an array or ``DenseTensor``.

    Raises ``NotImplementedError`` for ``SymmetricTensor`` — the excitation
    path is dense-only and must not silently densify a block-sparse tensor.
    """
    if isinstance(x, SymmetricTensor):
        raise NotImplementedError(
            "compute_excitations does not support SymmetricTensor inputs: the "
            "excitation contraction path is dense-only and densifying a "
            "block-sparse tensor would defeat the symmetry machinery (and can "
            "OOM at large chi/D). Convert the ground state to a dense iPEPS "
            "before computing excitations, or track symmetric-aware "
            "excitations as a follow-up."
        )
    if isinstance(x, Tensor):  # DenseTensor: todense() returns the stored buffer
        return x.todense()
    return jnp.asarray(x)


def _as_dense_env(env) -> CTMEnvironment:
    """Coerce a CTM environment to a raw-array ``CTMEnvironment``.

    Accepts the legacy raw-array ``CTMEnvironment`` (returned unchanged) or the
    Tensor-protocol ``CTMTensorEnv`` produced by ``optimize_gs_ad`` (each of the
    8 corner/edge fields is converted via ``.todense()``).
    """
    fields = tuple(env)
    if len(fields) != 8:
        raise ValueError(
            "compute_excitations expects an 8-tensor CTM environment "
            f"(4 corners + 4 edges); got {len(fields)} fields. Split-CTM "
            "environments are not supported by the excitation path."
        )
    if any(isinstance(f, Tensor) for f in fields):
        return CTMEnvironment(*(_as_dense_array(f) for f in fields))
    return env if isinstance(env, CTMEnvironment) else CTMEnvironment(*fields)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExcitationConfig:
    """Configuration for iPEPS excitation calculation.

    Attributes:
        chi:              CTM bond dimension.
        ctm_max_iter:     Maximum CTM iterations.
        ctm_conv_tol:     CTM convergence tolerance.
        num_excitations:  Number of lowest excitation energies to return.
        null_space_tol:   Threshold for filtering null-space of the norm
                          matrix (eigenvalues below this fraction of the
                          maximum are discarded).
    """

    chi: int = 20
    ctm_max_iter: int = 100
    ctm_conv_tol: float = 1e-8
    num_excitations: int = 3
    null_space_tol: float = 1e-2


@dataclass
class ExcitationResult:
    """Result of an iPEPS excitation calculation.

    Attributes:
        energies:             Excitation energies of shape
                              ``(num_k, num_excitations)``.
        momenta:              Momentum points ``(num_k, 2)``.
        ground_state_energy:  Ground state energy per site.
    """

    energies: np.ndarray
    momenta: np.ndarray
    ground_state_energy: float


# ---------------------------------------------------------------------------
# Mixed double-layer tensors
# ---------------------------------------------------------------------------


def _build_mixed_double_layer(
    A: jax.Array,
    B: jax.Array,
    position: str,
) -> jax.Array:
    """Build double-layer tensor with B substituted at *position*.

    Traces out the physical index to produce a closed double-layer tensor.

    Args:
        A: Ground state site tensor ``(D, D, D, D, d)``.
        B: Excitation perturbation tensor ``(D, D, D, D, d)``.
        position:
            ``"ket"`` — B in ket, A* in bra:
                ``a_mixed[uU,dD,lL,rR] = B[u,d,l,r,s] * conj(A[U,D,L,R,s])``
            ``"bra"`` — A in ket, B* in bra:
                ``a_mixed[uU,dD,lL,rR] = A[u,d,l,r,s] * conj(B[U,D,L,R,s])``

    Returns:
        Mixed double-layer tensor of shape ``(D^2, D^2, D^2, D^2)``.
    """
    D = A.shape[0]
    if position == "ket":
        # B in ket layer, A* in bra layer
        ao = jnp.einsum("udlrs,UDLRs->uUdDlLrR", B, jnp.conj(A))
    elif position == "bra":
        # A in ket layer, B* in bra layer
        ao = jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(B))
    else:
        raise ValueError(f"position must be 'ket' or 'bra', got {position!r}")
    return ao.reshape(D**2, D**2, D**2, D**2)


def _build_mixed_double_layer_open(
    A: jax.Array,
    B: jax.Array,
    position: str,
) -> jax.Array:
    """Build double-layer tensor with B substituted, physical indices open.

    Args:
        A: Ground state tensor ``(D, D, D, D, d)``.
        B: Excitation tensor ``(D, D, D, D, d)``.
        position: ``"ket"`` (B in ket, A* in bra) or ``"bra"`` (A in ket, B* in bra).

    Returns:
        Shape ``(D^2, D^2, D^2, D^2, d, d)`` with last two axes
        being ket and bra physical indices.
    """
    D = A.shape[0]
    d = A.shape[4]
    if position == "ket":
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", B, jnp.conj(A))
    elif position == "bra":
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", A, jnp.conj(B))
    else:
        raise ValueError(f"position must be 'ket' or 'bra', got {position!r}")
    return ao.reshape(D**2, D**2, D**2, D**2, d, d)


def _build_double_layer_BB_open(
    B: jax.Array,
) -> jax.Array:
    """Double-layer tensor with B in both ket and bra, physical indices open.

    Returns shape ``(D^2, D^2, D^2, D^2, d, d)``.
    """
    D = B.shape[0]
    d = B.shape[4]
    ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", B, jnp.conj(B))
    return ao.reshape(D**2, D**2, D**2, D**2, d, d)


# ---------------------------------------------------------------------------
# Mixed RDM contractions
# ---------------------------------------------------------------------------


def _rdm2x1_with_open_tensors(
    ao1: jax.Array,
    ao2: jax.Array,
    env: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Raw horizontal 2-site transition RDM from open double-layer tensors.

    Reuses the contraction structure from ``_rdm2x1`` but accepts
    pre-built open double-layer tensors (which may contain B substitutions).

    Args:
        ao1: Left site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        ao2: Right site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        env: CTM environment.
        d:   Physical dimension.

    Returns:
        Raw transition RDM of shape ``(d, d, d, d)`` in the grouped
        ``(ket1, ket2, bra1, bra2)`` convention of ``ipeps_rdm._rdm2x1`` —
        **neither trace-normalised nor Hermitian-symmetrised**.  A transition
        operator carries the excitation tensor's amplitude and is not
        Hermitian on its own; dividing by its own trace cancels that
        amplitude (#954), and symmetrising it mixes the ``B``-in-ket and
        ``B``-in-bra sectors.  Callers normalise against the B-independent
        pure-``AA`` contraction of the same geometry, which carries the same
        arbitrary environment phase and cancels it just as exactly (see
        ``_normalise_rdm`` for why that phase must not survive).
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env

    UL = jnp.einsum("ab,buc->auc", C1, T1)
    UR = jnp.einsum("cuf,fg->cug", T1, C2)
    LL = jnp.einsum("gi,idj->gdj", C4, T3)
    LR = jnp.einsum("jdk,mk->jdm", T3, C3)

    Lenv = einsum_promoted("auc,axg,gdj->ucxdj", UL, T4, LL)
    Renv = einsum_promoted("cuf,frm,jdm->curjd", UR, T2, LR)

    Lenv_ao1 = jnp.einsum("ucxdj,udxrst->crjst", Lenv, ao1)
    Renv_ao2 = jnp.einsum("curjd,udlrtv->cjltv", Renv, ao2)

    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao1, Renv_ao2)

    # (ket1, bra1, ket2, bra2) -> (ket1, ket2, bra1, bra2): the same transpose
    # ``ipeps_rdm._rdm2x1`` applies.  Reshaping the interleaved layout straight
    # to a matrix was #955 — the trace ran over (ket1=ket2, bra1=bra2), which
    # is not a trace, and the Hamiltonian contraction hit transposed axes.
    return rdm.transpose(0, 2, 1, 3)


def _rdm1x2_with_open_tensors(
    ao1: jax.Array,
    ao2: jax.Array,
    env: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Raw vertical 2-site transition RDM from open double-layer tensors.

    Args:
        ao1: Top site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        ao2: Bottom site open double-layer ``(D^2, D^2, D^2, D^2, d, d)``.
        env: CTM environment.
        d:   Physical dimension.

    Returns:
        Raw transition RDM ``(d, d, d, d)`` in the grouped
        ``(ket1, ket2, bra1, bra2)`` convention — unnormalised and
        unsymmetrised, for the same reasons as
        ``_rdm2x1_with_open_tensors`` (#954/#955).
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env

    top_row = einsum_promoted("ab,buc,ce->aue", C1, T1, C2)
    env_row1 = einsum_promoted("aue,alf,erg->ulfrg", top_row, T4, T2)
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, ao1)

    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", T4, ao2)
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, T2)

    bot_row = einsum_promoted("hj,jqk,ik->hqi", C4, T3, C3)
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)

    # (ket1, bra1, ket2, bra2) -> (ket1, ket2, bra1, bra2), as in the
    # horizontal helper (#955).
    return rdm.transpose(0, 2, 1, 3)


def _transition_trace(rdm: jax.Array) -> jax.Array:
    """Trace of a grouped ``(ket1, ket2, bra1, bra2)`` transition RDM."""
    return jnp.einsum("ijij->", rdm)


def _rdm2x1_mixed(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    d: int,
    sub_left: tuple[str, str],
    sub_right: tuple[str, str],
) -> jax.Array:
    """Horizontal 2-site RDM with specified ket/bra substitutions.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        env: CTM environment.
        d: Physical dimension.
        sub_left:  ``(ket, bra)`` for left site, each ``"A"`` or ``"B"``.
        sub_right: ``(ket, bra)`` for right site, each ``"A"`` or ``"B"``.

    Returns:
        Transition RDM ``(d, d, d, d)``, grouped ``(ket1, ket2, bra1, bra2)``,
        normalised by the pure-``AA`` contraction of the same geometry: that
        scalar is B-independent (so the RDM stays quadratic in B, #954) and
        carries the same arbitrary environment phase (so the gauge still
        cancels).  With all-``A`` substitutions this reduces to the standard
        ``_rdm2x1`` up to its Hermitian symmetrisation.
    """
    from tenax.algorithms.ipeps_rdm import _build_double_layer_open

    ao1 = _make_open_tensor(A, B, sub_left)
    ao2 = _make_open_tensor(A, B, sub_right)
    rdm = _rdm2x1_with_open_tensors(ao1, ao2, env, d)
    ao_AA = _build_double_layer_open(A)
    n0 = _transition_trace(_rdm2x1_with_open_tensors(ao_AA, ao_AA, env, d))
    return rdm / n0


def _rdm1x2_mixed(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    d: int,
    sub_top: tuple[str, str],
    sub_bottom: tuple[str, str],
) -> jax.Array:
    """Vertical 2-site RDM with specified ket/bra substitutions.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        env: CTM environment.
        d: Physical dimension.
        sub_top:    ``(ket, bra)`` for top site.
        sub_bottom: ``(ket, bra)`` for bottom site.

    Returns:
        Transition RDM ``(d, d, d, d)``, grouped and normalised by the
        pure-``AA`` contraction of the same geometry — see
        ``_rdm2x1_mixed`` (#954/#955).
    """
    from tenax.algorithms.ipeps_rdm import _build_double_layer_open

    ao1 = _make_open_tensor(A, B, sub_top)
    ao2 = _make_open_tensor(A, B, sub_bottom)
    rdm = _rdm1x2_with_open_tensors(ao1, ao2, env, d)
    ao_AA = _build_double_layer_open(A)
    n0 = _transition_trace(_rdm1x2_with_open_tensors(ao_AA, ao_AA, env, d))
    return rdm / n0


def _make_open_tensor(
    A: jax.Array,
    B: jax.Array,
    sub: tuple[str, str],
) -> jax.Array:
    """Build open double-layer tensor for given (ket, bra) substitution.

    Args:
        A: Ground state tensor.
        B: Excitation tensor.
        sub: ``(ket_type, bra_type)`` where each is ``"A"`` or ``"B"``.

    Returns:
        Open double-layer tensor ``(D^2, D^2, D^2, D^2, d, d)``.
    """
    from tenax.algorithms.ipeps_rdm import _build_double_layer_open

    ket_type, bra_type = sub
    if ket_type == "A" and bra_type == "A":
        return _build_double_layer_open(A)
    elif ket_type == "B" and bra_type == "A":
        return _build_mixed_double_layer_open(A, B, "ket")
    elif ket_type == "A" and bra_type == "B":
        return _build_mixed_double_layer_open(A, B, "bra")
    elif ket_type == "B" and bra_type == "B":
        return _build_double_layer_BB_open(B)
    else:
        raise ValueError(f"Invalid substitution: {sub}")


# ---------------------------------------------------------------------------
# Norm and energy functionals
# ---------------------------------------------------------------------------


def _compute_norm(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    d: int,
) -> jax.Array:
    r"""Compute :math:`\langle\Phi_k(B)|\Phi_k(B)\rangle` — the norm of the excitation state.

    For a 1x1 unit cell, the dominant contribution is the on-site term
    (B at the same position in ket and bra). Off-diagonal contributions
    from B at neighboring sites enter with momentum phases
    :math:`e^{i k \cdot r}`.

    The norm is a sesquilinear form in (B*, B): every contraction below is
    divided by the **B-independent** pure-``AA`` contraction of its geometry,
    never by its own trace, so scaling B scales the norm quadratically
    (#954) while the environment's arbitrary phase still cancels.
    """
    from tenax.algorithms.ipeps_rdm import _build_double_layer_open

    # On-site term: B in ket, B* in bra at same site, A elsewhere
    ao_BB = _build_double_layer_BB_open(B)
    ao_AA = _build_double_layer_open(A)

    # B-independent normalisation: <psi|psi> under each contraction geometry.
    n0_h = _transition_trace(_rdm2x1_with_open_tensors(ao_AA, ao_AA, env, d))
    n0_v = _transition_trace(_rdm1x2_with_open_tensors(ao_AA, ao_AA, env, d))

    # Horizontal on-site: (BB, AA) and (AA, BB); vertical likewise.  The norm
    # is the plain trace of each transition RDM.  All four windows measure
    # the SAME quantity — the on-site overlap <Phi_r|Phi_r> — so they are
    # averaged, not summed: summing counted that overlap four times while
    # each energy window contributes a *different* bond operator once, which
    # scaled every generalized eigenvalue by exactly 1/4 on an exact product
    # state (review P1 on #961; each off-site pair below lives in exactly
    # one window, so those are correctly counted once).
    norm_onsite = 0.25 * (
        _transition_trace(_rdm2x1_with_open_tensors(ao_BB, ao_AA, env, d)) / n0_h
        + _transition_trace(_rdm2x1_with_open_tensors(ao_AA, ao_BB, env, d)) / n0_h
        + _transition_trace(_rdm1x2_with_open_tensors(ao_BB, ao_AA, env, d)) / n0_v
        + _transition_trace(_rdm1x2_with_open_tensors(ao_AA, ao_BB, env, d)) / n0_v
    )

    # Off-site terms: B at neighboring sites with momentum phases
    ao_Bket = _build_mixed_double_layer_open(A, B, "ket")
    ao_Bbra = _build_mixed_double_layer_open(A, B, "bra")

    phase_x = jnp.exp(1j * k[0])
    phase_y = jnp.exp(1j * k[1])

    norm_offsite = (
        phase_x
        * _transition_trace(_rdm2x1_with_open_tensors(ao_Bket, ao_Bbra, env, d))
        / n0_h
        + jnp.conj(phase_x)
        * _transition_trace(_rdm2x1_with_open_tensors(ao_Bbra, ao_Bket, env, d))
        / n0_h
        + phase_y
        * _transition_trace(_rdm1x2_with_open_tensors(ao_Bket, ao_Bbra, env, d))
        / n0_v
        + jnp.conj(phase_y)
        * _transition_trace(_rdm1x2_with_open_tensors(ao_Bbra, ao_Bket, env, d))
        / n0_v
    )

    return (norm_onsite + norm_offsite).real


def _compute_excitation_energy(
    A: jax.Array,
    B: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    d: int,
) -> jax.Array:
    r"""Compute :math:`\langle\Phi_k(B)|(H - E_{gs})|\Phi_k(B)\rangle`.

    Uses the shifted Hamiltonian ``H' = H - (E_gs / n_bonds) * I`` per
    bond so that excitation eigenvalues are directly the excitation gaps.

    Contracts 2-site RDMs with B substituted in various positions,
    weighted by momentum phases.
    """
    from tenax.algorithms.ipeps_rdm import _build_double_layer_open

    H = hamiltonian_gate.reshape(d, d, d, d)
    # Shift Hamiltonian: subtract E_gs/2 per bond (2 bonds per site).  The
    # identity must live in the same grouped (ket1, ket2, bra1, bra2) layout
    # as the gate and the transition RDMs (#955): eye(d*d) reshaped is
    # delta(ket1, bra1) * delta(ket2, bra2) in that layout.
    Id4 = jnp.eye(d * d).reshape(d, d, d, d)
    H_shifted = H - (E_gs / 2.0) * Id4

    ao_AA = _build_double_layer_open(A)
    ao_BB = _build_double_layer_BB_open(B)
    ao_Bket = _build_mixed_double_layer_open(A, B, "ket")
    ao_Bbra = _build_mixed_double_layer_open(A, B, "bra")

    # B-independent normalisation per geometry, as in _compute_norm (#954).
    n0_h = _transition_trace(_rdm2x1_with_open_tensors(ao_AA, ao_AA, env, d))
    n0_v = _transition_trace(_rdm1x2_with_open_tensors(ao_AA, ao_AA, env, d))

    # The transition RDM is grouped (ket1, ket2, bra1, bra2) and the gate is
    # (out1, out2, in1, in2) = <o1 o2|H|i1 i2>, so the expectation pairs the
    # RDM's bra axes with the gate's out axes: Tr(rho H).  Pairing axes
    # elementwise ("ijkl,ijkl") instead computes Tr(rho H^T) — identical for
    # the real-symmetric gates the oracle tests use, but sign-flipped for
    # complex Hermitian entries: Sy (x) I on |+y,+y> gave -0.5 instead of
    # +0.5 (review P2 on #961).
    def _e_h(ao1, ao2):
        rdm = _rdm2x1_with_open_tensors(ao1, ao2, env, d)
        return jnp.einsum("ijkl,klij->", rdm, H_shifted) / n0_h

    def _e_v(ao1, ao2):
        rdm = _rdm1x2_with_open_tensors(ao1, ao2, env, d)
        return jnp.einsum("ijkl,klij->", rdm, H_shifted) / n0_v

    phase_x = jnp.exp(1j * k[0])
    phase_y = jnp.exp(1j * k[1])

    # On-site contributions (B at the same site in ket and bra), then
    # off-site (B in ket at one site, B* in bra at the neighbor) with
    # momentum phases.
    energy = (
        _e_h(ao_BB, ao_AA)
        + _e_h(ao_AA, ao_BB)
        + _e_v(ao_BB, ao_AA)
        + _e_v(ao_AA, ao_BB)
        + phase_x * _e_h(ao_Bket, ao_Bbra)
        + jnp.conj(phase_x) * _e_h(ao_Bbra, ao_Bket)
        + phase_y * _e_v(ao_Bket, ao_Bbra)
        + jnp.conj(phase_y) * _e_v(ao_Bbra, ao_Bket)
    )

    return energy.real


# ---------------------------------------------------------------------------
# H_eff and N matrix construction via AD
# ---------------------------------------------------------------------------


def _make_basis(D: int, d: int) -> list[jax.Array]:
    """Generate orthonormal basis vectors for B tensor space.

    Returns a list of ``D^4 * d`` basis tensors, each of shape
    ``(D, D, D, D, d)``.
    """
    basis_size = D**4 * d
    basis = []
    for i in range(basis_size):
        b = jnp.zeros(basis_size).at[i].set(1.0)
        basis.append(b.reshape(D, D, D, D, d))
    return basis


def _build_H_and_N(
    A: jax.Array,
    env: CTMEnvironment,
    k: jax.Array,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    d: int,
    config: ExcitationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build H_eff(k) and N(k) matrices using automatic differentiation.

    For basis vector :math:`e_m` (m-th unit vector in B-parameter space):

    .. math::

        N_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|\Phi_k(B)\rangle\big|_{B=e_m}

        H_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|(H-E_{gs})|\Phi_k(B)\rangle\big|_{B=e_m}

    Both functionals are real-valued sesquilinear forms ``f(B) = B^dag M B``
    with ``M`` Hermitian, so the m-th column is ``M e_m = grad_{B*} f`` at
    ``B = e_m``.  That gradient is assembled from **separate real and
    imaginary coordinate derivatives**, ``M e_m = (df/dx + i df/dy) / 2``:
    differentiating a real-only basis loses every imaginary matrix element
    (the float cotangent cannot carry them, #956), and going through JAX's
    complex-cotangent convention instead invites conjugation mistakes — the
    (x, y) route needs neither.

    Args:
        A:                Optimized ground state tensor.
        env:              Converged CTM environment.
        k:                Momentum vector ``(kx, ky)``.
        hamiltonian_gate: 2-site Hamiltonian.
        E_gs:             Ground state energy per site.
        d:                Physical dimension.
        config:           ExcitationConfig.

    Returns:
        ``(H_eff, N_mat)`` each of shape ``(basis_size, basis_size)``.
    """
    D = A.shape[0]
    basis_size = D**4 * d
    basis = _make_basis(D, d)

    # Stack basis tensors into a single JAX array: (basis_size, D, D, D, D, d)
    B_stacked = jnp.stack(basis)

    def energy_fn(B):
        return _compute_excitation_energy(A, B, env, k, hamiltonian_gate, E_gs, d)

    def norm_fn(B):
        return _compute_norm(A, B, env, k, d)

    y0 = jnp.zeros_like(B_stacked[0])

    def _matrix(fn):
        # grad_{B*} f = (df/dx + i df/dy) / 2 at B = x + iy, evaluated at
        # each real basis vector (y = 0).  Row m of the vmapped result holds
        # M e_m, i.e. the m-th column of M, so the matrix is the transpose
        # (plain, not conjugate: the rows already ARE the columns).
        def f_xy(x, y):
            return fn(x + 1j * y)

        gx, gy = jax.vmap(jax.grad(f_xy, argnums=(0, 1)), in_axes=(0, None))(
            B_stacked, y0
        )
        cols = 0.5 * (gx + 1j * gy)  # (basis_size, D, D, D, D, d)
        return np.array(cols.reshape(basis_size, basis_size).T)

    return _matrix(energy_fn), _matrix(norm_fn)


# ---------------------------------------------------------------------------
# Generalized eigenvalue problem
# ---------------------------------------------------------------------------


def _project_out_ground_state(
    H_eff: np.ndarray,
    N_mat: np.ndarray,
    A: jax.Array,
    k: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Remove the ground-state direction from the excitation pencil.

    ``B \propto A`` is *exactly* null at any ``k != 0``: replacing the site
    tensor by itself leaves the state untouched, so
    :math:`|\Phi_k(A)\rangle = \sum_r e^{ikr}|GS\rangle = N_s\,\delta_{k,0}
    |GS\rangle`.  The full norm form annihilates that direction through its
    infinite separation sum, but this module truncates the sum to
    nearest-neighbour separations, which misrepresents the null as
    ``1 + 2\cos k_x + 2\cos k_y`` — as low as ``-3`` near the M point
    (#961 review round 2).  Modes with a large ground-state component then
    carry spuriously negative norm and are silently discarded by the
    solver's null filter, distorting the spectrum near M.  Projecting the
    direction out *before* solving removes the artifact at its source
    while keeping the physical (tangent-space) quotient exact — rescaling
    the off-site norm windows instead would restore positivity by making
    the metric's k-dependence wrong for every mode that overlaps A.

    **At** ``k = 0`` **the same direction is not null — it is the ground
    state itself** (#961 review round 3), and there the removal has to be
    orthogonal with respect to the *physical* metric ``N``, not the
    Euclidean one: tangent tensors are generally not ground-state-orthogonal
    under ``N`` (measured on an optimized ``D=2`` Heisenberg state:
    ``sin(angle(N a, a)) = 0.58``), so restricting with ``I - |a><a|``
    leaves ground-state weight in the retained metric and admixes the
    ``omega ~ 0`` direction into every retained mode — the Gamma point grew
    a spurious low level at ``0.113`` where the ``N``-orthogonal reduction
    puts the lowest physical mode at ``0.365``.  The oblique projector

    .. math::

        \Pi = I - \frac{a\,(a^\dagger N)}{a^\dagger N a}

    annihilates ``a`` and maps onto the ``N``-orthogonal complement
    ``\{v : a^\dagger N v = 0\}``; sandwiching both matrices restricts the
    pencil to that subspace, and the exact null it leaves along ``a`` is
    dropped by the solver's null filter.  Away from Gamma the direction is
    being deleted as spurious rather than quotiented out, any transverse
    complement is equivalent to truncation order, and the Euclidean
    projector is kept.

    The remaining gauge redundancy of the ansatz (``B`` obtained from ``A``
    by bond gauge transformations) is smaller in norm and stays with the
    solver's relative null filter, as in the reference implementations.
    """
    a = np.asarray(A).ravel().astype(np.complex128)
    a = a / np.linalg.norm(a)
    # Gamma modulo reciprocal lattice vectors, not literal zero: every phase
    # in the pencil is e^{i k r} with integer r, so k = (2 pi, 0) assembles
    # matrices *identical* to k = 0 and must take the same projector -- a
    # literal comparison handed physically equivalent momenta different
    # spectra (#961 review round 4).  Folding into (-pi, pi] first makes the
    # test exact up to the float representation of 2 pi.
    k_folded = np.mod(np.asarray(k, dtype=float) + np.pi, 2.0 * np.pi) - np.pi
    if np.allclose(k_folded, 0.0):
        na = np.asarray(N_mat).conj().T @ a
        denom = a.conj() @ np.asarray(N_mat) @ a
        if abs(denom) < 1e-12 * max(np.linalg.norm(na), 1e-300):
            # The truncated metric thinks the ground state has no norm --
            # degenerate input the oblique quotient would amplify into an
            # unbounded projector.  The Euclidean deletion is the honest
            # remaining move: it removes the direction without dividing by
            # its vanishing N-weight.
            P = np.eye(a.size, dtype=np.complex128) - np.outer(a, a.conj())
        else:
            P = np.eye(a.size, dtype=np.complex128) - np.outer(a, na.conj()) / denom
    else:
        P = np.eye(a.size, dtype=np.complex128) - np.outer(a, a.conj())
    return P.conj().T @ H_eff @ P, P.conj().T @ N_mat @ P


def _solve_excitations(
    H_eff: np.ndarray,
    N_mat: np.ndarray,
    num_excitations: int,
    null_tol: float = 1e-3,
) -> np.ndarray:
    """Solve generalized eigenvalue problem ``H v = omega N v``.

    Steps:
    1. Symmetrize H and N.
    2. Eigendecompose N to find and project out null space.
    3. Solve reduced GEV in the non-null subspace.
    4. Return lowest excitation energies.

    Args:
        H_eff:            Effective Hamiltonian matrix.
        N_mat:            Norm matrix.
        num_excitations:  Number of excitation energies to return.
        null_tol:         Threshold for null-space filtering (relative to
                          largest N eigenvalue).

    Returns:
        Array of the lowest *num_excitations* excitation energies.
    """
    # Symmetrize
    H_eff = 0.5 * (H_eff + H_eff.conj().T)
    N_mat = 0.5 * (N_mat + N_mat.conj().T)

    # Single eigendecomposition of N — use eigenvectors directly for N^{-1/2}
    # transform (avoids redundant projection + re-eigendecomposition which can
    # introduce numerical errors on ill-conditioned norm matrices).
    eigvals_N, P = np.linalg.eigh(N_mat)

    # eigh returns sorted eigenvalues; largest is last
    max_eigval = eigvals_N[-1] if len(eigvals_N) > 0 else 1.0
    if max_eigval < 1e-15:
        return np.zeros(num_excitations)

    safe = eigvals_N > null_tol * max_eigval
    if not np.any(safe):
        return np.zeros(num_excitations)

    # N^{-1/2} regularised ordinary eigenvalue problem
    P_safe = P[:, safe]
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_N[safe]))
    H_tilde = inv_sqrt @ P_safe.conj().T @ H_eff @ P_safe @ inv_sqrt
    H_tilde = 0.5 * (H_tilde + H_tilde.conj().T)
    eigvals = np.linalg.eigvalsh(H_tilde)

    # Always return exactly num_excitations values (pad with zeros if the
    # safe subspace is smaller than num_excitations).
    result = np.zeros(num_excitations)
    n_fill = min(num_excitations, len(eigvals))
    result[:n_fill] = eigvals[:n_fill]
    return result


# ---------------------------------------------------------------------------
# Momentum path utilities
# ---------------------------------------------------------------------------


def make_momentum_path(
    path_type: str = "brillouin",
    num_points: int = 20,
) -> list[tuple[float, float]]:
    r"""Generate momentum path through the Brillouin zone.

    For a square lattice with lattice constant 1:

    ``path_type="brillouin"``:
        :math:`\Gamma(0,0) \to X(\pi,0) \to M(\pi,\pi) \to \Gamma(0,0)`

    ``path_type="diagonal"``:
        :math:`\Gamma(0,0) \to M(\pi,\pi)`

    Args:
        path_type: Type of momentum path.
        num_points: Total number of momentum points.

    Returns:
        List of ``(kx, ky)`` tuples.
    """
    if path_type == "brillouin":
        # Three segments: Gamma->X, X->M, M->Gamma
        n1 = num_points // 3
        n2 = num_points // 3
        n3 = num_points - n1 - n2

        path = []
        # Gamma -> X: (0,0) -> (pi,0)
        for i in range(n1):
            t = i / max(n1, 1)
            path.append((t * np.pi, 0.0))

        # X -> M: (pi,0) -> (pi,pi)
        for i in range(n2):
            t = i / max(n2, 1)
            path.append((np.pi, t * np.pi))

        # M -> Gamma: (pi,pi) -> (0,0)
        for i in range(n3):
            t = i / max(n3, 1)
            path.append(((1 - t) * np.pi, (1 - t) * np.pi))

        return path

    elif path_type == "diagonal":
        path = []
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            path.append((t * np.pi, t * np.pi))
        return path

    else:
        raise ValueError(f"Unknown path_type: {path_type!r}")


# ---------------------------------------------------------------------------
# Main excitation function
# ---------------------------------------------------------------------------


def compute_excitations(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    E_gs: float,
    momenta: list[tuple[float, float]],
    config: ExcitationConfig,
) -> ExcitationResult:
    """Compute excitation spectrum at given momentum points.

    For each momentum point, constructs the effective Hamiltonian and norm
    matrices using AD (Ponsioen et al. 2022), then solves the generalized
    eigenvalue problem for the lowest excitation energies.

    Args:
        A:                 Optimized ground state tensor ``(D, D, D, D, d)``.
                           Accepts a raw ``jax.Array`` or a Tensor object
                           (e.g. the ``DenseTensor`` returned by
                           ``optimize_gs_ad``).
        env:               Converged CTM environment for A.  Accepts the
                           raw-array ``CTMEnvironment`` or the Tensor-based
                           ``CTMTensorEnv`` returned by ``optimize_gs_ad``.
        hamiltonian_gate:  2-site Hamiltonian ``(d, d, d, d)``.  Accepts a raw
                           ``jax.Array`` or a Tensor object.
        E_gs:              Ground state energy per site.
        momenta:           List of ``(kx, ky)`` momentum points.
        config:            ExcitationConfig.

    Returns:
        ExcitationResult with energies and momenta.
    """
    # Accept the Tensor-protocol outputs of ``optimize_gs_ad`` directly by
    # coercing the site tensor, gate, and environment to raw arrays.
    A = _as_dense_array(A)
    hamiltonian_gate = _as_dense_array(hamiltonian_gate)
    env = _as_dense_env(env)

    d = A.shape[-1]

    all_energies = []
    for kx, ky in momenta:
        k = jnp.array([kx, ky])
        H_eff, N_mat = _build_H_and_N(
            A,
            env,
            k,
            hamiltonian_gate,
            E_gs,
            d,
            config,
        )
        H_eff, N_mat = _project_out_ground_state(H_eff, N_mat, A, k)
        excitation_energies = _solve_excitations(
            H_eff,
            N_mat,
            config.num_excitations,
            config.null_space_tol,
        )
        all_energies.append(excitation_energies)

    return ExcitationResult(
        energies=np.array(all_energies),
        momenta=np.array(momenta),
        ground_state_energy=E_gs,
    )
