"""iPESS (infinite Projected Entangled Simplex State) on the kagome lattice."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from tenax.algorithms.auto_mpo import spin_half_ops, spin_one_ops

D_PHYS_DEFAULT = 3  # spin-1


def _site_ops(d: int) -> dict[str, np.ndarray]:
    """Return ``{"Sz", "Sp", "Sm", "Id"}`` for physical dimension ``d``."""
    if d == 2:
        return spin_half_ops()
    if d == 3:
        return spin_one_ops()
    raise ValueError(f"Unsupported physical dimension d={d}; expected 2 or 3.")


def kagome_triangle_xxz_hamiltonian(
    delta: float, d: int = D_PHYS_DEFAULT
) -> np.ndarray:
    """Build the 3-site XXZ Hamiltonian on a kagome triangle.

    H_tri = H_12 + H_23 + H_31 with each pair term

        H_ij = delta * Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j).

    Args:
        delta: Ising anisotropy on the SzSz channel.
        d: Physical dimension per site (2 for spin-1/2, 3 for spin-1).

    Returns:
        Hermitian ``(d**3, d**3)`` numpy array.
    """
    ops = _site_ops(d)
    Sz = ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    Id = ops["Id"]

    # H_12: acts on sites 1,2; identity on site 3.
    h12 = delta * np.kron(np.kron(Sz, Sz), Id) + 0.5 * (
        np.kron(np.kron(Sp, Sm), Id) + np.kron(np.kron(Sm, Sp), Id)
    )
    # H_23: identity on site 1; acts on sites 2,3.
    h23 = delta * np.kron(Id, np.kron(Sz, Sz)) + 0.5 * (
        np.kron(Id, np.kron(Sp, Sm)) + np.kron(Id, np.kron(Sm, Sp))
    )
    # H_31: acts on sites 1,3; identity on site 2.
    h31 = delta * np.kron(Sz, np.kron(Id, Sz)) + 0.5 * (
        np.kron(Sp, np.kron(Id, Sm)) + np.kron(Sm, np.kron(Id, Sp))
    )

    return h12 + h23 + h31


def make_triangle_gate(
    H: np.ndarray, dt: complex, d: int = D_PHYS_DEFAULT
) -> np.ndarray:
    """Compute ``exp(-dt * H)`` and reshape to a 3-site Trotter gate.

    Args:
        H: ``(d**3, d**3)`` Hamiltonian matrix.
        dt: Time step. Real positive ``dt`` gives imaginary-time evolution;
            imaginary ``dt`` (e.g. ``1j * t``) gives real-time evolution.
        d: Physical dimension per site.

    Returns:
        ``(d, d, d, d, d, d)`` complex128 array — the 3-site Trotter gate.
    """
    gate = expm(-dt * H)
    return gate.reshape(d, d, d, d, d, d).astype(np.complex128)


def hosvd_truncate(
    theta: jnp.ndarray, D_max: int, d: int = D_PHYS_DEFAULT
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
    """Truncate a contracted 3-site tensor back into PESS form via HOSVD.

    Args:
        theta: Rank-6 tensor of shape ``(D_a, D_b, D_c, d, d, d)`` — the
            external bonds for sites a, b, c followed by their three physical
            indices, in matching order.
        D_max: Maximum internal bond dimension to keep on each leg of the
            truncated core. The actual kept dimension on a leg is
            ``min(D_max, D_x * d)`` where ``D_x`` is the external bond size.
        d: Physical dimension per site.

    Returns:
        A tuple ``(S_a, S_b, S_c, core, [lam_a, lam_b, lam_c])`` where
        ``S_x`` has shape ``(D_x_ext, D_x_int, d)``, ``core`` has shape
        ``(D_a_int, D_b_int, D_c_int)``, and each ``lam_x`` is a unit-norm
        singular-value vector along the corresponding internal bond. The
        bond spectra are factored OUT of ``core`` and live exclusively in
        the explicit ``lam_x`` vectors, so the reconstruction
        ``einsum("aip,i,bjq,j,ckr,k,ijk->abcpqr",
                 S_a, lam_a, S_b, lam_b, S_c, lam_c, core)``
        equals ``theta`` exactly when ``D_max >= D_x * d`` for every leg.
    """
    D_ext_a, D_ext_b, D_ext_c = theta.shape[0], theta.shape[1], theta.shape[2]

    theta_reordered = theta.transpose(0, 3, 1, 4, 2, 5)

    # Site a
    mat_a = theta_reordered.reshape(D_ext_a * d, D_ext_b * d * D_ext_c * d)
    U_a, _, _ = jnp.linalg.svd(mat_a, full_matrices=False)
    D_int_a = min(D_max, U_a.shape[1])
    U_a = U_a[:, :D_int_a]

    # Site b
    mat_b = theta_reordered.transpose(2, 3, 0, 1, 4, 5).reshape(
        D_ext_b * d, D_ext_a * d * D_ext_c * d
    )
    U_b, _, _ = jnp.linalg.svd(mat_b, full_matrices=False)
    D_int_b = min(D_max, U_b.shape[1])
    U_b = U_b[:, :D_int_b]

    # Site c
    mat_c = theta_reordered.transpose(4, 5, 0, 1, 2, 3).reshape(
        D_ext_c * d, D_ext_a * d * D_ext_b * d
    )
    U_c, _, _ = jnp.linalg.svd(mat_c, full_matrices=False)
    D_int_c = min(D_max, U_c.shape[1])
    U_c = U_c[:, :D_int_c]

    # Project onto truncated basis (complex-safe: use conj().T, not .T).
    theta_3mode = theta_reordered.reshape(D_ext_a * d, D_ext_b * d, D_ext_c * d)
    core = jnp.tensordot(U_a.conj().T, theta_3mode, axes=([1], [0]))
    core = jnp.tensordot(U_b.conj().T, core, axes=([1], [1]))
    core = core.transpose(1, 0, 2)
    core = jnp.tensordot(U_c.conj().T, core, axes=([1], [2]))
    core = core.transpose(1, 2, 0)

    # Extract per-bond singular value vectors from the core. We only need the
    # singular values, so skip the U/V allocations via compute_uv=False.
    sig_a = jnp.linalg.svd(core.reshape(D_int_a, D_int_b * D_int_c), compute_uv=False)
    sig_b = jnp.linalg.svd(
        core.transpose(1, 0, 2).reshape(D_int_b, D_int_a * D_int_c),
        compute_uv=False,
    )
    sig_c = jnp.linalg.svd(
        core.transpose(2, 0, 1).reshape(D_int_c, D_int_a * D_int_b),
        compute_uv=False,
    )

    # Factor the bond spectra OUT of ``core`` so they live exclusively in the
    # explicit lambda vectors. Without this step, ``core`` carries the spectra
    # baked into its matricizations AND the same spectra are re-applied as
    # gauges on the next SU step (via ``pess_simple_update_triangle``'s
    # ``S_a_w = lam_ext * R * lam_int``), squaring the relative bond spectrum
    # every iteration. That double-counting drives all six lambdas to a
    # rank-1 distribution within ~10 SU steps and pins the energy at the
    # classical 120° value (codex/triage on PR #387).
    #
    # Lorentzian-regularized inverse ``σ / (σ² + ε²)`` is smoother than a
    # hard ``1/σ`` near zero, which matters when ``D_max`` exceeds the rank
    # of ``mat_x(theta)`` and some kept singular values are numerically
    # vanishing.
    eps = 1e-12
    inv_a = (sig_a / (sig_a**2 + eps**2)).astype(core.dtype)
    inv_b = (sig_b / (sig_b**2 + eps**2)).astype(core.dtype)
    inv_c = (sig_c / (sig_c**2 + eps**2)).astype(core.dtype)

    norm_a = jnp.linalg.norm(sig_a)
    norm_b = jnp.linalg.norm(sig_b)
    norm_c = jnp.linalg.norm(sig_c)
    # Per-mode division strips the un-normalized spectrum from ``core``; the
    # scalar prefactor ``norm_a*norm_b*norm_c`` keeps reconstruction with
    # NORMALIZED lambdas exact:
    #   einsum("aip,i,bjq,j,ckr,k,ijk->abcpqr",
    #          S, lam_a, S, lam_b, S, lam_c, core) == theta
    # without it, identity-gate SU would still strip an overall scale per
    # step and the state norm would drift toward zero across many steps.
    K = (norm_a * norm_b * norm_c).astype(core.dtype)
    core = core * inv_a[:, None, None] * inv_b[None, :, None] * inv_c[None, None, :] * K

    lam_a = sig_a / norm_a
    lam_b = sig_b / norm_b
    lam_c = sig_c / norm_c

    S_a = U_a.reshape(D_ext_a, d, D_int_a).transpose(0, 2, 1)
    S_b = U_b.reshape(D_ext_b, d, D_int_b).transpose(0, 2, 1)
    S_c = U_c.reshape(D_ext_c, d, D_int_c).transpose(0, 2, 1)

    return S_a, S_b, S_c, core, [lam_a, lam_b, lam_c]


@dataclass(frozen=True)
class IPESSState:
    """Kagome iPESS parameters.

    R_a, R_b, R_c: rank-3 site tensors, shape (D, D, d). Axis order is
        ``(leg-to-T_d, leg-to-T_u, physical)`` — i.e. axis 0 is the
        bond toward the down-triangle's ``T_d``, axis 1 is the bond
        toward the up-triangle's ``T_u``. This matches the SU einsum
        in :func:`pess_simple_update_triangle`: when ``triangle="up"``
        ``T_u`` contracts axis 1 of each R, and when ``triangle="down"``
        ``T_d`` contracts axis 0. (The earlier docstring on this class
        had axis 0 / axis 1 swapped — the implementation has always
        used the convention documented above; codex P2 review on
        PR #387.)
    T_u, T_d: rank-3 simplex tensors, shape (D, D, D). Axis order
        ``(leg-to-R_a, leg-to-R_b, leg-to-R_c)``.
    lambdas: 6 bond singular-value vectors of length ``D``, ordered
        ``(a-up, b-up, c-up, a-down, b-down, c-down)``. The first
        three weight the R↔T_u bonds, the last three weight the
        R↔T_d bonds.
    """

    R_a: jax.Array
    R_b: jax.Array
    R_c: jax.Array
    T_u: jax.Array
    T_d: jax.Array
    lambdas: tuple[jax.Array, ...]

    @classmethod
    def random(
        cls,
        D: int,
        d: int = D_PHYS_DEFAULT,
        key: jax.Array | None = None,
        scale: float = 0.1,
    ) -> IPESSState:
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        def cmplx(k, shape):
            re = jax.random.normal(k, shape) * scale
            im = jax.random.normal(jax.random.fold_in(k, 1), shape) * scale
            return (re + 1j * im).astype(jnp.complex128)

        return cls(
            R_a=cmplx(keys[0], (D, D, d)),
            R_b=cmplx(keys[1], (D, D, d)),
            R_c=cmplx(keys[2], (D, D, d)),
            T_u=cmplx(keys[3], (D, D, D)),
            T_d=cmplx(keys[4], (D, D, D)),
            lambdas=tuple(jnp.ones(D) for _ in range(6)),
        )


jax.tree_util.register_pytree_node(
    IPESSState,
    lambda s: ((s.R_a, s.R_b, s.R_c, s.T_u, s.T_d, s.lambdas), None),
    lambda _aux, children: IPESSState(
        R_a=children[0],
        R_b=children[1],
        R_c=children[2],
        T_u=children[3],
        T_d=children[4],
        lambdas=children[5],
    ),
)


def _xxz_embed_inter(
    delta: float, d: int, site_1: int, site_2: int, n_sites: int = 3
) -> np.ndarray:
    """Embed the XXZ pair Hamiltonian on (sub-site ``site_1`` of left
    supersite, sub-site ``site_2`` of right supersite) into a rank-4
    ``(d_eff, d_eff, d_eff, d_eff)`` operator.

    Generalizes :func:`tenax.algorithms.coarse_grain._embed_inter` to
    arbitrary physical dimension and XXZ anisotropy. Acts as identity on
    every other sub-site of each supersite. Index layout
    ``(cg1_out, cg2_out, cg1_in, cg2_in)``.
    """
    ops = _site_ops(d)
    Sz = ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    Id = ops["Id"]
    d_eff = d**n_sites
    result = np.zeros((d_eff, d_eff, d_eff, d_eff), dtype=np.complex128)
    # XXZ pair = δ Sz Sz + 0.5 (S+ S- + S- S+)
    for Sa, Sb in [(delta * Sz, Sz), (0.5 * Sp, Sm), (0.5 * Sm, Sp)]:
        op1 = [Id.copy() for _ in range(n_sites)]
        op1[site_1] = Sa
        kron1 = op1[0]
        for o in op1[1:]:
            kron1 = np.kron(kron1, o)
        op2 = [Id.copy() for _ in range(n_sites)]
        op2[site_2] = Sb
        kron2 = op2[0]
        for o in op2[1:]:
            kron2 = np.kron(kron2, o)
        result = result + np.einsum("ik,jl->ijkl", kron1, kron2)
    return result


def kagome_xxz_pess_cg_gates(delta: float = 1.0, d: int = D_PHYS_DEFAULT):
    """Build :class:`tenax.algorithms.coarse_grain.CGGates` for kagome XXZ
    with an iPESS ``map_fn`` and ``init_fn``.

    The supersite is the up-triangle (Convention C). Phys-sub-leg
    ordering ``(a, b, c) = (u, v, w)``. Inter-supersite directions match
    :func:`tenax.algorithms.coarse_grain.kagome_cg_gates`:

    - ``"h"`` (horizontal): ``c`` of left ↔ ``b`` of right (down-tri bond ``b–c``).
    - ``"v"`` (vertical): ``b`` of top ↔ ``a`` of bottom (down-tri bond ``a–b``).
    - ``"diag"`` (diagonal): ``c`` of top ↔ ``a`` of bottom-right (down-tri bond ``a–c``).

    Together with ``h_intra`` (the up-triangle XXZ Hamiltonian) this
    captures all 6 kagome bonds per unit cell.

    Args:
        delta: XXZ anisotropy (Δ=1 is isotropic Heisenberg).
        d: Physical dimension per site (2 for spin-½, 3 for spin-1).

    Returns:
        :class:`CGGates` with ``map_fn = pess_to_kagome_supersite`` and
        ``init_fn`` returning a tuple
        ``(R_a, R_b, R_c, T_u, lam0, lam1, lam2, lam3, lam4, lam5)``
        suitable for :func:`tenax.optimize_gs_ad` warm-start.
    """
    from tenax.algorithms.coarse_grain import CGGates

    h_intra = jnp.asarray(
        kagome_triangle_xxz_hamiltonian(delta, d), dtype=jnp.complex128
    )
    h_inter = {
        "h": jnp.asarray(_xxz_embed_inter(delta, d, 2, 1), dtype=jnp.complex128),
        "v": jnp.asarray(_xxz_embed_inter(delta, d, 1, 0), dtype=jnp.complex128),
        "diag": jnp.asarray(_xxz_embed_inter(delta, d, 2, 0), dtype=jnp.complex128),
    }

    def _flat_map_fn(R_a, R_b, R_c, T_u, lam0, lam1, lam2, lam3, lam4, lam5):
        return pess_to_kagome_supersite(
            R_a, R_b, R_c, T_u, (lam0, lam1, lam2, lam3, lam4, lam5)
        )

    def _flat_init_fn(D: int, key: jax.Array) -> tuple[jax.Array, ...]:
        state = IPESSState.random(D=D, d=d, key=key)
        return (
            state.R_a,
            state.R_b,
            state.R_c,
            state.T_u,
            *state.lambdas,
        )

    return CGGates(
        h_intra=h_intra,
        h_inter=h_inter,
        n_sites=3,
        map_fn=_flat_map_fn,
        init_fn=_flat_init_fn,
    )


def pess_to_kagome_supersite(
    R_a: jax.Array,
    R_b: jax.Array,
    R_c: jax.Array,
    T_u: jax.Array,
    lambdas: tuple[jax.Array, ...] | jax.Array,
) -> jax.Array:
    """Build a square-iPEPS coarse-grained supersite from iPESS primitives.

    Implements the standard kagome → square coarse-graining (Liao 2019,
    "Convention C"): one supersite per up-triangle carrying ``T_u`` and
    its three ``R`` sites. ``T_d`` is absorbed implicitly via the bond
    gauges ``lambdas[3:6]`` (down-bond singular values produced by
    :func:`pess_simple_update`); each down-bond contributes ``sqrt(λ)``
    to the supersite and ``sqrt(λ)`` to the neighboring supersite.

    The resulting tensor has shape ``(D, D, D, D, d**3)``: 4 virtual legs
    (left, right, up, down) and a fused physical leg ordering
    ``(p_a, p_b, p_c)`` row-major. Three of the virtual legs carry the
    R-sites' down-bonds toward neighboring supersites (after the
    ``sqrt(λ_down)`` gauge is applied here), and the fourth is the
    Convention-C dummy: dimension 1 padded to ``D`` so the tensor
    matches the square-iPEPS rank-5 layout. Only ``A[:, :, :, 0, :]``
    is non-zero.

    Per the actual SU einsum convention (see
    :func:`pess_simple_update_triangle`), axis 0 of each ``R`` is the
    T_d-leg and axis 1 is the T_u-leg. ``T_u`` is contracted against
    axis 1 of each R; the remaining axis-0 legs become the supersite's
    three real virtual legs.

    Phys-leg ordering ``(p_a, p_b, p_c)`` matches
    :func:`kagome_triangle_xxz_hamiltonian` (whose ``np.kron`` order is
    ``(Sa, Sb, Sc)``), and matches the sub-leg ordering used by
    :func:`tenax.algorithms.coarse_grain.kagome_cg_gates` (sub-site 0
    = ``a`` = ``u``, 1 = ``b`` = ``v``, 2 = ``c`` = ``w``).

    .. warning::
        **Odd-D AD instability.** The dummy 4th virtual bond contributes
        ``D − 1`` zero singular values per direction to the square-CTM
        SVD projector. At even ``D`` (2, 4, 6, …) this stays absorbed
        and the AD path is numerically clean; at odd ``D`` (especially
        ``D = 3``) the projector spectrum is sufficiently degenerate
        that the converged CTM energy can flip between physically
        sensible values and zero across nearby ``chi`` choices, which
        cascades into ill-conditioned implicit-AD gradients. Until the
        AD path is hardened against this (e.g. Lorentzian-regularized
        eigh projector or QR-warmup → SVD with ``S_safe`` floor), use
        even ``D`` for any AD-driven optimization.

        This affects **only the AD path** (`build_pess_loss`,
        `optimize_pess_ad`). The simple-update flow
        (:func:`pess_simple_update_triangle`,
        :func:`pess_simple_update`) operates directly on the iPESS
        primitives via HOSVD truncation and does **not** construct
        this supersite; SU is unaffected at every ``D``, including
        odd values.

    Args:
        R_a, R_b, R_c: iPESS site tensors of shape ``(D, D, d)``,
            axes ``(T_d-leg, T_u-leg, phys)``.
        T_u: iPESS up-simplex tensor of shape ``(D, D, D)``, axes
            ``(R_a-leg, R_b-leg, R_c-leg)``.
        lambdas: 6 bond singular-value vectors of length ``D``, ordered
            ``(a-up, b-up, c-up, a-down, b-down, c-down)``. The up-bonds
            are absorbed fully (they live inside this triangle); the
            down-bonds enter as ``sqrt(λ)``.

    Returns:
        Rank-5 supersite of shape ``(D, D, D, D, d**3)``.
    """
    D = R_a.shape[0]
    d = R_a.shape[2]
    d_eff = d**3
    dtype = R_a.dtype

    # Up-bond lambdas: absorbed fully (these bonds live inside the up-triangle).
    lam_a_u = lambdas[0].astype(dtype)
    lam_b_u = lambdas[1].astype(dtype)
    lam_c_u = lambdas[2].astype(dtype)
    # Down-bond lambdas: sqrt-split; the other sqrt goes to the neighboring
    # supersite via the bond gauge. Use ``(λ² + 1e-28) ** 0.25`` instead of
    # ``sqrt(max(λ, 1e-14))`` so the gauge is smooth in the AD parameters.
    # The hard ``maximum`` would zero out gradients whenever a component
    # crossed the ``1e-14`` floor during L-BFGS, leaving that bond weight
    # permanently stuck (codex P1 review on PR #387). The squared form is
    # |λ|^{1/2} for |λ| ≫ 1e-14 and smoothly regularizes near zero.
    sqrt_lam_a_d = jnp.power(jnp.real(lambdas[3]) ** 2 + 1e-28, 0.25).astype(dtype)
    sqrt_lam_b_d = jnp.power(jnp.real(lambdas[4]) ** 2 + 1e-28, 0.25).astype(dtype)
    sqrt_lam_c_d = jnp.power(jnp.real(lambdas[5]) ** 2 + 1e-28, 0.25).astype(dtype)

    # Gauge each R: axis 0 (T_d-leg) gets sqrt(λ_down), axis 1 (T_u-leg)
    # gets λ_up.
    S_a = jnp.einsum("i,ijp,j->ijp", sqrt_lam_a_d, R_a, lam_a_u)
    S_b = jnp.einsum("i,ijp,j->ijp", sqrt_lam_b_d, R_b, lam_b_u)
    S_c = jnp.einsum("i,ijp,j->ijp", sqrt_lam_c_d, R_c, lam_c_u)

    # Contract gauged R's with T_u via the T_u-leg (axis 1 of S_x). Leaves
    # axis 0 (T_d-leg) of each S_x as the supersite's outgoing virtual leg.
    theta = jnp.einsum("xap,ybq,zcr,abc->xyzpqr", S_a, S_b, S_c, T_u)

    # Fuse the three physical legs into d_eff = d^3 in row-major order so the
    # basis matches kagome_triangle_xxz_hamiltonian's np.kron(np.kron(Sa, Sb), Sc).
    theta_phys = theta.reshape(D, D, D, d_eff)

    # Pad onto the square-iPEPS layout (D, D, D, D, d_eff). The 4th virtual
    # leg is Convention C's "dummy" — dimension 1 padded to D, with non-zero
    # weight only on the index-0 slice. Mirrors examples/kagome_xxz_pess.py:
    # the padding lets compute_energy_cg's existing 2-site RDMs (which assume
    # 4 nontrivial virtual legs) operate on the supersite without the CTM
    # itself ever touching the dummy index outside slot 0.
    A = jnp.zeros((D, D, D, D, d_eff), dtype=dtype)
    A = A.at[:, :, :, 0, :].set(theta_phys)
    return A


def pess_simple_update_triangle(
    state: IPESSState,
    gate: jax.Array,
    triangle: str,
    D_max: int,
) -> IPESSState:
    """One simple-update step on a single kagome triangle.

    Absorbs the triangle's external and internal bond weights into the three
    site tensors, contracts them with the chosen simplex tensor and the 3-site
    gate, then HOSVD-truncates the result back into iPESS form.

    Args:
        state: Input iPESS state.
        gate: 3-site gate of shape ``(d, d, d, d, d, d)``. Index order matches
            ``einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate)`` from the example
            (last 3 axes are ``ket``, first 3 are ``bra``).
        triangle: ``"up"`` updates ``T_u`` and the up-bond lambdas
            (indices 0, 1, 2); ``"down"`` updates ``T_d`` and the down-bond
            lambdas (indices 3, 4, 5).
        D_max: Maximum internal bond dimension kept by the HOSVD truncation.

    Returns:
        New :class:`IPESSState` with the updated site tensors, the chosen
        simplex tensor, and the matching internal lambda triplet replaced;
        the other simplex and the external lambdas are unchanged.

    Raises:
        ValueError: If ``triangle`` is not ``"up"`` or ``"down"``.
    """
    if triangle == "up":
        T = state.T_u
        ext_idx = (3, 4, 5)
        int_idx = (0, 1, 2)
        # ``IPESSState`` convention: R axis 0 = T_d-leg, axis 1 = T_u-leg.
        # For "up" SU the ext side (away from the simplex being updated) is
        # the T_d-leg (axis 0) and the int side (toward T_u being updated)
        # is the T_u-leg (axis 1). The body of this function assumes this
        # axis layout (ext on axis 0, int on axis 1, simplex contracts
        # axis 1 of R), so no transpose is needed.
        R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    elif triangle == "down":
        T = state.T_d
        ext_idx = (0, 1, 2)
        int_idx = (3, 4, 5)
        # For "down" SU the ext side is the T_u-leg (axis 1) and the int
        # side (toward T_d being updated) is the T_d-leg (axis 0).
        # Transpose axes 0 and 1 so the body sees ext on axis 0 again;
        # transpose back at the end. Without this, the simplex contraction
        # (which always contracts R's axis-1 leg via the einsum below)
        # would tie T_d to the T_u-leg of R, which is geometrically wrong:
        # the down-triangle gate would not actually evolve the down-bonds
        # and only the up-triangle would converge under SU.
        R_a = state.R_a.transpose(1, 0, 2)
        R_b = state.R_b.transpose(1, 0, 2)
        R_c = state.R_c.transpose(1, 0, 2)
    else:
        raise ValueError(f"triangle must be 'up' or 'down'; got {triangle!r}.")

    lam_ext = (
        state.lambdas[ext_idx[0]],
        state.lambdas[ext_idx[1]],
        state.lambdas[ext_idx[2]],
    )
    lam_int = (
        state.lambdas[int_idx[0]],
        state.lambdas[int_idx[1]],
        state.lambdas[int_idx[2]],
    )

    d = R_a.shape[2]

    # Weight site tensors with ext (axis 0) and int (axis 1) lambdas.
    S_a_w = jnp.einsum("i,ijd,j->ijd", lam_ext[0], R_a, lam_int[0])
    S_b_w = jnp.einsum("i,ijd,j->ijd", lam_ext[1], R_b, lam_int[1])
    S_c_w = jnp.einsum("i,ijd,j->ijd", lam_ext[2], R_c, lam_int[2])

    # Contract weighted site tensors with the simplex into theta, apply gate.
    theta = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", S_a_w, S_b_w, S_c_w, T)
    theta = jnp.einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate)

    # HOSVD-truncate back to iPESS form.
    S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new = hosvd_truncate(theta, D_max, d)

    # Strip the external lambdas back off so the returned R tensors are
    # "ungauged" again. Safe-divide to avoid division by zero.
    def _safe_inv(lam):
        return jnp.where(lam > 1e-12, 1.0 / lam, 0.0)

    inv_a = _safe_inv(lam_ext[0])
    inv_b = _safe_inv(lam_ext[1])
    inv_c = _safe_inv(lam_ext[2])
    R_a_new = jnp.einsum("i,ijd->ijd", inv_a, S_a_new)
    R_b_new = jnp.einsum("i,ijd->ijd", inv_b, S_b_new)
    R_c_new = jnp.einsum("i,ijd->ijd", inv_c, S_c_new)

    if triangle == "down":
        # After the SU body, R_x_new has axis 0 = the un-gauged T_u-leg and
        # axis 1 = the new T_d-leg from HOSVD. Transpose back so axis 0 =
        # T_d-leg per IPESSState convention.
        R_a_new = R_a_new.transpose(1, 0, 2)
        R_b_new = R_b_new.transpose(1, 0, 2)
        R_c_new = R_c_new.transpose(1, 0, 2)

    # Build new lambdas tuple with the int triplet replaced.
    new_lambdas = list(state.lambdas)
    new_lambdas[int_idx[0]] = lambdas_int_new[0]
    new_lambdas[int_idx[1]] = lambdas_int_new[1]
    new_lambdas[int_idx[2]] = lambdas_int_new[2]

    if triangle == "up":
        return replace(
            state,
            R_a=R_a_new,
            R_b=R_b_new,
            R_c=R_c_new,
            T_u=T_new,
            lambdas=tuple(new_lambdas),
        )
    return replace(
        state,
        R_a=R_a_new,
        R_b=R_b_new,
        R_c=R_c_new,
        T_d=T_new,
        lambdas=tuple(new_lambdas),
    )


def pess_simple_update(
    state: IPESSState,
    hamiltonian: np.ndarray,
    dt_schedule: list[tuple[float, int]],
    D_max: int,
) -> IPESSState:
    """Run alternating up/down triangle simple update on a kagome iPESS state.

    For each ``(dt, num_steps)`` segment in ``dt_schedule``, build the 3-site
    Trotter gate ``exp(-dt * H)`` once and run ``num_steps`` SU iterations.
    Each iteration performs one up-triangle update followed by one
    down-triangle update.

    Args:
        state: Input iPESS state.
        hamiltonian: ``(d**3, d**3)`` Hermitian Hamiltonian on a triangle
            (e.g. from :func:`kagome_triangle_xxz_hamiltonian`).
        dt_schedule: List of ``(dt, num_steps)`` tuples, e.g.
            ``[(0.1, 200), (0.01, 200), (0.001, 100)]``. Real positive ``dt``
            performs imaginary-time evolution.
        D_max: Maximum internal bond dimension preserved by the per-step
            HOSVD truncation.

    Returns:
        New :class:`IPESSState` after the full schedule.
    """
    d = state.R_a.shape[2]
    for dt, num_steps in dt_schedule:
        gate = jnp.asarray(make_triangle_gate(hamiltonian, dt, d))
        for _ in range(num_steps):
            state = pess_simple_update_triangle(state, gate, "up", D_max)
            state = pess_simple_update_triangle(state, gate, "down", D_max)
    return state


def pess_local_energy(state: IPESSState, h_tri: np.ndarray) -> jnp.ndarray:
    """Husimi-tree mean-field energy per kagome site for a 3-PESS state.

    Computes ``<H_tri>`` on the up-triangle and the down-triangle
    independently, treating the bond-λ singular values on the *external*
    legs of each triangle as a mean-field environment. With Vidal-canonical
    SU output (Σ λ² = 1), full λ_ext is absorbed on each side of the RDM,
    yielding λ_ext² weights in the squared amplitude — the canonical
    on-triangle reduced-density-matrix gauge. The two triangle
    energies are averaged; for an SU-converged state they should agree
    by triangle symmetry. ``E/site = (E_up + E_down) / 3``.

    This is the cheapest energy estimate consistent with the bond-gauge
    information SU produces. It does not match Liao 2017's MPS-projection
    measurement quantitatively (Liao uses ``D_mps ≈ 4·D²`` on a 1D MPS
    basis, which captures more correlation), but it tracks the
    SU-correctness of the kernel without going through the
    Convention-C + CTM measurement that ``build_pess_loss`` uses.

    Args:
        state: Input :class:`IPESSState`. Typically the output of
            :func:`pess_simple_update`.
        h_tri: ``(d**3, d**3)`` Hermitian triangle Hamiltonian in the
            ``(p_a, p_b, p_c)`` row-major basis (matches
            :func:`kagome_triangle_xxz_hamiltonian`).

    Returns:
        Real scalar JAX array — the energy per kagome site.
    """
    R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    T_u, T_d = state.T_u, state.T_d
    lam_au, lam_bu, lam_cu = state.lambdas[0:3]
    lam_ad, lam_bd, lam_cd = state.lambdas[3:6]

    dtype = R_a.dtype
    d = R_a.shape[2]
    h_tri_jax = jnp.asarray(h_tri).astype(dtype).reshape(d, d, d, d, d, d)

    def _triangle_energy(
        Ra: jax.Array,
        Rb: jax.Array,
        Rc: jax.Array,
        T: jax.Array,
        lam_int_a: jax.Array,
        lam_int_b: jax.Array,
        lam_int_c: jax.Array,
        lam_ext_a: jax.Array,
        lam_ext_b: jax.Array,
        lam_ext_c: jax.Array,
    ) -> jax.Array:
        """One triangle's energy with bond gauges (axis 0 = ext, axis 1 = int)."""
        # Full λ_ext on each external leg → λ_ext² in the squared amplitude.
        # In Vidal canonical form (Σ λ² = 1), the on-triangle RDM carries the
        # squared singular values on each external bond; absorbing full λ on
        # both bra and ket reproduces those weights in ⟨ψ|ψ⟩ and ⟨ψ|H|ψ⟩.
        # (The sqrt-gauge form is the right convention when *combining*
        # supersites in a 2D iPEPS — see pess_to_kagome_supersite — because
        # the neighbor contributes the other sqrt(λ). For the isolated-
        # triangle Husimi probe here, no neighbor is present, so we must put
        # the full λ on this side.)
        gauge_a = jnp.real(lam_ext_a).astype(dtype)
        gauge_b = jnp.real(lam_ext_b).astype(dtype)
        gauge_c = jnp.real(lam_ext_c).astype(dtype)

        # Full λ_int absorbed onto axis 1 of each R (the simplex-side bond).
        Q_a = jnp.einsum("i,ijp,j->ijp", gauge_a, Ra, lam_int_a.astype(dtype))
        Q_b = jnp.einsum("i,ijp,j->ijp", gauge_b, Rb, lam_int_b.astype(dtype))
        Q_c = jnp.einsum("i,ijp,j->ijp", gauge_c, Rc, lam_int_c.astype(dtype))

        # Triangle ket ψ[A, B, C, p_a, p_b, p_c]
        psi = jnp.einsum("Aap,Bbq,Ccr,abc->ABCpqr", Q_a, Q_b, Q_c, T)

        norm = jnp.einsum("ABCpqr,ABCpqr->", psi.conj(), psi)
        e_un = jnp.einsum(
            "ABCpqr,pqrPQR,ABCPQR->",
            psi.conj(),
            h_tri_jax,
            psi,
        )
        return jnp.real(e_un / (norm + 1e-30))

    # Up-triangle: int = up-bonds (R axis 1), ext = down-bonds (R axis 0).
    e_up = _triangle_energy(
        R_a,
        R_b,
        R_c,
        T_u,
        lam_au,
        lam_bu,
        lam_cu,
        lam_ad,
        lam_bd,
        lam_cd,
    )

    # Down-triangle: int = down-bonds (axis 0), ext = up-bonds (axis 1).
    # Swap R axes 0↔1 so the helper's "axis 1 = int" convention holds.
    e_dn = _triangle_energy(
        R_a.transpose(1, 0, 2),
        R_b.transpose(1, 0, 2),
        R_c.transpose(1, 0, 2),
        T_d,
        lam_ad,
        lam_bd,
        lam_cd,
        lam_au,
        lam_bu,
        lam_cu,
    )

    return (e_up + e_dn) / 3.0
