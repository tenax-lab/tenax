"""AD loss closure for kagome iPESS via honeycomb-supersite CTM.

Investigation summary
---------------------

This module wires the convention-A kagome iPESS coarse-graining
(:func:`tenax.algorithms.pess.pess_to_honeycomb_supersites`) into Tenax's
existing 2-sublattice implicit-AD CTM
(:func:`tenax.algorithms._ctm_energy_ad.ctm_energy_implicit`).

Key facts established by inspecting the existing infrastructure:

1. **CTM input convention.** The standard Tensor-protocol CTM expects
   rank-5 site tensors with leg labels ``(u, d, l, r, phys)`` whose four
   virtual legs all have the *same* dimension D. (See
   ``tenax.algorithms._ctm_tensor_init.initialize_ctm_tensor_env`` and the
   double-layer construction.) Asymmetric virtual dims (e.g. ``u=1, d=D``)
   cause the projector path to fail with a contraction-shape mismatch
   because the move builds fused indices ``chi*u2`` vs ``chi*d2`` of
   different sizes.

2. **Honeycomb supersite shape.** ``pess_to_honeycomb_supersites`` returns
   ``(A_u, A_d)`` of shape ``(d, d, d, D, D, D)``: three physical legs and
   only *three* D-bonds. To feed this into the square-lattice CTM, we
   pre-fuse the three physical legs into a single ``d**3`` leg and pad
   the supersite to four virtual legs. We choose the brick-wall convention
   ``(u_trivial, d=l_a, l=l_b, r=l_c)`` and pad the ``u`` leg to dimension D
   with the basis vector ``e_0 = (1, 0, ..., 0)``. The same convention is
   used for both A_u and A_d, so the dummy ``u/d`` bond contracts trivially
   between supersites and adds no real virtual dimensions to the manifold.

3. **Multisite CTM topology.** We use the existing ``CHECKERBOARD_NEIGHBORS``
   (a → b in all four directions) — the dummy-bond direction simply yields
   a trivial 1-dim effective contraction along that axis.

4. **Energy.** After convention-A coarse-graining, the kagome triangle
   Hamiltonian is purely intra-supersite: every up-triangle bond lives
   inside ``A_u`` and every down-triangle bond lives inside ``A_d``. There
   are *no* inter-supersite bonds for the triangle Hamiltonian, so energy
   is a sum of two 1-site expectation values:

       E_total = Tr(ρ_{A_u} · H_triangle) + Tr(ρ_{A_d} · H_triangle)

   We compute each ``ρ`` by contracting the converged 8-tensor environment
   around the supersite with the open-physical double-layer of A — this is
   the standard 1×1 RDM, simpler than the 2×1/1×2 RDMs used for NN bond
   energies.

Caveats / known limitations
---------------------------

* The dummy-bond padding inflates the iPEPS virtual manifold past the rank
  spanned by the parameterization (the dummy ``u`` leg is rank 1). SVD
  projectors during CTM may therefore see degenerate spectra. This is
  expected and is documented in the design plan
  (``docs/plans/2026-04-25-spin1-xxz-kagome-ipess-design.md``); it does not
  affect correctness, only conditioning. We use ``forward_gauge="phase"``
  and ``projector_method="eigh"`` (the project's defaults for AD) which are
  known to handle this regime.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import _build_double_layer_open_tensor
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, pess_to_honeycomb_supersites
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

__all__ = ["build_pess_loss"]


def _make_supersite_indices(D: int, d_fused: int) -> tuple[TensorIndex, ...]:
    """Build the 5-leg index tuple for a brick-wall-padded supersite.

    Layout: ``(u, d, l, r, phys)`` with all virtual legs at dimension D and
    physical leg at dimension ``d_fused = d**3``. Trivial U(1) charges
    (single sector at charge 0) — the symmetric-tensor charge conservation
    is unused here, only needed for ``TensorIndex`` construction.
    """
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d_fused, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )


def _supersite_to_iPEPS_tensor(  # noqa: N802
    A_super: jax.Array, indices: tuple[TensorIndex, ...]
) -> DenseTensor:
    """Convert a rank-6 honeycomb supersite into a rank-5 brick-wall iPEPS site.

    Pre-fuses the three physical legs into ``d**3`` and pads the ``u``
    direction with ``e_0`` so all four virtual legs share dimension D.

    Args:
        A_super: Supersite tensor of shape ``(d, d, d, D, D, D)`` with index
            order ``(p_a, p_b, p_c, l_a, l_b, l_c)`` (matches the output of
            :func:`pess_to_honeycomb_supersites`).
        indices: Pre-built 5-leg ``(u, d, l, r, phys)`` index tuple from
            :func:`_make_supersite_indices`.

    Returns:
        ``DenseTensor`` of shape ``(D, D, D, D, d**3)`` with leg order
        ``(u, d, l, r, phys)``. The ``u`` leg is occupied only at index 0
        (Kronecker-delta padding).
    """
    d_a, d_b, d_c, D_a, D_b, D_c = A_super.shape
    d_fused = d_a * d_b * d_c
    # Fuse physical legs. The fusion order (p_a, p_b, p_c) must match how
    # the triangle Hamiltonian's d**3 basis was constructed (np.kron(np.kron(s_a,
    # s_b), s_c)) so that ⟨H_triangle⟩ = Tr(ρ · H) is meaningful.
    A_3leg = A_super.reshape(d_fused, D_a, D_b, D_c)
    D = D_a
    if D_b != D or D_c != D:
        raise ValueError(
            "Supersite virtual legs must all share the same bond dimension D; "
            f"got shape {A_super.shape}."
        )
    # Pad the u direction with e_0 = (1, 0, ..., 0).
    e_0 = jnp.zeros(D, dtype=A_super.dtype).at[0].set(1.0)
    A_iPEPS = jnp.einsum("u,Pdlr->udlrP", e_0, A_3leg)
    return DenseTensor(A_iPEPS, indices)


def _rdm1x1_multisite(A: Tensor, env: CTMTensorEnv) -> jax.Array:
    r"""Single-site RDM from a converged CTM environment.

    Contracts the standard 8-tensor environment (4 corners + 4 edges) around
    a single supersite ``A``::

        C1 — T1 — C2
        |    |    |
        T4 — ao — T2
        |    |    |
        C4 — T3 — C3

    Returns a dense Hermitian density matrix of shape
    ``(d_fused, d_fused)`` with unit trace, where ``d_fused`` is the
    physical dimension of ``A``.

    Args:
        A:   Site tensor with labels ``(u, d, l, r, phys)``.
        env: Converged ``CTMTensorEnv`` for the same site.

    Returns:
        ``(d_fused, d_fused)`` dense RDM in the (ket, bra) convention.
    """
    ao = _build_double_layer_open_tensor(A)  # (u2, d2, l2, r2, phys, phys_bra)

    # Pairwise contractions following the corner/edge connectivity used
    # in _rdm2x1_tensor (see _ctm_tensor_energy.py for the full bond table).

    # C1·T1: contract c1_r ↔ t1_l
    C1 = env.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, env.T1)  # (c1_d, u2, t1_r)

    # T1·C2: contract t1_r ↔ c2_l (use already-contracted UL form)
    C2 = env.C2.relabel("c2_l", "t1_r")
    UR = contract(UL, C2)  # (c1_d, u2, c2_d)

    # C4·T3: contract c4_u ↔ t3_r
    C4 = env.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, env.T3)  # (c4_r, d2, t3_l)

    # T3·C3: contract t3_l ↔ c3_l
    C3 = env.C3.relabel("c3_l", "t3_l")
    LR = contract(LL, C3)  # (c4_r, d2, c3_u)

    # Combine top + T4 + bottom into the left/right environment around ao.
    # Bond connectivity from _ctm_tensor_init.CTMTensorEnv comments:
    #   c1_d ↔ t4_d, t4_u ↔ c4_r, c2_d ↔ t2_u, t2_d ↔ c3_u
    T4 = env.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    UL_T4 = contract(UR, T4)  # (u2, c2_d, l2, c4_r)
    Lside = contract(UL_T4, LR)  # (u2, c2_d, l2, d2, c3_u)

    T2 = env.T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})
    Rside = contract(Lside, T2)  # (u2, l2, d2, r2)

    # Sandwich ao: the four virtual legs (u2, d2, l2, r2) close, leaving
    # (phys, phys_bra) as the only remaining indices.
    rdm_t = contract(Rside, ao, output_labels=["phys", "phys_bra"])
    rdm = rdm_t.todense()  # (d_fused, d_fused) — always small.

    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


def build_pess_loss(
    hamiltonian: np.ndarray | jnp.ndarray,
    config: CTMConfig,
) -> Callable[[IPESSState], jnp.ndarray]:
    """Build the AD loss closure for kagome iPESS optimization.

    The returned closure:

      1. Calls :func:`pess_to_honeycomb_supersites` on the input state to
         produce ``(A_u, A_d)`` of shape ``(d, d, d, D, D, D)``.
      2. Pre-fuses physical legs to ``d**3`` and pads each supersite to a
         brick-wall iPEPS shape ``(D, D, D, D, d**3)`` with one trivial
         dummy bond.
      3. Hands the two sites off to :func:`ctm_energy_implicit` (Tenax's
         standard implicit-differentiation CTM AD path) configured with a
         2-site checkerboard topology.
      4. Computes the triangle energy as the sum of two 1-site expectation
         values using a custom ``energy_fn`` that builds the 1-site RDM on
         each supersite from the converged environments.

    Args:
        hamiltonian: ``(d**3, d**3)`` Hermitian triangle Hamiltonian, e.g.
            from :func:`tenax.algorithms.pess.kagome_triangle_xxz_hamiltonian`.
        config: ``CTMConfig`` controlling chi, max_iter, conv_tol, projector
            method, gauge, and GMRES backward solver. The
            ``forward_gauge="phase"`` and ``projector_method="eigh"``
            defaults are recommended.

    Returns:
        ``loss_fn(state: IPESSState) -> jnp.ndarray`` returning a real
        scalar (sum of triangle energies for both up and down supersites).
        Differentiable via ``jax.grad`` through the implicit-AD CTM.
    """
    H = jnp.asarray(hamiltonian, dtype=jnp.complex128)

    def loss_fn(state: IPESSState) -> jnp.ndarray:
        A_u_super, A_d_super = pess_to_honeycomb_supersites(state)
        D = A_u_super.shape[-1]
        d = A_u_super.shape[0]
        d_fused = d**3

        indices = _make_supersite_indices(D, d_fused)
        A_u = _supersite_to_iPEPS_tensor(A_u_super, indices)
        A_d = _supersite_to_iPEPS_tensor(A_d_super, indices)

        site_tensors = {(0, 0): A_u, (1, 0): A_d}

        def _triangle_energy_fn(site_tensors_, envs_, _gate_unused):
            """Energy = ⟨H⟩_{A_u} + ⟨H⟩_{A_d} via 1-site RDMs.

            ``_gate_unused`` is required by the ``ctm_energy_implicit`` API
            (it normally carries a 2-site bond gate). We close over the
            triangle Hamiltonian ``H`` from the outer scope instead.
            """
            rdm_u = _rdm1x1_multisite(site_tensors_[(0, 0)], envs_[(0, 0)])
            rdm_d = _rdm1x1_multisite(site_tensors_[(1, 0)], envs_[(1, 0)])
            E_u = jnp.einsum("ij,ji->", rdm_u, H)
            E_d = jnp.einsum("ij,ji->", rdm_d, H)
            return (E_u + E_d).real

        # ``ctm_energy_implicit`` requires a gate argument; we pass H but the
        # custom ``energy_fn`` ignores it. ``id(gate)`` is part of the VJP
        # cache key, so reusing the same Hamiltonian array across calls
        # keeps the compiled backward warm.
        return ctm_energy_implicit(
            site_tensors,
            CHECKERBOARD_NEIGHBORS,
            H,
            chi=config.chi,
            max_iter=config.max_iter,
            conv_tol=config.conv_tol,
            projector_method=config.projector_method,
            renormalize=config.renormalize,
            projector_backward=config.projector_backward,
            qr_warmup_steps=config.qr_warmup_steps,
            chi_ramp=config.chi_ramp,
            forward_gauge=config.forward_gauge,
            conv_method=config.ctm_conv_method,
            min_iter=config.min_iter,
            gmres_tol=config.gmres_tol,
            gmres_maxiter=config.gmres_maxiter,
            gmres_restart=config.gmres_restart,
            energy_fn=_triangle_energy_fn,
            arnoldi_precheck=False,
        )

    return loss_fn
