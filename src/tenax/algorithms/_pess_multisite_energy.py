"""Per-bond gates and energy wrapper for the kagome 3-site multisite iPESS.

Companion to :func:`tenax.algorithms.pess.pess_to_kagome_3site_multisite`.
The 3-site multisite encoding maps each kagome sublattice onto one ``d=2``
(or ``d=3``) site of a 3-site multisite iPEPS keyed by ``{"u", "v", "w"}``;
the 4 kagome a-b/a-c bonds become NN bonds on the kagome
:func:`tenax.core.lattice.kagome` Lattice (handled by the standard 2-site
NN RDMs), and the 2 kagome b-c bonds — encoded as 3-body correlations
inside ``S_u`` because the encoding absorbs both ``T_u`` and ``T_d`` into
``S_u`` — are routed through a 3-site marginalised RDM helper that closes
a 1×3 (or 3×1) block (w-u-v) and traces over u's physical leg. See
``docs/plans/2026-05-05-multisite-kagome-pess.md`` "ACTIVE PLAN" Phase B
and Phase B-revised.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor_energy import (
    _normalise_rdm_for_energy,
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
)
from tenax.algorithms.pess import _site_ops
from tenax.contraction.contractor import contract
from tenax.core.tensor import Tensor

__all__ = [
    "kagome_xxz_pair_hamiltonian",
    "kagome_3site_bond_gates",
    "compute_energy_pess_3site_multisite",
]


def kagome_xxz_pair_hamiltonian(delta: float = 1.0, d: int = 2) -> np.ndarray:
    """Single XXZ pair Hamiltonian on two ``d``-dimensional sites.

    ``H = δ Sz Sz + 0.5 (S+ S- + S- S+)``. Returned as ``(d, d, d, d)`` array
    with index layout ``(site1_out, site2_out, site1_in, site2_in)`` matching
    the gate convention consumed by ``_rdm{2x1,1x2}_tensor_2site``'s
    ``einsum("ijkl,ijkl->", rdm, H)`` contraction.
    """
    ops = _site_ops(d)
    Sz = ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    H = delta * np.einsum("ik,jl->ijkl", Sz, Sz)
    H = H + 0.5 * np.einsum("ik,jl->ijkl", Sp, Sm)
    H = H + 0.5 * np.einsum("ik,jl->ijkl", Sm, Sp)
    return H.astype(np.complex128)


def kagome_3site_bond_gates(
    delta: float = 1.0, d: int = 2
) -> dict[frozenset, jax.Array]:
    """Per-bond XXZ gates for the 3-site multisite kagome iPESS.

    Returns 6 entries — one per kagome bond — keyed by
    ``frozenset({(name_a, dir_a), (name_b, dir_b)})`` so the order in which
    the energy iterator emits a bond does not matter (both
    ``(coord, direction)`` and ``(neighbor, reverse_direction)`` hash to the
    same bond ID).  Direction labels match
    :func:`tenax.core.lattice.kagome` 's neighbor map:

    Up-triangle (NN-h, "right"/"left"):
      - ``u-v``: ``frozenset({("u","right"), ("v","left")})``
      - ``v-w``: ``frozenset({("v","right"), ("w","left")})``
      - ``w-u``: ``frozenset({("w","right"), ("u","left")})``

    Down-triangle (NN-v, "bottom"/"top"):
      - ``u-v``: ``frozenset({("u","bottom"), ("v","top")})``
      - ``v-w``: ``frozenset({("v","bottom"), ("w","top")})``
      - ``w-u``: ``frozenset({("w","bottom"), ("u","top")})``

    All 6 gates are the same XXZ pair Hamiltonian (uniform kagome).
    """
    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta, d))
    return {
        frozenset({("u", "right"), ("v", "left")}): H_pair,
        frozenset({("v", "right"), ("w", "left")}): H_pair,
        frozenset({("w", "right"), ("u", "left")}): H_pair,
        frozenset({("u", "bottom"), ("v", "top")}): H_pair,
        frozenset({("v", "bottom"), ("w", "top")}): H_pair,
        frozenset({("w", "bottom"), ("u", "top")}): H_pair,
    }


def _rdm_3site_marginal_vw_row(
    S_u: Tensor,
    S_v: Tensor,
    S_w: Tensor,
    env_u: CTMTensorEnv,
    env_v: CTMTensorEnv,
    env_w: CTMTensorEnv,
) -> jax.Array:
    """3-site marginalised 2-site RDM for the v–w bond mediated by ``T_u``.

    Closes the multisite CTM environment around a horizontal 1×3 block
    laid out as ``w`` (left), ``u`` (middle), ``v`` (right) on the kagome
    Lattice — matching the up-triangle b–c bond ``v.right ↔ w.left``,
    which is structurally a 3-body correlation on the multisite iPEPS
    because both ``T_u`` and ``T_d`` are absorbed into ``S_u``.  The
    middle site's physical leg is traced (via the closed double-layer
    ``ac_u``); the v and w physical legs stay open.

    Returns a dense rank-4 array of shape ``(d, d, d, d)`` in
    ``(s_v_ket, s_w_ket, s_v_bra, s_w_bra)`` order, symmetrised and
    trace-normalised, ready to contract with the same XXZ gate
    convention used by :func:`_rdm2x1_tensor_2site`.

    The peak intermediate is bounded by ``χ²·D⁶`` (matching the supersite
    diagonal-RDM cost), with the boundary halves consuming their open
    ``ao`` factor at the standard ``χ²·D⁴·d²`` floor.
    """
    ao_W = _build_double_layer_open_tensor(S_w)
    ac_U = _build_double_layer_tensor(S_u)
    ao_V = _build_double_layer_open_tensor(S_v)

    # ----- Column W (leftmost; full corners on the left side) -----
    C1 = env_w.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, env_w.T1)  # (c1_d, u2, t1_r)
    C4 = env_w.C4.relabel("c4_u", "t3_r")
    LL = contract(C4, env_w.T3)  # (c4_r, d2, t3_l)
    T4 = env_w.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    UL_T4 = contract(UL, T4)  # (u2, t1_r, l2, c4_r)
    Lenv = contract(UL_T4, LL)  # (u2, t1_r, l2, d2, t3_l)
    ao_W_lbl = ao_W.relabels({"phys": "phys_W", "phys_bra": "phys_braW"})
    site_env_W = contract(Lenv, ao_W_lbl)
    # → (t1_r, r2, t3_l, phys_W, phys_braW)

    # ----- Column V (rightmost; full corners on the right side) -----
    T1_V = env_v.T1.relabels({"t1_l": "t1_lV", "u2": "u2V", "t1_r": "t1_rV"})
    T3_V = env_v.T3.relabels({"t3_r": "t3_rV", "d2": "d2V", "t3_l": "t3_lV"})
    C2 = env_v.C2.relabel("c2_l", "t1_rV")
    UR = contract(T1_V, C2)  # (t1_lV, u2V, c2_d)
    C3 = env_v.C3.relabel("c3_l", "t3_lV")
    LR = contract(T3_V, C3)  # (t3_rV, d2V, c3_u)
    T2 = env_v.T2.relabels({"t2_u": "c2_d", "r2": "r2V", "t2_d": "c3_u"})
    UR_T2 = contract(UR, T2)  # (t1_lV, u2V, r2V, c3_u)
    Renv = contract(UR_T2, LR)  # (t1_lV, u2V, r2V, t3_rV, d2V)
    ao_V_lbl = ao_V.relabels(
        {
            "u2": "u2V",
            "d2": "d2V",
            "l2": "l2V",
            "r2": "r2V",
            "phys": "phys_V",
            "phys_bra": "phys_braV",
        }
    )
    site_env_V = contract(Renv, ao_V_lbl)
    # → (t1_lV, t3_rV, l2V, phys_V, phys_braV)

    # ----- Column U (middle; T1, T3 only — no corners; ac closes phys) -----
    # Boundary chi labels match the W (left) and V (right) halves.  Inner
    # D² labels ``r2`` (W boundary) and ``l2V`` (V boundary) are shared
    # across the column-W/U/V seam.
    T1_U = env_u.T1.relabels({"t1_l": "t1_r", "u2": "u2_U", "t1_r": "t1_lV"})
    T3_U = env_u.T3.relabels({"t3_r": "t3_l", "d2": "d2_U", "t3_l": "t3_rV"})
    ac_U_lbl = ac_U.relabels({"u2": "u2_U", "d2": "d2_U", "l2": "r2", "r2": "l2V"})
    T1_acU = contract(T1_U, ac_U_lbl)  # (t1_r, t1_lV, d2_U, r2, l2V)
    site_env_U = contract(T1_acU, T3_U)
    # → (t1_r, t1_lV, r2, l2V, t3_l, t3_rV)

    # ----- Combine: site_env_W · site_env_U · site_env_V -----
    WU = contract(site_env_W, site_env_U)
    # shared: t1_r, r2, t3_l → (phys_W, phys_braW, t1_lV, l2V, t3_rV)
    rdm_t = contract(
        WU,
        site_env_V,
        output_labels=["phys_V", "phys_W", "phys_braV", "phys_braW"],
    )
    # → (s_v_ket, s_w_ket, s_v_bra, s_w_bra)

    rdm = rdm_t.todense()
    d_phys = rdm.shape[0]
    rdm_mat = rdm.reshape(d_phys * d_phys, d_phys * d_phys)
    rdm_mat = _normalise_rdm_for_energy(rdm_mat, "_rdm_3site_marginal_vw_row")
    return rdm_mat.reshape(d_phys, d_phys, d_phys, d_phys)


def _rdm_3site_marginal_vw_col(
    S_u: Tensor,
    S_v: Tensor,
    S_w: Tensor,
    env_u: CTMTensorEnv,
    env_v: CTMTensorEnv,
    env_w: CTMTensorEnv,
) -> jax.Array:
    """3-site marginalised 2-site RDM for the v–w bond mediated by ``T_d``.

    Vertical sibling of :func:`_rdm_3site_marginal_vw_row`: closes a 3×1
    column with ``w`` on top, ``u`` in the middle, and ``v`` on the bottom
    — matching the down-triangle b–c bond ``v.bottom ↔ w.top`` on the
    kagome Lattice.  Traces u's physical leg; returns ρ_(vw) of shape
    ``(d, d, d, d)`` in ``(s_v_ket, s_w_ket, s_v_bra, s_w_bra)`` order.
    """
    ao_W = _build_double_layer_open_tensor(S_w)
    ac_U = _build_double_layer_tensor(S_u)
    ao_V = _build_double_layer_open_tensor(S_v)

    # ----- Top row (env_w): full corners on the top -----
    C1 = env_w.C1.relabel("c1_r", "t1_l")
    UL = contract(C1, env_w.T1)  # (c1_d, u2, t1_r)
    C2 = env_w.C2.relabel("c2_l", "t1_r")
    top_row = contract(UL, C2)  # (c1_d, u2, c2_d)
    T4_T = env_w.T4.relabels({"t4_d": "c1_d"})  # (c1_d, l2, t4_u)
    T2_T = env_w.T2.relabels({"t2_u": "c2_d"})  # (c2_d, r2, t2_d)
    top_T4 = contract(top_row, T4_T)  # (u2, c2_d, l2, t4_u)
    env_top = contract(top_T4, T2_T)  # (u2, l2, t4_u, r2, t2_d)
    ao_W_lbl = ao_W.relabels({"phys": "phys_W", "phys_bra": "phys_braW"})
    site_env_W = contract(env_top, ao_W_lbl)
    # → (t4_u, t2_d, d2, phys_W, phys_braW)

    # ----- Bottom row (env_v): full corners on the bottom -----
    T4_V = env_v.T4.relabels({"t4_d": "t4_dV", "l2": "l2V", "t4_u": "t4_uV"})
    T2_V = env_v.T2.relabels({"t2_u": "t2_uV", "r2": "r2V", "t2_d": "t2_dV"})
    C4 = env_v.C4.relabel("c4_u", "t3_r")
    T3_V = env_v.T3.relabel("d2", "d2V")
    C3 = env_v.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, T3_V)  # (c4_r, d2V, t3_l)
    bot_row = contract(C4_T3, C3)  # (c4_r, d2V, c3_u)
    T4_VB = T4_V.relabel("t4_uV", "c4_r")  # (t4_dV, l2V, c4_r)
    T2_VB = T2_V.relabel("t2_dV", "c3_u")  # (t2_uV, r2V, c3_u)
    bot_T4 = contract(bot_row, T4_VB)  # (d2V, c3_u, t4_dV, l2V)
    env_bot = contract(bot_T4, T2_VB)  # (d2V, t4_dV, l2V, r2V, t2_uV)
    ao_V_lbl = ao_V.relabels(
        {
            "u2": "u2V",
            "d2": "d2V",
            "l2": "l2V",
            "r2": "r2V",
            "phys": "phys_V",
            "phys_bra": "phys_braV",
        }
    )
    site_env_V = contract(env_bot, ao_V_lbl)
    # → (t4_dV, t2_uV, u2V, phys_V, phys_braV)

    # ----- Middle row (env_u): T4, T2 only — no corners; ac closes phys -----
    # Vertical chi seam: top side ↔ env_w's t4_u/t2_d; bottom side ↔ env_v's
    # t4_dV/t2_uV.  Inner D² seam: top side ↔ env_w's d2 (label ``d2``);
    # bottom side ↔ env_v's u2V (label ``u2V``).
    T4_U = env_u.T4.relabels({"t4_d": "t4_u", "l2": "l2_U", "t4_u": "t4_dV"})
    T2_U = env_u.T2.relabels({"t2_u": "t2_d", "r2": "r2_U", "t2_d": "t2_uV"})
    ac_U_lbl = ac_U.relabels({"u2": "d2", "d2": "u2V", "l2": "l2_U", "r2": "r2_U"})
    T4_acU = contract(T4_U, ac_U_lbl)
    # contracts l2_U → (t4_u, t4_dV, d2, u2V, r2_U)
    site_env_U = contract(T4_acU, T2_U)
    # contracts r2_U → (t4_u, t4_dV, d2, u2V, t2_d, t2_uV)

    # ----- Combine: site_env_W · site_env_U · site_env_V -----
    WU = contract(site_env_W, site_env_U)
    # shared: t4_u, t2_d, d2 → (phys_W, phys_braW, t4_dV, u2V, t2_uV)
    rdm_t = contract(
        WU,
        site_env_V,
        output_labels=["phys_V", "phys_W", "phys_braV", "phys_braW"],
    )

    rdm = rdm_t.todense()
    d_phys = rdm.shape[0]
    rdm_mat = rdm.reshape(d_phys * d_phys, d_phys * d_phys)
    rdm_mat = _normalise_rdm_for_energy(rdm_mat, "_rdm_3site_marginal_vw_col")
    return rdm_mat.reshape(d_phys, d_phys, d_phys, d_phys)


def compute_energy_pess_3site_multisite(
    site_tensors: dict,
    envs: dict,
    neighbors: dict,
    bond_gates: dict[frozenset, jax.Array | Tensor],
    d: int | None = None,
) -> jax.Array:
    """Per-microscopic-site energy of the 3-site multisite kagome iPESS.

    Implements the Phase B-revised "4 NN + 2 marginalised-3-site" energy
    formula (see ``project_kagome_3site_energy_bug.md`` and Task B.4 in
    ``docs/plans/2026-05-05-multisite-kagome-pess.md``):

    * **4 NN bonds** — the 2 a-b and 2 a-c kagome bonds (1 up-triangle
      + 1 down-triangle each) map to multisite NN bonds with full-D
      virtual content.  Computed by the standard 2-site RDM primitives
      :func:`_rdm2x1_tensor_2site` (``right``/``left``) and
      :func:`_rdm1x2_tensor_2site` (``bottom``/``top``).
    * **2 marginalised-3-site bonds** — the 2 b-c kagome bonds
      (up-triangle + down-triangle) are encoded as 3-body correlations
      inside ``S_u`` because the encoding absorbs both ``T_u`` and ``T_d``
      into ``S_u``.  Computed by closing a 1×3 (resp. 3×1) block ``w-u-v``
      around the multisite CTM environments and tracing u's physical leg
      (:func:`_rdm_3site_marginal_vw_row` and
      :func:`_rdm_3site_marginal_vw_col`).

    The buggy 6-NN summation that this replaces would treat the
    dim-1-padded multisite v-w bonds as physical NN bonds; the resulting
    2-site RDMs at those bonds reduce to mean-field outer products of
    single-site density matrices and L-BFGS exploits them to drive the
    energy below the variational ground state (see
    ``project_kagome_3site_energy_bug.md`` for the empirical signature).

    Args:
        site_tensors: ``{name: Tensor}`` mapping sublattice names
            ``{"u", "v", "w"}`` to iPEPS site tensors (typically obtained
            from :func:`tenax.algorithms.pess.pess_to_kagome_3site_multisite`
            wrapped as :class:`tenax.core.tensor.DenseTensor`).
        envs:         ``{name: CTMTensorEnv}`` converged environments per
            sublattice (from :func:`tenax.algorithms._ctm_tensor_convergence._ctm_tensor_multisite`).
        neighbors:    ``{name: {"left": name, "right": name, "top": name,
                      "bottom": name}}`` — typically
                      ``tenax.core.lattice.kagome().neighbor_map``.
        bond_gates:   Per-bond gate dict from
                      :func:`kagome_3site_bond_gates`.
        d:            Physical dimension (inferred from the first site
                      tensor if ``None``).

    Returns:
        Real scalar — total kagome-XXZ bond energy divided by
        ``len(site_tensors)``.

    Raises:
        KeyError: when a kagome bond visited by the dispatcher has no
            matching entry in ``bond_gates``.
    """
    if d is None:
        first_A = next(iter(site_tensors.values()))
        phys_idx = [i for i in first_A.indices if i.label == "phys"]
        d = phys_idx[0].dim if phys_idx else first_A.indices[-1].dim

    n_sites = len(site_tensors)

    def _gate(bond_id: frozenset) -> jax.Array:
        gate = bond_gates[bond_id]
        if isinstance(gate, Tensor):
            return gate.todense().reshape(d, d, d, d)
        return jnp.asarray(gate).reshape(d, d, d, d)

    # ----- 4 NN bonds: u–v (right, bottom) + u–w (right, bottom from w) -----
    # Iterating ("u", right), ("u", bottom), ("w", right), ("w", bottom)
    # visits exactly the 4 a-b/a-c kagome bonds once each.  The v-iterations
    # would land on the dim-1 v-w multisite bonds and are deliberately
    # skipped — their physical content is recovered via the marginalised
    # 3-site RDMs below.
    nn_visits = (
        ("u", "right"),
        ("u", "bottom"),
        ("w", "right"),
        ("w", "bottom"),
    )
    total_energy = jnp.array(0.0)
    for name, direction in nn_visits:
        nb_name = neighbors[name][direction]
        reverse_dir = "left" if direction == "right" else "top"
        bond_id = frozenset([(name, direction), (nb_name, reverse_dir)])
        H = _gate(bond_id)
        A = site_tensors[name]
        B = site_tensors[nb_name]
        env_A = envs[name]
        env_B = envs[nb_name]
        if direction == "right":
            rdm = _rdm2x1_tensor_2site(A, B, env_A, env_B)
        else:
            rdm = _rdm1x2_tensor_2site(A, B, env_A, env_B)
        total_energy = total_energy + jnp.einsum("ijkl,ijkl->", rdm, H)

    # ----- 2 marginalised-3-site bonds: v–w via T_u (h-row) and T_d (v-col) -----
    S_u = site_tensors["u"]
    S_v = site_tensors["v"]
    S_w = site_tensors["w"]
    env_u = envs["u"]
    env_v = envs["v"]
    env_w = envs["w"]

    H_h = _gate(frozenset({("v", "right"), ("w", "left")}))
    rdm_vw_h = _rdm_3site_marginal_vw_row(S_u, S_v, S_w, env_u, env_v, env_w)
    total_energy = total_energy + jnp.einsum("ijkl,ijkl->", rdm_vw_h, H_h)

    H_v = _gate(frozenset({("v", "bottom"), ("w", "top")}))
    rdm_vw_v = _rdm_3site_marginal_vw_col(S_u, S_v, S_w, env_u, env_v, env_w)
    total_energy = total_energy + jnp.einsum("ijkl,ijkl->", rdm_vw_v, H_v)

    return total_energy.real / n_sites
