"""Test oracle helpers for issue #463 split-CTM double-layer projector.

Proven by spike: fused_env_to_split converts a trusted fused CTM env to a
split env (lossless at chi_I = chi*D); used as the correctness oracle.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
    max_abs_normalize,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor
from tenax.linalg import svd as tensor_svd


def make_site(D, d, seed):
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(d, dtype=np.int32)
    idx = [
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, zd.copy(), FlowDirection.IN, label="phys"),
    ]
    return DenseTensor(data, idx)


def heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def evolve_physical_site(A, gate, D, n_steps=40, dt=0.05):
    """Simple-update evolve a 1-site ansatz and return the PHYSICAL tensor.

    The name is a contract: the returned tensor carries ``sqrt(lambda)`` on all
    four legs, so a CTM contracting it sees the simple-update state rather than
    a bare Vidal ``Gamma`` with every bond weight dropped.

    Two things this used to get wrong, both of them #667:

    * It ran only the (A.r<->B.l) and (A.d<->B.u) bonds, so half the lattice
      bonds never acquired Schmidt weight.  The sweep now cycles all four.
    * It discarded the gate's *right*-site output (``A, _, lam_h = ...``), so
      ``A`` only ever received the left-site update.  Both sites are now kept.

    Note this helper currently has **no callers**, so nothing exercises it; it
    is fixed rather than deleted to keep the module's stated contract honest.
    """
    from tenax.algorithms.ipeps_simple_update import (
        _make_trotter_gate_tensor,
        _simple_update_checkerboard_sweep,
        _to_physical_pair,
    )

    tg = _make_trotter_gate_tensor(gate, dt, site_tensor=A)
    A, _B, lam_h, lam_v = _simple_update_checkerboard_sweep(A, A, tg, D, n_steps)
    A_phys, _ = _to_physical_pair(A, _B, lam_h, lam_v)
    A, _ = max_abs_normalize(A_phys)
    return A


def _unfuse_d2_leg(T, fused_label, ket_label, bra_label, D):
    labels = T.labels()
    ax = labels.index(fused_label)
    arr = T.todense()
    new_shape = arr.shape[:ax] + (D, D) + arr.shape[ax + 1 :]
    arr = arr.reshape(new_shape)
    sym = U1Symmetry()
    flow = T.indices[ax].flow
    new_indices = list(T.indices[:ax])
    new_indices.append(
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), flow, label=ket_label
        )
    )
    new_indices.append(
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), flow, label=bra_label
        )
    )
    new_indices.extend(T.indices[ax + 1 :])
    return DenseTensor(arr, new_indices)


def _svd_split(T, left_labels, right_labels, chi_I, ket_relabels, bra_relabels):
    U_t, s, Vh_t, _ = tensor_svd(
        T,
        left_labels=left_labels,
        right_labels=right_labels,
        new_bond_label="_I",
        max_singular_values=chi_I,
    )
    T_ket, T_bra = absorb_sqrt_singular_values(U_t, s, Vh_t, "_I")
    return T_ket.relabels(ket_relabels), T_bra.relabels(bra_relabels)


def fused_env_to_split(fused_env, D, chi_I):
    T1u = _unfuse_d2_leg(fused_env.T1, "u2", "u_ket", "u_bra", D)
    T1_ket, T1_bra = _svd_split(
        T1u,
        ["t1_l", "u_ket"],
        ["u_bra", "t1_r"],
        chi_I,
        {"t1_l": "t1k_l", "_I": "t1k_I"},
        {"_I": "t1b_I", "t1_r": "t1b_r"},
    )
    T1_ket = T1_ket.transpose(
        tuple(T1_ket.labels().index(lbl) for lbl in ["t1k_l", "u_ket", "t1k_I"])
    )
    T1_bra = T1_bra.transpose(
        tuple(T1_bra.labels().index(lbl) for lbl in ["t1b_I", "u_bra", "t1b_r"])
    )
    T2u = _unfuse_d2_leg(fused_env.T2, "r2", "r_ket", "r_bra", D)
    T2_ket, T2_bra = _svd_split(
        T2u,
        ["t2_u", "r_ket"],
        ["r_bra", "t2_d"],
        chi_I,
        {"t2_u": "t2k_u", "_I": "t2k_I"},
        {"_I": "t2b_I", "t2_d": "t2b_d"},
    )
    T2_ket = T2_ket.transpose(
        tuple(T2_ket.labels().index(lbl) for lbl in ["t2k_u", "r_ket", "t2k_I"])
    )
    T2_bra = T2_bra.transpose(
        tuple(T2_bra.labels().index(lbl) for lbl in ["t2b_I", "r_bra", "t2b_d"])
    )
    T3u = _unfuse_d2_leg(fused_env.T3, "d2", "d_ket", "d_bra", D)
    T3_ket, T3_bra = _svd_split(
        T3u,
        ["t3_r", "d_ket"],
        ["d_bra", "t3_l"],
        chi_I,
        {"t3_r": "t3k_r", "_I": "t3k_I"},
        {"_I": "t3b_I", "t3_l": "t3b_l"},
    )
    T3_ket = T3_ket.transpose(
        tuple(T3_ket.labels().index(lbl) for lbl in ["t3k_r", "d_ket", "t3k_I"])
    )
    T3_bra = T3_bra.transpose(
        tuple(T3_bra.labels().index(lbl) for lbl in ["t3b_I", "d_bra", "t3b_l"])
    )
    T4u = _unfuse_d2_leg(fused_env.T4, "l2", "l_ket", "l_bra", D)
    T4_ket, T4_bra = _svd_split(
        T4u,
        ["t4_d", "l_ket"],
        ["l_bra", "t4_u"],
        chi_I,
        {"t4_d": "t4k_d", "_I": "t4k_I"},
        {"_I": "t4b_I", "t4_u": "t4b_u"},
    )
    T4_ket = T4_ket.transpose(
        tuple(T4_ket.labels().index(lbl) for lbl in ["t4k_d", "l_ket", "t4k_I"])
    )
    T4_bra = T4_bra.transpose(
        tuple(T4_bra.labels().index(lbl) for lbl in ["t4b_I", "l_bra", "t4b_u"])
    )
    return SplitCTMTensorEnv(
        C1=fused_env.C1,
        C2=fused_env.C2,
        C3=fused_env.C3,
        C4=fused_env.C4,
        T1_ket=T1_ket,
        T1_bra=T1_bra,
        T2_ket=T2_ket,
        T2_bra=T2_bra,
        T3_ket=T3_ket,
        T3_bra=T3_bra,
        T4_ket=T4_ket,
        T4_bra=T4_bra,
    )


def norm_spectrum(C):
    s = np.linalg.svd(np.asarray(C.todense()), compute_uv=False)
    s = np.sort(s)[::-1]
    return s / (s[0] + 1e-300)


def eff_rank(s, tol=1e-6):
    return int(np.sum(s > tol))
