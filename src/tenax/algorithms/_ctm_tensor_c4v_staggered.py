"""The 2-sublattice C4v CTM sweep for charged C4v input (#762, #763).

A bosonic C4v ``SymmetricTensor`` ``A`` whose virtual charges are not all
self-dual describes the 2-sublattice state ``(A, negate(A))`` (see
``_ctm_c4v_charges``).  This module runs its CTM with one corner and one
edge, as ``_c4v_sweep`` does for self-dual input, and keeps the symmetry and
the lattice mirror exact.

The state, in the C1/T1 orientation of the #905 flow table:

- ``C``: the corner of an A-site, legs ``(c_a IN, c_b OUT)``.
- ``T``: the edge above an A-site (its first site is a B-site), legs
  ``(t_l IN, D2 OUT, t_r OUT)``; ``D2`` pairs with the ``u2`` face of ``a``.
- ``a``: the double layer of ``A`` (``_build_double_layer_tensor``).

One step enlarges the corner as ``_c4v_sweep`` does, truncates it with one
block projector (an SVD per mirror pair of blocks, an eigh per self-partner
block, one ranking that never splits a pair), and sets ``C_new = P^T Q P``
and ``T_new = negate(P^T (T a) P)``, which is again the edge above an
A-site.  Design: ``docs/superpowers/specs/2026-11-22-c4v-charged-u1-design.md``.

Spec 2 (``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``)
differentiates the sweep at a frozen layout.  So the projector is split
into the eager ranking (``_c4v_staggered_rank``, ``_c4v_staggered_layout``)
and the assembly of P from its columns (``_c4v_staggered_assemble``), which
the traced builder ``_c4v_staggered_projectors_at`` shares.  The eager
outputs do not change.
"""

from __future__ import annotations

__all__ = [
    "_c4v_staggered_assemble",
    "_c4v_staggered_blocks",
    "_c4v_staggered_enlarged_corner",
    "_c4v_staggered_initial_state",
    "_c4v_staggered_layout",
    "_c4v_staggered_loop",
    "_c4v_staggered_project",
    "_c4v_staggered_projectors",
    "_c4v_staggered_projectors_at",
    "_c4v_staggered_rank",
    "_c4v_staggered_sweep",
    "_c4v_staggered_to_full_env",
    "_check_c4v_staggered_invariants",
    "_ctm_tensor_c4v_staggered",
]

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ad_primitives import regularized_eigh, truncated_svd_ad
from tenax.algorithms._ctm_c4v_charges import (
    _dual,
    _dual_charge,
    _negate_charges,
    _pairs,
)
from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo
from tenax.algorithms._ctm_env_dense import _in_documented_leg_order
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_sv_diff,
    _forced_corner_rank,
    _max_virtual_bond_dim,
)
from tenax.algorithms._ctm_tensor_init import (
    IN,
    OUT,
    CTMTensorEnv,
    _build_double_layer_tensor,
    _fuse_pair_by_label,
)
from tenax.algorithms._ctm_tensor_moves import _flip_leg_flow
from tenax.contraction.contractor import contract
from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor, Tensor
from tenax.linalg import _dense_svd

# Values below this fraction of the largest are dropped: the cutoff of the SVD
# projector in ``_ctm_projector.py``.
_REL_CUTOFF = 1e-14


def _c4v_staggered_projectors(
    Qf: SymmetricTensor, chi: int
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The block projector of the 2-sublattice sweep: ``(P_left, P_right)``.

    ``Qf`` is the enlarged corner with the legs ``fused`` (IN, bottom face)
    and ``col`` (OUT, right face).  Both faces carry the same flow-weighted
    charges, so ``col`` holds the duals of the ``fused`` charges, every block
    has the key (q, q), and block q joins the fused states of charge q with
    those of charge ``sym.dual(q)``.  The lattice mirror maps block q to
    block ``sym.dual(q)``, transposed, with the same singular values.

    - A mirror pair (q, dual(q)), q != dual(q): one SVD, Q_q = U S V^T.  The
      projector takes the kept columns of U in sector q and the same columns
      of V in sector dual(q).
    - A self-partner block, q == dual(q) (0 for U(1); also n/2 for Z_n with
      n even): eigh of the symmetrised block, ranked by |lambda|.

    All values are ranked together; a pair value counts once in each of its
    two sectors.  The chi largest are kept, except that a cut between the
    two copies of one pair value drops both (the dimension is then chi - 1).
    Values below ``_REL_CUTOFF`` of the largest are dropped.  This is the
    ranking of the dense eigh of ``_c4v_sweep``: the dense Q has eigenvalues
    +s and -s for each pair value s.

    ``P_left`` is ``(fused OUT, chi_new IN)`` and contracts with the fused
    face; ``P_right`` is ``(col IN, chi_new_r OUT)`` and contracts with the
    col face.  They hold the same arrays, one dense P on both faces; only the
    keys and flows differ.

    Raises:
        RuntimeError: the two faces of ``Qf`` do not carry the same
            flow-weighted charges (not a 2-sublattice corner).
        ValueError: nothing survives the cutoff (a zero corner), or chi = 1
            and the largest value belongs to a mirror pair.
    """
    sym, fused, col, mats = _c4v_staggered_blocks(Qf)
    decomp, keep = _c4v_staggered_rank(sym, mats, chi)
    cols = {}
    for q in sorted(keep):
        kind, U, V = decomp[q]
        ns = sorted(keep[q])
        cols[q] = U[:, ns]
        if kind == "pair":
            cols[_dual_charge(sym, q)] = V[:, ns]
    return _c4v_staggered_assemble(sym, fused, col, cols)


def _c4v_staggered_blocks(Qf: SymmetricTensor):
    """The faces of the enlarged corner and its blocks: ``(sym, fused, col, mats)``.

    ``mats[q]`` is block (q, q): the fused states of charge q are its rows,
    the col states of charge q (the fused states of charge ``dual(q)``) its
    columns.  Metadata decides everything here, so it is safe under tracing.

    Raises:
        RuntimeError: the two faces of ``Qf`` do not carry the same
            flow-weighted charges (not a 2-sublattice corner).
    """
    labels = Qf.labels()
    fpos, cpos = labels.index("fused"), labels.index("col")
    fused, col = Qf.indices[fpos], Qf.indices[cpos]
    sym = fused.symmetry
    if not np.array_equal(col.charges, _dual(sym, fused.charges)):
        raise RuntimeError(
            "charged C4v projector: the two faces of the enlarged corner do not "
            "carry the same flow-weighted charges, so the env is not a "
            "2-sublattice env"
        )
    # Block (q, q): rows are the fused states of charge q, columns the col
    # states of charge q, which are the fused states of charge dual(q).
    mats = {
        int(key[fpos]): (block if fpos < cpos else block.T)
        for key, block in Qf.blocks.items()
    }
    return sym, fused, col, mats


def _c4v_staggered_rank(sym, mats, chi: int):
    """Decompose every block and rank all values together: ``(decomp, keep)``.

    Eager: the ranking reads the values on the host.  ``decomp[q]`` is
    ``("self", W, None)`` for a self-partner block and ``("pair", U, V)``
    for a mirror pair, keyed by the first charge of the pair in sorted
    order.  ``keep[q]`` lists the kept columns.  The ranking is the one
    described in ``_c4v_staggered_projectors``.

    Raises:
        ValueError: nothing survives the cutoff (a zero corner), or chi = 1
            and the largest value belongs to a mirror pair.
    """
    decomp, cands, done = {}, [], set()
    for q in sorted(mats):
        if q in done:
            continue
        qd = _dual_charge(sym, q)
        M = mats[q]
        if qd == q:
            # Self-partner block: symmetric by the mirror; rank by |lambda|.
            lam, W = jnp.linalg.eigh(0.5 * (M + M.T))
            decomp[q] = ("self", W, None)
            for n, v in enumerate(np.abs(np.asarray(lam))):
                cands.append((float(v), q, n, 0))
            done.add(q)
        else:
            # Mirror pair (q, dual(q)): one SVD.  Each value has two copies,
            # one state in sector q and one in sector dual(q).
            U, S, Vh = _dense_svd(M, full_matrices=False)
            decomp[q] = ("pair", U, Vh.T)
            for n, v in enumerate(np.asarray(S)):
                cands += [(float(v), q, n, 0), (float(v), q, n, 1)]
            done.update((q, qd))

    vmax = max((c[0] for c in cands), default=0.0)
    alive = sorted(
        (c for c in cands if c[0] > 0.0 and c[0] >= _REL_CUTOFF * vmax),
        key=lambda c: (-c[0], c[1], c[2], c[3]),
    )
    if not alive:
        raise ValueError("charged C4v projector: the enlarged corner is zero")

    copies: dict[tuple[int, int], int] = {}
    for _v, q, n, _copy in alive[:chi]:
        copies[(q, n)] = copies.get((q, n), 0) + 1
    keep: dict[int, list[int]] = {}
    for (q, n), count in copies.items():
        # A cut between the two copies of a pair value would keep one state
        # of a mirror pair without its partner: drop both.
        if decomp[q][0] == "pair" and count == 1:
            continue
        keep.setdefault(q, []).append(n)
    if not keep:
        raise ValueError(
            f"ctm_tensor_c4v: chi={chi} keeps no state of the charged C4v "
            "corner. Its largest value belongs to a mirror pair (q, dual(q)), "
            "which needs two states. Use chi >= 2."
        )
    return decomp, keep


def _c4v_staggered_assemble(sym, fused, col, cols):
    """``(P_left, P_right)`` from one column block per kept charge.

    One array per sector on both faces.  On the col face, the states of
    fused charge x have col charge dual(x), and so does their chi_new_r leg.
    The chi index lists the kept charges in sorted order, each with
    ``cols[x].shape[1]`` states.
    """
    left, right, chi_charges = {}, {}, []
    for x in sorted(cols):
        xd = _dual_charge(sym, x)
        left[(x, x)] = cols[x]
        right[(xd, xd)] = cols[x]
        chi_charges += [x] * cols[x].shape[1]
    chi_charges = np.asarray(chi_charges, dtype=np.int32)
    P_left = SymmetricTensor(
        left,
        (
            fused.flip_flow(),
            TensorIndex.from_charges(sym, chi_charges, IN, label="chi_new"),
        ),
    )
    P_right = SymmetricTensor(
        right,
        (
            col.flip_flow(),
            TensorIndex.from_charges(
                sym, _dual(sym, chi_charges), OUT, label="chi_new_r"
            ),
        ),
    )
    return P_left, P_right


def _c4v_staggered_layout(Qf: SymmetricTensor, chi: int) -> BondLayout:
    """The kept states per chi charge under the free ranking (eager).

    The layout that ``_c4v_staggered_projectors(Qf, chi)`` builds: a mirror
    pair keeps the same count in q and in ``dual(q)``.  Spec 2 freezes it
    for an AD run (``_c4v_staggered_projectors_at``) and re-runs it at each
    step to check the frozen layout.
    """
    sym, _fused, _col, mats = _c4v_staggered_blocks(Qf)
    decomp, keep = _c4v_staggered_rank(sym, mats, chi)
    dims = {}
    for q, ns in keep.items():
        dims[q] = len(ns)
        if decomp[q][0] == "pair":
            dims[_dual_charge(sym, q)] = len(ns)
    return BondLayout.from_dims(dims)


def _c4v_staggered_projectors_at(
    Qf: SymmetricTensor, layout: BondLayout
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The block projector at a frozen layout, traceable (spec 2, section 2).

    The keys, flows and column order of ``_c4v_staggered_projectors`` for
    the layout that the free ranking chose, with differentiable
    decompositions:

    - A mirror pair (q, dual(q)), q the first of the two in sorted order:
      ``truncated_svd_ad(M_q, k_q)``.  U goes to sector q, V to sector
      dual(q).  Both copies of a pair value come from one SVD, so no
      backward term divides by their zero gap (#1178).
    - A self-partner block: ``regularized_eigh`` of 0.5 (M + M^T), and the
      k_q largest |lambda| in ascending index order (a stable argsort with
      a static k: the eager tie-break).
    - A charge with k_q = 0 gets no decomposition.

    The column signs may differ from the eager projector's
    (``truncated_svd_ad`` fixes them with ``_fix_svd_signs``); the AD sweep
    aligns them.

    Raises:
        RuntimeError: as ``_c4v_staggered_blocks``.
        ValueError: the layout keeps a charge that has no block, or more
            states than its block has.
    """
    sym, fused, col, mats = _c4v_staggered_blocks(Qf)
    for q, k in layout.dims:
        if q not in mats or k > min(mats[q].shape):
            shape = tuple(mats[q].shape) if q in mats else None
            raise ValueError(
                f"charged C4v projector: the layout keeps {k} states of charge "
                f"{q}, but the enlarged corner's block of that charge has shape "
                f"{shape}"
            )
    cols = {}
    for q in sorted(mats):
        k = layout.dim_of(q)
        if k == 0 or q in cols:
            continue
        qd = _dual_charge(sym, q)
        M = mats[q]
        if qd == q:
            lam, W = regularized_eigh(0.5 * (M + M.T))
            cols[q] = W[:, jnp.sort(jnp.argsort(-jnp.abs(lam))[:k])]
        else:
            U, _s, Vh = truncated_svd_ad(M, k)
            cols[q] = U
            cols[qd] = Vh.T
    return _c4v_staggered_assemble(sym, fused, col, cols)


def _leg(T: Tensor, label: str) -> TensorIndex:
    """The index of ``T`` with this label."""
    return T.indices[T.labels().index(label)]


def _c4v_staggered_enlarged_corner(C: Tensor, T: Tensor, a: Tensor) -> Tensor:
    """Q = C · T_top · T_left · a, with its two open faces fused.

    The contraction of step 1 of ``_c4v_sweep``, which stays unchanged:
    C.c_b joins the top edge's t_l and C.c_a the left edge's t_r; the top
    edge's D2 is a.u2, the left edge's is a.l2.  The bottom face (the open
    chi leg of T_left, a.d2) is fused into ``fused`` (IN); the right face
    (the open chi leg of T_top, a.r2) into ``col`` (OUT).
    """
    Q = contract(C.relabel("c_b", "t_l"), T)  # (c_a, D2, t_r)
    Q = Q.relabels({"D2": "u2", "t_r": "o_r"})
    T_left = T.relabels({"t_r": "c_a", "D2": "l2", "t_l": "o_d"})
    Q = contract(Q, T_left)  # (u2, o_r, o_d, l2)
    Q = contract(Q, a)  # (o_r, o_d, d2, r2)
    Qf = _fuse_pair_by_label(Q, "o_d", "d2", "fused", IN)
    return _fuse_pair_by_label(Qf, "o_r", "r2", "col", OUT)  # (fused, col)


def _c4v_staggered_initial_state(
    a: SymmetricTensor,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The trace boundary: C = [1], and T the ket-bra identity on D2.

    C is 1 x 1 with the identity charge.  T has chi legs of dimension 1 with
    the identity charge and the full D2 index that pairs with ``a.u2``; its
    data is the ket-bra identity delta_kk', whose nonzero entries all carry
    the identity charge.  The state satisfies I1-I3.
    """
    u2 = _leg(a, "u2")
    sym = u2.symmetry
    e = np.asarray([sym.identity()], dtype=np.int32)
    C = SymmetricTensor(
        {(int(e[0]), int(e[0])): jnp.ones((1, 1), dtype=a.dtype)},
        (
            TensorIndex.from_charges(sym, e, IN, label="c_a"),
            TensorIndex.from_charges(sym, e, OUT, label="c_b"),
        ),
    )
    D = int(round(np.sqrt(u2.dim)))
    data = np.zeros((1, D * D, 1))
    data[0, np.arange(D) * (D + 1), 0] = 1.0  # delta_kk'; u2 is fused ket-major
    T = SymmetricTensor.from_dense(
        jnp.asarray(data, dtype=a.dtype),
        (
            TensorIndex.from_charges(sym, e, IN, label="t_l"),
            u2.flip_flow().relabel("D2"),
            TensorIndex.from_charges(sym, e, OUT, label="t_r"),
        ),
    )
    return C, T


def _check_c4v_staggered_invariants(
    C: SymmetricTensor,
    T: SymmetricTensor,
    a: SymmetricTensor,
    a_B: SymmetricTensor,
) -> None:
    """Check the charge invariants of the 2-sublattice state.

    - I1: the C-T chi bonds and the T.D2-a.u2 bond pair opposite flows with
      identical charges.
    - I2: the two legs of C carry the same flow-weighted charges at each
      position, and so do the two chi legs of T; each chi leg holds as many
      states of charge q as of ``sym.dual(q)``.
    - I3: ``negate(T)``, the edge above a B-site, pairs with ``a_B``, the
      double layer of the B-site tensor.

    Index metadata only.

    Raises:
        RuntimeError: the message starts with the invariant that fails.
    """
    sym = _leg(C, "c_a").symmetry

    def s(X, label):
        idx = _leg(X, label)
        return sym.flow_charge(idx.flow, idx.charges)

    if not (
        _pairs(_leg(C, "c_b"), _leg(T, "t_l"))
        and _pairs(_leg(C, "c_a"), _leg(T, "t_r"))
    ):
        raise RuntimeError("I1: a C-T chi bond does not pair")
    if not _pairs(_leg(T, "D2"), _leg(a, "u2")):
        raise RuntimeError("I1: T.D2 does not pair with a.u2")
    if not np.array_equal(s(C, "c_a"), s(C, "c_b")):
        raise RuntimeError("I2: the legs of C carry different flow-weighted charges")
    if not np.array_equal(s(T, "t_l"), s(T, "t_r")):
        raise RuntimeError(
            "I2: the chi legs of T carry different flow-weighted charges"
        )
    for X, label in ((C, "c_a"), (T, "t_l")):
        charges = s(X, label)
        for q in set(charges.tolist()):
            qd = _dual_charge(sym, q)
            if np.sum(charges == q) != np.sum(charges == qd):
                raise RuntimeError(
                    f"I2: leg {label} holds {int(np.sum(charges == q))} states "
                    f"of charge {q} but {int(np.sum(charges == qd))} of its "
                    f"mirror partner {qd}"
                )
    if not _pairs(_leg(_negate_charges(T), "D2"), _leg(a_B, "u2")):
        raise RuntimeError("I3: negate(T).D2 does not pair with a_B.u2")


def _c4v_staggered_project(
    Qf: SymmetricTensor,
    T: SymmetricTensor,
    a: SymmetricTensor,
    P_left: SymmetricTensor,
    P_right: SymmetricTensor,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """``(C_new, T_new)`` from a projector, before the normalisation.

    C_new = P^T Q P is the enlarged corner projected on both faces.  T·a is
    the edge above a B-site; projected on both faces and negated, it is the
    edge above an A-site again.  Traceable: the AD sweep calls it with the
    traced projector.
    """
    C_new = contract(contract(P_left, Qf), P_right)
    C_new = C_new.relabels({"chi_new": "c_a", "chi_new_r": "c_b"})

    T_with_a = contract(T.relabel("D2", "u2"), a)  # (t_l, t_r, d2, l2, r2)
    Tg = _fuse_pair_by_label(T_with_a, "t_l", "l2", "fl", IN)
    Tg = _fuse_pair_by_label(Tg, "t_r", "r2", "fr", OUT)  # (fl, d2, fr)
    T_new = contract(P_left.relabel("fused", "fl"), Tg)  # (chi_new, d2, fr)
    T_new = contract(T_new, P_right.relabel("col", "fr"))  # (chi_new, d2, chi_new_r)
    T_new = _negate_charges(T_new).relabels(
        {"chi_new": "t_l", "d2": "D2", "chi_new_r": "t_r"}
    )
    return C_new, T_new


def _c4v_staggered_sweep(
    C: SymmetricTensor, T: SymmetricTensor, a: SymmetricTensor, chi: int
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """One step of the 2-sublattice C4v sweep; returns ``(C_new, T_new)``.

    ``_c4v_staggered_project`` with the eager projector; both outputs are
    normalised by ``max_abs``, as in ``_c4v_sweep``.
    """
    Qf = _c4v_staggered_enlarged_corner(C, T, a)
    P_left, P_right = _c4v_staggered_projectors(Qf, chi)
    C_new, T_new = _c4v_staggered_project(Qf, T, a, P_left, P_right)
    C_new = C_new * (1.0 / (C_new.max_abs() + 1e-15))
    T_new = T_new * (1.0 / (T_new.max_abs() + 1e-15))
    return C_new, T_new


def _c4v_staggered_to_full_env(C: SymmetricTensor, T: SymmetricTensor) -> CTMTensorEnv:
    """The eight env tensors around an A-site, with the #905 flows.

    Relabels and single-leg duals only.  A dual keeps the flow-weighted
    charges, and I2 makes the two legs of C, and the two chi legs of T, carry
    the same ones, so every bond pairs once the flows are set:

    - C1..C4 are C, with ``c_a`` on the IN leg and ``c_b`` on the OUT leg.
    - T1 and T4 are T; T2 and T3 are T with the D2 leg dualled (``r2`` and
      ``d2`` are IN in the #905 table).
    - The chi joins follow the 2-site RDM code (``_rdm2x1_tensor_2site``,
      ``_rdm1x2_tensor_2site``): C1.c1_d joins T4.t4_d.

    Unlike ``_c4v_to_full_env``, every chi bond pairs opposite flows, and so
    do the bonds from env to ``negate(env)`` in the 2-site energy.  The
    fields come out in the documented leg order.
    """
    env = CTMTensorEnv(
        C1=C.relabels({"c_a": "c1_d", "c_b": "c1_r"}),
        C2=C.relabels({"c_a": "c2_l", "c_b": "c2_d"}),
        C3=C.relabels({"c_a": "c3_u", "c_b": "c3_l"}),
        C4=C.relabels({"c_a": "c4_u", "c_b": "c4_r"}),
        T1=T.relabels({"t_l": "t1_l", "D2": "u2", "t_r": "t1_r"}),
        T2=_flip_leg_flow(T.relabels({"t_l": "t2_u", "D2": "r2", "t_r": "t2_d"}), "r2"),
        T3=_flip_leg_flow(T.relabels({"t_l": "t3_l", "D2": "d2", "t_r": "t3_r"}), "d2"),
        T4=T.relabels({"t_l": "t4_u", "D2": "l2", "t_r": "t4_d"}),
    )
    return _in_documented_leg_order(env)


def _c4v_staggered_loop(
    A: SymmetricTensor, chi: int, max_iter: int, conv_tol: float
) -> tuple[SymmetricTensor, SymmetricTensor, SymmetricTensor, CTMConvergeInfo]:
    """The loop of ``ctm_tensor_c4v``: ``(C, T, a, info)``.

    Start from the trace boundary, check I1-I3 on it, and sweep until the
    corner singular values change by less than ``conv_tol``
    (``_ctm_sv_diff`` with the forced-rank floor of #903), or for
    ``max_iter`` sweeps.  ``info.converged`` says which; ``info.sv_diff`` is
    the last sv change (inf before the second sweep).
    """
    a = _build_double_layer_tensor(A)
    a_B = _build_double_layer_tensor(_negate_charges(A))
    C, T = _c4v_staggered_initial_state(a)
    _check_c4v_staggered_invariants(C, T, a, a_B)
    max_rank = _forced_corner_rank(_max_virtual_bond_dim(a))
    prev_sv, diff, converged, iterations = None, float("inf"), False, 0
    for iterations in range(1, max_iter + 1):
        C, T = _c4v_staggered_sweep(C, T, a, chi)
        current_sv = _dense_svd(C.todense(), compute_uv=False)  # chi x chi
        if prev_sv is not None:
            diff = float(_ctm_sv_diff(current_sv, prev_sv, max_rank=max_rank))
            if diff < conv_tol:
                converged = True
                break
        prev_sv = current_sv
    info = CTMConvergeInfo(
        converged=converged,
        iterations=iterations,
        sv_diff=diff,
        best_iteration=iterations,
        step_multiplier=float("nan"),
    )
    return C, T, a, info


def _ctm_tensor_c4v_staggered(
    A: SymmetricTensor, chi: int, max_iter: int, conv_tol: float
) -> CTMTensorEnv:
    """Run the 2-sublattice C4v CTM; return the env of an A-site.

    ``_c4v_staggered_loop``, expanded to the eight env tensors.
    ``ctm_tensor_c4v`` validates ``A`` first.
    """
    C, T, _a, _info = _c4v_staggered_loop(A, chi, max_iter, conv_tol)
    return _c4v_staggered_to_full_env(C, T)
