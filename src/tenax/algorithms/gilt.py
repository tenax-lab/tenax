"""Graph-Independent Local Truncation (GILT) and Gilt-TNR.

GILT removes short-range entanglement (most importantly corner-double-line
loops) from a 2D tensor network by inserting a low-rank matrix ``Q ~ identity``
on a bond, computed from the spectrum of that bond's plaquette environment.
Combined with TRG this gives Gilt-TNR: unlike plain TRG, the coarse-graining
then flows toward proper fixed points instead of accumulating CDL junk.

References:
    - M. Hauru, C. Delcamp, S. Mizera, PRB 97, 045111 (2018) — GILT and
      Gilt-TNR (the algorithm implemented here).
    - N. Ebel, T. Kennedy, S. Rychkov, PRX 15, 031047 (2025), App. C —
      the iterative cascade recipe and lap structure this module follows.

Conventions:
    - ``gilt_eps`` is measured against the environment singular-value
      spectrum normalized by its SUM (the convention of Hauru et al.'s
      reference code and Ebel et al.; some papers, e.g. Guo & Wei,
      normalize by the largest value instead — the two differ by roughly
      a factor of the effective spectrum count).
    - The bond matrix ``Q`` is split by SVD and singular values below
      ``split_factor * gilt_eps`` are dropped individually (Ebel et al.'s
      printed rule). Any two such truncation rules that preserve the
      network to O(eps) differ only at O(eps^2) per round.
    - Site tensors use the library leg convention ``(up, down, left,
      right)`` with flows (IN, OUT, IN, OUT), matching ``trg.py``.

The plaquette environment gram is built with label-based ``contract`` calls
(block-sparse for ``SymmetricTensor``), while the cascade itself runs on the
dense gram — a ``(chi^2, chi^2)`` object, small compared to the chi^6
contraction that produces it. The composed ``Q`` is charge-conserving
whenever the input tensor is, and is re-wrapped into the input's tensor type
before absorption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import (
    absorb_sqrt_singular_values,
    max_abs_normalize,
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

_LEG_LABELS = ("up", "down", "left", "right")

# Plaquette geometry: corners (TL, TR, BL, BR) = (B1, B2, B2, B1) on the
# checkerboard. Each bond names (corner_i, leg_i, corner_j, leg_j) — the two
# ends of the cut bond — and the corner chain order used to build the
# environment gram (start at corner_i, go around the loop, end at corner_j).
# Corner ids: 0=TL, 1=TR, 2=BL, 3=BR.
_BONDS = {
    "top": {"ends": (0, "right", 1, "left"), "chain": (0, 2, 3, 1)},
    "bottom": {"ends": (2, "right", 3, "left"), "chain": (2, 0, 1, 3)},
    "right": {"ends": (1, "down", 3, "up"), "chain": (1, 0, 2, 3)},
    "left": {"ends": (0, "down", 2, "up"), "chain": (0, 1, 3, 2)},
}

# The lap visits bonds in this order (Ebel et al., footnote 68).
_BOND_ORDER = ("top", "bottom", "right", "left")

# Internal plaquette bond labels: each corner's two inward legs, relabeled so
# that shared labels contract. corner id -> {leg: plaquette-bond name}.
_PLAQ_WIRING = {
    0: {"right": "tb", "down": "lb"},  # TL
    1: {"left": "tb", "down": "rb"},  # TR
    2: {"right": "bb", "up": "lb"},  # BL
    3: {"left": "bb", "up": "rb"},  # BR
}


@dataclass
class GiltConfig:
    """Configuration for the GILT plaquette filter.

    Attributes:
        gilt_eps:       Filtering strength epsilon, measured against the
                        SUM-normalized environment spectrum (see module
                        docstring). 0.0 disables filtering exactly.
        convergence_eps:    Cascade stop criterion: done when all retained
                        singular values of the candidate Q sit within this
                        distance of 1 (Ebel et al. Eq. C2).
        max_laps:       Maximum number of laps over the four plaquette
                        bonds (Ebel et al. Eq. C4); a lap in which every
                        bond needs at most one refinement ends the stage.
        max_cascade_iterations: Hard cap on cascade iterations per bond.
        split_factor:   Q-split truncation: singular values of Q below
                        ``split_factor * gilt_eps`` are dropped.
    """

    gilt_eps: float = 1e-6
    convergence_eps: float = 1e-2
    max_laps: int = 4
    max_cascade_iterations: int = 200
    split_factor: float = 1e-3


@dataclass
class GiltTNRConfig:
    """Configuration for Gilt-TNR coarse-graining.

    Attributes:
        max_bond_dim:   Maximum bond dimension chi after each TRG split.
        num_steps:      Number of coarse-graining iterations (each halves
                        the number of lattice sites, like ``trg``).
        svd_trunc_err:  Optional maximum truncation error per TRG SVD.
        gilt:           Parameters of the GILT filter applied before each
                        TRG step.
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    svd_trunc_err: float | None = None
    gilt: GiltConfig = field(default_factory=GiltConfig)


def _index_of(T: Tensor, label: str):
    """Return the TensorIndex of the leg with the given label."""
    for idx in T.indices:
        if idx.label == label:
            return idx
    raise ValueError(f"tensor has no leg labeled {label!r}: {T.labels()}")


def _double_layer(T: Tensor, relabel_map: dict[str, str]) -> Tensor:
    """Contract T with its bra copy over the legs NOT in relabel_map.

    The legs named in ``relabel_map`` stay open: the ket copy's leg gets the
    mapped name, the bra copy's gets the mapped name uppercased. All other
    legs keep their labels on both copies and therefore contract.
    """
    ket = T.relabels(relabel_map)
    bra = T.bar().relabels({k: v.upper() for k, v in relabel_map.items()})
    return contract(ket, bra)


def _bond_gram(corners: tuple[Tensor, ...], bond: str) -> jax.Array:
    """Dense environment gram M[i, j, I, J] = (E E^dagger) for one bond.

    ``i`` is the cut end at corner_i, ``j`` the end at corner_j; capitals
    are the bra copies. Built by chaining the four double-layer corners
    around the plaquette (chi^6 cost, block-sparse for SymmetricTensor),
    then densified — a (r, r, r, r) object with r the cut bond dimension.

    For ``bond="top"`` the ket layer of E is the plaquette with the top
    bond cut open (ends i, j); the bra copy closes every outer leg::

                 │                     │
                 ▼                     ▼
                 │                     │
               ┌─┴──┐                ┌─┴──┐
          ──▶──┤ B1 ├─▶─── i  j ──▶──┤ B2 ├──▶──
               └─┬──┘                └─┬──┘
                 │                     │
                 │                     │
                 ▼                     ▼
                 │                     │
                 │                     │
               ┌─┴──┐                ┌─┴──┐
          ──▶──┤ B2 ├───────▶────────┤ B1 ├──▶──
               └─┬──┘                └─┬──┘
                 │                     │
                 ▼                     ▼
                 │                     │
    """
    ci, leg_i, cj, leg_j = _BONDS[bond]["ends"]
    layers = {}
    for c in range(4):
        wiring = dict(_PLAQ_WIRING[c])
        if c == ci:
            wiring[leg_i] = "i"
        if c == cj:
            wiring[leg_j] = "j"
        layers[c] = _double_layer(corners[c], wiring)
    chain = _BONDS[bond]["chain"]
    M = layers[chain[0]]
    for c in chain[1:]:
        M = contract(M, layers[c])
    dense = M.todense()
    perm = [M.labels().index(lbl) for lbl in ("i", "j", "I", "J")]
    return jnp.transpose(dense, perm)


def _optimal_q(
    M: jax.Array, ca: np.ndarray, cb: np.ndarray, sym, eps: float
) -> jax.Array:
    """The optimal bond matrix Q from the environment gram, sector-resolved.

    The gram (as an operator on bond matrices) exactly conserves the
    charge difference across the bond, so its eigenproblem is solved per
    charge-difference sector — a dense eigh would mix accidentally
    near-degenerate eigenvectors across sectors, and the weight function's
    steep eps-shoulder amplifies that mixing into charge-violating leakage
    in Q. The spectrum is sum-normalized over ALL sectors (normalizing
    only the identity sector effectively rescales gilt_eps), while the
    trace vector t — and hence Q itself — lives purely in the identity
    sector, so Q is charge-conserving by construction.
    """
    ra, rb = M.shape[0], M.shape[1]
    mat = M.reshape(ra * rb, ra * rb)
    mat = 0.5 * (mat + mat.conj().T)
    dq = sym.fuse(ca[:, None], sym.dual(cb)[None, :])
    dq_flat = np.asarray(dq).ravel()
    dq_id = int(sym.identity())

    sector_eigs = {}
    total = 0.0
    for val in np.unique(dq_flat):
        fidx = np.flatnonzero(dq_flat == val)
        sub = mat[fidx[:, None], fidx[None, :]]
        if int(val) == dq_id:
            w, U = jnp.linalg.eigh(sub)
            sector_eigs[int(val)] = (fidx, w, U)
        else:
            w = jnp.linalg.eigvalsh(sub)
        total += float(jnp.sum(jnp.sqrt(jnp.clip(w, 0.0, None))))

    if dq_id not in sector_eigs or total <= 0.0:
        return jnp.zeros((ra, rb), dtype=M.dtype)
    fidx0, w0, U0 = sector_eigs[dq_id]
    sh = jnp.sqrt(jnp.clip(w0, 0.0, None)) / total
    eye_flat = jnp.eye(ra, rb, dtype=M.dtype).reshape(ra * rb)
    tvec = U0.conj().T @ eye_flat[fidx0]
    tp = tvec * sh**2 / (sh**2 + eps**2)
    q_flat = jnp.zeros(ra * rb, dtype=M.dtype).at[fidx0].set(U0 @ tp)
    return q_flat.reshape(ra, rb)


def _blockwise_svd_split(
    Q: jax.Array, ca: np.ndarray, cb: np.ndarray, cut: float
) -> tuple[jax.Array, jax.Array, jax.Array, np.ndarray]:
    """Per-charge-block truncated SVD split of a charge-conserving Q.

    Singular values below ``cut`` are dropped individually per block
    (with a global keep-at-least-one fallback). Returns dense assembled
    ``(L, s, R, charges)`` with ``L = u sqrt(s)`` (ra, k),
    ``R = sqrt(s) vh`` (k, rb), s the kept singular values, and the kept
    bond charges. Blockwise SVD keeps the split exactly charge-conserving
    — a dense SVD would rotate degenerate singular pairs across sectors.
    """
    ra, rb = Q.shape
    pieces = []
    for q in np.unique(np.concatenate([ca, cb])):
        ia = np.flatnonzero(ca == q)
        jb = np.flatnonzero(cb == q)
        if len(ia) == 0 or len(jb) == 0:
            continue
        block = Q[ia[:, None], jb[None, :]]
        u, s, vt = jnp.linalg.svd(block, full_matrices=False)
        keep = np.asarray(s >= cut)
        pieces.append((q, ia, jb, u, s, vt, keep))
    if not pieces:
        raise ValueError(
            "GILT bond has no charge-matched (row, column) pair — "
            "the two ends of the bond carry disjoint charge sectors"
        )
    if not any(p[6].any() for p in pieces):
        # keep the single largest singular value overall
        best = max(pieces, key=lambda p: float(p[4][0]) if p[4].size else -1.0)
        best[6][0] = True
    k_total = sum(int(p[6].sum()) for p in pieces)
    L = jnp.zeros((ra, k_total), dtype=Q.dtype)
    R = jnp.zeros((k_total, rb), dtype=Q.dtype)
    s_all = []
    charges = []
    off = 0
    for q, ia, jb, u, s, vt, keep in pieces:
        kq = int(keep.sum())
        if kq == 0:
            continue
        u, s, vt = u[:, keep], s[keep], vt[keep]
        sq = jnp.sqrt(s)
        L = L.at[ia[:, None], np.arange(off, off + kq)[None, :]].set(u * sq)
        R = R.at[np.arange(off, off + kq)[:, None], jb[None, :]].set(sq[:, None] * vt)
        s_all.append(s)
        charges.extend([q] * kq)
        off += kq
    return L, jnp.concatenate(s_all), R, np.asarray(charges, dtype=ca.dtype)


def _gilt_cascade(
    M: jax.Array, ca: np.ndarray, cb: np.ndarray, sym, config: GiltConfig
) -> tuple[jax.Array, int, int]:
    """Recursive optimization of the bond matrix Q (Hauru et al.'s scheme).

    Given the dense gram M[i, j, I, J] and the charge arrays of the two
    bond ends: compute the optimal Q from the environment spectrum
    (sum-normalized, weights S^2/(S^2 + eps^2)), SVD-split it at
    ``split_factor * gilt_eps``; if its retained spectrum is flat (all
    within ``convergence_eps`` of 1, Ebel et al. Eq. C2) stop, otherwise
    absorb the split halves into the gram and recurse. The innermost
    (flat, truncated) Q is included in the composition, and the composed
    Q is meant to be absorbed by the caller even when the first pass is
    already flat — a flat spectrum can still carry a nontrivial rank cut
    (e.g. on a pure corner-double-line tensor), and dropping it would
    skip exactly the truncation GILT exists to make.

    Returns ``(Q_total, k, rank)``: the composed bond matrix in the
    original bond basis, the number of refinement iterations (k == 0
    means the environment was flat on the first pass), and the retained
    rank of the innermost Q.
    """
    eps = config.gilt_eps
    cut = config.split_factor * eps
    left_factors: list[jax.Array] = []
    right_factors: list[jax.Array] = []
    k = 0
    while True:
        Qn = _optimal_q(M, ca, cb, sym, eps)
        L, s, R, kept_charges = _blockwise_svd_split(Qn, ca, cb, cut)
        if (
            float(jnp.max(jnp.abs(s - 1.0))) < config.convergence_eps
            or k >= config.max_cascade_iterations
        ):
            core = L @ R
            rank = int(s.shape[0])
            break
        left_factors.append(L)
        right_factors.append(R)
        M = jnp.einsum("ia,bj,ijIJ,IA,BJ->abAB", L, R, M, L.conj(), R.conj())
        ca = cb = kept_charges
        k += 1
    Q = core
    for m in reversed(left_factors):
        Q = m @ Q
    for m in reversed(right_factors):
        Q = Q @ m
    return Q, k, rank


def _split_and_absorb(
    corners_pair: tuple[Tensor, Tensor],
    bond: str,
    Q: jax.Array,
    cut: float,
) -> tuple[Tensor, Tensor]:
    """Split Q by truncated SVD and absorb the halves into (B1, B2).

    Rows of Q live on corner_i's cut leg, columns on corner_j's. The split
    drops singular values below ``cut`` individually and absorbs
    ``u sqrt(s)`` into corner_i's leg and ``sqrt(s) vh`` into corner_j's.
    """
    B1, B2 = corners_pair
    ci, leg_i, cj, leg_j = _BONDS[bond]["ends"]
    corner_of = (B1, B2, B2, B1)
    Ti, Tj = corner_of[ci], corner_of[cj]

    qi = _index_of(Ti, leg_i).flip_flow().relabel("qi")
    qj = _index_of(Tj, leg_j).flip_flow().relabel("qj")
    if isinstance(Ti, SymmetricTensor):
        Q_t: Tensor = SymmetricTensor.from_dense(Q, (qi, qj), tol=1e-8)
    else:
        Q_t = DenseTensor(Q, (qi, qj))

    _, s_full, _, _ = truncated_svd(Q_t, ["qi"], ["qj"], new_bond_label="qk")
    n_keep = max(1, int(jnp.sum(s_full >= cut)))
    U, s, Vh, _ = truncated_svd(
        Q_t, ["qi"], ["qj"], new_bond_label="qk", max_singular_values=n_keep
    )
    g1, g2 = absorb_sqrt_singular_values(U, s, Vh, "qk")

    Ti_new = contract(Ti, g1.relabel("qi", leg_i)).relabel("qk", leg_i)
    Tj_new = contract(g2.relabel("qj", leg_j), Tj).relabel("qk", leg_j)

    new_pair = {ci: Ti_new, cj: Tj_new}
    B1_new = new_pair.get(0, new_pair.get(3, B1))
    B2_new = new_pair.get(1, new_pair.get(2, B2))
    return B1_new, B2_new


def gilt_plaquette(T: Tensor, config: GiltConfig) -> tuple[Tensor, Tensor, dict]:
    """Apply the full GILT stage to the plaquette of a uniform lattice.

    Starting from the checkerboard pair ``B1 = B2 = T``, visits the four
    plaquette bonds in the order (top, bottom, right, left), running the
    cascade and absorbing the resulting bond matrices, for up to
    ``max_laps`` laps; a lap in which every bond converges in at most one
    refinement ends the stage (Ebel et al. Eq. C4).

    The plaquette and its four named bonds (arrows = charge flow, in on
    up/left, out on down/right)::

                 │                     │
                 ▼                     ▼
                 │                     │
               ┌─┴──┐      top       ┌─┴──┐
          ──▶──┤ B1 ├───────▶────────┤ B2 ├──▶──
               └─┬──┘                └─┬──┘
                 │                     │
                 │                     │
          left   ▼                     ▼  right
                 │                     │
                 │                     │
               ┌─┴──┐     bottom     ┌─┴──┐
          ──▶──┤ B2 ├───────▶────────┤ B1 ├──▶──
               └─┬──┘                └─┬──┘
                 │                     │
                 ▼                     ▼
                 │                     │

    Args:
        T:      Site tensor with legs (up, down, left, right).
        config: GILT parameters.

    Returns:
        ``(B1, B2, info)`` — the filtered checkerboard pair (B1 on the
        TL/BR sublattice, B2 on TR/BL) and an info dict with the
        per-lap cascade iteration counts (``"laps"``) and the final leg
        dimensions of B1 (``"bond_dims"``).
    """
    if not isinstance(T, Tensor):
        raise TypeError(f"gilt_plaquette() requires a Tensor, got {type(T).__name__}")
    if any(idx.symmetry.is_fermionic for idx in T.indices):
        # The gram is densified and its legs reordered with a plain
        # transpose (no Koszul signs), and the double layer uses ``bar()``
        # (no fermionic twists) — both are only correct for bosonic
        # braiding. Fermionic GILT needs a sign-aware audit of the whole
        # plaquette wiring; reject rather than silently corrupt.
        raise NotImplementedError(
            "gilt_plaquette() does not support fermionic symmetries"
        )
    B1, B2 = T, T
    if config.gilt_eps == 0.0:
        return B1, B2, {"laps": [], "bond_dims": _leg_dims(B1)}
    cut = config.split_factor * config.gilt_eps
    lap_log = []
    for _ in range(config.max_laps):
        kmaxes = []
        for bond in _BOND_ORDER:
            corners = (B1, B2, B2, B1)
            ci, leg_i, cj, leg_j = _BONDS[bond]["ends"]
            idx_i = _index_of(corners[ci], leg_i)
            idx_j = _index_of(corners[cj], leg_j)
            M = _bond_gram(corners, bond)
            Q, kmax, rank = _gilt_cascade(
                M, idx_i.charges, idx_j.charges, idx_i.symmetry, config
            )
            kmaxes.append(kmax)
            # Absorb whenever the cascade refined the bond or the flat Q
            # carries a rank cut; skip only the exact no-op (flat AND
            # full-rank), where absorbing would just rotate the bond gauge.
            if kmax >= 1 or rank < M.shape[0]:
                B1, B2 = _split_and_absorb((B1, B2), bond, Q, cut)
        lap_log.append(kmaxes)
        if all(k <= 1 for k in kmaxes):
            break
    return B1, B2, {"laps": lap_log, "bond_dims": _leg_dims(B1)}


def _leg_dims(T: Tensor) -> dict[str, int]:
    return {lbl: _index_of(T, lbl).dim for lbl in _LEG_LABELS}


def _ln_step_pair(
    BA: Tensor,
    BB: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None,
) -> tuple[Tensor, jax.Array]:
    """One Levin-Nave TRG step on the checkerboard pair (BA, BB).

    BA (TL/BR sublattice) is split on (left, down) | (up, right), BB
    (TR/BL) on (left, up) | (right, down); the four half-tensors are
    contracted around the un-filtered plaquette. The output tensor lives
    on the 45-degree rotated lattice with half the sites, and its legs
    come out in the library flow convention (up IN, down OUT, left IN,
    right OUT) with no extra permutation.

    The four half-tensors around the un-filtered plaquette (bare index
    lines; the split bonds become the coarse legs up=nw, down=se,
    left=sw, right=ne)::

                 │                     │
                 │ nw                  │ ne
                 │                     │
               ┌─┴──┐      tbond     ┌─┴──┐
               │ K2 ├────────────────┤ P1 │
               └─┬──┘                └─┬──┘
                 │                     │
                 │                     │
          lbond  │                     │  rbond
                 │                     │
                 │                     │
               ┌─┴──┐     bbond      ┌─┴──┐
               │ K1 ├────────────────┤ P2 │
               └─┬──┘                └─┬──┘
                 │                     │
                 │ sw                  │ se
                 │                     │
    """
    U1, s1, V1, _ = truncated_svd(
        BA,
        left_labels=["left", "down"],
        right_labels=["up", "right"],
        new_bond_label="ka",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    P1, K1 = absorb_sqrt_singular_values(U1, s1, V1, "ka")
    # P1: (left, down, ka) — bond becomes the new NE leg
    # K1: (ka, up, right) — bond becomes the new SW leg

    U2, s2, V2, _ = truncated_svd(
        BB,
        left_labels=["left", "up"],
        right_labels=["right", "down"],
        new_bond_label="kb",
        max_singular_values=max_bond_dim,
        max_truncation_err=svd_trunc_err,
    )
    P2, K2 = absorb_sqrt_singular_values(U2, s2, V2, "kb")
    # P2: (left, up, kb) — bond becomes the new SE leg
    # K2: (kb, right, down) — bond becomes the new NW leg

    # Corners of the un-filtered plaquette: TL=K2, TR=P1, BL=K1, BR=P2.
    TL = K2.relabels({"kb": "nw", "right": "tbond", "down": "lbond"})
    TR = P1.relabels({"ka": "ne", "left": "tbond", "down": "rbond"})
    BL = K1.relabels({"ka": "sw", "up": "lbond", "right": "bbond"})
    BR = P2.relabels({"kb": "se", "left": "bbond", "up": "rbond"})

    top = contract(TL, TR)  # contracts tbond -> (nw, lbond, ne, rbond)
    bottom = contract(BL, BR)  # contracts bbond -> (sw, lbond, rbond, se)
    C = contract(top, bottom, output_labels=("nw", "se", "sw", "ne"))
    C = C.relabels({"nw": "up", "se": "down", "sw": "left", "ne": "right"})
    return max_abs_normalize(C)


def gilt_tnr_step(T: Tensor, config: GiltTNRConfig) -> tuple[Tensor, jax.Array, dict]:
    """One Gilt-TNR step: GILT the plaquette, then one TRG contraction.

    Returns:
        ``(T_new, log_norm, info)`` — the coarse tensor (half the sites,
        45-degree rotated frame), its log normalization, and the GILT
        info dict from :func:`gilt_plaquette`.
    """
    B1, B2, info = gilt_plaquette(T, config.gilt)
    T_new, log_norm = _ln_step_pair(B1, B2, config.max_bond_dim, config.svd_trunc_err)
    return T_new, log_norm, info


def gilt_tnr(tensor: Tensor, config: GiltTNRConfig) -> jax.Array:
    """Gilt-TNR coarse-graining for a 2D square lattice partition function.

    Drop-in counterpart of :func:`tenax.algorithms.trg.trg`: same input
    convention (a site tensor with legs (up, down, left, right) repeated
    over the infinite lattice), same log-normalization accounting, but with
    the GILT filter applied to the plaquette before every TRG step. At and
    near criticality this removes the corner-double-line short-range
    entanglement that plain TRG accumulates, improving the free energy at
    equal bond dimension.

    Args:
        tensor: Initial site tensor (DenseTensor or SymmetricTensor).
        config: GiltTNRConfig parameters.

    Returns:
        Scalar JAX array: estimated log(Z)/N (free energy per site up to sign).
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"gilt_tnr() requires a Tensor, got {type(tensor).__name__}")
    T = tensor
    log_norm_total = jnp.zeros((), dtype=T.dtype)
    for step in range(config.num_steps):
        T, log_norm, _ = gilt_tnr_step(T, config)
        log_norm_total = log_norm_total + log_norm / (2.0 ** (step + 1))
    return log_norm_total
