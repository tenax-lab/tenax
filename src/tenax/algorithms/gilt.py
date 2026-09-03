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
(block-sparse for ``SymmetricTensor``). Rather than densify it into a full
``(r, r, r, r)`` (i.e. ``chi^4``) array, the gram is kept in the
charge-difference-block form it already has: reshaped to a matrix
``mat[(i, j), (I, J)]`` it is exactly block-diagonal in
``dq = fuse(charge_i, dual(charge_j))`` (the plaquette environment conserves
the charge flowing across the cut bond), so only the ``dq`` diagonal blocks
are nonzero. The cascade — spectrum, optimal ``Q``, and the per-iteration
absorb — runs on those blocks directly (:class:`_BondGram`), which for a
``Z_n`` grading is an ``n``-fold memory saving over the dense ``chi^4`` object
and never materializes it. The composed ``Q`` is charge-conserving whenever
the input tensor is, and is re-wrapped into the input's tensor type before
absorption.
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


@dataclass
class _BondGram:
    """Environment gram in charge-difference-block form (never densified).

    The gram, reshaped to a matrix ``mat[(i, j), (I, J)]`` with row/column
    flat index ``i * rb + j``, is exactly block-diagonal in the charge
    difference ``dq = fuse(charge_i, dual(charge_j))`` across the cut bond
    (a ``dq`` off-diagonal block would violate the plaquette's charge
    conservation, and is verified to carry exactly zero weight). Only those
    diagonal blocks are stored.

    Attributes:
        blocks:   ``{dq_value: (n_dq, n_dq) raw matrix}`` — the un-symmetrized
                  sub-block ``mat[fidx, fidx]`` for each charge-difference
                  sector, with ``fidx = sectors[dq_value]``.
        sectors:  ``{dq_value: fidx}`` — the flat ``i * rb + j`` positions of
                  each sector, ascending (as ``np.flatnonzero`` returns).
        ra, rb:   Dimensions of the two cut ends (row leg ``i``/``I`` and
                  column leg ``j``/``J``); ``ra`` is the cut bond dimension.
        ca, cb:   Per-state charges of the ``i``/``I`` and ``j``/``J`` legs,
                  in the dense (``todense``) ordering.
        dq_id:    The identity charge difference — the sector that carries the
                  optimal ``Q`` (``Q`` lives purely there).
        sym:      The leg symmetry.
        dtype:    Block dtype.
        dense_full: For a non-symmetric (``DenseTensor``) gram only, the full
                  ``(ra*rb, ra*rb)`` raw matrix. There is no block structure to
                  exploit (a single trivial charge sector), so it is kept whole
                  and the absorb uses the original dense einsum — reproducing
                  the pre-refactor dense path bit-for-bit. ``None`` for the
                  block-sparse (``SymmetricTensor``) path, which never
                  materializes the full matrix.
    """

    blocks: dict[int, jax.Array]
    sectors: dict[int, np.ndarray]
    ra: int
    rb: int
    ca: np.ndarray
    cb: np.ndarray
    dq_id: int
    sym: object
    dtype: object
    dense_full: jax.Array | None = None

    @property
    def dim(self) -> int:
        """The cut bond dimension (rows of the gram operator)."""
        return self.ra


def _dq_sectors(
    ca: np.ndarray, cb: np.ndarray, rb: int, sym
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Partition the flat ``(i, j)`` index space by charge difference ``dq``.

    Returns ``(sectors, local_of)`` where ``sectors[dq] = fidx`` are the flat
    ``i * rb + j`` positions with ``fuse(ca[i], dual(cb[j])) == dq`` (ascending),
    and ``local_of[p]`` is the position of flat index ``p`` within its own
    sector — the inverse map used to scatter blocks into their sector matrix.
    """
    dq_flat = np.asarray(sym.fuse(ca[:, None], sym.dual(cb)[None, :])).ravel()
    sectors: dict[int, np.ndarray] = {}
    local_of = np.empty(dq_flat.size, dtype=np.int64)
    for val in np.unique(dq_flat):
        fidx = np.flatnonzero(dq_flat == val)
        sectors[int(val)] = fidx
        local_of[fidx] = np.arange(fidx.size)
    return sectors, local_of


def _gram_from_tensor(M: Tensor) -> _BondGram:
    """Build the block-diagonal :class:`_BondGram` from the 4-leg gram tensor.

    ``M`` has legs labeled ``i, j, I, J`` (capitals the bra copies, sharing
    the charge structure of ``i, j``). For a ``SymmetricTensor`` the sector
    matrices are scattered directly from ``M``'s charge blocks — the full
    ``chi^4`` dense array is never formed. For a ``DenseTensor`` (a single
    trivial charge sector) the whole matrix is one block; there is no symmetry
    to exploit, so it degenerates to the former dense path.
    """
    perm = tuple(M.labels().index(lbl) for lbl in ("i", "j", "I", "J"))
    idx_i = M.indices[perm[0]]
    idx_j = M.indices[perm[1]]
    sym = idx_i.symmetry
    ca = np.asarray(idx_i.charges)
    cb = np.asarray(idx_j.charges)
    ra, rb = int(idx_i.dim), int(idx_j.dim)
    dq_id = int(sym.identity())
    dtype = M.dtype

    sectors, local_of = _dq_sectors(ca, cb, rb, sym)
    blocks: dict[int, jax.Array] = {
        val: jnp.zeros((fidx.size, fidx.size), dtype=dtype)
        for val, fidx in sectors.items()
    }

    if isinstance(M, SymmetricTensor):
        for key, block in M.blocks.items():
            q_i, q_j = int(key[perm[0]]), int(key[perm[1]])
            q_I, q_J = int(key[perm[2]]), int(key[perm[3]])
            dq_r = int(sym.fuse(np.array([q_i]), sym.dual(np.array([q_j])))[0])
            dq_c = int(sym.fuse(np.array([q_I]), sym.dual(np.array([q_J])))[0])
            if dq_r != dq_c:
                # A dq off-diagonal block: the per-sector spectrum ignores it
                # and the plaquette gram carries none (charge is conserved
                # across the cut). Skipping keeps us faithful and dense-free.
                continue
            rows_i = np.flatnonzero(ca == q_i)
            rows_j = np.flatnonzero(cb == q_j)
            cols_I = np.flatnonzero(ca == q_I)
            cols_J = np.flatnonzero(cb == q_J)
            flat_rows = (rows_i[:, None] * rb + rows_j[None, :]).ravel()
            flat_cols = (cols_I[:, None] * rb + cols_J[None, :]).ravel()
            lr = local_of[flat_rows]
            lc = local_of[flat_cols]
            bp = jnp.transpose(block, perm).reshape(flat_rows.size, flat_cols.size)
            blocks[dq_r] = blocks[dq_r].at[lr[:, None], lc[None, :]].add(bp)
        return _BondGram(blocks, sectors, ra, rb, ca, cb, dq_id, sym, dtype)

    # DenseTensor: a single trivial charge sector, no symmetry to exploit.
    # Keep the whole matrix so the absorb can use the original dense einsum
    # (bit-for-bit identical to the pre-refactor dense path).
    mat = jnp.transpose(M.todense(), perm).reshape(ra * rb, ra * rb)
    for val, fidx in sectors.items():
        blocks[val] = mat[fidx[:, None], fidx[None, :]]
    return _BondGram(blocks, sectors, ra, rb, ca, cb, dq_id, sym, dtype, dense_full=mat)


def _bond_gram(corners: tuple[Tensor, ...], bond: str) -> _BondGram:
    """Environment gram (E E^dagger) for one bond, as a :class:`_BondGram`.

    ``i`` is the cut end at corner_i, ``j`` the end at corner_j; capitals
    are the bra copies. Built by chaining the four double-layer corners
    around the plaquette (chi^6 cost, block-sparse for SymmetricTensor),
    then packed into charge-difference blocks (never a dense ``chi^4`` array;
    see :class:`_BondGram`) with ``r`` the cut bond dimension.

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
    return _gram_from_tensor(M)


def _optimal_q(gram: _BondGram, eps: float) -> jax.Array:
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

    Each ``gram.blocks[val]`` is the raw (un-symmetrized) sector sub-matrix;
    symmetrizing it here is identical to slicing the symmetrized full matrix,
    since the sector is on the block diagonal.
    """
    ra, rb, dq_id = gram.ra, gram.rb, gram.dq_id

    sector_eigs = None
    total = 0.0
    for val in sorted(gram.blocks):
        sub_raw = gram.blocks[val]
        sub = 0.5 * (sub_raw + sub_raw.conj().T)
        if val == dq_id:
            w, U = jnp.linalg.eigh(sub)
            sector_eigs = (gram.sectors[val], w, U)
        else:
            w = jnp.linalg.eigvalsh(sub)
        total += float(jnp.sum(jnp.sqrt(jnp.clip(w, 0.0, None))))

    if sector_eigs is None or total <= 0.0:
        return jnp.zeros((ra, rb), dtype=gram.dtype)
    fidx0, w0, U0 = sector_eigs
    sh = jnp.sqrt(jnp.clip(w0, 0.0, None)) / total
    eye_flat = jnp.eye(ra, rb, dtype=gram.dtype).reshape(ra * rb)
    tvec = U0.conj().T @ eye_flat[fidx0]
    tp = tvec * sh**2 / (sh**2 + eps**2)
    q_flat = jnp.zeros(ra * rb, dtype=gram.dtype).at[fidx0].set(U0 @ tp)
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


def _absorb_gram(
    gram: _BondGram, L: jax.Array, R: jax.Array, kept_charges: np.ndarray
) -> _BondGram:
    """Absorb the split halves (L, R) into the gram, sector by sector.

    Realizes ``M'[a,b,A,B] = sum_{ijIJ} L[i,a] R[b,j] M[i,j,I,J]
    conj(L[I,A]) conj(R[B,J])`` — but per charge-difference sector rather
    than as a dense ``chi^4`` einsum. Since ``L`` is charge-block-diagonal
    (``L[i,a]`` nonzero only where ``ca[i] == kept_charges[a]``) and likewise
    ``R``, the row map ``P[(a,b),(i,j)] = L[i,a] R[b,j]`` couples a new sector
    only to the same old sector, so each new block is
    ``P_val @ old_block @ P_val^H``. The new bond legs both carry
    ``kept_charges``.

    A ``DenseTensor`` gram (``dense_full`` set) has no block structure to
    exploit; it takes the original dense einsum on the full 4-leg matrix
    instead, reproducing the pre-refactor path exactly (per-sector matmuls
    would reorder the floating-point sums, which the near-critical flow
    amplifies).
    """
    kc = np.asarray(kept_charges)
    k = int(kc.size)
    sym = gram.sym
    rb_old = gram.rb
    sectors, _ = _dq_sectors(kc, kc, k, sym)

    if gram.dense_full is not None:
        # No block structure to exploit: run the original dense einsum on the
        # full 4-leg gram, bit-for-bit as before, and re-slice its sectors.
        m4 = gram.dense_full.reshape(gram.ra, rb_old, gram.ra, rb_old)
        mp = jnp.einsum("ia,bj,ijIJ,IA,BJ->abAB", L, R, m4, L.conj(), R.conj())
        new_full = mp.reshape(k * k, k * k)
        blocks = {v: new_full[f[:, None], f[None, :]] for v, f in sectors.items()}
        return _BondGram(
            blocks, sectors, k, k, kc, kc, gram.dq_id, sym, gram.dtype, new_full
        )

    blocks: dict[int, jax.Array] = {}
    for val, new_fidx in sectors.items():
        old_block = gram.blocks.get(val)
        old_fidx = gram.sectors.get(val)
        if old_block is None or old_fidx is None:
            blocks[val] = jnp.zeros((new_fidx.size, new_fidx.size), dtype=gram.dtype)
            continue
        new_a, new_b = new_fidx // k, new_fidx % k
        old_i, old_j = old_fidx // rb_old, old_fidx % rb_old
        p = L[old_i[None, :], new_a[:, None]] * R[new_b[:, None], old_j[None, :]]
        blocks[val] = p @ old_block @ p.conj().T
    return _BondGram(blocks, sectors, k, k, kc, kc, gram.dq_id, sym, gram.dtype)


def _gilt_cascade(gram: _BondGram, config: GiltConfig) -> tuple[jax.Array, int, int]:
    """Recursive optimization of the bond matrix Q (Hauru et al.'s scheme).

    Given the block-diagonal gram :class:`_BondGram`: compute the optimal Q
    from the environment spectrum (sum-normalized, weights S^2/(S^2 + eps^2)),
    SVD-split it at ``split_factor * gilt_eps``; if its retained spectrum is
    flat (all within ``convergence_eps`` of 1, Ebel et al. Eq. C2) stop,
    otherwise absorb the split halves into the gram and recurse. The innermost
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
        Qn = _optimal_q(gram, eps)
        L, s, R, kept_charges = _blockwise_svd_split(Qn, gram.ca, gram.cb, cut)
        if (
            float(jnp.max(jnp.abs(s - 1.0))) < config.convergence_eps
            or k >= config.max_cascade_iterations
        ):
            core = L @ R
            rank = int(s.shape[0])
            break
        left_factors.append(L)
        right_factors.append(R)
        gram = _absorb_gram(gram, L, R, kept_charges)
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
        # The gram's blocks are reordered with a plain transpose (no Koszul
        # signs) when packed into charge-difference sectors, and the double
        # layer uses ``bar()`` (no fermionic twists) — both are only correct
        # for bosonic braiding. Fermionic GILT needs a sign-aware audit of the
        # whole plaquette wiring; reject rather than silently corrupt.
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
            gram = _bond_gram(corners, bond)
            Q, kmax, rank = _gilt_cascade(gram, config)
            kmaxes.append(kmax)
            # Absorb whenever the cascade refined the bond or the flat Q
            # carries a rank cut; skip only the exact no-op (flat AND
            # full-rank), where absorbing would just rotate the bond gauge.
            if kmax >= 1 or rank < gram.dim:
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
