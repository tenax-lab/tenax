"""Gilt-HOTRG: HOTRG coarse-graining with GILT bond filtering.

Drop-in counterpart of :func:`tenax.algorithms.hotrg.hotrg`: same input
convention (a uniform site tensor with legs ``(up, down, left, right)``
repeated over the infinite lattice), same ``log(Z)/N`` normalization
accounting, but with the GILT filter applied to the two inequivalent lattice
bonds (one horizontal, one vertical) before every HOTRG coarse-graining move.
At and near criticality this removes the corner-double-line short-range
entanglement that plain HOTRG accumulates, improving the free energy at equal
bond dimension.

The filter is applied UNIFORMLY, in contrast to :func:`gilt_tnr`. GILT computes
a near-identity bond matrix ``Q`` from the plaquette environment of a bond;
Gilt-TNR absorbs its two split halves into the two *checkerboard* sublattice
tensors (``B1``/``B2``) that the 45-degree TRG frame consumes. HOTRG, however,
coarse-grains a single uniform tensor, so here both halves of ``Q`` are
absorbed into the two ends of the *same* inequivalent bond of the *same*
tensor, leaving the lattice a single site tensor. This mirrors the reference
implementation ``tnrg.gilt.gilt_step_hdm`` (the graded HOTRG-with-GILT round
that reproduces the filtered Ising fixed point).

Reuse: the GILT cascade (:func:`tenax.algorithms.gilt._gilt_cascade` and its
per-charge-sector ``_optimal_q`` / blockwise split) is geometry-independent —
it needs only the dense bond gram, the two bond-end charge arrays, and the
symmetry — so it is applied verbatim; only the environment (built with uniform
corners) and the symmetric absorption are specific to the HOTRG geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from tenax.algorithms._tensor_utils import absorb_sqrt_singular_values
from tenax.algorithms.gilt import (
    _BONDS,
    GiltConfig,
    _bond_gram,
    _gilt_cascade,
    _index_of,
)
from tenax.algorithms.hotrg import _hotrg_step_horizontal, _hotrg_step_vertical
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# The two inequivalent bonds of the uniform square lattice, named in the
# plaquette convention of :mod:`tenax.algorithms.gilt`: ``"top"`` is a
# horizontal bond (it cuts the right<->left legs), ``"right"`` is a vertical
# bond (down<->up legs). Filtering one of each covers the whole lattice by
# translation invariance.
_HORIZONTAL_BOND = "top"
_VERTICAL_BOND = "right"


@dataclass
class GiltHOTRGConfig:
    """Configuration for Gilt-HOTRG coarse-graining.

    Attributes:
        max_bond_dim:    Maximum bond dimension chi after each HOTRG move.
        num_steps:       Number of coarse-graining iterations.
        direction_order: "alternating" (default), "horizontal", or "vertical",
                         matching :class:`HOTRGConfig`.
        svd_trunc_err:   Optional maximum truncation error per HOSVD.
        gilt:            GILT filter parameters applied before every HOTRG move
                         (``gilt.gilt_eps == 0.0`` disables filtering exactly,
                         recovering plain HOTRG).
        device_mesh:     Optional 1-D ``jax.sharding.Mesh`` for multi-GPU dense
                         Gilt-HOTRG, forwarded to the HOTRG move (see
                         :class:`HOTRGConfig`). Shards the dominant chi^6
                         ``T_merged`` intermediate over the mesh; the GILT
                         filter is comparatively small. Pure layout hint —
                         same free energy as single-device. Dense path only
                         (a no-op on ``SymmetricTensor``, whose block-sparse
                         HOTRG is already small).
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    direction_order: str = "alternating"
    svd_trunc_err: float | None = None
    gilt: GiltConfig = field(default_factory=GiltConfig)
    device_mesh: Mesh | None = None


def _filter_bond(T: Tensor, bond: str, config: GiltConfig) -> Tensor:
    """GILT-filter one inequivalent uniform-lattice bond and absorb both
    halves of the bond matrix into the *same* tensor, keeping the lattice
    uniform.

    ``bond`` is a key of :data:`tenax.algorithms.gilt._BONDS`; its two ends
    ``(leg_i, leg_j)`` are opposite legs of the site tensor (right/left for a
    horizontal bond, down/up for a vertical one).
    """
    if config.gilt_eps == 0.0:
        return T
    _ci, leg_i, _cj, leg_j = _BONDS[bond]["ends"]
    idx_i = _index_of(T, leg_i)
    idx_j = _index_of(T, leg_j)
    # environment gram of this bond on the uniform lattice (all four plaquette
    # corners are the same site tensor), then the geometry-independent cascade.
    gram = _bond_gram((T, T, T, T), bond)
    Q, _k, _rank = _gilt_cascade(
        gram, idx_i.charges, idx_j.charges, idx_i.symmetry, config
    )
    # split Q charge-conservingly (drop s < split_factor*eps individually,
    # keep >=1) and absorb u*sqrt(s) into leg_i, sqrt(s)*vh into leg_j.
    cut = config.split_factor * config.gilt_eps
    qi = idx_i.flip_flow().relabel("qi")
    qj = idx_j.flip_flow().relabel("qj")
    if isinstance(T, SymmetricTensor):
        Q_t: Tensor = SymmetricTensor.from_dense(Q, (qi, qj), tol=1e-8)
    else:
        Q_t = DenseTensor(Q, (qi, qj))
    _, s_full, _, _ = truncated_svd(Q_t, ["qi"], ["qj"], new_bond_label="qk")
    n_keep = max(1, int(jnp.sum(s_full >= cut)))
    U, s, Vh, _ = truncated_svd(
        Q_t, ["qi"], ["qj"], new_bond_label="qk", max_singular_values=n_keep
    )
    g1, g2 = absorb_sqrt_singular_values(U, s, Vh, "qk")
    # both halves into the single uniform site tensor (leg_i and leg_j get the
    # same new bond dimension, so the lattice stays consistent).
    T = contract(T, g1.relabel("qi", leg_i)).relabel("qk", leg_i)
    T = contract(g2.relabel("qj", leg_j), T).relabel("qk", leg_j)
    return T


def gilt_hotrg_step(
    T: Tensor, config: GiltHOTRGConfig, horizontal: bool = True
) -> tuple[Tensor, jax.Array, dict]:
    """One Gilt-HOTRG step: GILT-filter the two uniform bonds, then one HOTRG
    coarse-graining move.

    Returns:
        ``(T_new, log_norm, info)`` — the coarse site tensor (still legs
        ``(up, down, left, right)``), its log normalization from the HOTRG
        move, and an info dict with the filtered leg dimensions.
    """
    Tf = _filter_bond(T, _HORIZONTAL_BOND, config.gilt)
    Tf = _filter_bond(Tf, _VERTICAL_BOND, config.gilt)
    step_fn = _hotrg_step_horizontal if horizontal else _hotrg_step_vertical
    T_new, log_norm = step_fn(
        Tf,
        config.max_bond_dim,
        config.svd_trunc_err,
        device_mesh=config.device_mesh,
    )
    info = {
        "filtered_bond_dims": {lbl: _index_of(Tf, lbl).dim for lbl in Tf.labels()},
    }
    return T_new, log_norm, info


def gilt_hotrg(tensor: Tensor, config: GiltHOTRGConfig) -> jax.Array:
    """Gilt-HOTRG coarse-graining for a 2D square-lattice partition function.

    Drop-in counterpart of :func:`tenax.algorithms.hotrg.hotrg` with GILT bond
    filtering before every step.

    Args:
        tensor: Initial site tensor (``DenseTensor`` or ``SymmetricTensor``)
                with 4 legs labeled ``("up", "down", "left", "right")``.
        config: :class:`GiltHOTRGConfig` parameters.

    Returns:
        Scalar JAX array: estimated ``log(Z)/N`` (free energy per site).
    """
    valid_directions = ("alternating", "horizontal", "vertical")
    if config.direction_order not in valid_directions:
        raise ValueError(
            f"Invalid direction_order {config.direction_order!r}. "
            f"Must be one of {valid_directions}."
        )
    if not isinstance(tensor, Tensor):
        raise TypeError(f"gilt_hotrg() requires a Tensor, got {type(tensor).__name__}")

    T = tensor
    log_norm_total = jnp.zeros((), dtype=T.dtype)
    for step in range(config.num_steps):
        if config.direction_order == "alternating":
            horizontal = step % 2 == 0
        else:
            horizontal = config.direction_order == "horizontal"
        T, log_norm, _ = gilt_hotrg_step(T, config, horizontal=horizontal)
        # Each HOTRG step halves the number of tensors (same accounting as
        # plain HOTRG; the GILT filter is Q ~ identity and preserves Z to
        # O(gilt_eps), so it carries no separate log-norm term).
        log_norm_total = log_norm_total + log_norm / (2.0 ** (step + 1))
    return log_norm_total
