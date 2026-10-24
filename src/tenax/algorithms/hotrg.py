"""Higher-Order Tensor Renormalization Group (HOTRG) algorithm.

HOTRG improves upon TRG by using Higher-Order Singular Value Decomposition
(HOSVD) to compute truncation isometries. Instead of pairwise SVD splits,
HOTRG constructs an optimal projector by computing the truncated SVD of the
"environment tensor" M obtained by contracting two tensors over shared bonds.

Reference: Xie et al., PRB 86, 045139 (2012).

Algorithm (horizontal coarse-graining step):
  1. Form M[u,U,d,D] = sum_{l,r} T[u,d,l,r] * T[U,D,r,l]
     (contract two adjacent tensors over their shared left-right bonds)
  2. Reshape M to (d_u*d_U, d_d*d_D) and SVD to get paired isometries
     U_u of shape (d_u^2, chi) and U_d of shape (d_d^2, chi).
  3. Contract the two T tensors over the shared bond, apply paired
     isometries to compress the doubled up/down indices:
     T_new[a,b,l,r] = U_u[(u,U),a] * T_merged[(u,U),(d,D),l,r] * U_d[(d,D),b]

The vertical step is analogous with l/r bonds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tenax.algorithms._tensor_utils import max_abs_normalize
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.tensor import DenseTensor, Tensor
from tenax.linalg import eigh


def _shard_leg(T: Tensor, label: str, device_mesh: Mesh | None) -> Tensor:
    """Constrain ``T``'s ``label`` leg to be sharded over ``device_mesh``.

    A pure layout hint (``with_sharding_constraint``) — never changes numerics.
    Applied to the dominant chi^6 ``T_merged`` intermediate inside the HOTRG
    steps so it stays at ~1/N per device (dense large-chi multi-GPU HOTRG;
    HOTRG is forward-only so there is no backward SVD-VJP replication wall).

    No-op when ``device_mesh is None``, for non-``DenseTensor`` inputs (dense
    large-chi is the target regime; block-sparse HOTRG is small), or when the
    leg dimension is not divisible by the device count (the early small-bond
    steps, which carry no memory pressure).
    """
    if device_mesh is None or not isinstance(T, DenseTensor):
        return T
    n = device_mesh.devices.size
    axis = T.labels().index(label)
    leaves, treedef = jax.tree_util.tree_flatten(T)
    if not leaves or leaves[0].shape[axis] % n != 0:
        return T
    spec = [None] * len(T.labels())
    spec[axis] = device_mesh.axis_names[0]
    sharding = NamedSharding(device_mesh, PartitionSpec(*spec))
    leaves = [with_sharding_constraint(x, sharding) for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)


@dataclass
class HOTRGConfig:
    """Configuration for HOTRG coarse-graining.

    Attributes:
        max_bond_dim:    Maximum bond dimension chi after each coarse-graining step.
        num_steps:       Number of coarse-graining iterations.
        direction_order: Order of coarse-graining directions.
                         "alternating": alternate horizontal/vertical (default).
                         "horizontal": horizontal only.
                         "vertical": vertical only.
        svd_trunc_err:   Optional maximum truncation error per HOSVD.
        device_mesh:     Optional 1-D ``jax.sharding.Mesh`` for multi-GPU dense
                         HOTRG. When set, each step shards the dominant chi^6
                         ``T_merged`` intermediate over the mesh (~1/N per-device
                         peak, extends the chi ceiling). Pure layout hint —
                         same free energy as single-device. Opt-in; ``None``
                         (default) is unchanged single-device behaviour. Only
                         affects the dense path (block-sparse HOTRG is small).
        isometry:        How the coarse-bond projector W W^dagger is found.
                         "svd" (default): W = the leading left singular vectors of
                         the two-site RING tensor (the original tenax choice).
                         "eigh": Xie et al.'s HOSVD -- W = the leading eigenvectors
                         of the density matrix rho = M M^dagger of one fused pair of
                         the OPEN two-site tensor, every other leg traced against the
                         conjugate copy (see ``_hosvd_isometry``).
        side:            ``isometry="eigh"`` only: "first" / "second" = which fused
                         pair's density matrix supplies W ((up,U) / (down,D) in a
                         horizontal move, (left,L) / (right,R) in a vertical one);
                         "auto" (default) keeps the one with the smaller discarded
                         weight, ties -> first.  Ignored by ``isometry="svd"``.
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    direction_order: str = "alternating"
    svd_trunc_err: float | None = None
    device_mesh: Mesh | None = None
    isometry: str = "svd"
    side: str = "auto"


_ISOMETRIES = ("svd", "eigh")
_SIDES = ("auto", "first", "second")


def _hosvd_isometry(
    T_merged: Tensor,
    first: tuple[str, str],
    second: tuple[str, str],
    max_bond_dim: int,
    trunc_err: float | None,
    side: str,
    new_bond_label: str = "a",
) -> tuple[Tensor, float, str]:
    """Xie et al.'s HOTRG isometry (HOSVD, PRB 86, 045139): the leading eigenvectors of the
    density matrix ``rho = M M^dagger`` of ONE fused pair of the open two-site tensor
    ``M = T_merged``, every other leg traced against the conjugate copy.

    ``first`` / ``second`` name the two fused pairs (horizontal move: ``("up", "U")`` and
    ``("down", "D")``; vertical: ``("left", "L")`` and ``("right", "R")``).  The first pair's
    legs flow IN, the second's OUT.  ``side``: ``"first"`` / ``"second"`` picks the pair whose
    density matrix supplies W; ``"auto"`` builds both and keeps the one with the smaller
    discarded weight (ties -> first) -- the per-move choice of the original HOTRG.

    Returns ``(W, discarded_weight, side_used)`` with W labelled ``(first..., new_bond_label)``
    and flows ``(IN, IN, OUT)`` whichever pair it came from (the second pair's eigenvectors
    are read off the conjugate copy's legs, which flow IN, and relabelled), so the caller
    applies ``W.dagger()`` on the first pair and ``W`` relabelled on the second: ONE
    isometry, the projector ``W W^dagger`` on the coarse bond.  ``trunc_err`` (the same dial
    as ``svd_trunc_err``) drops trailing eigenvalues while the discarded fraction of the
    trace stays below it; ``max_bond_dim`` always caps.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    total = jnp.real(T_merged.norm() ** 2)  # tr(rho) for either pair

    def decompose(pair: tuple[str, str], rows_from_conj: bool):
        primed = {lab: f"{lab}__conj" for lab in pair}
        rho = contract(T_merged, T_merged.bar().relabels(primed))
        if rows_from_conj:
            left, right = [primed[lab] for lab in pair], list(pair)
        else:
            left, right = list(pair), [primed[lab] for lab in pair]
        W, w = eigh(
            rho,
            left_labels=left,
            right_labels=right,
            new_bond_label=new_bond_label,
            max_eigenvalues=max_bond_dim,
        )
        if trunc_err is not None:
            _, w_all = eigh(
                rho,
                left_labels=left,
                right_labels=right,
                new_bond_label=new_bond_label,
                max_eigenvalues=None,
            )
            w_all = jnp.real(w_all)
            discarded = total - jnp.cumsum(w_all)
            n_keep = int(jnp.sum(discarded > trunc_err * total)) + 1
            n_keep = max(1, min(n_keep, max_bond_dim, int(w_all.shape[0])))
            if n_keep < int(w.shape[0]):
                W, w = eigh(
                    rho,
                    left_labels=left,
                    right_labels=right,
                    new_bond_label=new_bond_label,
                    max_eigenvalues=n_keep,
                )
        err = float(total - jnp.sum(jnp.real(w)))
        if rows_from_conj:
            W = W.relabels({primed[s]: f for s, f in zip(pair, first)})
        return W, err

    if side == "first":
        W, err = decompose(first, False)
        return W, err, "first"
    if side == "second":
        W, err = decompose(second, True)
        return W, err, "second"
    W1, err1 = decompose(first, False)
    W2, err2 = decompose(second, True)
    return (W1, err1, "first") if err1 <= err2 else (W2, err2, "second")


def hotrg(
    tensor: Tensor,
    config: HOTRGConfig,
) -> jax.Array:
    """HOTRG coarse-graining for a 2D square lattice partition function.

    Uses Higher-Order SVD (HOSVD) for computing truncation isometries,
    providing better accuracy than TRG at the same bond dimension.

    Args:
        tensor: Initial site tensor (DenseTensor or SymmetricTensor) with
                4 legs labeled ("up", "down", "left", "right").
        config: HOTRGConfig parameters. Set ``config.device_mesh`` to a 1-D
                ``jax.sharding.Mesh`` to shard the dominant chi^6 intermediate
                over multiple GPUs (~1/N per-device peak, higher chi ceiling;
                dense path only). See ``examples/probe_hotrg_multigpu.py``.

    Returns:
        Scalar JAX array: estimated log(Z)/N (free energy per site).
    """
    valid_directions = ("alternating", "horizontal", "vertical")
    if config.direction_order not in valid_directions:
        raise ValueError(
            f"Invalid direction_order {config.direction_order!r}. "
            f"Must be one of {valid_directions}."
        )

    if not isinstance(tensor, Tensor):
        raise TypeError(f"hotrg() requires a Tensor, got {type(tensor).__name__}")
    if config.isometry not in _ISOMETRIES:
        raise ValueError(
            f"Invalid isometry {config.isometry!r}. Must be one of {_ISOMETRIES}."
        )
    if config.side not in _SIDES:
        raise ValueError(f"Invalid side {config.side!r}. Must be one of {_SIDES}.")

    T = tensor
    log_norm_total = jnp.zeros((), dtype=T.dtype)
    # Runs eager (``truncated_svd`` does host-side rank truncation and is not
    # jit-safe). When ``device_mesh`` is set, each step re-shards its input
    # ``up`` leg (see ``_hotrg_step_*``); eager contractions over sharded inputs
    # keep the dominant chi^6 ``T_merged`` sharded (~1/N per device) without ever
    # materializing it replicated.
    mesh = config.device_mesh

    for step in range(config.num_steps):
        if config.direction_order == "alternating":
            step_fn = _hotrg_step_horizontal if step % 2 == 0 else _hotrg_step_vertical
        elif config.direction_order == "horizontal":
            step_fn = _hotrg_step_horizontal
        else:
            step_fn = _hotrg_step_vertical

        T, log_norm = step_fn(
            T,
            config.max_bond_dim,
            config.svd_trunc_err,
            device_mesh=mesh,
            isometry=config.isometry,
            side=config.side,
        )

        # Each HOTRG step halves the number of tensors.
        log_norm_total = log_norm_total + log_norm / (2.0 ** (step + 1))

    return log_norm_total


def _hotrg_step_horizontal(
    T: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
    device_mesh: Mesh | None = None,
    isometry: str = "svd",
    side: str = "auto",
) -> tuple[Tensor, jax.Array]:
    """Single horizontal HOTRG coarse-graining step (polymorphic).

    Contracts two adjacent tensors horizontally and uses SVD to find
    the optimal truncation isometries for the paired up and down bonds.

    Args:
        T:             Site tensor with labels ("up", "down", "left", "right").
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed up/down bonds.
    """
    # Multi-GPU: shard the input "up" leg so the eager contractions below keep
    # the dominant chi^6 T_merged at ~1/N per device. "up" is present in T_merged
    # (up,down,left,U,D,right) and in every step's input, so re-sharding it here
    # each step handles the leg rotation across horizontal/vertical alternation.
    if isometry not in _ISOMETRIES:
        raise ValueError(f"isometry must be one of {_ISOMETRIES}, got {isometry!r}")
    T = _shard_leg(T, "up", device_mesh)
    # Step 1: Form environment M by contracting T with itself over (left, right).
    # T has labels (up, down, left, right). Second copy relabeled to avoid collision.
    if isometry == "svd":
        T_copy = T.relabels({"up": "U", "down": "D", "left": "right", "right": "left"})
        # T_copy: (U, D, right, left) — shares "left" and "right" with T
        M = contract(T, T_copy)  # contracts left↔left, right↔right → (up, down, U, D)

        # Step 2: Isometries via SVD of environment
        # Group (up, U) vs (down, D) → get paired isometries
        U_iso, _, Vh_iso, _ = truncated_svd(
            M,
            left_labels=["up", "U"],
            right_labels=["down", "D"],
            new_bond_label="a",
            max_singular_values=max_bond_dim,
            max_truncation_err=svd_trunc_err,
        )
    # U_iso: (up, U, a),  Vh_iso: (a, down, D)

    # Step 3: Merge two T copies over the shared horizontal bond
    T_left = T.relabel("right", "k")  # (up, down, left, k)
    T_right = T.relabels({"up": "U", "down": "D", "left": "k"})  # (U, D, k, right)
    T_merged = contract(T_left, T_right)  # contracts k → (up, down, left, U, D, right)
    if isometry == "eigh":
        U_iso, _, _ = _hosvd_isometry(
            T_merged, ("up", "U"), ("down", "D"), max_bond_dim, svd_trunc_err, side
        )

    # Step 4: Apply isometries to compress (up, U) → a and (down, D) → b
    # Dagger flips flow directions so contracted legs have opposite flows,
    # which is required for SymmetricTensor charge conservation.
    # Use two-step contraction (multi-tensor symmetric contraction has
    # limitations when different tensor pairs share different bonds).
    # ONE isometry on both sides of the new vertical bond: U^dagger on (up, U) and the
    # SAME U on (down, D), so the bond carries the projector U U^dagger.  Using V from
    # the SVD of M on the down side inserts V U^dagger instead, which is a projector
    # only when M is symmetric positive (V = U): a real reflection-symmetric T.  For a
    # reflection-asymmetric T (every post-GILT tensor) the untruncated step was off
    # the exact contraction by 5e-4 on gauge invariants, and for a complex T (M complex
    # symmetric, V = conj U) by 3e-2 (measured 2026-10-10).
    U_iso_dag = U_iso.dagger()  # (up_out, U_out, a_in)
    Vh_iso_b = U_iso.relabels(
        {"up": "down", "U": "D", "a": "b"}
    )  # (down_in, D_in, b_out)
    T_tmp = contract(U_iso_dag, T_merged)  # contracts up, U → (a, down, left, D, right)
    T_new = contract(T_tmp, Vh_iso_b, output_labels=("a", "b", "left", "right"))
    T_new = T_new.relabels({"a": "up", "b": "down"})

    # Step 5: Normalize
    T_new, log_norm = max_abs_normalize(T_new)
    return T_new, log_norm


def _hotrg_step_vertical(
    T: Tensor,
    max_bond_dim: int,
    svd_trunc_err: float | None = None,
    device_mesh: Mesh | None = None,
    isometry: str = "svd",
    side: str = "auto",
) -> tuple[Tensor, jax.Array]:
    """Single vertical HOTRG coarse-graining step (polymorphic).

    Analogous to horizontal step but contracts along the up-down direction.

    Args:
        T:             Site tensor with labels ("up", "down", "left", "right").
        max_bond_dim:  Maximum chi after truncation.
        svd_trunc_err: Optional maximum truncation error per HOSVD.

    Returns:
        (T_new, log_norm) where T_new has compressed left/right bonds.
    """
    # Multi-GPU: shard the input "up" leg (present in T_merged
    # (up,left,right,down,L,R)) so the eager contractions keep chi^6 at ~1/N.
    if isometry not in _ISOMETRIES:
        raise ValueError(f"isometry must be one of {_ISOMETRIES}, got {isometry!r}")
    T = _shard_leg(T, "up", device_mesh)
    # Step 1: Form environment M by contracting T with itself over (up, down).
    if isometry == "svd":
        T_copy = T.relabels({"left": "L", "right": "R", "up": "down", "down": "up"})
        # T_copy: (down, up, L, R) — shares "up" and "down" with T
        M = contract(T, T_copy)  # contracts up↔up, down↔down → (left, right, L, R)

        # Step 2: Isometries via SVD of environment
        U_iso, _, Vh_iso, _ = truncated_svd(
            M,
            left_labels=["left", "L"],
            right_labels=["right", "R"],
            new_bond_label="a",
            max_singular_values=max_bond_dim,
            max_truncation_err=svd_trunc_err,
        )
    # U_iso: (left, L, a),  Vh_iso: (a, right, R)

    # Step 3: Merge two T copies over the shared vertical bond
    T_top = T.relabel("down", "k")  # (up, k, left, right)
    T_bottom = T.relabels({"left": "L", "right": "R", "up": "k"})  # (k, down, L, R)
    T_merged = contract(T_top, T_bottom)  # contracts k → (up, left, right, down, L, R)
    if isometry == "eigh":
        U_iso, _, _ = _hosvd_isometry(
            T_merged, ("left", "L"), ("right", "R"), max_bond_dim, svd_trunc_err, side
        )

    # Step 4: Apply isometries to compress (left, L) → a and (right, R) → b
    # Dagger flips flow directions for SymmetricTensor charge conservation.
    # Use two-step contraction (see horizontal step comment).
    U_iso_dag = U_iso.dagger()
    Vh_iso_b = U_iso.relabels(
        {"left": "right", "L": "R", "a": "b"}
    )  # one isometry, see the horizontal step
    T_tmp = contract(U_iso_dag, T_merged)  # contracts left, L → (up, right, down, R, a)
    T_new = contract(T_tmp, Vh_iso_b, output_labels=("up", "down", "a", "b"))
    T_new = T_new.relabels({"a": "left", "b": "right"})

    # Step 5: Normalize
    T_new, log_norm = max_abs_normalize(T_new)
    return T_new, log_norm
