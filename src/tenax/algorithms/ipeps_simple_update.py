"""iPEPS simple update functions for imaginary time evolution.

Contains 2-site Tensor-protocol simple update routines and the Trotter gate
builder, extracted from the monolithic ``ipeps.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# 2-site Tensor-protocol simple update                                #
# ------------------------------------------------------------------ #

# Relative cutoff for the 1/lambda pseudo-inverse.  The Vidal update divides the
# outer lambdas back out, so a bond direction whose Schmidt value has gone to
# zero would otherwise produce inf * 0 = NaN (observed at D=3, dt=0.01).  Null
# directions are projected out instead, which is the standard simple-update
# regularization.  Lambdas are normalized to max == 1, so this is absolute too.
_LAMBDA_PINV_CUTOFF = 1e-12


def _inv_lambda(lam: jax.Array) -> jax.Array:
    """Pseudo-inverse of a bond-weight vector: 1/lam, or 0 on null directions."""
    return jnp.where(lam > _LAMBDA_PINV_CUTOFF, 1.0 / jnp.where(lam > 0, lam, 1.0), 0.0)


def _normalise_lambda(sigma: jax.Array) -> jax.Array:
    """Scale a Schmidt spectrum to ``max = 1`` exactly.

    ``sigma / (max(sigma) + EPS)`` was a no-op while ``max(sigma) >> EPS`` and
    inverted the moment it was not: at ``max(sigma) ~ 1e-15`` it returned a
    spectrum whose maximum was **0.643** rather than 1, the next sweep absorbed
    that shrunken lambda into ``theta``, ``sigma`` shrank again, and the state
    reached exactly zero two steps later.  An additive absolute epsilon in a
    denominator is the same defect as #748; the guard has to be relative.

    ``sigma`` is non-negative and descending, so ``max`` is ``sigma[0]``; an
    all-zero spectrum returns all zeros rather than dividing by nothing.
    """
    smax = jnp.max(sigma)
    return sigma / jnp.where(smax > 0, smax, 1.0)


def _truncation_base_charges(A: Tensor, leg: str) -> np.ndarray | None:
    """The canonical bond charge layout to impose on the SVD, or ``None``.

    ``base_charges`` pins the new bond's per-sector *keep counts* to the old
    bond's layout.  That is a constraint on the truncation, not a relabelling:
    it stops the SVD from keeping the globally largest singular values.

    The fermionic single-site path needs it, because there ``A.l`` and ``A.r``
    are the same physical bond and the next step crashes if the layout drifts
    (#558/#559/#563).  On the bosonic 2-site checkerboard they are *different*
    bonds, nothing downstream requires a fixed layout, and imposing it is
    actively wrong: measured on U(1)-Sz D=3 at step 0 it kept
    ``[4.611, 1.428, 0.159]`` when the true top-3 were
    ``[6.378, 4.611, 4.183]`` -- discarding the largest singular value of
    ``theta`` and retaining 25.6% of the weight where the optimal choice keeps
    87.0%.  Repeated, that drives ``lambda`` to zero and the state with it
    (#865).

    Returning ``None`` restores the global truncation, which is what minimises
    the 2-norm truncation error and what the dense path has always done.
    """
    if not A.indices[A.labels().index(leg)].symmetry.is_fermionic:
        return None
    return np.asarray(A.indices[A.labels().index(leg)].charges)


def _to_physical_tensor(gamma: Tensor, lam_h: jax.Array, lam_v: jax.Array) -> Tensor:
    """Convert a Vidal ``Gamma`` site tensor to the physical iPEPS tensor.

    The simple-update routines keep the state in Vidal (``Gamma``-``lambda``)
    form: bare site tensors with the Schmidt weights held separately on the
    bonds.  The tensor a CTM contracts is the symmetric-gauge one, carrying
    ``sqrt(lambda)`` on **all four** virtual legs, so that every bond of the
    lattice picks up ``sqrt(lambda) * sqrt(lambda) = lambda`` exactly once.

    Args:
        gamma: Bare site tensor with labels ``(u, d, l, r, phys)``.
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.

    Returns:
        The physical site tensor, normalized.
    """
    sqrt_h = jnp.sqrt(lam_h)
    sqrt_v = jnp.sqrt(lam_v)
    out = scale_bond_axis(gamma, "l", sqrt_h)
    out = scale_bond_axis(out, "r", sqrt_h)
    out = scale_bond_axis(out, "u", sqrt_v)
    out = scale_bond_axis(out, "d", sqrt_v)
    norm = float(out.norm())
    if norm > EPS:
        out = out * (1.0 / norm)
    return out


def _to_physical_pair(
    A: Tensor, B: Tensor, lam_h: jax.Array, lam_v: jax.Array
) -> tuple[Tensor, Tensor]:
    """Both checkerboard sites out of Vidal form.

    Four of the five call sites converted ``A`` and ``B`` on adjacent lines;
    this is those two calls, so that the site-to-bond mapping has one home.

    .. note::
        Both sites receive the *same* ``lam_h`` on their ``l`` and ``r`` legs
        and the same ``lam_v`` on ``u`` and ``d``.  A checkerboard has **four**
        inequivalent nearest-neighbour bonds (``A.r<->B.l``, ``B.r<->A.l``,
        ``A.d<->B.u``, ``B.d<->A.u``), so two spectra cannot describe it and
        ``steps % 4`` decides which bond's gauge is stamped on the lattice.
        That is #851, and it is deliberately **not** fixed here -- this helper
        reproduces the existing behaviour exactly.  It is written down in one
        place so the fix lands once rather than five times.
    """
    return (
        _to_physical_tensor(A, lam_h, lam_v),
        _to_physical_tensor(B, lam_h, lam_v),
    )


def _simple_update_checkerboard_sweep(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    max_D: int,
    steps: int,
    lam_h: jax.Array | None = None,
    lam_v: jax.Array | None = None,
    *,
    phase0: int = 0,
) -> tuple[Tensor, Tensor, jax.Array, jax.Array]:
    """Run ``steps`` phases of the four-bond checkerboard simple-update sweep.

    One implementation, because there were **five** byte-identical copies of
    this loop -- ``ipeps()``, ``examples/su_symmetric_ctm_e2e.py``,
    ``tests/_split_ctm_oracle.py``, ``tests/test_split_ctm_2site.py`` and
    ``tests/test_split_ctm_2site_ad.py``.  #667 had to be applied to each of
    them by hand, which is the failure shape that also produced #828, #829 and
    #842: a fix landing on some of N copies.

    The cycle covers every bond of the unit cell exactly once per four steps::

        phase 0: horizontal, left=A right=B
        phase 1: vertical,   top=A  bottom=B
        phase 2: horizontal, left=B right=A
        phase 3: vertical,   top=B  bottom=A

    Driving only phases 0 and 1 leaves ``B.r<->A.l`` and ``B.d<->A.u``
    unevolved, so ``A`` is always the left/top site of every gate and half the
    lattice bonds never acquire Schmidt weight at all -- a spuriously dimerized
    state whose 1-site RDM is indefinite (#667).

    Returns the state in Vidal form; call :func:`_to_physical_pair` for the
    tensors a CTM should contract.

    Args:
        A, B:   Bare Vidal ``Gamma`` tensors, labels ``(u, d, l, r, phys)``.
        gate:   Trotter gate, labels ``(si, sj, si_out, sj_out)``.
        max_D:  Maximum bond dimension after each truncated SVD.
        steps:  Number of phases to run (not cycles).
        lam_h:  Horizontal spectrum to start from; ones if omitted.
        lam_v:  Vertical spectrum to start from; ones if omitted.
        phase0: Phase to start the cycle on.  A caller resuming a sweep must
                pass the phase it stopped on, or the first bond is evolved
                twice and the second never.

    .. note::
        Two spectra for four inequivalent bonds is #851; see
        :func:`_to_physical_pair`.  This reproduces the existing behaviour
        exactly and does not fix it.
    """
    labels = A.labels()
    if lam_h is None:
        lam_h = jnp.ones(A.indices[labels.index("r")].dim)
    if lam_v is None:
        lam_v = jnp.ones(A.indices[labels.index("d")].dim)

    for step in range(steps):
        phase = (phase0 + step) % 4
        if phase == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, max_D
            )
        elif phase == 1:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, max_D
            )
        elif phase == 2:
            B, A, lam_h = _simple_update_2site_horizontal_tensor(
                B, A, gate, lam_h, lam_v, max_D
            )
        else:
            B, A, lam_v = _simple_update_2site_vertical_tensor(
                B, A, gate, lam_h, lam_v, max_D
            )

    return A, B, lam_h, lam_v


def _simple_update_2site_horizontal_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the horizontal bond (A.r <-> B.l).

    Works polymorphically with DenseTensor and SymmetricTensor.

    Returns the state in Vidal (``Gamma``-``lambda``) form: the site tensors are
    **bare**, with the bond weights carried only by the returned lambda.  Call
    :func:`_to_physical_tensor` to get the tensor a CTM should contract.

    The caller must evolve **all four** bonds of the checkerboard unit cell --
    this call covers ``A.r <-> B.l`` only.  Driving just this bond and its
    vertical partner leaves ``B.r <-> A.l`` and ``B.d <-> A.u`` unevolved and
    the state spuriously dimerized (#667).

    Args:
        A:     Left site tensor with labels (u, d, l, r, phys).
        B:     Right site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_h_new).
    """
    # 1. Absorb outer lambdas onto A: u<-lam_v, d<-lam_v, l<-lam_h
    #    + shared lambda on A.r<-lam_h
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)

    # 2. Absorb outer lambdas onto B: u<-lam_v, d<-lam_v, r<-lam_h (NOT l)
    B_abs = scale_bond_axis(B, "u", lam_v)
    B_abs = scale_bond_axis(B_abs, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.r with B.l
    A_left = A_abs.relabel("r", "shared")
    B_right = B_abs.relabels(
        {
            "u": "u_B",
            "d": "d_B",
            "l": "shared",
            "r": "r_B",
            "phys": "phys_B",
        }
    )
    theta = contract(A_left, B_right)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. Truncated SVD.  ``base_charges`` pins the new bond's charge layout to
    #    the old one, which is a *constraint*, not a free choice: it fixes the
    #    per-sector keep counts, so the truncation can no longer take the
    #    globally largest singular values.  Only the fermionic single-site path
    #    needs it (#558/#559/#563), where ``A.l`` and ``A.r`` are the same
    #    physical bond and the next step crashes if the layout drifts.  Here
    #    they are different bonds, and imposing it discarded the *largest*
    #    singular value of theta -- see :func:`_truncation_base_charges`.
    base_charges = _truncation_base_charges(A, "r")
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "d", "l", "si_out"],
        right_labels=["u_B", "d_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
        base_charges=base_charges,
    )

    # 6. New lambda, normalised to max 1.
    lam_h_new = _normalise_lambda(sigma)

    # 7. Reconstruct A_new from U: labels are (u, d, l, si_out, bond_new)
    #    Transpose so bond_new is in the r position: (u, d, l, bond_new, si_out)
    #    Gamma stays BARE (#667): the shared bond's weight lives in lam_h_new and
    #    is re-absorbed in full by step 1 of the next sweep.  Scaling it in here
    #    as well made the bond carry lambda**1.5, which drove the state to the
    #    product state -- see ``_to_physical_tensor`` for how it comes back.
    A_new = U.transpose((0, 1, 2, 4, 3))
    A_new = A_new.relabels({"bond_new": "r", "si_out": "phys"})

    # 8. Reconstruct B_new from Vh: labels are (bond_new, u_B, d_B, r_B, sj_out)
    #    Transpose so bond_new is in the l position: (u_B, d_B, bond_new, r_B, sj_out)
    B_new = Vh.transpose((1, 2, 0, 3, 4))
    B_new = B_new.relabels(
        {"bond_new": "l", "u_B": "u", "d_B": "d", "r_B": "r", "sj_out": "phys"}
    )

    # 9. Remove outer lambdas
    inv_lam_v = _inv_lambda(lam_v)
    inv_lam_h = _inv_lambda(lam_h)
    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "d", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)

    B_new = scale_bond_axis(B_new, "u", inv_lam_v)
    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize each tensor
    norm_A = float(A_new.norm())
    if norm_A > EPS:
        A_new = A_new * (1.0 / norm_A)
    norm_B = float(B_new.norm())
    if norm_B > EPS:
        B_new = B_new * (1.0 / norm_B)

    return A_new, B_new, lam_h_new


def _simple_update_2site_vertical_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the vertical bond (A.d <-> B.u).

    Works polymorphically with DenseTensor and SymmetricTensor.  Returns bare
    Vidal ``Gamma`` tensors -- see the horizontal counterpart above.

    Args:
        A:     Top site tensor with labels (u, d, l, r, phys).
        B:     Bottom site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_v_new).
    """
    # 1. Absorb outer lambdas onto A: u<-lam_v, l<-lam_h, r<-lam_h
    #    + shared lambda on A.d<-lam_v
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)

    # 2. Absorb outer lambdas onto B: d<-lam_v, l<-lam_h, r<-lam_h (NOT u)
    B_abs = scale_bond_axis(B, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "l", lam_h)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.d with B.u
    A_top = A_abs.relabel("d", "shared")
    B_bottom = B_abs.relabels(
        {
            "u": "shared",
            "d": "d_B",
            "l": "l_B",
            "r": "r_B",
            "phys": "phys_B",
        }
    )
    theta = contract(A_top, B_bottom)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. Truncated SVD.  See the horizontal counterpart for why the canonical
    #    charge layout is imposed only on the fermionic path (#563, #865).
    base_charges = _truncation_base_charges(A, "d")
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "l", "r", "si_out"],
        right_labels=["d_B", "l_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
        base_charges=base_charges,
    )

    # 6. New lambda, normalised to max 1.
    lam_v_new = _normalise_lambda(sigma)

    # 7. Reconstruct A_new from U: labels are (u, l, r, si_out, bond_new)
    #    Transpose so bond_new is in the d position: (u, bond_new, l, r, si_out)
    #    Gamma stays BARE -- see the horizontal counterpart above (#667).
    A_new = U.transpose((0, 4, 1, 2, 3))
    A_new = A_new.relabels({"bond_new": "d", "si_out": "phys"})

    # 8. Reconstruct B_new from Vh: labels are (bond_new, d_B, l_B, r_B, sj_out)
    #    Transpose so bond_new is in the u position: (bond_new, d_B, l_B, r_B, sj_out)
    #    Already in the right order for (u, d, l, r, phys)
    B_new = Vh.relabels(
        {"bond_new": "u", "d_B": "d", "l_B": "l", "r_B": "r", "sj_out": "phys"}
    )

    # 9. Remove outer lambdas
    inv_lam_v = _inv_lambda(lam_v)
    inv_lam_h = _inv_lambda(lam_h)
    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)
    A_new = scale_bond_axis(A_new, "r", inv_lam_h)

    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "l", inv_lam_h)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize each tensor
    norm_A = float(A_new.norm())
    if norm_A > EPS:
        A_new = A_new * (1.0 / norm_A)
    norm_B = float(B_new.norm())
    if norm_B > EPS:
        B_new = B_new * (1.0 / norm_B)

    return A_new, B_new, lam_v_new


def _make_trotter_gate_tensor(
    hamiltonian_gate: jax.Array | Tensor,
    dt: float,
    site_tensor: Tensor | None = None,
) -> Tensor:
    """Build the Trotter gate exp(-dt * H) as a 4-leg Tensor.

    When *site_tensor* is provided, the gate indices use the same symmetry
    and physical charges as the site tensor's physical leg.  If the site
    tensor is a ``SymmetricTensor``, the gate is built via ``from_dense``
    so that it matches the type and can be contracted directly.

    Args:
        hamiltonian_gate: 2-site Hamiltonian as (d,d,d,d) array or Tensor.
        dt: Imaginary time step.
        site_tensor: Optional reference tensor whose physical index provides
                     symmetry and charge information for the gate.

    Returns:
        Tensor with labels (si, sj, si_out, sj_out).
    """
    if isinstance(hamiltonian_gate, Tensor):
        H_dense = hamiltonian_gate.todense()
    else:
        H_dense = jnp.asarray(hamiltonian_gate)

    d = H_dense.shape[0]
    H_mat = H_dense.reshape(d * d, d * d)
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(H_mat)
    gate_mat = eigvecs @ jnp.diag(jnp.exp(-dt * eigvals)) @ eigvecs.conj().T
    gate_4leg = gate_mat.reshape(d, d, d, d)

    # Derive index metadata from site tensor's physical leg if available
    if site_tensor is not None:
        phys_idx = site_tensor.indices[-1]  # last leg = phys
        sym = phys_idx.symmetry
        charges = phys_idx.charges
    else:
        sym = U1Symmetry()
        charges = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )

    if isinstance(site_tensor, SymmetricTensor):
        return SymmetricTensor.from_dense(gate_4leg, indices)

    return DenseTensor(gate_4leg, indices)
