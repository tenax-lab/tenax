"""iPEPS configuration dataclasses and environment NamedTuples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax


@dataclass
class CTMConfig:
    """Configuration for CTM environment computation.

    Attributes:
        chi:          Bond dimension of CTM environment tensors.
        max_iter:     Maximum CTM iterations before declaring convergence.
        conv_tol:     Convergence tolerance (based on singular value change
                      between CTM iterations).
        renormalize:  Whether to renormalize environment tensors at each step
                      to prevent exponential growth (always recommended).
    """

    chi: int = 20
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True
    projector_method: str = "eigh"  # "eigh" or "qr"
    qr_warmup_steps: int = 3  # eigh warm-up iterations before QR kicks in
    chi_I: int | None = None  # interlayer bond dim for split-CTMRG; None => chi_I = chi
    ctm_method: str = "split"  # "split" (better scaling) or "standard"


@dataclass
class iPEPSConfig:
    """Configuration for iPEPS simple update optimization.

    Attributes:
        max_bond_dim:          PEPS virtual bond dimension D.
        num_imaginary_steps:   Number of imaginary time evolution steps.
        dt:                    Imaginary time step size.
        ctm:                   CTM configuration for environment computation.
        svd_trunc_err:         SVD truncation error for simple update.
        gate_order:            Order of bond updates: "sequential" or "random".
        su_init:               If True, ``optimize_gs_ad`` initializes the site
                               tensor via simple update (``ipeps()``) instead of
                               random initialization.  Ignored when ``A_init``
                               is provided explicitly.
    """

    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = field(default_factory=CTMConfig)
    svd_trunc_err: float | None = None
    gate_order: str = "sequential"
    unit_cell: str = "1x1"  # "1x1" or "2site"
    # AD ground-state optimization settings
    gs_optimizer: str = "adam"
    gs_learning_rate: float = 1e-3
    gs_num_steps: int = 200
    gs_conv_tol: float = 1e-8
    su_init: bool = False


class CTMEnvironment(NamedTuple):
    """The 8 CTM environment tensors (4 corners + 4 edge tensors).

    Corner convention (looking at a single site):
        C1 --- T1 --- C2
        |             |
        T4    [A]    T2
        |             |
        C4 --- T3 --- C3

    Corners (chi x chi tensors):
        C1: top-left     C2: top-right
        C3: bottom-right C4: bottom-left

    Edges (chi x D^2 x chi tensors, where D = PEPS bond dim):
        T1: top    T2: right
        T3: bottom T4: left

    All shapes use chi for environment bonds, D^2 for the PEPS bond (physical
    space of the doubled layer A * A^* is D^2 = D*D).
    """

    C1: jax.Array  # shape (chi, chi)
    C2: jax.Array  # shape (chi, chi)
    C3: jax.Array  # shape (chi, chi)
    C4: jax.Array  # shape (chi, chi)
    T1: jax.Array  # shape (chi, D2, chi) — top edge
    T2: jax.Array  # shape (chi, D2, chi) — right edge
    T3: jax.Array  # shape (chi, D2, chi) — bottom edge
    T4: jax.Array  # shape (chi, D2, chi) — left edge


class SplitCTMEnvironment(NamedTuple):
    """Split CTM environment keeping ket and bra layers separate.

    Corners are shared between ket and bra (shape ``(chi, chi)``).
    Each edge is split into a ket half ``(chi, D, chi_I)`` and a bra
    half ``(chi_I, D, chi)`` connected by an interlayer bond ``chi_I``.

    This reduces the projector cost from O(chi^3 * D^6) to
    O(chi^3 * D^3) (arXiv:2502.10298).
    """

    C1: jax.Array  # (chi, chi)
    C2: jax.Array  # (chi, chi)
    C3: jax.Array  # (chi, chi)
    C4: jax.Array  # (chi, chi)
    T1_ket: jax.Array  # (chi, D, chi_I)
    T1_bra: jax.Array  # (chi_I, D, chi)
    T2_ket: jax.Array  # (chi, D, chi_I)
    T2_bra: jax.Array  # (chi_I, D, chi)
    T3_ket: jax.Array  # (chi, D, chi_I)
    T3_bra: jax.Array  # (chi_I, D, chi)
    T4_ket: jax.Array  # (chi, D, chi_I)
    T4_bra: jax.Array  # (chi_I, D, chi)
