"""Single-site split-CTM AD energy entry points (#463 Phase 2).

Mirrors ``ctm_energy_explicit`` / ``ctm_energy_implicit`` from
``_ctm_energy_ad`` but on the split (ket/bra-separate) double layer.
Single-site only: the split forward (``ctm_split_tensor``) converges one
site as an isolated 1×1 iPEPS.  Multisite has no split forward yet.
"""

from __future__ import annotations

__all__ = ["ctm_energy_split_explicit"]


def _extract_single_site(site_tensors):
    if len(site_tensors) != 1:
        raise NotImplementedError(
            "split-CTM (fuse_virtual_legs=False) supports only the single-site "
            f"(recipe='1x1') path; got {len(site_tensors)} sites."
        )
    ((_coord, A),) = site_tensors.items()
    return A


def ctm_energy_split_explicit(
    site_tensors,
    neighbors,
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    backward_steps: int | None = None,
    chi_I: int | None = None,
    renormalize: bool = True,
    energy_fn=None,
    **_ignored,
):
    """Single-site iPEPS energy with explicit (unrolled) split-CTM AD."""
    A = _extract_single_site(site_tensors)
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn (e.g. coarse-grain) is not supported on the split "
            "path yet; use fuse_virtual_legs=True."
        )
    if backward_steps is not None:
        raise ValueError(
            "backward_steps (TBPTT) is not supported on the split explicit path; "
            "set gs_explicit_ad_backward_steps=None or use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor,
    )
    from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit

    env = ctm_split_tensor_converge_explicit(
        A,
        chi=chi,
        chi_I=chi_I,
        renormalize=renormalize,
        num_steps=backprop_steps,
        warmup_steps=warmup_steps,
    )
    return compute_energy_split_ctm_tensor(A, env, gate)
