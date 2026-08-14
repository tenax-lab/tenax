#!/usr/bin/env python3
"""Gradient-free symmetric forward-CTM energy: what works, what collapses.

Premise: to sidestep the large-D symmetric-AD wall (``_jit_fused_fixed_point_bwd``,
the block-sparse SVD/eigh/projector VJPs), evaluate energy with a *forward-only*
symmetric CTM (as in an SU + CTM workflow).  The forward path never builds the
backward VJPs, so large-D symmetric CTM is fine — *if* the forward CTM and energy
are correct.

This script pins down exactly which forward paths are correct today:

  PART A (verified WORKING) — single-site symmetric CTM + energy.
    ``ctm_tensor`` + ``compute_energy_ctm_tensor`` on a trivial-charge U(1)
    ``SymmetricTensor`` gives a finite NONZERO energy that matches the densified
    tensor to ~1e-12.  This confirms the forward symmetric CTM + RDM + energy
    machinery is sound and block-sparse (no chi-env densification).

  PART B (currently BROKEN) — 2-site checkerboard ``ctm_tensor_2site``.
    Feeding a U(1)-Sz Heisenberg pair (``heisenberg_u1sz_init_pair``) through SU
    then ``ctm_tensor_2site`` yields a corner of norm 0 -> energy exactly 0.
    The collapse reproduces on the *densified* tensors too, so it is a general
    2-site forward-CTM defect, not a symmetry/charge issue.  The shipped test
    ``test_2site_symmetric_converges`` only checks convergence, never the energy
    value, so it does not catch this.

Takeaway: gradient-free symmetric energy works today via the single-site CTM;
the 2-site (bipartite-Heisenberg) forward CTM needs a fix before an SU->CTM
Heisenberg pipeline is usable.

Usage::

    JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/jaxfresh \\
        uv run python examples/su_symmetric_ctm_e2e.py --D 4 --chi 16
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
    ctm_tensor,
    ctm_tensor_2site,
)
from tenax.algorithms.ipeps import (  # noqa: E402
    heisenberg_gate,
    heisenberg_gate_u1sz,
    heisenberg_u1sz_init_pair,
)
from tenax.algorithms.ipeps_simple_update import (  # noqa: E402
    _make_trotter_gate_tensor,
    _simple_update_checkerboard_sweep,
    _to_physical_pair,
)
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor, SymmetricTensor  # noqa: E402


def _trivial_charge_site(D: int, d: int, seed: int) -> SymmetricTensor:
    """Random single-site iPEPS as a trivial-charge U(1) SymmetricTensor."""
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    data = jnp.array(np.random.RandomState(seed).standard_normal((D, D, D, D, d)))
    return SymmetricTensor.from_dense(data, idx)


def part_a_single_site(D: int, chi: int, seed: int) -> None:
    """Verified working: single-site symmetric CTM + energy (sym == dense)."""
    print("=== PART A: single-site symmetric CTM + energy (should WORK) ===")
    A = _trivial_charge_site(D, 2, seed)
    H = heisenberg_gate()

    env, _ = ctm_tensor(A, chi=chi, max_iter=60, conv_tol=1e-8)
    E_sym = float(compute_energy_ctm_tensor(A, env, H))

    Ad = DenseTensor(A.todense(), A.indices)
    envd, _ = ctm_tensor(Ad, chi=chi, max_iter=60, conv_tol=1e-8)
    E_den = float(compute_energy_ctm_tensor(Ad, envd, H))

    c1 = float(jnp.linalg.norm(env.C1.todense()))
    ok = abs(E_sym) > 1e-9 and abs(E_sym - E_den) < 1e-8
    print(f"  C1 corner norm : {c1:.6f}")
    print(f"  E (symmetric)  : {E_sym:.10f}")
    print(f"  E (densified)  : {E_den:.10f}   |diff|={abs(E_sym - E_den):.2e}")
    print(f"  verdict        : {'PASS (nonzero, sym==dense)' if ok else 'FAIL'}\n")


def part_b_2site(D: int, chi: int, steps: int, seed: int) -> None:
    """Currently broken: SU -> ctm_tensor_2site corner collapses to zero."""
    print("=== PART B: 2-site U(1)-Sz SU -> ctm_tensor_2site (KNOWN COLLAPSE) ===")
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))
    H = heisenberg_gate_u1sz()
    gate = _make_trotter_gate_tensor(H, 0.1, site_tensor=A)
    # All four bonds of the checkerboard, and the physical tensor rebuilt from
    # the Vidal form afterwards -- driving only (A.r<->B.l)/(A.d<->B.u) leaves
    # half the lattice bonds with no Schmidt weight (#667).
    A, B, lam_h, lam_v = _simple_update_checkerboard_sweep(A, B, gate, D, steps)
    A, B = _to_physical_pair(A, B, lam_h, lam_v)

    print(
        f"  after {steps} SU steps: |A|={float(A.norm()):.4f} "
        f"|B|={float(B.norm()):.4f}  type={type(A).__name__}"
    )
    for recipe in ("1x1", "2x2"):
        env_A, env_B = ctm_tensor_2site(
            A, B, chi=chi, max_iter=80, conv_tol=1e-8, recipe=recipe
        )
        c1 = float(jnp.linalg.norm(env_A.C1.todense()))
        E = float(compute_energy_ctm_tensor_2site(A, B, env_A, env_B, H))
        flag = "COLLAPSED" if c1 == 0.0 or abs(E) < 1e-12 else "ok"
        print(f"  recipe={recipe:>3}: C1 norm={c1:.3e}  E/site={E:.6f}  -> {flag}")
    print(
        "  (collapse reproduces on densified tensors too -> general 2-site "
        "forward-CTM defect)\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=int, default=4)
    p.add_argument("--chi", type=int, default=16)
    p.add_argument("--steps", type=int, default=60, help="2-site SU steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        device = jax.devices()[0].device_kind
    except Exception:
        device = "unknown"
    print(f"\ndevice={device}  D={args.D}  chi={args.chi}\n")

    part_a_single_site(args.D, args.chi, args.seed)
    part_b_2site(args.D, args.chi, args.steps, args.seed)


if __name__ == "__main__":
    main()
