"""Regenerate the frozen physical simple-update tensor for the #772 fixture.

Run:  JAX_PLATFORMS=cpu uv run python scripts/gen_su_fixture.py
Paste the printed literal into ``tests/_su_fixtures.py``.

The state is a D=2 Heisenberg simple-update ground state.  It is the smallest
*physical* state that violates the retained-spectrum precondition the covariant
characteristic equations depend on (#772), which is why it is the fixture.

**This script no longer reproduces the frozen literal, and must not be used to
refresh it** (#667).  That literal was generated when simple update converged to
the product state, and its defining pathology -- a retained spectrum collapsing
with chi, usable rank 3 -- is a consequence of that bug.  Simple update is fixed,
so this now builds a genuinely entangled state (``retained_smin_rtol`` 4.0e-04
rather than 3.0e-08 at chi=4); pasting it in would silently replace the
pathological state #772/#778/#784/#785 exist to exercise with a healthy one.
Kept for provenance and for generating *new* fixtures, not for refreshing that one.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def build():
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=6, max_iter=100, conv_tol=1e-10),
    )
    E, tensors, _ = ipeps(gate, None, cfg)
    A = tensors[0]
    A = A * (1.0 / (A.norm() + 1e-10))
    return float(E), A


if __name__ == "__main__":
    E, A = build()
    print(f"# labels = {list(A.labels())}")
    # The energy is printed as a comment, NOT as a paste-in literal (#836).
    # ``ipeps`` evaluates it with a 2-site CTM at the config's chi=6, and this
    # state is the one whose chi=6 environment is rank-3 of 6 -- that sweep
    # stops on max_iter rather than converging, so E moves ~7e-3 with the
    # iteration budget while conv_tol does nothing.  A frozen literal for it
    # cannot hold; the test compares states physically instead.
    print(f"# E_su (diagnostic only, not reproducible -- see #836) = {E!r}")
    print("PHYSICAL_SU_D2_DATA = np.array(")
    print(f"    {np.asarray(A.todense()).tolist()!r}")
    print(")")
    # Emitted rather than assumed: simple update does NOT preserve the
    # (OUT, IN, OUT, IN, IN) layout `_wrap_as_dense_tensor` starts from -- the
    # returned tensor is all-OUT.  Hardcoding the initial layout would give the
    # fixture different flows from the state it claims to be.
    print(f"PHYSICAL_SU_D2_FLOWS = {tuple(int(i.flow) for i in A.indices)!r}")
    print(f"PHYSICAL_SU_D2_LABELS = {tuple(A.labels())!r}")
