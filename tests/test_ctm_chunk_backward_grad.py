"""#632 Increment 2 gate byproduct — forward-chunk grads are exact through the
(monolith) implicit-AD backward.

Increment 1 (#668) threads ``ctm_chunk_size`` through the *forward* 1x1 CTM absorb
(chunked ``lax.map``). The Increment-2 gate established that chunking the *backward*
is a NO-GO (it defeats XLA rematerialization and increases peak memory —
``docs/superpowers/handoffs/2026-07-01-chunk-ctm-absorb-increment2-backward-gate.md``),
so the backward stays monolith. This guards the correctness half that IS shipped:
``value_and_grad`` with a chunked forward reaches a bit-identical fixed point, so
its gradient (via the monolith adjoint) matches the non-chunked gradient exactly.

Increment 1's tests were forward-only (converge parity); this closes the
``value_and_grad``-through-the-chunked-forward coverage gap. Tiny D=2 on CPU.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402


def _indices(D: int):
    sym = U1Symmetry()
    bc = np.zeros(D, dtype=np.int32)
    pc = np.zeros(2, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )


def _energy_and_grad(D: int, chi: int, chunk):
    idx = _indices(D)
    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
    key = jax.random.PRNGKey(0)
    data0 = jax.random.normal(key, (D, D, D, D, 2))
    # near-product init -> well-separated leading CTM eigenvalue (clean adjoint)
    data0 = 0.02 * data0
    data0 = data0.at[0, 0, 0, 0, :].add(1.0)
    data0 = data0 / (jnp.linalg.norm(data0) + 1e-10)

    def loss(data):
        A = DenseTensor(data, idx)
        return ctm_energy_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            max_iter=50,
            conv_tol=1e-10,
            forward_gauge="phase",
            adjoint_method="fixed_point",
            recipe="1x1",
            ctm_chunk_size=chunk,
        )

    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)


def test_forward_chunk_grad_matches_monolith():
    """value_and_grad with ctm_chunk_size (chunked forward, monolith backward)
    matches the non-chunked gradient to machine precision (bit-identical fixed
    point => same adjoint operator)."""
    e_off, g_off = _energy_and_grad(D=2, chi=6, chunk=None)
    e_on, g_on = _energy_and_grad(D=2, chi=6, chunk=2)
    assert abs(e_off - e_on) < 1e-12, abs(e_off - e_on)
    dg = float(np.max(np.abs(g_off - g_on)))
    assert dg < 1e-10, dg
