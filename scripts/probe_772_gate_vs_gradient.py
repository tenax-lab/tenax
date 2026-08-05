"""#772: does the residual gate predict gradient error at all?

The tolerance question only has a principled answer if `covariant_residual`
tracks how wrong the gradient is.  On the physical fixture under #778's clamp
the residual grows 4x from chi=4 to chi=12 while max|g| is bit-identical, which
suggests it does not.  Settle it by checking the gradient against a directional
finite difference at the BEST and WORST residual.

Run: JAX_PLATFORMS=cpu uv run python scripts/probe_772_gate_vs_gradient.py
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, "tests")

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from _su_fixtures import physical_su_d2  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402


def _gate(delta=1.0):
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


A = physical_su_d2()
H = _gate()
KW = dict(max_iter=300, conv_tol=1e-13, on_root_residual="warn")


def energy_and_grad(state, chi):
    return M.asym_root_implicit_energy_and_grad(state, H, chi=chi, **KW)


print(f"{'chi':>4} {'cov_resid':>12} {'AD':>18} {'FD':>18} {'rel err':>11}")
print("-" * 68)

rng = np.random.RandomState(0)
base = np.asarray(A.todense())
v = rng.standard_normal(base.shape)
v /= np.linalg.norm(v)

for chi in (4, 12):
    _E, g, d = M.asym_root_implicit_energy_and_grad(
        A, H, chi=chi, return_diagnostics=True, **KW
    )
    ad = float(np.sum(np.asarray(g) * v))

    def energy_at(t, chi=chi):
        At = DenseTensor(jnp.asarray(base + t * v), A.indices)
        E, _ = energy_and_grad(At, chi)
        return float(jnp.real(E))

    h = 1e-5
    fd = (energy_at(h) - energy_at(-h)) / (2 * h)
    rel = abs(fd - ad) / max(abs(fd), 1e-300)
    print(
        f"{chi:>4} {d['covariant_residual']:>12.3e} {ad:>18.10e} "
        f"{fd:>18.10e} {rel:>11.2e}"
    )

print(
    "\nIf rel err stays ~1e-8 while cov_resid grows 4x, the residual is not\n"
    "measuring gradient error and the gate cannot be tightened onto it."
)
