"""#772: what tolerance does `root_residual_warn` need against the #778 rank cap?

Both `root_residual` and `covariant_residual` are compared against the single
`root_residual_warn` knob inside `asym_root_implicit_energy_and_grad`, and
`on_root_residual` defaults to "raise".  #778 reports a forward root residual of
2.8e-06 against a 1e-06 default, but does not break the two quantities out per
chi.  This does.

Run: JAX_PLATFORMS=cpu uv run python scripts/probe_772_tolerance.py
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, "tests")

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from _su_fixtures import physical_su_d2  # noqa: E402

eps13 = float(np.finfo(np.float64).eps ** (1.0 / 3.0))
print(f"main's default clamp: eps^(1/3) = {eps13:.4e}\n")


def _gate(delta=1.0):
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


A = physical_su_d2()
H = _gate()

hdr = f"{'state':>10} {'chi':>4} {'root_resid':>12} {'cov_resid':>12} {'max|g|':>12}"
print(hdr)
print("-" * len(hdr))

worst_cov = 0.0
for label, state in (("physical", A), ("random", None)):
    for chi in (4, 6, 8, 12):
        if state is None:
            from test_ctm_root_implicit_asym import _site_tensor

            state_ = _site_tensor(D=2)
        else:
            state_ = state
        try:
            _E, g, d = M.asym_root_implicit_energy_and_grad(
                state_,
                H,
                chi=chi,
                max_iter=300,
                conv_tol=1e-13,
                return_diagnostics=True,
                on_root_residual="warn",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{label:>10} {chi:>4}  ERROR {type(exc).__name__}: {exc}")
            continue
        gmax = float(np.abs(np.asarray(g)).max())
        print(
            f"{label:>10} {chi:>4} {d['root_residual']:>12.3e} "
            f"{d['covariant_residual']:>12.3e} {gmax:>12.3e}"
        )
        if label == "physical":
            worst_cov = max(worst_cov, float(d["covariant_residual"]))

print(f"\nworst physical covariant_residual = {worst_cov:.3e}")
print(f"current root_residual_warn default = 1.0e-06 -> {'PASS' if worst_cov < 1e-6 else 'REJECTS'}")
