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

from _su_fixtures import physical_su_d2  # noqa: E402

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402

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

hdr = (
    f"{'state':>10} {'chi':>4} {'rank':>5} {'root_resid':>12} "
    f"{'cov_resid':>12} {'max|g|':>12} {'gate':>12} {'verdict':>8}"
)
print(hdr)
print("-" * len(hdr))

# The gate is *derived*, not a constant: `_root_residual_tolerance` relaxes the
# base tolerance only where the rank clamp actually bound, in proportion to how
# many directions it clamped.  Reading a flat 1e-6 here would report REJECTS for
# rows the production default accepts, which is the failure this probe exists to
# measure -- so resolve it exactly the way the library does.
BASE_TOL = 1e-6
worst = None
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
        rank = int(d["usable_rank"])
        gate = M._root_residual_tolerance(BASE_TOL, rank, chi, jnp.complex128)
        cov = float(d["covariant_residual"])
        worst_resid = max(float(d["root_residual"]), cov)
        ok = worst_resid <= gate
        print(
            f"{label:>10} {chi:>4} {rank:>5} {d['root_residual']:>12.3e} "
            f"{cov:>12.3e} {gmax:>12.3e} {gate:>12.3e} "
            f"{'PASS' if ok else 'REJECTS':>8}"
        )
        if label == "physical" and (worst is None or worst_resid / gate > worst[0]):
            worst = (worst_resid / gate, chi, rank, worst_resid, gate)

if worst is not None:
    margin, chi, rank, resid, gate = worst
    print(
        f"\ntightest physical margin: chi={chi} rank={rank} "
        f"residual {resid:.3e} against the resolved gate {gate:.3e} "
        f"({margin:.2f}x) -> {'PASS' if margin <= 1.0 else 'REJECTS'}"
    )
    print(
        f"a flat {BASE_TOL:.1e} would have called this "
        f"{'PASS' if resid <= BASE_TOL else 'REJECTS'} -- which is the #772 "
        "misreport this probe exists to catch."
    )
