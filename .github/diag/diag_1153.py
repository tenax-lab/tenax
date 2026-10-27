"""Throwaway diagnostic for #1153's macOS-only failure (not for merge).

Runs the #1028 pair fixture of tests/test_flow_slope_check_1111.py and prints,
for every adjoint convergence check, the absolute residual, |b| and tol, plus
the diagnostics at that moment; and whether any gesdd result was non-finite.
"""

import platform
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy

import tenax
import tenax.algorithms._ctm_energy_ad as cea
import tenax.linalg as tl
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, heisenberg_u1sz_init_pair
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

LABEL = sys.argv[1]
print(f"=== {LABEL}: tenax from {tenax.__file__}")
print("python", sys.version.split()[0], "jax", jax.__version__, "numpy",
      np.__version__, "scipy", scipy.__version__, "machine", platform.machine(),
      "mac_ver", platform.mac_ver()[0])
try:
    print("cpu", subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True).stdout.strip())
except Exception as e:  # noqa: BLE001
    print("cpu ?", e)

G2 = heisenberg_gate().todense()
CHECKS = []
SVD = {"calls": 0, "nonfinite": 0}

_real_conv = cea._adjoint_converged


def spy_conv(residual, b_norm, tol):
    ok = _real_conv(residual, b_norm, tol)
    d = cea._F3_LAST_DIAGNOSTICS
    CHECKS.append((float(residual), float(b_norm), tol, ok,
                   d.get("n_iter"), d.get("diverged")))
    return ok


cea._adjoint_converged = spy_conv

_real_gesdd = tl._gesdd


def _note(finite):
    SVD["calls"] += 1
    SVD["nonfinite"] += int(not bool(finite))


def spy_gesdd(m, full_matrices, compute_uv):
    r = _real_gesdd(m, full_matrices, compute_uv)
    leaves = r if isinstance(r, tuple) else (r,)
    fin = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    jax.debug.callback(_note, fin)
    return r


tl._gesdd = spy_gesdd


def params():
    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    return (jnp.asarray(A.todense()), jnp.asarray(B.todense()))


def energy(p, **kw):
    return jnp.real(cea.ctm_energy_implicit(
        {(0, 0): _wrap_as_dense_tensor(p[0]), (1, 0): _wrap_as_dense_tensor(p[1])},
        CHECKERBOARD_NEIGHBORS, G2, chi=8, max_iter=20, projector_method="svd",
        forward_gauge="bond_phase", conv_method="elementwise",
        projector_backward="flow", **kw))


for kw in ({}, {"gmres_maxiter": 800}, {"gmres_tol": 1e-7}, {"gmres_tol": 1e-8}):
    jax.clear_caches()
    CHECKS.clear()
    SVD.update(calls=0, nonfinite=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e, g = jax.value_and_grad(lambda p: energy(p, **kw))(params())
    d = dict(cea.get_last_implicit_ad_diagnostics())
    print(f"--- {LABEL} {kw or 'defaults'}")
    print(f"  E={float(e)!r} |g|={float(sum(jnp.vdot(x, x).real for x in g))**0.5!r}")
    print("  fwd stat", d.get("forward_stationarity_residual"), "sv_diff",
          d.get("forward_sv_diff"))
    print("  used", d.get("projector_backward_used"), "| flow_rel",
          d.get("flow_adjoint_residual"), "| ratio", d.get("flow_slope_ratio"))
    print("  reason", (d.get("flow_fallback_reason") or "")[:90])
    for i, (r, b, tol, ok, n, dv) in enumerate(CHECKS):
        print(f"  check{i}: abs={r!r} |b|={b!r} rel={r / b if b else r!r} "
              f"tol={tol} ok={ok} n_iter={n} diverged={dv}")
    print("  gesdd calls", SVD["calls"], "non-finite", SVD["nonfinite"], flush=True)
