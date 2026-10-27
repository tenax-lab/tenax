"""Throwaway diagnostic 2 for #1153's macOS-only stall (not for merge).

Captures the flow attempt's joint adjoint solve (#1153's
``_jit_joint_adjoint_solve``) on the #1028 pair fixture, then: prints its
outputs, re-measures the true residual of the returned lambda, checks that the
saved pullback is linear and repeatable, re-solves with other settings, and
saves the pullback's leaves so the operator can be replayed elsewhere.
Usage: diag_1153b.py OUT.npz [REPLAY.npz]
"""

import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_energy_ad as cea
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, heisenberg_u1sz_init_pair
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

OUT = sys.argv[1]
REPLAY = sys.argv[2] if len(sys.argv) > 2 else None
G2 = heisenberg_gate().todense()
CALLS = []
_real_fv = cea._fusion_variant


def spy_fv(prog, params_data_tuple, *, by_bond_dim):
    fn = _real_fv(prog, params_data_tuple, by_bond_dim=by_bond_dim)
    if getattr(prog, "__name__", "") != "_jit_joint_adjoint_solve":
        return fn

    def wrapped(*args, **kw):
        out = fn(*args, **kw)
        CALLS.append({"fn": fn, "args": args, "kw": kw, "out": out})
        return out

    return wrapped


cea._fusion_variant = spy_fv


def params():
    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    return (jnp.asarray(A.todense()), jnp.asarray(B.todense()))


def energy(p):
    return jnp.real(cea.ctm_energy_implicit(
        {(0, 0): _wrap_as_dense_tensor(p[0]), (1, 0): _wrap_as_dense_tensor(p[1])},
        CHECKERBOARD_NEIGHBORS, G2, chi=8, max_iter=20, projector_method="svd",
        forward_gauge="bond_phase", conv_method="elementwise",
        projector_backward="flow"))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    jax.grad(energy)(params())
d = cea.get_last_implicit_ad_diagnostics()
print("used", d.get("projector_backward_used"), "| flow_rel", d.get("flow_adjoint_residual"))
print("joint-solve calls:", [(c["kw"]["neumann"], c["kw"]["tol"]) for c in CALLS])
flow = CALLS[0]
fn, (vjp, rhs, init, use_init), kw = flow["fn"], flow["args"], flow["kw"]

if REPLAY:
    z = np.load(REPLAY)
    leaves, treedef = jax.tree.flatten(vjp)
    assert len(leaves) == int(z["n_vjp"]), (len(leaves), int(z["n_vjp"]))
    vjp = jax.tree.unflatten(treedef, [jnp.asarray(z[f"vjp{i}"]) for i in range(len(leaves))])
    rhs = tuple(jnp.asarray(z[f"rhs{i}"]) for i in range(len(rhs)))
    print(f"REPLAYING the operator from {REPLAY}")
    flow["out"] = fn(vjp, rhs, init, use_init, **kw)
else:
    leaves = jax.tree.leaves(vjp)
    np.savez(OUT, n_vjp=len(leaves), **{f"vjp{i}": np.asarray(x) for i, x in enumerate(leaves)},
             **{f"rhs{i}": np.asarray(x) for i, x in enumerate(rhs)})
    print("saved", OUT, "vjp leaves", len(leaves), "bytes", sum(np.asarray(x).nbytes for x in leaves))


def nrm(t):
    return float(sum(jnp.vdot(x, x).real for x in jax.tree.leaves(t))) ** 0.5


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def jt(v):
    return tuple(vjp(tuple(v))[1])


jt_jit = jax.jit(lambda f, v: tuple(f(tuple(v))[1]))


def true_resid(lam, op=jt):
    return nrm(sub(rhs, sub(lam, op(lam))))


n_div, n_conv, n_k, n_lam, n_res, lam, abs_res, b_norm, _aux = flow["out"]
lam = tuple(lam)
print(f"flow solve: neumann div={bool(n_div)} conv={bool(n_conv)} n_k={int(n_k)} "
      f"n_resid={float(n_res)!r}")
print(f"  reported abs={float(abs_res)!r} |b|={float(b_norm)!r} |lam|={nrm(lam)!r} "
      f"|lam_neumann|={nrm(n_lam)!r}")
print(f"  true resid eager={true_resid(lam)!r} jit={true_resid(lam, lambda v: jt_jit(vjp, v))!r}")

key = jax.random.PRNGKey(7)
v1 = tuple(jax.random.normal(jax.random.fold_in(key, i), x.shape, x.dtype) for i, x in enumerate(rhs))
v2 = tuple(jax.random.normal(jax.random.fold_in(key, 100 + i), x.shape, x.dtype) for i, x in enumerate(rhs))
for name, v in (("lam", lam), ("rand", v1)):
    a, b = jt(v), jt(v)
    c = jt_jit(vjp, v)
    s = jt(tuple(1e3 * x for x in v))
    print(f"  J^T on {name}: |v|={nrm(v):.6e} |J^T v|/|v|={nrm(a) / nrm(v):.6e} "
          f"repeat={nrm(sub(a, b)) / nrm(a):.3e} eager-vs-jit={nrm(sub(a, c)) / nrm(a):.3e} "
          f"scale={nrm(sub(s, tuple(1e3 * x for x in a))) / nrm(s):.3e}")
s12 = jt(tuple(x + y for x, y in zip(v1, v2)))
print(f"  additivity={nrm(sub(s12, tuple(x + y for x, y in zip(jt(v1), jt(v2))))) / nrm(s12):.3e}")
print(f"  eps*|A|*|lam| ~ {np.finfo(np.float64).eps * nrm(lam) * (1 + nrm(jt(v1)) / nrm(v1)):.3e} (|A| from the random v)")

for label, kw2 in (
    ("neumann tol1e-8", {**kw, "tol": 1e-8}),
    ("neumann restart60", {**kw, "restart": 60}),
    ("neumann restart10", {**kw, "restart": 10}),
    ("gmres-only tol1e-6", {**kw, "neumann": False}),
    ("gmres-only tol1e-8", {**kw, "neumann": False, "tol": 1e-8}),
):
    o = fn(vjp, rhs, init if kw2["neumann"] else tuple(jnp.zeros_like(x) for x in rhs),
           use_init, **kw2)
    lam2 = tuple(o[5])
    print(f"  re-solve {label}: reported abs={float(o[6]):.6e} true={true_resid(lam2):.6e} "
          f"|lam|={nrm(lam2):.6e} n_k={int(o[2])}")

from jax.scipy.sparse.linalg import gmres

x, _ = gmres(lambda v: sub(v, jt(v)), rhs, tol=1e-12, atol=1e-12, restart=30, maxiter=40,
             solve_method="incremental")
print(f"  jax.scipy gmres (eager matvec, tol 1e-12): true={true_resid(tuple(x)):.6e} |x|={nrm(x):.6e}")
sys.stdout.flush()
