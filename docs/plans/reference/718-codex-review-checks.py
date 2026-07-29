"""Both Codex claims, with the dtype reality accounted for.

The existing test state is REAL (every leaf of y* is float64), so:
  * #719's phase null direction cannot exist there -- a real tensor times
    exp(i.theta) leaves the manifold -- but it should appear for a complex state,
    which is the production regime and which no test covers.
  * #720's per-direction concern is testable on the real state; for real data the
    bond gauge group is orthogonal, so a real antisymmetric generator is the
    right probe.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/yjkao/tenax/tests")

from test_ctm_root_implicit_asym import _gate, _site_tensor  # noqa: E402

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint  # noqa: E402
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

chi = 4
gate = _gate(1.0)


def build(A):
    env, a_arr, _m = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
    root, res = M.asym_root_parametrize(env, a_arr, chi, polish_steps=30)
    rc = M.asym_root_to_covariant_convention(root)
    tilde = M.remove_inverse_roots(rc.env, rc.s)
    return (tilde, rc.u, rc.s, rc.v), rc, a_arr, res


def nrm(t):
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(t))))


def pieces(A, y_star, rc, a_arr):
    template = initialize_ctm_tensor_env(A, chi)

    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_energy(
            A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    A_data = A.todense()
    energy, vjp_e = jax.vjp(energy_of, A_data, y_star[0], y_star[2])
    gd, tb, Sb = vjp_e(jnp.ones((), dtype=energy.dtype))
    y_bar = (
        tb,
        tuple(jnp.zeros_like(x) for x in y_star[1]),
        Sb,
        tuple(jnp.zeros_like(x) for x in y_star[3]),
    )
    return template, A_data, gd, y_bar


# =====================================================================
# (2) per-direction Eq. 88 on the real state
# =====================================================================
A = _site_tensor()
y_star, rc, a_arr, res = build(A)
d2 = a_arr.shape[0]
template, A_data, grad_direct, y_bar = pieces(A, y_star, rc, a_arr)


def Fy(y, a=None, c=None):
    return M.asym_characteristic_residual_covariant(
        y, a_arr if a is None else a, rc if c is None else c, chi
    )


_, vjp_y = jax.vjp(Fy, y_star)
F_bar, sres = _solve_root_adjoint(
    lambda v: vjp_y(v)[0], y_bar, tol=1e-11, maxiter=800, restart=40
)


def F_of_consts(U_star, U_perp, Vh_star, Vh_perp, s_star_inv):
    c = rc._replace(
        U_star=U_star,
        U_perp=U_perp,
        Vh_star=Vh_star,
        Vh_perp=Vh_perp,
        s_star_inv=s_star_inv,
    )
    return M.asym_characteristic_residual_covariant(y_star, a_arr, c, chi)


_, vjp_c = jax.vjp(
    F_of_consts, rc.U_star, rc.U_perp, rc.Vh_star, rc.Vh_perp, rc.s_star_inv
)
Ub, Upb, Vb, Vpb, sib = vjp_c(F_bar)


def pair(g, delta):
    return float(jnp.real(jnp.sum(jnp.conj(g) * delta)))


def dE(Xs):
    KX = [jnp.kron(X, jnp.eye(d2, dtype=X.dtype)) for X in Xs]
    tot = 0.0
    for k in range(4):
        km, kp = (k - 1) % 4, (k + 1) % 4
        Us, Up = rc.U_star[k], rc.U_perp[k]
        Vs, Vp = rc.Vh_star[k], rc.Vh_perp[k]
        si = rc.s_star_inv[k]
        tot += pair(Ub[k], Us @ Xs[k] - KX[km] @ Us)
        tot += pair(Upb[k], -KX[km] @ Up)
        tot += pair(Vb[k], Vs @ KX[kp] - Xs[k] @ Vs)
        tot += pair(Vpb[k], Vp @ KX[kp])
        tot += pair(sib[k], si @ Xs[k] - Xs[k] @ si)
    return tot


print(f"REAL state: root residual {res:.3e}, adjoint solve {float(sres):.3e}")
print(f"  |grad_direct| = {float(jnp.linalg.norm(grad_direct)):.4e}")
print()
print("=== (2) Eq. 88: global generator vs one generator per direction ===")
rng = np.random.RandomState(0)
zero = jnp.zeros((chi, chi))
for trial in range(3):
    G = jnp.asarray(rng.standard_normal((chi, chi)))
    X = 0.5 * (G - G.T)  # real antisymmetric = orthogonal gauge
    X = X / jnp.linalg.norm(X)
    glob = dE([X] * 4)
    per = [dE([X if j == i else zero for j in range(4)]) for i in range(4)]
    print(
        f"  trial {trial}: global {glob:+.3e}   per-direction "
        f"{[f'{v:+.3e}' for v in per]}"
    )

# =====================================================================
# (1) the phase null direction, on a COMPLEX state
# =====================================================================
print()
print("=== (1) global env phase on a COMPLEX state ===")
rngc = np.random.RandomState(7)
base = np.asarray(A.todense())
cdata = base + 0.25j * rngc.standard_normal(base.shape)
cdata = jnp.asarray(cdata / np.linalg.norm(cdata))
Ac = DenseTensor(cdata, A.indices)
yc, rcc, ac_arr, resc = build(Ac)
print(
    f"  complex state: root residual {resc:.3e}"
    f"   leaf dtype {jax.tree.leaves(yc)[0].dtype}"
)


def Fyc(y):
    return M.asym_characteristic_residual_covariant(y, ac_arr, rcc, chi)


print(f"  |F(y*)| = {nrm(Fyc(yc)):.3e}")
phase = (
    M.AsymEnv(*[1j * t for t in yc[0]]),
    tuple(jnp.zeros_like(x) for x in yc[1]),
    tuple(jnp.zeros_like(x) for x in yc[2]),
    tuple(jnp.zeros_like(x) for x in yc[3]),
)
_, jvpc = jax.linearize(Fyc, yc)
print(f"  |dir|              = {nrm(phase):.4e}")
print(f"  |d_yF . phase_dir| = {nrm(jvpc(phase)):.4e}   <- ~0 => exact null direction")

rngd = np.random.RandomState(3)
gen = jax.tree.map(
    lambda x: jnp.asarray(
        rngd.standard_normal(x.shape)
        + (1j * rngd.standard_normal(x.shape) if jnp.iscomplexobj(x) else 0)
    ).astype(x.dtype),
    yc,
)
gen = jax.tree.map(lambda x: x * (nrm(phase) / nrm(gen)), gen)
print(f"  |d_yF . random|    = {nrm(jvpc(gen)):.4e}   (same norm, for scale)")

templatec, Ac_data, gdc, ybc = pieces(Ac, yc, rcc, ac_arr)
inner = float(
    jnp.real(
        sum(
            jnp.sum(jnp.conj(n) * b)
            for n, b in zip(jax.tree.leaves(phase), jax.tree.leaves(ybc))
        )
    )
)
print(
    f"  <null_dir, y_bar>  = {inner:.4e}   (must be ~0 for the solve to be consistent)"
)

print()
print(
    "  dtypes: F(y*) S-block",
    jax.tree.leaves(Fyc(yc))[12].dtype,
    " y_bar S-block",
    jax.tree.leaves(ybc)[12].dtype,
)


# FD check that the energy really does move along the global phase
def energy_phase(theta):
    rot = M.AsymEnv(*[jnp.exp(1j * theta) * t for t in yc[0]])
    A_live = DenseTensor(Ac_data, Ac.indices)
    return float(
        jnp.real(
            M.asym_energy(A_live, M.absorb_inverse_roots(rot, yc[2]), templatec, gate)
        )
    )


for h in (1e-4, 1e-5, 1e-6):
    print(
        f"  FD dE/dtheta h={h:.0e} = "
        f"{(energy_phase(h) - energy_phase(-h)) / (2 * h):+.6e}"
    )
print(f"  E(0) = {energy_phase(0.0):.12f}   E(0.7) = {energy_phase(0.7):.12f}")
try:
    _, vjp_yc = jax.vjp(Fyc, yc)
    Fbc, sresc = _solve_root_adjoint(
        lambda v: vjp_yc(v)[0], ybc, tol=1e-11, maxiter=800, restart=40
    )
except Exception as exc:
    print(f"  ADJOINT SOLVE CRASHES on a complex state: {type(exc).__name__}")
    print(f"    {str(exc)[:160]}")
    raise SystemExit(0)
print(f"  adjoint solve residual = {float(sresc):.3e}")


def a_from_A(a_data):
    A_live = DenseTensor(a_data, Ac.indices)
    a_t = M._build_double_layer_tensor(A_live)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    return a_t.transpose(perm).todense()


_, vjp_pc = jax.vjp(
    lambda ad: M.asym_characteristic_residual_covariant(yc, a_from_A(ad), rcc, chi),
    Ac_data,
)
g0 = gdc - vjp_pc(Fbc)[0]
print(f"  |grad| = {float(jnp.linalg.norm(g0)):.4e}")
for alpha in (1.0, 10.0):
    Fb2 = jax.tree.map(lambda b, nn: b + alpha * nn, Fbc, phase)
    g2 = gdc - vjp_pc(Fb2)[0]
    print(
        f"  alpha={alpha:5.1f}: |grad change| = "
        f"{float(jnp.linalg.norm(g2 - g0)):.4e}"
        f"  rel {float(jnp.linalg.norm(g2 - g0) / jnp.linalg.norm(g0)):.3e}"
    )
