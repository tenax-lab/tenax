"""Does the frozen-c root track the true CTMRG fixed point?

Every internal check passes, and the explicit reference is confirmed correct by
FD of its own map (9 digits).  So the question left is whether

    G(p) = E( y  such that  F(y; c, p) = 0 ),   c frozen at p0

is the same function as the true fixed-point energy.  Re-running
converge+parametrize at p +/- h cannot answer this: that route is
gauge-discontinuous (|ydot_fd| blows up as h shrinks).  Solving F by Newton from
y* with c frozen is gauge-stable, so it isolates the modelling assumption from
the gauge noise.

If FD(G) matches the implicit gradient, the implicit path differentiates its own
function correctly and G != E_true -- i.e. the frozen-c root does not track the
fixed point.  If FD(G) matches the explicit gradient instead, the bug is in the
gradient assembly after all.
"""

import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/yjkao/tenax/tests")

from test_ctm_root_implicit_asym import _gate, _site_tensor  # noqa: E402

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint  # noqa: E402
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

A = _site_tensor()
chi = 4
gate = _gate(1.0)
A_data = A.todense()
template = initialize_ctm_tensor_env(A, chi)


def a_from_A(a_data):
    A_live = DenseTensor(a_data, A.indices)
    a_t = M._build_double_layer_tensor(A_live)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    return a_t.transpose(perm).todense()


env, a_arr, _m = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
root, res0 = M.asym_root_parametrize(env, a_arr, chi, polish_steps=30)
root_cov = M.asym_root_to_covariant_convention(root)
S_star = root_cov.s
tilde = M.remove_inverse_roots(root_cov.env, S_star)
y_star = (tilde, root_cov.u, S_star, root_cov.v)
print(f"root residual at p0: {res0:.3e}")


def nrm(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(tree))))


def newton_root(a_data, y_init, iters=10, tol=1e-13):
    """Solve F(y; c frozen, p) = 0 by Newton from y_init."""
    a_live = a_from_A(a_data)

    def Fy(y):
        return M.asym_characteristic_residual_covariant(y, a_live, root_cov, chi)

    y = y_init
    for _ in range(iters):
        F0 = Fy(y)
        r = nrm(F0)
        if r < tol:
            break
        _, jvp = jax.linearize(Fy, y)
        dy, _resid = _solve_root_adjoint(
            jvp, jax.tree.map(lambda x: -x, F0), tol=1e-12, maxiter=800, restart=40
        )
        y = jax.tree.map(lambda b, c: b + c, y, dy)
    return y, nrm(Fy(y))


def energy_at(a_data, y):
    A_live = DenseTensor(a_data, A.indices)
    return float(
        M.asym_energy(A_live, M.absorb_inverse_roots(y[0], y[2]), template, gate)
    )


key = jax.random.PRNGKey(21)
dA = jax.random.normal(key, A_data.shape, dtype=A_data.dtype)
dA = dA / jnp.linalg.norm(dA)

y0, r0 = newton_root(A_data, y_star)
print(
    f"newton at p0: |F| = {r0:.3e}   |y0 - y*| = {nrm(jax.tree.map(lambda b, c: b - c, y0, y_star)):.3e}"
)
print(f"E(p0) = {energy_at(A_data, y0):.12f}")

print()
print("=== FD of G(p) = E at the frozen-c root ===")
for h in (1e-4, 1e-5, 1e-6):
    yp, rp = newton_root(A_data + h * dA, y_star)
    ym, rm = newton_root(A_data - h * dA, y_star)
    ep, em = energy_at(A_data + h * dA, yp), energy_at(A_data - h * dA, ym)
    print(
        f"  h={h:.0e}  |F|+/- = {rp:.1e}/{rm:.1e}   FD(G) = {(ep - em) / (2 * h):+.10f}"
    )

print()
print("  implicit (IFT)           = +0.3519056655")
print("  explicit backprop + FD   = +0.3607559898")
