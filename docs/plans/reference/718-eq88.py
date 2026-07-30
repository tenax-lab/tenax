"""Is Eq. 88's freezing of U*, V* licensed now that the §V.3 package is in?

The implicit gradient treats c = (U*, U_perp, Vh*, Vh_perp, s*inv) as constants.
The error it therefore commits is

    dE/dc . dc/dp ,   with   dE/dc = -Fbar . d_c F

using the *same* Fbar the gradient already solved for.  So the size of
`Fbar . d_c F` is a direct, assembly-independent measure of whether Eq. 88's
null-space restriction is discarding a gauge (norm ~ 0) or something real.
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

env, a_arr, _meta = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
root, res = M.asym_root_parametrize(env, a_arr, chi, polish_steps=30)
print("forward root residual  =", res)

root_cov = M.asym_root_to_covariant_convention(root)
S_star = root_cov.s
tilde = M.remove_inverse_roots(root_cov.env, S_star)
y_star = (tilde, root_cov.u, S_star, root_cov.v)

template = initialize_ctm_tensor_env(A, chi)


def energy_of(a_data, env_tilde, S_all):
    A_live = DenseTensor(a_data, A.indices)
    return M.asym_energy(
        A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
    )


energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S_star)
grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
y_bar = (
    tilde_bar,
    tuple(jnp.zeros_like(x) for x in root_cov.u),
    S_bar,
    tuple(jnp.zeros_like(x) for x in root_cov.v),
)


def F_of_y(y):
    return M.asym_characteristic_residual_covariant(y, a_arr, root_cov, chi)


F0, vjp_y = jax.vjp(F_of_y, y_star)
print(
    "covariant residual     =",
    float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(F0)))),
)

F_bar, solve_resid = _solve_root_adjoint(
    lambda v: vjp_y(v)[0], y_bar, tol=1e-10, maxiter=600, restart=40
)
print("adjoint solve residual =", float(solve_resid))


# ---- cotangents on the frozen constants ----
def F_of_consts(U_star, U_perp, Vh_star, Vh_perp, s_star_inv):
    c = root_cov._replace(
        U_star=U_star,
        U_perp=U_perp,
        Vh_star=Vh_star,
        Vh_perp=Vh_perp,
        s_star_inv=s_star_inv,
    )
    return M.asym_characteristic_residual_covariant(y_star, a_arr, c, chi)


_, vjp_c = jax.vjp(
    F_of_consts,
    root_cov.U_star,
    root_cov.U_perp,
    root_cov.Vh_star,
    root_cov.Vh_perp,
    root_cov.s_star_inv,
)
cots = vjp_c(F_bar)
names = ("U_star", "U_perp", "Vh_star", "Vh_perp", "s_star_inv")


# scale: the same Fbar contracted with d_p F, i.e. the adjoint part of the grad
def F_of_p(a_data):
    A_live = DenseTensor(a_data, A.indices)
    a_t = M._build_double_layer_tensor(A_live)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    return M.asym_characteristic_residual_covariant(
        y_star, a_t.transpose(perm).todense(), root_cov, chi
    )


_, vjp_p = jax.vjp(F_of_p, A_data)
adj_part = vjp_p(F_bar)[0]
grad = grad_direct - adj_part

print()
print(f"|grad_direct|          = {float(jnp.linalg.norm(grad_direct)):.6e}")
print(f"|Fbar . d_p F|         = {float(jnp.linalg.norm(adj_part)):.6e}")
print(f"|grad|                 = {float(jnp.linalg.norm(grad)):.6e}")
print()
print("=== Eq. 88 check: |Fbar . d_c F| for each frozen constant ===")
for name, ct in zip(names, cots):
    nrm = float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in ct)))
    print(
        f"  {name:11s} {nrm:.6e}   ({nrm / float(jnp.linalg.norm(grad)):.3e} of |grad|)"
    )

# ---------------------------------------------------------------------------
# Eq. 88 only needs dE/dc to vanish along the GAUGE directions: the in-subspace
# rotations dU* = U* X and dVh* = X Vh*.  Any component orthogonal to those is
# harmless, because the parametrisation U = U* + U_perp u already carries every
# non-gauge variation of the retained subspace.
# ---------------------------------------------------------------------------
print()
print("=== gauge vs non-gauge split of the cotangents ===")
U_bar, _Up_bar, Vh_bar, _Vp_bar, _s_bar = cots
for k in range(4):
    Us, Vhs = root_cov.U_star[k], root_cov.Vh_star[k]
    Ub, Vb = U_bar[k], Vh_bar[k]

    # gauge part of the U* cotangent: <Ubar, U* X> = <U*^H Ubar, X>
    g_U = Us.conj().T @ Ub
    # non-gauge remainder
    n_U = Ub - Us @ g_U

    # gauge part of the Vh* cotangent: <Vbar, X Vh*> = <Vbar Vh*^H, X>
    g_V = Vb @ Vhs.conj().T
    n_V = Vb - g_V @ Vhs

    print(
        f"  k={k}  U*:  gauge {float(jnp.linalg.norm(g_U)):.4e}"
        f"   non-gauge {float(jnp.linalg.norm(n_U)):.4e}"
        f"   total {float(jnp.linalg.norm(Ub)):.4e}"
    )
    print(
        f"        Vh*: gauge {float(jnp.linalg.norm(g_V)):.4e}"
        f"   non-gauge {float(jnp.linalg.norm(n_V)):.4e}"
        f"   total {float(jnp.linalg.norm(Vb)):.4e}"
    )

# ---------------------------------------------------------------------------
# The gauge acts on ALL of c at once, so projecting U* and Vh* separately is
# only valid if their generators are independent.  They may not be: Vd[k] takes
# its Eq. 73 root from direction k+1, so the bond index that U*_k pairs with
# need not be the one Vh*_k pairs with.  If the generators are linked, the two
# cotangents must be summed before the gauge condition is read off, and the sum
# can be far smaller than either term.
# ---------------------------------------------------------------------------
print()
print("=== paired gauge condition: |gU[k] (+/-) gV[j]^H| over offsets ===")
gU, gV = {}, {}
for k in range(4):
    Us, Vhs = root_cov.U_star[k], root_cov.Vh_star[k]
    gU[k] = Us.conj().T @ U_bar[k]
    gV[k] = Vh_bar[k] @ Vhs.conj().T

for off in (0, 1, 3):
    for sign in (+1, -1):
        for adj in (False, True):
            worst = 0.0
            for k in range(4):
                b = gV[(k + off) % 4]
                b = b.conj().T if adj else b
                worst = max(worst, float(jnp.linalg.norm(gU[k] + sign * b)))
            base = max(float(jnp.linalg.norm(gU[k])) for k in range(4))
            print(
                f"  off={off} sign={sign:+d} adj={adj!s:5s}: worst = {worst:.4e}"
                f"   (vs |gU| alone {base:.4e})"
            )
