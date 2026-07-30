"""Where does the per-bond gauge antisymmetry (+/-2.121e-3 on bonds 2,3) come from?

Two FINITE-transform tests, each decisive on its own and independent of the
cotangent pairing that produced the antisymmetry:

  A.  Is F covariant under a gauge on ONE bond?   ||F(g.y, g.c)|| should stay ~1e-12.
  B.  Is E invariant under a gauge on ONE bond?   E(g.env, g.S) should be unchanged.

Bond j lives on four env legs (C_j right, T_j both, C_{j+1} left), so

    C_{k+1} -> W_{k-1}' C_{k+1} W_k
    T_{k+1} -> W_k' T_{k+1} W_k
    S_k     -> W_k' S_k W_k
    U*_k    -> kron(W_{k-1},I)' U*_k W_k        U_perp_k  -> kron(W_{k-1},I)' U_perp_k
    Vh*_k   -> W_k' Vh*_k kron(W_{k+1},I)       Vh_perp_k -> Vh_perp_k kron(W_{k+1},I)
    s*inv_k -> W_k' s*inv_k W_k

If A holds per bond but B fails on exactly bonds 2 and 3 with opposite signs, the
bug is a 2<->3 swap in the energy's leg->bond assignment, not in F.  If A fails,
it is in the residual's assignment.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/yjkao/tenax/tests")

from test_ctm_root_implicit_asym import _gate, _site_tensor  # noqa: E402

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402

A = _site_tensor()
chi = 4
gate = _gate(1.0)
A_data = A.todense()
template = initialize_ctm_tensor_env(A, chi)

env, a_arr, _m = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
root, res0 = M.asym_root_parametrize(env, a_arr, chi, polish_steps=30)
rc = M.asym_root_to_covariant_convention(root)
S_star = rc.s
tilde = M.remove_inverse_roots(rc.env, S_star)
y_star = (tilde, rc.u, S_star, rc.v)
d2 = a_arr.shape[0]
n = chi * d2
print(f"root residual {res0:.3e}")


def nrm(t):
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(t))))


def gauge_env(env_t, S, Ws):
    """Apply the per-bond gauge to (env_tilde, S)."""
    corners, edges = [], []
    for k in range(4):
        km = (k - 1) % 4
        C = Ws[km].conj().T @ getattr(env_t, f"C{k + 1}") @ Ws[k]
        E = jnp.einsum(
            "ai,ixj,jb->axb", Ws[k].conj().T, getattr(env_t, f"T{k + 1}"), Ws[k]
        )
        corners.append(C)
        edges.append(E)
    S_g = tuple(Ws[k].conj().T @ S[k] @ Ws[k] for k in range(4))
    return M.AsymEnv(*corners, *edges), S_g


def gauge_consts(c, Ws):
    KW = [jnp.kron(W, jnp.eye(d2, dtype=W.dtype)) for W in Ws]
    Us, Up, Vs, Vp, si = [], [], [], [], []
    for k in range(4):
        km, kp = (k - 1) % 4, (k + 1) % 4
        Us.append(KW[km].conj().T @ c.U_star[k] @ Ws[k])
        Up.append(KW[km].conj().T @ c.U_perp[k])
        Vs.append(Ws[k].conj().T @ c.Vh_star[k] @ KW[kp])
        Vp.append(c.Vh_perp[k] @ KW[kp])
        si.append(Ws[k].conj().T @ c.s_star_inv[k] @ Ws[k])
    return c._replace(
        U_star=tuple(Us),
        U_perp=tuple(Up),
        Vh_star=tuple(Vs),
        Vh_perp=tuple(Vp),
        s_star_inv=tuple(si),
    )


def energy_of(env_t, S):
    return float(
        jnp.real(M.asym_energy(A, M.absorb_inverse_roots(env_t, S), template, gate))
    )


E0 = energy_of(tilde, S_star)
F0 = nrm(M.asym_characteristic_residual_covariant(y_star, a_arr, rc, chi))
print(f"E0 = {E0:.12f}   |F(y*)| = {F0:.3e}")

rng = np.random.RandomState(0)
G = rng.standard_normal((chi, chi))
X = 0.5 * (G - G.T)
X = X / np.linalg.norm(X)
eye = jnp.eye(chi)


def Ws_for(active, t):
    W = jnp.asarray(expm(t * X))
    return [W if j in active else eye for j in range(4)]


print()
print("=== A: is F still a root after a gauge on a SINGLE bond? ===")
for t in (0.1, 0.5):
    for active in ([0], [1], [2], [3], [0, 1, 2, 3]):
        Ws = Ws_for(active, t)
        env_g, S_g = gauge_env(tilde, S_star, Ws)
        c_g = gauge_consts(rc, Ws)
        y_g = (env_g, rc.u, S_g, rc.v)
        r = nrm(M.asym_characteristic_residual_covariant(y_g, a_arr, c_g, chi))
        print(f"  t={t:.1f} bonds {str(active):12s} |F| = {r:.3e}")

print()
print("=== B: is E invariant under a gauge on a SINGLE bond? ===")
for t in (0.1, 0.5):
    for active in ([0], [1], [2], [3], [0, 1, 2, 3]):
        Ws = Ws_for(active, t)
        env_g, S_g = gauge_env(tilde, S_star, Ws)
        Eg = energy_of(env_g, S_g)
        print(
            f"  t={t:.1f} bonds {str(active):12s} E = {Eg:+.12f}   dE = {Eg - E0:+.3e}"
        )

print()
print("=== B': E invariance with the gauge on the env only (S left alone) ===")
for active in ([0], [1], [2], [3], [0, 1, 2, 3]):
    Ws = Ws_for(active, 0.1)
    env_g, _ = gauge_env(tilde, S_star, Ws)
    Eg = energy_of(env_g, S_star)
    print(f"  bonds {str(active):12s} dE = {Eg - E0:+.3e}")
