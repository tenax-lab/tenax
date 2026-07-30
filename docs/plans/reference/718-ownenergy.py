"""Energy contracted directly in the MODULE's own convention, as ground truth.

The module's env is a genuine CTM fixed point of its own sweep (F vanishes at
1e-16), so contracting the 2x1 RDM with the module's own leg convention gives
the energy that env actually represents.  Then find which relabelling of the
env, if any, makes ``asym_energy`` (the production RDM) reproduce it.

Module convention ring, read off the sweep:
    C1.1-T1.0  T1.2-C2.0  C2.1-T2.0  T2.2-C3.0
    C3.1-T3.0  T3.2-C4.0  C4.1-T4.0  T4.2-C1.0

Production convention, from the ``_rdm2x1_tensor`` docstring:
    C1[0]-T4[0]  C1[1]-T1[0]  T1[2]-C2[0]  C2[1]-T2[0]
    T2[2]-C3[0]  C3[1]-T3[2]  T3[0]-C4[1]  C4[0]-T4[2]
i.e. the same ring with C4 transposed and T3, T4 reversed.
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
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    _build_double_layer_open_tensor,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402

A = _site_tensor()
chi = 4
gate = _gate(1.0)
template = initialize_ctm_tensor_env(A, chi)
env, a_arr, meta = M.converge(A, chi, max_iter=400, conv_tol=1e-13)

ao_t = _build_double_layer_open_tensor(A)
lab = list(ao_t.labels())
perm = tuple(lab.index(x) for x in ("u2", "d2", "l2", "r2", "phys", "phys_bra"))
ao = jnp.asarray(ao_t.transpose(perm).todense())


def rdm_h(e, ao_o):
    """2x1 RDM, module convention.  Returns (s1_ket, s2_ket, s1_bra, s2_bra).

    Ring: C1[a,b] T1_L[b,u,c] T1_R[c,U,e] C2[e,f] T2[f,r,g] C3[g,h]
          T3_R[h,D,i] T3_L[i,d,j] C4[j,k] T4[k,l,a]
    Sites: ao_L[u,d,l,m,s,S], ao_R[U,D,m,r,t,T]  (m is the shared horizontal bond)
    """
    UL = jnp.einsum("ab,buc->auc", e.C1, e.T1)
    LL = jnp.einsum("jk,idj->kid", e.C4, e.T3)
    Lenv = jnp.einsum("auc,kla,kid->ucldi", UL, e.T4, LL)
    Lao = jnp.einsum("ucldi,udlmsS->cimsS", Lenv, ao_o)

    UR = jnp.einsum("cUe,ef->cUf", e.T1, e.C2)
    LR = jnp.einsum("gh,hDi->gDi", e.C3, e.T3)
    Renv = jnp.einsum("cUf,frg,gDi->cUrDi", UR, e.T2, LR)
    Rao = jnp.einsum("cUrDi,UDmrtT->cimtT", Renv, ao_o)

    return jnp.einsum("cimsS,cimtT->stST", Lao, Rao)


def energy_from_rdm(rdm, H):
    d = rdm.shape[0]
    mat = rdm.reshape(d * d, d * d)
    mat = 0.5 * (mat + mat.conj().T)
    mat = mat / jnp.trace(mat)
    return jnp.einsum("ijkl,ijkl->", mat.reshape(d, d, d, d), H)


def rotate_ao(x):
    return jnp.transpose(x, (3, 2, 0, 1, 4, 5))


def own_energy(e):
    E_h = energy_from_rdm(rdm_h(e, ao), gate)
    E_v = energy_from_rdm(rdm_h(M.rotate_env(e), rotate_ao(ao)), gate)
    return float(jnp.real(E_h + E_v)), float(jnp.real(E_h)), float(jnp.real(E_v))


E_own, E_h, E_v = own_energy(env)
print(f"module-convention energy (own contraction) = {E_own:.12f}")
print(f"  E_h = {E_h:+.12f}   E_v = {E_v:+.12f}")

variants = {
    "as-is": env,
    "C4 transposed": env._replace(C4=env.C4.T),
    "T3,T4 reversed": env._replace(
        T3=jnp.transpose(env.T3, (2, 1, 0)), T4=jnp.transpose(env.T4, (2, 1, 0))
    ),
    "C4.T + T3,T4 reversed": env._replace(
        C4=env.C4.T,
        T3=jnp.transpose(env.T3, (2, 1, 0)),
        T4=jnp.transpose(env.T4, (2, 1, 0)),
    ),
}
print("\n=== asym_energy (production RDM) on relabelled envs ===")
for tag, e in variants.items():
    val = float(jnp.real(M.asym_energy(A, e, template, gate)))
    print(f"  {tag:24s} E = {val:.12f}   diff vs own = {val - E_own:+.3e}")

rng = np.random.RandomState(0)
G = rng.standard_normal((chi, chi))
X = 0.5 * (G - G.T) / np.linalg.norm(0.5 * (G - G.T))
eye = jnp.eye(chi)


def gauge_module(e, Ws):
    corners, edges = [], []
    for k in range(4):
        km = (k - 1) % 4
        corners.append(Ws[km].conj().T @ getattr(e, f"C{k + 1}") @ Ws[k])
        edges.append(
            jnp.einsum("ai,ixj,jb->axb", Ws[k].conj().T, getattr(e, f"T{k + 1}"), Ws[k])
        )
    return M.AsymEnv(*corners, *edges)


print("\n=== per-bond gauge invariance of the MODULE-convention energy ===")
for active in ([0], [1], [2], [3], [0, 1, 2, 3]):
    W = jnp.asarray(expm(0.1 * X))
    Ws = [W if j in active else eye for j in range(4)]
    print(
        f"  bonds {str(active):12s} dE = {own_energy(gauge_module(env, Ws))[0] - E_own:+.3e}"
    )

# --------------------------------------------------------------------------
# Intrinsic check: a correctly glued network gives a positive semidefinite RDM.
# A mis-glued one generally does not, and needs no external oracle to detect.
# --------------------------------------------------------------------------
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    _rdm1x2_tensor,
    _rdm2x1_tensor,
)

print("\n=== RDM spectra: is the contracted network a valid density matrix? ===")


def spec(rdm):
    d = rdm.shape[0]
    m = rdm.reshape(d * d, d * d)
    m = 0.5 * (m + m.conj().T)
    m = m / jnp.trace(m)
    return np.sort(np.linalg.eigvalsh(np.asarray(m)))


print(f"  module convention (own)      h {spec(rdm_h(env, ao))}")
print(
    f"  module convention (own)      v {spec(rdm_h(M.rotate_env(env), rotate_ao(ao)))}"
)
for tag, e in variants.items():
    ce = M._to_ctm_env(e, template)
    print(f"  production RDM, {tag:22s} h {spec(_rdm2x1_tensor(A, ce))}")
    print(f"  production RDM, {tag:22s} v {spec(_rdm1x2_tensor(A, ce))}")

print("\n=== E_h and E_v separately ===")
for tag, e in variants.items():
    ce = M._to_ctm_env(e, template)
    eh = float(jnp.real(energy_from_rdm(_rdm2x1_tensor(A, ce), gate)))
    ev = float(jnp.real(energy_from_rdm(_rdm1x2_tensor(A, ce), gate)))
    print(
        f"  {tag:24s} E_h {eh:+.12f} ({eh - E_h:+.1e})  E_v {ev:+.12f} ({ev - E_v:+.1e})"
    )

print("\n=== does the module _init_env need the same conversion? ===")
env_i, _a = M._init_env(A, chi)
print(
    f"  |C4 - C4.T|/|C4| = {float(jnp.linalg.norm(env_i.C4 - env_i.C4.T) / jnp.linalg.norm(env_i.C4)):.3e}"
)
for nm in ("T3", "T4"):
    t = getattr(env_i, nm)
    tr = jnp.transpose(t, (2, 1, 0))
    print(
        f"  |{nm} - reverse({nm})|/|{nm}| = {float(jnp.linalg.norm(t - tr) / jnp.linalg.norm(t)):.3e}"
    )
