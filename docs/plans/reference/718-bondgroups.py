"""Determine the TRUE bond grouping of an environment, with no convention reasoning.

A chi-bond of a CTM environment is carried by exactly four legs: one corner leg,
both legs of one edge, and one leg of the next corner.  Inserting a unitary W on
the outgoing legs of that group and W' on the incoming ones must leave the energy
untouched.  So: for every (edge, corner-leg, corner-leg) triple and every
orientation, evaluate the energy and keep the combinations that cancel.

Run on the production env first (control: the grouping must come out as the
documented one), then on the module's own converged env.
"""

import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/yjkao/tenax/tests")

from test_ctm_root_implicit_asym import _gate, _site_tensor  # noqa: E402

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor  # noqa: E402
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env  # noqa: E402

A = _site_tensor()
chi = 4
gate = _gate(1.0)
template = initialize_ctm_tensor_env(A, chi)

rng = np.random.RandomState(0)
G = rng.standard_normal((chi, chi))
X = 0.5 * (G - G.T)
W = jnp.asarray(expm(0.3 * X / np.linalg.norm(X)))
Wd = W.conj().T


def put(env_r, name, axis, mat):
    arr = getattr(env_r, name)
    if arr.ndim == 2:
        out = mat @ arr if axis == 0 else arr @ mat
    else:
        out = (
            jnp.einsum("ai,ixj->axj", mat, arr)
            if axis == 0
            else jnp.einsum("ixj,jb->ixb", arr, mat)
        )
    return env_r._replace(**{name: out})


corner_legs = [(nm, ax) for nm in ("C1", "C2", "C3", "C4") for ax in (0, 1)]


def bond_groups(env_r, tag):
    E0 = float(jnp.real(M.asym_energy(A, env_r, template, gate)))
    print(f"\n=== {tag}:  E = {E0:.12f} ===")
    hits = []
    for edge in ("T1", "T2", "T3", "T4"):
        for cl1, cl2 in itertools.combinations(corner_legs, 2):
            if cl1[0] == cl2[0]:
                continue
            for se, s1, s2 in itertools.product([0, 1], repeat=3):
                e = put(env_r, edge, 0, Wd if se == 0 else W)
                e = put(e, edge, 2, W if se == 0 else Wd)
                e = put(e, cl1[0], cl1[1], W if s1 == 0 else Wd)
                e = put(e, cl2[0], cl2[1], W if s2 == 0 else Wd)
                dE = abs(float(jnp.real(M.asym_energy(A, e, template, gate))) - E0)
                if dE < 1e-13:
                    hits.append((edge, cl1, cl2, dE))
    for edge, cl1, cl2, dE in hits:
        print(
            f"  bond group: {edge}(both) + {cl1[0]}.{cl1[1]} + {cl2[0]}.{cl2[1]}"
            f"   |dE| = {dE:.1e}"
        )
    if not hits:
        print(
            "  NO consistent bond group found for any edge -- env legs do not "
            "form chi-bonds in this convention"
        )
    return hits


env_p = ctm_tensor(A, chi=chi, max_iter=400, conv_tol=1e-12)
env_p = env_p[0] if isinstance(env_p, tuple) else env_p
env_prod = M.AsymEnv(
    *[
        jnp.asarray(getattr(env_p, n).todense())
        for n in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
    ]
)
bond_groups(env_prod, "production ctm_tensor env (control)")

env_m, a_arr, meta = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
bond_groups(env_m, f"module converge env ({meta['iters']} iters)")

env_i, _a = M._init_env(A, chi)
bond_groups(env_i, "module _init_env (before any sweep)")

env_1, _p = M.sweep(env_i, _a, chi, None)
bond_groups(env_1, "module after ONE sweep")
