"""Reference oracle: the paper's §V.3 characteristic equations, evaluated on the
reference implementation's own dumped fixed point.

Run from a directory holding ``dump.txt``/``dump2.txt`` as written by
``718-dump.jl``.  All five residual blocks come out at ~1e-12, the same order as
the reference's own ``|F|`` at that root, which is what pins every index
convention the port depends on:

* the enlarged corner is ordered ``(cut | outer)`` -- the port's quadrant helper
  returns ``(outer | cut)``, hence the transpose;
* the half-infinite environment glues ``EC[k]`` to ``EC[k+1]`` (the *upper*
  half), not two quadrants at one rotation (the left half the forward sweep
  uses);
* the corner takes ``PR[k-1]``, both of its projectors belonging to the same
  enlarged corner;
* ``lambda`` is complex -- real-projecting it alone moves ``|F1|`` from 2e-13
  to 1.6e0.
"""

import numpy as np


def load(path):
    out = {}
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("@"):
            i += 1
            continue
        _, name, shape = line.split()
        dims = tuple(int(x) for x in shape.split(","))
        n = int(np.prod(dims))
        v = np.empty(n, dtype=np.complex128)
        for j in range(n):
            re_, im_ = lines[i + 1 + j].split()
            v[j] = complex(float(re_), float(im_))
        out[name] = v.reshape(dims, order="F")
        i += 1 + n
    return out


d = load("dump.txt")
d.update(load("dump2.txt"))

P = d["peps"]
chi, D = d["C1"].shape[0], P.shape[1]
d2 = D * D
n = chi * d2

A = P.transpose((1, 3, 4, 2, 0))  # their (v1..v4,phys) -> my (u,d,l,r,phys)
a = np.einsum("udlrp,UDLRp->uUdDlLrR", A, A.conj()).reshape(d2, d2, d2, d2)


def rot_a(x):
    return np.transpose(x, (3, 2, 0, 1))


def fuse_edge(E):
    return E.reshape(E.shape[0], d2, E.shape[3])


def sqrtm(X, n_iter=40):
    eye = np.eye(X.shape[0], dtype=X.dtype)
    scale = np.sqrt(np.linalg.norm(X))
    Y, Z = X / scale**2, eye.copy()
    for _ in range(n_iter):
        Yn = 0.5 * (Y + np.linalg.inv(Z))
        Zn = 0.5 * (Z + np.linalg.inv(Y))
        Y, Z = Yn, Zn
    return Y * scale


def quartic(X):
    return sqrtm(sqrtm(X))


# ---- per-direction pieces, 0-based k ----
IS, S, Q, Ud, Vd, ULd, VRd, K_is, Et_f, Ct = ({} for _ in range(10))
rootL, rootR = {}, {}
for k in range(4):
    j = k + 1
    IS[k] = d[f"is{j}"]
    S[k] = d[f"S{j}"]
    Ct[k] = d[f"Ct{j}"]
    Et_f[k] = fuse_edge(d[f"Et{j}"])
    K_is[k] = np.kron(IS[k], np.eye(d2))
    s = IS[k]
    rootL[k] = quartic(s.conj().T @ s)
    rootR[k] = quartic(s @ s.conj().T)

ak = a
for k in range(4):
    j = k + 1
    km = (k - 1) % 4 + 1  # 1-based k-1
    Q[k] = np.einsum(
        "ce,efg,hic,fjik->gkhj",
        d[f"iCi{j}"],
        fuse_edge(d[f"Et{j}"]),
        fuse_edge(d[f"Et{km}"]),
        ak,
    ).reshape(n, n)
    ak = rot_a(ak)

for k in range(4):
    j = k + 1
    km, kp = (k - 1) % 4, (k + 1) % 4
    KL = np.kron(rootL[km], np.eye(d2))
    KR = np.kron(rootR[kp], np.eye(d2))
    Umat = d[f"U{j}"].reshape(n, chi)  # (chi,D,D,chi_new) -> (n, chi)
    Vmat = d[f"V{j}"].reshape(chi, n)  # (chi_new,chi,D,D) -> (chi, n)
    ULmat = d[f"UL{j}"].reshape(n, -1)  # (chi,D,D,perp)
    VRmat = d[f"VR{j}"].reshape(-1, n)  # (perp,chi,D,D)
    Ud[k] = Umat.conj().T @ KL
    Vd[k] = KR @ Vmat.conj().T
    ULd[k] = ULmat.conj().T @ KL
    VRd[k] = KR @ VRmat.conj().T

# sanity: these were already pinned, re-assert cheaply
for k in range(4):
    j = k + 1
    ref = d[f"Ud{j}"].reshape(chi, n)
    assert np.abs(Ud[k] - ref).max() < 1e-12, (k, np.abs(Ud[k] - ref).max())
print("Ud matches dump for all k")
for k in range(4):
    j = k + 1
    ref = d[f"Vd{j}"].reshape(n, chi)
    assert np.abs(Vd[k] - ref).max() < 1e-12, (k, np.abs(Vd[k] - ref).max())
print("Vd matches dump for all k")
for k in range(4):
    j = k + 1
    ref = d[f"EC{j}"].reshape(n, n)
    e = np.abs(Q[k].T - ref).max() / np.abs(ref).max()
    print(f"  EC_ref[{k}] == Q[{k}].T : {e:.2e}")

# ---- projectors and the half-infinite environment ----
PR, PL, PLpart, M = {}, {}, {}, {}
for k in range(4):
    kp = (k + 1) % 4
    PR[k] = Ud[k] @ Q[k].T @ K_is[k]
    PLpart[k] = Q[kp].T @ Vd[k]
    PL[k] = K_is[k] @ PLpart[k]
    M[k] = Q[k].T @ K_is[k] @ Q[kp].T


cols = {}
akl = a
for k in range(4):
    cols[k] = np.einsum("xfy,fjlr->xljyr", Et_f[k], akl).reshape(n, d2, n)
    akl = rot_a(akl)

print()
print("=== all five blocks, complex lambda ===")
tot = 0.0
for k in range(4):
    kp, km = (k + 1) % 4, (k - 1) % 4
    EC = {j: Q[j].T for j in range(4)}
    PRk = Ud[k] @ EC[k] @ K_is[k]
    PLk = K_is[k] @ EC[kp] @ Vd[k]
    Mk = EC[k] @ K_is[k] @ EC[kp]

    Sp = Ud[k] @ Mk @ Vd[k]
    lam_S = np.vdot(S[k], Sp)
    F4 = Sp / lam_S - S[k]
    F3 = ULd[k] @ Mk @ Vd[k] @ IS[k] / lam_S  # u = 0 at the root
    F5 = IS[k] @ Ud[k] @ Mk @ VRd[k] / lam_S  # v = 0 at the root

    PRm = Ud[km] @ EC[km] @ K_is[km]
    Cp = PRm @ EC[k] @ PLk
    lam_C = np.vdot(Ct[k], Cp)
    F1 = Cp / lam_C - Ct[k]

    Ep = np.einsum("ax,xjy,yb->ajb", PRk, cols[k], PLk)
    lam_E = np.vdot(Et_f[k], Ep)
    F2 = Ep / lam_E - Et_f[k]

    norms = [np.linalg.norm(x) for x in (F1, F2, F3, F4, F5)]
    tot += sum(n_**2 for n_ in norms)
    print(
        f"  k={k}  |F1|={norms[0]:.2e} |F2|={norms[1]:.2e} "
        f"|F3|={norms[2]:.2e} |F4|={norms[3]:.2e} |F5|={norms[4]:.2e}"
    )
    print(f"        lam_C={lam_C:.6e}  lam_E={lam_E:.6e}  lam_S={lam_S:.6e}")

print(f"\n  total |F| = {np.sqrt(tot):.3e}   (reference at its own root: ~1e-12)")
