"""Find the PEPS index mapping by requiring my quadrant == their EC."""

import itertools

import numpy as np


def load(path):
    """Read the dump written by 718-dump.jl (Julia is column-major)."""
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


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


d = load("dump.txt")
d.update(load("dump2.txt"))
P = d["peps"]  # (2,2,2,2,2) PEPSKit order
chi, D = d["C1"].shape[0], P.shape[1]
d2 = D * D


def build_a(A, ketmajor=True):
    """A[u,d,l,r,phys] -> a[u2,d2,l2,r2], each leg fused (ket,bra) or (bra,ket)."""
    x = np.einsum("udlrp,UDLRp->uUdDlLrR", A, A.conj())
    if not ketmajor:
        x = np.einsum("uUdDlLrR->UuDdLlRr", x)
    return x.reshape(d2, d2, d2, d2)


def fuse_edge(E, ketmajor=True):
    """(chi,D,D,chi) -> (chi,d2,chi)."""
    if not ketmajor:
        E = E.transpose(0, 2, 1, 3)
    return E.reshape(E.shape[0], d2, E.shape[3])


def quadrant(C1, T1, T4, a):
    return np.einsum("ce,efg,hic,fjik->gkhj", C1, T1, T4, a)


best = []
for perm in itertools.permutations(range(1, 5)):  # their virtual axes -> u,d,l,r
    for phys in (0,):
        A = P.transpose(perm + (phys,))
        for km_a in (True, False):
            a = build_a(A, km_a)
            for km_e in (True, False):
                for swap in (False, True):
                    errs = []
                    ak = a
                    for k in range(1, 5):
                        km = (k - 2) % 4 + 1
                        T1 = fuse_edge(d[f"Et{k}"], km_e)
                        T4 = fuse_edge(d[f"Et{km}"], km_e)
                        Q = quadrant(d[f"iCi{k}"], T1, T4, ak)
                        Q = Q.reshape(chi * d2, chi * d2)
                        R = d[f"EC{k}"].reshape(chi * d2, chi * d2)
                        if swap:
                            R = (
                                d[f"EC{k}"]
                                .transpose(3, 4, 5, 0, 1, 2)
                                .reshape(chi * d2, chi * d2)
                            )
                        errs.append(rel(Q, R))
                        ak = np.transpose(ak, (3, 2, 0, 1))  # rotate_a, per direction
                    best.append((max(errs), perm, km_a, km_e, swap))
best.sort()
print(
    "top candidates (max rel err over k, perm(their->u,d,l,r), a-ketmajor, E-ketmajor):"
)
for b in best[:8]:
    print(f"   {b[0]:.3e}   perm={b[1]} a_km={b[2]} E_km={b[3]} swap={b[4]}")
