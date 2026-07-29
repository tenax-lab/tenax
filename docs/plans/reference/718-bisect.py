"""Bisect: apply MY formulas to THEIR tensors, compare to THEIR intermediates.

No cross-convention mapping is needed for the (chi,chi) objects, and for the
fused cut index loading column-major then reshaping C-order reproduces the
chi-slowest convention used in the port.
"""

import numpy as np

D2 = 4  # d2 = D*D


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
            re, im = lines[i + 1 + j].split()
            v[j] = complex(float(re), float(im))
        out[name] = v.reshape(dims, order="F")  # Julia is column-major
        i += 1 + n
    return out


d = load("dump.txt")
d.update(load("dump2.txt"))


def sqrtm(A, n_iter=40):
    eye = np.eye(A.shape[0], dtype=A.dtype)
    scale = np.sqrt(np.linalg.norm(A))
    Y, Z = A / scale**2, eye.copy()
    for _ in range(n_iter):
        Yn = 0.5 * (Y + np.linalg.inv(Z))
        Zn = 0.5 * (Z + np.linalg.inv(Y))
        Y, Z = Yn, Zn
    return Y * scale


def quartic(A):
    return sqrtm(sqrtm(A))


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


print("=== 1. is = inv(S) ===")
for k in range(1, 5):
    mine = np.linalg.inv(d[f"S{k}"])
    print(f"  k={k}: {rel(mine, d[f'is{k}']):.3e}")

print("=== 2. Eq.73 roots ===")
for k in range(1, 5):
    s = d[f"is{k}"]
    print(
        f"  k={k} rootL(=4rt(s^H s)): {rel(quartic(s.conj().T @ s), d[f'rootL{k}']):.3e}"
        f"   rootR(=4rt(s s^H)): {rel(quartic(s @ s.conj().T), d[f'rootR{k}']):.3e}"
    )

print("=== 3. iCi = is_{k-1} Ct_k is_k  (tests the direction offset) ===")
for k in range(1, 5):
    km = (k - 2) % 4 + 1
    for label, cand in (
        ("is_{k-1} Ct is_k", d[f"is{km}"] @ d[f"Ct{k}"] @ d[f"is{k}"]),
        ("is_k Ct is_{k-1}", d[f"is{k}"] @ d[f"Ct{k}"] @ d[f"is{km}"]),
        ("is_k Ct is_k    ", d[f"is{k}"] @ d[f"Ct{k}"] @ d[f"is{k}"]),
    ):
        print(f"  k={k} {label}: {rel(cand, d[f'iCi{k}']):.3e}")

print("=== 4. Ud = U^H . kron(rootL_?, I) ===")
for k in range(1, 5):
    U = d[f"U{k}"].reshape(D2 * d[f"U{k}"].shape[0] // D2 * 1, -1) if False else None
    Uk = d[f"U{k}"]  # (chi, D, D, chi_new)
    n = Uk.shape[0] * Uk.shape[1] * Uk.shape[2]
    Umat = Uk.reshape(n, Uk.shape[3])  # C-order fuse -> chi slowest
    Udag = Umat.conj().T  # (chi_new, n)
    ref = d[f"Ud{k}"]  # (chi_new, chi, D, D)
    refmat = ref.reshape(ref.shape[0], n)
    best = None
    for j in range(1, 5):
        K = np.kron(d[f"rootL{j}"], np.eye(D2))
        e = rel(Udag @ K, refmat)
        if best is None or e < best[1]:
            best = (j, e)
        print(f"  k={k} rootL_{j}: {e:.3e}")
    print(f"    -> best j={best[0]} ({best[1]:.3e}); k-1 would be {(k - 2) % 4 + 1}")

print("=== 5. Vd = kron(rootR_?, I) . V^H ===")
for k in range(1, 5):
    Vk = d[f"V{k}"]  # (chi_new, chi, D, D)
    n = Vk.shape[1] * Vk.shape[2] * Vk.shape[3]
    Vmat = Vk.reshape(Vk.shape[0], n)  # (chi_new, n)
    Vdag = Vmat.conj().T  # (n, chi_new)
    ref = d[f"Vd{k}"]  # (chi, D, D, chi_new)
    refmat = ref.reshape(n, ref.shape[3])
    best = None
    for j in range(1, 5):
        K = np.kron(d[f"rootR{j}"], np.eye(D2))
        e = rel(K @ Vdag, refmat)
        if best is None or e < best[1]:
            best = (j, e)
        print(f"  k={k} rootR_{j}: {e:.3e}")
    print(f"    -> best j={best[0]} ({best[1]:.3e}); k+1 would be {k % 4 + 1}")
