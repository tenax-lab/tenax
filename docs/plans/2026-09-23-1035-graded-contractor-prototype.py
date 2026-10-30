"""Prototype graded contractor for the #1035 design (section 3.4).

Run from a checkout that has ``tests/_fermionic_fock_oracle.py`` (PR #1038):

    python docs/plans/2026-09-23-1035-graded-contractor-prototype.py

Plain numpy, dense, tiny clusters only; it is evidence for a design decision,
not a proposed implementation.
A graded tensor = array + ordered legs (name, flow).  Storage order IS the
Grassmann generator order.  Parity of index value = value % 2.
Rules (the whole convention, nothing fitted):
  * gtranspose: Koszul sign over inverted odd-odd pairs.
  * contract a pair: bring the two legs adjacent at the end (Koszul), then
    +1 if the OUT leg is first, (-1)^p if the IN leg is first.
  * bar: conj, flip flows, REVERSE leg order (Grassmann conjugation).
"""

import itertools
import sys
from pathlib import Path

import numpy as np

# the oracle lives in tests/ (PR #1038)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from _fermionic_fock_oracle import (  # noqa: E402
    _H2,
    _annihilate,
    _create,
    bonds_of,
    double_layer_energy,
    fock_psi,
    hop_energy,
    hop_energy_matvec,
    leg_dims,
    plain_double_layer,
    random_even_tensors,
    sites_of,
)


class G:
    def __init__(s, a, legs):
        assert a.ndim == len(legs), (a.shape, legs)
        s.a, s.legs = a, list(legs)

    names = property(lambda s: [n for n, _ in s.legs])


def _par(shape):
    return [np.arange(n) % 2 for n in shape]


def koszul(shape, perm):
    """sign array (in ORIGINAL axis order) for moving axes into perm order."""
    ps = _par(shape)
    nd = len(shape)
    sgn = np.zeros(shape, dtype=np.int64)
    pos = {ax: k for k, ax in enumerate(perm)}
    for i in range(nd):
        for j in range(i + 1, nd):
            if pos[i] > pos[j]:
                sh = [1] * nd
                sh[i] = shape[i]
                pi = ps[i].reshape(sh)
                sh = [1] * nd
                sh[j] = shape[j]
                pj = ps[j].reshape(sh)
                sgn = sgn + pi * pj
    return 1 - 2 * (sgn % 2)


def gtranspose(T, names):
    perm = [T.names.index(n) for n in names]
    a = (T.a * koszul(T.a.shape, perm)).transpose(perm)
    return G(a, [T.legs[p] for p in perm])


def plain_permute(T, names):  # sign-free: storage order changes, physics would too
    perm = [T.names.index(n) for n in names]
    return G(T.a.transpose(perm), [T.legs[p] for p in perm])


def gprod(A, B):
    return G(np.multiply.outer(A.a, B.a), A.legs + B.legs)


def gtrace(T, x, y):
    fx, fy = dict(T.legs)[x], dict(T.legs)[y]
    assert {fx, fy} == {"O", "I"}, (x, y, fx, fy)
    rest = [n for n in T.names if n not in (x, y)]
    T = gtranspose(T, rest + [x, y])
    d = np.einsum("...ii->...i", T.a)
    if fx == "I":  # IN leg first -> (-1)^p
        d = d * (1 - 2 * (np.arange(d.shape[-1]) % 2))
    return G(d.sum(-1), T.legs[:-2])


def gcontract(A, B, pairs):
    T = gprod(A, B)
    for x, y in pairs:
        T = gtrace(T, x, y)
    return T


def bar(T):
    flip = {"O": "I", "I": "O"}
    return G(
        np.conj(T.a).transpose(list(range(T.a.ndim))[::-1]),
        [(n + "~", flip[f]) for n, f in reversed(T.legs)],
    )


# ---------------------------------------------------------------- network
FLOW = {"u": "O", "d": "I", "l": "O", "r": "I", "p": "I"}  # tenax's flows


def ket_sites(R, C, As):
    name = {}
    for b, (s, x, t, y) in enumerate(bonds_of(R, C)):
        name[(s, x)] = name[(t, y)] = f"b{b}"
    out = {}
    for n, s in enumerate(sites_of(R, C)):
        legs, arr, keep = [], As[s], []
        for k, x in enumerate("udlr"):
            if (s, x) in name:
                legs.append((f"{name[(s, x)]}{'s' if x in 'rd' else 't'}", FLOW[x]))
                keep.append(k)
        a = arr[tuple(slice(None) if k in keep else 0 for k in range(4))]
        out[s] = G(a, legs + [(f"p{n}", "I")])
    return out


def network_value(R, C, As, bond=None, order=None, perturb=None):
    """<psi|O|psi> (O = hop on `bond` or identity) by graded contraction."""
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    K = ket_sites(R, C, As)
    if perturb:
        K = {s: perturb(T) for s, T in K.items()}
    E = {}
    for s in sites:
        k, b = K[s], bar(K[s])
        E[s] = gprod(b, k)  # bra first, then ket (any order: both even)
        if bond is None or s not in bond:
            n = n_of[s]
            E[s] = gtrace(E[s], f"p{n}~", f"p{n}")
    tensors = [E[s] for s in (order or sites)]
    if bond is not None:
        s, t = n_of[bond[0]], n_of[bond[1]]
        # O = sum h |Ps Pt><ps pt| ; legs: creation half (s,t) then annihilation half (t,s)
        a = np.einsum("ABab->ABba", _H2)
        tensors.append(
            G(a, [(f"O{s}", "I"), (f"O{t}", "I"), (f"o{t}", "O"), (f"o{s}", "O")])
        )
    T = tensors[0]
    for U in tensors[1:]:
        T = gprod(T, U)
        # contract every pair that is now complete
        while True:
            nm = T.names
            done = False
            for x in nm:
                y = None
                if x.startswith("b") and x.endswith(("s", "t")):
                    y = x[:-1] + ("t" if x[-1] == "s" else "s")
                elif x.startswith("b") and x.endswith(("s~", "t~")):
                    y = x[:-2] + ("t~" if x[-2] == "s" else "s~")
                elif x.startswith("O"):
                    y = f"p{x[1:]}~"
                elif x.startswith("o"):
                    y = f"p{x[1:]}"
                if y and y in nm:
                    T = gtrace(T, x, y)
                    done = True
                    break
            if not done:
                break
    assert T.a.ndim == 0, T.names
    return float(T.a)


def graded_energy(R, C, As, **kw):
    norm = network_value(R, C, As, **kw)
    return sum(
        network_value(R, C, As, bond=(s, t), **kw) for s, _, t, _ in bonds_of(R, C)
    ) / norm, norm


def z_gauge(R, C, As):  # Z on every bond's t-side leg (u, l)
    out = {}
    for s, A in As.items():
        A = A.copy()
        for ax, x in ((0, "u"), (2, "l")):
            if A.shape[ax] == 2:
                idx = [slice(None)] * 5
                idx[ax] = 1
                A[tuple(idx)] *= -1
        out[s] = A
    return out


# ---------- (2) odd tensors in the global order
def random_parity(shape, parity, rng):
    A = rng.standard_normal(shape)
    for k in itertools.product(*[range(n) for n in shape]):
        if sum(k) % 2 != parity:
            A[k] = 0.0
    return A


def fock_ordered(R, C, As):
    """|psi> = O_1 O_2 ... O_N |0>, row-major left to right (defined order)."""
    sites = sites_of(R, C)
    sid = {s: n for n, s in enumerate(sites)}
    mode, M = {}, len(sites)
    for s, x, t, y in bonds_of(R, C):
        mode[(s, x)], mode[(t, y)] = M, M + 1
        M += 2
    vec = np.zeros(1 << M)
    vec[0] = 1.0
    for s in reversed(sites):  # apply O_N first
        A, new = As[s], np.zeros_like(vec)
        for k in itertools.product(*[range(n) for n in A.shape]):
            if A[k] == 0:
                continue
            v = vec
            for leg, bit in reversed(list(zip("udlrp", k))):
                if bit:
                    v = _create(v, sid[s] if leg == "p" else mode[(s, leg)])
            new = new + A[k] * v
        vec = new
    for s, x, t, y in bonds_of(R, C):
        vec = vec + _annihilate(_annihilate(vec, mode[(s, x)]), mode[(t, y)])
    return vec[: 1 << len(sites)]


def complete_pairs(T):
    while True:
        nm, done = T.names, False
        for x in nm:
            y = None
            if x.startswith("b") and x.endswith(("s", "t")):
                y = x[:-1] + ("t" if x[-1] == "s" else "s")
            elif x.startswith("b") and x.endswith(("s~", "t~")):
                y = x[:-2] + ("t~" if x[-2] == "s" else "s~")
            elif x.startswith("O"):
                y = f"p{x[1:]}~"
            elif x.startswith("o"):
                y = f"p{x[1:]}"
            if y and y in nm:
                T = gtrace(T, x, y)
                done = True
                break
        if not done:
            return T


def global_value(R, C, bra_As, ket_As, op_bond=None):
    """<bra|O|ket> with every tensor in the true global order:
    bar(K_N) ... bar(K_1)  [O]  K_1 ... K_N."""
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    Kb, Kk = ket_sites(R, C, bra_As), ket_sites(R, C, ket_As)
    seq = [bar(Kb[s]) for s in reversed(sites)]
    if op_bond is not None:
        s, t = n_of[op_bond[0]], n_of[op_bond[1]]
        a = np.einsum("ABab->ABba", _H2)
        seq.append(
            G(a, [(f"O{s}", "I"), (f"O{t}", "I"), (f"o{t}", "O"), (f"o{s}", "O")])
        )
        touched = {s, t}
    else:
        touched = set()
    seq += [Kk[s] for s in sites]
    T = seq[0]
    for U in seq[1:]:
        T = complete_pairs(gprod(T, U))
    # identity on untouched physical legs: pair bra p~ (OUT) with ket p (IN)
    for n in range(len(sites)):
        if n not in touched:
            T = gtrace(T, f"p{n}~", f"p{n}")
    assert T.a.ndim == 0, T.names
    return float(T.a)


def local_value(R, C, bra_As, ket_As):
    """Regrouped as per-site double layers bar(K_s) K_s, in site order (CTM-style)."""
    sites = sites_of(R, C)
    Kb, Kk = ket_sites(R, C, bra_As), ket_sites(R, C, ket_As)
    T = None
    for n, s in enumerate(sites):
        E = gtrace(gprod(bar(Kb[s]), Kk[s]), f"p{n}~", f"p{n}")
        T = E if T is None else complete_pairs(gprod(T, E))
    return float(T.a)


# ---------------------------------------------------------------- checks
def check_even():
    """(A) graded == Fock on even states; order- and signed-reorder-invariant."""
    for R, C in [(2, 2), (2, 3), (3, 2)]:
        rng = np.random.default_rng(7)
        for trial in range(3):
            As = random_even_tensors(R, C, rng)
            Eg, ng = graded_energy(R, C, As)
            Ez = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
            Eh = hop_energy(R, C, fock_psi(R, C, As), fermion=False)
            Erule, _ = double_layer_energy(
                R,
                C,
                {s: plain_double_layer(A) for s, A in z_gauge(R, C, As).items()},
                rule=True,
            )
            Eo, _ = graded_energy(R, C, As, order=list(reversed(sites_of(R, C))))
            Es, _ = graded_energy(
                R, C, As, perturb=lambda T: gtranspose(T, T.names[::-1])
            )
            Ep, _ = graded_energy(
                R, C, As, perturb=lambda T: plain_permute(T, T.names[::-1])
            )
            print(
                f"{R}x{C} t{trial}: |graded-Fock| {abs(Eg - Ez):.1e}  norm>0 {ng > 0}  "
                f"|graded-HCB| {abs(Eg - Eh):.1e}  |graded-#1037 rule| {abs(Eg - Erule):.1e}  "
                f"order {abs(Eg - Eo):.1e}  signed-reorder {abs(Eg - Es):.1e}  "
                f"plain-reorder {abs(Eg - Ep):.1e}",
                flush=True,
            )


def check_mutants():
    """(B) each of the three rules is necessary."""
    global bar, gtrace, gtranspose

    def worst():
        w = 0.0
        for R, C in [(2, 2), (2, 3)]:
            rng = np.random.default_rng(7)
            for _ in range(3):
                As = random_even_tensors(R, C, rng)
                Ez = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
                w = max(w, abs(graded_energy(R, C, As)[0] - Ez))
        return w

    flip = {"O": "I", "I": "O"}
    good = (bar, gtrace, gtranspose)

    def bar_keep_order(T):  # tenax's bar(): conj + flip, order kept
        return G(np.conj(T.a), [(n + "~", flip[f]) for n, f in T.legs])

    def gtrace_no_orientation(T, x, y):
        rest = [n for n in T.names if n not in (x, y)]
        T = gtranspose(T, rest + [x, y])
        return G(np.einsum("...ii->...", T.a), T.legs[:-2])

    print(f"baseline                         worst {worst():.1e}")
    bar = bar_keep_order
    print(f"MUTANT bar keeps leg order       worst {worst():.1e}")
    bar = good[0]
    gtrace = gtrace_no_orientation
    print(f"MUTANT no pair-orientation sign  worst {worst():.1e}")
    gtrace = good[1]
    gtranspose = plain_permute
    print(f"MUTANT sign-free reorder         worst {worst():.1e}")
    gtranspose = good[2]


def check_odd():
    """(C) odd site tensors (excitation shape) in the global order, and the
    sign that a per-site regrouping loses."""
    for R, C in [(2, 2), (2, 3)]:
        rng = np.random.default_rng(11)
        sites = sites_of(R, C)
        bg = random_even_tensors(R, C, rng)
        Bs = {
            s: random_parity([leg_dims(R, C, *s)[x] for x in "udlr"] + [2], 1, rng)
            for s in sites
        }
        wN = wH = 0.0
        table = []
        for x in sites:
            row = []
            for y in sites:
                kx = dict(bg)
                kx[x] = Bs[x]
                by = dict(bg)
                by[y] = Bs[y]
                px = fock_ordered(R, C, z_gauge(R, C, kx))
                py = fock_ordered(R, C, z_gauge(R, C, by))
                Ng = global_value(R, C, by, kx)
                Hg = sum(
                    global_value(R, C, by, kx, op_bond=(s, t))
                    for s, _, t, _ in bonds_of(R, C)
                )
                wN = max(wN, abs(Ng - py @ px))
                wH = max(wH, abs(Hg - py @ hop_energy_matvec(R, C, px, fermion=True)))
                row.append("+" if local_value(R, C, by, kx) / Ng > 0 else "-")
            table.append("".join(row))
        print(
            f"odd-B {R}x{C}: {len(sites) ** 2} (x,y) pairs, worst |N-N_Fock| {wN:.1e}, "
            f"|H-H_Fock| {wH:.1e}; per-site regroup sign rows x, cols y: {' '.join(table)}",
            flush=True,
        )


if __name__ == "__main__":
    check_even()
    check_mutants()
    check_odd()
