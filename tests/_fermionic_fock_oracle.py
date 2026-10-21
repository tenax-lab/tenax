"""Exact fermionic reference for small open-boundary PEPS clusters (#1037).

Every sign here comes from the fermionic operator algebra -- no drawing, no
swap gates, no convention to get wrong.  Use it to check any contraction
scheme (a double layer, a graded contractor, an odd-operator string) on a
cluster small enough to hold in Fock space.

**The fermionic PEPS.**  Sites are ``(i, j)`` in row-major order.  A site
tensor ``A[u, d, l, r, p]`` (tenax's storage order; boundary legs have
dimension 1, bond legs dimension 2 with index == parity) is the parity-even
operator

    A_s = sum A[u,d,l,r,p] (a_u^+)^u (a_d^+)^d (a_l^+)^l (a_r^+)^r (c_s^+)^p .

Each bond ``s.x -- t.y`` (``s`` before ``t``) is contracted by the pair
projection ``(1 + a_{t,y} a_{s,x})``, after which every virtual mode is
projected onto its vacuum.  Physical modes come first in the Jordan-Wigner
order, so the result is a vector over physical Fock states with bit ``n`` of
the index = occupation of site ``n``.

:func:`sign_formula` is the same object in closed form, and the tests check the
two agree exhaustively.
"""

from __future__ import annotations

import itertools

import numpy as np

LEGS = "udlr"


def sites_of(R: int, C: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(R) for j in range(C)]


def bonds_of(R: int, C: int) -> list[tuple[tuple[int, int], str, tuple[int, int], str]]:
    """``(s, leg_s, t, leg_t)`` with ``s`` before ``t`` in row-major order."""
    return [((i, j), "r", (i, j + 1), "l") for i in range(R) for j in range(C - 1)] + [
        ((i, j), "d", (i + 1, j), "u") for i in range(R - 1) for j in range(C)
    ]


def leg_dims(R: int, C: int, i: int, j: int, D: int = 2) -> dict[str, int]:
    return {
        "u": D if i > 0 else 1,
        "d": D if i < R - 1 else 1,
        "l": D if j > 0 else 1,
        "r": D if j < C - 1 else 1,
    }


def random_even_tensors(R: int, C: int, rng: np.random.Generator) -> dict:
    """Random real parity-even site tensors ``A[u, d, l, r, p]``."""
    out = {}
    for s in sites_of(R, C):
        dm = leg_dims(R, C, *s)
        shape = [dm["u"], dm["d"], dm["l"], dm["r"], 2]
        A = rng.standard_normal(shape)
        for k in itertools.product(*[range(n) for n in shape]):
            if sum(k) % 2:
                A[k] = 0.0
        out[s] = A
    return out


# ------------------------------------------------------------------ #
# Fock-space primitives on bit-indexed state vectors                  #
# ------------------------------------------------------------------ #


def _parity(x: np.ndarray) -> np.ndarray:
    """Parity of the population count of non-negative int64s (NumPy 1.x-safe;
    ``np.bitwise_count`` needs NumPy 2)."""
    x = x.astype(np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        x = x ^ (x >> shift)
    return x & 1


def _jw_sign(src: np.ndarray, k: int) -> np.ndarray:
    return 1 - 2 * _parity(src & ((1 << k) - 1))


def _create(vec: np.ndarray, k: int) -> np.ndarray:
    idx = np.arange(vec.size)
    src = idx[((idx >> k) & 1) == 0]
    out = np.zeros_like(vec)
    out[src | (1 << k)] = _jw_sign(src, k) * vec[src]
    return out


def _annihilate(vec: np.ndarray, k: int) -> np.ndarray:
    idx = np.arange(vec.size)
    src = idx[((idx >> k) & 1) == 1]
    out = np.zeros_like(vec)
    out[src & ~(1 << k)] = _jw_sign(src, k) * vec[src]
    return out


def _fock_state(R: int, C: int, As: dict, site_order: list) -> np.ndarray:
    """Build fermionic state by applying site operators in the given order.

    Shared implementation for fock_psi and fock_psi_ordered. The site_order
    parameter determines the order in which parity-even site operators are
    applied: site_order[0] is applied first (innermost, closest to |0>), so
    forward site_order gives O_N ... O_1 |0> and reversed gives O_1 ... O_N.
    """
    sites = sites_of(R, C)
    sid = {s: n for n, s in enumerate(sites)}
    mode, M = {}, len(sites)
    for s, x, t, y in bonds_of(R, C):
        mode[(s, x)], mode[(t, y)] = M, M + 1
        M += 2
    vec = np.zeros(1 << M, dtype=np.result_type(*As.values()))
    vec[0] = 1.0
    for s in site_order:
        A, new = As[s], np.zeros_like(vec)
        for k in itertools.product(*[range(n) for n in A.shape]):
            if A[k] == 0:
                continue
            v = vec
            # (a_u^+)^u ... (c^+)^p acts right-to-left: p first, u last
            for leg, bit in reversed(list(zip("udlrp", k))):
                if bit:
                    v = _create(v, sid[s] if leg == "p" else mode[(s, leg)])
            new = new + A[k] * v
        vec = new
    for s, x, t, y in bonds_of(R, C):
        vec = vec + _annihilate(_annihilate(vec, mode[(s, x)]), mode[(t, y)])
    return vec[: 1 << len(sites)]  # every virtual mode in its vacuum


def fock_psi(R: int, C: int, As: dict) -> np.ndarray:
    """The fermionic state of the cluster, over physical Fock states.

    Parity-even site operators commute: order is free. Returns O_N ... O_1 |0>.
    """
    return _fock_state(R, C, As, sites_of(R, C))


def z_gauge(R: int, C: int, As: dict) -> dict:
    """Z on every bond's t-side leg (u, l).  Rule 2's "OUT leg first" pairing
    equals the oracle's ``(1 + a_t a_s)`` projector in this gauge.  The sign
    is per bond, so the s-side legs would do equally well."""
    out = {}
    for s, A in As.items():
        A = np.array(A, copy=True)
        for ax in (0, 2):
            if A.shape[ax] == 2:
                idx = [slice(None)] * A.ndim
                idx[ax] = 1
                A[tuple(idx)] *= -1
        out[s] = A
    return out


def fock_psi_ordered(R: int, C: int, As: dict) -> np.ndarray:
    """``O_1 O_2 ... O_N |0>`` (row-major, left to right).  Equal to
    ``fock_psi`` for even tensors; defines the order when some are odd."""
    return _fock_state(R, C, As, list(reversed(sites_of(R, C))))


def hop_energy(R: int, C: int, psi: np.ndarray, *, fermion: bool) -> float:
    """``<psi| -sum_<st> (c_s^+ c_t + h.c.) |psi> / <psi|psi>``.

    ``fermion=False`` drops the Jordan-Wigner strings: hard-core bosons.
    """
    sid = {s: n for n, s in enumerate(sites_of(R, C))}
    if fermion:
        cr, an = _create, _annihilate
    else:

        def cr(v, k):
            idx = np.arange(v.size)
            src = idx[((idx >> k) & 1) == 0]
            out = np.zeros_like(v)
            out[src | (1 << k)] = v[src]
            return out

        def an(v, k):
            idx = np.arange(v.size)
            src = idx[((idx >> k) & 1) == 1]
            out = np.zeros_like(v)
            out[src & ~(1 << k)] = v[src]
            return out

    h = np.zeros_like(psi)
    for s, _, t, _ in bonds_of(R, C):
        a, b = sid[s], sid[t]
        h -= cr(an(psi, b), a) + cr(an(psi, a), b)
    # vdot conjugates its first argument: <psi|H|psi>, valid for complex states
    return float((np.vdot(psi, h) / np.vdot(psi, psi)).real)


def ground_energy(R: int, C: int, *, fermion: bool) -> float:
    """Exact even-parity ground energy (the sector a parity-even PEPS lives in)."""
    N = R * C
    H = np.zeros((1 << N, 1 << N))
    for col in range(1 << N):
        e = np.zeros(1 << N)
        e[col] = 1.0
        H[:, col] = hop_energy_matvec(R, C, e, fermion=fermion)
    even = np.array([bin(s).count("1") % 2 == 0 for s in range(1 << N)])
    return float(np.linalg.eigvalsh(H[np.ix_(even, even)])[0])


def hop_energy_matvec(R: int, C: int, v: np.ndarray, *, fermion: bool) -> np.ndarray:
    sid = {s: n for n, s in enumerate(sites_of(R, C))}
    out = np.zeros_like(v)
    for s, _, t, _ in bonds_of(R, C):
        a, b = sid[s], sid[t]
        if fermion:
            out -= _create(_annihilate(v, b), a) + _create(_annihilate(v, a), b)
        else:
            idx = np.arange(v.size)
            for x, y in ((a, b), (b, a)):  # b_x^+ b_y, no strings
                src = idx[(((idx >> y) & 1) == 1) & (((idx >> x) & 1) == 0)]
                out[(src & ~(1 << y)) | (1 << x)] -= v[src]
    return out


# ------------------------------------------------------------------ #
# Closed form                                                          #
# ------------------------------------------------------------------ #


def sign_formula(R: int, C: int, occ, phys: dict) -> int:
    """Exponent (mod 2) of the fermionic sign of one configuration.

    Put every operator on the row-major line (site by site, ``u,d,l,r,p``
    within a site).  The sign is ``(-1)**(X + Y)``: ``X`` counts pairs of
    occupied bond arcs that cross, ``Y`` counts (occupied arc, occupied
    physical operator strictly inside it) pairs.
    """
    pos = {
        (s, x): n * 5 + "udlrp".index(x)
        for n, s in enumerate(sites_of(R, C))
        for x in "udlrp"
    }
    arcs = [
        (pos[(s, x)], pos[(t, y)])
        for b, (s, x, t, y) in enumerate(bonds_of(R, C))
        if occ[b]
    ]
    X = sum(
        1
        for (a1, b1), (a2, b2) in itertools.combinations(arcs, 2)
        if (a1 < a2 < b1) != (a1 < b2 < b1)
    )
    inside = [pos[(s, "p")] for s, v in phys.items() if v]
    Y = sum(1 for a, b in arcs for q in inside if a < q < b)
    return (X + Y) % 2


def site_bits(R: int, C: int, occ, s) -> tuple[dict, int]:
    bits = {x: 0 for x in LEGS}
    for b, (a, x, t, y) in enumerate(bonds_of(R, C)):
        if occ[b]:
            if a == s:
                bits[x] = 1
            if t == s:
                bits[y] = 1
    return bits, sum(bits.values()) % 2


# ------------------------------------------------------------------ #
# Double-layer contraction with a pluggable sign rule                  #
# ------------------------------------------------------------------ #

# h[P_s, P_t, p_s, p_t] for -(c_s^+ c_t + h.c.) in the local (s, t) basis
_H2 = np.zeros((2, 2, 2, 2))
_H2[1, 0, 0, 1] = _H2[0, 1, 1, 0] = -1.0


def derived_site_exponent(k, b) -> int:
    """#1037's derived per-site rule; ``k``/``b`` are ket/bra ``(u,d,l,r,p)`` bits."""
    ku, kd, kl, kr, _ = k
    bu, bd, bl, br, _ = b
    return (
        ku + bu + bl
        + ku * kd + ku * kl + ku * kr + kd * kr + kd * bl + kl * kr + kl * bl
        + kr * bd + kr * bl + bu * bd + bu * bl + bu * br + bd * bl
    )  # fmt: skip


def derived_hop_exponent(kind: str, ks, bs, kt, bt) -> int:
    """#1037's operator rule on the two hop sites (``s`` before ``t``)."""
    if kind == "h":
        return ks[0] + ks[1] + ks[2] + ks[3] + bs[0] + kt[0]
    return ks[3] + bs[2]


def double_layer_energy(
    R: int, C: int, sites_E: dict, rule: bool
) -> tuple[float, float]:
    """Contract a cluster from per-site double-layer tensors.

    ``sites_E[s]`` has legs ``(u, d, l, r, p, U, D, L, R, P)`` (ket then bra).
    With ``rule=True`` the #1037 site and hop signs are applied; with
    ``rule=False`` the tensors are contracted as given.
    """
    sites = sites_of(R, C)
    lab, cnt = {}, [0]

    def new():
        cnt[0] += 1
        return cnt[0]

    for s, x, t, y in bonds_of(R, C):
        for layer in (0, 1):
            lab[(s, x, layer)] = lab[(t, y, layer)] = new()
    for s in sites:
        for x in LEGS:
            for layer in (0, 1):
                if (s, x, layer) not in lab:  # not setdefault: new() must not run
                    lab[(s, x, layer)] = new()
    base = cnt[0] + 1
    pid = {s: (base + 2 * n, base + 1 + 2 * n) for n, s in enumerate(sites)}

    def signed(s, extra):
        E = np.array(sites_E[s], copy=True)
        if not rule:
            return E
        half = E.shape[:5]
        for k in itertools.product(*[range(n) for n in half]):
            for b in itertools.product(*[range(n) for n in half]):
                if (derived_site_exponent(k, b) + extra(k, b)) % 2:
                    E[k + b] *= -1
        return E

    def net(bond):
        args = []
        for s in sites:
            extra = lambda k, b: 0  # noqa: E731
            if bond is not None:
                kind, s0, t0 = bond
                z = (0,) * 5
                if s == s0:
                    extra = lambda k, b, kd=kind: derived_hop_exponent(kd, k, b, z, z)  # noqa: E731
                elif s == t0:
                    extra = lambda k, b, kd=kind: derived_hop_exponent(kd, z, z, k, b)  # noqa: E731
            sub = [lab[(s, x, 0)] for x in LEGS] + [pid[s][0]]
            sub += [lab[(s, x, 1)] for x in LEGS] + [pid[s][1]]
            args += [signed(s, extra), sub]
        for s in sites:
            if bond is None or s not in bond[1:]:
                args += [np.eye(2), [pid[s][0], pid[s][1]]]
        if bond is not None:
            _, s, t = bond
            args += [_H2, [pid[s][1], pid[t][1], pid[s][0], pid[t][0]]]
        return real_scalar(np.einsum(*args, [], optimize="greedy"))

    bonds = [("h" if x == "r" else "v", s, t) for s, x, t, _ in bonds_of(R, C)]
    norm = net(None)
    return sum(net(b) for b in bonds) / norm, norm


def real_scalar(x) -> float:
    """A closed network that must be real (a norm, a Hermitian expectation
    value): check the imaginary residual, then return the real part.  Complex
    tensors give a complex dtype even when the value is real."""
    z = complex(np.asarray(x).reshape(-1)[0])
    assert abs(z.imag) <= 1e-10 * max(1.0, abs(z.real)), z
    return z.real


def plain_double_layer(A: np.ndarray) -> np.ndarray:
    """``A (x) conj(A)`` with no sign: legs ``(u,d,l,r,p, U,D,L,R,P)``."""
    return np.einsum("udlrp,UDLRP->udlrpUDLRP", A, A.conj())


def plain_amplitudes(R: int, C: int, As: dict) -> np.ndarray:
    """The sign-free ket contraction, indexed like :func:`fock_psi`."""
    sites = sites_of(R, C)
    lab, cnt = {}, [0]
    for s, x, t, y in bonds_of(R, C):
        cnt[0] += 1
        lab[(s, x)] = lab[(t, y)] = cnt[0]
    for s in sites:
        for x in LEGS:
            if (s, x) not in lab:
                cnt[0] += 1
                lab[(s, x)] = cnt[0]
    phys = {s: cnt[0] + 1 + n for n, s in enumerate(sites)}
    args = []
    for s in sites:
        args += [As[s], [lab[(s, x)] for x in LEGS] + [phys[s]]]
    out = [phys[s] for s in reversed(sites)]  # site 0 = least significant bit
    return np.einsum(*args, out, optimize="greedy").reshape(-1)
