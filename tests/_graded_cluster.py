"""Build #1038's oracle clusters out of SymmetricTensors and contract them with
tenax.core._graded, so the graded contractor can be checked against Fock."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from _fermionic_fock_oracle import _H2, bonds_of, fock_psi, sites_of

from tenax.algorithms._graded_double_layer import build_graded_double_layer
from tenax.core._graded import graded_bar, graded_contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT
FLOW = {"u": OUT, "d": IN, "l": OUT, "r": IN}
SYM = FermionParity()


def _index(n: int, flow, label) -> TensorIndex:
    return TensorIndex.from_charges(
        SYM, np.arange(n, dtype=np.int32) % 2, flow, label=label
    )


def ket_site(R: int, C: int, s, A: np.ndarray, *, aux: bool = False) -> SymmetricTensor:
    """``A[u,d,l,r,p]`` as a SymmetricTensor over its bond legs and ``p``.

    Boundary legs (dimension 1, parity 0) are dropped.  ``aux=True`` prepends
    a dimension-1 **odd** leg ``x`` (flow IN), which is how an odd-parity
    ``A`` is stored as an even tensor; its bra partner is contracted with it.
    """
    name = {}
    for b, (a, x, t, y) in enumerate(bonds_of(R, C)):
        name[(a, x)] = f"b{b}"
        name[(t, y)] = f"b{b}"
    n = sites_of(R, C).index(s)
    keep = [k for k, x in enumerate("udlr") if (s, x) in name]
    arr = A[tuple(slice(None) if k in keep else 0 for k in range(4))]
    idx = [_index(2, FLOW["udlr"[k]], name[(s, "udlr"[k])]) for k in keep]
    idx.append(_index(2, IN, f"p{n}"))
    if aux:
        arr = arr[None]
        idx.insert(
            0, TensorIndex.from_charges(SYM, np.array([1], np.int32), IN, label="x")
        )
    return SymmetricTensor.from_dense(jnp.asarray(arr), tuple(idx))


def bra_site(ket: SymmetricTensor, n: int, *, touched: bool) -> SymmetricTensor:
    """``graded_bar`` of a ket site, legs renamed so they pair with the bra
    network (``b*`` -> ``B*``), the ket (``p`` if untouched) or the operator."""
    m = {lab: "B" + lab[1:] for lab in ket.labels() if lab.startswith("b")}
    m[f"p{n}"] = f"P{n}" if touched else f"p{n}"
    if "x" in ket.labels():
        m["x"] = "x"
    return graded_bar(ket).relabels(m)


def hop_operator(ns: int, nt: int) -> SymmetricTensor:
    """``-(c_s^+ c_t + h.c.)``: legs ``(P_s, P_t, p_t, p_s)`` -- the creation
    half (s, t) then the annihilation half (t, s)."""
    return bond_operator(_H2, ns, nt)


def gate_operator(ns: int, nt: int, tau: float) -> SymmetricTensor:
    """``exp(tau (c_s^+ c_t + h.c.))`` = ``exp(-tau h)``, legs as
    :func:`hop_operator`."""
    w, v = np.linalg.eigh(_H2.reshape(4, 4))
    return bond_operator((v * np.exp(-tau * w)) @ v.T, ns, nt)


def bond_operator(h2: np.ndarray, ns: int, nt: int) -> SymmetricTensor:
    """``h2[P_s, P_t, p_s, p_t]`` (the local ``(s, t)`` basis) stored with
    the annihilation half reversed; ``<P_s P_t|`` is ``<0| c_t c_s``, so the
    raw transpose carries no sign."""
    data = np.einsum("ABab->ABba", np.reshape(h2, (2, 2, 2, 2)))
    idx = (
        _index(2, IN, f"P{ns}"),
        _index(2, IN, f"P{nt}"),
        _index(2, OUT, f"p{nt}"),
        _index(2, OUT, f"p{ns}"),
    )
    return SymmetricTensor.from_dense(jnp.asarray(data), idx)


def cluster_value(
    R,
    C,
    bra_As,
    ket_As,
    *,
    op_bond=None,
    order="global",
    ket_aux=None,
    bra_aux=None,
    override=None,
):
    """``<bra| O |ket>`` by graded contraction.

    ``override`` maps a site to a prebuilt ket SymmetricTensor (used as both
    ket and bra) -- e.g. after an SVD regauge.
    ``order="global"``: ``bar(K_N) ... bar(K_1) [O] K_1 ... K_N``.
    ``order="per_site"``: ``[bar(K_s) K_s]`` site by site (what a CTM builds),
    then ``O``.  ``ket_aux`` / ``bra_aux`` name the one site whose ket / bra
    tensor is odd; it is stored with the auxiliary leg (see ``ket_site``).
    """
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    touched = set() if op_bond is None else {n_of[op_bond[0]], n_of[op_bond[1]]}
    override = override or {}
    kets = {
        s: override.get(s) or ket_site(R, C, s, ket_As[s], aux=s == ket_aux)
        for s in sites
    }
    bras = {
        s: bra_site(
            override.get(s) or ket_site(R, C, s, bra_As[s], aux=s == bra_aux),
            n_of[s],
            touched=n_of[s] in touched,
        )
        for s in sites
    }
    op = None if op_bond is None else hop_operator(n_of[op_bond[0]], n_of[op_bond[1]])
    if order == "global":
        seq = (
            [bras[s] for s in reversed(sites)]
            + ([op] if op is not None else [])
            + [kets[s] for s in sites]
        )
    elif order == "per_site":
        seq = [graded_contract(bras[s], kets[s]) for s in sites] + (
            [op] if op is not None else []
        )
    else:
        raise ValueError(order)
    t = seq[0]
    for u in seq[1:]:
        t = graded_contract(t, u)
    assert t.ndim == 0, t.labels()
    return complex(np.asarray(t.todense()).reshape(-1)[0])


def cluster_energy(R, C, As, **kw):
    norm = cluster_value(R, C, As, As, **kw).real
    e = sum(
        cluster_value(R, C, As, As, op_bond=(s, t), **kw).real
        for s, _, t, _ in bonds_of(R, C)
    )
    return e / norm, norm


def fock_of_kets(R: int, C: int, kets: dict) -> np.ndarray:
    """The Fock state that graded ket tensors represent, for comparison with
    the oracle -- bond legs of any dimension, parity layout and flow (e.g.
    after an SVD).

    The oracle's bonds are one mode each (index == parity), so every bond
    index is summed explicitly as a flavour, embedded at its parity slot.
    Rule 2 pairs a bond with +1 when the earlier site's leg is OUT and
    ``(-1)^p`` when it is IN; against the oracle's ``(1 + a_t a_s)`` that is
    a Z on the later site's leg exactly when the earlier leg is IN
    (:func:`z_gauge` is the case where every earlier leg is IN).
    """
    sites, bonds = sites_of(R, C), bonds_of(R, C)
    where = {s: {} for s in sites}  # site -> {bond label: leg letter}
    for b, (s, x, t, y) in enumerate(bonds):
        where[s][f"b{b}"], where[t][f"b{b}"] = x, y
    dense, par, flow = {}, {}, {}
    for n, s in enumerate(sites):
        K, labs = kets[s], list(kets[s].labels())
        assert sorted(labs) == sorted([*where[s], f"p{n}"]), (s, labs)
        dense[s] = np.asarray(K.todense())  # one site tensor: small
        par[s] = {lab: ix.charges for lab, ix in zip(labs, K.indices)}
        flow[s] = {lab: ix.flow for lab, ix in zip(labs, K.indices)}
        assert list(par[s][f"p{n}"]) == [0, 1]
    twisted = {  # (site, bond label) carrying the Z
        (t, f"b{b}") for b, (s, _, t, _) in enumerate(bonds) if flow[s][f"b{b}"] == IN
    }
    dims = [len(par[s][f"b{b}"]) for b, (s, *_) in enumerate(bonds)]
    psi = 0
    for f in itertools.product(*map(range, dims)):
        As = {}
        for n, s in enumerate(sites):
            labs = list(kets[s].labels())
            A = dense[s]
            slot, sign = [0, 0, 0, 0], 1.0
            for lab in [lab for lab in labs if lab != f"p{n}"]:
                q = int(par[s][lab][f[int(lab[1:])]])
                slot["udlr".index(where[s][lab])] = q
                if (s, lab) in twisted and q:
                    sign = -sign
            take = tuple(
                f[int(lab[1:])] if lab != f"p{n}" else slice(None) for lab in labs
            )
            dm = [2 if (x in where[s].values()) else 1 for x in "udlr"]
            out = np.zeros([*dm, 2], dtype=A.dtype)
            out[tuple(slot)] = sign * A[take]
            As[s] = out
        psi = psi + fock_psi(R, C, As)
    return psi


def production_site(A: np.ndarray) -> SymmetricTensor:
    """``A[u,d,l,r,p]`` in production's convention: labels
    ``(u, d, l, r, phys)``, flows ``FLOW`` then ``phys`` IN, boundary legs
    kept (dimension 1, parity 0) -- the input of the double-layer builders."""
    idx = [
        TensorIndex.from_charges(
            SYM, np.arange(n, dtype=np.int32) % 2, FLOW[x], label=x
        )
        for n, x in zip(A.shape[:4], "udlr")
    ]
    idx.append(_index(2, IN, "phys"))
    return SymmetricTensor.from_dense(jnp.asarray(A), tuple(idx))


def double_layer_value(R, C, As, *, op=None):
    """``<psi| O |psi>`` from graded double layers
    (:func:`build_graded_double_layer`), contracted site by site with
    ``graded_contract`` -- the per-site order a CTM builds.

    ``op=None`` is the norm.  ``op=((s, t), h2)`` puts the two-site operator
    ``h2[P_s, P_t, p_s, p_t]`` (the local ``(s, t)`` basis, as
    :func:`bond_operator`) on sites ``s, t``, whose double layers keep their
    physical legs open.
    """
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    rename = {s: {} for s in sites}
    for b, (s, x, t, y) in enumerate(bonds_of(R, C)):
        rename[s][f"{x}2"] = rename[t][f"{y}2"] = f"b{b}"
    on = () if op is None else op[0]
    seq = []
    for s in sites:
        n = n_of[s]
        if s in on:
            a = build_graded_double_layer(production_site(As[s]), phys_bra=f"P{n}")
            m = {"phys": f"p{n}"}
        else:
            a = build_graded_double_layer(production_site(As[s]))
            m = {}
        m |= {f"{x}2": f"open_{n}_{x}" for x in "udlr"} | rename[s]
        seq.append(a.relabels(m))
    if op is not None:
        (s, t), h2 = op
        seq.append(bond_operator(h2, n_of[s], n_of[t]))
    out = seq[0]
    for u in seq[1:]:
        out = graded_contract(out, u)
    arr = np.asarray(out.todense()).reshape(-1)  # only dimension-1 legs remain
    assert arr.size == 1, out.labels()
    return complex(arr[0])


def double_layer_energy(R, C, As):
    norm = double_layer_value(R, C, As).real
    e = sum(
        double_layer_value(R, C, As, op=((s, t), _H2)).real
        for s, _, t, _ in bonds_of(R, C)
    )
    return e / norm, norm
