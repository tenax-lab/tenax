"""Build #1038's oracle clusters out of SymmetricTensors and contract them with
tenax.core._graded, so the graded contractor can be checked against Fock."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _fermionic_fock_oracle import _H2, bonds_of, sites_of

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
    data = np.einsum("ABab->ABba", _H2)
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
