#!/usr/bin/env python3
"""Prototype: lax.scan-fused, fixed-shape, jittable CTM step (the #569 lever).

The #569 analysis showed batch_blocksparse fixes neither runtime (host-dispatch-
bound eager vjp) nor compile (symmetric op-count blows up). The real lever is to
run the CTM iteration as a *single compiled body* over *fixed shapes* via
``jax.lax.scan`` -- so the whole step is jittable end-to-end AND the compile time
is bounded (independent of the number of iterations), unlike a Python-unrolled
loop whose compiled graph grows with the iteration count.

This prototype demonstrates that on the classical 2D Ising CTMRG (C4v symmetric),
which is validatable against the exact Onsager magnetization. It shows:

  1. correctness  -- magnetization matches Onsager in the ordered phase;
  2. jittable + differentiable  -- d<m>/dbeta via jax.grad through the scan;
  3. bounded compile  -- compile(scan, length=N) is ~flat in N, while
                         compile(python-unrolled N) grows ~linearly.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Classical 2D Ising bulk tensor (C4v symmetric) and exact references
# --------------------------------------------------------------------------- #
def ising_bulk(beta, spin_insert=False, h=0.0):
    """Square-lattice Ising bulk tensor a[u, l, d, r] (each leg dim 2).

    ``a = sum_s g(s) W[s,u] W[s,l] W[s,d] W[s,r]`` with W = sqrt(B),
    B[s,s'] = exp(beta s s'), s in {+1,-1}. ``h`` is a small longitudinal field
    (site weight ``exp(beta h s)``) that breaks the up/down symmetry so the
    ordered phase polarizes -- without it the CTMRG converges to the symmetric
    <sigma>=0 fixed point and the eigh backward sees degenerate eigenvalues.
    With ``spin_insert`` the site sum is weighted by ``s`` (magnetization
    numerator).
    """
    s = jnp.array([1.0, -1.0])
    B = jnp.exp(beta * jnp.outer(s, s))  # 2x2 bond Boltzmann
    w, V = jnp.linalg.eigh(B)
    W = V @ jnp.diag(jnp.sqrt(w)) @ V.T  # symmetric sqrt(B)
    site = jnp.exp(beta * h * s)  # symmetry-breaking field
    weight = (s * site) if spin_insert else site
    return jnp.einsum("s,su,sl,sd,sr->uldr", weight, W, W, W, W)


def onsager_magnetization(beta):
    """Exact spontaneous magnetization (ordered phase, beta > beta_c)."""
    m8 = 1.0 - 1.0 / np.sinh(2.0 * beta) ** 4
    return float(m8 ** 0.125) if m8 > 0 else 0.0


# --------------------------------------------------------------------------- #
# Lorentzian-regularized symmetric eigh (stable backward at degeneracies)
# --------------------------------------------------------------------------- #
@jax.custom_vjp
def safe_eigh(A):
    w, V = jnp.linalg.eigh(A)
    return w, V  # plain tuple (must match the fwd rule's primal output)


def _safe_eigh_fwd(A):
    w, V = jnp.linalg.eigh(A)
    return (w, V), (w, V)


def _safe_eigh_bwd(res, g):
    """Standard symmetric-eigh VJP with a Lorentzian-regularized 1/(w_i - w_j).

    The C4v corner has degenerate eigenvalues, where the bare ``1/(w_i - w_j)``
    is 0/0 -> NaN. Replace it with ``(w_i-w_j)/((w_i-w_j)^2 + eps)`` (Francuz/
    iPEPS-AD trick) so the gradient stays finite through the projector.
    """
    w, V = res
    gw, gV = g
    eps = 1e-12
    diff = w[:, None] - w[None, :]
    F = diff / (diff * diff + eps)  # regularized 1/(w_i - w_j); 0 on diagonal
    VtgV = V.T @ gV
    inner = jnp.diag(gw) + F * VtgV
    dA = V @ inner @ V.T
    return (0.5 * (dA + dA.T),)


safe_eigh.defvjp(_safe_eigh_fwd, _safe_eigh_bwd)


# --------------------------------------------------------------------------- #
# CTMRG move (C4v: one corner C[chi,chi], one edge T[chi,chi,d])
# --------------------------------------------------------------------------- #
def ctm_step(C, T, a):
    """One fixed-shape C4v CTMRG renormalization: (C, T) -> (C', T').

    Enlarged corner -> Hermitian eigen-projector (truncate to chi) -> renormalize
    corner and edge. Shapes in == shapes out, so this is a valid scan body.
    """
    chi = C.shape[0]
    d = a.shape[0]
    # Enlarged corner Cp[(A,Dn),(Bo,Rn)] from C, top-edge T, left-edge T, bulk a.
    Cp = jnp.einsum("ij,iaU,jbL,ULdr->adbr", C, T, T, a)  # (chi,d,chi,d)
    Cp = Cp.reshape(chi * d, chi * d)
    Cp = 0.5 * (Cp + Cp.T)  # symmetric (C4v) -> Hermitian eigh
    w, Z = safe_eigh(Cp)  # Lorentzian-regularized backward (degenerate-safe)
    # keep the chi largest-magnitude eigenvalues -> isometry P (chi*d, chi)
    order = jnp.argsort(-jnp.abs(w))[:chi]
    P = Z[:, order]  # (chi*d, chi)
    C_new = (P.T @ Cp @ P)
    C_new = C_new / jnp.max(jnp.abs(C_new))
    # Edge renormalization: absorb one bulk a into T, project both bond legs.
    # T[i,a,u]; bulk a[u,l,D,r]: u contracts T's phys, l & r are the bond-side
    # bulk legs paired across the two projectors, D is the new physical leg.
    Tmid = jnp.einsum("iau,ulDr->iDlar", T, a)  # i,a bond; D new phys; l,r legs
    # group rows (i, l) and cols (a, r): but l is a bulk leg of size d that must
    # pair across the two projectors. Use P reshaped to (chi, d, chi).
    Pr = P.reshape(chi, d, chi)  # (bond, phys, new_bond)
    # T_new[A,B,D] = sum_{i,l,a,r} Pr[i,l,A] Tmid[i,D,l,a,r] Pr[a,r,B]
    T_new = jnp.einsum("ilA,iDlar,arB->ABD", Pr, Tmid, Pr)
    T_new = T_new / jnp.max(jnp.abs(T_new))
    return C_new, T_new


def magnetization(C, T, a, a_sigma):
    """<sigma> = Z1(a_sigma) / Z1(a), where Z1 is the 1-site environment closure."""

    def z1(bulk):
        return jnp.einsum(
            "ab,cd,ef,gh,bcp,deq,fgr,has,pqrs->",
            C, C, C, C, T, T, T, T, bulk,
        )

    return z1(a_sigma) / z1(a)


def init_ctm(a, chi):
    """Initialize C (chi,chi) and T (chi,chi,d) from the bulk, padded to chi."""
    d = a.shape[0]
    C0 = jnp.einsum("uldr->ul", a)  # (d,d) corner = trace two legs (d,r)
    T0 = jnp.einsum("uldr->uld", a)  # T[bond_u, bond_l, phys_d]: sum over r
    C = jnp.zeros((chi, chi)).at[:d, :d].set(C0 / jnp.max(jnp.abs(C0)))
    T = jnp.zeros((chi, chi, d)).at[:d, :d, :].set(T0 / jnp.max(jnp.abs(T0)))
    return C, T


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def converge_scan(beta, chi, n_steps, h=0.0):
    """Scan-fused fixed-shape CTMRG -> magnetization (jittable, differentiable)."""
    a = ising_bulk(beta, h=h)
    a_sig = ising_bulk(beta, spin_insert=True, h=h)
    C, T = init_ctm(a, chi)

    def body(carry, _):
        C, T = carry
        return ctm_step(C, T, a), None

    (C, T), _ = jax.lax.scan(body, (C, T), None, length=n_steps)
    return magnetization(C, T, a, a_sig)


def compile_seconds(fn, *args):
    t0 = time.perf_counter()
    jax.jit(fn).lower(*args).compile()
    return time.perf_counter() - t0


def main():
    beta = 0.5  # ordered phase (beta_c ~ 0.4407); Onsager m here is well-defined
    chi = 16
    H = 0.02  # small symmetry-breaking field (m -> spontaneous m as H -> 0+)
    a = ising_bulk(beta, h=H)
    a_sig = ising_bulk(beta, spin_insert=True, h=H)

    print(f"# 2D Ising CTMRG prototype (scan-fused) | beta={beta} chi={chi} h={H}")
    print(f"# device: {jax.devices()[0].platform}")

    # (1) correctness: magnetization vs Onsager (small field -> ~spontaneous m)
    m = float(converge_scan(beta, chi, 80, h=H))
    m_exact = onsager_magnetization(beta)
    print(f"\n[1] correctness:  <sigma>_CTMRG = {m:.6f}   Onsager(h=0) = {m_exact:.6f}"
          f"   |diff| = {abs(m - m_exact):.2e}")

    # (2) jittable + differentiable end-to-end
    g = jax.grad(lambda b: converge_scan(b, chi, 80, h=H))(beta)
    print(f"[2] jittable + differentiable:  d<sigma>/dbeta = {float(g):+.4f} "
          f"(finite: {np.isfinite(float(g))})")

    # (3) bounded compile: scan(N) vs python-unrolled(N)
    C0, T0 = init_ctm(a, chi)

    def scan_fn(C, T, N):
        (C, T), _ = jax.lax.scan(lambda c, _: (ctm_step(c[0], c[1], a), None),
                                 (C, T), None, length=N)
        return magnetization(C, T, a, a_sig)

    def unrolled_fn(C, T, N):
        for _ in range(N):
            C, T = ctm_step(C, T, a)
        return magnetization(C, T, a, a_sig)

    print("\n[3] compile time (trace+XLA), scan-fused vs python-unrolled:")
    print(f"{'N steps':>8} {'scan_s':>9} {'unrolled_s':>12}")
    for N in (4, 16, 64, 128):
        cs = compile_seconds(lambda C, T: scan_fn(C, T, N), C0, T0)
        cu = compile_seconds(lambda C, T: unrolled_fn(C, T, N), C0, T0)
        print(f"{N:>8} {cs:>8.2f}s {cu:>11.2f}s")
    print("\n-> scan_s ~flat in N (one compiled body); unrolled_s grows with N. "
          "That is the compile-wall fix: a jittable, fixed-shape CTM-AD step.")


if __name__ == "__main__":
    main()
