"""Shared helpers for validating symmetric 2-site CTM AD (spec 2026-10-09).

Private: used by tests and benchmarks/symmetric_ad only.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms.ctm_layout import _dense_gate, energy_2site, normalized_pair
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs, make_ctm_energy_fn
from tenax.core.tensor import DenseTensor, Tensor

_FD_HS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def make_2site_loss(gate, ctm_cfg, envs_init):
    """The optimizer's 2-site loss (``ipeps_optimize.py`` loss_fn), seeded
    with ``envs_init`` so the traced projector inherits its layout.

    Returns ``(loss, cache)``; ``loss((A, B))`` normalises each tensor and
    returns the fused implicit-AD energy, and ``cache["envs"]`` is the
    environment the loss starts from.
    """
    chis = sorted({int(e.C1.indices[0].dim) for e in envs_init.values()})
    if chis != [ctm_cfg.chi]:
        shown = chis[0] if len(chis) == 1 else chis
        raise ValueError(
            f"envs_init chi {shown} does not match ctm_cfg.chi {ctm_cfg.chi}"
        )
    g = _dense_gate(gate)
    cache = {"envs": envs_init}

    def _energy_fn(site_tensors, envs, gate_):
        return compute_energy_ctm_tensor_2site(
            site_tensors[(0, 0)],
            site_tensors[(1, 0)],
            envs[(0, 0)],
            envs[(1, 0)],
            gate_,
            gate_.shape[0],
        )

    f = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS,
        gate=g,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache=cache,
        use_explicit=False,
        explicit_warmup=0,
        explicit_steps=0,
        energy_fn=_energy_fn,
    )

    def loss(params):
        A_n, B_n = normalized_pair(*params)
        return f({(0, 0): A_n, (1, 0): B_n})

    return loss, cache


def _densify(t: Tensor) -> DenseTensor:
    return DenseTensor(t.todense(), t.indices)


def densify_pair(A: Tensor, B: Tensor) -> tuple[DenseTensor, DenseTensor]:
    return _densify(A), _densify(B)


def densify_envs(envs: dict) -> dict:
    """Small objects only (D=2, chi<=8 fixtures): every env tensor densified."""
    return {c: CTMTensorEnv(*(_densify(t) for t in env)) for c, env in envs.items()}


def oracle_value_and_grad(A, B, gate, ctm_cfg, envs_init):
    """Approach C: the same loss on densified tensors and environment.

    The site tensors are densified *inside* the differentiated function, so
    the CTM, its projectors and their SVD backward all run on
    ``DenseTensor``s (``_compute_2x2_projector`` routes to the block-sparse
    pipeline only when an input is a ``SymmetricTensor``) while the gradient
    is taken with respect to the symmetric blocks: it is ``P^T`` of the dense
    gradient, ``P`` the embedding of the block data.  The off-block part of
    the dense gradient is discarded: it is not a direction of the symmetric
    loss, and where a singular value is degenerate across two sectors the
    dense SVD backward fills it with ``1/(s_i^2 - s_j^2)`` noise (measured
    |g_off| ~ 1.7e4 against |g| ~ 0.13 on the U(1) D=2 chi=4 fixture).

    Returns ``(energy, (gA, gB))`` with the gradients as dense arrays.  The
    loss is real, so the raw cotangent is the Euclidean gradient
    (``_euclidean_grads`` is ``conj``, the identity on real leaves).
    """
    loss, _ = make_2site_loss(gate, ctm_cfg, densify_envs(envs_init))
    e, (gA, gB) = jax.value_and_grad(lambda p: loss(densify_pair(*p)))((A, B))
    return float(e), (jnp.asarray(gA.todense()), jnp.asarray(gB.todense()))


def _arr(t):
    return t.todense() if isinstance(t, Tensor) else t


def grad_dot(g, d) -> float:
    """Real inner product of two (A, B) pairs, densified per site."""
    return float(
        sum(
            jnp.vdot(
                jnp.ravel(jnp.asarray(_arr(a))), jnp.ravel(jnp.asarray(_arr(b)))
            ).real
            for a, b in zip(g, d, strict=True)
        )
    )


def converged_energy(A, B, gate, ctm_cfg, env_init):
    """Energy at normalised (A, B) from a tight eager CTM warm-started at
    ``env_init``; returns ``(energy, converged)``."""
    tight = dataclasses.replace(
        ctm_cfg,
        conv_tol=1e-12,
        max_iter=1000,
        plateau_patience=None,
        on_unconverged="warn",
    )
    A_n, B_n = normalized_pair(A, B)
    envs, info = python_loop_ctm_converge(
        {(0, 0): A_n, (1, 0): B_n},
        CHECKERBOARD_NEIGHBORS,
        **ctm_converge_kwargs(tight, env_init=env_init),
    )
    return energy_2site(A_n, B_n, envs, gate), bool(info.converged)


def fd_directional(A, B, dA, dB, gate, ctm_cfg, envs_init, hs=_FD_HS):
    """Central FD of the loss along (dA, dB) with an h scan.

    Each E(+-h) is a converged CTM.  Over the h values whose two solves both
    converged, ``fd`` is the median of the three adjacent ones with the
    smallest spread, ``fd_unc`` that spread, ``h_best`` the middle h.
    """
    scan = []
    for h in hs:
        ep, cp = converged_energy(A + h * dA, B + h * dB, gate, ctm_cfg, envs_init)
        em, cm = converged_energy(A - h * dA, B - h * dB, gate, ctm_cfg, envs_init)
        scan.append((h, (ep - em) / (2 * h), cp and cm))
    ok = [s for s in scan if s[2]]
    best = None
    for i in range(len(ok) - 2):
        vals = [ok[i + k][1] for k in range(3)]
        spread = max(vals) - min(vals)
        if best is None or spread < best[0]:
            best = (spread, float(np.median(vals)), ok[i + 1][0])
    if best is None:
        return {
            "fd": float("nan"),
            "fd_unc": float("inf"),
            "h_best": None,
            "scan": scan,
        }
    return {"fd": best[1], "fd_unc": best[0], "h_best": best[2], "scan": scan}
