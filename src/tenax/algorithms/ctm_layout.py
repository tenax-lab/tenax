"""Chi sector layout of a CTM environment, and an eager seed that fixes it.

The layout is the per-sector size of every environment leg.  Under AD the
traced projector keeps the layout it is seeded with (#929), so a run seeded
from a converged eager environment (``envs_init``) has a frozen layout.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.core.tensor import SymmetricTensor, Tensor

Layout = dict

_FIELDS = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")


def layout_of(envs: dict) -> Layout:
    """Map (coord, field, leg label) to its sorted ((charge, size), ...)."""
    out: Layout = {}
    for coord in sorted(envs):
        env = envs[coord]
        for field in _FIELDS:
            for ix in getattr(env, field).indices:
                pairs = zip(
                    np.asarray(ix.sectors).tolist(),
                    np.asarray(ix.multiplicities).tolist(),
                )
                out[(coord, field, ix.label)] = tuple(sorted(pairs))
    return out


def same_layout(a: Layout, b: Layout) -> bool:
    """True iff every leg has the same (charge, size) pairs."""
    return a == b


def normalized_pair(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """The normalisation the 2-site loss applies (``ipeps_optimize`` loss_fn)."""
    return A * (1.0 / (A.norm() + 1e-10)), B * (1.0 / (B.norm() + 1e-10))


def _dense_gate(gate):
    return gate.todense() if isinstance(gate, Tensor) else jnp.asarray(gate)


def energy_2site(A_n: Tensor, B_n: Tensor, envs: dict, gate) -> float:
    g = _dense_gate(gate)
    return float(
        compute_energy_ctm_tensor_2site(
            A_n, B_n, envs[(0, 0)], envs[(1, 0)], g, g.shape[0]
        )
    )


def _converge(st, cfg, env_init):
    return python_loop_ctm_converge(
        st, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg, env_init=env_init)
    )


def _corner_values(t: Tensor) -> np.ndarray:
    if isinstance(t, SymmetricTensor):
        vals = [
            np.linalg.svd(np.asarray(b), compute_uv=False) for b in t.blocks.values()
        ]
        s = np.concatenate(vals) if vals else np.zeros(0)
    else:
        s = np.linalg.svd(np.asarray(t.todense()), compute_uv=False)
    return np.sort(s)[::-1]


def _cut_gap(envs: dict, chi: int) -> float:
    gaps = []
    for env in envs.values():
        for field in ("C1", "C2", "C3", "C4"):
            s = _corner_values(getattr(env, field))
            if len(s) > chi and s[chi - 1] > 0:
                gaps.append((s[chi - 1] - s[chi]) / s[chi - 1])
    return float(min(gaps)) if gaps else float("inf")


class SeedResult(NamedTuple):
    envs_init: dict
    layout: Layout
    energy_cold: float
    energy_forced: float
    stable: bool
    cut_gap: float
    cold_converged: bool
    cold_iterations: int


def symmetric_layout_seed(
    A: Tensor,
    B: Tensor,
    gate,
    ctm_cfg,
    *,
    forced_sweeps: int = 300,
    energy_tol: float = 1e-10,
    gap_extra_chi: int = 4,
) -> SeedResult:
    """Converge an eager 2-site CTM cold, check it with forced sweeps, and
    return it as ``envs_init`` with its layout.

    ``stable`` is False when ``forced_sweeps`` more sweeps move the energy by
    ``energy_tol`` or more, or change the layout: a cold CTM can certify
    convergence on a point it later leaves (#1096).
    """
    A_n, B_n = normalized_pair(A, B)
    st = {(0, 0): A_n, (1, 0): B_n}
    cold, info = _converge(st, ctm_cfg, None)
    e_cold = energy_2site(A_n, B_n, cold, gate)
    forced_cfg = dataclasses.replace(
        ctm_cfg,
        conv_tol=0.0,
        max_iter=forced_sweeps,
        min_iter=forced_sweeps,
        plateau_patience=None,
        on_unconverged="warn",
    )
    forced, _ = _converge(st, forced_cfg, cold) if forced_sweeps > 0 else (cold, None)
    e_forced = energy_2site(A_n, B_n, forced, gate)
    lay = layout_of(cold)
    stable = abs(e_forced - e_cold) < energy_tol and same_layout(lay, layout_of(forced))
    big_cfg = dataclasses.replace(
        ctm_cfg, chi=ctm_cfg.chi + gap_extra_chi, on_unconverged="warn"
    )
    big, _ = _converge(st, big_cfg, None)
    return SeedResult(
        envs_init=cold,
        layout=lay,
        energy_cold=e_cold,
        energy_forced=e_forced,
        stable=bool(stable),
        cut_gap=_cut_gap(big, ctm_cfg.chi),
        cold_converged=bool(info.converged),
        cold_iterations=int(info.iterations),
    )
