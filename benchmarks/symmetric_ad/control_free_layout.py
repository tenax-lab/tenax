"""Control: AD without the layout seed (ruling R17).

The loss is built with an empty cache (``cache = {}``), i.e. a cold-started
implicit-AD loss, which keeps the frozen cold-init layout (R16).
Twenty plain-GD steps (step 0.01 on normalised tensors); after each step we
record the loss energy, the converged CTM energy at the same tensors (before the step)
(``converged_energy`` adapts the layout), their gap, the forward verdict and
the wall time.  This measures what seeding buys.  Nothing is gated.

Under AD the layout is fixed by tracing (#929), so a "free layout" AD run
does not exist; the loss cache is never written by the loss (Task 4).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from benchmarks.symmetric_ad.pipeline import MODELS, REF_TOL, SYMS, _ctm  # noqa: E402
from tenax import CTMConfig, ipeps, iPEPSConfig  # noqa: E402
from tenax.algorithms import _ctm_energy_ad  # noqa: E402
from tenax.algorithms._symmetric_ad_validation import (  # noqa: E402
    converged_energy,
)
from tenax.algorithms.ctm_layout import (  # noqa: E402
    normalized_pair,
)
from tenax.algorithms.ipeps import xxz_gate  # noqa: E402

STEPS = 20
LR = 0.01


def run_control(model, sym, D, chi, seed, out_dir, *, su_steps=200, steps=STEPS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init, gate_fn = SYMS[sym]
    gate_dense, gate = xxz_gate(MODELS[model]), gate_fn(MODELS[model])
    A0, B0 = init(D, jax.random.PRNGKey(seed))
    cfg = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=su_steps, dt=0.05, ctm=CTMConfig(chi=chi)
    )
    _, (A, B), _ = ipeps(gate_dense, (A0, B0), cfg, compute_energy=False)
    # AD without the seed: cold-started loss (empty cache).
    loss, _ = _cold_loss(gate, _ctm(chi))
    A, B = normalized_pair(A, B)
    rows = []
    for k in range(steps):
        t0 = time.time()
        e, (gA, gB) = jax.value_and_grad(loss)((A, B))
        diag = _ctm_energy_ad.get_last_implicit_ad_diagnostics()
        # Converged CTM at the tensors the loss was evaluated at, then step.
        e_conv, conv = converged_energy(A, B, gate, _ctm(chi), None, REF_TOL)
        A, B = A - LR * gA, B - LR * gB
        A, B = normalized_pair(A, B)
        rows.append(
            {
                "step": k,
                "loss_E": float(e),
                "converged_E": e_conv,
                "gap": float(e) - e_conv,
                "converged": conv,
                "forward_converged": diag.get("forward_converged"),
                "wall": time.time() - t0,
            }
        )
        (out_dir / "control_partial.json").write_text(
            json.dumps({"steps": rows}, indent=2, default=str)
        )
    rec = {
        "model": model,
        "sym": sym,
        "D": D,
        "chi": chi,
        "seed": seed,
        "steps": rows,
    }
    (out_dir / "control.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def _cold_loss(gate, ctm_cfg):
    """The optimizer's 2-site loss with an empty env cache (no seed)."""
    from tenax.algorithms._ctm_tensor_convergence import (  # noqa: PLC0415
        CHECKERBOARD_NEIGHBORS,
    )
    from tenax.algorithms._ctm_tensor_energy import (  # noqa: PLC0415
        compute_energy_ctm_tensor_2site,
    )
    from tenax.algorithms.ctm_layout import _dense_gate  # noqa: PLC0415
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: PLC0415

    g = _dense_gate(gate)
    cache = {}

    def _energy_fn(sites, envs, gate_):
        return compute_energy_ctm_tensor_2site(
            sites[(0, 0)],
            sites[(1, 0)],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--sym", choices=sorted(SYMS), required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--su-steps", type=int, default=200)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cache_dir is not None:
        a.cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(a.cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    run_control(a.model, a.sym, a.D, a.chi, a.seed, a.out, su_steps=a.su_steps)


if __name__ == "__main__":
    main()
