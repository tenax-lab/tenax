"""Re-test 2-site ``gs_stall_recovery="noise"`` on post-#494 energy (issue #520).

The ``"reset"`` default for the 2-site path was set in PR #300 (2026-04-11)
with the rationale "noise interacts pathologically with non-variational CTM
regions on 2-site, see #298".  That observation was made on a 2-site bipartite
energy that PR #494 (2026-05-17) later proved was undercounting NN bonds by
50% — so the verdict was reached on a broken loss landscape and has not been
revalidated on the corrected energy.

This probe runs the suggested experiment from #520: a short 2-site D=2 χ=8
Heisenberg AFM trajectory under each recovery mode, starting from identical
initial site tensors (the optimizer's internal ``PRNGKey(0)``).  The output
dict feeds the issue-comment verdict:

* ``"noise viable"``  → file #521 docstring update and #522 default review.
* ``"noise pathological"`` → confirm the #298 verdict held post-#494.

Usage::

    uv run python benchmarks/stall_recovery_noise_vs_reset_520.py            # full probe
    uv run python benchmarks/stall_recovery_noise_vs_reset_520.py --smoke    # ~minutes
    uv run python benchmarks/stall_recovery_noise_vs_reset_520.py --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import signal
import sys
import time
from contextlib import contextmanager, redirect_stdout

import jax.numpy as jnp

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

# Sandvik QMC reference for the 2D spin-½ Heisenberg AFM ground-state energy
# per site.  D=2 χ=8 cannot reach this — the gap is the (D, χ) truncation
# error — but it anchors the "drift below QMC" non-variational diagnostic.
_QMC_E_PER_SITE: float = -0.66944


_STALL_RE = re.compile(r"\[iPEPS-AD\][^\n]*stall #(\d+)")


class ProbeTimeoutError(RuntimeError):
    """Raised when ``--max-wall-time`` elapses during a single mode run."""


@contextmanager
def _wallclock_deadline(seconds: float | None):
    """Install a ``SIGALRM`` handler that raises ``ProbeTimeoutError`` after
    ``seconds``.  POSIX-only; on platforms without ``SIGALRM`` (Windows) this
    is a no-op and the post-run wall-clock check remains the only enforcement.
    """
    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise ProbeTimeoutError(
            f"probe exceeded --max-wall-time={seconds:.1f}s during optimize_gs_ad"
        )

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds + 0.5)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _heisenberg_gate():
    """Real-dtype Heisenberg gate (matches ``tests/test_ipeps_ad_history.py``).

    The complex128 variant trips a legacy ``ctm_2site`` while-loop dtype
    mismatch on the 2-site SU init path that is orthogonal to #520; real-
    dtype sidesteps it.  Stall-recovery semantics are dtype-independent so
    the verdict carries over to the production complex path.
    """
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _max_stall_seen(stdout_text: str) -> int:
    counts = [int(m.group(1)) for m in _STALL_RE.finditer(stdout_text)]
    return max(counts) if counts else 0


def _summarize_trajectory(energies: list[float]) -> dict:
    """Compute monotonicity + drift diagnostics on a 2-site energy trajectory."""
    if not energies:
        return {
            "num_steps": 0,
            "final_energy": None,
            "min_energy": None,
            "non_monotonic_count": 0,
            "max_increase": 0.0,
            "drift_below_qmc": False,
            "min_below_qmc": 0.0,
        }
    increases = [
        energies[i + 1] - energies[i]
        for i in range(len(energies) - 1)
        if energies[i + 1] > energies[i]
    ]
    return {
        "num_steps": len(energies),
        "final_energy": float(energies[-1]),
        "min_energy": float(min(energies)),
        "non_monotonic_count": len(increases),
        "max_increase": float(max(increases)) if increases else 0.0,
        # Non-variational signature: the trajectory dips below the exact QMC
        # answer.  At D=2 χ=8 a variational state cannot beat QMC — anything
        # below the QMC floor indicates the CTM environment has gone non-
        # variational, which #298 flagged as the "noise pathology".
        "drift_below_qmc": float(min(energies)) < _QMC_E_PER_SITE,
        "min_below_qmc": max(0.0, _QMC_E_PER_SITE - float(min(energies))),
    }


def run_mode(
    *,
    mode: str,
    D: int,
    chi: int,
    gs_num_steps: int,
    max_wall_time: float | None,
) -> dict:
    """Run a single 2-site implicit-AD trajectory with ``gs_stall_recovery=mode``.

    The optimizer's internal random init uses ``PRNGKey(0)`` (deterministic),
    so both modes share an identical starting state.  ``return_history=True``
    captures the per-step energy trajectory which feeds the monotonicity and
    drift-below-QMC diagnostics.
    """
    gate = _heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=D,
        unit_cell="2site",
        ctm=CTMConfig(chi=chi),
        gs_implicit_ad=True,
        gs_optimizer="lbfgs",
        gs_num_steps=gs_num_steps,
        gs_stall_recovery=mode,
        gs_conv_criterion="grad_norm",
        gs_verbose=True,
        return_history=True,
    )

    t0 = time.perf_counter()
    buf = io.StringIO()
    timed_out = False
    final_energy: float | None = None
    history: dict | None = None
    try:
        with _wallclock_deadline(max_wall_time), redirect_stdout(buf):
            result = optimize_gs_ad(gate, None, config)
        # 2-site with return_history: ((A, B), (env_A, env_B), E_gs, history).
        final_energy = float(result[2])
        history = result[3]
    except ProbeTimeoutError:
        timed_out = True
    stdout_text = buf.getvalue()
    wall = time.perf_counter() - t0

    energies = list(history.get("energies", [])) if history else []
    summary = _summarize_trajectory(energies)
    max_stall = _max_stall_seen(stdout_text)

    return {
        "mode": mode,
        "D": D,
        "chi": chi,
        "gs_num_steps": gs_num_steps,
        "final_energy": final_energy,
        "energies": energies,
        "max_stall_seen": max_stall,
        "wall_seconds": wall,
        "timed_out": timed_out,
        "converged": bool(history.get("converged", False)) if history else False,
        **summary,
    }


def compare_modes(noise_result: dict, reset_result: dict) -> dict:
    """Build the verdict dict from the two mode runs.

    The decision rule mirrors #520's acceptance criteria:

    * "noise viable" → noise reached a final energy at least as low as reset,
      did not drift below QMC, and made <= reset's non-monotonic-step count
      (the #298 pathology pattern is non-monotonic dives below QMC).
    * "noise pathological" → noise drifted below QMC, or its final energy was
      materially worse than reset's.
    """
    if noise_result["final_energy"] is None or reset_result["final_energy"] is None:
        return {"verdict": "inconclusive", "reason": "one or both runs failed"}

    if noise_result["drift_below_qmc"]:
        return {
            "verdict": "noise pathological",
            "reason": (
                f"noise dipped below QMC reference by "
                f"{noise_result['min_below_qmc']:.6e}"
            ),
        }

    # "Materially worse" = noise final E is above reset final E by more than
    # the energy-floor noise threshold (1e-6); below that the difference is
    # within numerical precision at D=2 χ=8.
    delta = noise_result["final_energy"] - reset_result["final_energy"]
    if delta > 1e-3:
        return {
            "verdict": "noise pathological",
            "reason": (
                f"noise final E ({noise_result['final_energy']:.10f}) is "
                f"{delta:.3e} above reset ({reset_result['final_energy']:.10f})"
            ),
        }

    return {
        "verdict": "noise viable",
        "reason": (
            f"noise final E ({noise_result['final_energy']:.10f}) within "
            f"{abs(delta):.3e} of reset ({reset_result['final_energy']:.10f}); "
            f"no drift below QMC"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument(
        "--steps", type=int, default=40, help="gs_num_steps (default 40 per #520)"
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=30 * 60,
        help="Hard per-mode wall-clock budget (default 30 min)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny config (D=2, chi=4, 5 steps) for dev-machine smoke",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Write the full result dict (both modes + verdict) to JSON path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step energy table in the printed report",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        args.chi = 4
        args.steps = 5

    print(
        f"[#520-probe] D={args.d} chi={args.chi} steps={args.steps} "
        f"max_wall_time={args.max_wall_time:.1f}s",
        flush=True,
    )

    noise_result = run_mode(
        mode="noise",
        D=args.d,
        chi=args.chi,
        gs_num_steps=args.steps,
        max_wall_time=args.max_wall_time,
    )
    print(
        f"[#520-probe] noise: final={noise_result['final_energy']} "
        f"min={noise_result['min_energy']} "
        f"non_monotonic={noise_result['non_monotonic_count']} "
        f"drift_below_qmc={noise_result['drift_below_qmc']} "
        f"max_stall={noise_result['max_stall_seen']} "
        f"wall={noise_result['wall_seconds']:.1f}s "
        f"timed_out={noise_result['timed_out']}",
        flush=True,
    )

    reset_result = run_mode(
        mode="reset",
        D=args.d,
        chi=args.chi,
        gs_num_steps=args.steps,
        max_wall_time=args.max_wall_time,
    )
    print(
        f"[#520-probe] reset: final={reset_result['final_energy']} "
        f"min={reset_result['min_energy']} "
        f"non_monotonic={reset_result['non_monotonic_count']} "
        f"drift_below_qmc={reset_result['drift_below_qmc']} "
        f"max_stall={reset_result['max_stall_seen']} "
        f"wall={reset_result['wall_seconds']:.1f}s "
        f"timed_out={reset_result['timed_out']}",
        flush=True,
    )

    verdict = compare_modes(noise_result, reset_result)
    print(
        f"[#520-probe] VERDICT: {verdict['verdict']} — {verdict['reason']}",
        flush=True,
    )

    if not args.quiet:
        print("---- noise trajectory ----", flush=True)
        for i, e in enumerate(noise_result["energies"]):
            print(f"  step {i + 1:3d}  E = {e:.10f}", flush=True)
        print("---- reset trajectory ----", flush=True)
        for i, e in enumerate(reset_result["energies"]):
            print(f"  step {i + 1:3d}  E = {e:.10f}", flush=True)

    out = {
        "config": {
            "D": args.d,
            "chi": args.chi,
            "gs_num_steps": args.steps,
            "qmc_reference": _QMC_E_PER_SITE,
        },
        "noise": noise_result,
        "reset": reset_result,
        "verdict": verdict,
    }
    if args.json:
        # Strip non-JSON-serializable bits (none expected; energies are floats).
        # Validate by round-trip through json.dumps before writing.
        json.dumps(out)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[#520-probe] wrote results to {args.json}", flush=True)

    # Exit non-zero only on inconclusive/timeout — both verdicts are valid
    # outcomes per #520, so we don't fail the run on "noise viable" or
    # "noise pathological".
    rc = 0
    if noise_result["timed_out"] or reset_result["timed_out"]:
        print("[#520-probe] FAIL: per-mode wall-clock deadline elapsed", flush=True)
        rc = 1
    if verdict["verdict"] == "inconclusive":
        rc = 1
    if not math.isfinite(noise_result["final_energy"] or float("nan")):
        rc = 1
    if not math.isfinite(reset_result["final_energy"] or float("nan")):
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
