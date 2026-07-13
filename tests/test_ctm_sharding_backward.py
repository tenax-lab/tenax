"""Rung-2 gate 2A (CI-able): the GSPMD-sharded dense CTM-AD *backward* produces
the same gradient as single-device.

Drives a fake-CPU-device subprocess (mirrors the rung-1 forward parity test).
Guards that GSPMD propagates through the implicit-AD custom_vjp + lax.while_loop
adjoint correctly: on a well-conditioned state the sharded gradient matches the
single-device gradient to machine precision.
"""

import os
import subprocess
import sys


def test_sharded_backward_grad_matches_single_device():
    """value_and_grad under a 4-device GSPMD mesh equals the single-device
    gradient to a tight tolerance (well-conditioned state; fake CPU devices,
    subprocess).

    Tolerance (#692): sharding reassociates the D²-axis sums in both the forward
    and the adjoint, so the sharded-vs-single gradient delta is O(eps*kappa).
    On a well-conditioned state that floor is machine-precision (~1e-15)
    locally, but **platform-dependent** — CI runners land at ~3e-7 on the
    gradient (energy delta ~9e-10) on this same D=4 χ=8 state (measured on the
    #692 Full-tests matrix), so the original 1e-8 gate was not portable. 1e-5
    clears the observed CI gradient noise by ~36x (relative to |grad|~1.8e-2
    that is ~5e-4 relative) while a real sharding/adjoint bug would give O(1)
    relative error. The single-device fixed point is stable across PR #676, so
    this is a tolerance-portability fix, not a masked regression.
    """
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    r = subprocess.run(
        [sys.executable, "tests/_rung2_grad_parity_subproc.py", "4", "8", "1e-5"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, (
        f"grad parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    )


def test_sharded_optimize_gs_ad_matches_single_device():
    """End-to-end: ``CTMConfig.device_mesh`` routes the whole implicit-AD
    optimization through GSPMD sharding with no change in result. A few
    ``optimize_gs_ad`` steps from a well-conditioned init, single vs 4-device
    sharded, must reach the same energy + tensor to <1e-8 (fake CPU devices)."""
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    r = subprocess.run(
        [sys.executable, "tests/_rung2_optimize_parity_subproc.py", "2", "8"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, (
        f"optimize parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    )
