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
    gradient to <1e-8 (well-conditioned state; fake CPU devices, subprocess)."""
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    r = subprocess.run(
        [sys.executable, "tests/_rung2_grad_parity_subproc.py", "4", "8", "1e-8"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, f"grad parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"


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
    assert r.returncode == 0, f"optimize parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
