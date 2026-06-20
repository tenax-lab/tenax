import os
import subprocess
import sys

import pytest

# Importable when pytest runs this file directly (prepend import mode puts the
# test dir on sys.path); guard the insert so it also works if invoked some other
# way.  Same import the subprocess driver uses.
sys.path.insert(0, os.path.dirname(__file__))
from _ctm_sharding_probe import heisenberg_dense_probe_energy

# Per device-count probe size: D² must divide the device count so every
# sharded axis (the double-layer D² legs AND the chi legs of the edges) splits
# evenly, otherwise XLA raises an IndivisibleError.
#   N=2: D=2 → D²=4 divisible by 2, chi=8 divisible by 2.
#   N=4: D=4 → D²=16 divisible by 4, chi=4 divisible by 4 (D=2 would give
#        D²=4 / 4 = 1 per device, which works, but the raw-D=2 legs the absorb
#        also touches are not divisible by 4 — D=4 shards cleanly everywhere).
_PARITY_PROBE = {2: (2, 8), 4: (4, 4)}


@pytest.mark.parametrize("n_dev", [2, 4])
def test_sharded_forward_matches_single_device(n_dev):
    """Dense CTM energy under an N-device GSPMD mesh equals the single-device
    result to <1e-8 (subprocess with fake CPU devices)."""
    # ``--xla_force_host_platform_device_count=N`` fabricates N devices on the
    # CPU platform only, so the subprocess MUST run on CPU.  On a GPU box JAX
    # otherwise defaults to the single real GPU, the N-way mesh all-gather
    # never finds its peers, and the child aborts on a rendezvous timeout.
    # ``JAX_PLATFORMS=cpu`` pins the platform so the fake-device trick (and the
    # GSPMD sharding it exercises) works regardless of host accelerators.
    D, chi = _PARITY_PROBE[n_dev]
    env = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
        JAX_PLATFORMS="cpu",
    )
    r = subprocess.run(
        [sys.executable, "tests/_ctm_sharding_parity_subproc.py", str(D), str(chi)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, f"parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"


def test_flag_off_is_unchanged():
    """device_mesh=None must give the exact same energy as not passing it.

    The sharding flag is opt-in: ``device_mesh=None`` (the default) must be a
    literal no-op, so the energy is bit-identical to omitting the argument.
    Both calls run single-device, so no fake-device env is needed.
    """
    e_default = heisenberg_dense_probe_energy(D=2, chi=8, seed=1)
    e_none = heisenberg_dense_probe_energy(D=2, chi=8, device_mesh=None, seed=1)
    assert e_default == e_none
