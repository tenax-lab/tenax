"""iPEPS D=8 square-lattice Heisenberg AFM: simple-update seed + forward-CTM
χ-scan showing the single-GPU memory wall and the #632 multi-GPU rescue.

    # full run (orchestrator): SU seed once, then scan χ × {1,2} GPU
    uv run python examples/heisenberg_d8_chi_scaling.py --outdir runs/d8_chi_scaling

    # quick validation (tiny end-to-end)
    uv run python examples/heisenberg_d8_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator):
    uv run python examples/heisenberg_d8_chi_scaling.py --cell --phase scan \
        --chi 64 --n-devices 1 --outdir runs/d8_chi_scaling --out /tmp/cell.json

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.

The D-agnostic formatting/plot/IO/mesh helpers are reused from the sibling D=4
driver (path-loaded as ``d4``) so the merged file is not modified.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

# Path-load the D=4 driver to reuse its D-agnostic pure helpers. Its top level
# imports only stdlib (jax/tenax live inside functions), so this stays jax-free.
_D4_PATH = pathlib.Path(__file__).resolve().parent / "heisenberg_d4_chi_scaling.py"
_d4_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _D4_PATH)
d4 = importlib.util.module_from_spec(_d4_spec)
_d4_spec.loader.exec_module(d4)

REFERENCE_E = d4.REFERENCE_E  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM
D = 8  # fixed bond dimension for this driver


def _parse_nvidia_smi(text):
    """Parse `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu
    --format=csv,noheader,nounits` into (index, name, mem_used_mib, util_pct)
    tuples. Lines that don't have all four fields are skipped."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem, util = parts[0], parts[1], parts[2], parts[3]
        try:
            rows.append((int(idx), name, int(float(mem)), int(float(util))))
        except ValueError:
            continue  # header row or [N/A] field -> skip, per the skip contract
    return rows


def select_free_a100s(rows, n, mem_threshold_mib=2048, util_threshold=50):
    """The n most-idle 80 GB A100 indices from parsed nvidia-smi rows.

    Idle = an A100 (never the DGX Display GPU) with memory.used and
    utilization below the thresholds. Sorted by (memory.used, index) so the
    most-idle device comes first; deterministic tiebreak by index. Raises
    RuntimeError if fewer than n idle A100s are available (so a row stops
    rather than landing on a busy or display GPU)."""
    free = [
        r for r in rows
        if "A100" in r[1] and "Display" not in r[1]
        and r[2] <= mem_threshold_mib and r[3] <= util_threshold
    ]
    free.sort(key=lambda r: (r[2], r[0]))
    if len(free) < n:
        raise RuntimeError(
            f"need {n} idle A100s, found {len(free)}: "
            + ", ".join(f"gpu{r[0]}({r[2]}MiB,{r[3]}%)" for r in free)
        )
    return [r[0] for r in free[:n]]


def free_a100_indices(n, mem_threshold_mib=2048, util_threshold=50):
    """Query nvidia-smi and return the n most-idle A100 indices."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return select_free_a100s(_parse_nvidia_smi(out), n, mem_threshold_mib, util_threshold)


def cuda_visible_for(n_devices):
    """CUDA_VISIBLE_DEVICES string pinning the n most-idle A100s right now."""
    return ",".join(str(i) for i in free_a100_indices(n_devices))


def build_grid(chi_ladder, device_counts):
    """Scan cells in device-major, chi-minor order (one row per n_devices).
    Reuses the D=4 module's frozen ``Cell`` dataclass with D=8."""
    return [
        d4.Cell(D=D, chi=chi, n_devices=n)
        for n in device_counts
        for chi in chi_ladder
    ]
