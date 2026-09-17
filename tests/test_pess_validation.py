"""End-to-end validation smoke tests for the kagome iPESS AD pipeline.

The smoke tests run a short SU + AD optimization at small ``D`` and
assert the per-kagome-site energy lands in the realistic ``D=2`` window
for each model. They are slow (CTM + L-BFGS at every step) and stay out
of the ``-m core`` required-CI bucket.

The published large-``D`` references (Liao 2019: −0.4378 spin-½; Picot
2016: ≈ −1.41 spin-1) are NOT achievable at ``D=2`` — at this bond
dimension the iPESS variational manifold is too small. The smoke
windows below are calibrated to the actual ``D=2`` behaviour: spin-½
locks onto the classical 120° energy ``−0.25`` (because the variational
manifold collapses at this resolution), spin-1 reaches roughly
``−1.0``. Anything outside these windows means a regression in the
pipeline (energy contract, supersite construction, or AD plumbing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _import_from_examples(module_name: str):
    """Side-effect import of an example module under ``examples/``."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    try:
        return __import__(module_name)
    finally:
        sys.path.pop(0)


@pytest.mark.slow
def test_kagome_spin12_d2_smoke():
    """Spin-½ kagome AFM at D=2, χ=8: per-site energy in ``[-0.42, -0.35]``.

    Re-pointed for #1002: the old window ``[-0.30, -0.20]`` encoded the
    Convention-C readout (rank-1-collapsed CTM; its "classical 120° AFM
    fixed point at ``E/site = -0.25``" was an artifact of the broken
    probe, not a property of the state). Through the exact blocking the
    same short-schedule pipeline reads ``E/site = -0.386`` (measured
    -0.386135 on CPU; the full-schedule SU state reads -0.386195,
    backend-identical, matching variPEPS to 1e-9 — see
    test_pess_supersite_exact). The window excludes the collapsed
    readouts (≈ -0.24..-0.25) above and unphysical drift below Liao
    2017's large-D limit -0.43752(6).
    """
    mod = _import_from_examples("kagome_spin12_pess_ad_benchmark")
    _, e_ad, _ = mod.run_kagome_spin12_benchmark(
        D=2,
        chi=8,
        max_iter=10,
        su_steps=((0.05, 100), (0.005, 50)),
    )
    assert -0.42 < e_ad < -0.35, f"E/site={e_ad:.6f} outside [-0.42, -0.35]"


@pytest.mark.slow
def test_kagome_spin1_d2_smoke():
    """Spin-1 kagome Heisenberg at D=2, χ=8: per-site energy in ``[-1.35, -1.20]``.

    Re-pointed for #1002: the old window ``[-1.20, -1.00]`` (and its
    "``D=2`` lands around ``-1.13``" calibration) encoded the
    Convention-C readout of the rank-1-collapsed CTM. Through the exact
    blocking the same short-schedule pipeline reads ``E/site = -1.270``
    (measured -1.270160 on CPU; the full-schedule SU state reads
    -1.270151 on GPU and agrees with the independent Husimi-tree probe
    ``pess_local_energy`` to 2.4e-4). The window excludes the old
    collapsed readouts (≈ -1.0..-1.13) above and unphysical drift below
    Picot 2016's large-D target ``≈ -1.41``.
    """
    mod = _import_from_examples("kagome_spin1_pess_ad_benchmark")
    _, e_ad, _ = mod.run_kagome_spin1_benchmark(
        D=2,
        chi=8,
        max_iter=5,
        su_steps=((0.1, 100), (0.01, 50)),
    )
    assert -1.35 < e_ad < -1.20, f"E/site={e_ad:.6f} outside [-1.35, -1.20]"
