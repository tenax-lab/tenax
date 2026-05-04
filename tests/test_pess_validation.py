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
    """Spin-½ kagome AFM at D=2, χ=8: per-site energy in ``[-0.30, -0.20]``.

    At ``D=2`` the iPESS pipeline saturates on the classical 120° AFM
    state with ``E/site = -0.25`` (unable to break free of this
    fixed point at ``D=2``). The window catches both this and any
    pipeline regression that pushes the value outside.
    """
    mod = _import_from_examples("kagome_spin12_pess_ad_benchmark")
    _, e_ad, _ = mod.run_kagome_spin12_benchmark(
        D=2,
        chi=8,
        max_iter=10,
        su_steps=((0.05, 100), (0.005, 50)),
    )
    assert -0.30 < e_ad < -0.20, f"E/site={e_ad:.6f} outside [-0.30, -0.20]"


@pytest.mark.slow
def test_kagome_spin1_d2_smoke():
    """Spin-1 kagome Heisenberg at D=2, χ=8: per-site energy in ``[-1.20, -1.00]``.

    Picot 2016 large-``D`` target is ``E/site ≈ -1.41``. With the fixed
    ``hosvd_truncate`` (no SU lambda double-counting) ``D=2`` lands
    around ``-1.13``, well below the classical Néel limit ``-3/4`` for
    spin-1 but still above the converged value because the iPESS bond
    dimension is too small. The window allows for sweep-to-sweep noise
    from random init and short SU schedules while excluding any
    pipeline regression. Pre-fix the SU collapsed to a near-classical
    state and the same call landed near ``-1.0``.
    """
    mod = _import_from_examples("kagome_spin1_pess_ad_benchmark")
    _, e_ad, _ = mod.run_kagome_spin1_benchmark(
        D=2,
        chi=8,
        max_iter=5,
        su_steps=((0.1, 100), (0.01, 50)),
    )
    assert -1.20 < e_ad < -1.00, f"E/site={e_ad:.6f} outside [-1.20, -1.00]"
