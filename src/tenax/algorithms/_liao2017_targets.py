"""Hand-digitized 3-PESS simple-update targets from Liao 2017.

Source: H.J. Liao et al., "Gapless spin-liquid ground state in the S=1/2
kagome antiferromagnet," PRL 118, 137202 (2017), arXiv:1610.04727,
Figure 1(a) (3-PESS simple update curve, blue circles).

Values are read off the figure to ±0.001 precision; treat as a
qualitative target band, not a numerical reference.
"""

from __future__ import annotations

# {D: E/site (3-PESS simple update, S=1/2 kagome AFM Heisenberg, Δ=1)}
LIAO2017_3PESS_SU_FIG1A: dict[int, float] = {
    4: -0.4290,
    6: -0.4340,
    8: -0.4360,
    10: -0.4365,
}

# Asymptotic extrapolation reported in Fig 1(b) inset:
LIAO2017_3PESS_SU_INF: float = -0.43752  # ±0.00006

# Tolerance band for "matches Liao within figure-readout error":
LIAO2017_FIGURE_READOUT_TOL: float = 0.002
