# iPEPS excitation spectrum and spectral function

**Status:** design, awaiting review. **Branch:** `docs/ipeps-spectrum-spec`.
**Reference:** Ponsioen, Hasik, Corboz, PRB 108, 195111 (arXiv:2306.13327).

## Goal

Add a tenax workflow that computes the iPEPS excitation spectrum and the
dynamical structure factor S(k,w), and validate it against the literature.
Success means:

- it passes the D=2 window oracle and the sum rule,
- it gives sensible results at D=3 and D=4 for the Heisenberg model,
- it reproduces Fig. 1 of the reference (J2=0.3, D=4, S((pi,0),w)).

## Problem

`compute_excitations` truncates the momentum sum to nearest neighbours in a
fixed environment (#1052) and has no spectral weights. Its lowest energy on
Hasik's J2=0 states is unphysical: -0.10 at D=2 and -0.27 at D=3 at (pi,0),
-0.10 and -0.46 at (pi/2,pi/2).

## Measured basis for this design

All numbers come from a throwaway probe (`sma_probe/so2_*` in worktree
`../tenax-sma`, at 679405ec) on Hasik's `j1j2_ipeps_states` (CC BY 4.0), in the
sublattice-rotated frame, on a 2x2 cell. X = (pi,0), M = (pi,pi) in that frame.

| | D=2 (chi=10, 16) | D=3 (chi=18) | Reference |
|---|---|---|---|
| Lowest w at X | 2.293 | 2.268 (see below) | series ~2.18, QMC ~2.16 |
| Lowest w at M (finite-D gap) | 0.649 | 0.308 (see below) | 0 as D goes to infinity |
| s(M) for S^y | 0.8086 | 1.592 | |
| S^y weight in the X magnon | 93% | 87% | |
| Sum rule at X, S^y | 0.203989 = 0.203989 | 0.196702 = 0.196702 | |

- D=2 is the same at chi=10 and 16 to all printed digits.
- D=3 values are fixed chi=18 eigenvectors evaluated at larger chi, not
  re-solved eigenvalues: X gives 2.2641, 2.2665, 2.2676, 2.2679 at
  chi=18, 27, 36, 54; M gives 0.2688, 0.3056, 0.3083 at chi=18, 36, 54.
- At D=3, chi=18 is too small. It gives negative modes (-4.42, -0.30 at X), and
  their Rayleigh quotients are positive at chi=27 and about +6 at chi=54.
- A full re-solve at larger chi has not been run.

Probe checks: N in random directions matches the window oracle to about 1e-5.
H in random pairs matches FD-FD to about 3e-5 or better at D=2.

Cost of one H column on 12 CPU cores: about 30 s at D=3 chi=18, about 135 s at
D=4 chi=16 (2 columns measured). H alone at D=4 chi=16 is about 19 h per k for
512 columns. Scaling by chi^3 (extrapolated) gives about 9 days per k at chi=36.
Log-norm N needs its own backward passes: separate N builds took about 47 s per
column at D=3 chi=18 on a busy host. Sharing the forward pass reduces the total,
by an amount not measured. Until then, assume up to twice the H figures.

## Scope

In: dense real `A`; momenta (pi,0), (0,pi), (pi,pi) in the frame of the inputs;
nearest-neighbour gate and an optional next-nearest gate; spectral weights.

Out: other momenta and larger cells (#1052 stays open); `SymmetricTensor`;
complex `A`; forward-mode AD.

Gamma is refused because the derivation below uses k != 0. Physical Gamma is
reached in the rotated frame through (pi,pi).

## Public API

In `src/tenax/algorithms/ipeps_excitations.py`. A new function, because the old
`env` and `E_gs` arguments have no meaning here and the old function accepts
momenta that this one refuses.

```python
@dataclass
class SpectrumConfig:
    chi: int
    ctm_sweeps: int = 30
    fd_step: float = 1e-4
    null_tol: float = 1e-4
    checkpoint_dir: Path | None = None

def compute_spectrum(
    A, gate, momentum, operators, config,
    *, gate_nnn=None, basis=None,
) -> SpectrumResult: ...

def spectral_function(result, name, omega, eta=0.1) -> np.ndarray: ...
```

- `operators` maps a name to a `d x d` matrix.
- `basis` is an optional matrix of orthonormal columns that restricts B. It is
  how a U(1) charge-sector saving is applied without `SymmetricTensor`.
- `A`, the gates and `operators` must share a frame, and `momentum` is in that
  frame.
- `SpectrumResult` holds `omega`, `weights` and `structure_factor` per operator,
  `rank`, and the diagnostics below.
- `spectral_function` broadens with a Lorentzian of half-width `eta`.

`compute_excitations` gets a `DeprecationWarning` naming `compute_spectrum`. It
does not delegate, because v1 cannot reach every momentum it accepts. Its
truncated-sum helpers are removed with it in a later release.

## Method

Take phases `phi_c = +-1` on the 2x2 cell, with independent ket and bra:

```
ket_c = A + eps phi_c X      bra_c = A + phi_c Y
```

Define two scalars, each evaluated after `ctm_sweeps` sweeps from the converged
uniform environment, with `recipe="2x2"` and `projector_backward="flow"`:

- `e(eps, Y)`: energy density from mixed RDMs, divided by their own traces.
- `l(eps, Y)`: log-norm density, the cell mean of
  `log|Z11 Z00 / (Z10 Z01)|`.

Then, for k != 0:

```
N_k(Y, X)          = d/deps d/dY l
(H_k - E0 N_k)(Y, X) = d/deps d/dY e
```

The disconnected terms vanish because the phases sum to zero, so no explicit E0
subtraction is needed.

### Bond placement for Z terms on the 2x2 cell

With `cx`, `cy`, `cxy` the right, bottom and diagonal neighbours of `c`:

```
Z11(c) = closed window of env[c], ket_c, bra_c
Z00(c) = C1[cxy] C2[cy] C3[c] C4[cx]
Z10(c) = C1[cy] T1[cy] C2[cy] with C3[c] T3[c] C4[c]
Z01(c) = C1[cx] T4[cx] C4[cx] with C2[c] T2[c] C3[c]
```

### Algorithm for one momentum

1. Converge the uniform environment with `_ctm_tensor_multisite` and stop its
   gradient.
2. For each basis vector X_i and each `eps = +h, -h`: one forward pass, then two
   backward passes on the bra, one for `e` and one for `l`.
3. A central difference in `eps` gives one column of each matrix.
4. Symmetrize. Keep eigenvectors of N above `null_tol` times the largest. Solve
   for H in that subspace.
5. For each operator, `B_O = O A`, `w_lambda = |v_lambda^T N B_O|^2` and
   `s(k) = B_O^T N B_O`. A complex `O` splits into real and imaginary parts,
   which works because N is real at these momenta.

### Decisions

- **Finite difference on the ket, reverse on the bra.** A spike with a
  Lorentzian `custom_jvp` SVD showed: first-order forward is correct after a safe
  sqrt of the zeroed Fishman values; reverse over forward is exact for S A but
  gives about 1e3 in random directions (FD-FD about 0.1); forward under jit
  diverges. Causes not found. The ket derivative stays behind one internal
  function so forward mode can replace it later.
- **Reverse-over-reverse is not used.** It gives NaN.
- **N from the log-norm, not local insertion.** Local insertion assumes an exact
  environment. Its asymmetry is finite-chi (8.6e-5, 6.4e-6, 2.8e-6 at chi=18, 27,
  36, independent of sweeps and h). The log-norm is symmetric to 2e-8 at D=3.
  Reverse mode on the local-insertion form is off by about 0.007 at D=2.
- **Jit one sweep, not the whole gradient.** A jit of the 20-sweep gradient did
  not finish compiling in 3400 s.
- **Full matrices.** `basis` is the first way to reduce cost.
- **Raw-array contractions.** Dense scope. Each one must pass the gauge test.

### New code

Mixed bond RDMs and Z terms exist in the probe. The mixed diagonal RDM does not:
`_rdm_diagonal_tensor` assumes bra = ket. It is a larger contraction in every H
backward pass, so J2 != 0 costs more than the figures above, by an amount not yet
measured. If the contractions grow large, they go in a private
`_ipeps_excitation_ctm.py`.

## Errors, raised before expensive work

- `SymmetricTensor` or complex `A`.
- A momentum outside the allowed three, with the message naming them.
- `basis` with the wrong shape or non-orthonormal columns.
- An operator or gate whose shape does not match `d`.
- A checkpoint that does not match the run fingerprint.

## Diagnostics

Stored in `SpectrumResult`. Failed checks warn, they do not raise. Thresholds are
starting values from the probe, not calibrated limits.

| Field | Meaning | Warns if |
|---|---|---|
| `energy_density` | e at eps=0 | CTM did not converge |
| `n_asymmetry`, `h_asymmetry` | max asymmetry | above 1e-4 |
| `n_min_eig` | lowest N eigenvalue / largest | below -1e-4 |
| `n_negative_modes` | modes with w < 0 | any |
| `sum_rule_deficit` | per operator, 1 - sum(w)/s(k) | above 1e-3 |
| `gauge_residual` | largest singular value of N on span(A, gauge vectors), `basis=None` only | above 1e-3 |
| `null_dim` | eigenvalues of N below `null_tol` | never |

- Negative modes are never dropped. The warning says to raise chi.
- The gauge span has rank `2 D^2 - 1` at these momenta (measured at D=2 and D=3).
  `gauge_residual` tests the property directly. A count of small eigenvalues does
  not: at D=3, chi=18 five of the 17 directions are lifted to about 1e-5. At D=2
  there are also 2 extra null directions, not identified, that H annihilates to
  about 1e-4.
- One run cannot judge chi convergence, so the guide says to compare at least two
  chi values.

## Checkpoints

- With `checkpoint_dir`, each column is written atomically when done, and a rerun
  resumes.
- Each checkpoint holds a fingerprint: a hash of A, gates, k, chi, sweeps, h and
  basis. A mismatch raises.
- Without `checkpoint_dir`, more than 100 columns warns. The probe lost a 4 h
  D=3 build to a timeout at about column 150 of 162.

## Testing

No test may rely only on a symmetric direction. A reversed T4/C4 ring in the
probe matched the energy to 10 digits and S^y A to 7 digits, and failed only on
random vectors.

### `core` tests, using `basis` to keep to a few columns

Fixture: Hasik's D=2 state at chi=10, inline with attribution.

- Random-gauge invariance of each contraction, at 2x2-cell level. The gauge is
  one matrix per row and column direction, so shared edge bonds stay closed.
- N against the window oracle in random directions. The oracle moves to a test
  helper with its Onsager certification.
- H against FD-FD on random pairs, to about 1e-4.
- A and the analytic gauge vectors are null in N.
- Sum rule in the subspace.
- Each error above, each warning on a deliberately bad case, and checkpoint
  resume equal to an uninterrupted run.

The PR reports a mutation count, with the ring reversal as a named mutant.

### Slow tests

Full D=2 spectrum at X and M (2.293 and 0.649), and chi=10 vs 16.

### Benchmarks, run by hand, results in the guide

| Step | Target | Pass criterion |
|---|---|---|
| 1. D=3 full re-solve, chi=36 | X, M | no negative modes; w(X) near 2.268 |
| 2. U(1) sector saving | D=4 column count | sets the Fig. 1 budget |
| 3. J2 energy | Hasik J2=0.3 state | matches peps-torch's listed energy |
| 4. D=4, J2=0 | X, M | w(X) below D=3 and toward 2.18; smaller gap at M |
| 5. Fig. 1 | J2=0.3, D=4, S((pi,0),w), eta=0.1 | s within 1% of 0.182; peak within 0.05 of 1.05 |

The reference does not give chi for Fig. 1. A miss gets investigated, not retuned.

## Delivery

One concern per branch:

1. `compute_spectrum`, `spectral_function`, tests and docs, for J2=0.
2. `gate_nnn` with the mixed diagonal RDM.
3. Deprecate `compute_excitations`.
4. Benchmark scripts and results.

Docs: `__all__`, the module docstring, `docs/guide/algorithms/ad_excitations.md`
and the `tenax-ipeps-workflow` skill. The guide states the chi-convergence
requirement and the v1 limits.

Issues: a new one for S(k,w), and a follow-up with the forward-mode findings.

## Risks

- **Fig. 1 may be unaffordable.** About 9 days per k for H at chi=36 on 12 CPU
  cores is an extrapolation, and N and J2 add to it. Benchmark 2 comes first. If
  the sector saving and the A100 are not enough, Fig. 1 waits, and the J2=0
  workflow still stands.
- **chi needed is unknown.** D=3 needs more than 18. The chi for D=4 is not
  measured.
- **Reverse-mode noise in H.** Agreement with FD-FD is 3e-5 or better at D=2. It
  could matter for small features at large D. Not measured at D>2.
- **Private API.** This builds on `_ctm_tensor_sweep_multisite`, whose signature
  can change.
