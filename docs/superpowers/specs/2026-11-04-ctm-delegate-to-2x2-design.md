# Delegate the public `ctm()` to the 2x2 recipe

**Status:** design, revised 2026-11-09 after review. **Branch:** `refactor/ctm-delegate-to-2x2`.

## Problem

The public `tenax.ctm()` runs a raw-array, 1-site corner-pair sweep with no
fixed point. It returns wrong environments and does not warn. The guide
(`docs/guide/algorithms/ipeps.md:154-160`) tells users to call it directly.

`ctm_tensor(recipe="2x2")` is the supported forward and gets the right answer
on the same inputs.

### Evidence

Same state, same `chi`, 800 sweep budget for `ctm()`.

Hasik's D=2 J2=0 state (`j1j2_ipeps_states`, CC BY 4.0), against the energies
peps-torch ships with that state. peps-torch gives -0.66023109 at every chi
from 16 to 192.

| chi | `ctm()` converged | `ctm()` E/site | `ctm_tensor` 2x2 converged | `ctm_tensor` 2x2 E/site |
|---|---|---|---|---|
| 16 | no (diff 2.0e-4) | -0.66098987 | yes, 11 sweeps | -0.6602310935 |
| 32 | no (diff 7.0e-5) | -0.65999932 | yes, 12 sweeps | -0.6602310935 |

`ctm()` at chi=16 lands below the converged energy of that state.

Ising PEPS at beta=0.2, where the norm is the classical 2D Ising partition
function, so C(1,0) has Onsager's exact value:

| chi | `ctm()` converged | `ctm()` C(1,0) error | 2x2 converged | 2x2 C(1,0) error |
|---|---|---|---|---|
| 8 | yes | below 1e-10 | yes | 2.5e-16 |
| 12 | no | 7.6e-4 | yes | 1.9e-16 |
| 16 | no | 4.3e-4 | yes | 2.2e-16 |
| 24 | no (diff 3.1e-7) | below 1e-7 | yes | 2.5e-16 |
| 32 | no | 4.3e-4 | yes | 2.2e-16 |

Increasing `chi` does not improve `ctm()` monotonically, so a chi scan does
not expose the failure either.

## Decision

`ctm()` keeps its signature, delegates to `ctm_tensor(recipe="2x2")`, and emits
a `DeprecationWarning`. Every existing caller gets the right answer at once;
removal follows in a later release. Tenax is pre-1.0 and legacy paths are a
fallback, not a constraint, but a warning period costs little here.

Rejected: removing `ctm()` now (breaks downstream code with no notice) and
warning only (callers keep receiving wrong energies until they migrate).

## Behaviour of `ctm(A, config, initial_env=None, *, return_meta=False)`

`ctm()` emits the warnings, checks `initial_env`, and calls a private helper,
`_ctm_via_2x2(A, config, *, return_meta=False)` in `ipeps_ctm_convergence.py`.
The helper does the rest of this table: it checks and wraps `A`, maps the
config, runs `ctm_tensor(recipe="2x2")` and converts the result. `fermionic_ctm`
and the fixture-only tests call the helper directly, so neither the library nor
those tests trigger the `DeprecationWarning`.

| Input | Behaviour |
|---|---|
| Any call | `DeprecationWarning` naming `ctm_tensor` as the replacement (in `ctm()` only) |
| `A` raw array with 5 legs `(u, d, l, r, s)` | Wrapped with `_wrap_as_dense_tensor` (`_ipeps_optimize_shared.py`) |
| `A` raw array with another number of legs | `ValueError` that names the 5-leg layout (see Input shape, below) |
| `A` `DenseTensor` | Passed through |
| `A` `SymmetricTensor` | `TypeError` naming `ctm_tensor`. Today's `ctm()` also raises `TypeError` here (measured), and accepting it would force `todense()` to return a raw `CTMEnvironment` |
| `config.chi`, `max_iter`, `conv_tol`, `renormalize` | Passed to `ctm_tensor` |
| `config.min_iter` | Passed to the new `ctm_tensor(min_iter=)` keyword (below) |
| `config.projector_method != "svd"` | `UserWarning` (in `ctm()`) that it is ignored: the 2x2 recipe always uses Fishman SVD. The helper always passes `projector_method="svd"`, because on the 2x2 recipe `"qr"` still starts the QR warm-up loop of `ctm_tensor` |
| `config.qr_warmup_steps` | Ignored: with `"svd"` passed, no warm-up runs |
| `initial_env` not `None` | `ValueError` with a migration message |
| `return_meta=True` | `(CTMEnvironment, CTMConvergenceInfo)`, as today |
| Return value | Raw-array `CTMEnvironment`, as today |

The conversion from `CTMTensorEnv` to `CTMEnvironment` is `_as_dense_env`. It
moves from `ipeps_excitations.py` to a new module, `_ctm_env_dense.py`, together
with `_as_dense_array`, which it calls. Behaviour and messages do not change.
These files import them from the new module:

- `ipeps_excitations.py`
- `ipeps_ctm_convergence.py` (the helper)
- `tests/test_ipeps_excitations.py:471,482`
- `tests/test_ipeps_spectrum_contractions.py:49`

The imports in `ipeps_ctm_convergence.py` stay local to the function body, to
avoid a cycle through `ipeps_rdm`.

### Input shape

Today's `ctm()` also runs a 3-leg array. Its docstring says that a ready
double-layer tensor works, but a 4-leg array is read as a site tensor whose last
leg is physical. `_wrap_as_dense_tensor` reads `shape[4]` without a check. So the
helper accepts only the 5-leg layout, raises `ValueError` for any other number of
legs, and the docstring is corrected. The plan checks that no caller passes
another shape.

### Why `initial_env` raises

No production code passes it. Its only users are the warm-start tests at
`tests/test_ipeps.py:180,198` and `tests/test_ipeps_core.py:191,209`. Supporting
it means a warm-start feature in `ctm_tensor`, which is out of scope. Raising is
better than silently cold-starting, because a caller expecting a warm start
would not see the difference.

### `ctm_tensor(min_iter=1)`

`ctm_tensor` currently early-exits on `conv_tol` alone. It gains a keyword-only
`min_iter`, default 1, which leaves every existing call unchanged. It uses the
same rule as the other entry points: gate `converged` on
`n_iter >= min(min_iter, max_iter)`, so a small `max_iter` stays convergeable
(#976). The `CTMConvergenceInfo` docstring, which says `ctm_tensor` takes no
`min_iter`, is updated.

`CTMConfig.min_iter` defaults to 10, so `ctm()` keeps requiring at least 10
sweeps.

## Other changes

- `fermionic_ctm` (`fermionic_ipeps.py:655`), non-Tensor branch: call
  `_ctm_via_2x2`. It keeps returning the `CTMEnvironment` that its docstring
  promises, keeps `CTMConfig.min_iter` (10), and does not trigger the warning. A
  direct `ctm_tensor` call would return a `CTMTensorEnv` and use `min_iter=1`.
  Nothing calls `fermionic_ctm` and it is not exported, so its new test is its
  only test.
- `examples/kagome_xxz_pess.py:606`, `examples/kagome_xxz_spin1_pess.py:344`,
  and `docs/guide/algorithms/ipeps.md:154-160`: move to `ctm_tensor`.
- Delete code that only the legacy 1-site sweep reaches:
  - `ipeps_ctm_convergence._ctm_sweep`
  - `ipeps_ctm_moves._ctm_{left,right,top,bottom}_move`
  - `ipeps_ctm_moves._ctm_move_qr`, with the `projector_method` argument of
    `_ctm_move`
  - the matching re-exports in `ipeps_ctm.py`

  `_ctm_move` and `_ctm_move_eigh` stay, because the 2-site moves call them.
  `_initialize_ctm_env` and `_build_double_layer` stay, because `ctm_2site` and
  tests use them. `_ctm_python_loop._ctm_sweep` is an unrelated function with the
  same name and stays.

## Tests

### New (`core`, exact oracles only)

- Ising PEPS at beta=0.2, chi in {8, 12, 16, 32}: `ctm()` gives
  `converged=True` and C(1,0) within 1e-12 of Onsager. The 2x2 error measured
  1.9e-16 to 2.5e-16 (Evidence). The old code fails this at chi=12, 16 and 32.
- Ising PEPS at beta=0.3: not measured yet. It is nearer beta_c (about 0.4407),
  so chi=8 may miss 1e-12. The plan measures the 2x2 error at each chi first.
  The test keeps the chi values whose error is at least ten times below 1e-12,
  or it drops beta=0.3.
- The `DeprecationWarning` fires; `projector_method="qr"` warns; `initial_env`
  raises; a raw array with 4 legs raises `ValueError`; `return_meta` has the
  right shape.
- `ctm_tensor(min_iter=)`: `n_iter >= min(min_iter, max_iter)` when converged,
  and the default leaves existing behaviour unchanged.
- `fermionic_ctm` on a raw array returns a `CTMEnvironment` and emits no
  `DeprecationWarning`.

Report a mutation count for the new tests in the PR.

### Existing

Each changes for one of the reasons below, and the PR lists every changed test
with its reason. A changed numeric assertion gets examined, not re-frozen.

- **Fixture only** (`ctm()` builds an environment for something else):
  `test_ipeps.py`, `test_ipeps_core.py`, `test_ipeps_excitations.py`,
  `test_ipeps_excitations_gauge.py`, `test_ipeps_rdm_gauge.py`,
  `test_normalise_rdm_zero_grad.py`, `test_rdm_gate_transpose_966.py`. They
  need a raw `CTMEnvironment`, so they move to `_ctm_via_2x2`: `ctm_tensor` plus
  the conversion, with the arguments of `ctm()` and no warning.
- **Pinned to QR or warm-up on `ctm()`**: `test_ipeps_core.py:368,382`, and
  `test_ctm_counts_qr_warmup_in_n_iter` and
  `test_ctm_warmup_counts_toward_min_iter` in
  `test_ctm_min_iter_contract_925_976.py`. They are removed: `ctm_2site` and
  `ctm_split` have no QR and no warm-up, so they cannot be retargeted. The new
  `projector_method="qr"` test replaces them. The parametrized cases of that
  file (`ctm`, `ctm_2site`, `ctm_split`) stay. On `ctm()` they then test
  `ctm_tensor(min_iter=)`.
- **Warm start**: the `initial_env` tests above. Replace with the
  `ValueError` test.
- **Contract tests that still apply**: `test_ctm_complex_site_tensors.py`,
  `test_ctm_convergence_meta.py`, `test_ctm_criterion_rank_blind_898.py`. Keep
  them against the delegating `ctm()`. If one fails, that is a finding about
  `ctm_tensor`, and it gets investigated before anything is changed.

## Out of scope

- `ctm_2site` and `ctm_split`. `ipeps()` imports `ctm_2site` directly. Their
  accuracy has not been measured, so nothing here claims they are broken.
- `ctm_tensor(recipe="1x1")`: 17 source files and 27 test files, already
  deprecated by #911. Separate follow-up.
- The raw `CTMEnvironment` type and `ipeps_rdm`. They are a data layout, not a
  defect.
- `compute_excitations` and its nearest-neighbour truncation (#1052).
- `ctm_tensor(recipe="2x2", projector_method="qr")`, which still runs the QR
  warm-up loop although the recipe ignores `projector_method`. `ctm()` avoids it
  by passing `"svd"`. A change to `ctm_tensor` is a separate decision.

## Risks

- **`ctm()` stops working under `jax.jit`.** Measured: today's `ctm()` runs under
  `jit`, while `ctm_tensor` raises `ConcretizationTypeError` under `jit`. Neither
  supports `jax.grad` (today's `ctm()` raises `ValueError` from `lax.while_loop`),
  so no gradient user is affected. A grep found no `jit` wrapper around `ctm()` in
  `src`, `tests`, `examples` or `benchmarks`, but it cannot see indirect use or
  downstream code. The `DeprecationWarning` text and the changelog state this.
- **Environments change for every `ctm()` caller**, and for the raw branch of
  `fermionic_ctm`, including results users have already recorded. That is the purpose of the change. The changelog says so
  plainly and cites the table above.
- **Complex site tensors** must keep working (#842). The kept contract test
  covers this.
