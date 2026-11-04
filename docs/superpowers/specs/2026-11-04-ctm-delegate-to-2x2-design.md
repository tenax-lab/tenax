# Delegate the public `ctm()` to the 2x2 recipe

**Status:** design, awaiting review. **Branch:** `refactor/ctm-delegate-to-2x2`.

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

| Input | Behaviour |
|---|---|
| Any call | `DeprecationWarning` naming `ctm_tensor` as the replacement |
| `A` raw array | Wrapped with `_wrap_as_dense_tensor` (`_ipeps_optimize_shared.py`) |
| `A` `DenseTensor` | Passed through |
| `A` `SymmetricTensor` | `TypeError` naming `ctm_tensor`. Today's `ctm()` also raises `TypeError` here (measured), and accepting it would force `todense()` to return a raw `CTMEnvironment` |
| `config.chi`, `max_iter`, `conv_tol`, `renormalize` | Passed to `ctm_tensor` |
| `config.min_iter` | Passed to the new `ctm_tensor(min_iter=)` keyword (below) |
| `config.projector_method != "svd"` | `UserWarning` that it is ignored: the 2x2 recipe always uses Fishman SVD |
| `config.qr_warmup_steps` | Ignored, because no QR sweep runs |
| `initial_env` not `None` | `ValueError` with a migration message |
| `return_meta=True` | `(CTMEnvironment, CTMConvergenceInfo)`, as today |
| Return value | Raw-array `CTMEnvironment`, as today |

The `CTMTensorEnv` to `CTMEnvironment` conversion reuses the field-by-field
logic of `_as_dense_env` in `ipeps_excitations.py`; that helper moves to a
module both can import. Imports stay local to the function body, to avoid a
cycle through `ipeps_rdm`.

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
  `ctm_tensor` directly, so the library never triggers its own warning.
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

- Ising PEPS at beta=0.2 and 0.3, chi in {8, 12, 16, 32}: `ctm()` gives
  `converged=True` and C(1,0) within 1e-12 of Onsager. The old code fails this
  at beta=0.2, chi=12, 16 and 32.
- The `DeprecationWarning` fires; `projector_method="qr"` warns; `initial_env`
  raises; `return_meta` has the right shape.
- `ctm_tensor(min_iter=)`: `n_iter >= min(min_iter, max_iter)` when converged,
  and the default leaves existing behaviour unchanged.
- `fermionic_ctm` on a raw array emits no `DeprecationWarning`.

Report a mutation count for the new tests in the PR.

### Existing

Each changes for one of the reasons below, and the PR lists every changed test
with its reason. A changed numeric assertion gets examined, not re-frozen.

- **Fixture only** (`ctm()` builds an environment for something else):
  `test_ipeps.py`, `test_ipeps_core.py`, `test_ipeps_excitations.py`,
  `test_ipeps_excitations_gauge.py`, `test_ipeps_rdm_gauge.py`,
  `test_normalise_rdm_zero_grad.py`, `test_rdm_gate_transpose_966.py`. Move to
  `ctm_tensor`.
- **Pinned to QR or warm-up on `ctm()`**: `test_ipeps_core.py:368,382`, and the
  `ctm`-specific cases in `test_ctm_min_iter_contract_925_976.py`. Retarget at
  `ctm_2site` or `ctm_split` where they exercise the same contract; otherwise
  remove.
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

## Risks

- **`ctm()` stops working under `jax.jit`.** Measured: today's `ctm()` runs under
  `jit`, while `ctm_tensor` raises `ConcretizationTypeError` under `jit`. Neither
  supports `jax.grad` (today's `ctm()` raises `ValueError` from `lax.while_loop`),
  so no gradient user is affected. A grep found no `jit` wrapper around `ctm()` in
  `src`, `tests`, `examples` or `benchmarks`, but it cannot see indirect use or
  downstream code. The `DeprecationWarning` text and the changelog state this.
- **Environments change for every `ctm()` caller**, including results users have
  already recorded. That is the purpose of the change. The changelog says so
  plainly and cites the table above.
- **Complex site tensors** must keep working (#842). The kept contract test
  covers this.
