# Flow-gradient slope check (#1111)

Date: 2026-10-17. Status: design approved in conversation; spec under review.
Branch: `fix/1111-flow-slope-check`, built on `feat/gmres-single-matvec` (#1112).

## Problem

`projector_backward="flow"` is the default under `forward_gauge="bond_phase"`.
`f_bwd` (`src/tenax/algorithms/_ctm_energy_ad.py`) keeps a flow gradient
unless one of three things happens:

- the flow adjoint fails to solve;
- its spectral precheck fails;
- the gradient is non-finite.

On a non-stationary forward, a *solved* flow adjoint can still give a gradient
that is orders of magnitude off. Main already returns one at D=3 chi=12. With
the single-matvec GMRES (#1112), more flow adjoints solve, so two D=2 fixtures
that used to fall back to frozen now return garbage. Data: #1111 and the
#1112 spec.

Two cheap signals do not separate good flow gradients from garbage ones:

- forward stationarity (garbage at 0.011, good at 0.15);
- adjoint amplification `||λ||/||b||` (garbage 109-120, good 29, mixed 113).

## Measured discriminator: slope ratio

Let `G` be the energy gradient and `u = -G/|G|`. Then

    r = (E(x + h u) - E(x)) / (-h |G|)

compares the energy change along the gradient's own descent direction with
the change the gradient predicts. The setup:

- one-sided, `h = 1e-4`;
- the same energy function as the forward (same `env_init`, `max_iter` and
  settings);
- CPU, branch code;
- finite-difference labels from two random directions.

| case | flow FD error | r (flow) | r (frozen) |
|---|---|---|---|
| D=2 #1028 fixture, max_iter=20 | 1497 / 39 | 0.000 | 1.005 |
| D=2, max_iter=30 | 7128 / 48 | 0.001 | 1.10 |
| D=3 chi=12, max_iter=20 | 86 / 25 | 0.002 | 1.07 |
| D=2 seed 3 (mixed) | 0.49 / 2.3 | 0.41 | 1.07 |
| good flow (limit cycle mi 20/80/300, random 1-site seeds, chi=8 starved, converged D=2 and D=3) | 0 - 0.76 | 0.97 - 1.24 | 1.00 - 1.54 |

- The gap between garbage and good flow gradients is about 500x.
- Central differences and `h = 1e-3` agree with the one-sided `h = 1e-4`
  result.
- The probe is in the session scratchpad: `gmres/slope_probe.py`, with logs
  `slope_*.log`.

## Design

### When the check runs

The check runs inside `f_bwd` only when all three conditions hold:

- `projector_backward == "flow"`;
- the flow backward has passed the existing acceptance tests (adjoint solved,
  precheck passed, gradient finite);
- the forward was **not stationary**:
  `forward_stationarity_residual > forward_stationarity_threshold`. This is
  the #841 threshold, `max(100*conv_tol, 1e-8)`.

Stationary forwards skip the check. Their gradients and their cost are
unchanged.

### Residuals

`f_fwd` appends two entries to its custom_vjp residuals:

- the forward energy `E0`;
- a Python bool `nonstationary`, computed from the stationarity residual and
  threshold of this forward.

The flag travels with the residuals instead of being read from the
module-global `_F3_LAST_DIAGNOSTICS`, because another forward can overwrite
those diagnostics before this backward runs. Every consumer of the residuals
tuple must be updated (`_bwd_core` unpacks `residuals`).

### Measurement

The measurement uses these quantities:

- `g` is the incoming cotangent (a scalar). If `g == 0`, the check is skipped
  and flow is kept, since there is no gradient to verify.
- `G = grads / g` is the energy gradient, a pytree with the leaves of
  `params_data_tuple`. `|G|` is its 2-norm over all leaves.
- `x` is `params_data_tuple` and `||x||` its 2-norm over all leaves.
- `h = 1e-4 * ||x||`, and `x_h = x - (h/|G|) * G`, applied leaf by leaf with
  `jax.tree`.
- `E_h = _compute_energy(site(x_h), _run_forward(site(x_h)).envs)` uses the
  same closure as `f`/`f_fwd`.
- `r = (E_h - E0) / (-h * |G|)`.
- If `|G| == 0` with `g != 0`, `r` is non-finite, so the check is inconclusive (see below).

Acceptance: keep flow if `0.5 <= r <= 2.0`. Define module constants
`_FLOW_SLOPE_RATIO_MIN = 0.5` and `_FLOW_SLOPE_RATIO_MAX = 2.0`, each with a
comment citing the table above. The lower bound catches a gradient that is
too large, as in every garbage case. The upper bound catches the symmetric
failure, a gradient that is too small.

### Rejection

A rejection reuses the existing #1028 fallback path, unchanged:

- discard flow;
- set `run["prev_lam_leaves"] = None`;
- warn once per run unless deferred to the caller's policy;
- redo the step with frozen projectors;
- report `projector_backward_used = "frozen_fallback"`.

`flow_fallback_reason` is one of:

- `"its gradient failed the slope check (measured/predicted dE = {r:.3g}, accepted range [0.5, 2])"`;
- `"the slope check could not evaluate ({detail})"`, for the inconclusive
  cases below.

### Inconclusive cases

The check is inconclusive when any of these holds:

- the extra forward raises any `Exception`;
- `E_h` or `r` is non-finite;
- the extra forward's `chi_post` differs from the residuals' `chi_post`
  (a chi bump changed the function).

An inconclusive check falls back to frozen with the "could not evaluate"
reason. It fails closed, because frozen is biased (#1028) but bounded.

### Isolation of the extra forward

The extra forward must leave no trace:

- **Diagnostics:** snapshot `dict(_F3_LAST_DIAGNOSTICS)` before the extra
  forward and restore it exactly afterwards. Then add `flow_slope_ratio`.
- **Warnings:** run the extra forward inside
  `warnings.catch_warnings(): warnings.simplefilter("ignore")`.
- **Run-state latches:** snapshot the run state's warning latches (the
  `*_warned` keys of `_run_state(...)`) and restore them, so the check cannot
  spend a once-per-run warning.
- **Caches:** `_cached` and `mutables` are not modified. `_run_forward` does
  not write `_cached["env_treedef"]`; only `f_fwd` does.

### Diagnostics

- `flow_slope_ratio` (float) is present when the check ran and produced a
  finite `r`. It is absent otherwise, including on stationary forwards and
  for `projector_backward != "flow"`.
- `flow_fallback_reason` is unchanged in meaning; it gains the two new
  reasons above.

### Tracing

`f_bwd` already pulls values to the host (`jax.device_get` in the
finite-gradient test), so the backward already requires concrete values. The
check adds no new tracing restriction.

### Cost

- Non-stationary flow steps cost one extra forward CTM evaluation, plus a
  frozen backward when flow is rejected.
- Stationary steps and `projector_backward="auto"` cost nothing extra.

## Testing

- **#1111 fixture: flow is rejected.** Replace the strict xfail
  `test_flow_gradient_on_the_1028_fixture_is_wrong_issue_1111`
  (`tests/test_projector_backward_default_1028.py`) with
  `test_slope_check_rejects_the_garbage_flow_gradient_issue_1111`. It uses
  the default gmres settings on the D=2 #1028 fixture and asserts:
  - `projector_backward_used == "frozen_fallback"`;
  - the reason contains `"slope check"`;
  - `flow_slope_ratio < 0.5`;
  - the gradient equals the frozen gradient (`allclose` 1e-10).
- **Good flow is kept.** On the D=2 chi=4 limit cycle at `max_iter=20`
  (random 1-site, `PRNGKey(2026)`, `bond_phase`, flow), assert
  `projector_backward_used == "flow"` and
  `0.5 <= flow_slope_ratio <= 2.0`.
- **Stationary forwards skip the check.** On a converged control, assert
  `flow_slope_ratio` is absent and the gradient is bit-identical to a run
  with the check disabled (monkeypatch the check function to raise if
  called).
- **Diagnostics are restored.** After a check runs, the `forward_*`
  diagnostics equal those recorded right after the forward
  (`forward_stationarity_residual`, `forward_iterations`,
  `forward_converged`).
- **Inconclusive means frozen.** Monkeypatch the extra-forward call to raise.
  Assert `"frozen_fallback"`, the reason `"could not evaluate"`, and no
  `flow_slope_ratio`.
- **Latches stay unspent.** The run state's `*_warned` latch values are the
  same before and after a backward whose check ran (read them through
  `_run_state` for the run token in the residuals).
- **Slow test: the D=3 hole.** D=3 chi=12 `max_iter=20` #1028-family fixture:
  flow is rejected, with `flow_slope_ratio < 0.5`.
- **Existing tests.** `tests/test_projector_backward_default_1028.py`,
  `tests/test_ctm_unconverged_policy.py`, `tests/test_adjoint_convergence_gate.py`
  and the #1070 policy tests must pass. Where a fixture's outcome changes
  because of the check, pin the new outcome with the measured numbers. Do not
  starve the solve to keep an old outcome.

## Docs

- CHANGELOG, "Behavior Changes": the flow slope check, with the table and its
  cost.
- `f_bwd` docstring: add why a slope check separates garbage flow gradients
  where stationarity cannot, and keep the existing measured rationale.
- `docs/guide/algorithms/ipeps_ad_paths.md`, in the #1028/flow section:
  one paragraph on the check and its diagnostics.
- Close #1111 with a summary comment after merge.

## Out of scope

- Changing the frozen path or the `"auto"` default.
- A slope check on frozen gradients.
- Making the thresholds configurable (YAGNI; constants with the evidence).
