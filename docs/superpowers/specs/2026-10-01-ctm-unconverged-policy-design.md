# CTM unconverged-forward policy (#1059 hotspot 8, #1060)

Status: design approved in conversation 2026-10-01. This document is the written spec for review.

## Problem

Every CTM forward in the iPEPS-AD optimizers returns a `CTMConvergeInfo` with a `converged` flag. On the production paths that flag is discarded:

| Call site | What happens today |
|---|---|
| `_run_forward` in `_ctm_energy_ad.py` | Only `_check_forward_stationarity` reads the flag; it warns once per cache build and nothing acts on it. |
| `_update_env_cache_2s` and 1-site `_update_env_cache` | The flag is ignored and the env is cached as the next warm start. |
| `loss_fn_fwd` (line-search φ) | Assigns `envs, _ =`, so the flag is dropped. |
| `_eval_fresh_2site` and its 1-site twin | Assigns `envs, _ =`, so the flag is dropped. |
| `ctm_tensor_2site` (public) | Returns envs only and warns internally. |

Production evidence (spinless t-V, D=3, 2-site checkerboard; see #1060 and its correction comment):

- **V=0.** The final reported energy came from a cold re-evaluation that hit the 500-sweep cap (`sv_diff` 3.9e-7 vs `conv_tol` 1e-10). It differs from the converged warm env by 3.3e-4.
- **V=2.** The seed's unconverged `bond_phase` forward reported E+V = −0.3024. That is below exact diagonalisation (−0.2918) and cylinder DMRG (−0.2913), so an unconverged env can report a sub-variational energy. The optimizer then took gradients at that non-fixed point.
- **V=1, chained line-search trials.** Each trial warm-started from the previous trial's *unconverged* env, and the chain landed on a different branch (E(α→0) ≠ E(0)).

The implicit-AD gradient is only valid at an element-wise fixed point (#841, #1057). A non-converged forward silently invalidates both the gradient and any energy reported from it.

### No single forward fix covers every failure

Two distinct non-convergence modes have been measured with `bond_phase` and element-wise convergence. Both come from a V=1 D=3 checkerboard state starting from a converged env (#1060 comment 5926626429, plus an independent check on the production r5 step-3 point):

| Mode | Signature | Where seen | Damping (#1061 `ctm_mixing`) |
|---|---|---|---|
| **Period-2 cycle** | Sweeps two apart agree to 5e-13 while consecutive sweeps differ by 3.9e-2. The spectra differ, so it is not a gauge rotation. Slow-mode multiplier λ ≈ −1 | χ=20 on the V=1 state; V=1 r5 step 3 at α=1e-6, χ=21 | β=0.3 converges (572 iterations) |
| **Slow wander** | Gauge-invariant spectra drift about 1.6e-3 per sweep. No 2-cycle | χ=14 on the V=1 state; V=2 seed at χ=12, where E+V wanders between −0.24 and −0.31 (below ED) | Does not cure it (β=0.3, 0.5 leave residuals of 8e-3 and 0.19) |

The within-sector singular-value gap at the χ cut does **not** predict failure: χ=16 converges with 0.39%, χ=14 fails with 0.67%, and the r5 step-3 failure has a clean 10.6% gap.

So there is no cheap pre-check that tells a caller whether a forward can be trusted. Damping helps one mode, and nothing yet fixes the other. That is why every consumer of a forward must check `converged`, independently of any improvement to the forward itself.

## Goal

Make every CTM forward that feeds a gradient or a reported energy act on `converged=False`. What it does depends on the site (below). The default is to fail loudly. An explicit opt-out keeps today's control flow but still warns.

Non-goals:

- Making the forward converge more often. That is the near-degenerate-cut masking (#1060) and the second failure mode under investigation, both separate work.
- Split-CTM paths.
- Multisite, PESS and root-implicit optimizers. Those come in a follow-up PR that uses the same helper.

## Design

### 1. Helper and config

New module `src/tenax/algorithms/_ctm_convergence_policy.py`:

```python
class CTMNotConvergedError(RuntimeError):
    """A CTM forward feeding a gradient or a reported energy did not reach its fixed point."""
    def __init__(self, info: CTMConvergeInfo, site: str, step: int | None = None): ...
    # attributes: info, site, step; message: site, step, iterations, sv_diff, conv_tol, chi,
    # and the signed step multiplier when available (see below)

class CTMNotConvergedWarning(UserWarning): ...

def check_ctm_converged(info, *, site: str, policy: str, step: int | None = None,
                        conv_tol: float | None = None) -> bool:
    """Return info.converged. If False:
    policy "raise" -> raise CTMNotConvergedError(info, site, step)
    policy "warn"  -> warnings.warn(..., CTMNotConvergedWarning), once per (site, step)."""
```

- **Mode in the message.** The message includes `step_multiplier = getattr(info, "step_multiplier", None)`. That field is added by #1061, so this PR does not depend on #1061 landing first. A `None` or NaN value (fewer than two comparable steps, e.g. right after a χ bump, or #1061 absent) is printed as `n/a`. Read it as: ≈ −1 is a flip cycle (damping helps), ≈ +1 is slow monotone convergence (damping hurts; raise `max_iter`), anything else is a wander. The same value is recorded in history as `ctm_step_multiplier`.
- `CTMConfig` gains `on_unconverged: Literal["raise", "warn"] = "raise"`, validated in `__post_init__`. It does not change `conv_method`, `conv_tol` or `max_iter`.
- `CTMNotConvergedError` and `CTMNotConvergedWarning` are exported in `src/tenax/__init__.py` `__all__` and noted in `README.md`, per CLAUDE.md.

### 2. Call-site rules (2-site and 1-site optimizers, public API)

| # | Site | Under `"raise"` (default) | Under `"warn"` |
|---|---|---|---|
| 1 | `_run_forward` (gradient forward) | `check_ctm_converged(site="gradient")` raises; handled by the optimizer (section 3) | warning, plus `ctm_converged=False` in history |
| 2 | `_update_env_cache_2s`, 1-site `_update_env_cache` | Never caches an unconverged env: keeps the previous one and logs. No raise; site 1 is authoritative | Same refusal to cache, plus a warning |
| 3 | `loss_fn_fwd` 2-site and 1-site (line-search φ) | Returns `+inf` and does not write the cache | Legacy behaviour (returns the energy), plus a warning |
| 4 | `_eval_fresh_2site` and 1-site twin (final / reported energy) | If unconverged, fall back to the optimizer's warm env, but only if it belongs to exactly these params and had `converged=True`. History records which env produced the energy. With no such env, raise | Legacy result, plus a warning |
| 5 | `ctm_tensor_2site` (public) | New kwarg `strict: bool = False`; `strict=True` raises | n/a |

Rationale:

- Site 2 refuses to cache in both modes, because poisoned warm starts caused the V=1 wrong-branch failure.
- Site 4 keeps the cold re-evaluation as the first choice. At χ=12, warm and cold can sit on different truncated fixed points about 3e-4 apart, so the history must say which env was used rather than hide the difference.

### 3. Optimizer handling of `CTMNotConvergedError`

Add `CTMNotConvergedError` to the existing `except CTMRGGradientError` clauses around `value_and_grad` in the 2-site and 1-site loops (precedent: #454 stall recovery). Two differences from that path:

1. **Restore the best env, don't clear it.** On reset, restore `best_env_cache` when it exists and matches the current χ (the #518 condition). Otherwise clear it, as today. Clearing forces the cold start that produced the V=0 and V=2 failures.
2. **Checkpoint and raise when recovery is impossible.** That is: the error occurs at `best_params` itself, the stall budget (`gs_stall_recovery_retries`) is spent, or `gs_stall_recovery != "reset"`. In each case write `ckpt.last.pkl` (if `gs_checkpoint_path` is set) and re-raise `CTMNotConvergedError` with step and diagnostics. Do not `break` and return the best energy silently.

History entries gain `ctm_converged`, `ctm_sv_diff` and `ctm_step_multiplier` (`None` when unavailable). Each reset logs `[iPEPS-AD] CTM forward not converged at step k (sweeps n, sv_diff x vs tol y, step multiplier m|n/a) — reset to best (#n/N)`.

### 4. Tests

New file `tests/test_ctm_unconverged_policy.py`. Add it to the explicit filename→marker map in `tests/conftest.py` as `"core"` so CI's required `-m core` runs it.

Non-convergence is forced for real with a small D=2 state and `CTMConfig(max_iter=2, conv_tol=1e-14)`. A one-call monkeypatch of `python_loop_ctm_converge` returning `converged=False` is used only where a specific step must fail (marked ★).

1. `check_ctm_converged`: passes when converged; raises with site/step/iterations/sv_diff; warns under `"warn"`. The message shows a numeric step multiplier when the info carries one, and `n/a` when it is missing or NaN.
2. `CTMConfig(on_unconverged="bogus")` raises `ValueError`.
3. Site 1: 2-site and 1-site `optimize_gs_ad` with `max_iter=2` raise `CTMNotConvergedError` after writing `ckpt.last.pkl` to a tmp path. Under `"warn"` the run completes with `ctm_converged=False` in history.
4. ★ Reset restores `best_env_cache` (the next forward's `env_init` is the best env, not `None`). A χ mismatch clears the cache (the #518 path).
5. ★ Site 2: an unconverged refresh leaves the cached env unchanged, in both modes.
6. ★ Site 3: φ returns `inf` under `"raise"`, and the energy plus a warning under `"warn"`.
7. ★ Site 4: falls back to the converged warm env and records it; raises with no converged env; under `"warn"` gives the legacy result plus a warning.
8. Site 5: `ctm_tensor_2site(strict=True)` raises; the default does not.

Regression inventory: run the full suite with the default `"raise"`. Each newly failing test is either pinned to `on_unconverged="warn"` with a comment explaining why it needs an unconverged forward, or, if it encoded the bug, fixed. Every pinned test is listed in the PR description.

## Compatibility

- **Behaviour change.** By default, runs that previously continued on an unconverged environment now raise, or reject the trial at the line search. Users who need the old flow set `CTMConfig(on_unconverged="warn")`.
- **Public API.** New exceptions and warnings; a new `strict` kwarg on `ctm_tensor_2site` with default `False`. No signature removals.
- **Unchanged.** Convergence criteria, tolerances and the forward itself.

## Follow-ups (not in this PR)

- The same helper in multisite, PESS and root-implicit optimizers.
- Split-CTM forwards surfacing a convergence flag.
- **Period-2 mode:** #1061 `ctm_mixing` (default off) is the fix. It should not become the default, because damping hurts when the multiplier is ≈ +1.
- **Wander mode:** open, with no known fix; damping does not cure it.
- **#1060 near-degenerate masking:** deprioritised. The within-sector gap at the cut does not predict failure.
- **Adjoint cost (not convergence detection):** near ρ ≈ +1 the adjoint `(I − Jᵀ)` is ill-conditioned, so Neumann and GMRES slow down under any gauge. Separately, `bond_phase` appears to cost the adjoint its warm start (about 2.3× iterations on a well-converged Heisenberg D=2 forward). Both are performance issues. The adjoint already has its own residual check and the Arnoldi pre-check (`CTMRGGradientError`).
