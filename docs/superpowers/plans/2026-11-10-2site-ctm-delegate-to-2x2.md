# 2-site `ipeps()` and `ctm_2site()` on the 2x2 Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the energy that `ipeps()` reports, and the environments that `ctm_2site()` returns, functions of the state: measure the 2-site pair with the 2x2 recipe of `ctm_tensor_2site`, sum all four bonds of the cell, and delete the legacy 2-site sweep.

**Architecture:** Mirror #1183 (`ctm()` → `ctm_tensor(recipe="2x2")`) for the checkerboard. The multisite loop `_ctm_tensor_multisite` gains a keyword-only `min_iter` (default 1) and reports its sweep count on a converged run. A private helper, `_ctm_2site_via_2x2(A, B, config, *, return_meta=False)` in `ipeps_ctm_convergence.py`, checks and wraps two raw 5-leg arrays, maps the `CTMConfig` fields, always passes `projector_method="svd"`, runs the multisite loop with `CHECKERBOARD_NEIGHBORS` and `recipe="2x2"`, and converts each environment with #1183's `_as_dense_env(_in_documented_leg_order(env))`. `compute_energy_ctm_2site` sums the four bonds and halves the sum. `ipeps()` calls the helper; `ctm_2site()` keeps its signature, warns (`DeprecationWarning`, and a `UserWarning` for an ignored `projector_method`) and returns what the helper returns. The legacy sweep, its moves and their re-exports are deleted.

**Tech Stack:** Python 3.11, JAX (x64), NumPy, pytest, ruff, pre-commit, uv.

**Spec:** No separate spec document. The design is the user's decision of 2026-11-10 ("Mirror #1183": `ipeps()` and `ctm_2site()` run `ctm_tensor_2site(recipe="2x2")` and return the raw `CTMEnvironment` pair in documented leg order; `ctm_2site()` deprecated like `ctm()`; legacy 2-site sweep and moves deleted; `compute_energy_ctm_2site()` sums all four bonds; all 8 RED tests stay). The defect analysis is the module docstring of `tests/test_ipeps_2site_energy.py` (commit `d6730510`). The pattern being mirrored is `docs/superpowers/specs/2026-11-04-ctm-delegate-to-2x2-design.md` and `docs/superpowers/plans/2026-11-09-ctm-delegate-to-2x2.md` (both arrive with #1183).

**Provenance:** Not dry-run. Every number in "Measured numbers" was read from probe output on 2026-11-10 at `d6730510` (main `2cb7c575` plus the RED test commit), CPU, x64; the probes are `/home/yjkao/.claude/jobs/1ad6c4e6/tmp/probes/probe_[e-q]*.py`, with logs beside them. They ran main's multisite loop, which has no `min_iter` gate before Task 1. The code blocks were written against `d6730510` merged with #1183's head `97ae6689` (`git merge-tree`, tree `9ec12fbb`); #1183 merged as `f14fca2e`. Every `old` text of a replace step was checked to occur exactly once in that tree. Expected test outcomes in the RED/GREEN steps are predictions from the probes, not runs of the edited code.

## Global Constraints

- **Prerequisite:** #1183 is merged into `main`. Task 0 merges `origin/main` into this branch. Nothing below runs before that: the helper needs `_ctm_env_dense._as_dense_env` and `_in_documented_leg_order`, which #1183 adds.
- `ctm_2site(A, B, config, *, return_meta=False)` keeps its signature. It returns `(CTMEnvironment, CTMEnvironment)`, or `(CTMEnvironment, CTMEnvironment, CTMConvergenceInfo)` with `return_meta=True`, as today. The info fields become Python scalars (`bool`, `int`, `float`).
- `ctm_2site()` emits one `DeprecationWarning` that names `ctm_tensor_2site` and `jax.jit`, with `stacklevel=2`. `_ctm_2site_via_2x2` emits no warning of its own, and passes `_warn_unconverged=False` to the loop, so `ipeps()` warns once (its own #839 `UserWarning`) when the CTM does not converge.
- `ipeps(hamiltonian_gate, initial_peps, config, *, compute_energy=True)` keeps its signature and return types. It still densifies the two site tensors (`A.todense()`, as today) before the CTM; the symmetric SU path is not changed.
- `_ctm_tensor_multisite` gains keyword-only `min_iter: int = 1`. Rule, as in `ctm_tensor` (#976, #1183): `converged` only when the total sweep count, QR warm-up included, is at least `min(min_iter, max_iter)`, with the caller's `max_iter` taken before the warm-up shortens the loop. The default 1 leaves `ctm_tensor_2site`, `ctm_multisite` and the excitation path unchanged. `ctm_tensor_2site` and `ctm_multisite` get no new public parameter.
- The `_return_status` info of `_ctm_tensor_multisite` reports `iterations` = sweeps that ran (warm-up included) on every exit; it reported `-1` on a converged run. Its readers: `ctm_tensor_2site(strict=True)` (reads it only when not converged) and `_ipeps_excitation_ctm._converged_env` (reads only `.converged`).
- The helper always passes `projector_method="svd"`: on the 2x2 recipe the projector is always Fishman SVD, but `"qr"` would still start the loop's QR warm-up. `config.qr_warmup_steps` is ignored.
- Raw `A` or `B`: only 5 legs `(u, d, l, r, s)` with four virtual legs of one dimension. Anything else gives a `ValueError` whose message contains `5 legs (u, d, l, r, s)`. A `DenseTensor` is passed through. A `SymmetricTensor` in either position gives a `TypeError` that names `ctm_tensor_2site`; never call `todense()` on it.
- The imports inside `_ctm_2site_via_2x2` stay local to the function body, as in `_ctm_via_2x2`. Write no comment that claims an import cycle.
- Out of scope: `ctm_tensor_2site`'s public signature (no `return_meta`, no `min_iter`); `ctm_split`; the symmetric SU path of `ipeps()`; `compute_excitations` (follow-up 1); the duplicate `_wrap_as_dense_tensor` in `ipeps.py:411` (the helper uses the one in `_ipeps_optimize_shared`, as `_ctm_via_2x2` does); CHANGELOG (written only in release PRs).
- A changed numeric assertion in an existing test gets examined, not re-frozen. The PR lists every changed test with its reason (Task 7 drafts the list).
- Run Python only through `uv run`, from `/home/yjkao/tenax`. Before each pytest run, check `uptime` and wait while the 1-minute load is above 12. One pytest run at a time, under `JAX_PLATFORMS=cpu taskset -c 0-11`. Redirect output to a file in `$W` and read the file afterwards; never pipe a run into `tail` or `head`. A run that can take more than 2 minutes goes in the background and ends with `exit $rc`. Pass `--no-cov`: `addopts` has `--cov=tenax --cov-report=xml`, which writes `coverage.xml` into the tree.
- A new test file needs a `_FILE_MARKERS` entry in `tests/conftest.py`, or `test_bucket_registry.py` fails.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`. pre-commit (ruff, ruff-format) must pass. If a hook rewrites a file, `git add` it and run the same `git commit` again: the failed commit left HEAD unchanged. Never use `--no-verify`, `--amend` after a hook failure, or `--reuse-message=HEAD`. Stage files by name; the untracked `out/` and `out2/` in the clone root are not ours.
- To bring the branch up to date: `git merge origin/main`, never rebase.
- No push, no PR and no `gh` call inside the tasks. After Task 7 the finishing skill asks the user. Arming `--auto` is the user's decision.

## Review Focus

1. The pair `ipeps()` returns, passed back to `ctm_2site()` (`DenseTensor`s, or a raw `A` with a `DenseTensor` `B`) → each sublattice converted on its own, same environments as the raw arrays, bit for bit. Pinned by `test_numpy_and_dense_tensor_inputs_match_the_jax_array` (Task 2; Task 5 adds the `ctm_2site` case).
2. A `SymmetricTensor` in the `B` position only → `TypeError` naming `ctm_tensor_2site`, not a silent `todense()` of `B` after `A` was wrapped. Pinned by `test_refuses_a_symmetric_tensor[B-*]` (Task 2).
3. A bad shape in the `B` position only, such as `(2, 2, 3, 3, 2)` → `ValueError` naming the 5-leg layout. Pinned by `test_rejects_a_bad_raw_shape[*-B-*]` (Task 2).
4. A state whose corners have rank 1 (singlets on every `A.r–B.l` bond) through `ipeps()` → the exact energy, −0.375, and one `UserWarning` that the CTM did not converge, instead of silence or a crash in the projector. Pinned by `test_ipeps_counts_every_bond_of_the_cell` (Task 4 turns its filter into `pytest.warns`).
5. A starved budget through `ipeps()` → exactly one "did not converge" warning, from `ipeps()`, not a second one from the loop inside it. Pinned by `test_ipeps_warns_when_its_ctm_did_not_converge` (Task 4).

## File Structure

| File | Change | Task |
|---|---|---|
| `src/tenax/algorithms/_ctm_tensor_convergence.py` | `_ctm_tensor_multisite(min_iter=)`, sweep count on every exit (1); `CTMConvergenceInfo` docstring (5) | 1, 5 |
| `src/tenax/algorithms/ipeps_ctm_convergence.py` | `_ctm_2site_via_2x2` (2); `ctm_2site()` delegates, legacy sweep and `_renormalize_env` deleted, module comment, two stale sentences in `ctm_split` (5) | 2, 5 |
| `src/tenax/algorithms/ipeps_rdm.py` | `compute_energy_ctm_2site` sums four bonds | 3 |
| `src/tenax/algorithms/ipeps.py` | `ipeps()` calls the helper; #839 text; docstring | 4 |
| `src/tenax/algorithms/ipeps_ctm_moves.py` | delete `_ctm_move_eigh`, `_ctm_move`, the four 2-site moves | 5 |
| `src/tenax/algorithms/ipeps_ctm.py` | drop eight re-exports | 5 |
| `pyproject.toml` | filter for the new `DeprecationWarning` | 5 |
| `tests/test_ctm_2site_delegates_to_2x2.py` | **new**, `core`: orientation, inputs, warnings | 2, 5 |
| `tests/conftest.py` | `core` bucket for the new file | 2 |
| `tests/test_ctm_min_iter_contract_925_976.py` | multisite cases and two multisite tests (1); helper cases (2); docstring (5) | 1, 2, 5 |
| `tests/test_rdm_validity_guard.py` | fixture from the helper; two tests re-derived for four bonds | 3 |
| `tests/test_ipeps_2site_energy.py` | dimer test asserts the warning (4); docstring wording (6) | 4, 6 |
| `tests/test_ctm_convergence_meta.py` | `ipeps()` starved config, one warning, helper, bomb target (4); `ctm_2site` starved config, jit test removed, docstrings (5) | 4, 5 |
| `tests/test_ctm_criterion_rank_blind_898.py` | routing test re-pointed at the multisite loop (4); the guarded-criterion test's docstring and message (5) | 4, 5 |
| `tests/test_ipeps.py` | `test_2site_heisenberg_D2_energy`: bound and docstring | 4 |
| `tests/test_su_667_product_state.py` | stale docstring and comment | 4 |
| `tests/test_ctm_complex_site_tensors.py` | module docstring note; one test docstring | 5 |
| `tests/test_ctm_tensor.py` | drop the `_ctm_2site_sweep` import and the two skipped legacy stubs | 5 |
| `docs/guide/algorithms/ipeps.md`, `docs/guide/algorithms/ctm.md`, `README.md`, `.claude/skills/tenax-ipeps-workflow/SKILL.md`, `.claude/skills/tenax-observables/SKILL.md` | the new measurement, the deprecation, four bonds, the skill's `dt=0.3` energy | 6 |

Checked and not changed: `src/tenax/__init__.py` and `src/tenax/algorithms/__init__.py` keep exporting `ctm_2site` and `compute_energy_ctm_2site` (deprecated, still public); `docs/api/algorithms.rst` autodocs the new docstrings; `tests/test_ipeps.py` imports `ctm_2site` and `compute_energy_ctm_2site` without calling them; `benchmarks/` only mention the legacy carry bug (#842) in comments.

## Measured numbers

Measured 2026-11-10 at `d6730510`, CPU, x64, on main's multisite loop (`_ctm_tensor_multisite(..., recipe="2x2", projector_method="svd")`) with #1183's leg reordering copied into the probe. "Raw" means the raw 2-site RDMs of `ipeps_rdm` on the converted environments; "four bonds" means half their sum. "Legacy" means the `ctm_2site` sweep of `d6730510`.

**That loop had no `min_iter` gate.** The helper passes `CTMConfig.min_iter` (default 10), so a run below that stopped before sweep 10 stops at sweep 10 through the helper. The guide example stops at sweep 10, not 8; its environment is then the one of the `conv_tol=1e-10` run at chi=10, which also stopped at sweep 10 (−0.6567903814). The `conv_tol=1e-2` case stops at sweep 10, not 3.

| state, chi | result |
|---|---|
| uniform rotated-gate SU state (40 × 0.1), chi=6, `conv_tol=1e-10` | 1-site `ctm_tensor` E1 = −0.6586443806 (13 sweeps). Pair `(A0, A0)`: four bonds = Tensor = E1 to 0. Gauge 1 / gauge 2: four bonds = Tensor to ≤ 1.1e-16, = E1 to 2.7e-11 / 3.7e-11. Without the corner reorder: 2.2e-5 / 4.0e-3 off. |
| convergence-meta `su_state` (rotated gate, 40 × 0.05), chi=6 | converges at sweep 11 (`conv_tol=1e-10`) and 13 (`1e-12`); criterion after 5, 8, 10, 12 sweeps: 1.8e-5, 1.7e-8, 2.0e-10, 2.9e-12; `conv_tol=1e-2` met at sweep 3 |
| #845 guard fixture (random seeds 0, 7, unnormalised), chi=4, 5 sweeps | not converged (criterion 0.111); all four RDMs trace 1, smallest eigenvalue 0.141; zeroing one corner of `env_B` kills two of the four bonds, one edge kills three |
| plain-gate SU state (200 × 0.05), the guide example | E = −0.6567903814 at chi = 10, 16, 24, 32 (10, 10, 11, 10 sweeps, `conv_tol=1e-10`); −0.6567903816 at chi=10, `conv_tol=1e-8` (8 sweeps). Bonds h AB −0.315558, h BA −0.332450, v AB −0.323706, v BA −0.341867: the two `AB` bonds alone give −0.639264 |
| dimer (singlets on every `A.r–B.l` bond), chi=4 | legacy env + four bonds: −0.375 exactly (converged at 10). 2x2 env: finite, −0.375 exactly, no RDM warning, not converged (rank-1 corners, criterion `inf`); the loop's own warning is a `RuntimeWarning` ("could not be certified") when `_warn_unconverged=True` |
| asymmetric pair (product + 0.3 noise, seeds 5 and 6; random symmetric H, seed 7), chi=8, `conv_tol=1e-12` | converged (criterion 8.1e-13); stored corner labels reversed (`C1` `(c1_r, c1_d)`, `C2` `(c2_d, c2_l)`, `C3` `(c3_l, c3_u)`, `C4` `(c4_u, c4_r)`); weakest corner asymmetry 0.057; four bonds = Tensor to 2.1e-16; stored order 2.25e-2 off; `env_A`/`env_B` swapped 0.22 off; any one corner transposed ≥ 4.5e-4 off |
| `ONES` product pair, chi=1 | criterion met at sweep 2; with `projector_method="qr", qr_warmup_steps=6`: 8 sweeps (6 + 2), also with `max_iter=10` |
| random pair (seeds 42, 43, normalised), chi=4, 5 sweeps | peak \|`env_A.C1`\| 1.0 with `renormalize=True`, 0.947 without (`env_B.C1`: 0.996) |
| Ising PEPS beta=0.2 as a pair, chi=8, `conv_tol=1e-12` | converged in 40 sweeps; `C1` (8, 8), `T1` (8, 4, 8) |
| legacy environment, four bonds (probe K) | the swap test's pair, raw, chi=6, `max_iter=200`, `conv_tol=1e-10`: E(A, B) = E(B, A) = −0.6419137799, not converged. The state of `test_2site_heisenberg_D2_energy` (chi=10, `max_iter=40`): −0.6507164561, not converged. Uniform pair: −0.6586628723 against E1 = −0.6586443806 (1.85e-5 off) |
| legacy, the order of the pair (probe L) | each environment is the same whichever order the pair is passed in (all 8 fields differ by 0.0), so its four-bond energy is swap-symmetric. Bonds of the swap test's pair: h AB −0.303228, v AB −0.317691, h BA −0.326271, v BA −0.336637. The RED docstring's −0.62569 and −0.66244 come through `ipeps()`, which normalises the pair first; the raw pair gives −0.62092 and −0.66291, so the legacy sweep is not scale-invariant |
| dimer through `ipeps()` at `d6730510` (probe M) | E = −0.750000000000 and no warning: the legacy loop certifies the dimer |
| the doc examples, verbatim (probes N, Q) | before: the guide's first example printed −0.633605, its second example, the README and the skill −0.630276, each with the `UserWarning` "CTM did not converge in ipeps()". 2x2 four bonds: −0.6567903670 (first example, float32 gate) and −0.6567903816 (the rest); both print −0.656790, with no warning. `ctm_tensor_2site` at chi=16 or 24 gives the same 9 digits. The second example's einsum gate equals `heisenberg_gate()` |
| the docs' `dt` claims (probe Q) | first guide example at `dt=0.01`: before 0.471383 with three warnings, now 0.4862144215 (so "E > 0" still holds). The skill's config at `dt=0.3`: before −0.528245 (the skill's "≈ −0.53"), now −0.6452164768, no warning |
| legacy `ctm_2site` on the new tests' inputs (probe O) | raw arrays run. A `DenseTensor` in either position, or a `SymmetricTensor`: `TypeError` from `jnp.conjugate`. Bad shapes: `(4, 4, 4)` runs without an error in either position; `(2, 2, 2, 2)` and `(2, 2, 2, 2, 2, 2)` raise einsum `ValueError`s; `(2, 2, 3, 3, 2)` raises a reshape `TypeError`. No message names the 5-leg layout or `ctm_tensor_2site` |
| complex pair (seeds 0, 7, complex128), chi=4, 6 sweeps (probe P) | no error, no warning; every field complex128; max\|T.imag\| 0.516 (`env_A`) and 0.354 (`env_B`); the criterion 9.9e-4 is a Python `float`; `iterations` 6 |

## Execution setup (once, before Task 0)

```bash
cd /home/yjkao/tenax
git status --short                      # Expected: only "?? out" and "?? out2"
git branch --show-current               # Expected: fix/2site-raw-rdm-zero-trace
test -x "$(git rev-parse --git-path hooks)/pre-commit" || uv run pre-commit install
W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2
mkdir -p "$W"
uptime                                  # wait while the 1-minute load is above 12
```

---

### Task 0: Merge #1183 into the branch

**Files:** none edited.

Done on 2026-11-10: #1183 merged as `f14fca2e` (07:02 UTC); the merge commit is `d0e668e4`, with no conflicts, and its tree is `9ec12fbb`, the tree the anchors were checked against; Step 3 printed `2` and `8 failed`.

- [x] **Step 1: Confirm #1183 merged**

```bash
cd /home/yjkao/tenax && gh pr view 1183 --json state,mergeCommit --jq '"\(.state) \(.mergeCommit.oid)"'
```

Expected: `MERGED <sha>`. Anything else: stop and report.

- [x] **Step 2: Merge**

```bash
cd /home/yjkao/tenax && git fetch origin && git merge origin/main
```

Expected: a merge commit, no conflicts (this branch touches only `tests/test_ipeps_2site_energy.py` and one `tests/conftest.py` entry; #1183 adds an entry elsewhere in the same dict). On a conflict in `tests/conftest.py`, keep both entries.

- [x] **Step 3: Check the prerequisites and the RED state**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
command grep -c "def _in_documented_leg_order\|def _as_dense_env" src/tenax/algorithms/_ctm_env_dense.py && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_2site_energy.py > "$W/t0_red.txt" 2>&1; echo "rc=$?"
```

Expected: `2`, then `rc=1`; `$W/t0_red.txt` ends with `8 failed`.

---

### Task 1: `min_iter` and the sweep count in the multisite loop

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` (`_ctm_tensor_multisite`: signature, docstring, gate, status)
- Test: `tests/test_ctm_min_iter_contract_925_976.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_ctm_tensor_multisite(site_tensors, neighbors, chi, max_iter=100, conv_tol=1e-8, renormalize=True, projector_method="svd", qr_warmup_steps=3, projector_backward="auto", recipe="2x2", _deprecation_stacklevel=3, hold_sweeps=0, hold_perturbation=..., *, min_iter=1, _return_status=False, _warn_unconverged=True)`. With `_return_status=True` it returns `(envs, CTMConvergeInfo, detail)`, and `CTMConvergeInfo.iterations` is the number of sweeps that ran, QR warm-up included, on every exit.

- [ ] **Step 1: Write the failing tests**

In `tests/test_ctm_min_iter_contract_925_976.py`, add to the imports (keep them sorted; ruff checks):

```python
from types import SimpleNamespace

from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_multisite,
)
```

After `_info_ctm_tensor`, add:

```python
def _multisite_status(cfg, **kwargs):
    """The status of the multisite loop on the checkerboard ``(A, A)``, with
    ``n_iter`` named as in ``CTMConvergenceInfo`` (the loop's own type calls it
    ``iterations``)."""
    site = _wrap_as_dense_tensor(A)
    _envs, info, _detail = _ctm_tensor_multisite(
        {(0, 0): site, (1, 0): site},
        CHECKERBOARD_NEIGHBORS,
        cfg.chi,
        max_iter=cfg.max_iter,
        conv_tol=cfg.conv_tol,
        _return_status=True,
        **kwargs,
    )
    return SimpleNamespace(converged=info.converged, n_iter=info.iterations)


def _info_ctm_tensor_multisite(cfg):
    return _multisite_status(cfg, min_iter=cfg.min_iter)
```

Add `pytest.param(_info_ctm_tensor_multisite, id="ctm_tensor_multisite"),` as the last entry of `_ENTRY_POINTS`. At the end of the file, add:

```python
def test_multisite_default_min_iter_keeps_the_early_exit():
    """``min_iter`` defaults to 1, so ``ctm_tensor_2site`` and ``ctm_multisite``,
    which do not pass it, keep the plain early exit: the stationary state
    certifies at the first comparison, sweep 2.  The count is reported on a
    converged run too; it was ``-1`` before."""
    info = _multisite_status(CTMConfig(chi=1, max_iter=40, conv_tol=1e-8))
    assert info.converged
    assert info.n_iter == 2


def test_multisite_min_iter_counts_the_warmup_against_the_callers_budget():
    """The cap is ``min(min_iter, max_iter)`` with the caller's ``max_iter``, and
    the QR warm-up sweeps count toward it.  Six warm-up sweeps leave four in the
    loop, and the loop's first sweep has nothing to compare with; a cap taken
    after the warm-up, ``min(10, 4) = 4``, would certify at sweep 8."""
    info = _multisite_status(
        CTMConfig(chi=1, max_iter=10, conv_tol=1e-8),
        projector_method="qr",
        qr_warmup_steps=6,
        min_iter=10,
    )
    assert info.converged
    assert info.n_iter == 10
```

- [ ] **Step 2: Run them to see them fail**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_min_iter_contract_925_976.py > "$W/t1_red.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=1`; 6 failed: `test_convergence_waits_for_min_iter[2|10|30-ctm_tensor_multisite]`, `test_min_iter_is_capped_at_max_iter[ctm_tensor_multisite]` and the warm-up test with `TypeError: ... unexpected keyword argument 'min_iter'`; the default test with `assert -1 == 2`. Every other case passes.

- [ ] **Step 3: Implement**

In `_ctm_tensor_multisite`'s signature, replace

```python
    *,
    _return_status: bool = False,
    _warn_unconverged: bool = True,
) -> dict[Coord, CTMTensorEnv]:
```

with

```python
    *,
    min_iter: int = 1,
    _return_status: bool = False,
    _warn_unconverged: bool = True,
) -> dict[Coord, CTMTensorEnv]:
```

In its docstring, replace

```
        hold_perturbation: See :func:`ctm_tensor_2site`.
        _return_status: Private.  When true, return ``(envs, info, detail)``
                      with ``info`` a :class:`CTMConvergeInfo` (``iterations``
                      is the sweeps actually run when not converged, ``-1``
                      otherwise) and ``detail`` the not-converged diagnostic
```

with

```
        hold_perturbation: See :func:`ctm_tensor_2site`.
        min_iter:     Minimum sweeps before ``converged`` can be set, capped at
                      ``max_iter`` so that a small budget stays convergeable
                      (#976).  QR warm-up sweeps count toward it.  The default,
                      1, keeps the plain early exit; ``ctm_2site`` passes
                      ``CTMConfig.min_iter``.
        _return_status: Private.  When true, return ``(envs, info, detail)``
                      with ``info`` a :class:`CTMConvergeInfo` (``iterations``
                      is the number of sweeps that ran, QR warm-up included,
                      converged or not) and ``detail`` the not-converged diagnostic
```

Replace

```python
    budget = max_iter

    # QR warm-up: run a few eigh iterations before switching to QR
```

with

```python
    budget = max_iter
    # ``converged`` needs ``min(min_iter, max_iter)`` sweeps, warm-up included,
    # as in ``ctm_tensor`` (#976).  The cap uses the caller's budget, before the
    # warm-up below shortens ``max_iter``.
    min_sweeps = min(min_iter, budget)

    # QR warm-up: run a few eigh iterations before switching to QR
```

Replace

```python
        if ever_measured:
            final_diff = sweep_diff
        if converged and hold_sweeps:
```

with

```python
        if ever_measured:
            final_diff = sweep_diff
        # ``budget - max_iter`` is the QR warm-up.  The last sweep of the budget
        # always passes this gate, because ``min_sweeps <= budget``.
        if converged and budget - max_iter + used < min_sweeps:
            converged = False
        if converged and hold_sweeps:
```

Replace

```python
        # Codex P2 on #1070: count the sweeps that ran, not the budget -- the
        # hold's budget and non-finite exits stop early.  ``budget - max_iter``
        # is the QR warm-up.
        info = CTMConvergeInfo(
            converged=bool(converged),
            iterations=(budget - max_iter + used) if not converged else -1,
            sv_diff=float(final_diff),
        )
```

with

```python
        # Codex P2 on #1070: count the sweeps that ran, not the budget -- the
        # hold's budget and non-finite exits stop early.  ``budget - max_iter``
        # is the QR warm-up.  A converged run reports its count too:
        # ``ctm_2site(return_meta=True)`` reads it as ``n_iter``.
        info = CTMConvergeInfo(
            converged=bool(converged),
            iterations=budget - max_iter + used,
            sv_diff=float(final_diff),
        )
```

- [ ] **Step 4: Run the contract tests, then the loop's other callers**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q \
  tests/test_ctm_min_iter_contract_925_976.py tests/test_ctm_unconverged_policy.py \
  tests/test_ctm_hold.py tests/test_ctm_tensor_return_info.py tests/test_recipe_1x1_deprecation.py \
  > "$W/t1_green.txt" 2>&1; rc=$?; echo "rc=$rc"; exit $rc
```

Expected: `rc=0`. Run it in the background if `uptime` shows load above 4 (the hold tests take a few minutes).

- [ ] **Step 5: Commit**

```bash
cd /home/yjkao/tenax && git add src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_ctm_min_iter_contract_925_976.py && \
git commit -m "feat(ctm): min_iter and a sweep count on every exit in the multisite loop

_ctm_tensor_multisite takes a keyword-only min_iter (default 1, so
ctm_tensor_2site and ctm_multisite keep their early exit), with the rule
ctm_tensor uses since #1183: converged only after min(min_iter, max_iter)
sweeps, QR warm-up included, capped at the caller's budget (#976).  Its
private status reported iterations=-1 on a converged run; it now reports
the sweeps that ran on every exit, which ctm_2site(return_meta=True) will
read as n_iter.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 2: `_ctm_2site_via_2x2`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ctm_convergence.py` (new function after `_ctm_via_2x2`; `__all__`)
- Create: `tests/test_ctm_2site_delegates_to_2x2.py`
- Modify: `tests/conftest.py` (`_FILE_MARKERS`)
- Modify: `tests/test_ctm_min_iter_contract_925_976.py` (the helper's cases)

**Interfaces:**
- Consumes: Task 1's `_ctm_tensor_multisite(..., min_iter=, _return_status=True, _warn_unconverged=False)`; #1183's `_as_dense_env`, `_in_documented_leg_order` (`tenax.algorithms._ctm_env_dense`); `_wrap_as_dense_tensor` (`tenax.algorithms._ipeps_optimize_shared`); `CHECKERBOARD_NEIGHBORS` (`tenax.algorithms._ctm_tensor_convergence`).
- Produces: `_ctm_2site_via_2x2(A, B, config: CTMConfig, *, return_meta: bool = False)` → `(CTMEnvironment, CTMEnvironment)` or `(CTMEnvironment, CTMEnvironment, CTMConvergenceInfo)`, info fields Python `bool`, `int`, `float`. Tasks 3, 4 and 5 call it.

- [ ] **Step 1: Write the test file**

Create `tests/test_ctm_2site_delegates_to_2x2.py`:

```python
"""``ctm_2site()`` and ``ipeps()`` run the 2x2 recipe of ``ctm_tensor_2site``
since 0.9.3.

The legacy sweep built each sublattice's new boundary from that sublattice's
own corners and edges, with a projector of its own, so the two environments did
not share a basis at the A|B seams and the energy moved with the bond gauge of
the pair; ``tests/test_ipeps_2site_energy.py`` pins that.  This file pins the
delegation.  ``_ctm_2site_via_2x2`` is ``ctm_2site()`` without its warnings: it
maps the ``CTMConfig`` fields onto the multisite loop of ``ctm_tensor_2site`` and
returns two raw-array ``CTMEnvironment`` named tuples whose legs are in the order
the raw 2-site RDMs contract.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _window_oracle import ising_peps

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_config import CTMConfig, CTMEnvironment
from tenax.algorithms.ipeps_ctm_convergence import (
    CTMConvergenceInfo,
    _ctm_2site_via_2x2,
)
from tenax.algorithms.ipeps_rdm import _rdm1x2_2site, _rdm2x1_2site
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

# A 1x1 product state: its chi=1 corners are stationary from the second sweep,
# so the stopping rule is all that moves.
ONES = jnp.ones((1, 1, 1, 1, 2))

# Every entry point that must return the environments of the 2x2 recipe.
ENTRY_POINTS = [
    pytest.param(_ctm_2site_via_2x2, id="helper"),
]


def _random_site(seed):
    A = jax.random.normal(jax.random.PRNGKey(seed), (2, 2, 2, 2, 2))
    return A / jnp.linalg.norm(A)


def _symmetric_site():
    sym = U1Symmetry()
    zeros = np.zeros(2, dtype=np.int32)
    legs = [
        ("u", FlowDirection.OUT),
        ("d", FlowDirection.IN),
        ("l", FlowDirection.OUT),
        ("r", FlowDirection.IN),
        ("phys", FlowDirection.IN),
    ]
    indices = tuple(
        TensorIndex.from_charges(sym, zeros.copy(), flow, label=label)
        for label, flow in legs
    )
    return SymmetricTensor.random_normal(indices=indices, key=jax.random.PRNGKey(0))


def _asymmetric_site(seed, eps=0.3):
    """A D=2 product state plus ``eps`` x Gaussian noise: no lattice symmetry."""
    A = np.zeros((2, 2, 2, 2, 2))
    A[0, 0, 0, 0, 0] = 1.0
    A += eps * np.random.default_rng(seed).standard_normal(A.shape)
    return jnp.asarray(A)


def _asymmetric_hamiltonian(seed=7):
    """A random real-symmetric two-site operator.  It has no site-exchange
    symmetry: ``max|H - SWAP H SWAP|`` is 1.48 (measured)."""
    M = np.random.default_rng(seed).standard_normal((4, 4))
    return jnp.asarray((M + M.T) / 2)


def _raw_energy(A, B, env_A, env_B, H):
    """Half the sum of the four bond energies, from the raw 2-site RDMs.

    Written out here, not read from ``compute_energy_ctm_2site``, so that the
    orientation test does not depend on that function's bond count."""
    H = H.reshape(2, 2, 2, 2)
    rdms = (
        _rdm2x1_2site(A, B, env_A, env_B, 2),
        _rdm2x1_2site(B, A, env_B, env_A, 2),
        _rdm1x2_2site(A, B, env_A, env_B, 2),
        _rdm1x2_2site(B, A, env_B, env_A, 2),
    )
    return 0.5 * sum(float(jnp.einsum("ijkl,klij->", r, H).real) for r in rdms)


def _assert_same_envs(envs, ref):
    for env, ref_env in zip(envs, ref, strict=True):
        for name in CTMEnvironment._fields:
            np.testing.assert_array_equal(getattr(env, name), getattr(ref_env, name))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry(entry):
    """The raw environments are the 2x2 environments, in the order the raw
    2-site RDMs contract.

    The multisite sweep stores every corner with its two legs reversed (``C1``
    as ``(c1_r, c1_d)``), and the raw RDMs contract by position.  On a
    lattice-symmetric state each corner is a symmetric matrix, so a transposed
    corner changes nothing.  This pair has no lattice symmetry and ``H`` no
    site-exchange symmetry, so the raw energy must equal the energy that
    ``compute_energy_ctm_tensor_2site`` takes, by label, from the environments of
    ``ctm_tensor_2site``.  Measured on this pair at d6730510: the criterion
    reaches 8.1e-13; the weakest corner asymmetry is 0.057; with the corners as
    the sweep stores them the raw energy is 2.3e-2 off, with ``env_A`` and
    ``env_B`` swapped 0.22 off, and each corner alone moves it by at least 4.5e-4.
    """
    A, B, H = _asymmetric_site(5), _asymmetric_site(6), _asymmetric_hamiltonian()
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env_A, env_B, info = entry(A, B, cfg, return_meta=True)
    assert info.converged

    # The regime, so the test cannot go vacuous: a corner that is a symmetric
    # matrix is blind to a transposition.
    asymmetry = [
        float(jnp.max(jnp.abs(C - C.T)) / jnp.max(jnp.abs(C)))
        for env in (env_A, env_B)
        for C in (env.C1, env.C2, env.C3, env.C4)
    ]
    assert min(asymmetry) > 5e-3, asymmetry

    At, Bt = _wrap_as_dense_tensor(A), _wrap_as_dense_tensor(B)
    tensor_A, tensor_B = ctm_tensor_2site(
        At, Bt, cfg.chi, max_iter=cfg.max_iter, conv_tol=cfg.conv_tol
    )
    E_raw = _raw_energy(A, B, env_A, env_B, H)
    E_tensor = float(compute_energy_ctm_tensor_2site(At, Bt, tensor_A, tensor_B, H, 2))
    assert abs(E_raw - E_tensor) < 1e-9, (E_raw, E_tensor)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad", ["A", "B"])
@pytest.mark.parametrize(
    "shape",
    [(4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2, 2, 2), (2, 2, 3, 3, 2)],
    ids=["3-leg", "4-leg", "6-leg", "unequal-virtual"],
)
def test_rejects_a_bad_raw_shape(entry, bad, shape):
    """Only 5 legs ``(u, d, l, r, s)`` with one virtual dimension is a site
    tensor, on either sublattice: ``_wrap_as_dense_tensor`` reads ``shape[0]``
    for all four virtual legs and ``shape[4]`` for the physical one, unchecked."""
    good = _random_site(0)
    pair = (jnp.ones(shape), good) if bad == "A" else (good, jnp.ones(shape))
    with pytest.raises(ValueError, match=r"5 legs \(u, d, l, r, s\)"):
        entry(*pair, CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad", ["A", "B"])
def test_refuses_a_symmetric_tensor(entry, bad):
    """The raw-array result would need ``todense()`` on a block-sparse tensor;
    the message names ``ctm_tensor_2site``, which keeps the blocks."""
    good = _random_site(0)
    pair = (_symmetric_site(), good) if bad == "A" else (good, _symmetric_site())
    with pytest.raises(TypeError, match="ctm_tensor_2site"):
        entry(*pair, CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_numpy_and_dense_tensor_inputs_match_the_jax_array(entry):
    """NumPy arrays are converted and ``DenseTensor``s (what ``ipeps()``
    returns) pass through, each sublattice on its own: every mix gives the
    environments of the JAX arrays, bit for bit."""
    A, B = _random_site(0), _random_site(7)
    cfg = CTMConfig(chi=4, max_iter=5)
    ref = entry(A, B, cfg)
    for pair in (
        (np.asarray(A), np.asarray(B)),
        (_wrap_as_dense_tensor(A), _wrap_as_dense_tensor(B)),
        (A, _wrap_as_dense_tensor(B)),
        (np.asarray(A), B),
    ):
        _assert_same_envs(entry(*pair, cfg), ref)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_returns_raw_environments_and_python_scalars(entry):
    """Two raw environments with ``jax.Array`` fields of shape ``(chi, chi)`` and
    ``(chi, D**2, chi)``, as ``ctm_2site()`` always returned.  The info holds
    Python scalars, because the multisite loop is a Python loop."""
    A = jnp.asarray(ising_peps(0.2))
    env_A, env_B, info = entry(
        A, A, CTMConfig(chi=8, max_iter=500, conv_tol=1e-12), return_meta=True
    )
    for env in (env_A, env_B):
        assert isinstance(env, CTMEnvironment)
        assert isinstance(env.C1, jax.Array) and env.C1.shape == (8, 8)
        assert isinstance(env.T1, jax.Array) and env.T1.shape == (8, 4, 8)
    assert isinstance(info, CTMConvergenceInfo)
    assert type(info.converged) is bool
    assert type(info.n_iter) is int
    assert type(info.diff) is float


def test_helper_runs_no_qr_warmup():
    """``"svd"`` is passed, so a ``"qr"`` config starts no warm-up loop: the
    stationary state stops at sweep 2, not after six warm-up sweeps (8 sweeps,
    measured with ``"qr"`` passed on)."""
    _env_A, _env_B, info = _ctm_2site_via_2x2(
        ONES,
        ONES,
        CTMConfig(
            chi=1, max_iter=40, min_iter=1, projector_method="qr", qr_warmup_steps=6
        ),
        return_meta=True,
    )
    assert info.converged
    assert info.n_iter == 2


@pytest.mark.parametrize("renormalize", [True, False])
def test_helper_passes_renormalize(renormalize):
    """``renormalize`` reaches the loop: with it the largest entry of
    ``env_A.C1`` is 1; without it, 0.947 on this pair (measured)."""
    env_A, _env_B = _ctm_2site_via_2x2(
        _random_site(42),
        _random_site(43),
        CTMConfig(chi=4, max_iter=5, renormalize=renormalize),
    )
    peak = float(jnp.max(jnp.abs(env_A.C1)))
    if renormalize:
        assert peak == pytest.approx(1.0, abs=1e-12)
    else:
        assert peak < 0.99


def test_helper_emits_no_deprecation_warning():
    """``ipeps()`` and the fixture-only tests call the helper, so they do not
    trigger ``ctm_2site()``'s ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ctm_2site_via_2x2(ONES, ONES, CTMConfig(chi=1, max_iter=5))
```

In `tests/conftest.py`, after the line `"test_ctm_delegates_to_2x2.py": "core",` (added by #1183), add:

```python
    # ``ctm_2site()`` and ``ipeps()`` run the 2x2 recipe of ``ctm_tensor_2site``
    # since 0.9.3.  The orientation pin compares the raw four-bond energy with
    # the Tensor energy on a pair without lattice symmetry, where a transposed
    # corner or swapped environments move it by 4.5e-4 or more.  A silent wrong
    # environment on a public entry point, so it belongs in the required gate.
    "test_ctm_2site_delegates_to_2x2.py": "core",
```

In `tests/test_ctm_min_iter_contract_925_976.py`, replace

```python
from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2
```

with

```python
from tenax.algorithms.ipeps_ctm_convergence import _ctm_2site_via_2x2, _ctm_via_2x2
```

after `_info_ctm_via_2x2` add

```python
def _info_ctm_2site_via_2x2(cfg):
    return _ctm_2site_via_2x2(A, A, cfg, return_meta=True)[2]
```

and in `_ENTRY_POINTS` insert `    pytest.param(_info_ctm_2site_via_2x2, id="ctm_2site_via_2x2"),` before the `ctm_tensor_multisite` entry that Task 1 added.

- [ ] **Step 2: Run them to see them fail**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_2site_delegates_to_2x2.py \
  tests/test_ctm_min_iter_contract_925_976.py > "$W/t2_red.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=2`, two collection errors, each `ImportError: cannot import name '_ctm_2site_via_2x2'`.

- [ ] **Step 3: Implement**

In `src/tenax/algorithms/ipeps_ctm_convergence.py`, in `__all__`, replace

```python
    "_ctm_via_2x2",
```

with

```python
    "_ctm_via_2x2",
    "_ctm_2site_via_2x2",
```

After the end of `_ctm_via_2x2` (its last line is `    return env`, followed by two blank lines and `def _renormalize_env`), insert:

```python
def _ctm_2site_via_2x2(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> (
    tuple[CTMEnvironment, CTMEnvironment]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo]
):
    """``ctm_2site()`` without its warnings: the 2x2 recipe of
    ``ctm_tensor_2site`` with the ``CTMConfig`` fields mapped, returned as two
    raw-array ``CTMEnvironment`` named tuples.

    For callers that need ``ctm_2site()``'s result and must not trigger its
    ``DeprecationWarning``: ``ipeps()`` and the fixture-only tests.

    Args:
        A:           Site tensor of sublattice A: a raw array with 5 legs
                     ``(u, d, l, r, s)`` and four virtual legs of one dimension,
                     or a ``DenseTensor``, which is passed through.
                     ``_wrap_as_dense_tensor`` reads ``shape[0]`` for all four
                     virtual legs without a check, so any other shape raises
                     ``ValueError`` here.  A ``SymmetricTensor`` raises
                     ``TypeError``: the raw result would need ``todense()``.
        B:           Site tensor of sublattice B, checked the same way.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` are passed on.  ``projector_method`` is not:
                     the 2x2 recipe always uses Fishman SVD, and ``"svd"`` is
                     passed because ``"qr"`` would still start the QR warm-up
                     loop.  So ``qr_warmup_steps`` has no effect.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`,
                     whose fields are Python scalars.

    Returns:
        ``(env_A, env_B)``, or ``(env_A, env_B, info)``.  Every field has its
        legs in the order ``CTMTensorEnv`` documents, which is the order the raw
        2-site RDMs of ``ipeps_rdm`` contract.  The loop's own not-converged
        warning is off: ``ipeps()`` warns itself, and ``ctm_2site()`` reports
        through ``info`` as it always did.
    """
    from tenax.algorithms._ctm_env_dense import _as_dense_env, _in_documented_leg_order
    from tenax.algorithms._ctm_tensor_convergence import (
        CHECKERBOARD_NEIGHBORS,
        _ctm_tensor_multisite,
    )
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.core.tensor import SymmetricTensor, Tensor

    sites = {}
    for name, coord, X in (("A", (0, 0), A), ("B", (1, 0), B)):
        if isinstance(X, SymmetricTensor):
            raise TypeError(
                f"ctm_2site() returns raw-array CTMEnvironments, so the "
                f"SymmetricTensor site tensor {name} would have to be densified. "
                "Call ctm_tensor_2site(A, B, chi, ...) instead: it keeps the "
                "blocks and returns two CTMTensorEnvs."
            )
        if not isinstance(X, Tensor):
            X = jnp.asarray(X)
            if X.ndim != 5 or len(set(X.shape[:4])) != 1:
                raise ValueError(
                    "Expected raw site tensors with 5 legs (u, d, l, r, s) and "
                    f"four virtual legs of one dimension; got shape {X.shape} "
                    f"for {name}."
                )
            X = _wrap_as_dense_tensor(X)
        sites[coord] = X
    envs, status, _detail = _ctm_tensor_multisite(
        sites,
        CHECKERBOARD_NEIGHBORS,
        config.chi,
        max_iter=config.max_iter,
        conv_tol=config.conv_tol,
        renormalize=config.renormalize,
        projector_method="svd",
        recipe="2x2",
        min_iter=config.min_iter,
        _return_status=True,
        _warn_unconverged=False,
    )
    # The 2x2 sweep stores each corner with its two legs reversed, and the raw
    # consumers contract by position: put every field in the documented order.
    env_A = _as_dense_env(_in_documented_leg_order(envs[(0, 0)]))
    env_B = _as_dense_env(_in_documented_leg_order(envs[(1, 0)]))
    if return_meta:
        info = CTMConvergenceInfo(
            converged=status.converged, n_iter=status.iterations, diff=status.sv_diff
        )
        return env_A, env_B, info
    return env_A, env_B
```

- [ ] **Step 4: Run the new file, the contract file and the bucket registry**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_2site_delegates_to_2x2.py \
  tests/test_ctm_min_iter_contract_925_976.py tests/test_bucket_registry.py > "$W/t2_green.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=0`. The new file collects 17 tests (1 + 8 + 2 + 1 + 1 for the one entry point, then 1 + 2 + 1); the contract file gains 4 cases, `test_convergence_waits_for_min_iter[2|10|30-ctm_2site_via_2x2]` and `test_min_iter_is_capped_at_max_iter[ctm_2site_via_2x2]`.

- [ ] **Step 5: Commit**

```bash
cd /home/yjkao/tenax && git add src/tenax/algorithms/ipeps_ctm_convergence.py tests/test_ctm_2site_delegates_to_2x2.py tests/conftest.py \
  tests/test_ctm_min_iter_contract_925_976.py && \
git commit -m "feat(ctm): _ctm_2site_via_2x2, the 2x2 recipe as a raw CTMEnvironment pair

The checkerboard counterpart of _ctm_via_2x2 (#1183): checks and wraps the
two site tensors, maps the CTMConfig fields onto the multisite loop with
CHECKERBOARD_NEIGHBORS and recipe=\"2x2\", always passes projector_method=
\"svd\", keeps the loop's not-converged warning off, and converts each
environment to the documented leg order the raw 2-site RDMs contract.  On a
pair without lattice symmetry the stored corner order moves the raw energy
by 2.3e-2 and swapped environments by 0.22; the new orientation test pins
both.  The min_iter contract tests run on the helper too.  Nothing calls
the helper yet.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 3: `compute_energy_ctm_2site` sums the four bonds

**Files:**
- Modify: `src/tenax/algorithms/ipeps_rdm.py:300-317` (`compute_energy_ctm_2site`)
- Modify: `tests/test_rdm_validity_guard.py` (fixture, two tests, docstrings)
- Test: `tests/test_ipeps_2site_energy.py` (existing RED tests `test_ipeps_counts_every_bond_of_the_cell`, `test_compute_energy_ctm_2site_counts_every_bond`)

**Interfaces:**
- Consumes: Task 2's `_ctm_2site_via_2x2` (the guard fixture).
- Produces: `compute_energy_ctm_2site(A, B, env_A, env_B, hamiltonian_gate, d)` → `0.5 * (h AB + h BA + v AB + v BA)`, same signature.

- [ ] **Step 1: Confirm the two dimer tests fail**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_2site_energy.py -k "counts_every_bond" > "$W/t3_red.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=1`, `2 failed`, each with `-0.75 == -0.375 ± 1.0e-10`.

- [ ] **Step 2: Implement**

In `src/tenax/algorithms/ipeps_rdm.py`, replace the whole of `compute_energy_ctm_2site`:

```python
    """Compute energy per site for a 2-site checkerboard iPEPS.

    E/site = E_horizontal + E_vertical (one bond of each type per site).
    """
    H = hamiltonian_gate.reshape(d, d, d, d)
    rdm_h = _rdm2x1_2site(A, B, env_A, env_B, d)
    rdm_v = _rdm1x2_2site(A, B, env_A, env_B, d)
    E_h = jnp.einsum("ijkl,klij->", rdm_h, H)
    E_v = jnp.einsum("ijkl,klij->", rdm_v, H)
    return (E_h + E_v).real
```

with

```python
    """Compute energy per site for a 2-site checkerboard iPEPS.

    The cell has four bonds: ``A.r-B.l`` and ``A.d-B.u`` (the bonds that start
    on ``A``) and ``B.r-A.l`` and ``B.d-A.u`` (the bonds that start on ``B``),
    and two sites, so E/site is half the sum of the four.  The same count as
    :func:`compute_energy_ctm_tensor_2site` (#493).  Until 0.9.3 this function
    summed only the two bonds that start on ``A``: singlets on every
    ``A.r-B.l`` bond then read -0.75 per site instead of -0.375, and the D=2
    simple-update state of the plain Heisenberg gate -0.6393 instead of
    -0.6568.

    ``env_A`` and ``env_B`` are the environments :func:`ctm_2site` returns:
    each field has its legs in the order ``CTMTensorEnv`` documents.
    """
    H = hamiltonian_gate.reshape(d, d, d, d)
    rdms = (
        _rdm2x1_2site(A, B, env_A, env_B, d),
        _rdm2x1_2site(B, A, env_B, env_A, d),
        _rdm1x2_2site(A, B, env_A, env_B, d),
        _rdm1x2_2site(B, A, env_B, env_A, d),
    )
    E = sum(jnp.einsum("ijkl,klij->", rdm, H) for rdm in rdms)
    # 4 bonds (2 that start on A + 2 that start on B) / 2 sites per cell.
    return 0.5 * E.real
```

- [ ] **Step 3: Move the #845 guard tests onto the 2x2 environment and four bonds**

The fixture's legacy environment cannot carry a four-bond sum: at `chi = D**2` its corner spectrum is flat, its corner basis is arbitrary, and its `B`-bonds are not checked anywhere. In `tests/test_rdm_validity_guard.py`:

(a) At the end of the module docstring, before the closing `"""`, add the paragraph:

```
Both measurements above came from the legacy ``ctm_2site`` sweep, whose two
environments did not share a basis at the A|B seams (removed in 0.9.3; see
``tests/test_ipeps_2site_energy.py``).  The guard stays: any other route to a
dead bond reaches the same normaliser.  The fixture now uses the 2x2 recipe,
and ``compute_energy_ctm_2site`` sums all four bonds of the cell.
```

(b) Replace the import `from tenax.algorithms.ipeps_ctm_convergence import ctm_2site` with `from tenax.algorithms.ipeps_ctm_convergence import _ctm_2site_via_2x2`.

(c) Replace the comment block above `CFG` and the `CFG` line

```python
# ``min_iter=1`` pins the environment to the converged early-stop point.  At
# ``chi = D**2 = 4`` the corner spectrum is exactly flat (see the module
# docstring), so the SV-diff criterion reads "converged" immediately while the
# corner *basis* keeps rotating sweep to sweep -- there is no unique fixed
# point.  This fixture's "healthy environment" premise therefore holds only at
# a pinned sweep count: with the default ``min_iter=10`` (capped to ``max_iter``
# by #925/#976) the loop is forced to 5 sweeps and lands on a rotated basis
# whose horizontal RDM is numerically near-invalid, so the guard fires on the
# nominally-healthy env.  Pinning ``min_iter=1`` lets the loop stop as soon as
# the SV criterion is met (~2 sweeps), reproducing the deterministic healthy
# basis these tests were written against.  This is the same gauge-dependent
# degeneracy documented in #780/#781, not a guard defect.
CFG = CTMConfig(chi=4, chi_I=4, max_iter=5, min_iter=1, conv_tol=1e-10)
```

with

```python
# The 2x2 recipe does not converge on these random tensors in 5 sweeps
# (criterion 0.111, measured at d6730510), and these tests do not need it to:
# all four bond RDMs of the fixture are density matrices (trace 1, smallest
# eigenvalue 0.141).
CFG = CTMConfig(chi=4, max_iter=5, conv_tol=1e-10)
```

(d) In the fixture, replace `    env_A, env_B = ctm_2site(A, B, CFG)` with `    env_A, env_B = _ctm_2site_via_2x2(A, B, CFG)`.

(e) In `test_a_dead_bond_warns_and_names_itself`, replace `with two bonds in the sum` with `with four bonds in the sum`.

(f) Replace the whole of `test_a_mutilated_environment_with_valid_rdms_is_silent` with:

```python
def test_a_mutilated_environment_with_valid_rdms_is_silent(env):
    """Zeroing ``C1`` of ``env_B`` leaves the two bonds that start on ``A`` --
    which never contract it -- with trace 1, so: no warning.

    This is the sharp negative control.  The guard keys on the RDM actually
    being a density matrix, not on the environment looking damaged, and this
    distinguishes those two.  A check that fired here would fire on healthy
    physics too.  (The energy sums all four bonds, and the two that start on
    ``B`` do contract this corner, so the control is run on the RDMs.)
    """
    A, B, env_A, env_B, _gate = env
    dead = _kill(env_B, "C1")

    rdms, caught = _rdm_warnings(
        lambda: (
            _rdm2x1_2site(A, B, env_A, dead, DPHYS),
            _rdm1x2_2site(A, B, env_A, dead, DPHYS),
        )
    )

    assert not caught, f"warned although both RDMs were valid: {caught}"
    assert all(jnp.isfinite(r).all() for r in rdms)
```

(g) Replace the whole of `test_the_energy_is_silently_short_by_exactly_the_dead_bond` with:

```python
def test_the_energy_is_silently_short_by_exactly_the_dead_bonds(env):
    """Pins the defect: the total is the surviving bonds alone, not an error.

    Zeroing ``C2`` of ``env_B`` kills the horizontal bond that starts on ``A``
    and the vertical bond that starts on ``B``; the other two survive.  Without
    the guard this number is indistinguishable from a real energy -- which is
    how it survived as a plausible-looking value ~2x off.
    """
    A, B, env_A, env_B, gate = env
    dead = _kill(env_B, "C2")
    H = gate.reshape(DPHYS, DPHYS, DPHYS, DPHYS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_full = float(compute_energy_ctm_2site(A, B, env_A, env_B, gate, DPHYS))
        E_dead = float(compute_energy_ctm_2site(A, B, env_A, dead, gate, DPHYS))
        alive = (
            _rdm2x1_2site(B, A, dead, env_A, DPHYS),
            _rdm1x2_2site(A, B, env_A, dead, DPHYS),
        )
    E_alive = 0.5 * sum(
        float(jnp.real(jnp.einsum("ijkl,klij->", rdm, H))) for rdm in alive
    )

    # The dead contributions are not wrong, they are *absent*.
    assert E_dead == pytest.approx(E_alive, abs=1e-12)
    assert E_dead != pytest.approx(E_full, abs=1e-6)
```

- [ ] **Step 4: Run**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_rdm_validity_guard.py tests/test_ipeps_2site_energy.py > "$W/t3_green.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=1`; every test in `test_rdm_validity_guard.py` passes. In `test_ipeps_2site_energy.py` 3 pass and 5 fail. The two dimer tests pass: the legacy environment is exact on the dimer, −0.375 with four bonds. The swap test passes too: the legacy environments do not depend on the order of the pair (probe L), so the four-bond sum is swap-symmetric. The 3 `ipeps()` gauge tests and the 2 `ctm_2site` gauge tests still fail.

- [ ] **Step 5: Commit**

```bash
cd /home/yjkao/tenax && git add src/tenax/algorithms/ipeps_rdm.py tests/test_rdm_validity_guard.py && \
git commit -m "fix(ipeps): compute_energy_ctm_2site sums all four bonds of the cell

The checkerboard cell has four bonds and two sites.  The raw energy added
the two bonds that start on A and dropped the two that start on B -- the
count #493 fixed in compute_energy_ctm_tensor_2site and not in this copy.
Singlets on every A.r-B.l bond read -0.75 per site instead of -0.375, and
the D=2 simple-update state of the plain Heisenberg gate -0.6393 instead
of -0.6568.

The #845 guard tests move to the 2x2 environment (the legacy one has a flat
corner spectrum at chi = D**2 and an arbitrary basis), and the two tests
whose premise was the two-bond sum are re-derived for four bonds.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 4: `ipeps()` measures with the 2x2 recipe

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py` (import at `:30`, docstring, the CTM call at `:591`, the #839 warning)
- Modify: `tests/test_ctm_convergence_meta.py`, `tests/test_ctm_criterion_rank_blind_898.py`, `tests/test_ipeps_2site_energy.py`, `tests/test_ipeps.py`, `tests/test_su_667_product_state.py`

**Interfaces:**
- Consumes: Task 2's `_ctm_2site_via_2x2`; Task 3's `compute_energy_ctm_2site`.
- Produces: `tenax.algorithms.ipeps` binds `_ctm_2site_via_2x2` at module scope (the bomb test patches it there) and no longer binds `ctm_2site`.

- [ ] **Step 1: Change the tests first**

(a) `tests/test_ipeps_2site_energy.py`, in `test_ipeps_counts_every_bond_of_the_cell`: replace

```python
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onverge", category=UserWarning)
        E = _measure(heisenberg_gate(), _dimer_pair(), CTMConfig(chi=4, max_iter=50))
```

with

```python
    # The corners of this state have rank 1, so the criterion refuses to
    # certify them (#898), and ipeps() must say so.
    with pytest.warns(UserWarning, match=r"CTM did not converge in ipeps\(\)"):
        E = _measure(heisenberg_gate(), _dimer_pair(), CTMConfig(chi=4, max_iter=50))
```

(Leave the `warnings.catch_warnings` in `test_compute_energy_ctm_2site_counts_every_bond` as it is.)

(b) `tests/test_ctm_convergence_meta.py`, replace the whole of `test_ipeps_warns_when_its_ctm_did_not_converge` with:

```python
def test_ipeps_warns_when_its_ctm_did_not_converge():
    """``ipeps()`` has no return slot for status, so it must warn (#747 policy).

    This is the actual user-facing harm in #839: a public entry point handing
    back an energy derived from a non-converged environment, silently.  The
    2x2 recipe converges on this state at sweep 13 (``conv_tol=1e-12``,
    measured at d6730510); after 10 sweeps its criterion is 2.0e-10, so the
    budget runs out.  Exactly one warning, from ``ipeps()``: the loop inside it
    must not tell the caller a second time.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=6, max_iter=10, conv_tol=1e-12),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ipeps(gate, None, config)
    ctm_warnings = [w for w in caught if "did not converge" in str(w.message)]
    assert len(ctm_warnings) == 1, [str(w.message)[:80] for w in ctm_warnings]
    assert issubclass(ctm_warnings[0].category, UserWarning)
    assert "CTM did not converge in ipeps()" in str(ctm_warnings[0].message)
```

(c) Same file, in `test_ipeps_energy_is_unchanged_by_the_warning_path`: replace the docstring's second paragraph

```
    ``ipeps()`` now calls ``ctm_2site(..., return_meta=True)``; that must be a
    pure addition, since every benchmark number in the repo comes through here.
```

with

```
    ``ipeps()`` reads the status from ``_ctm_2site_via_2x2(...,
    return_meta=True)``; that must be a pure addition, since every benchmark
    number in the repo comes through here.
```

and replace

```python
    from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site

    cfg = CTMConfig(chi=6, max_iter=100, conv_tol=1e-10)
    eA, eB = ctm_2site(A, B, cfg)
```

with

```python
    from tenax.algorithms.ipeps_ctm_convergence import _ctm_2site_via_2x2
    from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site

    cfg = CTMConfig(chi=6, max_iter=100, conv_tol=1e-10)
    eA, eB = _ctm_2site_via_2x2(A, B, cfg)
```

(d) Same file, in `test_su_warm_start_never_runs_the_measurement_ctm`, replace

```python
    """#937 review P2: optimize_gs_ad's su_init warm start wants only the
    state, so it must not pay for (or be warned about) ipeps()'s legacy
    measurement CTM.  The bomb below makes any measurement attempt loud;
    reverting every warm-start call site fires it on both unit cells."""
    import tenax.algorithms.ipeps as ipeps_mod
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    def _bomb(*a, **k):
        raise AssertionError(
            "su_init warm start invoked the legacy measurement CTM (#937)"
        )

    monkeypatch.setattr(ipeps_mod, "ctm_2site", _bomb)
```

with

```python
    """#937 review P2: optimize_gs_ad's su_init warm start wants only the
    state, so it must not pay for (or be warned about) ipeps()'s measurement
    CTM.  The bomb below makes any measurement attempt loud; reverting every
    warm-start call site fires it on both unit cells."""
    import tenax.algorithms.ipeps as ipeps_mod
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    def _bomb(*a, **k):
        raise AssertionError("su_init warm start invoked the measurement CTM (#937)")

    monkeypatch.setattr(ipeps_mod, "_ctm_2site_via_2x2", _bomb)
```

(The `raise` fits on one line now, 85 columns, so it is written the way `ruff format` writes it.)

(e) `tests/test_ctm_criterion_rank_blind_898.py`, replace the whole of `test_the_public_loop_is_the_one_ipeps_calls` with:

```python
def test_the_public_loop_is_the_one_ipeps_calls(monkeypatch):
    """Pin the routing the guard tests depend on.

    Since 0.9.3 ``ipeps()`` measures with the multisite loop of
    ``_ctm_tensor_convergence`` (through ``_ctm_2site_via_2x2``), which
    compares corner spectra with the guarded ``_ctm_sv_diff`` defined there.
    If the routing moves, the tests above keep passing and stop meaning
    anything -- so a spy on the guarded function asserts that ``ipeps()``
    reaches it.  The source check keeps ``ipeps_ctm_convergence``, whose
    ``ctm_split`` loop compares spectra too, on the imported copy (#898).
    """
    import inspect

    import tenax.algorithms._ctm_tensor_convergence as tensor_convergence
    from tenax.algorithms import ipeps_ctm_convergence
    from tenax.algorithms.ipeps import heisenberg_gate, ipeps
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    guarded = tensor_convergence._ctm_sv_diff
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return guarded(*args, **kwargs)

    monkeypatch.setattr(tensor_convergence, "_ctm_sv_diff", spy)
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=1,
        dt=0.1,
        ctm=CTMConfig(chi=4, max_iter=3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipeps(heisenberg_gate(), None, config)

    assert calls, (
        "ipeps() no longer compares corner spectra with the guarded "
        "_ctm_sv_diff of _ctm_tensor_convergence, so the guard tests above are "
        "no longer about the public path -- re-point them at whatever ipeps() "
        "reaches now"
    )
    source = inspect.getsource(ipeps_ctm_convergence)
    assert "def _ctm_sv_diff" not in source, (
        "ipeps_ctm_convergence defines _ctm_sv_diff again; it must import the "
        "guarded one from _ctm_tensor_convergence (#898)"
    )
```

(f) `tests/test_ipeps.py`, in `TestIPEPS2Site.test_2site_heisenberg_D2_energy`: replace the docstring's last paragraph

```
        The bound is loose because ``ipeps()``'s own energy comes from the
        legacy 2-site ``ctm_2site``, which does **not** converge on a genuinely
        entangled state -- measured -0.6294 to -0.6366 across chi=10..32 and
        max_iter=40..400, against the -0.6593 a converged 1x1 CTM gives for the
        same state.  Pinning it tightly would pin that non-convergence.  The
        physics assertion lives in ``test_su_667_product_state.py``, which
        measures the state rather than this environment.
```

with

```
        Since 0.9.3 ``ipeps()`` measures with the 2x2 recipe and sums the four
        bonds of the cell: -0.65679 here, the same to 10 digits at chi = 10,
        16, 24 and 32 (measured at d6730510 through the same recipe).  The
        bound sits above that and below every old reading, so it catches
        each defect on its own: the legacy environment with four bonds reads
        -0.6507, the 2x2 environment with the two bonds that start on ``A``
        -0.6393, and the old ``ipeps()`` -0.6303.  The physics assertion
        lives in ``test_su_667_product_state.py``.
```

and replace

```python
        assert float(energy) < -0.60, (
            f"Energy {float(energy)} not low enough — D=2 iPEPS should give "
            f"E < -0.60 even through the unconverged 2-site CTM"
        )
```

with

```python
        assert float(energy) < -0.655, (
            f"Energy {float(energy)} not low enough — the 2x2 measurement of "
            f"this D=2 state gives -0.65679"
        )
```

(g) `tests/test_su_667_product_state.py`: in the module docstring replace

```
**These tests deliberately do not assert on the energy ``ipeps()`` returns.**
That number comes from the legacy 2-site ``ctm_2site``, which does not converge
here (measured ``diff`` 1.6e-3 to 6.5e-2 after 600 sweeps at conv_tol=1e-10),
so freezing it would freeze noise -- the same trap #836 removed elsewhere.  The
state itself is well defined, so the assertions below measure the state with a
```

with

```
**These tests deliberately do not assert on the energy ``ipeps()`` returns.**
Until 0.9.3 that number came from the legacy 2-site ``ctm_2site``, which did not
converge here (measured ``diff`` 1.6e-3 to 6.5e-2 after 600 sweeps at
conv_tol=1e-10), so freezing it would have frozen noise -- the same trap #836
removed elsewhere.  ``ipeps()`` now measures with the 2x2 recipe, but at the
caller's chi and budget.  The state itself is well defined, so the assertions
below measure the state with a
```

and in `_run_su` replace the comment

```python
    # ipeps() may warn that its own (legacy) 2-site CTM did not converge (#839)
    # at the small chi used here.  That is a property of ctm_2site, not of the
    # state under test, and whether it fires depends on D -- so it is ignored
    # rather than asserted either way.
```

with

```python
    # ipeps() may warn that its measurement CTM did not converge (#839): two
    # sweeps rarely certify.  That is a property of this budget, not of the
    # state under test, and whether it fires depends on D -- so it is ignored
    # rather than asserted either way.
```

- [ ] **Step 2: Run the changed tests to see which fail before the source change**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_2site_energy.py tests/test_ctm_convergence_meta.py \
  tests/test_ctm_criterion_rank_blind_898.py "tests/test_ipeps.py::TestIPEPS2Site" > "$W/t4_red.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=1`, with 9 failures:

- `test_ipeps_energy_does_not_depend_on_the_bond_gauge[uniform|gauge1|gauge2]` and `test_ctm_2site_energy_does_not_depend_on_the_bond_gauge[1|2]`: the legacy environment.
- `test_ipeps_counts_every_bond_of_the_cell`: no warning, because the legacy loop certifies the dimer (probe M).
- `test_su_warm_start_never_runs_the_measurement_ctm[1x1|2site]`: `AttributeError`, because `ipeps` has no `_ctm_2site_via_2x2` to patch.
- `test_2site_heisenberg_D2_energy`: the legacy environment with four bonds reads −0.6507 (probe K), above −0.655.
- `test_the_public_loop_is_the_one_ipeps_calls`: the legacy loop calls the copy bound in `ipeps_ctm_convergence`, not the patched one.

These pass already: `test_ipeps_energy_does_not_depend_on_which_sublattice_is_called_A` (Task 3 made the sum swap-symmetric), and `test_ipeps_warns_when_its_ctm_did_not_converge` (the legacy loop does not converge on this state, #838, and has no warning of its own, so `ipeps()` warns once).

- [ ] **Step 3: Implement**

In `src/tenax/algorithms/ipeps.py`, replace `from tenax.algorithms.ipeps_ctm_convergence import ctm_2site` with `from tenax.algorithms.ipeps_ctm_convergence import _ctm_2site_via_2x2`.

In the `ipeps()` docstring, replace

```
    2. CTM environment computation -- initialise and iteratively absorb
       rows/columns until convergence.
    3. Compute energy per site using the CTM environment.
```

with

```
    2. CTM environments of the pair -- the 2x2 recipe of
       :func:`ctm_tensor_2site`, with ``config.ctm``'s ``chi``, ``max_iter``,
       ``conv_tol``, ``renormalize`` and ``min_iter``.
    3. Energy per site -- half the sum of the four bond energies of the cell.

    Until 0.9.3 steps 2-3 used a legacy 2-site CTM whose two environments did
    not share a basis, so the energy moved with the bond gauge of the pair,
    and summed only the two bonds that start on ``A``.
```

and replace

```
    Returns:
        ``(energy_per_site, (A, B), (env_A, env_B))``, or
        ``(None, (A, B), None)`` with ``compute_energy=False``.
```

with

```
    Returns:
        ``(energy_per_site, (A, B), (env_A, env_B))``, or
        ``(None, (A, B), None)`` with ``compute_energy=False``.  ``env_A`` and
        ``env_B`` are raw-array ``CTMEnvironment`` named tuples, each field with
        its legs in the order ``CTMTensorEnv`` documents.
```

Replace

```python
    # CTM environment (uses dense arrays)
    A_dense = A.todense()
    B_dense = B.todense()
    env_A, env_B, ctm_info = ctm_2site(A_dense, B_dense, config.ctm, return_meta=True)
```

with

```python
    # CTM environments (uses dense arrays): the 2x2 recipe of ctm_tensor_2site,
    # as raw CTMEnvironments in the leg order compute_energy_ctm_2site
    # contracts.  The legacy 2-site sweep it replaced in 0.9.3 gave energies
    # that moved with the bond gauge (tests/test_ipeps_2site_energy.py).
    A_dense = A.todense()
    B_dense = B.todense()
    env_A, env_B, ctm_info = _ctm_2site_via_2x2(
        A_dense, B_dense, config.ctm, return_meta=True
    )
```

Replace

```python
            f"value -- it can move with max_iter. Raise max_iter, or call "
            f"ctm_2site(..., return_meta=True) to inspect convergence "
            f"directly.",
```

with

```python
            f"value -- it can move with max_iter. Raise max_iter, or measure "
            f"the returned pair with ctm_tensor_2site(A, B, chi, ...), which "
            f"warns with the loop's own diagnosis (strict=True raises instead).",
```

- [ ] **Step 4: Run**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_2site_energy.py tests/test_ctm_convergence_meta.py \
  tests/test_ctm_criterion_rank_blind_898.py tests/test_ipeps.py tests/test_su_667_product_state.py \
  tests/test_ctm_complex_site_tensors.py tests/test_rdm_validity_guard.py > "$W/t4_green.txt" 2>&1; rc=$?; echo "rc=$rc"; exit $rc
```

Expected: `rc=1` with exactly 2 failures, `test_ctm_2site_energy_does_not_depend_on_the_bond_gauge[1|2]` (`ctm_2site` is still the legacy sweep until Task 5). Everything else passes. `tests/test_ipeps.py` is an `algorithm` file of several minutes: run this step in the background.

- [ ] **Step 5: Commit**

```bash
cd /home/yjkao/tenax && git add src/tenax/algorithms/ipeps.py tests/test_ipeps_2site_energy.py tests/test_ctm_convergence_meta.py \
  tests/test_ctm_criterion_rank_blind_898.py tests/test_ipeps.py tests/test_su_667_product_state.py && \
git commit -m "fix(ipeps): measure the 2-site pair with the 2x2 recipe

ipeps() measured its state with the legacy ctm_2site sweep, which built
each sublattice's new boundary from its own corners and edges and
renormalised it with a projector of its own, so the two environments did
not share a basis at the A|B seams.  On three D=2 simple-update states,
each in two bond gauges at chi=6 and 8, the reported energy ranged from
-0.7815 to +0.8171, several below the QMC ground-state energy -0.6694.
ipeps() now runs _ctm_2site_via_2x2 and agrees with the 1-site CTM of a
uniform state to 4e-11 in every gauge.

The #839 warning points at ctm_tensor_2site; the loop's own warning stays
off, so a starved budget warns once.  Tests: the routing test of #898 spies
on the guarded _ctm_sv_diff that ipeps() now reaches, the #937 bomb patches
the new name, the dimer test asserts the warning, and the D=2 smoke bound
tightens from -0.60 to -0.655 (measured -0.65679; the legacy environment
with four bonds reads -0.6507, the two A bonds alone -0.6393).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 5: `ctm_2site()` delegates and is deprecated; the legacy sweep goes

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ctm_convergence.py` (`ctm_2site`, `__all__`, imports, module comment; delete `_ctm_2site_sweep`, `_renormalize_env`)
- Modify: `src/tenax/algorithms/ipeps_ctm_moves.py` (delete `_ctm_move_eigh`, `_ctm_move`, `_ctm_{left,right,top,bottom}_move_2site`)
- Modify: `src/tenax/algorithms/ipeps_ctm.py` (drop eight re-exports)
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` (`CTMConvergenceInfo` docstring)
- Modify: `pyproject.toml` (`filterwarnings`)
- Modify: `tests/test_ctm_2site_delegates_to_2x2.py`, `tests/test_ctm_convergence_meta.py`, `tests/test_ctm_min_iter_contract_925_976.py`, `tests/test_ctm_complex_site_tensors.py`, `tests/test_ctm_tensor.py`, `tests/test_ctm_criterion_rank_blind_898.py`

**Interfaces:**
- Consumes: Task 2's `_ctm_2site_via_2x2`.
- Produces: `ctm_2site(A, B, config, *, return_meta=False)` = `_ctm_2site_via_2x2` plus the warnings.

- [ ] **Step 1: Write the failing tests**

(a) `tests/test_ctm_2site_delegates_to_2x2.py`: change the import to

```python
from tenax.algorithms.ipeps_ctm_convergence import (
    CTMConvergenceInfo,
    _ctm_2site_via_2x2,
    ctm_2site,
)
```

add `pytest.param(ctm_2site, id="ctm_2site"),` as the second entry of `ENTRY_POINTS`, and append:

```python
def test_ctm_2site_warns_that_it_is_deprecated():
    """Exactly one deprecation warning, naming the replacement and the lost
    ``jax.jit``."""
    with pytest.warns(DeprecationWarning, match="^ctm_2site is deprecated") as record:
        ctm_2site(ONES, ONES, CTMConfig(chi=1, max_iter=5))
    deprecations = [
        w for w in record if str(w.message).startswith("ctm_2site is deprecated")
    ]
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert "ctm_tensor_2site" in message
    assert "jax.jit" in message
    # Python's default filters show a DeprecationWarning only when it is blamed
    # on __main__ (PEP 565).  With stacklevel=1 it would be blamed on
    # ctm_2site()'s own module and hidden from every script.
    assert deprecations[0].filename == __file__


@pytest.mark.parametrize("method", ["qr", "eigh"])
def test_ctm_2site_warns_that_projector_method_is_ignored(method):
    """The 2x2 recipe always uses Fishman SVD: the environments are the ones
    ``"svd"`` gives, and the caller is told."""
    A, B = _random_site(0), _random_site(7)
    with pytest.warns(UserWarning, match="projector_method"):
        envs = ctm_2site(A, B, CTMConfig(chi=4, max_iter=5, projector_method=method))
    _assert_same_envs(envs, _ctm_2site_via_2x2(A, B, CTMConfig(chi=4, max_iter=5)))


def test_ctm_2site_default_config_gives_no_user_warning():
    """The ``projector_method`` warning fires only when it applies."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctm_2site(_random_site(0), _random_site(7), CTMConfig(chi=4, max_iter=5))
    assert not [w for w in caught if "projector_method" in str(w.message)]


def test_ctm_2site_return_meta_matches_the_helper():
    """``ctm_2site()`` is the helper plus the warnings: same environments, same
    info."""
    A = jnp.asarray(ising_peps(0.2))
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env_A, env_B, info = ctm_2site(A, A, cfg, return_meta=True)
    ref_A, ref_B, ref_info = _ctm_2site_via_2x2(A, A, cfg, return_meta=True)
    assert info == ref_info
    _assert_same_envs((env_A, env_B), (ref_A, ref_B))
```

(b) `tests/test_ctm_min_iter_contract_925_976.py` (its helper and multisite cases came in Tasks 1 and 2): in the module docstring, replace

```
pins the warm-up rule on ``ctm_tensor``.
```

with

```
pins the warm-up rule on ``ctm_tensor``.  Since 0.9.3 ``ctm_2site`` runs the
multisite loop of ``ctm_tensor_2site``, which takes the same gate as a
keyword-only ``min_iter`` (default 1): the loop and the helper behind
``ctm_2site`` run the same cases, and
``test_multisite_min_iter_counts_the_warmup_against_the_callers_budget``
pins the warm-up rule on the loop.
```

(c) `tests/test_ctm_convergence_meta.py`:
- Replace the whole `su_state` docstring

```
    """A real D=2 state and its gate.

    Deliberately the *unconverged* tau=2 simple-update state (40 steps at
    dt=0.05): its CTM provably does not converge at chi=6 (#838), which is what
    makes it a usable negative case here.  That holds for ``ctm_2site``; the
    2x2 recipe that ``ctm()`` runs since 0.9.3 converges on it in 16 sweeps at
    chi=6, so ``test_ctm_reports_status`` starves its budget instead.  Do not
    "improve" the state to a longer imaginary time -- a state whose CTM
    converges cannot exercise the ``converged=False`` branch.
    """
```

with

```
    """A real D=2 state and its gate.

    Deliberately the *unconverged* tau=2 simple-update state (40 steps at
    dt=0.05).  The legacy ``ctm_2site`` sweep did not converge on it at chi=6
    (#838), which made it a negative case for free.  The 2x2 recipe that
    ``ctm()`` and ``ctm_2site()`` run since 0.9.3 converges on it at chi=6 --
    ``ctm()`` at sweep 16, ``ctm_2site()`` at sweep 13 (``conv_tol=1e-12``) --
    so the status tests starve the budget instead, with budgets measured on
    this state.  Do not "improve" the state to a longer imaginary time: the
    starved budgets would then have to be measured again.
    """
```

- In `test_ctm_2site_reports_exhausted_max_iter`, replace

```python
    max_iter = 100
    _eA, _eB, info = ctm_2site(
        A, B, CTMConfig(chi=6, max_iter=max_iter, conv_tol=1e-10), return_meta=True
    )
```

with

```python
    # 10 sweeps at conv_tol=1e-12: the 2x2 recipe converges on this state at
    # sweep 13, and after sweep 10 its criterion is 2.0e-10 (measured at
    # d6730510), so the budget runs out.
    max_iter = 10
    _eA, _eB, info = ctm_2site(
        A, B, CTMConfig(chi=6, max_iter=max_iter, conv_tol=1e-12), return_meta=True
    )
```

and replace `    assert float(info.diff) > 1e-10` with `    assert float(info.diff) > 1e-12`.
- In `test_environments_are_bit_identical_with_and_without_meta`, replace the docstring's second paragraph

```
    ``diff`` was added to the ``lax.while_loop`` carry to be reported.  If that
    altered the iteration in any way, every energy in the suite would shift.
```

with

```
    ``diff`` was once added to a ``lax.while_loop`` carry to be reported; since
    0.9.3 the loop is the 2x2 recipe's.  If reading the status altered the
    iteration in any way, every energy in the suite would shift.
```

- Delete `test_info_fields_survive_jit` entirely (from its `def` line through `    assert float(diff) > 0.0` and the two blank lines after it). No producer of `CTMConvergenceInfo` runs under `jax.jit` any more. Delete `import jax` too: its only use was the `@jax.jit` in that test, and ruff does not flag an unused import in `tests/` (F401 is ignored there). `command grep -n "jax" tests/test_ctm_convergence_meta.py` must then print nothing.

(d) `tests/test_ctm_complex_site_tensors.py`: at the end of the module docstring, before the closing `"""`, add:

```

Since 0.9.3 ``ctm`` and ``ctm_2site`` run the 2x2 recipe, a Python loop, so the
carry constraint is gone.  The tests stay: the contract -- a complex state in, a
complex environment out, a real criterion -- does not depend on the loop.
```

and in `test_ipeps_accepts_a_complex_initial_state` replace

```
    ``num_imaginary_steps=0`` goes straight from the supplied state to
    ``ctm_2site``, which is the shortest path from user input to the raise.
```

with

```
    ``num_imaginary_steps=0`` goes straight from the supplied state to the
    measurement CTM, which is the shortest path from user input to the raise.
```

The 2x2 recipe keeps a complex pair complex: every field complex128, max|T.imag| 0.516 and 0.354 on this file's seeds and chi (probe P), against the test's floor of 1e-3.

(e) `tests/test_ctm_tensor.py`: remove the line `    _ctm_2site_sweep,` from the `from tenax.algorithms.ipeps_ctm import (` block, and delete both skipped stubs with their `@pytest.mark.skip(...)` decorators. Neither has a body. `test_2site_dense_matches_legacy` compared against `_ctm_2site_sweep`, which is deleted below. `test_ctm_tensor_dense_matches_ctm` compared against `_ctm_sweep`, which #1183 deleted, and its skip reason says "The legacy path is retained only for 2-site SU", which this task makes false. The first stub to delete:

```python
    @pytest.mark.skip(
        reason="Tensor CTM path diverged from legacy _ctm_sweep in PR #291 "
        "(differentiable Fishman SVD projector + per-absorption normalization). "
        "The legacy path is retained only for 2-site SU and is not expected "
        "to match the tensor path sweep-for-sweep."
    )
    def test_ctm_tensor_dense_matches_ctm(self, small_peps_dense):
        """Obsolete: legacy CTM sweep no longer matches tensor path."""

```

The second:

```python
    @pytest.mark.skip(
        reason="Tensor CTM path diverged from legacy _ctm_2site_sweep in "
        "PR #291 (differentiable Fishman SVD projector + per-absorption "
        "normalization). Each path is exercised in its own tests; cross-path "
        "equivalence after a single sweep is no longer a design goal."
    )
    def test_2site_dense_matches_legacy(self, small_peps_pair_dense):
        """Obsolete: legacy 2-site CTM sweep no longer matches tensor path."""

```

Each block ends with the blank line that separated the stub from the next method; delete that too, so one blank line stays between the neighbours.

(f) `tests/test_ctm_criterion_rank_blind_898.py`, in `test_the_public_ipeps_loop_uses_the_guarded_criterion`'s docstring, replace

```
    The three loops in that module (``ctm``, ``ctm_2site``, ``ctm_split``) all
    call the name bound at module scope, so binding it once is what routes all
    three.  ``ctm_2site`` is the one ``ipeps()`` reaches.
```

with

```
    Since 0.9.3 ``ctm`` and ``ctm_2site`` delegate to the 2x2 recipe, and
    ``ctm_split`` is the one loop left in that module; it calls the name bound
    at module scope.  ``ipeps()`` reaches the multisite loop of
    ``_ctm_tensor_convergence`` instead (pinned by the next test).
```

In the same docstring, put the routing into the past tense; replace

```
    copy the public path ran: ``ipeps.py`` imports ``ctm_2site`` from there and
    calls it, so every ``ipeps()`` call kept comparing ``[1, 0, ..., 0]``
```

with

```
    copy the public path ran: ``ipeps.py`` imported ``ctm_2site`` from there
    and called it, so every ``ipeps()`` call kept comparing ``[1, 0, ..., 0]``
```

and in the test's assertion message replace

```python
        "from _ctm_tensor_convergence -- the public ipeps() path compares "
        "corner spectra with an unguarded copy again (#898), so a rank-1 "
        "corner will certify as converged there no matter what this file's "
        "other tests say"
```

with

```python
        "from _ctm_tensor_convergence -- ctm_split compares corner spectra "
        "with an unguarded copy again (#898), so a rank-1 corner will certify "
        "as converged there no matter what this file's other tests say"
```

- [ ] **Step 2: Run them to see them fail**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_2site_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py \
  tests/test_ctm_convergence_meta.py tests/test_ipeps_2site_energy.py > "$W/t5_red.txt" 2>&1; echo "rc=$?"
```

Expected: `rc=1`, with 19 failures (probe O shows what the legacy `ctm_2site` does with these inputs):

- In the new file, 17:
  - `test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry[ctm_2site]`: the legacy environment.
  - `test_rejects_a_bad_raw_shape[*-*-ctm_2site]` (8): no message names the 5-leg layout. `(4, 4, 4)` runs without an error; the other shapes raise einsum `ValueError`s or a reshape `TypeError`.
  - `test_refuses_a_symmetric_tensor[*-ctm_2site]` (2): a `TypeError` from `jnp.conjugate` that does not name `ctm_tensor_2site`.
  - `test_numpy_and_dense_tensor_inputs_match_the_jax_array[ctm_2site]`: a `DenseTensor` raises `TypeError`.
  - `test_returns_raw_environments_and_python_scalars[ctm_2site]`: the info holds JAX arrays.
  - `test_ctm_2site_warns_that_it_is_deprecated` and `test_ctm_2site_warns_that_projector_method_is_ignored[qr|eigh]`: no warning.
  - `test_ctm_2site_return_meta_matches_the_helper`: other environments.
- In `test_ipeps_2site_energy.py`, 2: `test_ctm_2site_energy_does_not_depend_on_the_bond_gauge[1|2]`.

These pass: `test_ctm_2site_default_config_gives_no_user_warning` (the legacy code warns about nothing here); `test_ctm_2site_reports_exhausted_max_iter` (the legacy sweep does not converge on this state, #838, so its 10 sweeps run out); every `[helper]` case; the min_iter contract file, where only the docstring changed.

- [ ] **Step 3: Implement `ctm_2site()`**

In `src/tenax/algorithms/ipeps_ctm_convergence.py`, replace everything from `def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:` up to (not including) the line `# ---------------------------------------------------------------------------` that precedes `# Split-CTMRG: ket/bra layers kept separate (arXiv:2502.10298)` with:

```python
def ctm_2site(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> (
    tuple[CTMEnvironment, CTMEnvironment]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo]
):
    """Compute the CTM environments of a 2-site checkerboard pair with the 2x2 recipe.

    .. deprecated:: 0.9.3
        Use :func:`ctm_tensor_2site`.  ``ctm_2site()`` now runs its 2x2 recipe
        and returns the two environments as the raw-array
        :class:`CTMEnvironment` named tuples that ``ctm_2site()`` always
        returned.  The sweep it ran before built each sublattice's new boundary
        from that sublattice's own corners and edges, with a projector of its
        own, so the two environments did not share a basis at the A|B seams and
        the energy moved with the bond gauge of the pair: on three D=2
        simple-update states, each in two bond gauges at chi=6 and 8, the
        energy of ``ipeps()``, which measured with it, ranged from -0.7815 to
        +0.8171, while ``ctm_tensor_2site`` agreed with the 1-site CTM of each
        state to 4e-11.  So environments from ``ctm_2site()`` changed in 0.9.3,
        and ``ctm_2site()`` no longer runs under ``jax.jit``, because the 2x2
        recipe is a Python loop.

    Args:
        A:           Site tensor of sublattice A: a raw array with 5 legs
                     ``(u, d, l, r, s)`` and four virtual legs of one
                     dimension, or a ``DenseTensor``.  Any other shape raises
                     ``ValueError``.  A ``SymmetricTensor`` raises
                     ``TypeError``: call :func:`ctm_tensor_2site`, which keeps
                     the blocks.
        B:           Site tensor of sublattice B, the same way.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` apply.  A ``projector_method`` other than
                     ``"svd"`` is ignored with a ``UserWarning``, because the
                     2x2 recipe always uses Fishman SVD; ``qr_warmup_steps`` is
                     ignored too.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or exhausted
                     ``max_iter`` (#839).  Its fields are Python scalars.
                     Keyword-only and off by default so the return arity of
                     this public entry point is unchanged.

    Returns:
        ``(env_A, env_B)``, or ``(env_A, env_B, info)`` when
        ``return_meta=True``.  Each field has its legs in the order
        ``CTMTensorEnv`` documents, which is the order
        :func:`compute_energy_ctm_2site` contracts.  **The environments are not
        necessarily converged** -- that is what ``info`` is for.  One ``info``
        covers both sublattices: the loop converges when every corner does.
    """
    warnings.warn(
        "ctm_2site is deprecated and will be removed in a later release: call "
        "ctm_tensor_2site(A, B, chi, ...) on DenseTensor or SymmetricTensor site "
        "tensors with legs (u, d, l, r, phys). ctm_2site now runs the 2x2 recipe "
        "of ctm_tensor_2site, so its environments changed in 0.9.3 (the old "
        "sweep gave energies that moved with the bond gauge), and it no longer "
        "runs under jax.jit.",
        DeprecationWarning,
        stacklevel=2,
    )
    if config.projector_method != "svd":
        warnings.warn(
            f"ctm_2site() ignores projector_method={config.projector_method!r}: "
            "the 2x2 recipe always uses Fishman SVD projectors.",
            UserWarning,
            stacklevel=2,
        )
    return _ctm_2site_via_2x2(A, B, config, return_meta=return_meta)


```

(This deletes `_renormalize_env`, `_ctm_2site_sweep` and the legacy body of `ctm_2site`.) In `__all__`, delete the lines `    "_renormalize_env",` and `    "_ctm_2site_sweep",`. Delete the import block

```python
from tenax.algorithms.ipeps_ctm_moves import (
    _ctm_bottom_move_2site,
    _ctm_left_move_2site,
    _ctm_right_move_2site,
    _ctm_top_move_2site,
    _split_ctm_move,
)
```

and put back `from tenax.algorithms.ipeps_ctm_moves import _split_ctm_move` in its place. In the `ipeps_ctm_init` import block, delete `    _build_double_layer,` and `    _initialize_ctm_env,` (keep `_initialize_split_ctm_env`).

In the module comment block under `# ``_ctm_sv_diff`` is imported, not defined here.`, replace both paragraphs

```
# This module used to carry its own copy, and the copy is what the **public**
# path ran: ``ipeps()`` imports ``ctm_2site`` from here (``ipeps.py:30``, called
# at ``:462``), so #898's guard in ``_ctm_tensor_convergence`` protected the
# tensor loops while every ``ipeps()`` call kept the unguarded comparison --
# the rank-1 corner still normalised to ``[1, 0, ..., 0]``, still compared
# equal to an unrelated environment, and still reported convergence early.
# A fix that leaves the production path broken is worse than no fix, because
# the closed issue stops anyone looking.
#
# The guarded implementation is written to be shared: it uses ``jnp.where``
# rather than a Python ``if`` precisely so it can run inside the
# ``jax.lax.while_loop`` in this module, and its own docstring says the guard
# belongs in one place so "a future tenth loop inherits the guard instead of
# re-acquiring the bug".  This is that inheritance, and it also picks up the
# #670 zero-padding, which is a no-op on the dense path where the spectrum
# length is fixed by ``chi``.
```

with

```
# This module used to carry its own copy, and the copy is what the **public**
# path ran: until 0.9.3 ``ipeps()`` measured with ``ctm_2site`` from here, so
# #898's guard in ``_ctm_tensor_convergence`` protected the tensor loops while
# every ``ipeps()`` call kept the unguarded comparison -- the rank-1 corner
# still normalised to ``[1, 0, ..., 0]``, still compared equal to an unrelated
# environment, and still reported convergence early.  A fix that leaves the
# production path broken is worse than no fix, because the closed issue stops
# anyone looking.
#
# The guarded implementation is written to be shared: it uses ``jnp.where``
# rather than a Python ``if`` precisely so it could run inside the
# ``jax.lax.while_loop`` that ``ctm`` and ``ctm_2site`` ran before 0.9.3, and
# its own docstring says the guard belongs in one place so "a future tenth
# loop inherits the guard instead of re-acquiring the bug".  ``ctm_split``,
# the one loop left here, is that inheritance, and it also picks up the #670
# zero-padding, which is a no-op on the dense path where the spectrum length
# is fixed by ``chi``.
```

In `ctm_split`, two sentences point at loops that are gone (`ctm` lost its loop in #1183). Replace

```
        **The environment is not necessarily converged** -- that is what
        ``info`` is for.  Unlike :func:`ctm` / :func:`ctm_2site` this is a
        Python loop, so ``info``'s fields are Python scalars.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi
    # See ``ctm`` above: cap ``min_iter`` at the budget (#976).
```

with

```
        **The environment is not necessarily converged** -- that is what
        ``info`` is for.  Its fields are Python scalars.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi
    # Cap ``min_iter`` at the budget so a small ``max_iter`` stays convergeable
    # (#976).
```

Then run `uv run ruff format src/tenax/algorithms/ipeps_ctm_convergence.py`.

- [ ] **Step 4: Delete the legacy moves and the re-exports**

In `src/tenax/algorithms/ipeps_ctm_moves.py`: delete the functions `_ctm_move_eigh`, `_ctm_move`, `_ctm_left_move_2site`, `_ctm_right_move_2site`, `_ctm_top_move_2site` and `_ctm_bottom_move_2site` (everything from `def _ctm_move_eigh(` up to, not including, `def _split_ctm_projector(`), delete their six names from `__all__`, and replace the module docstring with `"""CTM algorithm for iPEPS — the split-CTM projector and directional moves."""`.

In `src/tenax/algorithms/ipeps_ctm.py`, delete these import blocks: `_ctm_2site_sweep`, `_renormalize_env` (from `ipeps_ctm_convergence`) and `_ctm_bottom_move_2site`, `_ctm_left_move_2site`, `_ctm_move`, `_ctm_move_eigh`, `_ctm_right_move_2site`, `_ctm_top_move_2site` (from `ipeps_ctm_moves`). Each block is three lines, `from ... import (`, the name, `)`.

Then:

```bash
cd /home/yjkao/tenax && uv run ruff check src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/ipeps_ctm_moves.py src/tenax/algorithms/ipeps_ctm.py
```

Delete exactly the imports it reports as unused (F401) in `ipeps_ctm_moves.py` and `ipeps_ctm_convergence.py`, and let `uv run ruff check --fix` sort an import block if it reports I001. Rerun until clean. Then:

```bash
cd /home/yjkao/tenax && command grep -rn "_ctm_2site_sweep\|_renormalize_env\|_ctm_move_eigh\|_ctm_move(\|_move_2site" src/ tests/ examples/ benchmarks/
```

Expected: no output.

- [ ] **Step 5: The filter and the info docstring**

In `pyproject.toml`, after the `"ignore:ctm is deprecated:DeprecationWarning",` entry that #1183 added, add:

```toml
    # ``ctm_2site()`` is deprecated in favour of ``ctm_tensor_2site`` (0.9.3):
    # it now runs the 2x2 recipe.  Its contract tests still call it on purpose;
    # test_ctm_2site_warns_that_it_is_deprecated
    # (tests/test_ctm_2site_delegates_to_2x2.py) un-suppresses it via
    # ``pytest.warns``.
    "ignore:ctm_2site is deprecated:DeprecationWarning",
```

In `src/tenax/algorithms/_ctm_tensor_convergence.py`, in the `CTMConvergenceInfo` docstring, replace

```
    ``converged`` and ``n_iter`` come straight out of a ``lax.while_loop``
    carry for :func:`ctm_2site`, so they are **JAX arrays**, not Python
    scalars.  That keeps it jittable; call ``bool(...)`` / ``int(...)`` at the
    point of use.  :func:`ctm_split` and :func:`ctm_tensor` run a Python loop
    and return Python scalars, and so does :func:`ctm`, which runs
    :func:`ctm_tensor` since 0.9.3.
```

with

```
    Every producer runs a Python loop and fills the fields with Python
    scalars: :func:`ctm_tensor`, :func:`ctm_split`, and, since 0.9.3,
    :func:`ctm` and :func:`ctm_2site`, which run the 2x2 recipe.  Before 0.9.3
    those two filled them with JAX arrays out of a ``lax.while_loop`` carry, so
    code that calls ``bool(...)`` / ``int(...)`` at the point of use works with
    both.
```

- [ ] **Step 6: Run**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_2site_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py \
  tests/test_ctm_convergence_meta.py tests/test_ipeps_2site_energy.py tests/test_ctm_complex_site_tensors.py \
  tests/test_ctm_criterion_rank_blind_898.py tests/test_rdm_validity_guard.py tests/test_ctm_tensor.py \
  tests/test_ctm_delegates_to_2x2.py tests/test_bucket_registry.py > "$W/t5_green.txt" 2>&1; rc=$?; echo "rc=$rc"; exit $rc
```

Expected: `rc=0`; `test_ipeps_2site_energy.py` gives `8 passed`. Background run (`test_ctm_tensor.py` is an `algorithm` file).

- [ ] **Step 7: Commit**

```bash
cd /home/yjkao/tenax && git add pyproject.toml src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/ipeps_ctm_moves.py \
  src/tenax/algorithms/ipeps_ctm.py src/tenax/algorithms/_ctm_tensor_convergence.py \
  tests/test_ctm_2site_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py tests/test_ctm_convergence_meta.py \
  tests/test_ctm_complex_site_tensors.py tests/test_ctm_tensor.py tests/test_ctm_criterion_rank_blind_898.py && \
git commit -m "refactor(ctm): ctm_2site() runs the 2x2 recipe and is deprecated

ctm_2site() keeps its signature and return types and now returns what
_ctm_2site_via_2x2 returns, behind a DeprecationWarning that names
ctm_tensor_2site, and a UserWarning when CTMConfig.projector_method is not
\"svd\" (the 2x2 recipe always uses Fishman SVD).  Its info fields are
Python scalars and it no longer runs under jax.jit.

The legacy sweep goes with it: _ctm_2site_sweep, _renormalize_env,
_ctm_move, _ctm_move_eigh and the four 2-site moves, and their re-exports
in ipeps_ctm.py.  test_info_fields_survive_jit is removed: no producer of
CTMConvergenceInfo runs under jit any more.  So are the two skipped,
bodiless stubs in test_ctm_tensor.py that compared against the legacy
sweeps (_ctm_sweep went in #1183).  The starved budget of
test_ctm_2site_reports_exhausted_max_iter drops from 100 sweeps at 1e-10
to 10 at 1e-12, because the 2x2 recipe converges on that state at sweep 13.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 6: Docs

**Files:**
- Modify: `docs/guide/algorithms/ipeps.md`, `docs/guide/algorithms/ctm.md`, `README.md`, `.claude/skills/tenax-ipeps-workflow/SKILL.md`, `.claude/skills/tenax-observables/SKILL.md`, `tests/test_ipeps_2site_energy.py` (docstring only)

- [ ] **Step 1: Run the examples as they are written**

The script runs every Python block that calls `ipeps(` and prints `Energy per site` in the guide, the README and the skill, unchanged, and two variants for the `dt` claims in the comments. That excludes the README's DMRG block and the guide's `fpeps(` block, which print the same words. The skill's Stage 2 block uses the `gate` of its Stage 1 block, so it runs after `gate = heisenberg_gate()`.

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && cat > "$W/examples.py" <<'PYEOF'
"""Run the doc examples that print ipeps()'s energy, as written."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yjkao/tenax")
SKILL = ".claude/skills/tenax-ipeps-workflow/SKILL.md"
PRELUDE = {SKILL: "from tenax import heisenberg_gate\ngate = heisenberg_gate()\n"}
# The two dt claims in the example comments, run with the dt they name.
VARIANTS = {
    ("docs/guide/algorithms/ipeps.md", 0): ("dt=0.05,", "dt=0.01,"),
    (SKILL, 0): ("dt=0.05,", "dt=0.3,"),
}
RECORD = (
    "import warnings\nwarnings.simplefilter('always')\n"
    "_rec = warnings.catch_warnings(record=True)\n_caught = _rec.__enter__()\n"
)
REPORT = "\nprint('warnings:', [f'{w.category.__name__}: {str(w.message)[:60]}' for w in _caught])\n"
env = dict(os.environ, JAX_PLATFORMS="cpu")
runs = failed = 0
for rel in ("docs/guide/algorithms/ipeps.md", "README.md", SKILL):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / rel).read_text(), re.S)
    ours = [b for b in blocks if "Energy per site" in b and re.search(r"\bipeps\(", b)]
    for i, block in enumerate(ours):
        cases = [("as written", block)]
        if (rel, i) in VARIANTS:
            old, new = VARIANTS[(rel, i)]
            assert block.count(old) == 1, (rel, i, old)
            cases.append((new.rstrip(","), block.replace(old, new)))
        for label, code in cases:
            src = RECORD + PRELUDE.get(rel, "") + code + REPORT
            res = subprocess.run([sys.executable, "-c", src], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
            runs += 1
            failed += res.returncode != 0
            print(f"{rel} block {i} ({label}): rc={res.returncode}")
            print("   ", res.stdout.strip().replace("\n", "\n    "))
            if res.returncode:
                print("   ", res.stderr.strip()[-800:])
print(f"{runs} runs, {failed} failed")
sys.exit(1 if failed else 0)
PYEOF
taskset -c 0-11 uv run python "$W/examples.py" > "$W/t6_examples.txt" 2>&1; rc=$?; echo "rc=$rc"; exit $rc
```

Six simple-update runs take several minutes: run this in the background. Expected: `rc=0` and `6 runs, 0 failed`. The four blocks as written each print `Energy per site: -0.656790` and `warnings: []` (probe N; with `min_iter=10` they stop at sweep 10, which changes the 10th digit at most). The `dt=0.01` variant of the guide's first block prints a positive energy, about `0.486214`, so the comment "E > 0" stays. The `dt=0.3` variant of the skill block prints about `-0.645216` (probe Q), which Step 5 writes into the skill's comment. If a printed number differs, use the printed value everywhere below and say so in the PR.

- [ ] **Step 2: `docs/guide/algorithms/ipeps.md`**

(a) Replace `print(f"Energy per site: {energy:.6f}")  # ~ -0.66` (the "Example -- 2D Heisenberg model" block) with `print(f"Energy per site: {energy:.6f}")  # -0.656790`, and `print(f"Energy per site: {energy:.6f}")  # ~ -0.63` (the "Simple update on the checkerboard" block) with `print(f"Energy per site: {energy:.6f}")  # -0.656790`. Keep the first block's comment `# dt=0.01 has not reached the ground state by 200 steps (E > 0)`: Step 1 checks it.

(b) Replace the paragraph

```
The energy `ipeps()` reports comes from the legacy 2-site CTM, which does not
converge on a genuinely entangled state — it sits ~0.02 above the truth. For an
accurate number, measure the returned state with `ctm_tensor(recipe="2x2")`
(D=2 gives −0.65933, χ-converged).
```

with

```
`ipeps()` measures the returned pair with the 2x2 recipe of `ctm_tensor_2site`
at `config.ctm.chi`, and reports half the sum of the four bond energies of the
cell. The example above prints −0.656790, and the energy is −0.6567903814 at
χ = 10, 16, 24 and 32 (`conv_tol=1e-10`). Before 0.9.3 the same example printed
−0.630276, with the warning "CTM did not converge in ipeps()": `ipeps()`
measured with a legacy 2-site CTM whose two environments did not share a basis,
so its energy moved with the bond gauge of the pair, and it summed only the two
bonds that start on `A`. The −0.65933 that this guide gave as the accurate
number came from `ctm_tensor` on `A` alone, which measures the uniform state of
`A`, a different state (#900).
```

(c) Replace the whole section from the heading line `` ### `ctm_2site()` -- standalone 2-site CTM `` up to, not including, the heading line `` ### `compute_energy_ctm_2site()` -- 2-site energy ``, with:

````
### `ctm_tensor_2site()` -- standalone 2-site CTM

`ctm_tensor_2site(A, B, chi, ...)` computes the environments of a checkerboard
pair `(A, B)` of `DenseTensor` or `SymmetricTensor` site tensors, such as the
pair `ipeps()` returns:

```python
from tenax import compute_energy_ctm_tensor_2site, ctm_tensor_2site

env_A, env_B = ctm_tensor_2site(A, B, chi=20, max_iter=100)
E = compute_energy_ctm_tensor_2site(A, B, env_A, env_B, H_bond, d=2)
```

It warns when the environments are not converged; `strict=True` raises
`CTMNotConvergedError` instead (see the CTM guide).

```{warning}
`ctm_2site()` is deprecated since 0.9.3 and emits a `DeprecationWarning`. It now
runs the 2x2 recipe of `ctm_tensor_2site` and returns the raw-array
`CTMEnvironment` pair it always returned, so its environments changed in 0.9.3.
The sweep it ran before built each sublattice's new boundary from that
sublattice's own corners and edges, with a projector of its own, so the two
environments did not share a basis at the A|B seams: on three D=2 simple-update
states, each in two bond gauges at χ = 6 and 8, the energy of `ipeps()` ranged
from −0.7815 to +0.8171, while `ctm_tensor_2site` agreed with the 1-site CTM of
each state to 4e-11. Three behaviours differ from before:

- `ctm_2site()` does not run under `jax.jit`, because the 2x2 recipe is a Python loop.
- A `CTMConfig.projector_method` other than `"svd"` is ignored with a
  `UserWarning`, because the 2x2 recipe always uses Fishman SVD.
- With `return_meta=True`, the fields of `info` are Python scalars, not JAX arrays.
```

````

The section drops the old argument table: the deprecated function's docstring documents its arguments, and the `ctm()` section that #1183 wrote has no table either.

(d) In the `compute_energy_ctm_2site()` section, replace

```
The energy includes one horizontal and one vertical bond per site:
$E/\text{site} = E_h + E_v$.
```

with

```
The cell has four bonds, `A.r–B.l`, `B.r–A.l`, `A.d–B.u` and `B.d–A.u`, and two
sites, so $E/\text{site} = \tfrac{1}{2}\,(E_h^{AB} + E_h^{BA} + E_v^{AB} + E_v^{BA})$.
Before 0.9.3 this function summed only the two bonds that start on `A` (#493
fixed the same count in `compute_energy_ctm_tensor_2site`): singlets on every
`A.r–B.l` bond then read −0.75 per site instead of −0.375. `A` and `B` are raw
arrays (`A.todense()` for the tensors `ipeps()` returns); `env_A` and `env_B`
are raw-array `CTMEnvironment`s, as `ipeps()` returns them.
```

- [ ] **Step 3: `docs/guide/algorithms/ctm.md`**

Replace

````
`ctm`, `ctm_2site`, `ctm_split` and `ctm_tensor` return an environment whether
or not the sweep met `conv_tol` — running out of `max_iter` is not an error.
Pass `return_meta=True` for a `CTMConvergenceInfo` saying which happened, rather
than inferring it from an energy that silently moves with `max_iter` (#839):

```python
from tenax import CTMConfig, ctm_2site

env_A, env_B, info = ctm_2site(A, B, CTMConfig(chi=16), return_meta=True)
if not bool(info.converged):
    print(f"stopped at max_iter after {int(info.n_iter)} sweeps, "
          f"criterion still {float(info.diff):.2e}")
```

`info.diff` is the convergence criterion — the change in the corner singular
values, not in the energy. `ipeps()` performs this check itself and warns.

`ctm_tensor` takes the same flag, and returns the info as a *third* element
after `(env, max_truncation_error)`:
````

with

````
`ctm_tensor`, `ctm_split` and the deprecated `ctm` and `ctm_2site` return an
environment whether or not the sweep met `conv_tol` — running out of `max_iter`
is not an error. Pass `return_meta=True` for a `CTMConvergenceInfo` saying which
happened, rather than inferring it from an energy that silently moves with
`max_iter` (#839). `info.diff` is the convergence criterion — the change in the
corner singular values, not in the energy. `ipeps()` performs this check itself
and warns.

`ctm_tensor` returns the info as a *third* element after
`(env, max_truncation_error)`:
````

and in the `min_iter` paragraph replace

```
`CTMConfig.min_iter` (default 10), and `ctm()` passes that value on.
```

with

```
`CTMConfig.min_iter` (default 10), and `ctm()` and `ctm_2site()` pass that
value on.
```

- [ ] **Step 4: `README.md`**

Replace `print(f"Energy per site: {energy:.6f}")  # ~ -0.63` with `print(f"Energy per site: {energy:.6f}")  # -0.656790`, and replace

```
The energy `ipeps()` reports is a quick estimate; for an accurate number, and
for the bond-weight options, see the [iPEPS guide](docs/guide/algorithms/ipeps.md).
```

with

```
`ipeps()` measures the state with the 2x2 CTM at the `chi` you pass; for chi
convergence and the bond-weight options, see the [iPEPS guide](docs/guide/algorithms/ipeps.md).
```

- [ ] **Step 5: The two skills**

`.claude/skills/tenax-ipeps-workflow/SKILL.md`: in the Stage 2 example, replace

```
    dt=0.05,                     # Imaginary time step δτ (0.3 gives E ≈ -0.53)
```

with (the value is Step 1's `dt=0.3` run; probe Q measured −0.6452164768)

```
    dt=0.05,                     # Imaginary time step δτ (0.3 gives E ≈ -0.645)
```

and replace

```
print(f"Energy per site: {energy:.6f}")  # ≈ -0.63 for Heisenberg
```

with

```
print(f"Energy per site: {energy:.6f}")  # -0.656790 for Heisenberg
```

Replace the whole passage after that example, from the paragraph that names the legacy CTM to the warm-start sentence:

````
The energy `ipeps()` reports comes from the legacy 2-site CTM, which sits ~0.02
above the truth on an entangled state. For an accurate number, measure the
returned pair with the Tensor CTM (`ctm_tensor_2site`, default `recipe="2x2"`):

```python
from tenax import ctm_tensor_2site, compute_energy_ctm_tensor_2site

env_A, env_B = ctm_tensor_2site(A, B, chi=16)
E = compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2)
```

The config above gives −0.6568 this way (vs −0.6328 from `ipeps()` itself); a
converged D=2 simple-update state reaches −0.65933, χ-converged. Use
`ipeps(..., compute_energy=False)` to get just `(None, (A, B), None)` as a warm
start.
````

with

````
`ipeps()` measures the returned pair with the 2x2 recipe of `ctm_tensor_2site`
at `config.ctm.chi`, and sums the four bonds of the cell (since 0.9.3; before
it, a legacy 2-site CTM printed −0.630276 here). To measure at another chi, or
a symmetric pair, call the Tensor CTM yourself:

```python
from tenax import ctm_tensor_2site, compute_energy_ctm_tensor_2site

env_A, env_B = ctm_tensor_2site(A, B, chi=16)
E = compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2)
```

The config above gives −0.656790 both ways. `ctm_tensor` on `A` alone measures
the uniform state of `A`, a different state (#900). Use
`ipeps(..., compute_energy=False)` to get just `(None, (A, B), None)` as a warm
start.
````

Then replace

```
Note: `ctm_2site()` is the legacy dense CTM used by simple update only;
for AD optimization with 2-site cells, use `optimize_gs_ad()` with
`unit_cell="2site"` which uses the Tensor-protocol multisite CTM.
```

with

```
Note: `ctm_2site()` is deprecated since 0.9.3 and runs the 2x2 recipe of
`ctm_tensor_2site`; call `ctm_tensor_2site()` for 2-site cells, and
`optimize_gs_ad()` with `unit_cell="2site"` for AD optimization.
```

`.claude/skills/tenax-observables/SKILL.md`: replace `# ipeps()'s own energy uses the legacy CTM, which reads ~0.02 high.` with `# ipeps() measures the same way at config.ctm.chi (since 0.9.3).`, and replace

```
The legacy `compute_energy_ctm_2site(A, B, env_A, env_B, gate, d)` takes raw
arrays and the `CTMEnvironment`s from `ctm_2site`.
```

with

```
`compute_energy_ctm_2site(A, B, env_A, env_B, gate, d)` takes raw arrays and
the `CTMEnvironment`s from the deprecated `ctm_2site`, and sums the same four
bonds.
```

- [ ] **Step 6: The RED test file's wording**

In the module docstring of `tests/test_ipeps_2site_energy.py`, replace

```
to ``+0.8171``, several below the exact ground state ``-0.6694``, while
```

with

```
to ``+0.8171``, several below the ground-state energy ``-0.6694`` (QMC), while
```

and replace

```
``-0.6743``; that difference, not a dropped bond, is most of the "~0.02 above
the truth" that the guide reports.
```

with

```
``-0.6743``; that difference, not a dropped bond, is most of the "~0.02 above
the truth" that the guide reported before 0.9.3.
```

- [ ] **Step 7: Build the docs**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
uv run sphinx-build -W -b html docs docs/_build/html > "$W/t6_docs.txt" 2>&1; rc=$?; echo "rc=$rc"; exit $rc
```

Expected: `rc=0` (the CI command, `ci.yml:694`). If `sphinx` is missing from the venv, run `uv sync --extra docs` first and say so in the report (it changes the venv, not the tree).

- [ ] **Step 8: Commit**

```bash
cd /home/yjkao/tenax && git add docs/guide/algorithms/ipeps.md docs/guide/algorithms/ctm.md README.md \
  .claude/skills/tenax-ipeps-workflow/SKILL.md .claude/skills/tenax-observables/SKILL.md tests/test_ipeps_2site_energy.py && \
git commit -m "docs(ipeps): the energy ipeps() reports is the 2x2 measurement of the pair

The guide and the skills said that ipeps()'s energy comes from a legacy
2-site CTM that sits ~0.02 above the truth, and the guide sent the reader
to ctm_tensor on A alone, which measures the uniform state of A instead.
The examples now print -0.656790 (the same to 10 digits at chi = 10, 16,
24, 32), and the skill's dt=0.3 comment gives -0.645 for -0.53.  The
ctm_2site section becomes the ctm_tensor_2site example plus the
deprecation note, and compute_energy_ctm_2site documents the four bonds.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ"
```

---

### Task 7: Verification, mutation count and the PR text

**Files:**
- Create (not committed): `$W/mutate.py`, `$W/mutation.txt`, `$W/pr_body.md`

- [ ] **Step 1: Lint**

```bash
cd /home/yjkao/tenax && uv run ruff check src tests && uv run ruff format --check src tests
```

Expected: `All checks passed!` and no file to reformat.

- [ ] **Step 2: The core gate (background, serial)**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q -m core > "$W/t7_core.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t7_core.txt"; exit $rc
```

Expected: `rc=0`. Copy the summary line into the PR body. Compare its warnings summary with the last green `Tests (Python 3.11)` log on `main` (`gh run list --branch main --workflow ci.yml --limit 1`, then `gh run view <id> --log | command grep -n "warnings summary" -A 60`): a new warning that names a file this branch touches is a finding.

- [ ] **Step 3: The touched `algorithm` files (background)**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
FILES=$(command grep -l "ipeps_ctm\b\|ipeps_ctm_moves\|ipeps_ctm_convergence\|ipeps_rdm\|from tenax.algorithms.ipeps import\|ctm_2site" tests/test_*.py | sort) && echo "$FILES" > "$W/t7_files.txt" && \
JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q -m "not core and not slow" $FILES > "$W/t7_algo.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t7_algo.txt"; exit $rc
```

Expected: `rc=0`; no `failed`, no `error`. `rc=5` ("no tests ran") is a failure: it means the markers no longer select the files. This can take an hour; check `uptime` before and leave it alone while it runs.

- [ ] **Step 4: Write the mutation script and check its anchors**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && cat > "$W/mutate.py" <<'PYEOF'
"""Mutation count for the 2-site delegation.

Every mutant edits a pristine copy of src/tenax under OUT/src, never the
working tree, and runs the new or changed test files against it.

    uv run python OUT/mutate.py OUT          # run every mutant
    uv run python OUT/mutate.py OUT --dry    # only check that each anchor is unique
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yjkao/tenax")
OUT = Path(sys.argv[1])
DRY = "--dry" in sys.argv[2:]
SRC = OUT / "src"
TESTS = [
    "tests/test_ctm_2site_delegates_to_2x2.py",
    "tests/test_ctm_min_iter_contract_925_976.py",
    "tests/test_ipeps_2site_energy.py",
    "tests/test_ctm_convergence_meta.py",
    "tests/test_rdm_validity_guard.py",
]
TC = "tenax/algorithms/_ctm_tensor_convergence.py"
CC = "tenax/algorithms/ipeps_ctm_convergence.py"
RD = "tenax/algorithms/ipeps_rdm.py"
IP = "tenax/algorithms/ipeps.py"
TAIL = '        recipe="2x2",\n        min_iter=config.min_iter,\n        _return_status=True,\n'
# (name, file, [(old, new), ...], the test expected to kill it)
MUTANTS = [
    ("M1 no min_iter gate", TC,
     [("        if converged and budget - max_iter + used < min_sweeps:\n"
       "            converged = False\n", "")],
     "test_convergence_waits_for_min_iter[10-ctm_tensor_multisite]"),
    ("M2 cap taken after the warm-up", TC,
     [("    min_sweeps = min(min_iter, budget)\n",
       "    min_sweeps = min(min_iter, budget - (projector_method == \"qr\")"
       " * min(qr_warmup_steps, budget))\n")],
     "test_multisite_min_iter_counts_the_warmup_against_the_callers_budget"),
    ("M3 min_iter default 10", TC,
     [("    min_iter: int = 1,\n    _return_status: bool = False,\n",
       "    min_iter: int = 10,\n    _return_status: bool = False,\n")],
     "test_multisite_default_min_iter_keeps_the_early_exit"),
    ("M4 -1 on a converged run", TC,
     [("            iterations=budget - max_iter + used,\n",
       "            iterations=(budget - max_iter + used) if not converged else -1,\n")],
     "test_convergence_waits_for_min_iter[2-ctm_tensor_multisite]"),
    ("M5 projector_method forwarded", CC,
     [('        projector_method="svd",\n' + TAIL,
       "        projector_method=config.projector_method,\n" + TAIL)],
     "test_helper_runs_no_qr_warmup"),
    ("M6 min_iter dropped", CC,
     [("        min_iter=config.min_iter,\n        _return_status=True,\n",
       "        _return_status=True,\n")],
     "test_convergence_waits_for_min_iter[10-ctm_2site]"),
    ("M7 renormalize dropped", CC,
     [('        renormalize=config.renormalize,\n        projector_method="svd",\n' + TAIL,
       '        projector_method="svd",\n' + TAIL)],
     "test_helper_passes_renormalize[False]"),
    ("M8 1x1 recipe", CC,
     [(TAIL, TAIL.replace('"2x2"', '"1x1"'))],
     "test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry"),
    ("M9 the loop warns too", CC,
     [("        _warn_unconverged=False,\n", "        _warn_unconverged=True,\n")],
     "test_ipeps_warns_when_its_ctm_did_not_converge"),
    ("M10 corners as stored", CC,
     [("    env_A = _as_dense_env(_in_documented_leg_order(envs[(0, 0)]))\n"
       "    env_B = _as_dense_env(_in_documented_leg_order(envs[(1, 0)]))\n",
       "    env_A = _as_dense_env(envs[(0, 0)])\n"
       "    env_B = _as_dense_env(envs[(1, 0)])\n")],
     "test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry"),
    ("M11 environments swapped", CC,
     [("    env_A = _as_dense_env(_in_documented_leg_order(envs[(0, 0)]))\n"
       "    env_B = _as_dense_env(_in_documented_leg_order(envs[(1, 0)]))\n",
       "    env_A = _as_dense_env(_in_documented_leg_order(envs[(1, 0)]))\n"
       "    env_B = _as_dense_env(_in_documented_leg_order(envs[(0, 0)]))\n")],
     "test_raw_energy_matches_the_tensor_energy_without_lattice_symmetry"),
    ("M12 no shape check", CC,
     [("            if X.ndim != 5 or len(set(X.shape[:4])) != 1:\n",
       "            if False:\n")],
     "test_rejects_a_bad_raw_shape"),
    ("M13 SymmetricTensor densified", CC,
     [("        if isinstance(X, SymmetricTensor):\n            raise TypeError(\n",
       "        if isinstance(X, SymmetricTensor):\n            X = X.todense()\n"
       "        if False:\n            raise TypeError(\n")],
     "test_refuses_a_symmetric_tensor"),
    ("M14 no DeprecationWarning", CC,
     [('    warnings.warn(\n        "ctm_2site is deprecated',
       '    0 and warnings.warn(\n        "ctm_2site is deprecated')],
     "test_ctm_2site_warns_that_it_is_deprecated"),
    ("M15 UserWarning always", CC,
     [('    if config.projector_method != "svd":\n'
       '        warnings.warn(\n            f"ctm_2site() ignores',
       '    if True:\n'
       '        warnings.warn(\n            f"ctm_2site() ignores')],
     "test_ctm_2site_default_config_gives_no_user_warning"),
    ("M16 the two A bonds only", RD,
     [("        _rdm2x1_2site(B, A, env_B, env_A, d),\n", ""),
      ("        _rdm1x2_2site(B, A, env_B, env_A, d),\n", ""),
      ("    return 0.5 * E.real\n", "    return E.real\n")],
     "test_ipeps_counts_every_bond_of_the_cell"),
    ("M17 four bonds, not halved", RD,
     [("    return 0.5 * E.real\n", "    return E.real\n")],
     "test_ipeps_counts_every_bond_of_the_cell"),
    ("M18 ipeps() swaps the environments", IP,
     [("compute_energy_ctm_2site(A_dense, B_dense, env_A, env_B, gate_dense, d)",
       "compute_energy_ctm_2site(A_dense, B_dense, env_B, env_A, gate_dense, d)")],
     "test_ipeps_energy_does_not_depend_on_the_bond_gauge"),
    ("M19 compute_energy=False still measures", IP,
     [("    if not compute_energy:\n        return None, (A, B), None\n", "")],
     "test_su_warm_start_never_runs_the_measurement_ctm"),
]


def run_tests(env):
    cmd = ["taskset", "-c", "0-11", sys.executable, "-m", "pytest", "-x", "-q",
           "--no-cov", "-p", "no:cacheprovider", *TESTS]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def main():
    if SRC.exists():
        shutil.rmtree(SRC)
    shutil.copytree(ROOT / "src" / "tenax", SRC / "tenax",
                    ignore=shutil.ignore_patterns("__pycache__"))
    pristine = {rel: (SRC / rel).read_text() for _, rel, _, _ in MUTANTS}
    for name, rel, pairs, _ in MUTANTS:
        text = pristine[rel]
        for old, new in pairs:
            n = text.count(old)
            assert n == 1, f"{name}: {n} matches in {rel} for {old!r}"
            text = text.replace(old, new)
    print(f"{len(MUTANTS)} mutants, every anchor unique", flush=True)
    if DRY:
        return 0
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1",
               JAX_PLATFORMS="cpu")
    where = subprocess.run([sys.executable, "-c", "import tenax; print(tenax.__file__)"],
                           cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip()
    assert where.startswith(str(SRC)), f"tenax imports from {where}, not the copy"
    base = run_tests(env)
    assert base.returncode == 0, "the pristine copy fails:\n" + base.stdout[-3000:]
    print("pristine copy: " + base.stdout.strip().splitlines()[-1], flush=True)
    killed = 0
    for name, rel, pairs, killer in MUTANTS:
        text = pristine[rel]
        for old, new in pairs:
            text = text.replace(old, new)
        (SRC / rel).write_text(text)
        for cache in list(SRC.rglob("__pycache__")):
            shutil.rmtree(cache)
        try:
            res = run_tests(env)
        finally:
            (SRC / rel).write_text(pristine[rel])
        (OUT / f"mutant_{name.split()[0]}.txt").write_text(res.stdout + res.stderr)
        lines = res.stdout.strip().splitlines()
        failed = [ln for ln in lines if ln.startswith("FAILED")]
        status = {0: "SURVIVED", 1: "killed"}.get(res.returncode, f"ERROR rc={res.returncode}")
        killed += res.returncode == 1
        print(f"{name}: {status}; expected killer {killer}; "
              f"{failed[0] if failed else (lines[-1] if lines else '')}", flush=True)
    print(f"{killed}/{len(MUTANTS)} killed", flush=True)
    return 0 if killed == len(MUTANTS) else 1


sys.exit(main())
PYEOF
uv run python "$W/mutate.py" "$W/mut" --dry
```

Expected: `19 mutants, every anchor unique`. An anchor that is not unique is a defect of this script, not of the code: make it longer, with a neighbouring line, and rerun `--dry`.

- [ ] **Step 5: Run the mutants (background; about 20 times the pristine run, which the script prints first)**

```bash
cd /home/yjkao/tenax && W=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/fu2 && uptime && \
uv run python "$W/mutate.py" "$W/mut" > "$W/mutation.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/mutation.txt"; exit $rc
```

Do not run any other probe or test against `src/` while it runs. Expected: `pristine copy: ... passed`, one `killed` line per mutant whose first `FAILED` line names the expected killer (or another test that asserts the same property), `19/19 killed`, `rc=0`. A `SURVIVED` mutant is a finding: decide whether it is equivalent (record why) or a gap (add a test, watch it fail on the mutant and pass on the code, then rerun). An `ERROR rc=...` line means the mutant broke collection or the copy; read `$W/mut/mutant_<id>.txt` before counting it.

- [ ] **Step 6: Draft the PR body**

Write `$W/pr_body.md` with these sections, taking every number from a file read in this task or from the plan's "Measured numbers":

1. The marker line `> 🤖 **AI-generated PR** — written by Claude Code, posted by @yingjerkao.`
2. **Summary**: `ipeps()` and `ctm_2site()` run the 2x2 recipe of `ctm_tensor_2site` and return the raw pair in documented leg order; `compute_energy_ctm_2site()` sums all four bonds; `ctm_2site()` is deprecated; the legacy 2-site sweep is deleted. Follow-up 2 of #1183. Plan path.
3. **Evidence**: the two defects with their numbers (gauge spread −0.7815 to +0.8171 over 12 runs; dimer −0.75 vs −0.375; the guide example −0.6393 vs −0.6568), the RED → GREEN run of `tests/test_ipeps_2site_energy.py` (8 failed at Task 0, 8 passed at Task 5), and the orientation numbers (2.3e-2, 0.22, ≥ 4.5e-4). The doc examples before and after, from `$W/t6_examples.txt` and probes N and Q: the guide's first example printed −0.633605, the second, the README and the skill −0.630276, each with the `UserWarning` "CTM did not converge in ipeps()"; now each prints −0.656790 with no warning. The skill's `dt=0.3` comment changes from −0.53 to −0.645.
4. **Changelog note** (for the release PR):
   - Fixed: the energy `ipeps()` reports for its 2-site pair. It moved with the bond gauge of the pair and counted two of the four bonds; it is now the 2x2 measurement, half the four-bond sum. Every recorded `ipeps()` energy changes.
   - Fixed: `compute_energy_ctm_2site` sums all four bonds (#493's count).
   - Changed: `ctm_2site()` runs the 2x2 recipe and is deprecated in favour of `ctm_tensor_2site`; its environments change; it no longer runs under `jax.jit`; a `projector_method` other than `"svd"` is ignored with a `UserWarning`, and `qr_warmup_steps` is ignored; `return_meta` info fields are Python scalars (were JAX arrays); a `DenseTensor` is accepted; a raw array that is not 5 legs with equal virtual dimensions raises `ValueError`; a `SymmetricTensor` raises `TypeError` naming `ctm_tensor_2site`.
   - Removed (private): `_ctm_2site_sweep`, `_renormalize_env`, `_ctm_move`, `_ctm_move_eigh`, `_ctm_{left,right,top,bottom}_move_2site`, and their re-exports in `ipeps_ctm.py`.
5. **Changed tests, with reasons**: every edit of Tasks 1–6, one line each (new cases, re-derived tests, starved budgets with their measured numbers, the removed jit test and the two removed stubs, the re-pointed bomb and routing tests, the tightened bound).
6. **Mutation count**: the 19 lines and the total from `$W/mutation.txt`.
7. **Verification**: the summary lines of `$W/t7_core.txt` and `$W/t7_algo.txt`; ruff clean; `sphinx-build -W` result (`$W/t6_docs.txt`); the example runs (`$W/t6_examples.txt`).
8. **Not in this PR**: `ctm_tensor_2site`'s public signature; `ctm_split`; the symmetric SU path; `compute_excitations` (follow-up 1); the duplicate `_wrap_as_dense_tensor` in `ipeps.py`.
9. The footer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`, a blank line, `https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`.

- [ ] **Step 7: Hand over**

```bash
cd /home/yjkao/tenax && git status --short && git log --oneline origin/main..HEAD
```

Expected: only `?? out` and `?? out2`; the RED test commit, the merge, the plan commit and the six task commits. Then use superpowers:finishing-a-development-branch. Push and open the PR only when the user says so, with `$W/pr_body.md` as the body (`gh pr edit` is broken in this clone; set a body later with `gh api repos/tenax-lab/tenax/pulls/<n> -X PATCH -F body=@<file>`). Do not arm `--auto`.
