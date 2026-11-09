# Delegate `ctm()` to the 2x2 Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public `tenax.ctm()` run `ctm_tensor(recipe="2x2")` behind a `DeprecationWarning`, so every caller gets a converged environment, then delete the legacy 1-site sweep that only `ctm()` reached.

**Architecture:** `ctm_tensor` gains a keyword-only `min_iter` (default 1). A private helper, `_ctm_via_2x2(A, config, *, return_meta=False)` in `ipeps_ctm_convergence.py`, checks and wraps a raw 5-leg array, maps the `CTMConfig` fields, always passes `projector_method="svd"`, runs `ctm_tensor(recipe="2x2")`, and converts the result to a raw-array `CTMEnvironment` with `_as_dense_env`, which moves to a new module, `_ctm_env_dense.py`. `ctm()` keeps its signature: it emits the warnings, refuses `initial_env`, and returns what the helper returns. `fermionic_ctm` and the fixture-only tests call the helper directly.

**Tech Stack:** Python 3.11, JAX (x64), NumPy, pytest, ruff, pre-commit, uv.

**Spec:** `docs/superpowers/specs/2026-11-04-ctm-delegate-to-2x2-design.md` (revised 2026-11-09).

**Provenance:** Dry-run 2026-11-09 on a `git archive` copy of 29e19367, with the main clone's venv and `PYTHONPATH` on the copy. Every `# edit` block applied in order. The RED and GREEN Expected lines of Tasks 1–4 and 6, the Task 5 counts, the Task 6 examine step and the Task 7 absence check matched. The per-file test counts in Task 5 Step 4 and Task 6 Step 6 matched a collection run. Ruff 0.17.3 (`uv.lock`) and 0.15.15 (pre-commit) passed on `src tests`. `mutate.py --dry` found 18 unique anchors, and its two test files collect 66 tests. At the end state, the 19 test files that the branch touches, or that import a touched module, gave 626 passed, 3 skipped, 1 xfailed in 1421 s; both kagome examples exited 0. The step times come from the per-file durations of that run. Not dry-run: the commits and their hooks, `-m core`, `sphinx-build`, and the mutation runs. The dry run found two defects: two cuts each left one extra blank line (ruff I001). Reading found four more: a comprehension layout, a docstring anchor that kept half a paragraph, an unverified issue reference, and a garbled count sentence. All six are fixed, and mutant M18 (`max_iter` dropped) is added.

## Global Constraints

- `ctm(A, config, initial_env=None, *, return_meta=False)` keeps its signature. It returns a raw-array `CTMEnvironment`, or `(CTMEnvironment, CTMConvergenceInfo)` with `return_meta=True`, as today.
- `ctm()` emits a `DeprecationWarning` that names `ctm_tensor` as the replacement. `_ctm_via_2x2` emits no warning, so neither `fermionic_ctm` nor the fixture-only tests trigger the `DeprecationWarning`.
- `ctm_tensor` gains a keyword-only `min_iter`, default 1, which leaves every existing call unchanged. Same rule as the other entry points: `converged` only when `n_iter >= min(min_iter, max_iter)` (#976). `max_iter` is the caller's, taken before the QR warm-up shortens the loop, and warm-up sweeps count.
- `CTMConfig.min_iter` (default 10) goes to `ctm_tensor(min_iter=)`, so `ctm()` keeps requiring at least 10 sweeps.
- The helper always passes `projector_method="svd"`, because on the 2x2 recipe `"qr"` still starts the QR warm-up loop of `ctm_tensor`. `config.projector_method != "svd"` gives a `UserWarning` in `ctm()`. `config.qr_warmup_steps` is ignored.
- Raw `A`: only 5 legs `(u, d, l, r, s)` with four virtual legs of one dimension. Any other shape gives a `ValueError` that names the 5-leg layout. A `DenseTensor` is passed through. A `SymmetricTensor` gives a `TypeError` that names `ctm_tensor`; never call `todense()` on it.
- `initial_env` not `None` gives a `ValueError` with a migration message.
- `_as_dense_env` and `_as_dense_array` move from `ipeps_excitations.py` to `src/tenax/algorithms/_ctm_env_dense.py`. Behaviour and messages do not change.
- The imports in `_ctm_via_2x2` stay local to the function body, as the spec says. Write no comment that claims an import cycle: there is none (checked 2026-11-09; `_ctm_env_dense` imports only `ipeps_config` and `core.tensor`, `_ipeps_optimize_shared` only `ipeps_config` and `core.*`).
- Out of scope (spec): `ctm_2site` and `ctm_split`; `ctm_tensor(recipe="1x1")`; the raw `CTMEnvironment` type and `ipeps_rdm`; `compute_excitations` and #1052; the QR warm-up of `ctm_tensor(recipe="2x2", projector_method="qr")`.
- A changed numeric assertion in an existing test gets examined, not re-frozen. The PR lists every changed test with its reason (Task 9 drafts the list).
- Run Python only through `uv run`, from the worktree root. Bare `python` imports the main clone's tenax.
- Before each pytest run, check `uptime` and wait while the 1-minute load is above 12. One pytest run at a time, under `JAX_PLATFORMS=cpu taskset -c 0-11`. Redirect output to a file in `$W` and read the file afterwards; never pipe a run into `tail` or `head`. A run that can take more than 2 minutes goes in the background and ends with `exit $rc`. Pass `--no-cov`: `addopts` has `--cov=tenax --cov-report=xml`, which writes `coverage.xml` into the tree.
- A new test file needs a `_FILE_MARKERS` entry in `tests/conftest.py`, or `test_bucket_registry.py` fails.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`. pre-commit (ruff, ruff-format) must pass. If a hook rewrites a file, `git add` it and run the same `git commit` again: the failed commit left HEAD unchanged. Never use `--no-verify`, `--amend` after a hook failure, or `--reuse-message=HEAD`.
- To bring the branch up to date: `git merge origin/main`, never rebase.
- No push, no PR and no `gh` call inside the tasks. After Task 9 the finishing skill asks the user. Arming `--auto` is the user's decision.

## Review Focus

1. A NumPy site array (users often build tensors in NumPy) → the same environment as the JAX array, bit for bit. Pinned by `test_numpy_and_dense_tensor_inputs_match_the_jax_array` (Task 3; Task 6 adds the `ctm` case).
2. A raw 5-leg array whose virtual legs differ, such as `(2, 2, 3, 3, 2)` → a `ValueError` that names the layout, not an environment built with wrong leg dimensions (`_wrap_as_dense_tensor` reads `shape[0]` for all four virtual legs, unchecked). Pinned by `test_rejects_a_bad_raw_shape[unequal-virtual-*]` (Task 3).
3. A `DenseTensor`, the type `ipeps()` returns, passed to `ctm()` → passed through, with the environment of its raw array. Today's `ctm()` raises `TypeError` on it (measured). Pinned by the test of item 1.
4. A complex site tensor through `ctm()` → the edges keep their imaginary part (max |Im T| > 1e-3; 0.57 measured), so no hidden `.real` (#842). Pinned by `test_ctm_keeps_the_imaginary_part_of_a_complex_state` (Task 6).
5. `projector_method="eigh"`, not only `"qr"` → a `UserWarning`, and the same environment as `"svd"`. Pinned by `test_ctm_warns_that_projector_method_is_ignored[eigh]` (Task 6).

## File Structure

| File | Change | Task |
|---|---|---|
| `src/tenax/algorithms/_ctm_tensor_convergence.py` | `ctm_tensor(min_iter=)`; two paragraphs of the `CTMConvergenceInfo` docstring | 1, 6 |
| `src/tenax/algorithms/_ctm_env_dense.py` | **new**: `_as_dense_array`, `_as_dense_env`, moved unchanged | 2 |
| `src/tenax/algorithms/ipeps_excitations.py` | import the two helpers from the new module | 2 |
| `src/tenax/algorithms/ipeps_ctm_convergence.py` | `_ctm_via_2x2` (3); `ctm()` delegates (6); `_ctm_sweep` deleted (7) | 3, 6, 7 |
| `src/tenax/algorithms/fermionic_ipeps.py` | raw branch of `fermionic_ctm` calls `_ctm_via_2x2` | 4 |
| `src/tenax/algorithms/ipeps_ctm_moves.py` | delete `_ctm_move_qr` and the 1-site moves; `_ctm_move` loses `projector_method` | 7 |
| `src/tenax/algorithms/ipeps_ctm.py` | drop six re-exports | 7 |
| `pyproject.toml` | filter for the new `DeprecationWarning` | 6 |
| `tests/test_ctm_delegates_to_2x2.py` | **new**, `core`: Onsager oracle, input checks, warnings, `fermionic_ctm` | 3, 4, 6 |
| `tests/conftest.py` | `core` bucket for the new file | 3 |
| `tests/test_ctm_min_iter_contract_925_976.py` | `ctm_tensor` and `_ctm_via_2x2` cases; two QR warm-up tests removed | 1, 3, 6 |
| `tests/test_ipeps_excitations.py`, `tests/test_ipeps_spectrum_contractions.py` | helper imports from `_ctm_env_dense` | 2 |
| `tests/test_ipeps.py`, `tests/test_ipeps_core.py`, `tests/test_ipeps_excitations.py`, `tests/test_ipeps_excitations_gauge.py`, `tests/test_ipeps_rdm_gauge.py`, `tests/test_normalise_rdm_zero_grad.py`, `tests/test_rdm_gate_transpose_966.py` | fixture environments from `_ctm_via_2x2` | 5 |
| `tests/test_ipeps.py`, `tests/test_ipeps_core.py` | warm-start and QR tests removed | 6 |
| `tests/test_ctm_convergence_meta.py` | starved budget 20 → 10; two docstrings | 6 |
| `tests/test_ctm_tensor.py` | drop the unused `_ctm_sweep` import | 7 |
| `docs/guide/algorithms/ipeps.md`, `docs/guide/algorithms/ctm.md` | `ctm_tensor` example, deprecation note, `min_iter` | 8 |
| `examples/kagome_xxz_pess.py`, `examples/kagome_xxz_spin1_pess.py` | `ctm_tensor` + `compute_energy_ctm_tensor` | 8 |

Not changed, checked: `tests/test_ctm_complex_site_tensors.py`, `tests/test_ctm_criterion_rank_blind_898.py` and the other contract tests stay on `ctm()`; `tests/test_split_ctm_tensor.py` imports `ctm` without using it; `docs/api/algorithms.rst` autodocs the new `ctm()` docstring. An AST scan found these `ctm()` callers outside tests: `fermionic_ipeps.py:655` and the two kagome examples. All callers pass 5-leg arrays. No `ctm()` call sits under `jax.jit` in `src`, `tests`, `examples` or `benchmarks`. `README.md` does not mention `ctm()`.

## Measured numbers

Measured 2026-11-09 on this branch at 29e19367, CPU, x64. C(1,0) error is the difference from Onsager's exact value, read off the window oracle in `tests/_window_oracle.py`. All runs use `CTMConfig(chi, max_iter=500, conv_tol=1e-12)` unless stated.

| state | path | chi 8 | chi 12 | chi 16 | chi 32 |
|---|---|---|---|---|---|
| Ising beta=0.2 | 2x2 (`ctm_tensor`) | 40 sweeps, 3.1e-16 | 14, 2.2e-16 | 14, 1.9e-16 | 14, 1.9e-16 |
| Ising beta=0.2 | old `ctm()` | converged at 318, 2.5e-14 | not converged, 7.6e-4 | not converged, 4.3e-4 | not converged, 4.3e-4 |
| Ising beta=0.3 | 2x2 | 49, 1.6e-14 | 54, 1.7e-16 | 43, 2.8e-16 | 57, 4.4e-16 |
| Ising beta=0.3 | old `ctm()` | not converged (diff 6.6e-10) | not (5.0e-9) | not (1.8e-6) | not (3.1e-7) |

Every 2x2 error is at least 64 times below 1e-12, so the beta=0.3 test keeps all four chi values. The old `fermionic_ctm` raw branch gives the old `ctm()` numbers. Other probes:

- `su_state` of `test_ctm_convergence_meta.py`, 2x2, chi=6, `conv_tol=1e-12`: converged at sweep 16 (diff 2.2e-13); the diff at sweep 10 is 7.7e-9. With `conv_tol=1e-2` it converges at sweep 3, or 10 with `min_iter`.
- `jnp.ones((1, 1, 1, 1, 2))`, chi=1: converged at sweep 2, diff 0. With `projector_method="qr", qr_warmup_steps=6`: sweep 7.
- Random D=2 state (`PRNGKey(42)`, normalised), chi=4, 5 sweeps: max |C1| = 1.0 with `renormalize=True`, 0.748 without.
- Random complex D=2 state (`PRNGKey(0)`, imaginary part from `fold_in(key, 1)`), chi=4, 6 sweeps: max |Im T| = 0.57 on the edges, 0.34 on the corners (old `ctm()`: 0.33 on the edges).
- NumPy and JAX inputs give bit-identical environments. The 2x2 runs above emit no warnings. `C1` is `(chi, chi)`, `T1` is `(chi, D**2, chi)`, raw `jax.Array`.
- Old `ctm()` on bad input: a 3-leg `(4, 4, 4)` array runs; `(2, 2, 2, 2)` and 6-leg arrays raise an einsum `ValueError`; `(2, 2, 3, 3, 2)` raises a reshape `TypeError`; `SymmetricTensor` and `DenseTensor` raise `TypeError: conjugate requires ndarray`. Its info fields are JAX arrays; `ctm_tensor`'s are Python scalars.
- Preview (old `ctm()` patched to the 2x2 delegation without `min_iter`; the 11 files that call `ctm()`): 10 failed, 356 passed, 1 skipped, 1 xfailed in 803 s. The failures are exactly the warm-start tests, `test_ctm_reports_status`, three `min_iter` cases on `ctm`, and the two QR warm-up tests. Every fixture-only test passed.

## Execution setup (once, before Task 1)

```bash
cd /home/yjkao/tenax-ctm-cleanup
git status --short                      # Expected: no output
git log --oneline -1                    # Expected: the plan commit, on refactor/ctm-delegate-to-2x2
uv run python -c "import tenax; print(tenax.__file__)"
# Expected: /home/yjkao/tenax-ctm-cleanup/src/tenax/__init__.py (the first run creates .venv)
test -x "$(git rev-parse --git-path hooks)/pre-commit" || uv run pre-commit install
W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2
mkdir -p "$W"
uptime                                  # wait while the 1-minute load is above 12
```

The edit steps are short Python scripts. Each script defines one helper, `edit(rel, old, new, end=None)`: it asserts that `old` (and `end`, if given) occurs exactly once, replaces `old` with `new` (or, with `end`, everything from `old` up to but not including `end`; `end=""` means to the end of the file), and writes nothing until every edit in the script has passed its check. A failed assertion leaves every file as it was.

---

### Task 1: `ctm_tensor(min_iter=)`

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` — the `n_iter` paragraph of the `CTMConvergenceInfo` docstring (`:1027-1041`), the `ctm_tensor` signature (`:1054-1066`) and Args, the loop (`:1304`, `:1367`)
- Test: `tests/test_ctm_min_iter_contract_925_976.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ctm_tensor(A, chi, max_iter=100, conv_tol=1e-8, renormalize=True, projector_method="svd", qr_warmup_steps=3, projector_backward="auto", recipe="2x2", *, min_iter: int = 1, return_meta: bool = False)`. With `return_meta=True` it returns `(CTMTensorEnv, float, CTMConvergenceInfo)`, and the info fields are Python `bool`, `int` and `float`. `converged` is True only if `n_iter >= min(min_iter, max_iter)`, where `max_iter` is the caller's and the QR warm-up sweeps count. Task 3 passes `min_iter=config.min_iter`.

- [ ] **Step 1: Write the failing tests**

```bash
# edit: Task 1, tests
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "tests/test_ctm_min_iter_contract_925_976.py"
edit(
    F,
    "violating ``CTMConvergenceInfo``'s own invariant (``n_iter == max_iter`` when\n"
    "not converged).\n",
    "violating ``CTMConvergenceInfo``'s own invariant (``n_iter == max_iter`` when\n"
    "not converged).  ``ctm_tensor`` takes the same gate as a keyword-only\n"
    "``min_iter`` (default 1, which keeps its plain early exit), so it runs the\n"
    "same cases.\n",
)
edit(
    F,
    "from tenax import CTMConfig, ctm, ctm_2site, ctm_split\n",
    "from tenax import CTMConfig, ctm, ctm_2site, ctm_split, ctm_tensor\n"
    "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n",
)
edit(
    F,
    "def _info_ctm_split(cfg):\n    return ctm_split(A, cfg, return_meta=True)[1]\n",
    "def _info_ctm_split(cfg):\n"
    "    return ctm_split(A, cfg, return_meta=True)[1]\n"
    "\n"
    "\n"
    "def _info_ctm_tensor(cfg):\n"
    "    return ctm_tensor(\n"
    "        _wrap_as_dense_tensor(A),\n"
    "        cfg.chi,\n"
    "        max_iter=cfg.max_iter,\n"
    "        conv_tol=cfg.conv_tol,\n"
    "        min_iter=cfg.min_iter,\n"
    "        return_meta=True,\n"
    "    )[2]\n",
)
edit(
    F,
    '    pytest.param(_info_ctm_split, id="ctm_split"),\n',
    '    pytest.param(_info_ctm_split, id="ctm_split"),\n'
    '    pytest.param(_info_ctm_tensor, id="ctm_tensor"),\n',
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
cat >> tests/test_ctm_min_iter_contract_925_976.py <<'PYEOF'


def test_ctm_tensor_default_min_iter_keeps_the_early_exit():
    """``min_iter`` defaults to 1, so a caller that does not pass it keeps the
    plain early exit: the stationary state certifies at the first comparison,
    sweep 2."""
    _env, _eps, info = ctm_tensor(
        _wrap_as_dense_tensor(A), 1, max_iter=40, conv_tol=1e-8, return_meta=True
    )
    assert info.converged
    assert info.n_iter == 2


def test_ctm_tensor_min_iter_counts_the_warmup_against_the_callers_budget():
    """The cap is ``min(min_iter, max_iter)`` with the caller's ``max_iter``,
    and the QR warm-up sweeps count toward it.  Six warm-up sweeps leave four
    in the loop; a cap taken after the warm-up, ``min(10, 4) = 4``, would
    certify at sweep 7."""
    _env, _eps, info = ctm_tensor(
        _wrap_as_dense_tensor(A),
        1,
        max_iter=10,
        conv_tol=1e-8,
        projector_method="qr",
        qr_warmup_steps=6,
        min_iter=10,
        return_meta=True,
    )
    assert info.converged
    assert info.n_iter == 10
PYEOF
```

Expected: `edited tests/test_ctm_min_iter_contract_925_976.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_min_iter_contract_925_976.py > "$W/t1_red.txt" 2>&1; echo "rc=$?"; tail -n 8 "$W/t1_red.txt"
```

Expected: `rc=1` and `5 failed, 15 passed`. The five failures are `test_convergence_waits_for_min_iter[2-ctm_tensor]`, `[10-ctm_tensor]`, `[30-ctm_tensor]`, `test_min_iter_is_capped_at_max_iter[ctm_tensor]` and `test_ctm_tensor_min_iter_counts_the_warmup_against_the_callers_budget`, each with `TypeError: ctm_tensor() got an unexpected keyword argument 'min_iter'`. `test_ctm_tensor_default_min_iter_keeps_the_early_exit` passes already: it is the guard that the default changes nothing.

- [ ] **Step 3: Implement**

```bash
# edit: Task 1, implementation
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/_ctm_tensor_convergence.py"
SIGNATURE_END = (
    ") -> tuple[CTMTensorEnv, float] | tuple[CTMTensorEnv, float, CTMConvergenceInfo]:\n"
)
edit(
    F,
    '    recipe: str = "2x2",\n    *,\n    return_meta: bool = False,\n' + SIGNATURE_END,
    '    recipe: str = "2x2",\n    *,\n    min_iter: int = 1,\n    return_meta: bool = False,\n'
    + SIGNATURE_END,
)
RETURN_META = "        return_meta:       When ``True``, return a third element, a\n"
edit(
    F,
    RETURN_META,
    '''\
        min_iter:          Minimum sweeps before ``converged`` can be set,
                           capped at ``max_iter`` so that a small budget stays
                           convergeable (#976).  QR warm-up sweeps count toward
                           it.  The default, 1, keeps the plain early exit on
                           ``conv_tol``.  The ``CTMConfig``-based entry points
                           apply the same rule with ``CTMConfig.min_iter``.
'''
    + RETURN_META,
)
edit(
    F,
    '\n    diff = float("inf")\n    n_iter = 0\n',
    '''
    diff = float("inf")
    n_iter = 0
    # ``converged`` needs ``min(min_iter, max_iter)`` sweeps, warm-up included,
    # as in the ``CTMConfig`` entry points (#976).  The cap uses the caller's
    # ``max_iter``, before the warm-up below shortens the loop's budget.  The
    # warm-up's own ``converged`` needs no gate: it survives only when the
    # warm-up spends the whole budget, and then ``n_iter == max_iter``.
    min_sweeps = min(min_iter, max_iter)
''',
)
edit(
    F,
    "\n            if diff < conv_tol:\n                converged = True\n                break\n",
    "\n            if diff < conv_tol and n_iter >= min_sweeps:\n"
    "                converged = True\n                break\n",
)
edit(
    F,
    "                   which one built the info.  The three ``CTMConfig``-based\n",
    '''\
                   which one built the info.  All four also gate convergence
                   on ``min(min_iter, max_iter)`` -- the warm-up counts toward
                   it and the cap keeps a small ``max_iter`` convergeable, so
                   ``n_iter >= min(min_iter, max_iter)`` whenever
                   ``converged`` is True (#976).  The three ``CTMConfig``-based
                   entry points (:func:`ctm`, :func:`ctm_2site`,
                   :func:`ctm_split`) read ``CTMConfig.min_iter`` (default
                   10); :func:`ctm_tensor` takes a keyword-only ``min_iter``
                   whose default, 1, keeps its plain early exit.
''',
    end="        diff:      Final value of the convergence criterion -- the max\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited src/tenax/algorithms/_ctm_tensor_convergence.py`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_min_iter_contract_925_976.py > "$W/t1_green.txt" 2>&1; echo "rc=$?"; tail -n 3 "$W/t1_green.txt"
```

Expected: `rc=0` and `20 passed`.

- [ ] **Step 5: Run the other `ctm_tensor` tests (background)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_tensor_return_info.py tests/test_ctm_tensor.py > "$W/t1_regress.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t1_regress.txt"; exit $rc
```

Expected (read `tail -n 5 "$W/t1_regress.txt"` after the notification): `rc=0`, and the summary line has no `failed` and no `error`. `test_ctm_tensor_return_info.py` has 15 tests and `test_ctm_tensor.py` 31, some of them skipped.

- [ ] **Step 6: Commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_ctm_min_iter_contract_925_976.py && git commit -F - <<'MSG'
feat(ctm): ctm_tensor takes min_iter, gated like the CTMConfig entry points

ctm_tensor early-exited on conv_tol alone.  It gains a keyword-only
min_iter (default 1, which changes no existing call) and certifies
convergence only at n_iter >= min(min_iter, max_iter), with the caller's
max_iter and the QR warm-up counted: the #976 rule of ctm, ctm_2site and
ctm_split.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

Expected: the hooks pass and the commit is made. If ruff-format rewrites a file, `git add` it and run the same commit command again.

---

### Task 2: Move `_as_dense_env` and `_as_dense_array` to `_ctm_env_dense`

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Create: `src/tenax/algorithms/_ctm_env_dense.py`
- Modify: `src/tenax/algorithms/ipeps_excitations.py:36-97` (imports; the "Input normalization" block)
- Test: `tests/test_ipeps_excitations.py:471,482`, `tests/test_ipeps_spectrum_contractions.py:27,46-53`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `tenax.algorithms._ctm_env_dense._as_dense_array(x: jax.Array | Tensor) -> jax.Array` and `_as_dense_env(env) -> CTMEnvironment`, with today's behaviour and messages: a raw `CTMEnvironment` comes back unchanged, a `CTMTensorEnv` has each field `.todense()`-ed, a `SymmetricTensor` raises `NotImplementedError`, and anything other than 8 fields raises `ValueError`. Task 3 imports `_as_dense_env` from here.

- [ ] **Step 1: Point the tests at the new module**

```bash
# edit: Task 2, tests
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "tests/test_ipeps_excitations.py"
edit(
    F,
    "        from tenax.algorithms.ipeps_excitations import _as_dense_env\n",
    "        from tenax.algorithms._ctm_env_dense import _as_dense_env\n",
)
edit(
    F,
    "        from tenax.algorithms.ipeps_excitations import _as_dense_array\n",
    "        from tenax.algorithms._ctm_env_dense import _as_dense_array\n",
)
F = "tests/test_ipeps_spectrum_contractions.py"
edit(
    F,
    "from tenax.algorithms._ctm_tensor_convergence import (\n",
    "from tenax.algorithms._ctm_env_dense import _as_dense_env\n"
    "from tenax.algorithms._ctm_tensor_convergence import (\n",
)
edit(
    F,
    "from tenax.algorithms.ipeps_excitations import (\n    _as_dense_env,\n",
    "from tenax.algorithms.ipeps_excitations import (\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited tests/test_ipeps_excitations.py` and `edited tests/test_ipeps_spectrum_contractions.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_spectrum_contractions.py > "$W/t2_red_a.txt" 2>&1; echo "rc=$?"; tail -n 4 "$W/t2_red_a.txt"; JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q "tests/test_ipeps_excitations.py::TestTensorInputAcceptance" > "$W/t2_red_b.txt" 2>&1; echo "rc=$?"; tail -n 4 "$W/t2_red_b.txt"
```

Expected: the first run `rc=2` with `1 error` during collection, `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_env_dense'`. The second run `rc=1` and `2 failed, 2 passed`: `test_split_env_rejected_with_clear_error` and `test_symmetric_tensor_rejected` fail with the same `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

```bash
# edit: Task 2, new module
cd /home/yjkao/tenax-ctm-cleanup && cat > src/tenax/algorithms/_ctm_env_dense.py <<'PYEOF'
"""Raw-array views of CTM environments for the dense-only code paths.

``optimize_gs_ad`` and ``ctm_tensor`` return Tensor-protocol objects: a
``DenseTensor`` site tensor and a ``CTMTensorEnv`` whose 8 fields are
``DenseTensor``s.  The excitation contractions and the raw-array result of
``ctm()`` use plain ``jax.Array``s and a raw-array ``CTMEnvironment``.  The
fields of a ``CTMTensorEnv`` share the leg and flow convention of the raw
``CTMEnvironment`` (verified to machine precision against
``compute_energy_ctm``), so ``.todense()`` on each field is correct and keeps
the converged environment.

For a ``DenseTensor``, ``.todense()`` returns the stored buffer, so it is free.
A ``SymmetricTensor`` would be densified here, against the project rule on
``todense()`` on the symmetric path (CLAUDE.md / AGENTS.md): CTM edges scale as
``chi * D**2 * chi``, and a block-sparse production run could run out of memory
or silently bypass the symmetry machinery.  So it is refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.core.tensor import SymmetricTensor, Tensor


def _as_dense_array(x: jax.Array | Tensor) -> jax.Array:
    """Return a raw ``jax.Array`` from an array or ``DenseTensor``.

    Raises ``NotImplementedError`` for ``SymmetricTensor`` — the excitation
    path is dense-only and must not silently densify a block-sparse tensor.
    """
    if isinstance(x, SymmetricTensor):
        raise NotImplementedError(
            "compute_excitations does not support SymmetricTensor inputs: the "
            "excitation contraction path is dense-only and densifying a "
            "block-sparse tensor would defeat the symmetry machinery (and can "
            "OOM at large chi/D). Convert the ground state to a dense iPEPS "
            "before computing excitations, or track symmetric-aware "
            "excitations as a follow-up."
        )
    if isinstance(x, Tensor):  # DenseTensor: todense() returns the stored buffer
        return x.todense()
    return jnp.asarray(x)


def _as_dense_env(env) -> CTMEnvironment:
    """Coerce a CTM environment to a raw-array ``CTMEnvironment``.

    Accepts the legacy raw-array ``CTMEnvironment`` (returned unchanged) or the
    Tensor-protocol ``CTMTensorEnv`` produced by ``optimize_gs_ad`` (each of the
    8 corner/edge fields is converted via ``.todense()``).
    """
    fields = tuple(env)
    if len(fields) != 8:
        raise ValueError(
            "compute_excitations expects an 8-tensor CTM environment "
            f"(4 corners + 4 edges); got {len(fields)} fields. Split-CTM "
            "environments are not supported by the excitation path."
        )
    if any(isinstance(f, Tensor) for f in fields):
        return CTMEnvironment(*(_as_dense_array(f) for f in fields))
    return env if isinstance(env, CTMEnvironment) else CTMEnvironment(*fields)
PYEOF
```

The two functions are copied verbatim from `ipeps_excitations.py:61-97`; the module docstring is the comment block from `:40-58`, reworded.

- [ ] **Step 4: Remove the block from `ipeps_excitations.py` and import from the new module**

```bash
# edit: Task 2, excitations
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/ipeps_excitations.py"
BANNER = "# ---------------------------------------------------------------------------\n"
edit(
    F,
    "from tenax.algorithms._einsum_compat import einsum_promoted\n",
    "from tenax.algorithms._ctm_env_dense import _as_dense_array, _as_dense_env\n"
    "from tenax.algorithms._einsum_compat import einsum_promoted\n",
)
edit(
    F,
    "from tenax.core.tensor import SymmetricTensor, Tensor\n",
    "from tenax.core.tensor import SymmetricTensor\n",
)
edit(
    F,
    BANNER + "# Input normalization\n",
    "",
    end=BANNER + "# Configuration and result dataclasses\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited src/tenax/algorithms/ipeps_excitations.py`. `Tensor` is no longer used there; `SymmetricTensor` still is (`:1016`).

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q "tests/test_ipeps_excitations.py::TestTensorInputAcceptance" > "$W/t2_green_a.txt" 2>&1; echo "rc=$?"; tail -n 3 "$W/t2_green_a.txt"
```

Expected: `rc=0` and `4 passed`.

Then both files, in the background:

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps_excitations.py tests/test_ipeps_spectrum_contractions.py > "$W/t2_green.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t2_green.txt"; exit $rc
```

Expected: `rc=0`; no `failed` and no `error` in the summary line (54 + 11 tests collected).

- [ ] **Step 6: Lint and commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src/tenax/algorithms/_ctm_env_dense.py src/tenax/algorithms/ipeps_excitations.py tests/test_ipeps_excitations.py tests/test_ipeps_spectrum_contractions.py && uv run ruff format --check src/tenax/algorithms/_ctm_env_dense.py src/tenax/algorithms/ipeps_excitations.py tests/test_ipeps_excitations.py tests/test_ipeps_spectrum_contractions.py
```

Expected: `All checks passed!` and `4 files already formatted`.

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/_ctm_env_dense.py src/tenax/algorithms/ipeps_excitations.py tests/test_ipeps_excitations.py tests/test_ipeps_spectrum_contractions.py && git commit -F - <<'MSG'
refactor(excitations): move _as_dense_env and _as_dense_array to _ctm_env_dense

ctm() will convert its CTMTensorEnv to a raw CTMEnvironment with the same
helper, so it moves out of the excitation module.  Behaviour and messages
are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 3: `_ctm_via_2x2`

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ctm_convergence.py` — `__all__` (`:5-16`); the new function goes right before `_renormalize_env` (`:229`), so it sits right after `ctm()`
- Create: `tests/test_ctm_delegates_to_2x2.py`
- Modify: `tests/conftest.py` — `_FILE_MARKERS`, after the `test_ctm_min_iter_contract_925_976.py` entry (`:189`)
- Modify: `tests/test_ctm_min_iter_contract_925_976.py`

**Interfaces:**
- Consumes: Task 1, `ctm_tensor(..., *, min_iter=1, return_meta=False)`; Task 2, `_as_dense_env` in `tenax.algorithms._ctm_env_dense`; the existing `_wrap_as_dense_tensor(arr) -> DenseTensor` in `tenax.algorithms._ipeps_optimize_shared`, which reads `shape[0]` for all four virtual legs and `shape[4]` for the physical leg, without checks.
- Produces: `tenax.algorithms.ipeps_ctm_convergence._ctm_via_2x2(A, config: CTMConfig, *, return_meta: bool = False) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]`, which emits no warning. In the new test file: `ONES`, `ISING_CASES`, `ENTRY_POINTS` (Task 6 adds `ctm`), `_c10_error(env, beta) -> float`, `_random_site(seed=42)`, `_symmetric_site()`.

- [ ] **Step 1: Write the new test file**

```bash
# edit: Task 3, new test file
cd /home/yjkao/tenax-ctm-cleanup && cat > tests/test_ctm_delegates_to_2x2.py <<'PYEOF'
"""``ctm()`` runs ``ctm_tensor(recipe="2x2")`` since 0.9.3.

The 1-site sweep that ``ctm()`` ran before reached no fixed point.  The oracle
here is the Ising PEPS: its norm is the classical 2D Ising partition function,
so C(1,0) has Onsager's exact value.  At beta=0.2 the old sweep stopped at
chi=12, 16 and 32 without converging, with C(1,0) errors of 7.6e-4, 4.3e-4 and
4.3e-4; at beta=0.3 it converged at no chi.  The 2x2 recipe converges at every
chi below in 14 to 57 sweeps, with errors of at most 1.6e-14 (measured
2026-11-09).

``_ctm_via_2x2`` is the delegation without ``ctm()``'s warnings.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import apply_op
from _window_oracle import closed_double_layer, correlators, ising_peps, onsager_c10

from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_config import CTMConfig, CTMEnvironment
from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo, _ctm_via_2x2
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

# A 1x1 product state: its chi=1 corners are stationary from the second sweep,
# so the stopping rule is all that moves.
ONES = jnp.ones((1, 1, 1, 1, 2))

ISING_CASES = [
    pytest.param(beta, chi, id=f"beta{beta}-chi{chi}")
    for beta in (0.2, 0.3)
    for chi in (8, 12, 16, 32)
]

# Every entry point that must return the environment of the 2x2 recipe.
ENTRY_POINTS = [pytest.param(_ctm_via_2x2, id="helper")]


def _c10_error(env, beta):
    """|C(1,0) - Onsager| from the window oracle, which shares no code with CTM."""
    A = ising_peps(beta)
    B = apply_op(A, np.diag([1.0, -1.0]))
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        1,
    )
    return abs(C[(1, 0)].real - onsager_c10(beta))


def _random_site(seed=42):
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


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("beta,chi", ISING_CASES)
def test_ising_c10_matches_onsager(entry, beta, chi):
    """Converged, and C(1,0) within 1e-12 of Onsager, at every chi."""
    env, info = entry(
        jnp.asarray(ising_peps(beta)),
        CTMConfig(chi=chi, max_iter=500, conv_tol=1e-12),
        return_meta=True,
    )
    assert info.converged
    # ``conv_tol`` reached ``ctm_tensor``: its default, 1e-8, stops earlier.
    assert info.diff < 1e-12
    assert _c10_error(env, beta) < 1e-12


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "shape",
    [(4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2, 2, 2), (2, 2, 3, 3, 2)],
    ids=["3-leg", "4-leg", "6-leg", "unequal-virtual"],
)
def test_rejects_a_bad_raw_shape(entry, shape):
    """Only 5 legs ``(u, d, l, r, s)`` with one virtual dimension is a site
    tensor.  ``_wrap_as_dense_tensor`` reads ``shape[0]`` for all four virtual
    legs and ``shape[4]`` for the physical one, unchecked.  The old sweep ran
    the 3-leg array and failed on the others with einsum or reshape errors."""
    with pytest.raises(ValueError, match=r"5 legs \(u, d, l, r, s\)"):
        entry(jnp.ones(shape), CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_refuses_a_symmetric_tensor(entry):
    """The raw-array result would need ``todense()`` on a block-sparse tensor;
    the message names ``ctm_tensor``, which keeps the blocks."""
    with pytest.raises(TypeError, match="ctm_tensor"):
        entry(_symmetric_site(), CTMConfig(chi=4, max_iter=5))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_numpy_and_dense_tensor_inputs_match_the_jax_array(entry):
    """A NumPy array is converted and a ``DenseTensor`` (what ``ipeps()``
    returns) passes through: both give the environment of the JAX array, bit
    for bit."""
    A = _random_site()
    cfg = CTMConfig(chi=4, max_iter=5)
    ref = entry(A, cfg)
    for other in (np.asarray(A), _wrap_as_dense_tensor(A)):
        env = entry(other, cfg)
        for name in CTMEnvironment._fields:
            np.testing.assert_array_equal(getattr(env, name), getattr(ref, name))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_returns_a_raw_environment_and_python_scalars(entry):
    """Raw ``jax.Array`` fields of shape ``(chi, chi)`` and ``(chi, D**2,
    chi)``, as ``ctm()`` always returned.  The info holds Python scalars,
    because ``ctm_tensor`` is a Python loop."""
    env, info = entry(
        jnp.asarray(ising_peps(0.2)),
        CTMConfig(chi=8, max_iter=500, conv_tol=1e-12),
        return_meta=True,
    )
    assert isinstance(env, CTMEnvironment)
    assert isinstance(env.C1, jax.Array) and env.C1.shape == (8, 8)
    assert isinstance(env.T1, jax.Array) and env.T1.shape == (8, 4, 8)
    assert isinstance(info, CTMConvergenceInfo)
    assert type(info.converged) is bool
    assert type(info.n_iter) is int


def test_helper_runs_no_qr_warmup():
    """``"svd"`` is passed, so a ``"qr"`` config starts no warm-up loop: the
    stationary state stops at sweep 2, not after six warm-up sweeps."""
    _env, info = _ctm_via_2x2(
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
    """``renormalize`` reaches ``ctm_tensor``: with it the corner's largest
    entry is 1; without it, 0.748 on this state (measured)."""
    env = _ctm_via_2x2(
        _random_site(), CTMConfig(chi=4, max_iter=5, renormalize=renormalize)
    )
    peak = float(jnp.max(jnp.abs(env.C1)))
    if renormalize:
        assert peak == pytest.approx(1.0, abs=1e-12)
    else:
        assert peak < 0.9


def test_helper_emits_no_deprecation_warning():
    """``fermionic_ctm`` and the fixture-only tests call the helper so that
    they do not trigger ``ctm()``'s ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ctm_via_2x2(ONES, CTMConfig(chi=1, max_iter=5))
PYEOF
```

- [ ] **Step 2: Add the helper to the `min_iter` contract file and register the new file**

```bash
# edit: Task 3, min_iter file and conftest
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "tests/test_ctm_min_iter_contract_925_976.py"
edit(
    F,
    "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n",
    "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n"
    "from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2\n",
)
edit(
    F,
    "def _info_ctm_tensor(cfg):\n",
    "def _info_ctm_via_2x2(cfg):\n"
    "    return _ctm_via_2x2(A, cfg, return_meta=True)[1]\n"
    "\n"
    "\n"
    "def _info_ctm_tensor(cfg):\n",
)
edit(
    F,
    '    pytest.param(_info_ctm_tensor, id="ctm_tensor"),\n',
    '    pytest.param(_info_ctm_tensor, id="ctm_tensor"),\n'
    '    pytest.param(_info_ctm_via_2x2, id="ctm_via_2x2"),\n',
)
edit(
    "tests/conftest.py",
    '    "test_ctm_min_iter_contract_925_976.py": "core",\n',
    '''\
    "test_ctm_min_iter_contract_925_976.py": "core",
    # ``ctm()`` runs ``ctm_tensor(recipe="2x2")`` since 0.9.3.  Onsager's C(1,0)
    # pins the environment it returns, next to the input checks and the
    # warnings: a silent wrong environment on a public entry point, so it
    # belongs in the required gate.  The 16 Onsager cases set its run time.
    "test_ctm_delegates_to_2x2.py": "core",
''',
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited tests/test_ctm_min_iter_contract_925_976.py` and `edited tests/conftest.py`.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py > "$W/t3_red.txt" 2>&1; echo "rc=$?"; tail -n 6 "$W/t3_red.txt"
```

Expected: `rc=2` and `2 errors` during collection, both `ImportError: cannot import name '_ctm_via_2x2' from 'tenax.algorithms.ipeps_ctm_convergence'`.

- [ ] **Step 4: Implement the helper**

```bash
# edit: Task 3, implementation
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/ipeps_ctm_convergence.py"
edit(F, '    "ctm",\n', '    "ctm",\n    "_ctm_via_2x2",\n')
RENORMALIZE = "def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:\n"
edit(
    F,
    RENORMALIZE,
    '''\
def _ctm_via_2x2(
    A: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]:
    """``ctm()`` without its warnings: ``ctm_tensor(recipe="2x2")`` with the
    ``CTMConfig`` fields mapped, returned as a raw-array ``CTMEnvironment``.

    For callers that need ``ctm()``'s result and must not trigger its
    ``DeprecationWarning``.

    Args:
        A:           Raw site array with 5 legs ``(u, d, l, r, s)`` and four
                     virtual legs of one dimension, or a ``DenseTensor``, which
                     is passed through.  ``_wrap_as_dense_tensor`` reads
                     ``shape[0]`` for all four virtual legs without a check, so
                     any other shape raises ``ValueError`` here.  A
                     ``SymmetricTensor`` raises ``TypeError``: the raw result
                     would need ``todense()``.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` are passed on.  ``projector_method`` is not:
                     the 2x2 recipe always uses Fishman SVD, and ``"svd"`` is
                     passed because ``"qr"`` would still start the QR warm-up
                     loop of ``ctm_tensor``.  So ``qr_warmup_steps`` has no
                     effect.
        return_meta: When True, also return the :class:`CTMConvergenceInfo`
                     of ``ctm_tensor``, whose fields are Python scalars.

    Returns:
        The raw-array ``CTMEnvironment``, or ``(env, info)``.
    """
    from tenax.algorithms._ctm_env_dense import _as_dense_env
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.core.tensor import SymmetricTensor, Tensor

    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "ctm() returns a raw-array CTMEnvironment, so a SymmetricTensor site "
            "tensor would have to be densified. Call ctm_tensor(A, chi, ...) "
            "instead: it keeps the blocks and returns a CTMTensorEnv."
        )
    if not isinstance(A, Tensor):
        A = jnp.asarray(A)
        if A.ndim != 5 or len(set(A.shape[:4])) != 1:
            raise ValueError(
                "Expected a raw site tensor with 5 legs (u, d, l, r, s) and four "
                f"virtual legs of one dimension; got shape {A.shape}."
            )
        A = _wrap_as_dense_tensor(A)
    env, _eps, info = ctm_tensor(
        A,
        config.chi,
        max_iter=config.max_iter,
        conv_tol=config.conv_tol,
        renormalize=config.renormalize,
        projector_method="svd",
        recipe="2x2",
        min_iter=config.min_iter,
        return_meta=True,
    )
    env = _as_dense_env(env)
    if return_meta:
        return env, info
    return env


'''
    + RENORMALIZE,
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited src/tenax/algorithms/ipeps_ctm_convergence.py`.

- [ ] **Step 5: Run the tests to verify they pass (about 40 s)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py > "$W/t3_green.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t3_green.txt"; exit $rc
```

Expected: `rc=0` and `43 passed` (19 in the new file, 24 in the `min_iter` file).

- [ ] **Step 6: Lint and commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src/tenax/algorithms/ipeps_ctm_convergence.py tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py tests/conftest.py && uv run ruff format --check src/tenax/algorithms/ipeps_ctm_convergence.py tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py tests/conftest.py
```

Expected: `All checks passed!` and `4 files already formatted`.

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/ipeps_ctm_convergence.py tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py tests/conftest.py && git commit -F - <<'MSG'
feat(ctm): _ctm_via_2x2 runs ctm_tensor(recipe="2x2") with ctm()'s arguments

A private helper that checks and wraps a raw 5-leg site array, maps the
CTMConfig fields (min_iter included), always passes projector_method="svd"
and returns a raw-array CTMEnvironment.  ctm() will delegate to it;
fermionic_ctm and the fixture-only tests will call it without the
deprecation warning.  The new core test file pins it against Onsager's
C(1,0) on the Ising PEPS at beta=0.2 and 0.3, chi 8 to 32.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 4: `fermionic_ctm` raw branch

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `src/tenax/algorithms/fermionic_ipeps.py:619-655` (`fermionic_ctm`)
- Test: `tests/test_ctm_delegates_to_2x2.py`

**Interfaces:**
- Consumes: Task 3, `_ctm_via_2x2(A, config, *, return_meta=False)` and the test helpers `ONES` and `_c10_error(env, beta)`.
- Produces: the raw branch of `fermionic_ctm(A, config)` returns `_ctm_via_2x2(A, CTMConfig(chi=config.ctm_chi, max_iter=config.ctm_max_iter, conv_tol=config.ctm_conv_tol))`: a raw `CTMEnvironment`, with `min_iter=10` and no `DeprecationWarning`. The `Tensor` branch does not change.

- [ ] **Step 1: Write the failing tests**

```bash
# edit: Task 4, tests
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


edit(
    "tests/test_ctm_delegates_to_2x2.py",
    "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n",
    "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n"
    "from tenax.algorithms.fermionic_ipeps import FPEPSConfig, fermionic_ctm\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
cat >> tests/test_ctm_delegates_to_2x2.py <<'PYEOF'


@pytest.mark.parametrize("chi", [12, 16])
def test_fermionic_ctm_raw_branch_reproduces_onsager(chi):
    """The raw branch returns the 2x2 environment as a ``CTMEnvironment``,
    without ``ctm()``'s ``DeprecationWarning``.  It ran the old 1-site sweep
    before, which missed C(1,0) by 7.6e-4 at chi=12 and 4.3e-4 at chi=16."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        env = fermionic_ctm(
            jnp.asarray(ising_peps(0.2)),
            FPEPSConfig(ctm_chi=chi, ctm_max_iter=500, ctm_conv_tol=1e-12),
        )
    assert isinstance(env, CTMEnvironment)
    assert _c10_error(env, 0.2) < 1e-12


def test_fermionic_ctm_raw_branch_keeps_ctmconfig_min_iter(monkeypatch):
    """The raw branch builds a ``CTMConfig``, so it keeps ``min_iter=10``; a
    direct ``ctm_tensor`` call would use ``min_iter=1``."""
    import tenax.algorithms.ipeps_ctm_convergence as conv

    seen = []

    def spy(A, config, *, return_meta=False):
        seen.append(config)
        return "env"

    monkeypatch.setattr(conv, "_ctm_via_2x2", spy)
    out = fermionic_ctm(
        ONES, FPEPSConfig(ctm_chi=1, ctm_max_iter=40, ctm_conv_tol=1e-8)
    )
    (cfg,) = seen
    assert out == "env"
    assert (cfg.chi, cfg.max_iter, cfg.conv_tol, cfg.min_iter) == (1, 40, 1e-8, 10)
PYEOF
```

Expected: `edited tests/test_ctm_delegates_to_2x2.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py -k fermionic > "$W/t4_red.txt" 2>&1; echo "rc=$?"; tail -n 6 "$W/t4_red.txt"
```

Expected: `rc=1` and `3 failed, 19 deselected`. `test_fermionic_ctm_raw_branch_reproduces_onsager[12]` and `[16]` fail on the error assertion (about 7.6e-4 and 4.3e-4: the old sweep). `test_fermionic_ctm_raw_branch_keeps_ctmconfig_min_iter` fails with `ValueError: not enough values to unpack (expected 1, got 0)`: the old branch never calls the helper.

- [ ] **Step 3: Implement**

```bash
# edit: Task 4, implementation
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/fermionic_ipeps.py"
edit(
    F,
    "    Otherwise falls back to the dense bosonic CTM.\n",
    "    Otherwise the raw array runs the same 2x2 recipe through\n"
    "    ``_ctm_via_2x2`` (what ``ctm()`` runs, without its ``DeprecationWarning``)\n"
    "    and comes back as a raw-array ``CTMEnvironment``.\n",
)
edit(
    F,
    "    from tenax.algorithms.ipeps_ctm_convergence import ctm\n",
    "    from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2\n",
)
edit(F, "    return ctm(A, ctm_cfg)\n", "    return _ctm_via_2x2(A, ctm_cfg)\n")
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited src/tenax/algorithms/fermionic_ipeps.py`.

- [ ] **Step 4: Run the tests to verify they pass (about 40 s)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py > "$W/t4_green.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t4_green.txt"; exit $rc
```

Expected: `rc=0` and `22 passed`.

- [ ] **Step 5: Lint and commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src/tenax/algorithms/fermionic_ipeps.py tests/test_ctm_delegates_to_2x2.py && uv run ruff format --check src/tenax/algorithms/fermionic_ipeps.py tests/test_ctm_delegates_to_2x2.py
```

Expected: `All checks passed!` and `2 files already formatted`.

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/fermionic_ipeps.py tests/test_ctm_delegates_to_2x2.py && git commit -F - <<'MSG'
fix(fpeps): fermionic_ctm's raw branch runs the 2x2 recipe

The raw-array branch called ctm(), whose 1-site sweep reached no fixed
point: on the Ising PEPS at beta=0.2 it missed C(1,0) by 7.6e-4 at chi=12
and 4.3e-4 at chi=16.  It now calls _ctm_via_2x2, so it keeps returning
a raw CTMEnvironment and CTMConfig's min_iter=10, without the coming
deprecation warning of ctm().

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 5: Fixture-only tests build their environment with `_ctm_via_2x2`

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

This task changes no behaviour: `ctm()` is still the old sweep here. These call sites use `ctm()` only to build an environment for something else, and they need a raw `CTMEnvironment`. After Task 6 a `ctm()` call would trigger the `DeprecationWarning`, so they move to the helper now. No RED step: the check is that the call sites moved (Step 1 and Step 3) and that the 7 files still pass (Step 4).

**Files:**
- Modify: `tests/test_ipeps.py:18-25,218,255,280,372`, `tests/test_ipeps_core.py:36-39,229,260,334`, `tests/test_ipeps_excitations.py:15,60,958`, `tests/test_ipeps_excitations_gauge.py:38,81`, `tests/test_ipeps_rdm_gauge.py:53,85`, `tests/test_normalise_rdm_zero_grad.py:108,114`, `tests/test_rdm_gate_transpose_966.py:38,71`

**Interfaces:**
- Consumes: Task 3, `_ctm_via_2x2(A, config)`.
- Produces: no fixture-only test calls `ctm()`. What still calls it: the contract tests (`TestCTM` in `test_ipeps.py` and `test_ipeps_core.py`, `test_ipeps_rdm_gauge.py::test_ctm_accepts_a_complex_site_tensor`, `test_ctm_complex_site_tensors.py`, `test_ctm_convergence_meta.py`, `test_ctm_criterion_rank_blind_898.py`, `test_ctm_min_iter_contract_925_976.py`), and the warm-start and QR tests that Task 6 removes.

- [ ] **Step 1: Count the `ctm()` calls before the change**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
import ast
from pathlib import Path

for name in ["test_ipeps.py", "test_ipeps_core.py", "test_ipeps_rdm_gauge.py",
             "test_ipeps_excitations.py", "test_ipeps_excitations_gauge.py",
             "test_normalise_rdm_zero_grad.py", "test_rdm_gate_transpose_966.py"]:
    tree = ast.parse(Path("tests", name).read_text())
    calls = sum(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ctm"
                for n in ast.walk(tree))
    print(name, calls)
PY
```

Expected: `14`, `13`, `2`, `2`, `1`, `1`, `1`, in that order.

- [ ] **Step 2: Move the call sites**

```bash
# edit: Task 5
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


HELPER_IMPORT = "from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2\n"

# The contract tests in TestCTM keep calling ctm(); only the fixtures move.
F = "tests/test_ipeps.py"
edit(
    F,
    "    ctm_split,\n)\nfrom tenax.algorithms.ipeps_optimize import optimize_gs_ad\n",
    "    ctm_split,\n)\n" + HELPER_IMPORT
    + "from tenax.algorithms.ipeps_optimize import optimize_gs_ad\n",
)
for G in ("tests/test_ipeps.py", "tests/test_ipeps_core.py"):
    edit(
        G,
        "        config = CTMConfig(chi=4, max_iter=5)\n"
        "        env = ctm(A, config)\n        return A, env\n",
        "        config = CTMConfig(chi=4, max_iter=5)\n"
        "        env = _ctm_via_2x2(A, config)\n        return A, env\n",
    )
    edit(
        G,
        "        config = CTMConfig(chi=8, max_iter=20)\n"
        "        env = ctm(A, config)\n        return A, env, d\n",
        "        config = CTMConfig(chi=8, max_iter=20)\n"
        "        env = _ctm_via_2x2(A, config)\n        return A, env, d\n",
    )
    edit(
        G,
        "        config = CTMConfig(chi=4, max_iter=20)\n        env = ctm(A, config)\n",
        "        config = CTMConfig(chi=4, max_iter=20)\n"
        "        env = _ctm_via_2x2(A, config)\n",
    )
edit(
    F,
    "        env = ctm(A, CTMConfig(chi=8, max_iter=100, conv_tol=1e-10))\n",
    "        env = _ctm_via_2x2(A, CTMConfig(chi=8, max_iter=100, conv_tol=1e-10))\n",
)
edit(
    "tests/test_ipeps_core.py",
    "from tenax.algorithms.ipeps_ctm import (\n    _build_double_layer,\n    ctm,\n)\n",
    "from tenax.algorithms.ipeps_ctm import (\n    _build_double_layer,\n    ctm,\n)\n"
    + HELPER_IMPORT,
)

F = "tests/test_ipeps_excitations.py"
edit(F, "from tenax.algorithms.ipeps_ctm import ctm\n", HELPER_IMPORT)
edit(
    F,
    "    config = CTMConfig(chi=8, max_iter=40)\n    env = ctm(A, config)\n",
    "    config = CTMConfig(chi=8, max_iter=40)\n    env = _ctm_via_2x2(A, config)\n",
)
edit(
    F,
    "        env = ctm(A, CTMConfig(chi=8, max_iter=40))\n",
    "        env = _ctm_via_2x2(A, CTMConfig(chi=8, max_iter=40))\n",
)

STATE_ENV = "    env = ctm(A, CTMConfig(chi=8, max_iter=200, conv_tol=1e-12))\n"
STATE_ENV_NEW = "    env = _ctm_via_2x2(A, CTMConfig(chi=8, max_iter=200, conv_tol=1e-12))\n"
F = "tests/test_ipeps_excitations_gauge.py"
edit(F, "from tenax.algorithms.ipeps_ctm_convergence import ctm\n", HELPER_IMPORT)
edit(F, STATE_ENV, STATE_ENV_NEW)
# test_ctm_accepts_a_complex_site_tensor in this file stays on ctm().
F = "tests/test_ipeps_rdm_gauge.py"
edit(
    F,
    "from tenax.algorithms.ipeps_ctm_convergence import ctm\n",
    "from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2, ctm\n",
)
edit(F, STATE_ENV, STATE_ENV_NEW)

F = "tests/test_normalise_rdm_zero_grad.py"
edit(
    F,
    "    from tenax.algorithms.ipeps_ctm import ctm\n",
    "    from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2\n",
)
edit(
    F,
    "    env = ctm(A, CTMConfig(chi=8, max_iter=40))\n",
    "    env = _ctm_via_2x2(A, CTMConfig(chi=8, max_iter=40))\n",
)

F = "tests/test_rdm_gate_transpose_966.py"
edit(F, "from tenax.algorithms.ipeps_ctm import ctm\n", HELPER_IMPORT)
edit(
    F,
    "    env = ctm(A, CTMConfig(chi=4, max_iter=50, conv_tol=1e-12))\n",
    "    env = _ctm_via_2x2(A, CTMConfig(chi=4, max_iter=50, conv_tol=1e-12))\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: seven `edited tests/...` lines.

- [ ] **Step 3: Count the `ctm()` calls after the change**

Run the command of Step 1 again. Expected: `10`, `10`, `1`, `0`, `0`, `0`, `0`. In each of `test_ipeps.py` and `test_ipeps_core.py` the 10 calls are the 8 of the `TestCTM` contract tests (2 of them in `test_ctm_with_initial_env`) and the 2 of `TestQRProjectors`. In `test_ipeps_rdm_gauge.py` the 1 call is `test_ctm_accepts_a_complex_site_tensor`.

- [ ] **Step 4: Run the 7 files (background, about 9 min)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ipeps_excitations.py tests/test_ipeps_excitations_gauge.py tests/test_ipeps_rdm_gauge.py tests/test_normalise_rdm_zero_grad.py tests/test_rdm_gate_transpose_966.py > "$W/t5.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t5.txt"; exit $rc
```

Expected: `rc=0`; the summary line has no `failed` and no `error` (280 tests collected: 98 + 28 + 54 + 65 + 19 + 8 + 8). A failure here is a finding about the fixture's new environment: investigate it, and do not change an assertion to make it pass.

- [ ] **Step 5: Lint and commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check tests && uv run ruff format --check tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ipeps_excitations.py tests/test_ipeps_excitations_gauge.py tests/test_ipeps_rdm_gauge.py tests/test_normalise_rdm_zero_grad.py tests/test_rdm_gate_transpose_966.py
```

Expected: `All checks passed!` and `7 files already formatted`.

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ipeps_excitations.py tests/test_ipeps_excitations_gauge.py tests/test_ipeps_rdm_gauge.py tests/test_normalise_rdm_zero_grad.py tests/test_rdm_gate_transpose_966.py && git commit -F - <<'MSG'
test: build fixture environments with _ctm_via_2x2, not ctm()

These call sites use ctm() only to get a raw CTMEnvironment for another
test.  They move to the helper, so they keep working without the coming
DeprecationWarning of ctm().  The contract tests of ctm() stay on it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 6: `ctm()` delegates and is deprecated

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ctm_convergence.py` — imports (`:19`); `ctm()` (`:97-226`)
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` — the JAX-array paragraph of the `CTMConvergenceInfo` docstring (`:1018-1022`)
- Modify: `pyproject.toml` — `filterwarnings` (`:130-164`)
- Test: `tests/test_ctm_delegates_to_2x2.py`, `tests/test_ipeps.py`, `tests/test_ipeps_core.py`, `tests/test_ctm_convergence_meta.py`, `tests/test_ctm_min_iter_contract_925_976.py`

**Interfaces:**
- Consumes: Task 3, `_ctm_via_2x2` and the test file's `ENTRY_POINTS`, `ONES`, `_random_site`; Task 1, `ctm_tensor(min_iter=)`.
- Produces: `ctm(A, config, initial_env=None, *, return_meta=False)`. It warns a `DeprecationWarning` whose message starts `ctm is deprecated` and names `ctm_tensor` and `jax.jit`; raises `ValueError` (message names `initial_env`) when `initial_env is not None`; warns a `UserWarning` (message names `projector_method`) when `config.projector_method != "svd"`; then returns `_ctm_via_2x2(A, config, return_meta=return_meta)`. `pyproject.toml` ignores `ctm is deprecated` warnings in the suite.

- [ ] **Step 1: Write the failing tests**

```bash
# edit: Task 6, tests
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "tests/test_ctm_delegates_to_2x2.py"
edit(
    F,
    "from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo, _ctm_via_2x2\n",
    "from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo, _ctm_via_2x2, ctm\n",
)
edit(
    F,
    'ENTRY_POINTS = [pytest.param(_ctm_via_2x2, id="helper")]\n',
    "ENTRY_POINTS = [\n"
    '    pytest.param(_ctm_via_2x2, id="helper"),\n'
    '    pytest.param(ctm, id="ctm"),\n'
    "]\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
cat >> tests/test_ctm_delegates_to_2x2.py <<'PYEOF'


def test_ctm_warns_that_it_is_deprecated():
    """Exactly one warning, naming the replacement and the lost ``jax.jit``."""
    with pytest.warns(DeprecationWarning, match="^ctm is deprecated") as record:
        ctm(ONES, CTMConfig(chi=1, max_iter=5))
    messages = [str(w.message) for w in record]
    deprecations = [m for m in messages if m.startswith("ctm is deprecated")]
    assert len(deprecations) == 1
    assert "ctm_tensor" in deprecations[0]
    assert "jax.jit" in deprecations[0]


def test_ctm_rejects_initial_env():
    """``ctm_tensor`` has no warm start, and a silent cold start would look the
    same to a caller who asked for one."""
    env = _ctm_via_2x2(ONES, CTMConfig(chi=1, max_iter=5))
    with pytest.raises(ValueError, match="initial_env"):
        ctm(ONES, CTMConfig(chi=1, max_iter=5), initial_env=env)


@pytest.mark.parametrize("method", ["qr", "eigh"])
def test_ctm_warns_that_projector_method_is_ignored(method):
    """The 2x2 recipe always uses Fishman SVD: the environment is the one
    ``"svd"`` gives, and the caller is told."""
    A = _random_site()
    with pytest.warns(UserWarning, match="projector_method"):
        env = ctm(A, CTMConfig(chi=4, max_iter=5, projector_method=method))
    ref = _ctm_via_2x2(A, CTMConfig(chi=4, max_iter=5))
    for name in CTMEnvironment._fields:
        np.testing.assert_array_equal(getattr(env, name), getattr(ref, name))


def test_ctm_default_config_gives_no_user_warning():
    """The ``projector_method`` warning fires only when it applies."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctm(_random_site(), CTMConfig(chi=4, max_iter=5))
    assert not [w for w in caught if "projector_method" in str(w.message)]


def test_ctm_keeps_the_imaginary_part_of_a_complex_state():
    """#842: complex site tensors keep working.  The edges of this state have
    imaginary parts up to 0.57 (measured); a ``.real`` on the way would zero
    them."""
    key = jax.random.PRNGKey(0)
    real = jax.random.normal(key, (2, 2, 2, 2, 2))
    imag = jax.random.normal(jax.random.fold_in(key, 1), (2, 2, 2, 2, 2))
    A = (real + 1j * imag).astype(jnp.complex128)
    env = ctm(A, CTMConfig(chi=4, max_iter=6, conv_tol=1e-10))
    assert jnp.iscomplexobj(env.T1)
    edges = (env.T1, env.T2, env.T3, env.T4)
    assert max(float(jnp.max(jnp.abs(t.imag))) for t in edges) > 1e-3


def test_ctm_return_meta_matches_the_helper():
    """``ctm()`` is the helper plus the warnings: same environment, same info."""
    A = jnp.asarray(ising_peps(0.2))
    cfg = CTMConfig(chi=8, max_iter=500, conv_tol=1e-12)
    env, info = ctm(A, cfg, return_meta=True)
    ref_env, ref_info = _ctm_via_2x2(A, cfg, return_meta=True)
    assert info == ref_info
    for name in CTMEnvironment._fields:
        np.testing.assert_array_equal(getattr(env, name), getattr(ref_env, name))
PYEOF
```

Expected: `edited tests/test_ctm_delegates_to_2x2.py`.

- [ ] **Step 2: Run the tests to verify they fail (about 50 s)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py > "$W/t6_red.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t6_red.txt"; exit $rc
```

Expected: `rc=1` and `19 failed, 25 passed`. All 22 helper and `fermionic_ctm` tests pass. The 19 failures are all on the old `ctm()`:
- `test_ising_c10_matches_onsager[beta0.2-chi12-ctm]`, `[beta0.2-chi16-ctm]`, `[beta0.2-chi32-ctm]` and the four `[beta0.3-*-ctm]` cases (not converged);
- the four `test_rejects_a_bad_raw_shape[*-ctm]` cases (the 3-leg array runs; the others raise einsum or reshape errors that do not match);
- `test_refuses_a_symmetric_tensor[ctm]`, `test_numpy_and_dense_tensor_inputs_match_the_jax_array[ctm]`, `test_returns_a_raw_environment_and_python_scalars[ctm]`;
- `test_ctm_warns_that_it_is_deprecated`, `test_ctm_rejects_initial_env`, both `test_ctm_warns_that_projector_method_is_ignored` cases, `test_ctm_return_meta_matches_the_helper`.

Three `ctm` tests pass already, as guards: `test_ising_c10_matches_onsager[beta0.2-chi8-ctm]` (the old sweep converged there, at 318 sweeps), `test_ctm_default_config_gives_no_user_warning` and `test_ctm_keeps_the_imaginary_part_of_a_complex_state`.

- [ ] **Step 3: Implement**

```bash
# edit: Task 6, implementation
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/ipeps_ctm_convergence.py"
edit(
    F,
    "]\n\n\nimport jax\nimport jax.numpy as jnp\n",
    "]\n\n\nimport warnings\n\nimport jax\nimport jax.numpy as jnp\n",
)
edit(
    F,
    "def ctm(\n",
    '''\
def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
    *,
    return_meta: bool = False,
) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]:
    """Compute the CTM environment of a 1x1 PEPS with the 2x2 recipe.

    .. deprecated:: 0.9.3
        Use :func:`ctm_tensor`.  ``ctm()`` now runs ``ctm_tensor(recipe="2x2")``
        and returns its environment as the raw-array :class:`CTMEnvironment`
        that ``ctm()`` always returned.  The 1-site sweep it ran before reached
        no fixed point: on the Ising PEPS at beta=0.2 it stopped at chi=12, 16
        and 32 without converging, with C(1,0) errors of 7.6e-4, 4.3e-4 and
        4.3e-4, where the 2x2 recipe converges with errors below 4e-16.  So
        environments from ``ctm()`` changed in 0.9.3, and ``ctm()`` no longer
        runs under ``jax.jit``, because ``ctm_tensor`` is a Python loop.

    Args:
        A:           Raw site array with 5 legs ``(u, d, l, r, s)`` and four
                     virtual legs of one dimension, or a ``DenseTensor``.  Any
                     other shape raises ``ValueError``.  A ``SymmetricTensor``
                     raises ``TypeError``: call :func:`ctm_tensor`, which keeps
                     the blocks.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` apply.  A ``projector_method`` other than
                     ``"svd"`` is ignored with a ``UserWarning``, because the
                     2x2 recipe always uses Fishman SVD; ``qr_warmup_steps`` is
                     ignored too.
        initial_env: Must be ``None``.  ``ctm_tensor`` has no warm start, so
                     anything else raises ``ValueError`` rather than start cold
                     in silence.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or exhausted
                     ``max_iter`` (#839).  Its fields are Python scalars.
                     Keyword-only and off by default so the return arity of
                     this public entry point is unchanged.

    Returns:
        The CTMEnvironment, or ``(env, info)`` when ``return_meta=True``.
        **The environment is not necessarily converged** -- that is what
        ``info`` is for.
    """
    warnings.warn(
        "ctm is deprecated and will be removed in a later release: call "
        "ctm_tensor(A, chi, ...) on a DenseTensor or SymmetricTensor site tensor "
        "with legs (u, d, l, r, phys). ctm now runs ctm_tensor(recipe='2x2'), so "
        "its environments changed in 0.9.3 (the old 1-site sweep reached no "
        "fixed point), and it no longer runs under jax.jit.",
        DeprecationWarning,
        stacklevel=2,
    )
    if initial_env is not None:
        raise ValueError(
            "ctm() no longer takes initial_env: it runs ctm_tensor(recipe='2x2'), "
            "which has no warm start. Leave initial_env as None."
        )
    if config.projector_method != "svd":
        warnings.warn(
            f"ctm() ignores projector_method={config.projector_method!r}: the 2x2 "
            "recipe always uses Fishman SVD projectors.",
            UserWarning,
            stacklevel=2,
        )
    return _ctm_via_2x2(A, config, return_meta=return_meta)


''',
    end="def _ctm_via_2x2(\n",
)
edit(
    "src/tenax/algorithms/_ctm_tensor_convergence.py",
    """\
    ``converged`` and ``n_iter`` come straight out of a ``lax.while_loop``
    carry for :func:`ctm` / :func:`ctm_2site`, so they are **JAX arrays**, not
    Python scalars.  That keeps the entry points jittable; call ``bool(...)``
    / ``int(...)`` at the point of use.  :func:`ctm_split` runs a Python loop
    and returns Python scalars.
""",
    """\
    ``converged`` and ``n_iter`` come straight out of a ``lax.while_loop``
    carry for :func:`ctm_2site`, so they are **JAX arrays**, not Python
    scalars.  That keeps it jittable; call ``bool(...)`` / ``int(...)`` at the
    point of use.  :func:`ctm_split` and :func:`ctm_tensor` run a Python loop
    and return Python scalars, and so does :func:`ctm`, which runs
    :func:`ctm_tensor` since 0.9.3.
""",
)
edit(
    "pyproject.toml",
    '    "ignore:compute_excitations is deprecated:DeprecationWarning",\n',
    '''\
    "ignore:compute_excitations is deprecated:DeprecationWarning",
    # ``ctm()`` is deprecated in favour of ``ctm_tensor`` (0.9.3): it now runs
    # the 2x2 recipe.  Its contract tests still call it on purpose;
    # test_ctm_warns_that_it_is_deprecated (tests/test_ctm_delegates_to_2x2.py)
    # un-suppresses it via ``pytest.warns``.
    "ignore:ctm is deprecated:DeprecationWarning",
''',
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited` lines for `ipeps_ctm_convergence.py`, `_ctm_tensor_convergence.py` and `pyproject.toml`. The old body of `ctm()` (its double layer, QR warm-up and `lax.while_loop`) is gone; `_ctm_sweep` is now unused and Task 7 deletes it.

- [ ] **Step 4: Examine the contract tests against the new `ctm()` before changing them**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q -rfE \
  tests/test_ipeps.py::TestCTM::test_ctm_with_initial_env \
  tests/test_ipeps.py::TestCTM::test_ctm_edge_tensors_change \
  tests/test_ipeps.py::TestQRProjectors::test_qr_ctm_converges \
  tests/test_ipeps.py::TestQRProjectors::test_qr_energy_is_finite \
  tests/test_ipeps_core.py::TestCTM::test_ctm_with_initial_env \
  tests/test_ipeps_core.py::TestCTM::test_ctm_edge_tensors_change \
  tests/test_ipeps_core.py::TestQRProjectors::test_qr_ctm_converges \
  tests/test_ipeps_core.py::TestQRProjectors::test_qr_energy_is_finite \
  tests/test_ctm_convergence_meta.py::test_ctm_reports_status \
  tests/test_ctm_min_iter_contract_925_976.py::test_ctm_counts_qr_warmup_in_n_iter \
  tests/test_ctm_min_iter_contract_925_976.py::test_ctm_warmup_counts_toward_min_iter \
  > "$W/t6_examine.txt" 2>&1; echo "rc=$?"; tail -n 12 "$W/t6_examine.txt"
```

Expected: `rc=1` and `6 failed, 5 passed`. Each result and what Step 5 does about it:
- The 4 warm-start tests fail with `ValueError: ctm() no longer takes initial_env` (spec: replaced by `test_ctm_rejects_initial_env`). Removed.
- The 4 QR tests pass, but vacuously: `ctm()` warns that it ignores `"qr"` and runs SVD, so they no longer test QR (spec: QR pinned on `ctm()`). Removed, together with the `heisenberg_gate` fixture that only they use. `test_qr_backward_compat` stays: it tests `CTMConfig` defaults.
- `test_ctm_reports_status` fails: the 2x2 recipe converges on `su_state` at sweep 16 (diff 2.2e-13), so 20 sweeps no longer starve it. This is a finding about the state, not about `ctm_tensor`. The starved budget becomes 10 sweeps (diff 7.7e-9 there), with the same assertions.
- `test_ctm_counts_qr_warmup_in_n_iter` fails: `ctm()` runs no warm-up now. `test_ctm_warmup_counts_toward_min_iter` passes for the wrong reason (`n_iter=8` comes from `min_iter`; no warm-up ran). Both removed (spec: `ctm_2site` and `ctm_split` have no QR and no warm-up, so they cannot be retargeted). `test_ctm_tensor_min_iter_counts_the_warmup_against_the_callers_budget` (Task 1) keeps the warm-up rule pinned on `ctm_tensor`.

If any result differs from this list, stop and investigate before Step 5.

- [ ] **Step 5: Change the contract tests**

```bash
# edit: Task 6, contract tests
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


QR_DOC = '''\
    """``CTMConfig``'s QR projector option.

    ``ctm()`` ran QR sweeps here until 0.9.3.  It now runs the 2x2 recipe,
    which always uses Fishman SVD and ignores ``projector_method`` with a
    ``UserWarning`` (``tests/test_ctm_delegates_to_2x2.py``), so only the
    config default is left to pin here.
    """

'''
for F in ("tests/test_ipeps.py", "tests/test_ipeps_core.py"):
    edit(
        F,
        "    def test_ctm_with_initial_env(self, small_peps_tensor):\n",
        "",
        end="    def test_ctm_no_renormalize(self, small_peps_tensor):\n",
    )
    edit(
        F,
        "    def test_ctm_edge_tensors_change(self, small_peps_tensor):\n",
        "\n",
        end="class TestComputeEnergyCTM:\n",
    )
    edit(
        F,
        '    """Tests for QR-based CTMRG projectors (Phase 1)."""\n',
        QR_DOC,
        end="    def test_qr_backward_compat(self):\n",
    )
edit(
    "tests/test_ipeps.py",
    "    def test_qr_ctm_converges(self):\n",
    "\n",
    end="class TestOptimizeGsAd2Site:\n",
)
edit("tests/test_ipeps_core.py", "\n    def test_qr_ctm_converges(self):\n", "", end="")
edit(
    "tests/test_ipeps_core.py",
    "* QR-projector smoke correctness.\n",
    "* The QR-projector config default.\n",
)

F = "tests/test_ctm_convergence_meta.py"
edit(
    F,
    """\
    Deliberately the *unconverged* tau=2 simple-update state (40 steps at
    dt=0.05): its CTM provably does not converge at chi=6 (#838), which is what
    makes it a usable negative case here.  Do not "improve" it to a longer
    imaginary time -- a state whose CTM converges cannot exercise the
    ``converged=False`` branch.
""",
    """\
    Deliberately the *unconverged* tau=2 simple-update state (40 steps at
    dt=0.05): its CTM provably does not converge at chi=6 (#838), which is what
    makes it a usable negative case here.  That holds for ``ctm_2site``; the
    2x2 recipe that ``ctm()`` runs since 0.9.3 converges on it in 16 sweeps at
    chi=6, so ``test_ctm_reports_status`` starves its budget instead.  Do not
    "improve" the state to a longer imaginary time -- a state whose CTM
    converges cannot exercise the ``converged=False`` branch.
""",
)
edit(
    F,
    """\
    starved = ctm(A, CTMConfig(chi=6, max_iter=20, conv_tol=1e-12), return_meta=True)[1]
    assert not bool(starved.converged)
    assert int(starved.n_iter) == 20
""",
    """\
    # 10 sweeps, not 20: since 0.9.3 ``ctm()`` runs the 2x2 recipe, which
    # converges on this state at sweep 16 (diff 2.2e-13).  At sweep 10 the diff
    # is 7.7e-9, so the budget still runs out.
    starved = ctm(A, CTMConfig(chi=6, max_iter=10, conv_tol=1e-12), return_meta=True)[1]
    assert not bool(starved.converged)
    assert int(starved.n_iter) == 10
""",
)
edit(
    F,
    '''\
    """``ctm`` / ``ctm_2site`` keep the flag as JAX arrays to stay jittable.

    Pins the documented contract: the entry points must remain usable under
    ``jit``, which forbids converting the carry to Python scalars internally.
    """
''',
    '''\
    """``ctm_2site`` keeps the flag as JAX arrays to stay jittable.

    Pins the documented contract: the entry point must remain usable under
    ``jit``, which forbids converting the carry to Python scalars internally.
    ``ctm()`` lost this in 0.9.3: it runs ``ctm_tensor``, a Python loop.
    """
''',
)

F = "tests/test_ctm_min_iter_contract_925_976.py"
edit(
    F,
    "def test_ctm_counts_qr_warmup_in_n_iter():\n",
    "",
    end="def test_ctm_tensor_default_min_iter_keeps_the_early_exit():\n",
)
edit(
    F,
    "same cases.\n",
    'same cases.  Since 0.9.3 ``ctm`` runs ``ctm_tensor(recipe="2x2")`` with\n'
    '``projector_method="svd"``, so it runs no QR warm-up: its two warm-up tests\n'
    "went with it, and\n"
    "``test_ctm_tensor_min_iter_counts_the_warmup_against_the_callers_budget``\n"
    "pins the warm-up rule on ``ctm_tensor``.\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
uv run ruff format tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ctm_convergence_meta.py tests/test_ctm_min_iter_contract_925_976.py
```

Expected: four `edited tests/...` lines, then `4 files left unchanged` from ruff format (the cuts keep two blank lines between top-level definitions).

- [ ] **Step 6: Run the tests to verify they pass (background)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_delegates_to_2x2.py tests/test_ctm_min_iter_contract_925_976.py tests/test_ctm_convergence_meta.py > "$W/t6_green_a.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t6_green_a.txt"; exit $rc
```

Expected: `rc=0` and `79 passed` (44 + 22 + 13).

Then the files whose contract tests call the new `ctm()`, in the background (about 5 min):

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ctm_complex_site_tensors.py tests/test_ctm_criterion_rank_blind_898.py tests/test_ipeps_rdm_gauge.py > "$W/t6_green_b.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t6_green_b.txt"; exit $rc
```

Expected: `rc=0`; no `failed` and no `error` in the summary line (94 + 24 + 7 + 54 + 19 = 198 tests collected). A failure in `test_ctm_complex_site_tensors.py`, `test_ctm_criterion_rank_blind_898.py` or `TestCTM` is a finding about `ctm_tensor` (spec): investigate before changing anything.

- [ ] **Step 7: Lint and commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src tests && uv run ruff format --check src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_ctm_delegates_to_2x2.py tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ctm_convergence_meta.py tests/test_ctm_min_iter_contract_925_976.py
```

Expected: `All checks passed!` and `7 files already formatted`.

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/_ctm_tensor_convergence.py pyproject.toml tests/test_ctm_delegates_to_2x2.py tests/test_ipeps.py tests/test_ipeps_core.py tests/test_ctm_convergence_meta.py tests/test_ctm_min_iter_contract_925_976.py && git commit -F - <<'MSG'
fix(ctm): ctm() runs the 2x2 recipe and is deprecated in favour of ctm_tensor

The public ctm() ran a raw-array 1-site sweep with no fixed point and
returned wrong environments without a warning: on the Ising PEPS at
beta=0.2 it stopped at chi=12, 16 and 32 with C(1,0) errors of 7.6e-4,
4.3e-4 and 4.3e-4.  It keeps its signature and now returns
_ctm_via_2x2(A, config), with a DeprecationWarning naming ctm_tensor.
initial_env raises ValueError (ctm_tensor has no warm start), a
projector_method other than "svd" warns that it is ignored, the info
fields are Python scalars, and ctm() no longer runs under jax.jit.

Contract tests: the warm-start and QR tests on ctm() are removed, and
the starved budget of test_ctm_reports_status drops from 20 to 10
sweeps, because the 2x2 recipe converges on su_state at sweep 16.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 7: Delete the legacy 1-site sweep

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each pytest run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ctm_convergence.py` — `__all__`, the moves import (`:45-55`), `_ctm_sweep` (`:60-74`)
- Modify: `src/tenax/algorithms/ipeps_ctm_moves.py` — `__all__` (`:5-20`), `_ctm_move_qr` (`:79-115`), `_ctm_move` (`:117-127`), the four 1-site moves (`:130-245`)
- Modify: `src/tenax/algorithms/ipeps_ctm.py` — six re-export blocks
- Modify: `tests/test_ctm_tensor.py:25-30`

**Interfaces:**
- Consumes: Task 6 — nothing calls `_ctm_sweep` any more.
- Produces: `_ctm_move(C1g, C2g, Tg, chi)`, without `projector_method`, returning `_ctm_move_eigh(C1g, C2g, Tg, chi)` for the 2-site moves. These names no longer exist in `ipeps_ctm_convergence`, `ipeps_ctm_moves` or `ipeps_ctm`: `_ctm_sweep`, `_ctm_move_qr`, `_ctm_left_move`, `_ctm_right_move`, `_ctm_top_move`, `_ctm_bottom_move`. What stays: `_ctm_move_eigh`, the 2-site and split moves, `_initialize_ctm_env`, `_build_double_layer`, and `_ctm_python_loop._ctm_sweep` (an unrelated function with the same name).

- [ ] **Step 1: Write the absence check and watch it fail**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && cat > "$W/t7_absent.py" <<'PYEOF'
"""Names that only the legacy 1-site sweep reached must be gone."""

import inspect

import tenax.algorithms.ipeps_ctm as shim
import tenax.algorithms.ipeps_ctm_convergence as conv
import tenax.algorithms.ipeps_ctm_moves as moves

ONE_SITE = ["_ctm_left_move", "_ctm_right_move", "_ctm_top_move", "_ctm_bottom_move"]
GONE = {
    conv: ["_ctm_sweep", *ONE_SITE],
    moves: ["_ctm_move_qr", *ONE_SITE],
    shim: ["_ctm_sweep", "_ctm_move_qr", *ONE_SITE],
}
present = [f"{m.__name__}.{n}" for m, names in GONE.items() for n in names if hasattr(m, n)]
if "projector_method" in inspect.signature(moves._ctm_move).parameters:
    present.append("_ctm_move(projector_method=)")
print(len(present), "still present:", present)
raise SystemExit(1 if present else 0)
PYEOF
uv run python "$W/t7_absent.py"; echo "rc=$?"
```

Expected: `17 still present: [...]` and `rc=1`.

- [ ] **Step 2: Delete**

```bash
# edit: Task 7
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


F = "src/tenax/algorithms/ipeps_ctm_convergence.py"
edit(F, '    "_ctm_sweep",\n', "")
edit(
    F,
    "    _ctm_bottom_move,\n    _ctm_bottom_move_2site,\n"
    "    _ctm_left_move,\n    _ctm_left_move_2site,\n"
    "    _ctm_right_move,\n    _ctm_right_move_2site,\n"
    "    _ctm_top_move,\n    _ctm_top_move_2site,\n",
    "    _ctm_bottom_move_2site,\n    _ctm_left_move_2site,\n"
    "    _ctm_right_move_2site,\n    _ctm_top_move_2site,\n",
)
# From the second blank line after the imports: a comment block follows them
# now, and ruff's isort wants one blank line before it, not two.
edit(F, "\ndef _ctm_sweep(\n", "", end="# ``_ctm_sv_diff`` is imported, not defined here.\n")

F = "src/tenax/algorithms/ipeps_ctm_moves.py"
edit(
    F,
    '    "_ctm_move_eigh",\n    "_ctm_move_qr",\n    "_ctm_move",\n'
    '    "_ctm_left_move",\n    "_ctm_right_move",\n'
    '    "_ctm_top_move",\n    "_ctm_bottom_move",\n',
    '    "_ctm_move_eigh",\n    "_ctm_move",\n',
)
edit(
    F,
    "def _ctm_move_qr(\n",
    '''\
def _ctm_move(
    C1g: jax.Array,
    C2g: jax.Array,
    Tg: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """CTM truncation with the eigh projector; the 2-site moves call this."""
    return _ctm_move_eigh(C1g, C2g, Tg, chi)


''',
    end="def _ctm_left_move_2site(\n",
)

F = "src/tenax/algorithms/ipeps_ctm.py"
CONV = "from tenax.algorithms.ipeps_ctm_convergence import (\n"
MOVES = "from tenax.algorithms.ipeps_ctm_moves import (\n"
edit(F, CONV + "    _ctm_sweep as _ctm_sweep,\n)\n", "")
edit(
    F,
    "from tenax.algorithms.ipeps_ctm_moves import (  # noqa: F401\n"
    "    _ctm_bottom_move as _ctm_bottom_move,\n)\n"
    + MOVES + "    _ctm_bottom_move_2site as _ctm_bottom_move_2site,\n)\n",
    "from tenax.algorithms.ipeps_ctm_moves import (  # noqa: F401\n"
    "    _ctm_bottom_move_2site as _ctm_bottom_move_2site,\n)\n",
)
for name in ("_ctm_left_move", "_ctm_move_qr", "_ctm_right_move", "_ctm_top_move"):
    edit(F, MOVES + f"    {name} as {name},\n)\n", "")

edit(
    "tests/test_ctm_tensor.py",
    "    _ctm_2site_sweep,\n    _ctm_sweep,\n    _initialize_ctm_env,\n",
    "    _ctm_2site_sweep,\n    _initialize_ctm_env,\n",
)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited` lines for the three source files and `tests/test_ctm_tensor.py`. The skip reason of `TestConvergence::test_ctm_tensor_dense_matches_ctm` in `tests/test_ctm_tensor.py:264` still names `_ctm_sweep` as history; leave it.

- [ ] **Step 3: Run the absence check to verify it passes**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uv run python "$W/t7_absent.py"; echo "rc=$?"
```

Expected: `0 still present: []` and `rc=0`.

- [ ] **Step 4: Lint**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src tests && uv run ruff format --check src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/ipeps_ctm_moves.py src/tenax/algorithms/ipeps_ctm.py tests/test_ctm_tensor.py
```

Expected: `All checks passed!` (an unused import would show as F401) and `4 files already formatted`.

- [ ] **Step 5: Run the tests that reach `_ctm_move` and the shim (background, about 3 min)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q tests/test_ctm_tensor.py tests/test_ctm_delegates_to_2x2.py tests/test_ctm_convergence_meta.py tests/test_ctm_complex_site_tensors.py tests/test_ipeps_core.py > "$W/t7.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t7.txt"; exit $rc
```

Expected: `rc=0`; no `failed` and no `error` in the summary line. `ctm_2site` (in `test_ctm_convergence_meta.py` and `test_ctm_complex_site_tensors.py`) reaches `_ctm_move` through the 2-site moves.

- [ ] **Step 6: Commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add src/tenax/algorithms/ipeps_ctm_convergence.py src/tenax/algorithms/ipeps_ctm_moves.py src/tenax/algorithms/ipeps_ctm.py tests/test_ctm_tensor.py && git commit -F - <<'MSG'
refactor(ctm): delete the legacy 1-site sweep and its QR move

Only the old ctm() reached _ctm_sweep, the four 1-site moves and
_ctm_move_qr.  _ctm_move keeps its eigh body for the 2-site moves and
loses projector_method; the shim drops the six re-exports.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 8: Docs and examples

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`.

**Files:**
- Modify: `docs/guide/algorithms/ipeps.md:152-161` ("Using CTM standalone")
- Modify: `docs/guide/algorithms/ctm.md` ("Checking whether the CTM actually converged", before the `ctm_tensor_2site` paragraph)
- Modify: `examples/kagome_xxz_pess.py:30-36,599-608`
- Modify: `examples/kagome_xxz_spin1_pess.py:28-30,329,343-346`

**Interfaces:**
- Consumes: the deprecation text and measured numbers of Task 6; Task 1, `min_iter`.
- Produces: nothing code-facing. Guide example signatures: `ctm_tensor(A, chi=20, max_iter=100, return_meta=True) -> (env, eps_T, info)` and `compute_energy_ctm_tensor(A, env, hamiltonian_gate, d=2)`.

- [ ] **Step 1: Write the guide text**

````bash
# edit: Task 8, guide section
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && mkdir -p "$W" && cat > "$W/ipeps_md_section.md" <<'MDEOF'
## Using CTM standalone

`ctm_tensor()` computes the environment of an existing site tensor `A`: a
`DenseTensor` or `SymmetricTensor` with 5 legs `(u, d, l, r, phys)`, such as
the tensors `ipeps()` and `optimize_gs_ad()` return. A raw `(D, D, D, D, d)`
array becomes one with `DenseTensor(data, indices)`, where the five
`TensorIndex` legs are labelled `u`, `d`, `l`, `r`, `phys` with flows `OUT`,
`IN`, `OUT`, `IN`, `IN`.

```python
from tenax import compute_energy_ctm_tensor, ctm_tensor

env, _eps_T, info = ctm_tensor(A, chi=20, max_iter=100, return_meta=True)
if not info.converged:
    print(f"not converged after {info.n_iter} sweeps, diff {info.diff:.2e}")
E = compute_energy_ctm_tensor(A, env, hamiltonian_gate, d=2)
```

```{warning}
`ctm()` is deprecated since 0.9.3 and emits a `DeprecationWarning`. It now runs
`ctm_tensor(recipe="2x2")` and returns the raw-array `CTMEnvironment` it always
returned, so its environments changed in 0.9.3. The 1-site sweep it ran before
reached no fixed point: on the Ising PEPS at beta=0.2, where C(1,0) has
Onsager's exact value, it stopped at chi=12, 16 and 32 without converging, with
C(1,0) errors of 7.6e-4, 4.3e-4 and 4.3e-4. The 2x2 recipe converges at the
same chi with errors below 4e-16. Three behaviours differ from before:

- `ctm()` does not run under `jax.jit`, because `ctm_tensor` is a Python loop.
- `initial_env` raises `ValueError`, because `ctm_tensor` has no warm start.
- A `CTMConfig.projector_method` other than `"svd"` is ignored with a
  `UserWarning`, because the 2x2 recipe always uses Fishman SVD.
```

MDEOF
````

- [ ] **Step 2: Apply the docs and examples changes**

```bash
# edit: Task 8, docs and examples
cd /home/yjkao/tenax-ctm-cleanup && uv run python - <<'PY'
from pathlib import Path

ROOT, texts = Path("/home/yjkao/tenax-ctm-cleanup"), {}
W = ROOT / ".superpowers/sdd/2026-11-09-ctm-delegate-to-2x2"


def edit(rel, old, new, end=None):
    text = texts.get(rel) or (ROOT / rel).read_text()
    for marker in (old, end):
        if marker:
            n = text.count(marker)
            assert n == 1, f"{rel}: {n} matches for {marker[:70]!r}"
    i = text.index(old)
    j = i + len(old) if end is None else (text.index(end) if end else len(text))
    assert j >= i, f"{rel}: the end marker comes before {old[:40]!r}"
    texts[rel] = text[:i] + new + text[j:]


edit(
    "docs/guide/algorithms/ipeps.md",
    "## Using CTM standalone\n",
    (W / "ipeps_md_section.md").read_text(),
    end="## 2-site checkerboard unit cell\n",
)
NEXT = "`ctm_tensor_2site` warns instead when the environment is not converged (or\n"
edit(
    "docs/guide/algorithms/ctm.md",
    NEXT,
    "`ctm_tensor` also takes a keyword-only `min_iter` (default 1). With it,\n"
    "`converged` is set only after `min(min_iter, max_iter)` sweeps, QR warm-up\n"
    "sweeps included. The `CTMConfig`-based entry points apply the same rule with\n"
    "`CTMConfig.min_iter` (default 10), and `ctm()` passes that value on.\n"
    "\n" + NEXT,
)

WRAP = "from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor  # noqa: E402\n"
OLD_BODY = (
    "    config = CTMConfig(chi=chi, max_iter=100)\n"
    "    env = ctm(A, config)\n"
    "\n"
    "    E = compute_energy_ctm(A, env, jnp.array(H_eff_4leg), d_eff)\n"
)
NEW_BODY = (
    "    A_t = _wrap_as_dense_tensor(A)\n"
    "    env, _ = ctm_tensor(A_t, chi, max_iter=100)\n"
    "\n"
    "    E = compute_energy_ctm_tensor(A_t, env, jnp.array(H_eff_4leg), d_eff)\n"
)
F = "examples/kagome_xxz_pess.py"
edit(
    F,
    "from tenax import (  # noqa: E402\n    CTMConfig,\n    iPEPSConfig,\n    optimize_gs_ad,\n)\n"
    "from tenax.algorithms.ipeps_ctm import ctm  # noqa: E402\n"
    "from tenax.algorithms.ipeps_rdm import compute_energy_ctm  # noqa: E402\n",
    "from tenax import (  # noqa: E402\n    CTMConfig,\n    compute_energy_ctm_tensor,\n"
    "    ctm_tensor,\n    iPEPSConfig,\n    optimize_gs_ad,\n)\n" + WRAP,
)
edit(
    F,
    "    # compute_energy_ctm returns E_h + E_v = 2*<H_tri>, correctly\n",
    "    # compute_energy_ctm_tensor returns E_h + E_v = 2*<H_tri>, correctly\n",
)
edit(F, OLD_BODY, NEW_BODY)

F = "examples/kagome_xxz_spin1_pess.py"
edit(
    F,
    "from tenax import CTMConfig, spin_one_ops  # noqa: E402\n"
    "from tenax.algorithms.ipeps_ctm import ctm  # noqa: E402\n"
    "from tenax.algorithms.ipeps_rdm import compute_energy_ctm  # noqa: E402\n",
    "from tenax import compute_energy_ctm_tensor, ctm_tensor, spin_one_ops  # noqa: E402\n"
    + WRAP,
)
edit(
    F,
    "    ``compute_energy_ctm`` returns E_h + E_v from horizontal and vertical\n",
    "    ``compute_energy_ctm_tensor`` returns E_h + E_v from horizontal and vertical\n",
)
edit(F, OLD_BODY, NEW_BODY)
for rel, text in texts.items():
    (ROOT / rel).write_text(text)
    print("edited", rel)
PY
```

Expected: `edited` lines for the two guide pages and the two examples. `CTMConfig` stays imported in `kagome_xxz_pess.py` (`:697` still uses it); in the spin-1 example it had no other use.

- [ ] **Step 3: Run the two examples**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && for ex in kagome_xxz_pess kagome_xxz_spin1_pess; do JAX_PLATFORMS=cpu taskset -c 0-11 uv run python examples/$ex.py --D 2 --chi 8 --steps 5 > "$W/t8_$ex.txt" 2>&1; echo "$ex rc=$?"; command grep -n "E/site\|deprecated" "$W/t8_$ex.txt"; done
```

Expected: `kagome_xxz_pess rc=0` and `kagome_xxz_spin1_pess rc=0`, each followed by one `E/site = <finite number>` line and no `deprecated` line. Measured in the plan's dry run: `E/site = 0.115582` (the old code gave 0.115578) and `E/site = -0.354355` (old: -0.350443). The spin-1 change comes from the environment, not from the energy function: on the same 2x2 environment, `compute_energy_ctm` and `compute_energy_ctm_tensor` agree to 10 digits at chi=8. `pess_to_ipeps` fills only `A[:, :, :, 0, :]` in that example, so the right virtual leg carries one index, the corner spectrum is rank-collapsed, and `ctm_tensor` never certifies convergence on this state (`diff=inf`, the #898 guard). The example does not check convergence. Record both energies, the old values and this reason for the PR.

- [ ] **Step 4: Build the docs as CI does (background, a few minutes)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && taskset -c 0-11 uv run --locked --extra docs sphinx-build -W -b html docs "$W/docs_html" > "$W/t8_docs.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t8_docs.txt"; exit $rc
```

Expected: `rc=0` and `build succeeded.` near the end of `$W/t8_docs.txt`. The `.. deprecated:: 0.9.3` directive in the `ctm()` docstring renders through the `docs/api/algorithms.rst` autodoc entry (precedent: `compute_excitations`).

- [ ] **Step 5: Commit**

```bash
cd /home/yjkao/tenax-ctm-cleanup && git add docs/guide/algorithms/ipeps.md docs/guide/algorithms/ctm.md examples/kagome_xxz_pess.py examples/kagome_xxz_spin1_pess.py && git commit -F - <<'MSG'
docs(ctm): ctm_tensor replaces ctm() in the guide and the kagome examples

The standalone-CTM section of the iPEPS guide showed ctm().  It now shows
ctm_tensor with return_meta, and a warning gives the deprecation, the
measured Ising errors of the old sweep, and the three behaviours that
differ.  The CTM guide documents ctm_tensor(min_iter=).  The two kagome
PESS examples compute their energy with ctm_tensor and
compute_energy_ctm_tensor.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
MSG
```

---

### Task 9: Verification, mutation count and the PR text

**Run conventions:** work in `/home/yjkao/tenax-ctm-cleanup` (branch `refactor/ctm-delegate-to-2x2`) with `W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2`. Python only through `uv run`. Check `uptime` before each run (wait while the 1-minute load is above 12), run one at a time under `JAX_PLATFORMS=cpu taskset -c 0-11`, redirect to a file in `$W` and read it afterwards (never pipe a run into `tail`). Runs longer than ~2 minutes go in the background and end with `exit $rc`. No push, no PR, no `gh` call: the PR text is a file for the user's go-ahead.

**Files:**
- Create (not committed): `$W/mutate.py`, `$W/mutation.txt`, `$W/pr_body.md`

**Interfaces:**
- Consumes: everything above; the final code text of Tasks 1, 3, 4 and 6 (the mutant anchors below quote it).
- Produces: the verification record and the PR body for `superpowers:finishing-a-development-branch`.

- [ ] **Step 1: Lint the whole tree**

```bash
cd /home/yjkao/tenax-ctm-cleanup && uv run ruff check src tests && uv run ruff format --check src tests
```

Expected: `All checks passed!` and `... files already formatted`, nothing to reformat.

- [ ] **Step 2: Run the required `core` gate (background; long)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q -m core > "$W/t9_core.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t9_core.txt"; exit $rc
```

Expected: `rc=0`; the summary line has no `failed` and no `error`. Copy that line into the PR body.

- [ ] **Step 3: Run the non-core tests of the touched files (background, about 10 min)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest --no-cov -q -m "not core" tests/test_ipeps.py tests/test_ipeps_excitations.py tests/test_ipeps_excitations_gauge.py tests/test_ipeps_rdm_gauge.py tests/test_ctm_tensor.py tests/test_split_ctm_tensor.py tests/test_ipeps_spectrum_contractions.py tests/test_normalise_rdm_zero_grad.py tests/test_ctm_criterion_rank_blind_898.py > "$W/t9_algo.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/t9_algo.txt"; exit $rc
```

Expected: `rc=0`; no `failed` and no `error` in the summary line. The four `algorithm` files run: `test_ipeps.py`, `test_ipeps_excitations.py`, `test_ctm_tensor.py`, and `test_split_ctm_tensor.py`, which imports `ctm` from the shim that Task 7 edits. The `-m core` gate of Step 2 does not run them, and neither does CI on a PR without the `run-full-tests` label. The other files are `core`, so `-m "not core"` deselects them. `rc=5` ("no tests ran") is a failure here: it means the markers no longer select those four files. Copy the summary line into the PR body.

- [ ] **Step 4: Write the mutation script and check its anchors**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && cat > "$W/mutate.py" <<'PYEOF'
"""Mutation count for the ctm() delegation.

Every mutant edits a pristine copy of src/tenax under OUT/src, never the
worktree, and runs the two new or changed test files against it.

    uv run python OUT/mutate.py OUT          # run every mutant
    uv run python OUT/mutate.py OUT --dry    # only check that each anchor is unique
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yjkao/tenax-ctm-cleanup")
OUT = Path(sys.argv[1])
DRY = "--dry" in sys.argv[2:]
SRC = OUT / "src"
TESTS = [
    "tests/test_ctm_delegates_to_2x2.py",
    "tests/test_ctm_min_iter_contract_925_976.py",
]
TC = "tenax/algorithms/_ctm_tensor_convergence.py"
CC = "tenax/algorithms/ipeps_ctm_convergence.py"
FI = "tenax/algorithms/fermionic_ipeps.py"
QR_CAP = (
    "    min_sweeps = min(\n"
    '        min_iter, max_iter - (projector_method == "qr") * min(qr_warmup_steps, max_iter)\n'
    "    )\n"
)
DIRECT = (
    "    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor\n"
    "    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor\n"
    "\n"
    "    return ctm_tensor(\n"
    "        _wrap_as_dense_tensor(A),\n"
    "        ctm_cfg.chi,\n"
    "        max_iter=ctm_cfg.max_iter,\n"
    "        conv_tol=ctm_cfg.conv_tol,\n"
    "    )[0]\n"
)
# (name, file, [(old, new), ...], the test expected to kill it)
MUTANTS = [
    ("M1 no min_iter gate", TC,
     [("\n            if diff < conv_tol and n_iter >= min_sweeps:\n",
       "\n            if diff < conv_tol:\n")],
     "test_convergence_waits_for_min_iter[10-ctm_tensor]"),
    ("M2 cap taken after the warm-up", TC,
     [("    min_sweeps = min(min_iter, max_iter)\n", QR_CAP)],
     "test_ctm_tensor_min_iter_counts_the_warmup_against_the_callers_budget"),
    ("M3 min_iter default 10", TC,
     [("    min_iter: int = 1,\n", "    min_iter: int = 10,\n")],
     "test_ctm_tensor_default_min_iter_keeps_the_early_exit"),
    ("M4 projector_method forwarded", CC,
     [('        projector_method="svd",\n',
       "        projector_method=config.projector_method,\n")],
     "test_helper_runs_no_qr_warmup"),
    ("M5 min_iter dropped", CC,
     [("        min_iter=config.min_iter,\n", "")],
     "test_convergence_waits_for_min_iter[10-ctm_via_2x2]"),
    ("M6 renormalize dropped", CC,
     [("        renormalize=config.renormalize,\n", "")],
     "test_helper_passes_renormalize[False]"),
    ("M7 conv_tol dropped", CC,
     [("        conv_tol=config.conv_tol,\n", "")],
     "test_ising_c10_matches_onsager"),
    ("M18 max_iter dropped", CC,
     [("        max_iter=config.max_iter,\n        conv_tol=config.conv_tol,\n",
       "        conv_tol=config.conv_tol,\n")],
     "test_min_iter_is_capped_at_max_iter[ctm_via_2x2]"),
    ("M8 no shape check", CC,
     [("        if A.ndim != 5 or len(set(A.shape[:4])) != 1:\n", "        if False:\n")],
     "test_rejects_a_bad_raw_shape"),
    ("M9 SymmetricTensor densified", CC,
     [("    if isinstance(A, SymmetricTensor):\n        raise TypeError(\n",
       "    if isinstance(A, SymmetricTensor):\n        A = A.todense()\n"
       "    if False:\n        raise TypeError(\n")],
     "test_refuses_a_symmetric_tensor"),
    ("M10 no DeprecationWarning", CC,
     [('    warnings.warn(\n        "ctm is deprecated',
       '    0 and warnings.warn(\n        "ctm is deprecated')],
     "test_ctm_warns_that_it_is_deprecated"),
    ("M11 initial_env ignored", CC,
     [("    if initial_env is not None:\n        raise ValueError(\n",
       "    if False:\n        raise ValueError(\n")],
     "test_ctm_rejects_initial_env"),
    ("M12 UserWarning always", CC,
     [('    if config.projector_method != "svd":\n', "    if True:\n")],
     "test_ctm_default_config_gives_no_user_warning"),
    ("M13 1x1 recipe", CC,
     [('        recipe="2x2",\n', '        recipe="1x1",\n')],
     "test_ising_c10_matches_onsager"),
    ("M14 fermionic_ctm calls ctm()", FI,
     [("    from tenax.algorithms.ipeps_ctm_convergence import _ctm_via_2x2\n",
       "    from tenax.algorithms.ipeps_ctm_convergence import ctm\n"),
      ("    return _ctm_via_2x2(A, ctm_cfg)\n", "    return ctm(A, ctm_cfg)\n")],
     "test_fermionic_ctm_raw_branch_reproduces_onsager"),
    ("M15 fermionic_ctm calls ctm_tensor directly", FI,
     [("    return _ctm_via_2x2(A, ctm_cfg)\n", DIRECT)],
     "test_fermionic_ctm_raw_branch_reproduces_onsager"),
    ("M16 the helper warns", CC,
     [("    from tenax.algorithms._ctm_env_dense import _as_dense_env\n",
       '    warnings.warn("ctm is deprecated (mutant)", DeprecationWarning, stacklevel=2)\n'
       "    from tenax.algorithms._ctm_env_dense import _as_dense_env\n")],
     "test_helper_emits_no_deprecation_warning"),
    ("M17 fermionic_ctm min_iter=1", FI,
     [("        conv_tol=config.ctm_conv_tol,\n    )\n    return _ctm_via_2x2(A, ctm_cfg)\n",
       "        conv_tol=config.ctm_conv_tol,\n        min_iter=1,\n    )\n"
       "    return _ctm_via_2x2(A, ctm_cfg)\n")],
     "test_fermionic_ctm_raw_branch_keeps_ctmconfig_min_iter"),
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

Expected: `18 mutants, every anchor unique`.

- [ ] **Step 5: Run the mutants (background, about 20 min: 19 runs of about 1 min)**

```bash
cd /home/yjkao/tenax-ctm-cleanup && W=/home/yjkao/tenax-ctm-cleanup/.superpowers/sdd/2026-11-09-ctm-delegate-to-2x2 && uptime && uv run python "$W/mutate.py" "$W/mut" > "$W/mutation.txt" 2>&1; rc=$?; echo "rc=$rc" >> "$W/mutation.txt"; exit $rc
```

Expected: `pristine copy: 66 passed ...`, one `killed` line per mutant whose first `FAILED` line names the expected killer (or another test that asserts the same property), and `18/18 killed`, `rc=0`. A `SURVIVED` mutant is a finding: decide whether it is equivalent (record why) or a gap (add a test, watch it fail on the mutant and pass on the code, then rerun). An `ERROR rc=...` line means the mutant broke collection or the copy; read `$W/mut/mutant_<id>.txt` before counting it.

- [ ] **Step 6: Draft the PR body**

Write `$W/pr_body.md` with these sections, taking every number from a file read in this task or from the plan's "Measured numbers":

1. The marker line `> 🤖 **AI-generated PR** — written by Claude Code, posted by @yingjerkao.`
2. **Summary**: `ctm()` runs `ctm_tensor(recipe="2x2")` with a `DeprecationWarning`; the legacy sweep is deleted. Spec and plan paths.
3. **Evidence**: the Ising table of "Measured numbers", and the Hasik D=2 table of the spec (old `ctm()` -0.66098987 at chi=16 and -0.65999932 at chi=32, both not converged; 2x2 -0.6602310935 at both, in 11 and 12 sweeps; peps-torch -0.66023109).
4. **Changelog note** (for the release PR; CHANGELOG is written only in release PRs):
   - Changed: `ctm()` runs `ctm_tensor(recipe="2x2")` and is deprecated in favour of `ctm_tensor`. Its environments change for every caller, including recorded results; so do those of the raw branch of `fermionic_ctm`.
   - `ctm()` no longer runs under `jax.jit` (neither version supported `jax.grad`); `initial_env` raises `ValueError`; a `projector_method` other than `"svd"` is ignored with a `UserWarning`, and `qr_warmup_steps` is ignored; `return_meta` info fields are Python scalars (were JAX arrays); a `DenseTensor` is accepted (it raised `TypeError`); a raw array that is not 5 legs with equal virtual dimensions raises `ValueError`; a `SymmetricTensor` still raises `TypeError`, now naming `ctm_tensor`.
   - Added: `ctm_tensor(..., min_iter=1)`, keyword-only.
   - Removed (private): `_ctm_sweep`, `_ctm_move_qr`, `_ctm_{left,right,top,bottom}_move`, the `projector_method` argument of `_ctm_move`, and their re-exports in `ipeps_ctm.py`.
5. **Changed tests, with reasons**:
   - Fixture only, now `_ctm_via_2x2`: `test_ipeps.py` (`peps_and_env`, `peps_env`, `converged_peps_env`, `test_energy_product_state_up`), `test_ipeps_core.py` (`peps_and_env`, `peps_env`, `test_energy_product_state_up`), `test_ipeps_excitations.py` (`small_peps_and_env`, `TestEnergyFunctionalOracle::test_gamma_removes_the_ground_state_with_the_physical_metric`), `test_ipeps_excitations_gauge.py` and `test_ipeps_rdm_gauge.py` (`_state`), `test_normalise_rdm_zero_grad.py::test_excitation_norm_gradient_at_zero_B_is_finite`, `test_rdm_gate_transpose_966.py::test_rdm_single_site_energy`.
   - Removed, warm start (`initial_env` raises; `test_ctm_rejects_initial_env` replaces them): `TestCTM::test_ctm_with_initial_env` and `TestCTM::test_ctm_edge_tensors_change` in `test_ipeps.py` and `test_ipeps_core.py`.
   - Removed, QR on `ctm()` (passed vacuously after the change; `test_ctm_warns_that_projector_method_is_ignored` replaces them): `TestQRProjectors::test_qr_ctm_converges`, `::test_qr_energy_is_finite` and the `heisenberg_gate` fixture, in both files. `test_qr_backward_compat` stays.
   - Removed, QR warm-up on `ctm()`: `test_ctm_counts_qr_warmup_in_n_iter` (fails: no warm-up runs) and `test_ctm_warmup_counts_toward_min_iter` (passes for the wrong reason) in `test_ctm_min_iter_contract_925_976.py`. The same file gains the `ctm_tensor` and `_ctm_via_2x2` cases and two `ctm_tensor(min_iter=)` tests.
   - Changed number, examined: `test_ctm_convergence_meta.py::test_ctm_reports_status`, starved budget 20 → 10 sweeps, because the 2x2 recipe converges on `su_state` at sweep 16 (diff 2.2e-13); at sweep 10 the diff is 7.7e-9. Docstrings of `su_state` and `test_info_fields_survive_jit` updated.
   - Imports only: `test_ipeps_excitations.py`, `test_ipeps_spectrum_contractions.py` (`_ctm_env_dense`), `test_ctm_tensor.py` (unused `_ctm_sweep`). `tests/conftest.py`: `core` bucket for the new file.
   - Unchanged, against the new `ctm()`: `test_ctm_complex_site_tensors.py`, `test_ctm_criterion_rank_blind_898.py`, the remaining `TestCTM` tests, `test_ipeps_rdm_gauge.py::test_ctm_accepts_a_complex_site_tensor`.
6. **Mutation count**: the 18 lines and the total from `$W/mutation.txt`.
7. **Verification**: the summary lines of `$W/t9_core.txt` and `$W/t9_algo.txt`; ruff clean; `sphinx-build -W` result (`$W/t8_docs.txt`); the two example energies (`$W/t8_*.txt`), with the old values and the reason for the spin-1 change (Task 8 Step 3).
8. **Not in this PR** (spec "Out of scope").
9. The footer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`, a blank line, `https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`.

- [ ] **Step 7: Hand over**

```bash
cd /home/yjkao/tenax-ctm-cleanup && git status --short && git log --oneline origin/main..HEAD
```

Expected: no uncommitted changes, and the plan commit plus the eight task commits on top of the spec commits. Then use superpowers:finishing-a-development-branch. Push and open the PR only when the user says so, with `$W/pr_body.md` as the body (`gh pr edit` is broken in this clone; set a body later with `gh api repos/tenax-lab/tenax/pulls/<n> -X PATCH -F body=@<file>`). Do not arm `--auto`.
