# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Git Workflow

- Always open a PR instead of pushing directly to `main`.
- Merge PRs with `gh pr merge <number> --squash --delete-branch --auto` so CI must pass first.
- `main` has branch protection: `Tests (Python 3.11)`, `Tests (Python 3.12)`, and `Tests (macOS, Python 3.12)` must pass.
- Branch protection requires the PR branch to be up-to-date with `main`. If behind, merge main into the branch (`git merge origin/main`) rather than rebasing.
- **Pytest markers**: Tests are auto-marked by file name (`core`, `algorithm`, `slow`) via `conftest.py`. CI required checks run only `pytest -m core`; full suite runs on push to main or with the `run-full-tests` PR label.
- Local test commands:
  - `uv run pytest -m core` (fast)
  - `uv run pytest -m "not slow"` (skip expensive)
  - `uv run pytest` (all)
- On macOS CI/headless runs, force CPU backend for reproducibility:
  - `JAX_PLATFORMS=cpu uv run pytest ...`

## Documentation

- When adding new public API (algorithms, classes, functions), update both `README.md` (features list, example sections) and `src/tenax/__init__.py` (`__all__` exports).
- Sphinx docs live in `docs/`; build with `cd docs && make html`.
- Keep `README.md` example code consistent with actual function signatures and test usage.

## Skills

- Project skills live in `.claude/skills` and are made available to Codex via `.agents/skills` symlink.
