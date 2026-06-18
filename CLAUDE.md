# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

- Always open a PR instead of pushing directly to `main`.
- Merge PRs with `gh pr merge <number> --squash --delete-branch --auto` so CI must pass first.
- `main` has branch protection: `Tests (Python 3.11)`, `Tests (Python 3.12)`, and `Tests (macOS, Python 3.12)` must pass.
- Branch protection requires the PR branch to be up-to-date with `main`. If behind, merge main into the branch (`git merge origin/main`) rather than rebasing — rebase can get stuck on `--continue`.
- **Pytest markers**: Tests are auto-marked by file name (`core`, `algorithm`, `slow`) via `conftest.py`. CI required checks run only `pytest -m core`; full suite runs on push to main or with the `run-full-tests` PR label. Locally: `uv run pytest -m core` (fast), `uv run pytest -m "not slow"` (skip expensive), `uv run pytest` (all).
- **AI-authored GitHub comments must be labeled.** Any comment, PR, or issue an AI agent posts (`gh pr comment`, `gh issue comment`, `gh pr review`, `gh pr/issue create`) must carry a `🤖` AI marker in the body so humans can distinguish AI from human comments. A `PreToolUse` hook in `.claude/settings.json` enforces this — it blocks such `gh` commands when no `🤖` (or `AI-generated`) marker is present. Suggested form for comments: `> 🤖 **AI-generated comment** — written by Claude Code, posted by @<user>.`



## Coding Rules

- **Avoid `todense()` on the symmetric tensor path** unless the resulting dense matrix is guaranteed to be small (e.g. a local operator or a bond matrix after decomposition). Block-sparse operations (`SymmetricTensor` methods, `tenax.linalg.svd`/`qr`/`eigh`) should be used instead — they preserve sparsity, run faster at large bond dimension, and keep symmetry quantum numbers intact. Calling `todense()` on a large `SymmetricTensor` defeats the purpose of symmetric tensors.

## Documentation

- When adding new public API (algorithms, classes, functions), update both `README.md` (features list, example sections) and `src/tenax/__init__.py` (`__all__` exports).
- Sphinx docs live in `docs/`; build with `cd docs && make html`.
- Keep `README.md` example code consistent with actual function signatures and test usage.
