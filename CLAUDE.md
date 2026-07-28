# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Tenax is a JAX-based tensor-network library: DMRG/iDMRG on MPS, TRG/HOTRG, and iPEPS/fPEPS with CTM environments, built on label-based `DenseTensor`/`SymmetricTensor` operations with block-sparse U(1)/Z_n and fermionic symmetries.

## Git Workflow

- Always open a PR instead of pushing directly to `main`; merge with `gh pr merge <number> --squash --delete-branch --auto` so CI must pass first.
- Branch protection on `main` requires `Tests (Python 3.11)`, `Tests (Python 3.12)`, and `Tests (macOS, Python 3.12)`, and the PR branch must be up-to-date with `main`. If behind, `git merge origin/main` — don't rebase (rebase gets stuck on `--continue` here).
- Run `pre-commit install` once per clone — hooks (ruff, ruff-format) must pass before committing.
- Tests are auto-marked by file name (`core`, `algorithm`, `slow`) via `conftest.py`. CI required checks run only `pytest -m core`; the full suite runs on push to `main` or with the `run-full-tests` PR label. Locally: `uv run pytest -m core` (fast), `uv run pytest -m "not slow"`, or `uv run pytest` (all). On macOS/headless runs, force the CPU backend: `JAX_PLATFORMS=cpu uv run pytest ...`.
- **AI-authored GitHub comments must be labeled.** Any comment, PR, or issue an AI agent posts must carry a `🤖` marker so humans can tell it apart — a `PreToolUse` hook in `.claude/settings.json` blocks `gh` comment/create commands without one. Suggested form: `> 🤖 **AI-generated comment** — written by Claude Code, posted by @<user>.`

## Gotchas

- **Avoid `todense()` on the symmetric-tensor path** unless the result is guaranteed small (a local operator, or a bond matrix after decomposition). Use the block-sparse operations (`SymmetricTensor` methods, `tenax.linalg.svd`/`qr`/`eigh`) instead — densifying a large `SymmetricTensor` defeats the point of symmetric tensors.
- New public API must be exported in `src/tenax/__init__.py` (`__all__`) and reflected in `README.md`; keep README examples consistent with actual signatures and test usage.

## Skills

Task-specific guidance lives in `.claude/skills/` — workflows for DMRG, iPEPS, TRG, the symmetry system, AutoMPO, observables, debugging, benchmarking, and migrations from TeNPy/ITensor/quimb/Cytnx. Load the matching skill for those tasks instead of re-deriving the workflow.
