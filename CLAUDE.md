# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Tenax is a JAX-based tensor-network library: DMRG/iDMRG on MPS, TRG/HOTRG, and iPEPS/fPEPS with CTM environments, built on label-based `DenseTensor`/`SymmetricTensor` operations with block-sparse U(1)/Z_n and fermionic symmetries.

## Git Workflow

- Always open a PR instead of pushing directly to `main`; merge with `gh pr merge <number> --squash --auto` so CI must pass first. **Never pass `--delete-branch`** — `main` uses a merge queue, which deletes the head branch itself once the PR merges. Passing the flag deletes the branch the moment the PR *enters* the queue, which closes the PR and drops it from the queue (recover with `git push origin FETCH_HEAD:refs/heads/<branch>` from `refs/pull/<n>/head`, then reopen). The queue also picks the merge strategy, so `gh` reporting "The merge strategy for main is set by the merge queue" is normal, not a failure.
- **Read the review comments before merging — green CI is not review.** Codex reviews every PR, and its findings arrive *after* the checks go green, so a PR can be `CLEAN` and merge-ready while carrying an unread P1. Check both endpoints, since inline comments do not appear in the reviews list:

  ```
  gh api repos/tenax-lab/tenax/pulls/<n>/reviews  --paginate --jq '.[] | "[\(.state)] \(.user.login)\n\(.body)"'
  gh api repos/tenax-lab/tenax/pulls/<n>/comments --paginate --jq '.[] | "\(.path):\(.line)\n\(.body)"'
  ```

  `--paginate` is not optional: without it you get only the first page, and a rule against missing findings that silently reads the first 30 of them is worse than no rule.

  Verify a finding before acting on it *and* before dismissing it — Codex is often right about the mechanism but wrong about the blast radius, in both directions. Once a PR is queued it rejects pushes (GH006), so fixing anything means dequeuing via the `dequeuePullRequest` GraphQL mutation; the cheap moment to look is *before* arming `--auto`. This is not hypothetical: #847 entered the queue with an unread P2, merged as `f306d22`, and the hole had to be fixed on `main` as #848.
- Branch protection on `main` requires `Tests (Python 3.11)`, `Tests (Python 3.12)`, and `Tests (macOS, Python 3.12)`, and the PR branch must be up-to-date with `main`. If behind, `git merge origin/main` — don't rebase (rebase gets stuck on `--continue` here).
- **Don't arm `--auto` unasked.** Queueing is the point of no return, and it is the user's call, not a default.
- **Check what your `gh` actually supports before building a watcher on it, and make the watcher fail *closed*.** The `gh` here is 2.45.0, where `gh pr checks --json` exits with `unknown flag: --json` — a watcher built on it hits the error branch every poll and can never report (one ran 11 hours in a single session). Newer `gh` does support it, so verify with `gh --version` and `gh pr checks --help` rather than trusting either claim. Whatever you poll, exit only on an explicit success condition — `gh pr view <n> --json state --jq .state` returning `MERGED`/`CLOSED` — so a transient `gh` error retries instead of falling out of the loop as though it had succeeded.
- Run `pre-commit install` once per clone — hooks (ruff, ruff-format) must pass before committing.
- Tests are auto-marked by file name (`core`, `algorithm`, `slow`) via `conftest.py`. CI required checks run only `pytest -m core`; the full suite runs on push to `main` or with the `run-full-tests` PR label. Locally: `uv run pytest -m core` (fast), `uv run pytest -m "not slow"`, or `uv run pytest` (all). On macOS/headless runs, force the CPU backend: `JAX_PLATFORMS=cpu uv run pytest ...`.
- **AI-authored GitHub comments must be labeled.** Any comment, PR, or issue an AI agent posts must carry a `🤖` marker so humans can tell it apart — a `PreToolUse` hook in `.claude/settings.json` blocks `gh` comment/create commands without one. Suggested form: `> 🤖 **AI-generated comment** — written by Claude Code, posted by @<user>.`

## Gotchas

- **Avoid `todense()` on the symmetric-tensor path** unless the result is guaranteed small (a local operator, or a bond matrix after decomposition). Use the block-sparse operations (`SymmetricTensor` methods, `tenax.linalg.svd`/`qr`/`eigh`) instead — densifying a large `SymmetricTensor` defeats the point of symmetric tensors.
- New public API must be exported in `src/tenax/__init__.py` (`__all__`) and reflected in `README.md`; keep README examples consistent with actual signatures and test usage.

## Skills

Task-specific guidance lives in `.claude/skills/` — workflows for DMRG, iPEPS, TRG, the symmetry system, AutoMPO, observables, debugging, benchmarking, and migrations from TeNPy/ITensor/quimb/Cytnx. Load the matching skill for those tasks instead of re-deriving the workflow.
