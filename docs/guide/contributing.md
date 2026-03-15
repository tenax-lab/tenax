# Contributing

Tenax welcomes contributions of all kinds — bug fixes, new algorithms, documentation
improvements, and test coverage. This guide walks you through the development
workflow.

## Setting up the development environment

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax
uv sync --all-extras --dev
uv run pre-commit install
```

The last command installs pre-commit hooks that run **ruff** (lint + format)
on every commit. If a hook reformats your code, re-stage the changes and
commit again.

## Project structure

```
src/tenax/
├── core/           # DenseTensor, SymmetricTensor, TensorIndex, symmetry, lattice
├── contraction/    # Label-based contraction engine, opt_einsum integration
├── linalg.py       # SVD, QR, eigh decompositions
├── network/        # TensorNetwork, NetworkBlueprint, .net file parser
├── algorithms/     # DMRG, iDMRG, TRG, HOTRG, iPEPS, fPEPS, AutoMPO, CTM
└── __init__.py     # Public API (__all__ exports)
tests/              # Pytest test suite (mirrors src/ structure)
examples/           # Runnable example scripts
docs/               # Sphinx documentation (MyST + RST)
benchmarks/         # CLI benchmark suite
```

## Running tests

Tests are auto-marked by filename via `tests/conftest.py`:

| Marker | Files | What it covers |
|--------|-------|----------------|
| `core` | `test_tensor.py`, `test_contraction.py`, etc. | Tensor ops, symmetry, indexing |
| `algorithm` | `test_dmrg.py`, `test_ipeps.py`, etc. | DMRG, TRG, iPEPS, AutoMPO |
| `slow` | Tests decorated with `@pytest.mark.slow` | Expensive convergence benchmarks |

```bash
uv run pytest -m core              # Fast (~30s) — run before every commit
uv run pytest -m algorithm         # Algorithm tests (~2 min)
uv run pytest -m "not slow"        # Everything except expensive benchmarks
uv run pytest                      # Full suite
```

CI required checks run `pytest -m core` on Python 3.11, 3.12 (Linux) and
3.12 (macOS). The full suite runs on push to `main` or when a PR has the
`run-full-tests` label.

## Branch and PR workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes, write tests, commit.

3. Push and open a PR against `main`:
   ```bash
   git push -u origin feat/my-feature
   gh pr create
   ```

4. CI must pass before merge. PRs are squash-merged:
   ```bash
   gh pr merge <number> --squash --delete-branch --auto
   ```

**Branch protection:** `main` requires three passing checks — `Tests (Python 3.11)`,
`Tests (Python 3.12)`, and `Tests (macOS, Python 3.12)`. The PR branch must be
up-to-date with `main`.

## Writing tests

- Place tests in `tests/test_<module>.py` matching the source module.
- Add the filename to `_FILE_MARKERS` in `tests/conftest.py` so it gets
  the correct marker (`core` or `algorithm`).
- Use existing fixtures from `conftest.py` (e.g., `u1`, `rng`, `u1_sym_tensor_3leg`).
- For algorithm tests, compare against known reference values (exact solutions,
  published benchmarks).
- Mark expensive tests (>30s) with `@pytest.mark.slow`.

## Adding new public API

When you add a new public class, function, or algorithm:

1. Add the symbol to `src/tenax/__init__.py` (`__all__` and imports).
2. Add API docs in the appropriate `docs/api/*.rst` file using `autofunction`
   or `autoclass` directives.
3. Update `README.md` — add to the features list and/or add an example section.
4. If it's a new algorithm, add a tutorial in `docs/guide/algorithms/`.

## Code style

- **Formatter:** ruff format (enforced by pre-commit).
- **Linter:** ruff check with the project's `pyproject.toml` config.
- **Type annotations:** Use them on public API; not required for internal helpers.
- **Docstrings:** NumPy-style for public functions. Keep them concise.
- **No unnecessary abstractions:** Prefer simple, direct code. Three similar lines
  are better than a premature helper function.

## Building the documentation

```bash
uv sync --extra docs
cd docs && uv run make html
```

The site builds at `docs/_build/html/`. CI runs `sphinx-build -W` (warnings as
errors), so fix any warnings before pushing.

Docs use [MyST](https://myst-parser.readthedocs.io/) (Markdown) for guides and
reStructuredText for API reference pages.

## Contributing Claude Code skills

Tenax includes Claude Code skills in `.claude/skills/` that provide domain-specific
AI assistance. See {doc}`claude_code` for details on the plugin.

To add or update a skill:

1. Create or edit `.claude/skills/<skill-name>/SKILL.md`.
2. Open a PR — skills are just Markdown, no code changes needed.
3. After merge, a GitHub Actions workflow automatically syncs skills to the
   [tenax-toolkit](https://github.com/tenax-lab/tenax-toolkit) plugin repository.

## Reporting issues

File bugs and feature requests at
[github.com/tenax-lab/tenax/issues](https://github.com/tenax-lab/tenax/issues).
Include:

- Tenax version (`python -c "import tenax; print(tenax.__version__)"`)
- JAX version and backend (CPU/CUDA/TPU/Metal)
- Minimal reproducing code
- Full traceback
