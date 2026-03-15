# Claude Code Integration

Tenax ships an official [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
plugin that gives Claude domain-specific knowledge about tensor network
algorithms, API usage, and common workflows. When installed, Claude can
guide you through DMRG, iPEPS, TRG, and other calculations with
Tenax-specific code examples and troubleshooting.

## Installing the plugin

In any terminal with Claude Code available, first add the Tenax marketplace
(one-time setup), then install the plugin:

```bash
claude plugin marketplace add tenax-lab/tenax-toolkit
claude plugin install tenax-toolkit
```

This installs all Tenax skills into your Claude Code session. No API keys
or server setup required — skills are plain Markdown files that Claude reads
on demand.

## What the plugin provides

The plugin bundles **18 skills** covering the full Tenax workflow:

### Getting started
| Skill | Description |
|-------|-------------|
| `tenax-getting-started` | Installation, backend setup, first calculation |
| `tenax-tensor-ops` | DenseTensor, SymmetricTensor, contraction, SVD, eigh, QR |
| `tenax-symmetry` | U(1), Z_n, FermionParity, block-sparse tensors |
| `tenax-blueprint` | NetworkBlueprint, `.net` files, custom contractions |

### Algorithms
| Skill | Description |
|-------|-------------|
| `tenax-dmrg-workflow` | Finite DMRG, iDMRG, cylinder DMRG |
| `tenax-ipeps-workflow` | Simple update, AD optimization, excitations, lattice abstraction |
| `tenax-fermionic-ipeps` | fPEPS with graded tensors, spinless fermions, t-V model |
| `tenax-trg-workflow` | TRG, HOTRG, 2D Ising, phase transitions |
| `tenax-autompo` | Build Hamiltonians from symbolic operator descriptions |
| `tenax-observables` | Expectation values, correlations, entanglement entropy |
| `tenax-ed-comparator` | Exact diagonalization vs DMRG validation |

### Tools and migration
| Skill | Description |
|-------|-------------|
| `tenax-benchmark` | Performance benchmarks across CPU/GPU/TPU/Metal |
| `tenax-debugger` | Diagnose shape errors, NaN, convergence failures |
| `tenax-homework` | Generate homework problems for graduate courses |
| `tenax-migration-cytnx` | Migrate from Cytnx |
| `tenax-migration-itensor` | Migrate from ITensor |
| `tenax-migration-quimb` | Migrate from quimb |
| `tenax-migration-tenpy` | Migrate from TeNPy |

## Usage examples

Once the plugin is installed, Claude automatically activates the right skill
based on your question. Some examples:

**"How do I run DMRG for the Heisenberg model?"**
→ Activates `tenax-dmrg-workflow`, walks you through AutoMPO, MPS setup, and DMRG configuration.

**"My iPEPS energy is not converging"**
→ Activates `tenax-debugger`, checks CTM parameters, bond dimensions, and Trotter step size.

**"I'm coming from ITensor, how do I translate my code?"**
→ Activates `tenax-migration-itensor`, maps ITensor concepts to Tenax equivalents.

**"Set up a fermionic PEPS calculation for spinless fermions"**
→ Activates `tenax-fermionic-ipeps`, guides through FermionParity, gate construction, and `fpeps()`.

## For contributors

Skills are maintained in the main Tenax repository under `.claude/skills/`.
When changes are merged to `main`, a GitHub Actions workflow automatically
syncs them to the `tenax-lab/tenax-toolkit` plugin repository.

To add or update a skill:

1. Edit files in `.claude/skills/<skill-name>/SKILL.md`
2. Open a PR to `main`
3. After merge, the sync workflow creates a PR in `tenax-toolkit`

## Without the plugin

If you prefer not to install the plugin, you can still use Claude Code with
Tenax — it will use its general knowledge of Python, JAX, and tensor
networks. The plugin simply adds Tenax-specific guidance and up-to-date API
examples.

Alternatively, clone the repository and Claude Code will automatically read
the skills from `.claude/skills/` in the local checkout.
