---
name: tenax-homework
description: >
  Generate Tenax homework problems for a graduate solid state physics course.
  Creates scaffolded assignments with starter code, hints, and solutions, tied
  to specific course topics. Supports four tiers of prompt complexity: factual
  recall, analytical application, synthesis/research, and adversarial (testing
  whether the student/model identifies incorrect premises). Use this skill when
  the instructor asks to create homework, assignments, problem sets, exercises,
  or exam questions involving Tenax, tensor networks, DMRG, or computational
  physics. Also trigger for "generate a problem about [topic]", "create a
  benchmark prompt", "make an exercise for week N", or "design an adversarial
  question".
---

# Homework Scaffold Generator — Tenax Course Assignments

Generate pedagogically structured homework problems that teach Tenax while
reinforcing solid state physics concepts. Each problem includes starter code,
progressive hints, and a reference solution.

## Four-Tier Prompt Structure

Problems are organized into four tiers of increasing complexity, which also
serve as LLM benchmark prompts for evaluating AI model performance.

### Tier 1: Factual Recall
Direct questions with definitive answers. Tests basic knowledge.

**Example:** "What is the MPO index convention in Tenax? Write the shape
of a bulk MPO tensor for the spin-1/2 Heisenberg model and identify each index."

### Tier 2: Analytical Application
Apply concepts to solve a specific problem. Requires understanding, not just
recall.

**Example:** "Use Tenax's `idmrg` to compute the ground state energy per
site of the spin-1/2 Heisenberg chain at bond dimensions χ = 16, 32, 64, 128.
Plot E(χ) vs 1/χ and extrapolate to χ → ∞. Compare with the Bethe ansatz
result."

### Tier 3: Synthesis / Research
Open-ended problems requiring multiple steps, judgment, and interpretation.

**Example:** "Use Tenax to map out the phase diagram of the XXZ model
(H = Σ Δ Sz·Sz + (S+S- + S-S+)/2) as a function of Δ ∈ [-1, 2]. Identify
the ferromagnetic, XY, and Néel phases using appropriate order parameters
and entanglement entropy."

### Tier 4: Adversarial
Contains an incorrect premise or subtle error that the student (or AI model)
must identify and correct. Tests critical thinking.

**Example:** "The Heisenberg antiferromagnet on a 1D chain spontaneously
breaks SU(2) symmetry, developing Néel order in the ground state. Use Tenax
DMRG to compute the staggered magnetization m_s = (1/L) Σ (-1)^i ⟨Sz_i⟩ for
L = 20, 40, 80 and show it approaches a finite value as L → ∞."

*(The premise is wrong: the 1D Heisenberg chain does NOT have Néel order —
the Mermin-Wagner theorem forbids spontaneous breaking of continuous symmetry
in 1D. Students should find m_s → 0 as L → ∞ and explain why.)*

## Problem Templates by Course Topic

### Crystal Structure / Lattice Models (Weeks 1–3)

**Tier 2:** Build a Heisenberg Hamiltonian on a square lattice cylinder using
AutoMPO.

```python
# Starter code
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

Lx, Ly = 4, 3
N = Lx * Ly
auto = AutoMPO(L=N, d=2)

# TODO: Add Heisenberg bonds for a cylinder geometry
# Hint 1: Within-ring bonds are periodic in y: j = x*Ly + (y+1) % Ly
# Hint 2: Between-ring bonds connect site (x,y) to ((x+1),y)
# Hint 3: AutoMPO requires site indices in ascending order: use min(i,j), max(i,j)

# YOUR CODE HERE

mpo = auto.to_mpo(compress=True)
key = jax.random.PRNGKey(0)
mps = FiniteMPS.random(L=N, d=2, chi=16, key=key)
config = DMRGConfig(max_bond_dim=64, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
print(f"E/N = {result.energy / N:.8f}")
```

### Band Theory / 1D Systems (Weeks 4–6)

**Tier 2:** Compare the DMRG ground state energy of the tight-binding model
(XX model) with the exact analytic result from band theory.

**Tier 3:** Add a staggered potential to the tight-binding model and study the
metal-insulator transition as a function of potential strength, using
entanglement entropy as the diagnostic.

### Magnetism (Weeks 7–9)

**Tier 2:** Use iDMRG to compute the spin-1 Heisenberg chain and verify the
Haldane gap by computing the excitation energy.

**Tier 3:** Map the phase diagram of the bilinear-biquadratic spin-1 chain
H = Σ [cos(θ) S·S + sin(θ) (S·S)²] as a function of θ.

**Tier 4:** "The spin-1/2 Heisenberg antiferromagnet has a gap of approximately
0.41 J (the Haldane gap). Use iDMRG to compute this gap."
*(Wrong: the Haldane gap exists for integer spin, not half-integer. S=1/2
Heisenberg is gapless.)*

### Superconductivity / Correlations (Weeks 10–12)

**Tier 2:** Compute the spin-spin correlation function ⟨S^z_0 S^z_r⟩ for the
Heisenberg chain using DMRG. Verify the expected algebraic decay ~ (-1)^r / r.

**Tier 3:** Study pairing correlations in the t-J model using DMRG. Define
custom operators via `build_auto_mpo` with a site_ops dictionary for fermions.

### Phase Transitions (Weeks 13–14)

**Tier 2:** Use TRG to compute the specific heat of the 2D Ising model near T_c.

**Tier 3:** Use iPEPS to study the quantum phase transition in the 2D
transverse-field Ising model. Map the order parameter (magnetization) as a
function of h/J.

### 2D Systems / Advanced Topics (Weeks 15–16)

**Tier 3:** Compare cylinder DMRG (via AutoMPO) and iPEPS for the 2D
Heisenberg antiferromagnet. Which gives better energy per site? What are the
trade-offs?

**Tier 4:** "iPEPS with D=2 is sufficient to capture the ground state of the
2D Heisenberg model because the entanglement in 2D follows an area law, and
D=2 already satisfies the area law bound."
*(Misleading: while area law holds, the prefactor matters. D=2 gives
E ≈ -0.6548 vs exact E ≈ -0.6694. Higher D is needed for quantitative accuracy.
The area law guarantees efficiency scaling, not that the smallest D suffices.)*

## Generating a Problem

When the instructor requests a problem:

1. **Identify the topic** and appropriate tier.
2. **Write the problem statement** with clear physics context.
3. **Provide starter code** with TODO markers and progressive hints.
4. **Include a reference solution** (hidden from students) with expected output.
5. **Add a "what to report" section** — guide students on what to include in
   their writeup (plots, numerical values, physical interpretation).
6. **For Tier 4:** include the correct resolution of the adversarial premise
   in the solution.

## Benchmark Prompt Metadata

Each problem can be tagged for LLM benchmarking:

```json
{
  "tier": 2,
  "topic": "magnetism",
  "week": 8,
  "tenax_functions": ["idmrg", "build_bulk_mpo_heisenberg", "iDMRGConfig"],
  "physics_concepts": ["Haldane gap", "spin-1 chain", "gapped vs gapless"],
  "expected_difficulty": "medium",
  "adversarial": false
}
```

This metadata supports the instructor's goal of building a corpus of benchmark
prompts for evaluating AI models across complexity tiers.
