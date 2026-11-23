# Charged C4v CTM: a 2-sublattice forward sweep

**Status:** design, 2026-11-22. Spec 1 of 2: the forward CTM and its energy.
Spec 2 (AD optimisation) starts after this one merges.
**Branch:** `fix/c4v-charged-u1`. **Fixes:** #762, #763.

## Problem

`ctm_tensor_c4v` gives a wrong environment for a C4v `SymmetricTensor` whose
virtual charges are not self-dual (#762). On the #762 fixture at chi=16 it
raises instead (#763). `compute_energy_ctm_tensor` then reports a wrong
energy and gives no warning.

### Evidence

Two U(1) fixtures, D=2 and d=2. Each sets u, l = OUT with charges `vc`,
d, r = IN with charges `-vc`, and phys = IN with charges `pc`. The data is
random (key 42), projected with `symmetrize_c4v`.

| Fixture | `vc` | `pc` | Blocks | Note |
|---|---|---|---|---|
| #762 | [-1, 1] | [-2, 0] | 10 | well conditioned |
| test | [0, 1] | [0, 1] | 5 | equals `small_peps_u1`; a monomer-dimer state that converges slowly in chi |

Energy per site: J=1, the unrotated Heisenberg gate, `max_iter=200`,
`conv_tol=1e-10`. "Block-sparse" is the `SymmetricTensor` input; "dense" is
`DenseTensor(A.todense(), A.indices)`.

| Fixture | chi | `ctm_tensor_c4v`, block-sparse (today) | `ctm_tensor_c4v`, dense | `ctm_tensor`, dense | `ctm_tensor_2site(A, negate(A))`, block-sparse |
|---|---|---|---|---|---|
| #762 | 8 | 0.5647992010 | 0.2425305830 | 0.2425471974 | 0.242547197411 |
| #762 | 16 | `ValueError` (#763) | 0.2447028694 | 0.2447028690 | 0.244702868974 |
| test | 8 | 0.6493322679 | 0.3231512788 | 0.3231509655 | 0.373860482290 |
| test | 16 | 0.6422930424 | 0.3747802352 | 0.3747406900 | 0.374743084691 |

- #763: `ValueError: Size of label 'c' for operand 1 (4) does not match
  previous terms (5).` (`contract.py:324`).
- Z2 control (self-dual charges): block-sparse and dense `ctm_tensor_c4v`
  agree to 1.2e-16 (chi=4), 3.0e-9 (chi=8) and 4.3e-7 (chi=16).
- The block-sparse 2-site runs took 315-560 s. The dense runs took 10-74 s.

### A charged C4v tensor is a 2-sublattice state

- **Flow-signed charge s.** s is the charge on an OUT leg and `sym.dual` of
  the charge on an IN leg. A block is allowed when the s values of its legs
  sum to zero. Two legs contract when they have opposite flows and identical
  charges, that is, when their s values sum to zero.
- `symmetrize_c4v` permutes the four virtual legs. Its output keeps the block
  structure only if the four faces carry the same s at each position:
  s(u) = s(d) = s(l) = s(r).
- A bond then joins s with s. The uniform U(1) needs s + s = 0, which holds
  only for self-dual charges. Charged faces conserve the staggered U(1)
  instead: A on the sites with x + y even, and B = negate(A) on the other
  sites. peps-torch builds its U(1) C4v state the same way
  (`IPEPS_ABELIAN_C4V.get_bipartite_state` in `ipeps/ipeps_abelian_c4v.py`;
  Hasik, Poilblanc and Becca, SciPost Phys. 10, 012 (2021)).
- **negate(T)** replaces each charge q with `sym.dual(q)`. It keeps flows,
  labels and data. So `todense(B) == todense(A)`, and the uniform dense
  network of `todense(A)` is the same state.
- On the #762 fixture, the dense `ctm_tensor` and the block-sparse 2-site CTM
  on (A, negate(A)) agree to 12 digits at chi=8 and to 2.2e-11 at chi=16.

On the test fixture the same two CTMs differ by 5.1e-2 at chi=8 and by
2.4e-6 at chi=16. Its dense `ctm_tensor_c4v` corner has no exact (q, -q)
pairs (+0.54631 against -0.54430). A dense CTM is a reference for a charged
state only where its spectrum keeps the pairs exact.

### Why today's path fails

Today's path joins a charge q with `sym.dual(q)` where it needs q, at four
places:

1. `_c4v_sweep` makes the D2 leg of T_new from the d2 face of a. The next
   step joins that leg with the u2 face of a. Both faces carry the same s.
2. One projector P is built for the bottom face of Q (fused IN). The sweep
   relabels it onto the right face (fused OUT) and onto both faces of T·a.
   The two faces of Q carry the same s, so their charges are duals.
3. The `"eigh"` projector needs symmetric blocks. In an exactly C4v env the
   mirror maps block c of Q to block dual(c): Q_dual(c) = Q_c^T. So a charged
   block is not symmetric.
4. `_c4v_to_full_env` keeps the flows from before #905. Its docstring warns
   that four chi bonds are same-flow. The xfail note on
   `test_u1_energy_matches_dense` puts the first divergence of the energy at
   C1.c1_d, contracted IN against IN.

Block-sparse contraction matches blocks by charge value, so these joins
connect the wrong states. A ket-bra-symmetric env hides some of them: on
`small_peps_u1` the xfail note found the corner spectra of the two paths
equal to 1e-15. But the energy is wrong on both fixtures, and the #762
fixture raises at chi=16. For a self-dual charge, q = sym.dual(q), and all
four joins are right. This is why the Z2 control passes.

This design does not patch the four joins. It sends charged input to a new
sweep, built for the 2-sublattice state, and keeps the self-dual and dense
paths as they are.

## Scope

In scope: the forward `ctm_tensor_c4v` and `compute_energy_ctm_tensor` on
bosonic charged C4v input. This covers U(1) and Z_n (a Z3 charge 1 has
dual 2).

Out of scope:

- AD through the new sweep. That is spec 2.
- The general `ctm_tensor` on charged C4v input (#1210). On the #762 fixture
  it gives 0.6157994727 at chi=8. On the test fixture it raises `TypeError`.
  This spec only makes the energy routine refuse such an env.
- Complex data on the charged path. It is refused (section 1).
- Fermionic input. It is still densified.
- The claim at ctm.md:155-157 that C4v agrees with `recipe="2x2"` to 1e-12
  (#833). The #762 numbers bear on it (1.7e-5 at chi=8). A separate branch
  handles it.

## Design

### 1. Meaning, routing and validation

A charged C4v `SymmetricTensor` A describes the 2-sublattice state
(A, negate(A)). `ctm_tensor_c4v(A, ...)` returns the environment of an
A-site. `compute_energy_ctm_tensor(A, env, gate, d)` measures that state.

- **Self-dual:** every virtual charge q, at every position, has
  q == sym.dual(q). Examples: Z2, all-zero U(1), Z4 charge 2.
- **Charged:** a bosonic `SymmetricTensor` with at least one virtual charge
  q != sym.dual(q). U(1) [-1, 1] is charged: it is closed under negation as
  a set, but not at each position.

| Input to `ctm_tensor_c4v` | Path |
|---|---|
| `DenseTensor` | today's `_c4v_sweep`, unchanged |
| fermionic `SymmetricTensor` | densified, as today |
| bosonic `SymmetricTensor`, self-dual | today's `_c4v_sweep`, unchanged |
| bosonic `SymmetricTensor`, charged | the new 2-sublattice sweep |

**Validation.** On every bosonic `SymmetricTensor`, before any CTM work:

- The labels u, d, l, r and phys must exist. If not: `ValueError`.
- The four faces must carry the same s at each position. If not: a
  `ValueError` that names the first face that differs from u. This also
  refuses a tensor that conserves the uniform U(1) (s(d) = -s(u)), because
  C4v does not act on its blocks.
- There is no check of the C4v data itself.

**Refusals on the new path only:**

- `projector_method` other than `"eigh"` (the default):
  `NotImplementedError`. On this path `"eigh"` runs the block projector of
  section 2.
- A complex dtype: `NotImplementedError`. The mirror pairing of section 2
  holds only for real data.

**New private helpers.** They live in a leaf module, `_ctm_c4v_charges.py`,
that imports only `tenax.core`. So `_ctm_tensor_energy.py` can use them and
not import the sweep.

- `_negate_charges(T)`: replaces the charges of every index and every block
  key with their duals. It keeps flows, labels and data. Cost O(blocks), no
  `todense()`. Fermionic input raises.
- `_is_charged_c4v(A) -> bool`: a bosonic `SymmetricTensor` with four equal
  faces and a virtual charge that is not self-dual. `ctm_tensor_c4v` and
  `compute_energy_ctm_tensor` both use it.
- `_first_unequal_c4v_face(A) -> str | None`: the face check.
- All charge arithmetic goes through `sym.dual`. No code writes `-q`.

No new public API. The return type stays `CTMTensorEnv`.

### 2. The 2-sublattice sweep

A new module, `_ctm_tensor_c4v_staggered.py`, holds the sweep, its
projector, its initial state, the env expansion (section 3) and the loop.
`_c4v_sweep` does not change, so the dense AD paths that call it (the
`c4v_reference` path, the root-implicit C4v engine, `ipeps_optimize.py`)
are untouched.

**State.**

- C: the corner of an A-site.
- T: the edge above an A-site. Its first site is a B-site.
- a: the double layer of A (`_build_double_layer_tensor`).
- Flows follow the #905 table in the C1/T1 orientation: C (c1_d IN,
  c1_r OUT), T (t1_l IN, u2 OUT, t1_r OUT).

**Charge structure.** Here -x means `sym.dual(x)`, and x ⊕ y is the s of a
fused pair (the sum, for U(1)). Let s_k be the s of state k on a face of A,
the same on all four faces. The faces of a then carry
Δ_kk' = s_k ⊕ (-s_k') for the ket-bra pair (k, k'). Let γ be the s of the
legs of C, per position.

- The two legs of C carry (γ, γ).
- The legs of T, (t_l, D2, t_r), carry (-γ, -Δ, -γ).
- Both fused faces of the enlarged corner Q carry σ = (-γ) ⊕ Δ.

**Invariants.** A metadata check. The tests run it after every step.

- **I1:** every bond (C-T, T.D2-a.u2) pairs opposite flows with identical
  charges.
- **I2:** the two legs of C carry the same s at each position, and so do the
  two chi legs of T. Each chi leg holds as many states of s = c as of
  s = dual(c).
- **I3:** negate(T) pairs with negate(a) by I1, because negate(T) is the
  edge above a B-site.

**One step.**

1. **Enlarge.** Q = C · T_top · T_left · a, with the contraction pattern of
   `_c4v_sweep` and #905 flows. Fuse the bottom face B = (open chi leg of
   T_left, a.d2) and the right face R = (open chi leg of T_top, a.r2).
2. **Truncate.** B and R both carry σ, so a block of Q joins B-sector c with
   R-sector dual(c). The mirror maps block c to block dual(c),
   Q_dual(c) = Q_c^T, with the same singular values.
   - A pair of blocks (c, dual(c)), c != dual(c): one SVD, Q_c = U S V^T.
     The projector takes the kept columns of U in sector c and the kept
     columns of V in sector dual(c).
   - A self-partner block, c == dual(c) (0 for U(1); also n/2 for Z_n with
     n even): Q_c is symmetric. Its projector is the eigh of Q_c, ranked by
     |λ|, as today.
   - Rank all values over all blocks. A pair value counts once in each of
     its two sectors. Keep the chi largest. If the cut falls between the two
     copies of one value, drop both: the dimension is then chi - 1.
   - Drop values below 1e-14 · σ_max, the cutoff of the SVD projector in
     `_ctm_projector.py`.
   - This is the ranking of today's dense eigh: the dense Q has eigenvalues
     +σ and -σ for each pair value σ, and λ for each self-partner value.
3. **Project the corner.** P is the projector of step 2: one array, as in
   today's dense sweep. On each face it gets the label and flow of that face.
   A dual keeps s, so the blocks keep their keys in s. C_new = P^T Q P. It
   is diagonal: the kept S in the pair blocks, and the kept λ (with signs)
   in the self-partner blocks.
4. **Grow the edge.** T·a is the edge above a B-site. Project both of its
   faces with P, as in step 3. Then negate:
   T_new = negate(P^T (T·a) P) is again the edge above an A-site.
5. **Normalise and test.** Normalise C_new and T_new by `max_abs`, as
   today. Test convergence on the singular values of C, as today:
   `_ctm_sv_diff` with `max_rank=_forced_corner_rank(_max_virtual_bond_dim(a))`.
   The chi × chi `C.todense()` stays allowed.

In the dense picture this is today's sweep: one array P on all four faces,
and negation changes no data. Where no step splits a pair at the cut, the
kept space equals that of today's dense eigh. On such a state the two sweeps
give the same env up to a gauge on the chi bonds, and the same energy.

The sweep keeps U(1) and the mirror exactly. So on a hard state (the test
fixture) it can differ from both general CTMs.

**Initial state.** The trace boundary:

- C: 1 × 1, charge 0, value 1.
- T: chi legs of dimension 1 and charge 0. D2 is the full index that pairs
  with a.u2. The data is the ket-bra identity δ_kk', so only s = 0 entries
  are nonzero.

It satisfies I1-I3. The new path checks I1-I3 on it before the first step.

**Other arguments** behave as on today's path.

**Cost per step.** As today's C4v sweep, with a block SVD in place of eigh
and one negation (O(blocks), no data change).

The plan fixes the exact labels and flows of each step. I1-I3, the bond
check (test 5) and the energy tests (tests 7-9) check them.

### 3. Env expansion and energy

**Expansion.** `_c4v_staggered_to_full_env(C, T) -> CTMTensorEnv` builds the
eight tensors around an A-site, by relabels and duals only. A dual keeps s,
so I2 makes the charges match once the flows are set. Flows follow the #905
table. The joins follow the 2-site RDM code (`_rdm2x1_tensor_2site`,
`_rdm1x2_tensor_2site`):

| Bonds | Joins |
|---|---|
| 8 chi | C1.c1_r-T1.t1_l, T1.t1_r-C2.c2_l, C2.c2_d-T2.t2_u, T2.t2_d-C3.c3_u, C3.c3_l-T3.t3_l, T3.t3_r-C4.c4_u, C4.c4_r-T4.t4_u, T4.t4_d-C1.c1_d |
| 4 D² | T1.u2-a.u2, T2.r2-a.r2, T3.d2-a.d2, T4.l2-a.l2 |
| 4 env to negate(env) | T1.t1_r-T1'.t1_l, T3.t3_l-T3'.t3_r, T4.t4_u-T4'.t4_d, T2.t2_d-T2'.t2_u |

(' marks the negated env.) A comment in `_ctm_tensor_c4v.py` joins C1.c1_d
to T4.t4_u. The RDM code joins it to T4.t4_d, and the expansion follows the
RDM code. Today's `_c4v_to_full_env` stays for dense and self-dual input.

**Energy.** `compute_energy_ctm_tensor(A, env, gate, d)` gets one new
branch. It is taken only when `_is_charged_c4v(A)`.

1. **Guard.** The four D² legs of env must pair with the faces of a:
   opposite flows, identical charges. If not, `ValueError`: the env was not
   built for this A (for example, a general `ctm_tensor` env, #1210); build
   it with `ctm_tensor_c4v`. The guard compares index metadata only. It gets
   the face indices through the fusion helper that
   `_build_double_layer_tensor` uses, not a second copy of that rule, and it
   adds no contraction of RDM size.
2. **Sublattice B.** B = negate(A), and env_B = negate of all eight env
   tensors.
3. **Return** `compute_energy_ctm_tensor_2site(A, B, env, env_B, gate, d)`.

The B-A bonds are the negation images of the A-B bonds, so their RDMs are
equal to roundoff. The 2-site result, 0.5 (E_hAB + E_hBA + E_vAB + E_vBA),
is then E_h + E_v per site: the normalisation of the 1-site routine. Two of
the four RDMs are redundant. That cost buys reuse of the 2-site routine,
which the evidence table validates.

**Gate.** No change. The 2-site routine densifies the gate, and negation
keeps the physical basis order. A sublattice-rotated gate works as on the
dense path.

**Blast radius.**

- Only a bosonic charged C4v `SymmetricTensor` takes the new branch.
- The dense C4v callers (the C4v reference AD path, the C4v root-implicit
  engine) and the fermionic caller never take it.
- A general 1x1 caller (the 1x1 optimizer, the root-implicit engines,
  `_ctm_energy_ad.py`) takes it only with such a tensor. Today that input
  gives a wrong energy or a `TypeError` (#1210). With the guard it raises a
  `ValueError`.
- A state that conserves the uniform U(1) never takes it. Its bonds need
  s(d) = -s(u), and four equal faces would then force self-dual charges.

### 4. Tests

**Files.**

- New `tests/test_c4v_staggered.py`, registered as `core` in
  `_FILE_MARKERS` (`tests/conftest.py`): fast mechanism tests with no
  converged CTM. CI runs it on every PR.
- `tests/test_ctm_tensor_c4v.py` (`algorithm`): the CTM and energy tests.
  The full suite runs them: on push to main, or with the `run-full-tests`
  label plus a push.

**Core (mechanism) tests.**

1. **negate:** it is an involution; `todense()` does not change; every block
   still conserves charge; fermionic input raises.
2. **Routing:** a spy on the new sweep. Charged C4v input takes it: the #762
   fixture, and a Z3 fixture with a virtual charge 1 (dual 2). Dense, Z2,
   all-zero U(1) and fermionic input do not.
3. **Validation:**
   - Unequal faces raise a `ValueError` that names the face. This includes
     faces that are equal as multisets but not at each position.
   - `"qr"` on charged input raises `NotImplementedError`.
   - A complex dtype on charged input raises `NotImplementedError`.
4. **Pair rule,** on a synthetic Q with known mirror-paired blocks: a cut
   inside a (c, dual(c)) pair keeps equal counts and a dimension of at most
   chi. A self-partner block needs no partner.
5. **Invariants:** I1-I3 hold after each of the first steps, and all 16
   joins of the section 3 table pair correctly. On the #762 fixture at
   chi=4, and on the Z3 fixture.
6. **Energy routing:** a charged C4v A takes the new branch. A charged U(1)
   A that conserves the uniform way (s(d) = -s(u)) does not, so the standard
   symmetric 1x1 input stays on its own path.

**Algorithm (CTM) tests.**

7. **Energy equality,** #762 fixture, chi=8 and chi=16. The reference is
   today's dense `ctm_tensor_c4v` on `todense(A)`: the same algorithm on the
   same state. Require |E_new - E_ref| <= 1e-8, and normalised corner
   spectra equal to 1e-8. Before it compares, the test asserts the regime:
   - Both runs converged: one more sweep changes the normalised corner
     spectrum by less than 1e-9.
   - The cut does not split a pair: a gap of at least 5% at the cut in the
     enlarged-corner spectrum. The 11.7% (chi=8) and 9.7% (chi=16) measured
     so far are the gaps of the general CTM; the plan measures the C4v gaps
     first.
   - At least one charged block has σ >= 1e-3 · σ_max.
   - The state is not a product state.

   At chi=16 only, also require |E_new - E_gen| <= 1e-8 against the dense
   general `ctm_tensor` (measured, dense C4v against general: 4e-10). At
   chi=8 the C4v and 2x2 truncations differ by 1.7e-5 (0.2425305830 against
   0.2425471974), so chi=8 has no such check.

   This test replaces the strict xfail `test_u1_energy_matches_dense` and
   its `_EnergyGap762` class.
8. **#763:** the #762 fixture runs at chi=16. Today it raises `ValueError`.
9. **Test fixture (`small_peps_u1`), chi=16:**
   - The env is finite, I1-I3 hold, and the pairs stay whole.
   - A `slow` test compares the energy with the block-sparse
     `ctm_tensor_2site(A, negate(A))` value (about 9 min). The tolerance is
     1e-5, from the measured 2.4e-6 spread between the two 2-site CTMs. The
     plan measures the new value first. If the gap is larger, the plan stops
     and reports both numbers, and the user decides. A C4v-against-2x2 gap
     at finite chi is possible (1.7e-5 on the #762 fixture at chi=8), so a
     larger gap is not by itself a defect.
10. **Guard:** the energy of the #762 fixture with a general `ctm_tensor`
    env raises `ValueError`. One sweep is enough, because the guard reads
    metadata.
11. **Z2 control:** block-sparse against dense within 1e-12 at chi=4
    (measured 1.2e-16). This shows that the self-dual path did not change.
12. **Existing tests:** the dense and fermionic tests, and
    `test_u1_converges` (now on the new path), stay unchanged and pass.

**Mutation count.** Each mutation must make the named tests fail.

| Mutation | Tests that must fail |
|---|---|
| No negation of T_new | 5, 7 |
| No pair rule | 4 |
| Today's P / P-bar placement in place of per-face metadata | 5, 7 |
| Charged input sent to the old sweep | 2, 7, 8 |
| env_B = env | 7 |
| No energy guard | 10 |
| Energy routing without the four-equal-faces clause | 6 |
| Self-dual test done by multiset (passes [-1, 1]) | 2, 7 |
| Face test done by multiset | 3 |
| negate keeps the block keys | 1 |
| `-q` in place of `sym.dual(q)` | 2, 5 (Z3) |
| No complex refusal | 3 |

**Method.**

- RED first: each new test fails on main for the named reason.
- Commit before mutating. Mutate in place with `PYTHONDONTWRITEBYTECODE=1`,
  unique anchors and `git checkout --` in a `finally` block.
- Run the suites one at a time, gated on load.

### 5. Docs and housekeeping

**Docstrings.**

- `ctm_tensor_c4v`: the 2-sublattice meaning; the routing table; the
  refusals (unequal faces; a projector other than `"eigh"` and complex data
  on charged input); the measurement call, `compute_energy_ctm_tensor` with
  the same A; fermionic input is still densified. References: Hasik,
  Poilblanc and Becca, SciPost Phys. 10, 012 (2021); peps-torch
  `get_bipartite_state`.
- `compute_energy_ctm_tensor`: the new branch and its guard.
- `_c4v_to_full_env`: the "pre-#905" note points charged input to the new
  function. The note stays correct for dense and self-dual input.
- The `small_peps_u1` fixture: a new docstring. Today it explains the #762
  failure.

**Guide** (`docs/guide/algorithms/ctm.md`, the C4v section and the
standalone example):

- "Only valid for models without sublattice structure" and the example
  comment "(1-site, no sublattice)" get a note: with charged input the
  block-sparse state is (A, negate(A)), which is the uniform state of
  `todense(A)`.
- The example comment "(DenseTensor or SymmetricTensor)" becomes true for
  `ctm_tensor_c4v`. The example adds the energy call.
- Add the routing table and the measured cost: the new sweep against dense
  `ctm_tensor_c4v` at chi=8 and 16, with the numbers from the plan.
- One caveat: a dense CTM is not a U(1) reference on hard states. On the
  test fixture the dense and block-sparse 2-site CTMs differ by 5.1e-2 at
  chi=8.
- ctm.md:155-157 (#833) stays a separate fix: one concern per branch.

**Unchanged.** README and `src/tenax/__init__.py`: there is no new public
API, and the guide already claims the capability. No skill in
`.claude/skills/` names `ctm_tensor_c4v` (checked 2026-11-22).

**Issues and PR.**

- The PR body says "Fixes #762, Fixes #763".
- Spec 2 (AD) starts after this PR merges.
- A marked correction comment on #1210 needs the user's word. Two statements
  there are wrong:
  1. "does not conserve U(1)": the dense network conserves the staggered
     U(1).
  2. "(the charge-paired network above)": the convention of this spec is the
     2-sublattice network, which equals the dense network.

  If the comment goes up, the fix for #1210 follows the same convention.

## Points the plan settles

Each point has a method. None is open in the design.

- The exact labels and flows of each step: fixed in the plan, pinned by
  I1-I3 and test 5.
- The callers that pass a bosonic `SymmetricTensor` to `ctm_tensor_c4v`:
  the plan greps `src/` and `tests/` for them and runs each against the new
  validation before it writes the validation.
- The C4v cut gaps for the regime assert of test 7: measured before the
  test is written.
- The cost numbers for the guide: measured on the #762 fixture at chi=8 and
  16, one run at a time.
- The tolerance of test 9: the plan measures first, and a gap above 1e-5
  goes to the user.
