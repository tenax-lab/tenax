# Fermionic Implementation Audit (Historical)

This document records issues found and fixed during the fermionic audit
in PR #94 (`fermionic-fixes`, merged 2026-03-10). All items below are
resolved in the current codebase.

## Issues Found (all resolved)

1. **AutoMPO lacked fermionic site operators and JW string insertion.**
   `auto_mpo.py` had no spinless-fermion operator set and no mechanism to
   auto-insert Jordan-Wigner F operators between fermionic operators in
   multi-site terms. Fixed: `fermion_site_ops()` added (auto_mpo.py:70),
   JW insertion added (auto_mpo.py:570).

2. **Dagger twist phase used incorrect formula.**
   `SymmetricTensor.dagger()` computed the twist as the product
   `prod_i twist(q_i)` instead of the correct super-algebra formula
   `(-1)^{sum_{i<j} p_i p_j}`. Fixed: formula updated (tensor.py:975).

3. **Misleading fPEPS comments referenced swap gates.**
   Comments in `fermionic_ipeps.py` described the fermionic sign handling
   as "absorbing swap gates", when signs are actually handled by the
   graded tensor formalism (Koszul signs in transpose, contraction, SVD).
   Fixed: comments updated (fermionic_ipeps.py:401).

4. **No cross-validation of fermionic contraction against dense reference.**
   There were no tests comparing block-sparse fermionic contraction results
   against dense einsum to verify correctness of the Koszul sign logic.
   Fixed: dense cross-validation tests added (test_fermionic.py:869).

## Commits

Issues were addressed in the `fermionic-fixes` branch (PR #94):

| Issue | Fix | Commit |
|-------|-----|--------|
| 1a. Missing fermion operators | Added `fermion_site_ops()` returning C, Cd, N, F, Id | `8213f22` |
| 1b. No JW string insertion | Added `fermionic_ops` parameter to `AutoMPO` with `_insert_jw_strings()` | `6ece9e2` |
| 1c. `build_auto_mpo` passthrough | Added `fermionic_ops` kwarg, exported `fermion_site_ops` | `955c195` |
| 1d. End-to-end validation | Free fermion chain DMRG test matches exact energy | `e68c187` |
| 2. Dagger twist formula | Replaced with `(-1)^{sum_{i<j} p_i p_j}` | `6e31193` |
| 3. Misleading comments | Updated to describe graded tensor formalism | `fb53b77` |
| 4. Cross-validation tests | Added FermionParity, FermionicU1, and ProductSymmetry tests | `3428914` |
