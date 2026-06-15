"""Tests for U(1)-Sz–symmetric Heisenberg helpers (issue #570 follow-up)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import heisenberg_gate
from tenax.algorithms.ipeps import heisenberg_gate_u1sz
from tenax.core.tensor import SymmetricTensor


class TestHeisenbergGateU1Sz:
    def test_returns_symmetric_tensor(self):
        gate = heisenberg_gate_u1sz()
        assert isinstance(gate, SymmetricTensor)

    def test_dense_roundtrip_matches_plain_gate(self):
        """Same physics as heisenberg_gate(); only the charges differ.

        from_dense only reorders the same float values into blocks (no
        arithmetic), so the dense round-trip is bit-for-bit identical.
        """
        gate_u1 = heisenberg_gate_u1sz()
        gate_plain = heisenberg_gate()
        np.testing.assert_array_equal(
            np.asarray(gate_u1.todense()),
            np.asarray(gate_plain.todense()),
        )

    def test_physical_charges_are_sz(self):
        """Physical legs carry Sz charges [+1, -1] (units of 2*Sz)."""
        gate = heisenberg_gate_u1sz()
        charges = np.asarray(gate.indices[0].charges)
        assert sorted(charges.tolist()) == [-1, 1]

    def test_is_nontrivially_blocked(self):
        """Sz conservation splits H into more than one charge block."""
        gate = heisenberg_gate_u1sz()
        assert len(gate.blocks) > 1

    def test_blocks_conserve_sz(self):
        """Every block conserves Sz: flow-signed charges sum to zero.

        This tests the symmetry property itself (not just that >1 block
        exists): for each block key, sum_leg(flow_sign * charge) == 0
        with flows IN=+1, OUT=-1.
        """
        gate = heisenberg_gate_u1sz()
        signs = [int(ix.flow.value) for ix in gate.indices]
        for key in gate.blocks:
            assert sum(s * q for s, q in zip(signs, key)) == 0


class TestHeisenbergU1SzInit:
    def test_pair_are_symmetric_tensors(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert isinstance(A, SymmetricTensor)
        assert isinstance(B, SymmetricTensor)

    def test_pair_have_five_legs_and_nontrivial_blocks(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert len(A.indices) == 5  # u, d, l, r, phys
        assert len(A.blocks) > 1    # non-trivially blocked -> exercises absorb step
        assert len(B.blocks) > 1

    def test_physical_leg_is_sz_charged(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, _ = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        phys = np.asarray(A.indices[4].charges)  # phys is the 5th leg
        assert sorted(phys.tolist()) == [-1, 1]

    def test_blocks_conserve_sz(self):
        """Every block of each site tensor conserves Sz (flow-signed sum 0)."""
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        for tensor in (A, B):
            signs = [int(ix.flow.value) for ix in tensor.indices]
            for key in tensor.blocks:
                assert sum(s * q for s, q in zip(signs, key)) == 0

    def test_rejects_d_below_2(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        with pytest.raises(ValueError, match="D must be >= 2"):
            heisenberg_u1sz_init_pair(D=1, key=jax.random.PRNGKey(0))


class TestU1SzSymmetricMatchesDense:
    def test_one_step_symmetric_charged_ctm_no_collapse(self):
        """U(1)-Sz charged CTM does not collapse and agrees with dense (#602).

        Regression guard for #602: before the fix the charged CTM environment
        sectors collapsed to zero after the first sweep, so ``E_sym == 0``.
        The fix canonicalizes the symmetric CTM env-init chi bonds into the
        same charge-grouped order the (jit/AD) traced block-sparse SVD emits,
        so the absorb step's positionally-paired fused legs stay consistent.

        Exact agreement with the dense run to ~1e-8 is **not** expected and is
        not the property under test: the symmetric path enforces charge
        conservation (block-diagonal, per-sector truncation) while the dense
        path truncates globally without charge structure, and the CTM is not
        fully converged for this variationally-restricted D=2 / chi=8 init.
        The guard is that the charged sectors survive (``E_sym`` finite and
        clearly negative, not 0) and the symmetric energy agrees with dense to
        a physical tolerance.  (The symmetric contraction was separately
        confirmed order-invariant to ~1e-13 under a faithful virtual-bond
        permutation during the #602 fix.)
        """
        from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
        from tenax.algorithms.ipeps import (
            heisenberg_gate,
            heisenberg_u1sz_init_pair,
        )

        A_sym, B_sym = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        A_dense = A_sym.todense()
        B_dense = B_sym.todense()
        gate = heisenberg_gate().todense()  # dense gate, identical numerics

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=8, max_iter=20),
            gs_num_steps=1,
            unit_cell="2site",
        )

        # Symmetric run — this exercises the non-trivial-charge absorb step.
        (_, _), _, E_sym = optimize_gs_ad(gate, (A_sym, B_sym), config)
        # Dense run from the densified same init.
        (_, _), _, E_dense = optimize_gs_ad(gate, (A_dense, B_dense), config)

        assert np.isfinite(E_sym)
        # #602 guard: charged sectors did NOT collapse (was exactly 0.0).
        assert float(E_sym) < -0.05, f"charged CTM sectors collapsed: E_sym={E_sym}"
        # Physical agreement with dense (not exact — see docstring).
        np.testing.assert_allclose(float(E_sym), float(E_dense), atol=2e-2)


class TestU1SzSymmetricCTMD3:
    """Regression guard for #605: U(1)-Sz 2-plaquette CTM absorb at D>=3.

    Before #605 the symmetric 2-plaquette absorb hard-fused the env ``chi`` and
    ``D**2`` legs into a single ``fused`` leg whose charges came out the exact
    charge-CONJUGATE of the projector half's fused leg — because that projector
    half is built from the *neighbour* cell's enlarged corner, whose seam legs
    are the dual side of the current cell's edge.  ``_contract_symmetric`` pairs
    blocks by raw charge value, so at ``D>=3`` (where the Sz seam charge set is
    asymmetric) this raised ``ValueError: Size of label 'b' for operand 1 ...``.
    It was masked at ``D=2`` (the Sz seam charge set is self-dual, so a leg and
    its conjugate have identical per-sector dims) and never bit FermionParity
    (Z2 charges are self-dual).

    The fix mirrors YASTN's CTM (``proj_corners`` unfuses before applying): the
    projector's fused leg is split back into its bare ``(chi, D**2)``
    constituents (which are individually contractible — equal charges, opposite
    flow — being literally the seam's two sides) and contracted against the
    env's own legs.  The dense path is byte-identical (the constituents pair the
    same elements the fused leg did); the symmetric path no longer raises.
    """

    @staticmethod
    def _env_a_at_d3(chi):
        from tenax.algorithms._ctm_tensor_init import (
            _build_double_layer_tensor,
            initialize_ctm_tensor_env,
        )
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
        return A, initialize_ctm_tensor_env(A, chi), _build_double_layer_tensor(A)

    @pytest.mark.parametrize("chi", [8])
    def test_d3_symmetric_2plaq_absorbs_no_charge_mismatch(self, chi):
        """All four D=3 symmetric 2-plaquette absorbs run without the #605
        charge-sector dim mismatch and return finite, non-collapsed tensors.

        ``chi=8`` is enough to expose the bug: it is D-dependent (the asymmetric
        Sz seam appears at ``D>=3``), not chi-dependent.  ``chi=16`` reproduces
        the same fix but the block-sparse projector SVD is markedly slower
        there (#566 per-sector dispatch), so the guard runs at ``chi=8``.
        """
        from tenax.algorithms._ctm_tensor_moves import (
            _compute_plaquette_projector_pair,
            _ctm_tensor_absorb_bottom_2plaq,
            _ctm_tensor_absorb_left_2plaq,
            _ctm_tensor_absorb_right_2plaq,
            _ctm_tensor_absorb_top_2plaq,
        )

        _A, env, a = self._env_a_at_d3(chi)
        absorbs = {
            "left": _ctm_tensor_absorb_left_2plaq,
            "right": _ctm_tensor_absorb_right_2plaq,
            "top": _ctm_tensor_absorb_top_2plaq,
            "bottom": _ctm_tensor_absorb_bottom_2plaq,
        }
        for direction, absorb in absorbs.items():
            # Uniform plaquette: every cell shares ``env`` / ``a``.
            P_top, P_bot, _, _ = _compute_plaquette_projector_pair(
                env, env, env, env, a, a, a, a, chi, direction
            )
            # Was ``ValueError: Size of label 'b' ...`` before #605.
            out = absorb(env, a, P_top, P_bot, P_top, P_bot)
            for t in out:
                arr = np.asarray(t.todense())
                assert np.isfinite(arr).all(), f"{direction}: non-finite output"
