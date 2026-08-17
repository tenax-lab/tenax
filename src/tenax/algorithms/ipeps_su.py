"""Simple update with no stored bond spectrum (#882 Phase 2).

The defect class this module replaces is one defect wearing four issue
numbers.  Simple update stores each bond's Schmidt spectrum straight from the
SVD that produced it; an imaginary-time gate is **non-unitary**, so a gate on
any *other* bond invalidates that spectrum, and every subsequent step then
divides out and re-absorbs a number that is no longer the state's.  #667 (the
bond carried ``lambda**1.5``), #851 (two stored spectra for four inequivalent
bonds, so ``steps % 4`` chose the answer), #865 (a pinned per-sector layout
made the SVD discard the *largest* singular value) and #869 (a non-monotonic
D=3 energy) are all instances of it.

The rewrite's premise is that the storage goes away rather than the bugs being
fixed one at a time: :class:`_SUState` holds two site tensors in **absorbed
form** and has nowhere to put a spectrum, and each step re-derives the gauge
from the tensors themselves with
:func:`~tenax.algorithms.ipeps_gauge.gauge_fix`.  Vidal form exists only
transiently, inside one step, between the SVD and the ``sqrt(sigma)`` split
that immediately consumes it.

Everything here is underscore-private for v1.  Nothing is exported from
``tenax/__init__.py`` and nothing is documented in ``README.md``: the public
entry point arrives in #882's Phase 4, once ``ipeps()`` is wired to it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from tenax.algorithms._tensor_utils import absorb_sqrt_singular_values
from tenax.algorithms.ipeps_gauge import gauge_fix
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

#: Canonical leg order of a checkerboard site tensor, shared with
#: ``ipeps._wrap_as_dense_tensor`` and ``ipeps.heisenberg_u1sz_init_pair``.
_SITE_LABELS: tuple[str, ...] = ("u", "d", "l", "r", "phys")

#: The four virtual legs, in the same order.
_VIRTUAL_LEGS: tuple[str, ...] = ("u", "d", "l", "r")

#: Each bond as ``(IN end, OUT end)``, where an end is ``(site, leg)``.
#:
#: The **order within each entry is load-bearing twice over**, so it is written
#: here once and read three times below rather than re-derived at each use.
#:
#: *Geometry.*  The IN end is always the left/upper site of the pair, matching
#: ``ipeps_bp_gauge._BONDS`` and ``ipeps_simple_update._to_physical_pair``.  So
#: the IN end takes the gate's ``si`` leg and the OUT end takes ``sj``, which is
#: the convention every existing gate builder in this tree assumes.
#:
#: *Flow.*  ``r`` and ``d`` are ``IN`` and ``l`` and ``u`` are ``OUT`` in the
#: shipped iPEPS convention (``ipeps._wrap_as_dense_tensor``), and
#: :func:`~tenax.linalg.svd` stamps ``OUT`` on ``U``'s new leg and ``IN`` on
#: ``Vh``'s -- always, on both the dense and the block-sparse path.  Putting the
#: OUT end on the ``U`` side therefore reproduces the caller's own flows with no
#: flip anywhere, which is what keeps a whole run to **one** treedef and hence
#: one compile of the traced gauge; see :func:`_su_step`'s *Performance* note.
_BOND_ENDS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "h_AB": (("A", "r"), ("B", "l")),
    "h_BA": (("B", "r"), ("A", "l")),
    "v_AB": (("A", "d"), ("B", "u")),
    "v_BA": (("B", "d"), ("A", "u")),
}

#: Working labels, ``__``-prefixed so they cannot collide with a site leg
#: (``u``, ``d``, ``l``, ``r``, ``phys``) or a gate leg (``si``, ``sj``,
#: ``si_out``, ``sj_out``).
_SHARED = "__shared"
_NEW_BOND = "__bond_new"


@dataclass(frozen=True)
class _SUState:
    """An iPEPS checkerboard pair in absorbed form.

    Absorbed form means each bond's weight is already split ``sqrt(lambda)``
    into both of its ends, so ``A`` and ``B`` together **are** the
    wavefunction -- not one factor of it.  That is the same convention
    :func:`~tenax.algorithms.ipeps_gauge.gauge_fix` takes and returns, which is
    what lets a step hand its output straight to the next step's gauge.

    There is deliberately nowhere to store a bond spectrum.  An imaginary-time
    gate is non-unitary, so a spectrum cached on one bond is invalidated by a
    gate on any other, and every defect this module replaces was a defect in
    exactly that storage (#667, #851, #865, #869).  This is a *new* type rather
    than the existing iPEPS container with its lambda fields removed (#882
    §9.2): a type that used to have them invites them back the first time
    something wants to cache a spectrum, and the old container keeps its
    lambdas legitimately for the real-time path, where a unitary gate leaves
    the other bonds' spectra alone.

    ``test_su_state_has_no_lambda_fields`` is that premise as an executable
    assertion.  Do not add a field to satisfy a caller; re-derive the spectrum
    with ``gauge_fix``, which returns it as a diagnostic.
    """

    A: Tensor
    B: Tensor

    @classmethod
    def from_pair(cls, A: Tensor, B: Tensor) -> _SUState:
        """Build a state from an **absorbed-form** pair, checking its labels.

        The label check is the only validation available: absorbed form is a
        claim about how the bond weights are distributed between two tensors,
        and no property of the two tensors alone can distinguish it from Vidal
        form.  What this does catch is a pair with a leg named ``left`` or
        ``phys_B``, or a 4-leg tensor, i.e. the mistakes that would otherwise
        surface several frames deep inside a ``contract``.

        Args:
            A, B: Absorbed-form site tensors, labels ``(u, d, l, r, phys)`` in
                  any order.

        Returns:
            The state.

        Raises:
            ValueError: if either tensor's label set is not
                ``{u, d, l, r, phys}``.
        """
        for name, t in (("A", A), ("B", B)):
            labels = set(t.labels())
            if labels != set(_SITE_LABELS):
                raise ValueError(
                    f"{name} must be a checkerboard site tensor with labels "
                    f"{set(_SITE_LABELS)}; got {sorted(labels)}.  This type "
                    f"holds the pair in absorbed form -- the site tensors "
                    f"alone, with every bond weight already split into both "
                    f"of its ends."
                )
        return cls(A=A, B=B)

    @property
    def max_D(self) -> int:
        """The largest virtual bond dimension in the pair.

        Read off both tensors and all four virtual legs rather than off ``A.r``
        alone, because the four bonds are **not** constrained to share a
        dimension: a truncation to ``max_D`` on one bond leaves the other three
        as they were, and on the block-sparse path a bond's per-sector layout
        moves under truncation too.  ``phys`` is excluded -- it is not a bond.
        """
        return max(
            t.indices[t.labels().index(leg)].dim
            for t in (self.A, self.B)
            for leg in _VIRTUAL_LEGS
        )


def _reorder(t: Tensor, labels: tuple[str, ...]) -> Tensor:
    """Restore ``labels`` as the axis order.

    ``contract`` and :func:`~tenax.linalg.svd` both return legs in their own
    order.  Everything here is label-driven and does not care, but the axis
    order is part of a pytree's structure, so a step that returned a different
    one each time would recompile the traced gauge on every call.
    """
    current = t.labels()
    if current == labels:
        return t
    return t.transpose(tuple(current.index(lab) for lab in labels))


def _align_gate_to_ket(gate: Tensor, site: Tensor) -> Tensor:
    """The same gate, wired the way ``contract``'s block matching expects.

    ``contract`` pairs block-sparse legs **by charge value** and dense legs by
    position, and the one convention on which the two provably coincide is
    *opposite flows with element-wise equal charges* -- what ``bar()`` and
    ``flip_flow()`` produce, and what
    ``contractor._leg_pairing_fault`` documents as "the convention the block
    matching implements" (#834).  A ket site tensor has ``phys`` ``IN``, so a
    gate that acts on it must have ``si``/``sj`` ``OUT``.

    ``ipeps_simple_update._make_trotter_gate_tensor`` builds ``si``/``sj``
    ``IN`` and ``si_out``/``sj_out`` ``OUT`` -- the same flow as the ket on the
    contracted legs, i.e. exactly the pairing #834 is about.  That is not a
    cosmetic mismatch.  Measured on ``heisenberg_u1sz_init_pair(D=3)`` at
    ``dt=0.05``, one horizontal update's singular values come out

    ==========================  ====================================
    gate as built (``si`` IN)   6.378, 4.611, 4.183, 2.433, 1.959, …
    this function (``si`` OUT)  10.832, 7.578, 4.546, 0.159, 0.122, …
    the same state, densified   10.832, 7.578, 4.546, 0.159, 0.122, …
    ==========================  ====================================

    -- so the aligned gate reproduces the dense answer to machine precision
    and the as-built one does not.  (The dense route is the reference here
    because every leg in it carries the same charge array in the same order,
    so positional pairing *is* the intended pairing.  The shipped
    ``_simple_update_2site_horizontal_tensor`` inherits the discrepancy: on
    that pair it returns ``lambda = [1, 0.723, 0.656]`` block-sparse against
    ``[1, 0.700, 0.420]`` dense.  Filed as **#889**, not fixed here -- this
    module does not own that file.  **Delete this function when #889 lands**,
    and with it ``_su_step``'s call at the ``theta`` contraction: an aligned
    gate makes it the identity, and leaving it in place would then flip the
    flows back the wrong way.)

    **The mechanism, exactly**, because "it is #834's class" is not precise
    enough to act on and the obvious detector does *not* fire.  With ``theta``'s
    ``si``/``sj`` ``IN`` and the gate's ``si``/``sj`` also ``IN``, each product
    block's output key comes out as ``Q_theta - 2 (q_si + q_sj)``, which equals
    ``Q_theta`` only on the ``q_si + q_sj == 0`` sector; every other product
    lands on a key the output indices do not carry and is dropped by the
    per-block loop.  Measured, 278 blocks survive against 434 with the flows
    aligned (``||theta||`` 9.555 against 13.982).  Flipping all four gate flows
    makes the sum telescope back to ``Q_theta`` and nothing is dropped.
    ``contractor._leg_pairing_fault`` would **not** flag the bad pairing --
    same flows with element-wise equal charges is its early ``return None`` --
    and it is ``TENAX_STRICT_CONTRACT``-gated in any case, so the loss is
    silent by two independent routes.

    **A second symptom, same cause.**  Because ``U``'s new leg is stamped
    ``OUT`` and ``si_out`` is ``OUT``, the shipped path also hands back ``A.r``
    ``OUT`` (was ``IN``) and ``A.phys`` ``OUT`` (was ``IN``) after every call:
    it inverts the bond and physical flows on top of dropping blocks.  Here the
    bond half is handled by :data:`_BOND_ENDS`' choice of which end takes the
    ``U`` side, and the physical half by this function -- with the gate aligned,
    ``si_out`` comes back ``IN``, which is the site's own convention.
    ``test_su_step_returns_the_convention_it_accepts`` fires on ``phys`` if
    this call is removed, on the dense path as well as the symmetric one.

    Flipping **all four** flows, not two: the condition a block must satisfy is
    that its net charge is the identity, and negating every leg's contribution
    negates the net, so a charge-*conserving* gate keeps exactly the blocks it
    had.  ``SymmetricTensor.__init__`` validates that, so a gate that does not
    conserve charge raises here rather than silently losing blocks.

    Idempotent: a gate already dual to the ket is returned untouched, so a
    caller who builds one correctly is not punished for it -- and once
    ``_make_trotter_gate_tensor`` is fixed to build ``si`` ``OUT``, this
    becomes a no-op automatically.  **That is this workaround's removal
    trigger**; delete it when the gate builder is corrected, not before.

    Args:
        gate: Two-site gate, labels ``(si, sj, si_out, sj_out)``.
        site: A site tensor of the pair the gate will act on.

    Returns:
        ``gate``, or the same gate with every flow flipped.

    Raises:
        ValueError: if the gate's two input legs disagree on flow, or if either
            carries a different charge array from the site's ``phys`` leg.
            Neither is repairable by flipping flows, and both mis-pair
            *silently*: block matching is by charge **value**, so a gate whose
            ``si`` charges are a permutation of the site's would contract the
            wrong physical basis states together and return a plausible number.
    """
    phys = site.indices[site.labels().index("phys")]
    labels = gate.labels()
    si = gate.indices[labels.index("si")]
    sj = gate.indices[labels.index("sj")]
    if si.flow != sj.flow:
        raise ValueError(
            f"gate legs si ({si.flow.name}) and sj ({sj.flow.name}) must carry "
            f"the same flow; both contract against a site's phys leg, so no "
            f"single flip can align both."
        )
    for name, idx in (("si", si), ("sj", sj)):
        if not np.array_equal(np.asarray(idx.charges), np.asarray(phys.charges)):
            raise ValueError(
                f"gate leg {name} carries charges {list(np.asarray(idx.charges))} "
                f"but the site's phys leg carries "
                f"{list(np.asarray(phys.charges))}.  Block matching pairs by "
                f"charge value, so this would contract the wrong physical "
                f"basis states together without raising.  Build the gate with "
                f"``_make_trotter_gate_tensor(..., site_tensor=A)``."
            )
    if si.flow != phys.flow:
        return gate
    indices = tuple(idx.flip_flow() for idx in gate.indices)
    if isinstance(gate, SymmetricTensor):
        return SymmetricTensor(dict(gate.blocks), indices)
    return DenseTensor(gate.todense(), indices)


def _su_step(state: _SUState, gate: Tensor, max_D: int, bond: str) -> _SUState:
    """One simple-update step on one bond: gauge, gate, truncate.

    No lambda enters and none leaves.  The gauge is re-derived from the pair at
    the start of every step because the previous step's gate invalidated it --
    that is the cadence, and it is not a tunable (#882 §2).

    The four stages:

    1. :func:`~tenax.algorithms.ipeps_gauge.gauge_fix` on the pair.  It takes
       and returns **absorbed form**, and the ``BondWeights`` it also returns
       are the BP fixed-point spectrum *already absorbed into that pair*.  They
       are dropped here.  Absorbing them again would put ``lambda**1.5`` on
       every bond, which is #667's mechanism verbatim and has shipped on this
       very function once already.  Measured *here*, re-absorbing them inside
       this function moves the stepped state by 2.7e-01 (dense ``h_AB`` and
       ``h_BA``) and 2.8e-01 (``v_AB``, ``v_BA``), independently reproduced at
       2.663e-01.  (``gauge_fix``'s own docstring quotes 9.0e-02 to 8.4e-01 for
       the same mistake; that is Phase 1's measurement of re-absorbing at
       *its* boundary, on a range of pairs, and is not a number about this
       function.)
    2. Contract the two sites sharing ``bond`` across it, and apply ``gate`` to
       the two open physical legs (see :func:`_align_gate_to_ket` for why the
       gate's flows are checked first).
    3. Truncated SVD of the result across the same bond, keeping at most
       ``max_D`` singular values, with ``base_charges=None``.
    4. Split ``sqrt(sigma)`` into **both** factors, so the pair comes back in
       absorbed form and the next step's gauge is exact on it.

    ``base_charges=None`` is not an omission.  It is what
    ``ipeps_simple_update._truncation_base_charges`` already returns for dense
    and bosonic-symmetric legs, and imposing the pin is #865: pinned per-sector
    keep counts made the SVD keep ``[4.611, 1.428, 0.159]`` where the top three
    of that same ``theta`` were ``[6.378, 4.611, 4.183]`` -- discarding the
    **largest** singular value and retaining 25.6% of the weight against an
    optimal 87.0%, at step 0.  (Those six numbers are quoted from
    ``_truncation_base_charges``, and they were taken on the shipped gate
    wiring, i.e. on the contraction :func:`_align_gate_to_ket` corrects; the
    aligned spectrum of the same ``theta`` is ``[10.832, 7.578, 4.546]``.  The
    conclusion is unaffected -- a pin that discards the largest value is wrong
    either way -- but the numbers are not this module's.)  Only the fermionic
    single-site path needs the pin, and this module is bosonic (#882 Phase 2).

    ``sqrt`` comes from
    :func:`~tenax.algorithms._tensor_utils.absorb_sqrt_singular_values`, which
    carries a backward-safe guard.  A bare ``jnp.sqrt`` -- what
    ``ipeps_simple_update._to_physical_tensor`` uses -- has adjoint
    ``0.5/sqrt(s)``, i.e. ``+inf`` at ``s == 0``, which is #789's NaN-VJP shape.
    Nothing here is differentiated yet; the point is that it stays safe when
    something is.

    The output is deliberately **not** renormalised.  An overall scale is not
    observable, and it cannot compound across steps either: ``gauge_fix``
    rescales its input by max-abs before the first BP message
    (``ipeps_bp_gauge._prepare``), so every step's SVD sees a unit-scale pair
    whatever scale the previous one left behind.

    Args:
        state: The pair, in absorbed form.
        gate:  Two-site gate, labels ``(si, sj, si_out, sj_out)``.  ``si`` acts
               on the bond's left/upper site and ``sj`` on its right/lower one.
        max_D: Cap on the updated bond's dimension after truncation.
        bond:  One of ``"h_AB"``, ``"h_BA"``, ``"v_AB"``, ``"v_BA"``.

    Returns:
        A new :class:`_SUState`, in absorbed form, with the same leg order,
        leg labels, flows and dimensions as the input.

        **Not the same per-leg charge layout**, and the way it moves is the
        opposite of what one would guess.  The *updated* bond keeps its layout;
        every *untouched* virtual leg is permuted, by the internal
        ``gauge_fix``, which derives each leg afresh from its own
        ``eigh``/``svd``.  Measured, symmetric ``D=3``, one ``h_AB`` step::

            A.u  (0,1,-1) -> (-1,0,1)   changed    A.r  (0,1,-1) -> (0,1,-1)
            A.d  (0,1,-1) -> (-1,0,1)   changed    B.l  (0,1,-1) -> (0,1,-1)
            A.l  (0,1,-1) -> ( 1,-1,0)  changed    B.r  (0,1,-1) -> ( 1,-1,0)

        The *multiset* is preserved on every leg, so no sector is created or
        lost, ``phys`` is stable at ``(1,-1)`` so a gate built once from the
        initial pair stays valid, and the layout settles after the first call,
        so there is no churn.  What it does mean is that **a stepped tensor and
        a freshly built one must not be paired on a shared bond**:
        ``A_stepped.l = (1,-1,0)`` against ``B_fresh.r = (0,1,-1)`` is #834's
        silent mis-pairing, since block matching goes by charge value and dense
        contraction by position.  Evolve both sites of a pair together, which
        is what ``_su_step`` and the evolve loop above it do.

    Raises:
        ValueError: if ``bond`` is not one of the four, or if ``gate``'s input
            legs do not match the site's ``phys`` leg (see
            :func:`_align_gate_to_ket`).

    Warns:
        RuntimeWarning: if the internal ``gauge_fix`` did not converge.  The
            gauge sets the basis the truncation is taken in, so a failed solve
            does not raise anything and does not corrupt the state -- it
            quietly costs truncation quality on *every* step of a run.  That is
            live rather than hypothetical: with the gate's flows left
            unaligned, BP hit ``max_iter=100`` at residual 1.399e-01 on a
            symmetric ``v_BA`` step and this function returned normally.  #870
            is the standing reason to distrust a silent BP: there, a residual
            falling monotonically to 9.8e-12 certified a state that had already
            gone to zero.

    Note:
        ``max_D`` should equal the pair's current bond dimension unless all
        four bonds are being grown together.  ``gauge_fix`` reads **one**
        ``D_h`` off ``A.r`` and one ``D_v`` off ``A.d`` and hands each to both
        bonds of that orientation (``ipeps_gauge._identity_weights``), so a
        pair whose two horizontal bonds have different dimensions is not
        something it can gauge.  Growing ``D`` therefore has to happen a full
        four-bond cycle at a time; the failure is loud (a shape error inside
        the next ``gauge_fix``), not silent.

    Performance:
        ``gauge_fix`` is called at its **default** ``tol=1e-6``.  That buys
        about seven digits, not fifteen -- measured, the weights land ~2.2e-07
        from exact in 14 sweeps, against ~1.8e-15 in 33 sweeps at
        ``tol=1e-14``.  Since the weights are dropped here, the only thing the
        tolerance affects is how close the truncation *basis* is to the true BP
        fixed point, and #882 §2 chose 1e-6 on cost grounds: Phase 1's perf gate
        is calibrated to it and closes at 379-391 ms against a 450 ms budget,
        so a tighter tolerance has ~50 ms of margin to spend and no measurement
        yet saying it is worth spending.

        The returned pair uses the same flow convention as the input -- see
        :data:`_BOND_ENDS` for how, and note that this is what makes a run
        compile once.  A pair's flow convention is part of the traced gauge's
        cache key, so alternating two conventions inside one run costs a second
        ~285 ms compile, which is most of the budget.
    """
    if bond not in _BOND_ENDS:
        raise ValueError(
            f"bond must be one of {sorted(_BOND_ENDS)}; got {bond!r}.  A "
            f"checkerboard unit cell has four inequivalent nearest-neighbour "
            f"bonds, not two -- evolving a subset leaves the state dimerised "
            f"(#667)."
        )

    # Absorbed form in, absorbed form out.  ``weights`` is a report, not state:
    # see stage 1 above for what re-absorbing it costs.  ``info`` is not a
    # report -- it is the only thing that can tell a caller the truncation
    # basis was derived from a solve that never converged, so it is checked
    # rather than dropped (see *Warns*).
    A, B, _weights, info = gauge_fix(state.A, state.B)
    if not info.converged:
        warnings.warn(
            f"gauge_fix did not converge before the {bond} update: "
            f"{info.iterations} sweeps, residual {info.residual:.3e}.  The "
            f"truncation below is taken in the basis that solve left behind, "
            f"so this step's kept singular values are not the state's -- "
            f"silently, and on every step if the cause persists.",
            RuntimeWarning,
            stacklevel=2,
        )
    pair = {"A": A, "B": B}
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]

    def _rename(leg_on_bond: str, prefix: str, phys_label: str) -> dict[str, str]:
        renames = {lg: prefix + lg for lg in _VIRTUAL_LEGS if lg != leg_on_bond}
        renames[leg_on_bond] = _SHARED
        renames["phys"] = phys_label
        return renames

    ren_i = _rename(leg_i, "__i", "si")
    ren_j = _rename(leg_j, "__j", "sj")

    theta = contract(pair[site_j].relabels(ren_j), pair[site_i].relabels(ren_i))
    theta = contract(theta, _align_gate_to_ket(gate, pair[site_i]))

    # ``U`` carries the OUT end of the bond and ``Vh`` the IN end, which is the
    # only grouping that reproduces the caller's flows; see ``_BOND_ENDS``.
    U, sigma, Vh, _sigma_full = truncated_svd(
        theta,
        left_labels=[ren_j[lg] for lg in _VIRTUAL_LEGS if lg != leg_j] + ["sj_out"],
        right_labels=[ren_i[lg] for lg in _VIRTUAL_LEGS if lg != leg_i] + ["si_out"],
        new_bond_label=_NEW_BOND,
        max_singular_values=max_D,
        base_charges=None,
    )
    F_j, F_i = absorb_sqrt_singular_values(U, sigma, Vh, _NEW_BOND)

    def _rebuild(F: Tensor, renames: dict[str, str], leg: str, phys: str) -> Tensor:
        # ``leg`` and ``"phys"`` are both excluded from the inverse map: the
        # SVD replaced the first with ``_NEW_BOND`` and the gate replaced the
        # second with ``si_out``/``sj_out``, so neither of their forward names
        # survives on ``F``.  Leaving them in worked -- ``relabels`` ignores a
        # key it does not find -- but it put two entries on ``"phys"`` and read
        # as though the gate's input leg were still there.
        back = {
            new: old for old, new in renames.items() if old != leg and old != "phys"
        }
        back[_NEW_BOND] = leg
        back[phys] = "phys"
        return _reorder(F.relabels(back), _SITE_LABELS)

    out = {
        site_j: _rebuild(F_j, ren_j, leg_j, "sj_out"),
        site_i: _rebuild(F_i, ren_i, leg_i, "si_out"),
    }
    return _SUState(A=out["A"], B=out["B"])
