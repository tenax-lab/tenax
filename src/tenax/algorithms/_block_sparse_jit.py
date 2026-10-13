"""``jax.jit`` that skips XLA's CPU fusion pass for block-sparse arguments.

XLA's instruction-fusion pass, on by default, merges chains of small
operations into one kernel: a reshape, a multiply and a transpose become one
loop over the data, with no intermediate arrays in memory.  "Fused" below
means compiled with it (the default); "unfused" means compiled with
``xla_disable_hlo_passes=fusion``.  Results are the same either way.

A block-sparse contraction traces one small op per pair of charge blocks.
The implicit-AD backward of a ``SymmetricTensor`` state is therefore mostly
reshapes, transposes, scatters and ``1x1`` dots: the D=3 U(1) chi=9
``dE/denv`` program is 497k jaxpr equations and 329k HLO instructions.  On
CPU, XLA's instruction-fusion pass is superlinear on such a program.  It also
duplicates cheap producers into each consumer, so the optimized module grows
7x (3.33M lines).  Measured (compile and warm run, fused against unfused):

=========================  ========  ===================  ===============
program                    ops       compile              warm run
=========================  ========  ===================  ===============
``dE/denv`` D=2 chi=16     30k       13.5 s / 13.4 s      4.4 / 3.1 ms
``dE/denv`` D=3 chi=6      570k      546 s / 308 s        14.7 / 10.5 ms
``dE/denv`` D=3 chi=9      627k      8569 s / 905 s       92 / 62 ms
sweep D=2 chi=8            52k       14.9 s / 27.0 s      5.1 / 8.2 ms
sweep D=3 chi=9            225k      145 s / 517 s        15.9 / 19.8 ms
=========================  ========  ===================  ===============

(U(1), StableHLO op counts.)  The LLVM optimization level does not matter
either way.  The forward sweep is the other way round, because its time goes
into the SVDs and the large absorption contractions, which fusion helps.  So
the decision is per program: wrap the backward programs with this function,
and keep plain ``jax.jit`` for the forward.

For the backward it is also per size.  At D=2 fusion wins for every
symmetry: the D=2 U(1) sweep VJP (64k ops) compiled in 20 s fused and 38 s
unfused, and the backward programs of a D=2 ``optimize_gs_ad`` step compiled
1.8-3.5x slower unfused.  From D=3 up ``dE/denv`` runs faster unfused for
U(1), Z2 and fermionic states alike, at a similar compile cost (warm run,
fused against unfused):

========  ===================  ===================
D, chi    Z2 ``dE/denv``       fermionic ``dE/denv``
========  ===================  ===================
2, 4      1.35 / 1.49 ms       1.68 / 2.32 ms
3, 9      8.65 / 7.41 ms       26.3 / 10.9 ms
4, 16     170 / 127 ms         446 / 287 ms
6, 36     11.5 / 10.0 s        33.0 / 16.2 s
========  ===================  ===================

The sweep VJP inside the other backward programs runs within 15% either way
at D>=3 and compiles 1.5-1.9x faster fused, so ``dE/denv`` decides.  The
size measure is the virtual bond dimension: a ``SymmetricTensor`` argument
with a virtual leg (``u``, ``d``, ``l``, ``r``) of dimension at least
``NO_FUSION_MIN_BOND_DIM``.  The block count alone misses the graded
states, which have 16 blocks per site at every D.  A U(1) state with many
charge sectors also qualifies from ``NO_FUSION_MIN_BLOCKS`` blocks: there
the fused compile is superlinear (D=3 chi=9 ``dE/denv``: 8569 s against
905 s).  Both tests are free; counting ops would trace and lower the program
a second time.

A wrapped call that is block-sparse, that large, and on CPU goes to
a variant compiled without the ``fusion`` pass.  Dense calls and every call on another backend
keep the default pipeline: on GPU the effect is unmeasured.  A call inside
another trace also takes the default variant: JAX compiles a nested jit into
the enclosing program and accepts ``compiler_options`` only at the top
level.  The two variants are separate ``jax.jit`` objects, so each keeps its
own trace cache.  ``jax.jit`` takes ``compiler_options`` from JAX 0.4.36; on
an older JAX every call keeps the default pipeline.
"""

from __future__ import annotations

__all__ = ["block_sparse_jit"]

import functools
import inspect

import jax
from jax._src import core as _jax_core

from tenax.core.tensor import SymmetricTensor

# Passed per compile, not through XLA_FLAGS: a process-wide flag would also
# drop fusion for dense programs and for every eager op's compile.
BLOCK_SPARSE_COMPILER_OPTIONS = {"xla_disable_hlo_passes": "fusion"}

# ``jax.jit`` takes ``compiler_options`` from JAX 0.4.36; the dependency floor
# is 0.4.30.
JIT_TAKES_COMPILER_OPTIONS = "compiler_options" in inspect.signature(jax.jit).parameters

# Virtual bond dimension from which the unfused backward runs faster, and the
# total charge blocks from which it compiles faster (see the module docstring).
NO_FUSION_MIN_BOND_DIM = 3
NO_FUSION_MIN_BLOCKS = 40
_VIRTUAL_LABELS = frozenset({"u", "d", "l", "r"})


def _is_symmetric(x) -> bool:
    return isinstance(x, SymmetricTensor)


def _call_platform(args, kwargs) -> str:
    """The platform this call compiles for.

    JAX places a jit call on the device of its committed arguments, and
    otherwise on the ``jax.default_device`` setting, and otherwise on the
    default backend.  ``jax.default_backend()`` alone is the process-wide
    default: in a process with both CPU and GPU it misses a call pinned to
    the other platform.
    """
    for x in jax.tree.leaves((args, kwargs)):
        if isinstance(x, jax.Array) and getattr(x, "committed", False):
            return next(iter(x.devices())).platform
    device = jax.config.jax_default_device
    if device is None:
        return jax.default_backend()
    if isinstance(device, str):
        return jax.devices(device)[0].platform
    return device.platform


def _is_large(symmetric) -> bool:
    if sum(len(t.blocks) for t in symmetric) >= NO_FUSION_MIN_BLOCKS:
        return True
    return any(
        ix.dim >= NO_FUSION_MIN_BOND_DIM
        for t in symmetric
        for ix in t.indices
        if ix.label in _VIRTUAL_LABELS
    )


def _wants_no_fusion(args, kwargs) -> bool:
    """A top-level CPU call with a large ``SymmetricTensor`` argument: a
    virtual leg of dimension ``NO_FUSION_MIN_BOND_DIM`` or more, or
    ``NO_FUSION_MIN_BLOCKS`` charge blocks in total.

    Inside another trace the call is a nested jit, which JAX compiles as part
    of the enclosing program and refuses ``compiler_options`` for; the
    enclosing compile decides fusion there.  The trace state is checked, not
    the arguments: an outer ``jax.jit`` that closes over a concrete tensor
    passes no tracer in.
    """
    if not JIT_TAKES_COMPILER_OPTIONS or not _jax_core.trace_state_clean():
        return False
    leaves = jax.tree.leaves((args, kwargs), is_leaf=_is_symmetric)
    symmetric = [leaf for leaf in leaves if _is_symmetric(leaf)]
    if not _is_large(symmetric):
        return False
    return _call_platform(args, kwargs) == "cpu"


def block_sparse_jit(fn=None, /, **jit_kwargs):
    """``jax.jit(fn, **jit_kwargs)``, without XLA fusion for block-sparse CPU calls.

    Usable as ``@block_sparse_jit`` or
    ``@partial(block_sparse_jit, static_argnames=...)``.  The returned function
    exposes both variants as ``.dense`` and ``.block_sparse``.
    """
    if fn is None:
        return functools.partial(block_sparse_jit, **jit_kwargs)
    dense = jax.jit(fn, **jit_kwargs)
    if JIT_TAKES_COMPILER_OPTIONS:
        block_sparse = jax.jit(
            fn, compiler_options=BLOCK_SPARSE_COMPILER_OPTIONS, **jit_kwargs
        )
    else:
        block_sparse = dense

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if _wants_no_fusion(args, kwargs):
            return block_sparse(*args, **kwargs)
        return dense(*args, **kwargs)

    call.dense = dense
    call.block_sparse = block_sparse
    return call
