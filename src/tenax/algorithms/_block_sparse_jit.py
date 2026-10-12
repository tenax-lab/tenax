"""``jax.jit`` that skips XLA's CPU fusion pass for block-sparse arguments.

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

For the backward it is also per size.  At D=2 fusion still wins: the D=2
U(1) sweep VJP (64k ops) compiled in 20 s fused and 38 s unfused, and the
backward programs of a D=2 ``optimize_gs_ad`` step compiled 1.8-3.5x slower
unfused.  At D=3 the same sweep VJP (203k ops) compiled in 464 s fused and
429 s unfused, and ran 23.1 ms against 18.5 ms.  Op count grows with the
number of charge-block pairs, so the size measure is the total number of
blocks of the ``SymmetricTensor`` arguments (the site tensors): 10 for a
D=2 U(1) 2-site state, 70 at D=3.  Fusion is dropped from
``NO_FUSION_MIN_BLOCKS`` blocks, between the two.  Counting blocks is free;
counting ops would trace and lower the program a second time.

A wrapped call that is block-sparse, at least that size, and on CPU goes to
a variant compiled without the ``fusion`` pass.  Dense calls and every call on another backend
keep the default pipeline: on GPU the effect is unmeasured.  A call inside
another trace also takes the default variant: JAX compiles a nested jit into
the enclosing program and accepts ``compiler_options`` only at the top
level.  The two variants are separate ``jax.jit`` objects, so each keeps its
own trace cache.
"""

from __future__ import annotations

__all__ = ["block_sparse_jit"]

import functools

import jax

from tenax.core.tensor import SymmetricTensor

# Passed per compile, not through XLA_FLAGS: a process-wide flag would also
# drop fusion for dense programs and for every eager op's compile.
BLOCK_SPARSE_COMPILER_OPTIONS = {"xla_disable_hlo_passes": "fusion"}

# Total charge blocks of the SymmetricTensor arguments from which the
# unfused variant compiles and runs faster (see the module docstring).
NO_FUSION_MIN_BLOCKS = 40


def _is_symmetric(x) -> bool:
    return isinstance(x, SymmetricTensor)


def _wants_no_fusion(args, kwargs) -> bool:
    """A top-level CPU call whose ``SymmetricTensor`` arguments hold at least
    ``NO_FUSION_MIN_BLOCKS`` charge blocks in total.

    Inside another trace (an argument is a tracer) the call is a nested jit,
    which JAX compiles as part of the enclosing program and refuses
    ``compiler_options`` for; the enclosing compile decides fusion there.
    """
    if jax.default_backend() != "cpu":
        return False
    leaves = jax.tree.leaves((args, kwargs), is_leaf=_is_symmetric)
    symmetric = [leaf for leaf in leaves if _is_symmetric(leaf)]
    if sum(len(t.blocks) for t in symmetric) < NO_FUSION_MIN_BLOCKS:
        return False
    inner = jax.tree.leaves((args, kwargs))
    return not any(isinstance(x, jax.core.Tracer) for x in inner)


def block_sparse_jit(fn=None, /, **jit_kwargs):
    """``jax.jit(fn, **jit_kwargs)``, without XLA fusion for block-sparse CPU calls.

    Usable as ``@block_sparse_jit`` or
    ``@partial(block_sparse_jit, static_argnames=...)``.  The returned function
    exposes both variants as ``.dense`` and ``.block_sparse``.
    """
    if fn is None:
        return functools.partial(block_sparse_jit, **jit_kwargs)
    dense = jax.jit(fn, **jit_kwargs)
    block_sparse = jax.jit(
        fn, compiler_options=BLOCK_SPARSE_COMPILER_OPTIONS, **jit_kwargs
    )

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if _wants_no_fusion(args, kwargs):
            return block_sparse(*args, **kwargs)
        return dense(*args, **kwargs)

    call.dense = dense
    call.block_sparse = block_sparse
    return call
