"""One function per vendor instruction.

Thin wrappers over the PIR node that carries each instruction.  They
decide nothing -- selection lives in `select`, availability in `caps`.
"""

from .relayout import fmadpp_operand_layout


def fmadpp(step):
    """The `fmacdpp{step}` family: `c += broadcast(a) * b`, in place.

    Void, and writes `c` through a reference, so it is a `call_stmt` rather
    than an SSA producer.  The gain over the raw text it replaces is that the
    operands are values: the def-use edges to the loads that produced `a` and
    `b` are real, and the write to `c` is a declared register access, so two
    accumulations on different accumulators are provably independent.
    """
    def emit(writer, C, A, B, row):
        want = fmadpp_operand_layout(step)
        got = getattr(A, 'layout', None)
        # `None` is *unknown*, not *wrong*: the sparse loader does not yet say
        # what it produces, and refusing to emit for want of an annotation
        # would turn a description into an obstacle.  A layout that is present
        # and disagrees is a different matter --- the instruction's DPP
        # pattern assumes this distribution, so a mismatch is a wrong kernel,
        # not a slow one.
        if got is not None and got != want:
            raise ValueError(
                f'fmacdpp{step} needs its broadcast operand at {want!r}, '
                f'got {got!r}')
        writer.call_stmt(f'tensorforge::fmacdpp{step}<{row}>', C, A, B,
                         writes=(C,))
    return emit


fmadpp16 = fmadpp(16)


fmadpp8 = fmadpp(8)


fmadpp4 = fmadpp(4)


def fmascalar(writer, C, A, B, row):
    """`c += a * b` with no cross-lane traffic -- the `step == 1` fallback."""
    writer.accumulate(C, writer.op('mul', C.type, A, B, hint='p'))
