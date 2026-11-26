# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.sycl_element_blocks`: an element's operands in global memory read
and written by 2D block messages of the sub-group over the element's own
storage (`memory.element_blocks`, `BatchLoop._emit_versions`,
`GlbToRegLoader._block_loops`, `StoreRegToGlb._block_loops`).

What a wrong answer would come from: a block where today's transfer moves
fewer lanes -- a guard's ragged slot, a slot past the operand's columns -- or
a block that takes a read the pseudo-IR forwards from a store, or a body that
reads blocks without the element's check.  The kernels are held here against
the text the generator writes; what the device computes is held by tfemu
(`~/tf-probe/pvcport/s6`) and the Max 1550's.
"""

import contextlib
import io
import re
import warnings
from types import SimpleNamespace

import pytest

import seissol_suite as suite
from tensorforge import toolchain
from tensorforge.backend.instructions.memory import element_blocks as eb
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator


@pytest.fixture(autouse=True)
def _no_ocloc(monkeypatch):
    """Every build as on a machine with no `ocloc` to ask, as in
    `test_sycl_dpas`: what is written does not depend on the test machine's
    IGC."""
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: None)


def _kernel(name='gpu_volume', order=6, precision='s', backend='oneapi',
            **options):
    descrs = DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-{precision}',
        name))[0]
    fp = Datatype.F64 if precision == 'd' else Datatype.F32
    gen = Generator(descrs, Context(arch='pvc', backend=backend, fp_type=fp,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel()


def _norm(text):
    """A kernel but for its name and the options comment."""
    text = re.sub(r'kernel_[0-9a-f]{16}', 'K', text)
    return re.sub(r'// options: [^\n]*', '', text)


ON = dict(sycl_element_blocks=True)
READ = re.compile(r'tensorforge::elementBlock2d<float, (\d), (\d+), (\d+)>\('
                  r'item\.get_sub_group\(\), (glb_m\d+), ')
WRITE = re.compile(r'tensorforge::elementBlock2dStore<float, (\d), (\d+), '
                   r'(\d+)>\(item\.get_sub_group\(\), (glb_m\d+), ')
CHECK = re.compile(r'bool (\w+) = tensorforge::elementBlocks<(\d+)>\('
                   r'item\.get_sub_group\(\), (glb_m\d+)\);')


def _bodies(text):
    """`[(then, else)]`: for every element body the check versions, the two
    bodies it chooses between, as text -- read off the braces after the `if`
    that follows the body's checks."""
    out = []

    def block(at):
        head = re.compile(r'\{\n').search(text, at)
        depth, k = 1, head.end()
        while depth:
            brace = re.compile(r'[{}]').search(text, k)
            depth += 1 if brace.group(0) == '{' else -1
            k = brace.end()
        return head.end(), k - 1

    at = 0
    while True:
        check = CHECK.search(text, at)
        if check is None:
            break
        last = check
        while True:
            more = CHECK.search(text, last.end())
            if more is None or text[last.end():more.start()].count('if (') > 0:
                break
            last = more
        iff = re.compile(r'\n\s*if \(').search(text, last.end())
        start, end = block(iff.end())
        rest = re.match(r'\s*else ', text[end + 1:])
        assert rest, text[end:end + 200]
        start2, end2 = block(end + 1 + rest.end() - 1)
        out.append((text[start:end], text[start2:end2]))
        at = end2
    assert out, 'no element check'
    return out




# -- off --------------------------------------------------------------------- #

@pytest.mark.parametrize('dpas', [False, True], ids=['fma', 'dpas'])
def test_off_and_auto_change_nothing(dpas):
    """The option off, and `auto` until the blocks are measured, write the
    kernel they always wrote."""
    plain = _norm(_kernel(sycl_dpas=dpas))
    assert _norm(_kernel(sycl_dpas=dpas, sycl_element_blocks=False)) == plain
    assert _norm(_kernel(sycl_dpas=dpas, sycl_element_blocks='auto')) == plain
    assert 'elementBlock' not in plain


@pytest.mark.parametrize('options', [
    dict(backend='esimd'), dict(backend='acpp'), dict(precision='d'),
], ids=['esimd', 'acpp', 'f64'])
def test_no_blocks_where_the_target_has_none(options):
    """The explicit vector, AdaptiveCpp -- no sub-group stated -- and double
    precision, whose blocks are not 32-bit: the kernel they always got."""
    kernel = _kernel(**options, **ON)
    assert 'elementBlock' not in kernel
    assert _norm(kernel) == _norm(_kernel(**options))


# -- on ---------------------------------------------------------------------- #

@pytest.mark.parametrize('dpas', [False, True], ids=['fma', 'dpas'])
def test_the_body_is_built_twice_behind_the_check(dpas):
    """One check an operand, the element's pointer at the surface's pitch;
    the blocks only behind it, and behind its failure today's body."""
    kernel = _kernel(sycl_dpas=dpas, **ON)
    checks = {(m.group(3), int(m.group(2))) for m in CHECK.finditer(kernel)}
    assert checks == {('glb_m0', 256), ('glb_m2', 256)}
    (then, otherwise), = _bodies(kernel)
    assert READ.search(then) and WRITE.search(then)
    assert 'elementBlock' not in otherwise


def test_the_fallback_is_todays_transfers():
    """Behind the check's failure today's transfers, in today's order and
    lane by lane.  Not today's text: the pseudo-IR takes a read from what
    the lane stored only where the store is certain to it (`pir.flow`), and
    a store of the block body is possible to a read of the fallback -- so
    the fallback keeps reads and writes today's body forwards, which cost
    the cold path time and nothing of what it computes."""
    transfers = re.compile(r'// (r\d+ = load\{g>r\}\(glb_m\d+\)|glb_m\d+ = '
                           r'store\{r>g\}\(r\d+\));')
    (then, otherwise), = _bodies(_kernel(sycl_dpas=True, **ON))
    today = _kernel(sycl_dpas=True)
    assert transfers.findall(otherwise) == transfers.findall(today)
    assert transfers.findall(then) == transfers.findall(today)
    assert 'elementBlock' not in otherwise


@pytest.mark.parametrize('dpas', [False, True], ids=['fma', 'dpas'])
def test_a_block_is_a_slot_of_nine_columns(dpas):
    """Eight columns a message and the ninth one of its own, the element's
    columns 256 bytes apart: the order-6 volume's I and Q."""
    (then, _), = _bodies(_kernel(sycl_dpas=dpas, **ON))
    shapes = {(int(r), int(p), int(c)) for r, p, c, _ in READ.findall(then)}
    assert shapes == {(8, 256, 9), (1, 256, 9)}
    assert {op for *_, op in READ.findall(then)} == {'glb_m0', 'glb_m2'}


def test_the_nine_by_nine_matrices_keep_their_reads():
    """A star matrix's column is 36 bytes: no surface (the messages want 64
    and a whole number of 16)."""
    kernel = _kernel(sycl_dpas=True, **ON)
    for name in ('glb_m1', 'glb_m3', 'glb_m4'):
        assert not re.search(r'elementBlock2d<[^>]*>\(item\.get_sub_group\(\), '
                             + name + ',', kernel)


def test_a_guarded_slot_of_an_accumulation_keeps_its_lanes():
    """Q's final write: the slots whose every lane the store writes go by
    blocks, the ragged ones of the accumulation -- row 0 and the rows past
    53 -- lane by lane, under their guards."""
    (then, _), = _bodies(_kernel(sycl_dpas=True, **ON))
    tail = then[then.rindex('glb_m2 = store{r>g}'):]
    assert WRITE.search(tail)
    assert re.search(r'if \(v\d+_g\) \{\s*glb_m2\[', tail)


def test_a_read_after_a_write_stays_a_read_the_ir_forwards():
    """The derivative writes each derivative and reads it back: those reads
    stay per lane, for `pir.forward` to take from the registers, and so do
    the writes a read follows -- in each of its bodies, no read of an
    operand after its first block write, and no block read after a write."""
    bodies = _bodies(_kernel('gpu_derivative', sycl_dpas=True, **ON))
    for then, _ in bodies:
        for op in {o for *_, o in WRITE.findall(then)}:
            first = next(m for m in WRITE.finditer(then) if m.group(4) == op)
            assert not re.search(r'= ' + op + r'\[', then[first.start():]), op
        for m in READ.finditer(then):
            op, before = m.group(4), then[:m.start()]
            assert not re.search(op + r'\[[^\]]*\] = ', before), op
            assert not re.search(r'elementBlock2dStore<[^>]*>\(item\.'
                                 r'get_sub_group\(\), ' + op + ',', before), op
    assert any(READ.search(t) for t, _ in bodies)
    assert any(WRITE.search(t) for t, _ in bodies)


def test_a_promise_takes_the_check_away():
    """`element_base_align` at the surfaces' grain or above: blocks in the
    one body, no check; below it the check stays."""
    kernel = _kernel(sycl_dpas=True, element_base_align=64, **ON)
    assert READ.search(kernel) and not CHECK.search(kernel)
    assert CHECK.search(_kernel(sycl_dpas=True, element_base_align=32, **ON))


def test_the_kernels_parse():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    for name in ('gpu_volume', 'gpu_derivative'):
        result = syntax.check_source(_kernel(name, sycl_dpas=True, **ON),
                                     shim=syntax.SYCL_SHIM)
        assert result.ok, (name, result.errors())


# -- the surfaces ------------------------------------------------------------ #

def _symbol(rows, columns, fptype=Datatype.F32, **fields):
    from tensorforge.backend.symbol import SymbolType
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.tensor import Tensor
    tensor = Tensor.__new__(Tensor)
    # The fields as a built tensor holds them; the two properties by the
    # values behind them, which their setters would check against a pattern
    # this tensor has none of.
    state = dict(_storage_order=None, simt_interleave=None, slot_major=None,
                 storage_surface=None, packing=None, storage_planar=False,
                 _storage_parts=1, addressing=Addressing.PTR_BASED)
    for k, v in fields.items():
        state['_' + k if k in ('storage_order', 'storage_parts') else k] = v
    tensor.__dict__.update(state)
    tensor.get_actual_shape = lambda: (rows, columns)
    return SimpleNamespace(stype=SymbolType.Global, obj=tensor, name='glb_m0',
                           get_fptype=lambda: fptype, stack=lambda: 1)


@pytest.mark.parametrize('rows,grain', [(64, 64), (56, 32), (120, 32),
                                        (128, 64), (40, 32), (24, 32),
                                        (20, 16), (16, 64)])
def test_the_grain_is_what_pitch_and_64_share(rows, grain):
    """SeisSol's padding to 64 bytes (NVIDIA's captures, or
    OVERRIDE_VECTORSIZE=64) puts every pitch on whole cache lines; its
    padding to 32 on PVC leaves the 32 of 64 the row shift reaches."""
    found = eb.surface(_symbol(rows, 9))
    assert found == eb.Surface(pitch=4 * rows, columns=9, rows=rows)
    assert found.grain == grain


@pytest.mark.parametrize('fields,why', [
    (dict(rows=9), 'at least 64'), (dict(rows=18), 'a whole number of 16'),
    (dict(fptype=Datatype.F64), 'not F32'),
    (dict(storage_parts=2), 'in parts'),
    (dict(storage_planar=True), 'in planes'),
    (dict(storage_order=(0, 1)), 'another order'),
], ids=['nine', 'eighteen', 'f64', 'parts', 'planar', 'order'])
def test_what_has_no_surface(fields, why):
    rows = fields.pop('rows', 64)
    fptype = fields.pop('fptype', Datatype.F32)
    answer = eb.surface_refusal(_symbol(rows, 9, fptype, **fields))
    assert answer is not None and why in answer, answer


def test_a_shared_operand_has_no_surface():
    from tensorforge.common.basic_types import Addressing
    answer = eb.surface_refusal(_symbol(64, 9, addressing=Addressing.NONE))
    assert answer is not None and 'own' in answer


@pytest.mark.parametrize('rows,first,slots', [
    (64, 0, [0, 1, 2, 3]), (56, 0, [0, 1, 2]), (120, 0, [0, 1, 2, 3, 4, 5, 6]),
    (64, 8, [0, 1, 2]), (40, 0, [0, 1]),
])
def test_a_slot_inside_its_columns(rows, first, slots):
    """A block moves 16 rows whole: the last slot of a leading dimension
    that is no multiple of 16 sticks out and stays per lane."""
    found = eb.Surface(pitch=4 * rows, columns=9, rows=rows)
    assert [s for s in range(-1, 10)
            if eb.slots_inside(found, first, s)] == slots


@pytest.mark.parametrize('lo,hi,expected', [
    (0, 9, [(0, 8), (8, 1)]), (0, 8, [(0, 8)]), (2, 9, [(2, 4), (6, 2), (8, 1)]),
    (0, 3, [(0, 2), (2, 1)]),
])
def test_the_messages_of_a_slot(lo, hi, expected):
    assert eb.chunks(lo, hi) == expected


def test_a_promise_covers_its_grain():
    ctx = SimpleNamespace(get_user_options=lambda: SimpleNamespace(
        element_base_align=32))
    assert not eb.needs_check(ctx, eb.Surface(224, 9, 56))
    assert eb.needs_check(ctx, eb.Surface(256, 9, 64))
    none = SimpleNamespace(get_user_options=lambda: SimpleNamespace(
        element_base_align=0))
    assert eb.needs_check(none, eb.Surface(224, 9, 56))


def test_the_header_comes_with_element_blocks():
    """`sycl_spmd.h` is included where a body reads or writes an element's
    operands by blocks, which the context says (`SyclLexic.element_blocks`)
    -- on its own, with none of the header's other users."""
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32,
                  options=Options(**ON))
    lex = ctx.target.lexic
    assert lex.element_blocks is True
    lex.split_broadcast = lex.dpas = lex.block_runs = lex.block2d = False
    assert 'tensorforge_device/sycl_spmd.h' in ctx.target.headers()
    lex.element_blocks = False
    assert 'tensorforge_device/sycl_spmd.h' not in ctx.target.headers()
