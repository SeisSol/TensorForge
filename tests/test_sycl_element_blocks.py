# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.sycl_element_blocks`: an element's operands in global memory read
and written by 2D block messages of the sub-group over the element's own
storage (`memory.element_blocks`, `BatchLoop._emit_versions`,
`GlbToRegLoader._block_loops`, `StoreRegToGlb._block_loops`) -- L1-uncached
under `Options.sycl_element_streaming` -- and the next element's operands
hinted by 2D block prefetches under `Options.sycl_element_prefetch`.

What a wrong answer would come from: a block where today's transfer moves
fewer lanes -- a guard's ragged slot, a slot past the operand's columns -- or
a block that takes a read the pseudo-IR forwards from a store, or a body that
reads blocks without the element's check.  The kernels are held here against
the text the generator writes; what the device computes is held by tfemu
(`~/tf-probe/pvcport/s6`) and the Max 1550's.
"""

import contextlib
import functools
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


#: The IGC probe of the cache-opts forms, as the module has it before the
#: fixture below stands in for it.
PROBE = toolchain.intel_block2d_cacheopts


@pytest.fixture(autouse=True)
def _no_ocloc(monkeypatch):
    """Every build as on a machine with no `ocloc` to ask, as in
    `test_sycl_dpas`: what is written does not depend on the test machine's
    IGC."""
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: None)
    monkeypatch.setattr(toolchain, 'intel_block2d_cacheopts',
                        lambda *a, **k: None)


def _kernel(name='gpu_volume', order=6, precision='s', backend='oneapi',
            **options):
    descrs = DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-{precision}',
        name))[0]
    fp = Datatype.F64 if precision == 'd' else Datatype.F32
    return _generate(descrs, backend=backend, fp=fp, **options).get_kernel()


@functools.lru_cache(maxsize=None)
def _cached(name, order, pvc, items):
    """`_kernel` once for each kernel and options, on the suite's layout or
    SeisSol's PVC one (`_pvc`)."""
    options = dict(items)
    if not pvc:
        return _kernel(name, order, **options)
    return _text(_pvc(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-s', name)), **options)


def _once(name='gpu_volume', order=6, pvc=False, **options):
    return _cached(name, order, pvc, tuple(sorted(options.items())))


def _pvc(desc):
    """SeisSol's own layout on PVC: an element's I and Q in columns of 56
    rows, yateto's padding to 32 bytes, where the suite's is NVIDIA's 64 --
    3.5 lane slots a column, the last one past the column's end."""
    for tensor in desc['tensors']:
        storage = tensor['storage']
        if tensor['addressing'] == 'n&+o&' and storage.get('shape') == [64, 9]:
            storage['shape'] = [56, 9]
            storage['sizes'] = [min(storage['sizes'][0], 56), 9]
            tensor['alignment'] = 32
    return desc


def _text(desc, **options):
    descrs = DescriptionReader(None, {}).read(desc)[0]
    return _generate(descrs, **options).get_kernel()


def _generate(descrs, backend='oneapi', fp=Datatype.F32, **options):
    gen = Generator(descrs, Context(arch='pvc', backend=backend, fp_type=fp,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


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


def _block(text, at):
    """`(start, end)` of the text inside the braces that open at or after
    `at`."""
    head = re.compile(r'\{\n').search(text, at)
    depth, k = 1, head.end()
    while depth:
        brace = re.compile(r'[{}]').search(text, k)
        depth += 1 if brace.group(0) == '{' else -1
        k = brace.end()
    return head.end(), k - 1


def _bodies(text):
    """`[(then, else)]`: for every element body the check versions, the two
    bodies it chooses between, as text -- read off the braces after the `if`
    that follows the body's checks."""
    out = []

    def block(at):
        return _block(text, at)

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
def test_off_changes_nothing(dpas):
    """The option off writes the kernel it always wrote."""
    plain = _norm(_once(sycl_dpas=dpas))
    assert _norm(_kernel(sycl_dpas=dpas, sycl_element_blocks=False)) == plain
    assert 'elementBlock' not in plain


# -- auto -------------------------------------------------------------------- #

def test_auto_under_fma_changes_nothing():
    """Under FMA the blocks lost localFluxAll (pvcport J6b): `auto` writes
    the kernel the option off writes."""
    assert (_norm(_once(sycl_element_blocks='auto'))
            == _norm(_once(sycl_dpas=False)))


@pytest.mark.parametrize('name,order', [
    ('gpu_volume', 6), ('gpu_localFluxAll', 6), ('gpu_volume', 7),
], ids=['volume-6', 'lfa-6', 'volume-7'])
def test_auto_takes_the_blocks_where_they_paid(name, order):
    """Under DPAS the volume -- its Q accumulated three times, from row 1 --
    and localFluxAll at order 6, and the volume at order 7, six slots a
    column, where pvcport J7 found it 6-10 % faster: the kernel `True`
    writes."""
    assert (_norm(_once(name, order, sycl_dpas=True,
                        sycl_element_blocks='auto'))
            == _norm(_once(name, order, sycl_dpas=True, **ON)))


@pytest.mark.parametrize('name,order', [
    ('gpu_neighboringFlux[0]', 6), ('gpu_localFluxAll', 8),
], ids=['nf-6', 'lfa-8'])
def test_auto_keeps_where_the_blocks_lost(name, order):
    """The neighbor flux, which writes its Q once, and localFluxAll at
    order 8, eight slots a column -- both behind with the blocks in J7 --
    get the kernel the option off writes."""
    assert (_norm(_once(name, order, sycl_dpas=True,
                        sycl_element_blocks='auto'))
            == _norm(_once(name, order, sycl_dpas=True)))


def test_auto_keeps_the_derivative_as_it_was():
    """The derivative reads each derivative it wrote back for the next and
    for the integral: `auto` writes the kernel the option off writes."""
    assert (_norm(_once('gpu_derivative', sycl_dpas=True,
                        sycl_element_blocks='auto'))
            == _norm(_once('gpu_derivative', sycl_dpas=True)))


def test_auto_keeps_a_derivative_cut_in_two_as_it_was():
    """The derivative cut into two sections (`fission=2`): the section after
    the cut reads back derivatives the one before stored, a load of nothing
    it stored itself -- the chain is asked of the kernel's operations
    (`element_blocks.chain_refusal`), and `auto` writes the kernel the
    option off writes, both sections."""
    cut = _once('gpu_derivative', sycl_dpas=True, sycl_element_blocks='auto',
                fission=2)
    assert len(re.findall(r'for \(size_t v\d+_batchId0', cut)) == 2
    assert _norm(cut) == _norm(_once('gpu_derivative', sycl_dpas=True,
                                     fission=2))


@pytest.mark.parametrize('name,why', [
    ('gpu_volume', None), ('gpu_localFluxAll', None),
    ('gpu_neighboringFlux[0]', None),
    ('gpu_derivative', 'dQ(1) read back for I'),
])
def test_the_chain_is_asked_of_the_kernel(name, why):
    """A tensor of the element's own that one operation writes and a later
    one reads to write another: the derivative's first derivative, read
    for the integral.  The volume's temporaries are no such tensor, nor is
    an accumulator its own operation's destination."""
    assert eb.chain_refusal(_descrs(name)) == why


def _transfers(*spec):
    """`'l m0 0'`: a load (`l`) or store (`s`) of `glb_m0` from row 0."""
    out = []
    for word in spec:
        kind, name, row = word.split()
        out.append(({'l': 'load', 's': 'store'}[kind], 'glb_' + name,
                    int(row)))
    return out


#: The volume's transfers, as `BatchLoop._element_plan` hands them on: I
#: (m0) and a star matrix read, Q (m2) from row 1 read and written back, a
#: product at a time.
VOLUME = ('l m0 0', 'l m1 0', 'l m2 1', 's m2 1', 'l m0 0', 'l m3 0',
          'l m2 1', 's m2 1')
#: localFluxAll's: Q from row 0.
LFA = ('l m0 0', 'l m1 0', 'l m2 0', 's m2 0', 'l m0 0', 'l m3 0', 'l m2 0',
       's m2 0')
#: The neighbor flux's: I and the flux matrix read, Q read and written once.
NF = ('l m0 0', 'l m1 0', 'l m2 0', 's m2 0')
#: The derivative's first two: Q (m1) into the integral (m0) and the first
#: derivative (m5), which is read back to be added to the integral.
DERIVATIVE = ('l m1 0', 's m0 0', 'l m1 0', 'l m4 0', 's m5 0', 'l m1 0',
              'l m6 0', 'l m5 0', 's m5 0', 'l m5 0', 'l m0 0', 's m0 0')
ONCE = 'no element operand written more than once'


@pytest.mark.parametrize('transfers,rows,why', [
    (VOLUME, 56, None), (VOLUME, 64, None), (VOLUME, 96, None),
    (VOLUME, 120, 'glb_m0: 8 lane slots a column'),
    (LFA, 88, None), (LFA, 120, 'glb_m0: 8 lane slots a column'),
    (NF, 56, ONCE), (('s m2 1',), 56, ONCE),
    (('s m2 0', 'l m2 0', 's m2 0'), 64, None),
    (DERIVATIVE, 64, 'glb_m5 read back for glb_m0'),
], ids=['volume-56', 'volume-64', 'volume-96', 'volume-120', 'lfa-88',
        'lfa-120', 'nf', 'written-once', 'into-itself', 'derivative'])
def test_the_auto_rule(transfers, rows, why):
    """J7's rule on the transfers alone: an accumulation is no chain, an
    operand the body wrote read back for another one's product is the
    derivative's chain; at most six slots a column, whichever operand; and
    some operand written more than once."""
    ctx = SimpleNamespace(get_user_options=lambda: SimpleNamespace(
        sycl_dpas=True))
    surfaces = {n: eb.Surface(4 * rows, 9, rows) for n in ('glb_m0',
                                                           'glb_m2')}
    assert eb.auto_refusal(ctx, _transfers(*transfers), surfaces) == why


@pytest.mark.parametrize('value,asked,auto', [
    (False, False, False), (True, True, False), ('auto', False, True),
])
def test_on_and_auto_are_told_apart(value, asked, auto):
    """`True` asks for the blocks outright, `auto` leaves them to the rule
    (`auto_refusal`): both may take blocks, and only `True` offers the
    next element's 2D prefetch (`Options.sycl_element_prefetch`)."""
    ctx = SimpleNamespace(get_user_options=lambda: SimpleNamespace(
        sycl_element_blocks=value))
    assert eb.asked(ctx) is asked and eb.auto(ctx) is auto
    assert eb.wanted(ctx) is (asked or auto)


@pytest.mark.parametrize('options,why', [
    (dict(), None), (dict(backend='esimd'), 'not SPMD SYCL under oneAPI'),
    (dict(backend='acpp'), 'not SPMD SYCL under oneAPI'),
    (dict(fp=Datatype.F64), 'not single precision'),
    (dict(arch='sm_90', backend='cuda'), 'not SYCL'),
], ids=['oneapi', 'esimd', 'acpp', 'f64', 'cuda'])
def test_the_targets_with_blocks(options, why):
    """SPMD SYCL under oneAPI on Intel, its sub-group of 16 stated, F32:
    the explicit vector has a backend of its own, and so has AdaptiveCpp."""
    arch = options.pop('arch', 'pvc')
    fp = options.pop('fp', Datatype.F32)
    ctx = Context(arch=arch, backend=options.pop('backend', 'oneapi'),
                  fp_type=fp, options=Options(**ON))
    answer = eb.target_refusal(ctx)
    assert (answer is None) if why is None else (why in (answer or '')), answer


def test_the_auto_rule_wants_dpas():
    ctx = SimpleNamespace(get_user_options=lambda: SimpleNamespace(
        sycl_dpas=False))
    assert eb.auto_refusal(ctx, _transfers(*LFA), {}) == 'not the DPAS path'


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
    kernel = _once(sycl_dpas=dpas, **ON)
    checks = {(m.group(3), int(m.group(2))) for m in CHECK.finditer(kernel)}
    assert checks == {('glb_m0', 256), ('glb_m2', 256)}
    (then, otherwise), = _bodies(kernel)
    assert READ.search(then) and WRITE.search(then)
    assert 'elementBlock' not in otherwise


def _element_body(text):
    """Today's element body: what the block of the element's bindings holds
    after them."""
    first = re.search(r'glb_m\d+ = &m\d+\[v\d+_batchId', text)
    start = text.rindex('{\n', 0, first.start())
    start, end = _block(text, start)
    return '\n'.join(line for line in text[start:end].split('\n')
                     if not re.search(r'= &m\d+\[v\d+_batchId', line))


def _same_text(text):
    """A body's lines but for the names the IR numbers and the comments."""
    text = re.sub(r'\bv\d+_', 'v_', text)
    text = re.sub(r'\b(i?r)\d+\b', r'\1', text)
    return [line.strip() for line in text.split('\n')
            if line.strip() and not line.strip().startswith('//')]


@pytest.mark.parametrize('name', ['gpu_volume', 'gpu_derivative'])
def test_the_fallback_is_todays_body(name):
    """Behind the check's failure today's body, the text it always was but
    for the names the IR numbers: the two arms exclude each other, so a
    store of the block body is no store to a read of the fallback
    (`pir.flow`), which forwards what today's body forwards -- pvcport J6b's
    fallback, without it, kept reads today's body does not make and ran up
    to 25 % slower, the derivative 69 %."""
    bodies = _bodies(_once(name, sycl_dpas=True, **ON))
    assert len(bodies) == 1
    (_, otherwise), = bodies
    assert (_same_text(otherwise)
            == _same_text(_element_body(_once(name, sycl_dpas=True))))
    assert 'elementBlock' not in otherwise


def test_the_two_bodies_weigh_as_the_larger():
    """The code fission and the instruction cache weigh counts the arm an
    element runs, not both (`Emitter._one_arm_of_two`): the volume's
    versioned body about today's, not twice it."""
    plain = _generate(_descrs('gpu_volume'), sycl_dpas=True).metrics
    both = _generate(_descrs('gpu_volume'), sycl_dpas=True, **ON).metrics
    assert plain.code_units <= both.code_units < 1.05 * plain.code_units
    assert (plain.written_code_units <= both.written_code_units
            < 1.05 * plain.written_code_units)


def test_the_two_bodies_split_no_kernel_today_s_does_not():
    """The order-6 derivative stays one section with the blocks, as without
    -- J6b's two bodies cut it in two, and it lost for that."""
    loops = re.compile(r'for \(size_t v\d+_batchId0')
    assert len(loops.findall(_once('gpu_derivative', sycl_dpas=True,
                                   **ON))) == 1
    assert len(loops.findall(_once('gpu_derivative', sycl_dpas=True))) == 1


def _descrs(name, order=6):
    return DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-s', name))[0]


@pytest.mark.parametrize('dpas', [False, True], ids=['fma', 'dpas'])
def test_a_block_is_a_slot_of_nine_columns(dpas):
    """Eight columns a message and the ninth one of its own, the element's
    columns 256 bytes apart: the order-6 volume's I and Q."""
    (then, _), = _bodies(_once(sycl_dpas=dpas, **ON))
    shapes = {(int(r), int(p), int(c)) for r, p, c, _ in READ.findall(then)}
    assert shapes == {(8, 256, 9), (1, 256, 9)}
    assert {op for *_, op in READ.findall(then)} == {'glb_m0', 'glb_m2'}


def test_the_nine_by_nine_matrices_keep_their_reads():
    """A star matrix's column is 36 bytes: no surface (the messages want 64
    and a whole number of 16)."""
    kernel = _once(sycl_dpas=True, **ON)
    for name in ('glb_m1', 'glb_m3', 'glb_m4'):
        assert not re.search(r'elementBlock2d<[^>]*>\(item\.get_sub_group\(\), '
                             + name + ',', kernel)


def test_a_guarded_slot_of_an_accumulation_keeps_its_lanes():
    """Q's final write: the slots whose every lane the store writes go by
    blocks, the ragged ones of the accumulation -- row 0 and the rows past
    53 -- lane by lane, under their guards."""
    (then, _), = _bodies(_once(sycl_dpas=True, **ON))
    tail = then[then.rindex('glb_m2 = store{r>g}'):]
    assert WRITE.search(tail)
    assert re.search(r'if \(v\d+_g\) \{\s*glb_m2\[', tail)


@pytest.mark.parametrize('form', [{}, dict(element_base_align=64),
                                  dict(sycl_dpas_block2d=True,
                                       sycl_core_threads=32)],
                         ids=['dpeb', 'dpebp', 'dpbt32eb'])
def test_a_held_element_of_a_block_is_read_before_its_select(form):
    """IGC 2.41.5's form (`Options.unguard_window_loads`): the last slot of
    the order-7 I and Q, rows 80-83 of 96, read by its block and held to its
    lanes by a select over the element in a variable of its own, never over
    `v_eblk[3]` inline -- the front end reads an inline element in the
    select's true arm alone, and the select is the guarded write the option
    is there to avoid (`GlbToRegLoader._block_loops`).  pvcport J8: the
    neighbor flux on the bench layout wrong in one cell of every element
    under IGC 2.41.5, the ninth column's chain sunk into that branch."""
    kernel = _once('gpu_neighboringFlux[0]', 7, sycl_dpas=True,
                   unguard_window_loads=True, **form, **ON)
    assert not re.search(r'\? \(v\d+_eblk\[\d+\]\)', kernel)
    named = re.findall(r'float (v\d+_el) = v\d+_eblk\[\d+\];', kernel)
    # I's and Q's, eight columns each: the ninth is a block of its own, a
    # float; the fallback body reads as today.
    assert len(named) == 2 * 8
    for name in named:
        assert re.search(r'= \(v\d+_g \? ' + name + r' : 0\.0f\);', kernel)


def test_without_the_held_form_the_element_keeps_its_guard():
    """Another IGC than 2.41.5: the slot's elements written under the
    lane guard as before, nothing named."""
    kernel = _once('gpu_neighboringFlux[0]', 7, sycl_dpas=True,
                   unguard_window_loads=False, element_base_align=64, **ON)
    assert not re.search(r'float v\d+_el = v\d+_eblk\[', kernel)
    assert re.search(r'if \(v\d+_g\) \{\s*r\d+\[\d+\] = \(v\d+_eblk\[0\]\);',
                     kernel)


def test_a_read_after_a_write_stays_a_read_the_ir_forwards():
    """The derivative writes each derivative and reads it back: those reads
    stay per lane, for `pir.forward` to take from the registers, and so do
    the writes a read follows -- in each of its bodies, no read of an
    operand after its first block write, and no block read after a write."""
    bodies = _bodies(_once('gpu_derivative', sycl_dpas=True, **ON))
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


def test_a_transfer_inside_a_region_keeps_its_lanes():
    """The volume's last product under a guard of the element's own (a
    `GuardedRegion`): its transfers -- Q's last write among them -- stay
    lane by lane; the blocks take only the body's own transfers."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GuardLiteral
    descrs = _descrs('gpu_volume')
    flag = SubTensor(Tensor([1], Addressing.PTR_BASED, BoundingBox([0], [1]),
                            alias='c', datatype=Datatype.BOOL))
    descrs[-1].condition = [GuardLiteral(flag)]
    kernel = _generate(descrs, sycl_dpas=True, **ON).get_kernel()
    (then, _), = _bodies(kernel)
    guard = re.search(r'bool (v\d+_data) = glb_m\d+\[0\];\s*if \(\1\) ',
                      then)
    start, end = _block(then, guard.end())
    assert READ.search(then[:guard.start()])
    assert 'glb_m2 = store{r>g}' in then[start:end]
    assert 'elementBlock' not in then[start:end]


def test_a_promise_takes_the_check_away():
    """`element_base_align` at the surfaces' grain or above: blocks in the
    one body, no check; below it the check stays."""
    kernel = _once(sycl_dpas=True, element_base_align=64, **ON)
    assert READ.search(kernel) and not CHECK.search(kernel)
    assert CHECK.search(_once(sycl_dpas=True, element_base_align=32, **ON))


def test_the_kernels_parse():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    for name, pvc in (('gpu_volume', False), ('gpu_derivative', False),
                      ('gpu_volume', True), ('gpu_neighboringFlux[0]', True)):
        result = syntax.check_source(_once(name, pvc=pvc, sycl_dpas=True,
                                           **ON),
                                     shim=syntax.SYCL_SHIM)
        assert result.ok, (name, pvc, result.errors())


# -- the last slot of a column of no whole number of slots ------------------- #

READ_AT = re.compile(r'tensorforge::elementBlock2d<float, (\d), (\d+), (\d+)>'
                     r'\(item\.get_sub_group\(\), (glb_m\d+), (\w+|\([^)]*\)), '
                     r'(\d+)\)')
WRITE_AT = re.compile(r'tensorforge::elementBlock2dStore<float, (\d), (\d+), '
                      r'(\d+)>\(item\.get_sub_group\(\), (glb_m\d+), (\w+), '
                      r'(\d+), (\w+)\)')


def test_the_last_slot_of_a_ragged_column_is_the_block_before_its_end():
    """The neighbor flux on SeisSol's PVC layout, 56 rows a column: rows
    48-55 of I and Q read as the block of rows 40-55, rotated down by 8
    lanes (`sycl::shift_group_left`); Q written back as that block, its
    lanes 0-7 the slot before's upper ones rotated down, 8-15 the slot's own
    rotated up (`shift_group_right`), merged by lane -- and no lane of
    either reads or writes alone, where J6b's blocks left the slot nine
    messages a lane."""
    (then, otherwise), = _bodies(_once('gpu_neighboringFlux[0]', pvc=True,
                                       sycl_dpas=True, **ON))
    reads = {(m.group(4), m.group(5)) for m in READ_AT.finditer(then)}
    assert ('glb_m0', '40') in reads and ('glb_m2', '40') in reads
    assert {m.group(2) for m in READ_AT.finditer(then)} == {'224'}
    writes = {(m.group(4), m.group(5)) for m in WRITE_AT.finditer(then)}
    assert ('glb_m2', '40') in writes
    block = re.search(r'sycl::vec<float, 8> (v\d+_eblk) = tensorforge::'
                      r'elementBlock2d<float, 8, 224, 9>\(item\.get_sub_group'
                      r'\(\), glb_m0, 40, 0\);', then).group(1)
    assert re.search(r'sycl::shift_group_left\(item\.get_sub_group\(\), \('
                     + block + r'\[0\]\), 8\)', then)
    merge = re.search(r'float (v\d+)_emerge = (v\d+_\w+) \? (v\d+)_erot : '
                      r'(v\d+)_erot;', then)
    assert merge
    assert re.search(r'float ' + merge.group(3) + r'_erot = sycl::'
                     r'shift_group_left\(item\.get_sub_group\(\), \w+, 8\);',
                     then)
    assert re.search(r'float ' + merge.group(4) + r'_erot = sycl::'
                     r'shift_group_right\(item\.get_sub_group\(\), \w+, 8\);',
                     then)
    assert not re.search(r'glb_m[02]\[', then)
    assert re.search(r'glb_m2\[', otherwise)


def test_a_padded_tail_is_written_whole():
    """The derivative's copy of Q into its first derivative, 56 rows of 64:
    today's write zeroes rows 56-63 with the data, so its last slot stays
    the whole block at row 48 -- the block that ends with row 55 would
    leave them as they were."""
    (then, _), = _bodies(_once('gpu_derivative', sycl_dpas=True, **ON))
    at = {m.group(5) for m in WRITE_AT.finditer(then)
          if m.group(4) == 'glb_m3'}
    assert '48' in at and '40' not in at


@pytest.mark.parametrize('rows,first,shifts', [
    (56, 0, {3: 8}), (120, 0, {7: 8}), (64, 0, {}), (64, 1, {3: 1}),
    (56, 1, {3: 9}), (40, 0, {2: 8}), (16, 0, {}),
])
def test_the_lanes_a_ragged_slot_lies_past_its_block(rows, first, shifts):
    """A slot leaving its columns at their end is the block of the 16 rows
    that end with them, its lane `l` that block's lane `l + shift`; inside
    the columns, or before them, it has none."""
    found = eb.Surface(pitch=4 * rows, columns=9, rows=rows)
    assert {s: eb.overlap_shift(found, first, s) for s in range(-1, 10)
            if eb.overlap_shift(found, first, s) is not None} == shifts


def test_the_rotation_is_spelled_by_the_sycl_lexic():
    """Down the sub-group by `shift_group_left`, up by `shift_group_right`;
    the explicit vector, and a lexic with no sub-group, spell none."""
    def lexic(backend):
        return Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                       options=Options(**ON)).target.lexic
    spmd = lexic('oneapi')
    assert (spmd.sub_group_shift('{0}', 8)
            == 'sycl::shift_group_left(item.get_sub_group(), {0}, 8)')
    assert (spmd.sub_group_shift('{0}', 3, down=False)
            == 'sycl::shift_group_right(item.get_sub_group(), {0}, 3)')
    assert lexic('esimd').sub_group_shift('{0}', 8) is None
    cuda = Context(arch='sm_90', backend='cuda',
                   fp_type=Datatype.F32).target.lexic
    assert cuda.sub_group_shift('{0}', 8) is None


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
    that is no multiple of 16 sticks out, and is the block that ends with
    the column (`overlap_shift`)."""
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


# -- streaming and the 2D prefetch ------------------------------------------- #

STREAMED = re.compile(r'tensorforge::elementBlock2d(Store)?<float, \d, \d+, '
                      r'\d+, (\d)>\(item\.get_sub_group\(\), glb_m\d+, ')
PREFETCH2D = re.compile(r'tensorforge::elementPrefetch2d<(\d+)>\('
                        r'item\.get_sub_group\(\), pf_(glb_m\d+)\);')
STREAM = dict(sycl_element_streaming=True, **ON)
AHEAD = dict(sycl_element_prefetch=True, prefetch_data=True, **ON)


def test_streaming_puts_the_cache_option_on_every_block():
    """Each block read and write of the element through IGC's cache-opts
    form, L1 uncached (2); the per-lane transfers and the fallback as they
    were."""
    kernel = _once(sycl_dpas=True, **STREAM)
    plain = _once(sycl_dpas=True, **ON)
    blocks = STREAMED.findall(kernel)
    assert blocks and {c for _, c in blocks} == {'2'}
    assert {w for w, _ in blocks} == {'', 'Store'}
    assert not READ.search(kernel) and not WRITE.search(kernel)
    assert len(blocks) == len(READ.findall(plain)) + len(WRITE.findall(plain))
    assert ([o for _, o in _bodies(kernel)]
            == [o for _, o in _bodies(plain)])


def test_streaming_without_blocks_changes_nothing(monkeypatch):
    """Nothing to stream where no block is read: the kernel it always got,
    and no IGC asked."""
    def asked(*args, **kwargs):
        raise AssertionError('a build without blocks asked for the IGC')
    monkeypatch.setattr(toolchain, 'intel_block2d_cacheopts', asked)
    plain = _norm(_kernel(sycl_dpas=True))
    assert _norm(_kernel(sycl_dpas=True, sycl_element_streaming=True)) == plain
    assert _norm(_kernel(sycl_dpas=True, sycl_element_prefetch=True,
                         prefetch_data=True)) == _norm(
        _kernel(sycl_dpas=True, prefetch_data=True))


@pytest.mark.parametrize('options,option', [
    (STREAM, 'sycl_element_streaming'), (AHEAD, 'sycl_element_prefetch'),
], ids=['streaming', 'prefetch'])
def test_an_igc_without_the_forms_is_named_at_generation(monkeypatch, options,
                                                         option):
    """Never a quiet return to the messages without the cache option: the
    option and the reason."""
    from tensorforge.common.exceptions import GenerationError
    monkeypatch.setattr(toolchain, 'intel_block2d_cacheopts',
                        lambda *a, **k: False)
    with pytest.raises(GenerationError, match=option + '.*cache-opts'):
        _kernel(sycl_dpas=True, **options)
    monkeypatch.setattr(toolchain, 'intel_block2d_cacheopts',
                        lambda *a, **k: True)
    assert _kernel(sycl_dpas=True, **options)


@pytest.mark.parametrize('order,nbytes', [(6, 2304), (8, 4608)])
def test_the_next_elements_i_and_q_by_2d_prefetches(order, nbytes):
    """The operands with an element surface, I and Q, by one 2D prefetch of
    their storage each; the star matrices keep a hint a lane."""
    kernel = _once(order=order, sycl_dpas=True, **AHEAD)
    assert sorted(PREFETCH2D.findall(kernel)) == [
        (str(nbytes), 'glb_m0'), (str(nbytes), 'glb_m2')]
    lanes = _once(order=order, sycl_dpas=True, prefetch_data=True, **ON)
    assert not PREFETCH2D.search(lanes)
    assert '&pf_glb_m0[' in lanes and '&pf_glb_m0[' not in kernel
    assert '&pf_glb_m1[' in kernel


def test_the_probe_reads_the_binary(tmp_path):
    """The forms resolved: a yes.  Left undefined, or a failed build: a no.
    No `ocloc` at all: no answer."""
    from test_sycl_dpas import _ocloc
    probe = PROBE
    patch = ('__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0', False)
    dirs = {}
    for name in ('old', 'new', 'failed'):
        dirs[name] = tmp_path / name
        dirs[name].mkdir()
    undefined = (toolchain.BLOCK2D_CACHEOPTS_BUILTINS[2], False)
    assert probe(ocloc=_ocloc(dirs['old'], [('probe', True), undefined,
                                            patch])) is False
    assert probe(ocloc=_ocloc(dirs['new'], [('probe', True), patch])) is True
    assert probe(ocloc=_ocloc(dirs['failed'], [('probe', True), patch],
                              status=1)) is False
    assert probe(ocloc=str(tmp_path / 'nowhere')) is None


def test_the_streamed_and_prefetching_kernels_parse():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    for options in (STREAM, AHEAD):
        result = syntax.check_source(_once(sycl_dpas=True, **options),
                                     shim=syntax.SYCL_SHIM)
        assert result.ok, (options, result.errors())
