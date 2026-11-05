# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The launches gemmforge-chainforge makes of a kernel, built by gemmforge
and chainforge.

SeisSol generated its GPU kernels with those two before TensorForge
(`--device_codegen gemmforge-chainforge`): yateto groups runs of GEMMs into
chains, which chainforge builds as one kernel each, and hands every other
GEMM and every copy-scale-add to gemmforge, one kernel each.  This module
does what yateto does there, from the exported description TensorForge is
built from -- the same operations, the same layouts, the same constants --
so that what the two generators make of a kernel is measured on the same
input and in the same run (`Config.generator`).

## What yateto hands them

The cut is `suite.chain_partition`; the rest is here, read off yateto's
`codegen/fused_gemms/external_generator.py`, `codegen/gemm/gemmgen.py` and
`codegen/copyscaleadd/csa_gen.py`:

* a matrix is its storage box and the region an operation reaches in it.
  gemmforge counts rows and columns to the box's end (`produce_dense_matrix`),
  chainforge over the box's extent (`gen_dense_matrix`); both place the
  region relative to the box's start.
* a GEMM whose result and first operand both promise aligned columns has its
  row range rounded out to the alignment -- both ends for gemmforge
  (`gemm.Description`), the end only for chainforge (`FusedGemms`), which
  also keeps one matrix per tensor until a GEMM reaches it differently.
* a chain's temporaries are chainforge's own (`generate_tmp_matrix`); a
  factor known only at run time is gemmforge's argument by name
  (`_ForgeFactor`); a copy-scale-add runs over the range of what it reads.

Checked against yateto for SeisSol's elastic kernels at orders 3, 4 and 6, in
both precisions, for sm_86 and gfx90a, with and without chainforge: the same
matrices and factors reach the two generators, and the code they generate is
the same but for the routines' names -- which digest the matrices' printed
form, numpy's integers and all, and so differ from one numpy to the next.

## The translation unit

The entry points of `harness.driver_bench`, with the generated kernels and
launchers in the workload's namespace, every routine once.  A tensor is one
buffer for every launch that names it, laid out as the export stores it --
the layout TensorForge's kernels read -- and filled the same way: the
pattern, or the description's numbers for a batch-constant operand.  A
factor known only at run time is 0.5, as the TensorForge driver passes it.
So the checksum of `check` compares the generators on equal inputs.

`CHECK_ERR` is defined empty, which is what `-DNDEBUG` makes of
TensorForge's: the measurement build checks nothing either way.

Both generators are optional (`requirements-forge.txt`); without them, a
configuration that asks for them builds nothing and says why.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / 'src'), str(ROOT / 'tests')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from harness import driver_bench                                # noqa: E402
from harness.driver_bench import Buffer, Region                 # noqa: E402

from suite import Launch                                        # noqa: E402

#: The backends the two generators emit for: chainforge's lexicon knows
#: CUDA and HIP and nothing else.
BACKENDS = ('cuda', 'hip')

#: How the export spells an addressing mode, as the generators spell it.
ADDRESSING = {'n&+o&': 'pointer_based', 'n*N+o&': 'strided', '&': 'none'}

#: Bytes per element of the exported datatypes.
ELEMENT = {'f64': 8, 'f32': 4}

#: What a factor known only at run time is passed as, the TensorForge
#: driver's value (`driver_emit.collect_operands`).
RUNTIME_FACTOR = 0.5


def missing() -> Optional[str]:
    """Why the generators cannot be used here, or `None`."""
    absent = [name for name in ('gemmforge', 'chainforge')
              if importlib.util.find_spec(name) is None]
    if absent:
        return (f'{" and ".join(absent)} not importable; pip install -r '
                f'tools/bench/requirements-forge.txt')
    return None


def versions() -> Dict[str, str]:
    """The version of each generator, and the commit it was installed from
    where pip recorded one -- which for a generator nobody releases any more
    is the version."""
    import json
    from importlib import metadata
    out = {}
    for name in ('gemmforge', 'chainforge'):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            spec = importlib.util.find_spec(name)
            if spec is not None:
                out[name] = f'from {Path(spec.origin).parent}'
            continue
        text = dist.version
        try:
            commit = json.loads(dist.read_text('direct_url.json') or '{}')[
                'vcs_info']['commit_id']
            text += f' ({commit[:12]})'
        except (KeyError, TypeError, ValueError):
            pass
        out[name] = text
    return out


@dataclass
class Built:
    """A workload's translation unit, the symbol it is named by, and the
    kernel each of its launches runs, in order."""
    source: str
    symbol: str
    kernels: List[str]


class ForgeFactor(str):
    """A factor gemmforge takes as an argument of the routine: named by the
    argument, and spelled as zero in the routine's name, which has no room
    for a name (yateto's `_ForgeFactor`)."""

    def __int__(self):
        return 0


# ----------------------------------------------------------------------
# What a generator is handed
# ----------------------------------------------------------------------

def _layout(tensor: Dict) -> List[Tuple[int, int]]:
    """The storage box of a tensor, per dimension."""
    storage = tensor['storage']
    starts = storage.get('start') or [0] * len(storage['shape'])
    sizes = storage.get('sizes') or storage['shape']
    return [(a, a + b) for a, b in zip(starts, sizes)]


def _range(ref: Dict, dim: int) -> Tuple[int, int]:
    lo, hi = ref['bbox']
    return lo[dim], hi[dim]


def _aligned_upper(x: int, reals: int) -> int:
    return x + (reals - x % reals) % reals


def _factor(tensors: Dict[str, Dict], op: Dict):
    """The factor of an operation: its number, or its name where only the
    run knows it."""
    alpha = (op.get('linear') or {}).get('alpha')
    if alpha is None:
        return 1.0
    values = tensors[alpha['name']]['values']
    if values is None:
        return alpha['name']
    return float(values['data'][0][1]) if values['data'] else 0.0


def _beta(op: Dict) -> float:
    return 1.0 if (op.get('linear') or {}).get('add') else 0.0


def _gemmforge_matrix(gf, tensor: Dict, m: Tuple[int, int],
                      n: Tuple[int, int]):
    box = _layout(tensor)
    return gf.DenseMatrix(
        num_rows=box[0][1], num_cols=box[1][1],
        addressing=ADDRESSING[tensor['addressing']],
        bbox=[m[0] - box[0][0], n[0] - box[1][0],
              m[1] - box[0][0], n[1] - box[1][0]])


def _chainforge_matrix(cf, tensor: Dict, m: Tuple[int, int],
                       k: Tuple[int, int]):
    box = _layout(tensor)
    return cf.DenseMatrix(
        num_rows=box[0][1] - box[0][0], num_cols=box[1][1] - box[1][0],
        addressing=cf.Addressing.str2addr(ADDRESSING[tensor['addressing']]),
        bbox=[m[0] - box[0][0], k[0] - box[1][0],
              m[1] - box[0][0], k[1] - box[1][0]],
        alias=tensor['name'],
        is_tmp=bool(tensor['flags'].get('temporary')))


def _mnk(op: Dict) -> Tuple[Tuple[int, int], Tuple[int, int],
                            Tuple[int, int]]:
    """The ranges of a GEMM, from what its operands reach: `m` from the
    first, `n` from the second, `k` where they meet."""
    a, b = op['args']
    k = (max(_range(a, 1)[0], _range(b, 0)[0]),
         min(_range(a, 1)[1], _range(b, 0)[1]))
    return _range(a, 0), _range(b, 1), k


@dataclass
class Routine:
    """One launch as a generator makes it: the generator, which operations
    it computes, and the operands its call site names, in order."""
    kind: str
    generator: object
    operations: Tuple[int, ...]
    #: For gemmforge: the tensors of the call, (name, read-only) per matrix;
    #: chainforge names its own (`generate_call_site`).
    operands: Tuple[Tuple[str, bool], ...] = ()
    #: The factor gemmforge takes by argument, where it takes one.
    argument: Optional[object] = None

    @property
    def base(self) -> str:
        return self.generator.get_base_name()

    @property
    def launcher(self) -> str:
        return (f'launcher_{self.base}' if self.kind == 'chain'
                else self.base)

    @property
    def kernel(self) -> str:
        return f'kernel_{self.base}'


def lower(description: Dict, launch: Launch, backend: str, arch: str,
          datatype: str) -> Routine:
    """What yateto hands gemmforge or chainforge for one launch, registered
    with the generator -- named, not yet generated.

    `description` is the launch's own (`suite.split_description`): its
    temporaries are temporary where the launch alone touches them, and
    global arrays where another launch does.
    """
    import gemmforge as gf
    tensors = {t['name']: t for t in description['tensors']}
    ops = description['operations']
    fp = {'F64': 'double', 'F32': 'float'}[datatype]

    if launch.kind == 'csa':
        op, = ops
        term, result = op['args'][0], op['result']
        m, n = _range(term, 0), _range(term, 1)
        generator = gf.CsaGenerator(gf.vm_factory(arch, backend, fp_type=fp))
        alpha = _factor(tensors, op)
        generator.set(_gemmforge_matrix(gf, tensors[term['name']], m, n),
                      _gemmforge_matrix(gf, tensors[result['name']], m, n),
                      alpha, _beta(op))
        return Routine('csa', generator, launch.operations,
                       ((term['name'], True), (result['name'], False)),
                       argument=alpha)

    if launch.kind == 'gemm':
        op, = ops
        a, b = op['args']
        result = op['result']
        ta, tb, tc = (tensors[a['name']], tensors[b['name']],
                      tensors[result['name']])
        m, n, k = _mnk(op)
        if ta['alignment'] > 0 and tc['alignment'] > 0:
            reals = tc['alignment'] // ELEMENT[tc['datatype']]
            m = (m[0] - m[0] % reals, _aligned_upper(m[1], reals))
        factor = _factor(tensors, op)
        by_argument = isinstance(factor, str) or factor < 0
        alpha = ForgeFactor('alpha') if by_argument else float(factor)
        generator = gf.GemmGenerator(gf.vm_factory(arch, backend, fp_type=fp))
        generator.set(False, False, _gemmforge_matrix(gf, ta, m, k),
                      _gemmforge_matrix(gf, tb, k, n),
                      _gemmforge_matrix(gf, tc, m, n), alpha, _beta(op))
        return Routine('gemm', generator, launch.operations,
                       ((a['name'], True), (b['name'], True),
                        (result['name'], False)),
                       argument=factor if by_argument else None)

    if launch.kind == 'chain':
        import chainforge.common as cf
        from chainforge.backend.generator import Generator as ChainForge
        cache: Dict[str, object] = {}
        temporaries: Dict[str, object] = {}
        gemms = []

        def matrix(tensor: Dict, rows, cols):
            if tensor['flags'].get('temporary'):
                return temporaries[tensor['name']]
            made = _chainforge_matrix(cf, tensor, rows, cols)
            if not (tensor['name'] in cache
                    and made.is_same(cache[tensor['name']])):
                cache[tensor['name']] = made
            return cache[tensor['name']]

        for op in ops:
            a, b = op['args']
            result = op['result']
            ta, tb, tc = (tensors[a['name']], tensors[b['name']],
                          tensors[result['name']])
            m, n, k = _mnk(op)
            aligned = ta['alignment'] > 0 and tc['alignment'] > 0
            if aligned:
                reals = tc['alignment'] // ELEMENT[tc['datatype']]
                m = (m[0], _aligned_upper(m[1], reals))
            left, right = matrix(ta, m, k), matrix(tb, k, n)
            if tc['flags'].get('temporary'):
                out = cf.generate_tmp_matrix(op1=left, op2=right)
                temporaries[tc['name']] = cache[tc['name']] = out
            else:
                out = matrix(tc, m, n)
            factor = _factor(tensors, op)
            gemms.append(cf.GemmDescr(
                trans_a=False, trans_b=False, a=left, b=right, c=out,
                alpha=factor, beta=_beta(op), strict_match=False,
                prefer_align=aligned))
        context = cf.Context(arch=arch, backend=backend,
                             fp_type=cf.FloatingPointType.str2enum(fp))
        generator = ChainForge(gemms, context)
        generator.register()
        return Routine('chain', generator, launch.operations)

    raise NotImplementedError(
        f'operation {launch.operations[0]} is neither a GEMM nor a '
        f'copy-scale-add gemmforge takes')


# ----------------------------------------------------------------------
# The translation unit
# ----------------------------------------------------------------------

def _ctype(datatype: str) -> str:
    return {'F64': 'double', 'F32': 'float'}[datatype]


def _host(tensor: Dict) -> Optional[List[float]]:
    """A batch-constant tensor's numbers in its storage: column-major over
    its box, zero where the description lists no entry."""
    values = tensor.get('values')
    if (values is None or tensor['addressing'] != '&'
            or tensor['storage'].get('type') != 'bbox'):
        return None
    box = _layout(tensor)
    strides = [1]
    for lo, hi in box[:-1]:
        strides.append(strides[-1] * (hi - lo))
    volume = strides[-1] * (box[-1][1] - box[-1][0]) if box else 1
    out = [0.0] * volume
    for index, value in values['data']:
        if any(not lo <= i < hi for i, (lo, hi) in zip(index, box)):
            continue
        out[sum((i - lo) * s for i, (lo, _), s
                in zip(index, box, strides))] = float(value)
    return out


def _buffers(pieces: Sequence[Dict], datatype: str,
             regions: Optional[Dict[str, Region]]) -> Dict[str, Buffer]:
    """One buffer per tensor of the batch or batch-constant operator any
    launch names, by its name."""
    written = {op['result']['name'] for piece in pieces
               for op in piece['operations']}
    out: Dict[str, Buffer] = {}
    for piece in pieces:
        for tensor in piece['tensors']:
            name = tensor['name']
            if name in out or tensor['addressing'] not in ADDRESSING:
                continue
            if tensor['flags'].get('temporary'):
                continue        # a chain's own
            box = _layout(tensor)
            per = 1
            for lo, hi in box:
                per *= hi - lo
            out[name] = Buffer(
                stem=f'b{len(out)}', alias=name, ctype=_ctype(datatype),
                per=per, addressing=ADDRESSING[tensor['addressing']],
                written=name in written, host=_host(tensor),
                summed=(name in regions) if regions is not None
                else name in written,
                region=(regions or {}).get(name))
    return out


def _literal(value, ctype: str) -> str:
    number = RUNTIME_FACTOR if isinstance(value, str) else float(value)
    return f'static_cast<{ctype}>({number!r})'


def _gemmforge_argument(buffer: Buffer, read_only: bool) -> str:
    if buffer.addressing != 'pointer_based':
        return f'd_{buffer.stem}'
    if read_only and buffer.written:
        return f'const_cast<const {buffer.ctype}**>(d_p_{buffer.stem})'
    return f'd_p_{buffer.stem}'


def _chainforge_argument(buffer: Buffer) -> str:
    """chainforge's launchers take every matrix mutable, read or written;
    yateto casts the constness of a read one away, and so does this."""
    if buffer.addressing != 'pointer_based':
        return f'd_{buffer.stem}'
    if buffer.written:
        return f'd_p_{buffer.stem}'
    return f'const_cast<{buffer.ctype}**>(d_p_{buffer.stem})'


def _call(routine: Routine, buffers: Dict[str, Buffer], ctype: str) -> str:
    """The call of one launch, as yateto writes it with the bench's
    buffers: offsets zero, no batch flags, the stream the driver hands in."""
    stream = 'DEV_STREAM_PTR(stream)'
    if routine.kind == 'chain':
        names = {}
        for gemm in routine.generator.gemm_list:
            for matrix in (gemm.mat_a, gemm.mat_b, gemm.mat_c):
                if matrix.alias in buffers:
                    names[matrix.alias] = _chainforge_argument(
                        buffers[matrix.alias])
        last = routine.generator.gemm_list[-1]
        alpha, beta = last.alpha, last.beta
        call = routine.generator.generate_call_site(
            names, {name: '0' for name in names},
            alpha if isinstance(alpha, float) else _literal(alpha, ctype),
            beta if isinstance(beta, float) else _literal(beta, ctype),
            'batch', 'nullptr', stream)
        return f'  {call}'
    args = []
    if routine.kind == 'csa' or routine.argument is not None:
        args.append(_literal(routine.argument, ctype))
    for name, read_only in routine.operands:
        args += [_gemmforge_argument(buffers[name], read_only), '0']
    args += ['batch', 'nullptr', stream]
    return f'  {routine.launcher}({", ".join(args)});'


def build(pieces: Sequence[Tuple[Launch, Dict]], workload, unit,
          regions: Optional[Dict[str, Region]]) -> Built:
    """The translation unit of `workload` cut into `pieces`, every launch
    built by gemmforge or chainforge."""
    reason = missing()
    if reason:
        raise RuntimeError(reason)
    backend = unit.target.backend
    if backend not in BACKENDS:
        raise ValueError(f'gemmforge-chainforge emits CUDA and HIP, not '
                         f'{backend}')
    datatype = unit.datatype.name
    ctype = _ctype(datatype)
    routines = []
    with contextlib.redirect_stdout(io.StringIO()):
        for launch, piece in pieces:
            routines.append(lower(piece, launch, backend, unit.target.arch,
                                  datatype))
        code: Dict[str, str] = {}
        for routine in routines:
            if routine.base in code:
                continue
            routine.generator.generate()
            code[routine.base] = (routine.generator.get_kernel() + '\n'
                                  + routine.generator.get_launcher())
    buffers = _buffers([piece for _, piece in pieces], datatype, regions)
    decls, allocs, fills, frees = driver_bench.buffer_code(
        list(buffers.values()))
    launches = [_call(routine, buffers, ctype) for routine in routines]
    body = '\n'.join(code.values())
    symbol = 'forge_' + hashlib.sha256(
        '\n'.join([body] + launches).encode()).hexdigest()[:16]
    source = (driver_bench.tu_head(backend, '#ifndef CHECK_ERR\n'
                                   '#define CHECK_ERR\n#endif')
              + driver_bench.entry_points(
                  driver_bench.slug(workload.name), decls, allocs, fills,
                  frees, launches,
                  driver_bench.checksum_code(list(buffers.values())),
                  symbol, len(launches), body))
    return Built(source, symbol, [r.kernel for r in routines])
