# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Record what SeisSol's code generator hands a GPU exporter (yateto interface 7), for one configuration.
  seissol_export.py EQUATION SOLVER ORDER PRECISION OUT.json [--mechanisms N]
Runs seissol21/codegen/generate.py (its yateto: codegen/yateto -> submodules/yateto) with TensorForge as
the device code generator, and swaps TensorForge's routine generator for one that records every kernel
description and generates nothing.  Kernels are named as yateto names them (`namespace::kernel`,
`family[group]` for a family member).  The CPU half is generated as SeisSol would (gemm tools: none)."""
import contextlib, io, json, os, pathlib, sys, tempfile, time

CODEGEN = pathlib.Path(os.environ.get("SEISSOL_CODEGEN", pathlib.Path(__file__).resolve().parents[2] / "seissol21" / "codegen"))
TF = pathlib.Path(__file__).resolve().parents[1] / "src"

equation, solver, order, precision, out = sys.argv[1:6]
out = os.path.abspath(out)          # the generator runs from the codegen directory
mechanisms = int(sys.argv[sys.argv.index('--mechanisms') + 1]) if '--mechanisms' in sys.argv else 0

sys.path.insert(0, str(TF))
sys.path.insert(0, str(CODEGEN))
os.chdir(CODEGEN)

import tensorforge                                                   # noqa: E402
from yateto.codegen.factory import ExportGenerator                  # noqa: E402
from yateto.codegen import visitor                                  # noqa: E402

# yateto (50200d3) describes `spaceTimePredictorRhs` twice in one poroelastic kernel, with alignment 64 and
# then 0, and asserts on it.  For the recording, a tensor described twice with nothing but its alignment
# differing keeps the lesser alignment -- the one both occurrences are true of -- and the config says so.
from yateto.codegen import factory as _factory                      # noqa: E402

workarounds = set()
_handle_tensor = _factory.ExportFactory._handleTensor


def _tolerant_handle_tensor(self, tensor, *args, **kwargs):
    stored = self.tensors.get(tensor['name'])
    if stored is not None and stored != tensor:
        rest = lambda d: {k: v for k, v in d.items() if k != 'alignment'}
        if rest(stored) == rest(tensor):
            least = min(stored.get('alignment') or 0, tensor.get('alignment') or 0)
            stored['alignment'] = tensor['alignment'] = least
            workarounds.add(f'{tensor["name"]}: alignment described twice, the lesser kept ({least})')
    return _handle_tensor(self, tensor, *args, **kwargs)


_factory.ExportFactory._handleTensor = _tolerant_handle_tensor

pending, recorded = [], {}


class Recorder(ExportGenerator):
    def add_kernel(self, description):
        pending.append(description)

    def generate(self, cpp, cache):
        cpp('// recorded for TensorForge, not generated')


def name_pending(name):
    """Called where yateto names what the outlines just built."""
    if not pending:
        return
    names = [name] if len(pending) == 1 else [f'{name}[{i}]' for i in range(len(pending))]
    for n, d in zip(names, pending):
        key, k = n, 2
        while key in recorded:
            key = f'{n}#{k}'; k += 1
        recorded[key] = d
    pending.clear()


original_generate = visitor.OptimizedKernelGenerator.generate


def named_generate(self, cpp, header, name, kernelOutlines, *args, **kwargs):
    name_pending(name)
    return original_generate(self, cpp, header, name, kernelOutlines, *args, **kwargs)


visitor.OptimizedKernelGenerator.generate = named_generate
tensorforge.get_routine_generator = lambda yateto: (lambda arch, attrs=None: Recorder(arch, attrs))

import generate                                                     # noqa: E402

start = time.monotonic()
with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ['generate.py', '--equations', equation, '--solver', solver, '--order', order,
                '--precision', precision, '--matricesDir', str(CODEGEN / 'matrices'), '--outputDir', tmp,
                '--host_arch', 'hsw', '--device_backend', 'cuda', '--device_arch', 'sm_86',
                '--device_vendor', 'nvidia', '--numMechanisms', str(mechanisms), '--memLayout', 'auto',
                '--multipleSimulations', '1', '--PlasticityMethod', 'nb', '--gemm_tools', 'none',
                '--device_codegen', 'tensorforge', '--drQuadRule', 'stroud']
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        generate.main()
name_pending('<unnamed>')

config = dict(equation=equation, solver=solver, order=int(order), precision=precision,
              mechanisms=mechanisms, device='cuda/sm_86', host='hsw', plasticity='nb', dr_quad='stroud',
              mem_layout='auto', multiple_simulations=1, workarounds=sorted(workarounds))
pathlib.Path(out).write_text(json.dumps({
    'source': ('SeisSol codegen (seissol21, master plus the damage and viscoacoustic systems) through '
               'seissol/yateto 50200d3, interface 7; recorded by tools/seissol_export.py'),
    'config': config,
    'descriptions': recorded}, separators=(',', ':')))
print(f'{equation} {solver} O{order} {precision}: {len(recorded)} kernels, '
      f'{time.monotonic() - start:.0f} s, {pathlib.Path(out).stat().st_size // 1024} kB')
