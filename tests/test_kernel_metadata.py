# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a generated kernel says about itself, and where.

The launch -- block, multiplications per block, shared memory, cooperative or
not -- is decided once (`Generator.launch_config`), published in the header
(`launch_info_<kernel>` and the `launch_config_<kernel>` function that adds
the grid), and the launcher launches what that function decides.  The
kernel's comment block states the launch, the operands and the operations in
index notation, and the same as data on one `tensorforge-meta` line.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from test_amd_blgp import _kernel_at

from tensorforge.generators.launch import launch_info_initializer

ROOT = Path(__file__).resolve().parent.parent


def _generator(name, backend, arch, **options):
    import contextlib
    import io
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    from test_amd_packed_dpp import CASES
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_meta__' + name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    ctx = Context(arch=arch, backend=backend, fp_type=case.DTYPE,
                  options=Options(**options))
    gen = Generator(case.descr_list(), ctx, attrs=getattr(case, 'ATTRS', None))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _meta(src):
    line = next(ln for ln in src.splitlines()
                if ln.strip().startswith('// tensorforge-meta:'))
    return json.loads(line.split(':', 1)[1])


@pytest.mark.parametrize('backend,arch', [('cuda', 'sm_86'), ('hip', 'gfx942'),
                                          ('esimd', 'pvc')])
def test_the_header_states_the_launch_the_launcher_uses(backend, arch):
    gen = _generator('local_flux', backend, arch)
    name, header, launcher = (gen.get_base_name(), gen.get_header(),
                              gen.get_launcher())
    config = gen.launch_config()
    assert (f'launch_info_{name} = {launch_info_initializer(config)};'
            in header)
    assert re.search(rf'tensorforge::LaunchConfig launch_config_{name}\('
                     r'size_t numElements0, void\* streamPtr = nullptr\);',
                     header)
    assert header.count('void launcher_') == 1
    # the launcher launches what the function decides
    assert f'= launch_config_{name}(numElements0, streamPtr);' in launcher
    assert f'config.block[1] = {config.block[1]};' in launcher
    assert config.threads_per_block == (config.block[0] * config.block[1]
                                        * config.block[2])
    assert config.shared_bytes == config.shared_elements * 4


def test_the_types_are_defined_once_however_many_headers_carry_them():
    header = _generator('local_flux', 'cuda', 'sm_86').get_header()
    assert header.startswith('#ifndef TENSORFORGE_LAUNCH_TYPES\n')
    assert header.count('struct LaunchConfig {') == 1


def test_the_comment_block_states_launch_operations_and_metadata():
    gen = _generator('local_flux', 'cuda', 'sm_86')
    src = gen.get_kernel()
    assert f'// launch: {gen.launch_config().describe()}' in src
    assert re.search(r'//   [mt]\d+\[i,j\] \+?= [mt]\d+\[i,k\] × [mt]\d+\[k,j\]',
                     src)
    meta = _meta(src)
    assert meta == gen.kernel_info()
    assert meta['launch'] == gen.launch_config().to_dict()
    assert len(meta['operations']) == sum(len(d.operations())
                                          for d in gen.descr_list)


def test_a_merged_run_states_its_members():
    src = _kernel_at('local_flux', 'gfx942', merge_variants=True)
    assert re.search(r'//   for \d+ iterations', src)
    assert re.search(r'//     v\d+ ∈ \{m\d+(, m\d+)+\}', src)
    loops = _meta(src)['loops']
    assert loops and loops[0]['kind'] == 'for'
    assert len(loops[0]['holes'][0]['members']) == loops[0]['iterations']


def test_the_host_parser_reads_the_operations_back():
    spec = importlib.util.spec_from_file_location(
        'parse_generated', ROOT / 'tools' / 'host' / 'parse_generated.py')
    parse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parse)
    src = _kernel_at('local_flux', 'gfx942')
    descrs = parse.parse_kernel(src)
    assert len(descrs) == len(_meta(src)['operations'])
    assert all(' × ' in d.summary() for d in descrs)
