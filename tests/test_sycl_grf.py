# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register file a SYCL kernel on Intel asks for (`Options.sycl_grf`).

IGC gives a thread 128 general registers by default, and almost every large
TensorForge kernel of SeisSol spills there (oneAPI 2025 on PVC); with 256 the
spills shrink and most of them run faster -- not all: the volume kernel went
from 187 to 202 us a call, the threads in flight halved.  So it is a choice a
kernel at a time, stated as a DPC++ kernel property, and nothing where the
compiler is not DPC++.
"""
import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / 'cases'
PROPERTY = re.compile(r'sycl::ext::oneapi::experimental::properties\{'
                      r'sycl::ext::intel::experimental::(grf_size<(\d+)>|'
                      r'grf_size_automatic)\}, \[=\]')


def _kernel(backend='oneapi', arch='pvc', **options):
    path = next(CASES.rglob('addressing_none.py'))
    spec = importlib.util.spec_from_file_location('tf_grf__case', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    gen = Generator(case.descr_list(),
                    Context(arch=arch, backend=backend, fp_type=case.DTYPE,
                            options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), gen.get_helper_headers()


def _asked(src):
    m = PROPERTY.search(src)
    if m is None:
        return None
    return m.group(2) or 'auto'


@pytest.mark.parametrize('grf, asked', [('default', None), ('128', '128'),
                                        ('256', '256'), ('auto', 'auto')])
def test_spmd_asks_for_what_it_is_told(grf, asked):
    src, headers = _kernel(sycl_grf=grf)
    assert _asked(src) == asked
    assert len(PROPERTY.findall(src)) == (asked is not None)


def test_the_explicit_vector_takes_the_large_file_unless_told_otherwise():
    """Its register images are sized for 256 (`esimd-register-image-fit`)."""
    assert _asked(_kernel(backend='esimd')[0]) == '256'
    assert _asked(_kernel(backend='esimd', sycl_grf='128')[0]) == '128'


def test_only_dpcpp_on_intel_states_it():
    """AdaptiveCpp has neither the property nor its header."""
    src, _ = _kernel(backend='acpp', sycl_grf='256')
    assert _asked(src) is None


def test_an_unknown_file_is_refused():
    with pytest.raises(ValueError):
        _kernel(sycl_grf='192')
