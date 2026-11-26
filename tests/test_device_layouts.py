# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The device layouts yateto generates this side's kernels against.

yateto gives a tensor a host layout and a device layout, the host's unless one
is set, and generates a kernel for the device against the device layouts only
for an exporter that declares it (`YatetoFrontend.DEVICE_LAYOUTS`).  The
description states them where it states every layout -- the storage, the
alignment -- so this side reads them as it reads any, and declaring it is all
there is to do here.
"""

from __future__ import annotations

import copy

import tensorforge
import seissol_suite
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend
from tensorforge.generators.generator import Generator


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "double"
    alignment = 64


def _derivative():
    """The o4 double-precision derivative, as yateto describes it with every
    tensor held as the host holds it: Q's 20 rows padded to 24 for 64 bytes."""
    return seissol_suite.description("elastic-linearck", "elastic-linearck-o4-d",
                                     "gpu_derivative")


def _narrowed(description, name):
    """The same description, with `name` held as a device whose columns are
    padded to 16 bytes only: 20 rows of doubles."""
    narrowed = copy.deepcopy(description)
    tensor = next(t for t in narrowed["tensors"] if t["name"] == name)
    assert tensor["storage"]["sizes"][0] == 24
    tensor["storage"]["shape"][0] = 20
    tensor["storage"]["sizes"][0] = 20
    tensor["alignment"] = 16
    return narrowed


def _kernel(description):
    descrs, _ = DescriptionReader(_Arch(), {}).read(description)
    generator = Generator(descrs, Context(arch="sm_86", backend="cuda",
                                          fp_type=Datatype.F64), attrs={})
    generator.generate()
    return generator.get_kernel()


class TestTheDeclaration:
    def test_the_frontend_declares_it_on_the_class(self):
        # yateto asks what is registered, before it makes an exporter
        assert YatetoFrontend.__dict__["DEVICE_LAYOUTS"] is True

    def test_it_is_what_yateto_is_handed(self):
        assert tensorforge.get_routine_generator(None).DEVICE_LAYOUTS is True


class TestADeviceLayout:
    def test_the_storage_stated_is_the_storage_read(self):
        _, cache = DescriptionReader(_Arch(), {}).read(_narrowed(_derivative(), "Q"))
        q = cache["Q"]
        assert q.shape[0] == 20
        assert list(q.get_bbox().upper()) == [20, 9]
        assert q.alignment == 16

    def test_the_kernel_addresses_it(self):
        assert _kernel(_narrowed(_derivative(), "Q")) != _kernel(_derivative())
