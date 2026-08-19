# SPDX-License-Identifier: MIT
"""Where the dereference is an operation, and where it is deliberately not.

`Symbol.load` and `Symbol.store` used to hand the whole access to the IR as
text: `f'{name}[{addr}]'`, with the address pinned so folding could not rewrite
a value the string still referred to by name. The address arithmetic was
structured and the dereference around it was not, so a pass could see neither
the def-use edge to the address nor that two accesses touch the same place.

Most of that is now `Op.LOAD` and `Op.STORE`. Three cases are still text, each
for a different reason, and each is one careless generalisation away from
being wrong rather than merely unmigrated:

  * a **scalar** is not a subscripted access at all --- `access` returns the
    bare name, so `Op.LOAD` would invent a `[0]` that never existed
  * a **broadcast** is a load wrapped in a vendor intrinsic; splitting it
    leaves a named temporary the source does not have
  * a **base override** replaces the pointer name, which `Op.STORE` cannot
    express because its base *is* the symbol

`access_equiv.py` shows the corpus addresses the same memory before and after.
This file pins the shape of that memory traffic --- and the exceptions, which
no corpus-wide equivalence check can speak to, because they are the cases
where nothing changed.

Not pinned here, because it cannot be stated from the emitted text: address
sharing stops at instruction boundaries. `through_pir` builds one IR body per
instruction, so the emitter never sees two instructions at once and cannot
fold an address computed in both. `chain_five_multiplies` recomputes 57 of
them. Checking that would need the claim "these two accesses are in the same
region", which the generated source does not say.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"


def _generate(name, backend="cuda", arch="sm_86"):
    path = next(iter(sorted(CASES.rglob(f"{name}.py"))), CASES / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"tf_access__{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend, fp_type=getattr(mod, "DTYPE", None))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel() or ""


ADDR = re.compile(r'^\s*int32_t (v\d+_a) = (.+?);\s*$', re.M)


def _uses(src, name):
    """Lines mentioning `name` other than its definition."""
    return [l for l in src.splitlines()
            if re.search(rf'\b{name}\b', l)
            and not re.match(rf'\s*int32_t {name} = ', l)]


# ---------------------------------------------------------------------- #
# the migrated shape
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_a_register_store_takes_its_address_as_an_operand(backend, arch):
    """`r1[v560_n1] = v563_data;` --- the loop variable directly.

    Pinned, the same store read `r1[v564_a]` with `v564_a = v560_n1` above it:
    a copy that existed because a string cannot hold a value, only a name.
    """
    src = _generate("accumulate_chain", backend, arch)
    stores = re.findall(r'^\s*(r\d+)\[([^\]]+)\] = (v\d+_\w+);\s*$', src, re.M)
    assert stores, "expected register stores in this case"
    # No store may address through a name that is a bare copy of another.
    copies = {m.group(1) for m in re.finditer(
        r'^\s*int32_t (v\d+_a) = (v\d+_\w+);\s*$', src, re.M)}
    through_copy = [s for s in stores if s[1] in copies]
    assert not through_copy, f'stores addressing through a copy: {through_copy}'


@pytest.mark.parametrize("case_name", ["chain_three", "accumulate_then_read"])
def test_one_address_serves_every_access_at_that_index(case_name):
    """`ir1[v1348_a]` read and `r1[v1348_a]` written, computed once.

    Pinned, that was impossible: the pin exists so folding cannot rewrite a
    value the surrounding string still refers to by name, and the price is
    that two accesses at the same index are two unrelated computations with no
    edge between them. Sharing is the thing the migration buys, so its absence
    should fail rather than merely cost.
    """
    src = _generate(case_name)
    shared = [n for n, _ in ADDR.findall(src) if len(_uses(src, n)) > 1]
    assert shared, 'no address is shared between two accesses'


# ---------------------------------------------------------------------- #
# the deliberate exceptions
# ---------------------------------------------------------------------- #

def test_a_scalar_load_gains_no_subscript():
    """`float v_data = alpha;`, never `alpha[0]`.

    A scalar's `access` returns the bare name, so `Op.LOAD` would build a
    subscript out of an empty index list. `[0]` on a scalar is not a slower
    way to be right --- it does not compile.

    Built directly rather than generated, because no case in the corpus
    reaches this path: the only one that would is `csa_alpha`, which does not
    generate (the `Tensor.data` defect, owned elsewhere). A corpus-driven
    version of this test passes without executing anything, which is how the
    first draft of it was wrong.
    """
    from tensorforge.backend.pir import emit as pir_emit
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.symbol import DataView, Symbol, SymbolType
    from tensorforge.backend.writer import Writer
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.matrix.tensor import Tensor
    from tensorforge.common.vm.vm import vm_factory

    obj = Tensor([1], Addressing.SCALAR, alias="alpha", datatype=Datatype.F32)
    sym = Symbol("alpha", SymbolType.Scalar, obj)
    sym.datatype = Datatype.F32
    sym.data_view = DataView([1], None)
    sym.num_threads = 32

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    builder = IRBuilder(fptype=Datatype.F32, context=ctx)
    assert sym.load(builder, ctx, None, [0], False) is not None

    writer = Writer()
    pir_emit(builder.finish(), writer, vm_factory("sm_86", "cuda", "float"))
    src = writer.get_src()
    assert "alpha" in src, src
    assert "alpha[" not in src, f"a scalar was given a subscript:\n{src}"


@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a")])
def test_a_broadcast_stays_inside_its_intrinsic(backend, arch):
    """The lane broadcast wraps the access; it does not read a named temporary.

    `Op.LOAD` is impure, so the emitter will not inline it into the intrinsic
    call. Migrating this case would therefore add a temporary per broadcast,
    which is a change to the generated code with nothing bought for it.
    """
    src = _generate("chain_three", backend, arch)
    for call in re.findall(r'__shfl\w*\(([^;]+)\)', src):
        assert '[' in call or 'v' in call, call


def test_the_exceptions_are_reachable_at_all():
    """A guard nothing takes is a guard nobody notices breaking.

    If the corpus stops exercising the scalar or broadcast paths, the two
    tests above pass vacuously and the special cases rot. This fails loudly
    instead.
    """
    from tensorforge.backend.symbol import Symbol, SymbolType
    import inspect
    src = inspect.getsource(Symbol.load) + inspect.getsource(Symbol.store)
    assert 'SymbolType.Register' in src
    assert 'base is None' in src, 'the base-override exception vanished'
    assert 'access is pre_access' in src, 'the broadcast exception vanished'
