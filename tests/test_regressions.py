# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Host-only regressions for two silent-wrong-value codegen defects.

Both were found in a generated poroelastic SeisSol kernel dump, both produce
plausible-looking code and wrong numbers, and neither is visible in any case
the suite had before.  They need no GPU to pin down: one is a numeric
statement the host interpreter can make, the other a statement about which
dimension a register staging spreads across lanes.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from pathlib import Path

import pytest

import kernel_eval
from tensorforge.backend.instructions.memory.load import GlbToRegLoader
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"


def _load(name):
    path = next(iter(sorted(CASES.rglob(f"{name}.py"))), CASES / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"tf_regr__{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(name, backend="cuda", arch="sm_86"):
    mod = _load(name)
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen


def _walk(instrs):
    for ins in instrs or []:
        yield ins
        for attr in ("_instructions", "instructions", "_region"):
            sub = getattr(ins, attr, None)
            if isinstance(sub, (list, tuple)):
                yield from _walk(sub)


def _register_loaders(gen):
    section = gen._section
    stream = list(section.global_ir) + list(section.stream)
    return [i for i in _walk(stream) if isinstance(i, GlbToRegLoader)]


def _destination(src, seed, tid, preset=None):
    mem = kernel_eval.evaluate(src, tid=tid, seed=seed, globals_only=True,
                               preset=preset)
    return {k: v for k, v in mem.items() if k[0] == "m0"}


# ----------------------------------------------------------------------
# The accumulator has to carry forward
# ----------------------------------------------------------------------

@pytest.mark.parametrize("term", range(4))
def test_every_accumulated_term_reaches_the_destination(term):
    """Zeroing any one term's operand must move the result.

    A reference-free sensitivity test, which is what this defect needs: the
    kernel it produced was internally consistent and only *some* of the terms
    went missing.  With the stale bias in place the destination held the first
    write plus the last term, so zeroing an operand of any term in between
    left the output bit-identical.
    """
    src = _generate("accumulate_chain").get_kernel()
    # operand order in the launcher is dest, then (a, b) per descriptor
    operand = f"m{1 + 2 * term}"
    for seed in (7, 101):
        for tid in (0, 3, 11):
            base = _destination(src, seed, tid)
            zeroed = _destination(src, seed, tid, preset={operand: 0.0})
            assert zeroed != base, (
                f"term {term} ({operand}) does not affect the destination: "
                f"its contribution is being overwritten rather than "
                f"accumulated")


def test_accumulation_chain_stores_once():
    """The chain stays in registers: one store, no reload of the destination.

    Not a correctness statement on its own --- with the invalidation fix the
    store/reload form computes the right answer too --- but the round trip per
    term is the cost the writers-vs-boxes distinction exists to avoid, and it
    is what made the stale bias reachable in the first place.
    """
    src = _generate("accumulate_chain").get_kernel()
    assert src.count("store{r>g}") == 1, src.count("store{r>g}")
    assert "load{g>r}(glb_m0)" not in src


# ----------------------------------------------------------------------
# The register lane axis has to follow the lead index
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_lead_index_off_dim0_is_lane_resident(backend, arch):
    """A staged operand spreads the dimension carrying the lead index.

    Operand ``A`` of this case is ``[1, 20]`` with the contraction index on
    dimension 0 and the destination's lead index on dimension 1.  Staging it
    with dimension 0 across lanes put the lane-distributed index into the
    register axis; ``Symbol.load`` then found a loop constant where it
    expected the lane axis and emitted a cross-lane broadcast, handing every
    lane element ``[0, 0]``.
    """
    gen = _generate("lead_index_off_dim0", backend, arch)
    loaders = _register_loaders(gen)
    assert loaders, "expected the operand to be staged in registers"
    assert any(ldr._dest.lead_dims == [1] for ldr in loaders), (
        "no register image spreads dimension 1 across lanes: "
        + repr([(l._dest.name, l._dest.lead_dims) for l in loaders]))


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_lead_index_off_dim0_needs_no_cross_lane_read(backend, arch):
    """...and therefore reads its own lane, with no shuffle at all.

    The user-visible form of the same statement.  Checking the operand's own
    register array rather than the whole kernel matters: the *other* operand
    carries no lead index and is broadcast entirely legitimately, so a blanket
    search for cross-lane primitives would report it.  The defect showed up as
    ``readlane(r0[0], 0)`` on CUDA and ``broadcast<32, 1, 0>(r0[0])`` on HIP,
    i.e. one element standing in for twenty.
    """
    gen = _generate("lead_index_off_dim0", backend, arch)
    staged = [ldr for ldr in _register_loaders(gen)
              if list(ldr._bbox.sizes()) == [1, 20]]
    assert staged, "expected operand A to be staged in registers"
    name = staged[0]._dest.name
    src = gen.get_kernel()
    shuffle = re.compile(
        rf"(?:readlane|readfirstlane|__shfl\w*|broadcast\s*<[^>]*>)"
        rf"\s*\(\s*{re.escape(name)}\[")
    hit = shuffle.search(src)
    assert hit is None, (
        f"{name} holds the lead index lane by lane, but it is read through a "
        f"cross-lane primitive: {src[hit.start():hit.start() + 60]!r}"
        if hit else "")


# ----------------------------------------------------------------------
# Every result has to reach memory
# ----------------------------------------------------------------------

_COMPUTE = re.compile(r"//\s*(\w+) = \+\((?P<ops>[^)]*)\) \+ (?:None|name: (\w+))")
# `glb_mN = store{r>g}(rM);` but `sN = store{r>s}(localShrMem0, rM);`
_STORED = re.compile(r"//\s*\w+ = store\{r>[gs]\}\((?:\w+,\s*)?(\w+)\)")


def _dropped_results(src):
    """Result arrays that are neither stored nor read again.

    A multilinear writes into a register array; that array then has to be
    stored, handed to a later step as its bias, or consumed as an operand.
    One that is none of those has been computed and thrown away, which is what
    a lost write looks like from the outside --- no crash, no diagnostic, just
    a term missing from the answer.

    ``auto& irN = rM;`` is deliberately not counted as a use: that is how a
    compute writes its result in place, i.e. a definition.
    """
    produced, consumed = [], set()
    for match in _COMPUTE.finditer(src):
        produced.append(match.group(1))
        consumed.update(re.findall(r"\b[rs]\d+\b", match.group("ops")))
        if match.group(3):
            consumed.add(match.group(3))
    consumed.update(_STORED.findall(src))
    return [r for r in produced if r not in consumed]


# ----------------------------------------------------------------------
# A sliced write covers its slice, and only its slice
# ----------------------------------------------------------------------

_STORE_HEAD = re.compile(r"//\s*(glb_\w+) = store\{r>g\}\((\w+)\);")
_FOR_BOUNDS = re.compile(r"for \(int32_t \w+ = (-?\d+); \w+ < (-?\d+);")
# The literal is typed now (`0.0f`, not `0`): the neutral element comes
# from `writer.const(..., ftype)` rather than the string "0".  What this
# test is about is that the columns get *defined*, not how the zero is
# spelled.
_ZERO_TO_GLOBAL = re.compile(r"\bglb_\w+\[\w+\] = 0(?:\.0+[fF]?)?;")


def _store_loops(src):
    """Per global store: the bounds of the loops it walks.

    The emitted store is a lead loop over thread-blocks plus one counted loop
    per remaining dimension.  Reading the bounds back is the cheapest exact
    statement about *which* elements a store touches --- a slice written at the
    wrong offset, or one that spills past its box, shows up here and nowhere
    else short of running the kernel.
    """
    lines = src.splitlines()
    out = []
    for i, line in enumerate(lines):
        head = _STORE_HEAD.search(line)
        if not head:
            continue
        bounds = []
        for follow in lines[i + 1:i + 12]:
            m = _FOR_BOUNDS.search(follow)
            if m:
                bounds.append((int(m.group(1)), int(m.group(2))))
        out.append((head.group(2), bounds))
    return out


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_eqspp_window_write_defines_the_whole_tensor(backend, arch):
    """No slicing offset: the box is the eqspp window, the rest is zero.

    ``sliced_write`` assigns ``D[:, 6:13]`` with no offset, so it addresses
    ``D`` itself and its box is the range yateto knows the result can be
    nonzero in.  Columns 0..5 are therefore zero, not unspecified --- and
    nothing else in the kernel writes them, so this store has to.  The
    poroelastic space-time predictor reads them straight back.
    """
    src = _generate("sliced_write", backend, arch).get_kernel()
    assert _ZERO_TO_GLOBAL.search(src), (
        "columns outside the eqspp window are left as they were; nothing "
        "else defines them")

    stores = _store_loops(src)
    assert len(stores) == 1, f"expected one store, got {len(stores)}"
    _, bounds = stores[0]
    assert bounds and bounds[0] == (0, 1), (
        f"the destination spans one thread-block on the lead axis, "
        f"the store walks {bounds[0] if bounds else None}")


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_view_write_touches_nothing_outside_its_slice(backend, arch):
    """With an offset the destination is a slice and owns only its own box.

    The mirror image of the case above, and the reason the two cannot share a
    rule: ``kernel_0bf208a83b`` writes ``m2`` column by column through views,
    and zero-filling around any one of them wipes the other twelve.
    """
    src = _generate("sliced_write_view", backend, arch).get_kernel()
    assert not _ZERO_TO_GLOBAL.search(src), (
        "a slice is zero-filling around itself; the rest of the tensor "
        "belongs to other descriptors")


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_sliced_accumulation_writes_every_term(backend, arch):
    """Slicing and accumulation together still write every term.

    ``_deferred_stores`` holds one entry per symbol name.  Deferring an atomic
    update therefore makes it collide with the next slice of the same tensor:
    the second displaced the first, and one term ended up in a register array
    nothing ever read.
    """
    src = _generate("sliced_accumulate", backend, arch).get_kernel()
    dropped = _dropped_results(src)
    assert not dropped, f"result(s) computed and discarded: {dropped}"
    assert src.count("store{r>g}") == 3, (
        f"three writes were produced, {src.count('store{r>g}')} reach memory")
    assert not _ZERO_TO_GLOBAL.search(src), (
        "an accumulation is zero-filling around its slice")


# ----------------------------------------------------------------------
# A load may not be overtaken by a store to what it reads
# ----------------------------------------------------------------------

_GLB_LOAD = re.compile(r"//\s*(\w+) = load\{g>r\}\((glb_\w+)\);")
_GLB_STORE = re.compile(r"//\s*(glb_\w+) = store\{r>g\}\((\w+)\);")
_USE = re.compile(r"//\s*(\w+) = \+\(([^)]*)\)(?: \+ name: (\w+))?")


def _stale_reads(src):
    """Loads whose tensor is stored to before the loaded value is consumed.

    `MoveLoads` splits a load into transfer and wait so the transfer can go
    early.  Between the two positions nothing may write what it reads, or the
    consumer gets the value from before that write.  Reading the emitted order
    back is an exact test: the transfer is where the comment sits.
    """
    seq = []
    for line in src.splitlines():
        t = line.strip()
        m = _GLB_LOAD.match(t)
        if m:
            seq.append(("load", m.group(1), m.group(2)))
            continue
        m = _GLB_STORE.match(t)
        if m:
            seq.append(("store", m.group(1), m.group(2)))
            continue
        m = _USE.match(t)
        if m:
            read = set(re.findall(r"\b[rs]\d+\b", m.group(2)))
            if m.group(3):
                read.add(m.group(3))
            seq.append(("compute", m.group(1), read))

    stale = []
    for i, (kind, reg, tensor) in enumerate(seq):
        if kind != "load":
            continue
        for j in range(i + 1, len(seq)):
            if seq[j][0] == "compute" and reg in seq[j][2]:
                if any(k == "store" and t == tensor
                       for k, t, _ in seq[i + 1:j]):
                    stale.append((reg, tensor))
                break
    return stale


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_accumulated_tensor_is_read_after_its_last_write(backend, arch):
    src = _generate("accumulate_then_read", backend, arch).get_kernel()
    stale = _stale_reads(src)
    assert not stale, (
        "load(s) hoisted above a store to the same tensor, so the consumer "
        f"sees a value from before that store: {stale}")


@pytest.mark.parametrize("name", ["accumulate_chain", "sliced_accumulate",
                                  "sliced_write", "sliced_write_view",
                                  "accumulate_then_read"])
@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_no_stale_global_read(name, backend, arch):
    """The same invariant across every case that writes a tensor twice."""
    src = _generate(name, backend, arch).get_kernel()
    assert not _stale_reads(src)


# ----------------------------------------------------------------------
# The harness has to compare the buffer the reference describes
# ----------------------------------------------------------------------

def _all_case_paths():
    return sorted(p for p in (CASES).rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _all_case_paths(), ids=lambda p: p.stem)
def test_case_names_a_single_output(path):
    """A case that writes several tensors must say which one it means.

    `reference()` returns one array and receives one `dest_in`, so the
    snapshot handed to it and the buffer read back afterwards have to be the
    same operand.  They used to be chosen independently --- the input
    preparation took the last sink, the comparison the first --- which agreed
    only as long as every case had exactly one.  A case with two then had its
    intermediate compared against a reference for its result, and reported a
    kernel bug that was not there.
    """
    from harness import driver_emit

    mod = _load(path.stem)
    if not hasattr(mod, "descr_list"):
        pytest.skip("not a case module")
    ctx = Context(arch="sm_86", backend="cuda", fp_type=getattr(mod, "DTYPE", None))
    try:
        gen = Generator(mod.descr_list(), ctx)
        gen.generate()
    except Exception:
        # cases pinned as XFAIL do not build a descriptor list at all
        pytest.skip("case does not generate on this target")
    sinks = [o.alias or o.kernel_name
             for o in driver_emit.collect_operands(gen) if o.is_sink]
    assert sinks, "a case has to write something"
    declared = getattr(mod, "OUTPUT", None)
    if len(sinks) > 1:
        assert declared in sinks, (
            f"{mod.NAME} writes {sinks}; set OUTPUT to the one reference() "
            f"returns (currently {declared!r})")
    elif declared is not None:
        assert declared in sinks, f"{mod.NAME}: OUTPUT={declared!r} is not a sink"


# ----------------------------------------------------------------------
# A store may not read past its accumulator
# ----------------------------------------------------------------------

_REG_DECL = re.compile(r"float (r\d+)\[(\d+)\]")
_LOCAL = re.compile(r"int32_t (\w+) = (.+);$")
_SRC_READ = re.compile(r"float value = (r\d+)\[(\w+)\];")


def _expand(expr, env):
    for _ in range(14):
        new = re.sub(r"\b(v\d+\w*)\b",
                     lambda m: f"({env[m.group(1)]})" if m.group(1) in env
                     else m.group(1), expr)
        if new == expr:
            return new
        expr = new
    return expr


def _out_of_range_reads(src):
    """Register-array indices a global store uses that the array does not have.

    The accumulator's size is right there in its declaration, and the store's
    index expressions are local, so the check is exact: expand the local
    `int32_t` assignments, evaluate at every corner of the loop nest, and
    compare against the declared length.  A store that believes it holds more
    than it does shows up here as an index past the end --- and, on the way
    out, as elements written that were never computed.
    """
    lines = src.splitlines()
    sizes = {m.group(1): int(m.group(2))
             for m in (_REG_DECL.search(l) for l in lines) if m}
    bad = []
    for i, line in enumerate(lines):
        if not _STORE_HEAD.search(line):
            continue
        env, ranges = {}, []
        for follow in lines[i + 1:i + 400]:
            t = follow.strip()
            if _STORE_HEAD.search(t) or re.match(r"//\s*\w+ = (load|\+\()", t):
                break
            m = _FOR_BOUNDS.search(t)
            if m:
                var = re.search(r"int32_t (\w+) =", t).group(1)
                ranges.append((var, int(m.group(1)), int(m.group(2)) - 1))
                continue
            m = _LOCAL.match(t)
            if m:
                env[m.group(1)] = m.group(2)
            m = _SRC_READ.match(t)
            if not m:
                continue
            reg, idx = m.group(1), _expand(m.group(2), env)
            idx = re.sub(r"\(threadIdx\.x % \d+\)", "0", idx)
            names = [v for v, _, _ in ranges]
            for combo in itertools.product(*[(lo, hi) for _, lo, hi in ranges]):
                try:
                    value = eval(idx, {"__builtins__": {}}, dict(zip(names, combo)))
                except Exception:
                    break
                if reg in sizes and not 0 <= value < sizes[reg]:
                    bad.append((reg, value, sizes[reg]))
    return bad


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_narrow_write_does_not_claim_the_whole_tensor(backend, arch):
    """A write finds the image of an earlier *read* and must not adopt its box.

    ``_deferred_stores`` is keyed by symbol name and lives for the whole
    kernel, so the register image staged for reading ``D`` wide is what the
    later one-column write finds.  Adopting its data view made a one-element
    accumulator claim thirteen, and the store wrote all thirteen columns.
    """
    src = _generate("narrow_write_after_wide_read", backend, arch).get_kernel()
    bad = _out_of_range_reads(src)
    assert not bad, f"store reads past its accumulator: {bad[:5]}"

    stores = _store_loops(src)
    narrow = stores[0]
    assert len(narrow[1]) == 2 and narrow[1][1] == (0, 1), (
        f"the write covers one column; the store walks {narrow[1]}")


@pytest.mark.parametrize("name", ["accumulate_chain", "sliced_accumulate",
                                  "sliced_write", "sliced_write_view",
                                  "accumulate_then_read",
                                  "narrow_write_after_wide_read"])
@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_no_store_reads_past_its_accumulator(name, backend, arch):
    src = _generate(name, backend, arch).get_kernel()
    assert not _out_of_range_reads(src)


# ----------------------------------------------------------------------
# The accumulation bias must come from where the result goes
# ----------------------------------------------------------------------

_ADDR_DEF = r"int32_t {name} = ([^;]+);"


def _resolved_address(lines, name, depth=8):
    """The address expression for `name`, with intermediates substituted in.

    Address arithmetic is SSA now, so a shift and the term that follows it
    land in different statements::

        int32_t v629_a = v625_off + ((v614_n1 + 8) * 32);
        int32_t v630_a = v629_a + (v615_n2 * 416);

    Reading only the statement the load names would miss the `+ 8` entirely
    and report a correct kernel as a wrong one.  Substituting transitively
    asks the question the test means to ask -- does the address the bias is
    read through carry the offset, anywhere along the way.
    """
    if not re.fullmatch(r"v\d+_\w+", name.strip()):
        # A single-use address is folded into its subscript now that it is an
        # operand rather than a name inside a string, so the load may hand us
        # `(v322_lead + 384)` instead of a name to look up.  Substituting into
        # it is the same question; there is just one fewer hop to start from.
        expr = name
    else:
        expr = None
        for line in lines:
            m = re.search(_ADDR_DEF.format(name=re.escape(name)), line)
            if m:
                expr = m.group(1)
                break
        if expr is None:
            return None
    for _ in range(depth):
        names = [n for n in re.findall(r"\bv\d+_\w+\b", expr)]
        grown = expr
        for n in names:
            for line in lines:
                m = re.search(_ADDR_DEF.format(name=re.escape(n)), line)
                if m:
                    grown = grown.replace(n, f"({m.group(1)})")
                    break
        if grown == expr:
            break
        expr = grown
    return expr


_BIAS = re.compile(r"//\s*(\w+) = \+\(.*?\) \+ name: (\w+),")
_GUARD_LINE = re.compile(r"if \((.+)\) \{")
# The read used to be handed the fixed name `oldvalue`; it is an SSA
# value now, so the name varies.  The pair this test needs is still
# there -- which symbol is read, and through which address.
# The subscript may be a name or a folded expression: `s0[v298_a]` and
# `s0[(v322_lead + 384)]` are the same read, and which one appears depends on
# whether the address has more than one use.
_OLDVALUE = re.compile(r"\bfloat \w+ = (\w+)\[([^\]]+)\];")


def _predictor_descrs():
    """The space-time predictor's inner loop, reduced to four descriptors.

        t             = A x v                    (temporary, whole box)
        D[10:20, 12] += t[10:20, 12] x M0
        t[10:20, 8]  += s1 * D[10:20, 12] * s2   (read-modify-write on t)
        D[10:20, 11] += t[10:20, 11] x M1

    Every write slices the *lead* dimension, which pins the accumulator's
    origin, and `D` is read back in between.  That combination is what the
    three defects below need, and nothing in `cases/` produces it.
    """
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import MultilinearDescr

    M, N, T, LO, HI = 32, 13, 4, 10, 20

    def tensor(shape, alias, is_tmp=False, addressing=Addressing.PTR_BASED):
        return Tensor(shape, addressing,
                      BoundingBox([0] * len(shape), list(shape)),
                      alias=alias, is_tmp=is_tmp, datatype=Datatype.F32)

    def sliced(t, col):
        return SubTensor(t, BoundingBox([0, 0, 0], [HI - LO, 1, T]),
                         [LO, col, 0], sliced=True)

    a = tensor([M, N], "A")
    v = tensor([T], "v", addressing=Addressing.NONE)
    d = tensor([M, N, T], "D")
    t = tensor([M, N, T], "t", is_tmp=True, addressing=Addressing.STRIDED)
    s1 = tensor([], "s1", addressing=Addressing.SCALAR)
    s2 = tensor([], "s2", addressing=Addressing.SCALAR)

    def contract(dest_col, src_col, mat):
        return MultilinearDescr(
            dest=sliced(d, dest_col), ops=[sliced(t, src_col), SubTensor(mat)],
            target=[[0, 1, -1], [-1, 2]], permute=[[0, 1, 2], [0, 1]], add=True)

    return [
        MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(a), SubTensor(v)],
                         target=[[0, 1], [2]], permute=[[0, 1], [0]]),
        contract(12, 12, tensor([T, T], "M0")),
        MultilinearDescr(dest=sliced(t, 8),
                         ops=[SubTensor(s1), sliced(d, 12), SubTensor(s2)],
                         target=[[], [0, 1, 2], []],
                         permute=[[], [0, 1, 2], []], add=True),
        contract(11, 11, tensor([T, T], "M1")),
    ]


def _predictor_source(backend, arch):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    gen = Generator(_predictor_descrs(), ctx)
    gen.generate()
    return gen.get_kernel()


def _guard_after(lines, index):
    for line in lines[index + 1:index + 8]:
        m = _GUARD_LINE.search(line)
        if m:
            return re.sub(r"v\d+_lead", "LEAD", m.group(1))
    return None


# CUDA only: a vendor with atomic updates takes the `can_use_atomic`
# path instead, where the accumulation has no bias operand at all.
@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86")])
def test_bias_is_loaded_by_the_lanes_that_use_it(backend, arch):
    """The destination preload runs on the same lanes as the compute.

    `GlbToRegLoader` consumes a slicing offset while loading, so the image sits
    at origin 0 and element `s` lands in lane `s % T`.  Theta, though, is
    pinned on the destination's offset and shifts the whole lead loop by it.
    With the two out of step the lanes that did the arithmetic never loaded a
    bias and the ones that loaded it did nothing: `+=` quietly became `=`.
    """
    src = _predictor_source(backend, arch)
    lines = src.splitlines()
    loads = {m.group(1): i for i, l in enumerate(lines)
             for m in [re.match(r"\s*//\s*(\w+) = load\{g>r\}\(glb_\w+\);", l)] if m}
    checked = 0
    for i, line in enumerate(lines):
        m = _BIAS.search(line)
        if not m or m.group(2) not in loads:
            continue
        checked += 1
        assert _guard_after(lines, loads[m.group(2)]) == _guard_after(lines, i), (
            f"{m.group(2)} is loaded by different lanes than the compute that "
            f"uses it as bias: load guard "
            f"{_guard_after(lines, loads[m.group(2)])!r} vs compute guard "
            f"{_guard_after(lines, i)!r}")
    assert checked, "expected at least one register-resident bias"


# CUDA only: a vendor with atomic updates takes the `can_use_atomic`
# path instead, where the accumulation has no bias operand at all.
@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86")])
def test_shared_bias_carries_the_destination_offset(backend, arch):
    """A destination read live out of shared memory needs its slicing offset.

    `_get_target_symbol` falls back to the destination symbol itself for a
    shared temporary, and the compute then addresses it with its own loop
    indices.  The store adds the descriptor's offset on the way out; the read
    has to as well, or the accumulation takes its bias from the wrong
    elements --- `t[10:20, 8] += ...` read `t[0:10, 0]`.
    """
    src = _predictor_source(backend, arch)
    lines = src.splitlines()
    checked = 0
    for i, line in enumerate(lines):
        m = _BIAS.search(line)
        if not m or not m.group(2).startswith("s"):
            continue
        sym = m.group(2)
        read = next((_OLDVALUE.search(x) for x in lines[i:i + 400]
                     if _OLDVALUE.search(x)
                     and _OLDVALUE.search(x).group(1) == sym), None)
        assert read is not None, f"no bias read from {sym}"
        addr = _resolved_address(lines[i:i + 400], read.group(2))
        assert addr is not None
        checked += 1
        # the store that follows addresses the same symbol; both must agree
        store = next((x for x in lines[i:i + 500] if f"{sym}[" in x and "] = " in x
                      and "oldvalue" not in x), None)
        assert store is not None, f"no store back to {sym}"
        shifts = set(re.findall(r"\+ (\d+)\)", addr))
        assert shifts, (
            f"the bias read from {sym} carries no slicing offset "
            f"({addr.strip()}); the store does")
    assert checked, "expected a shared-memory bias"


# ----------------------------------------------------------------------
# What a writer actually writes decides whether a tensor is sliced
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_partial_writes_are_staged_for_the_whole_read(backend, arch):
    """Successive partial writes, then a read of the union.

    `t = Q; t += F0; t += F1; O = t x M`, where every descriptor declares the
    whole of `t` --- that is what yateto emits --- but `_analyze` intersects
    the range down to what each operand supports, so the accumulations write
    half of it.  Judged on the declared boxes those look like one writer
    covering everything, so the value was kept in registers; the image left
    behind then held only the last writer's rows, and the read that follows
    wants the union.  It was refused outright, which is where the elastic
    build stopped.

    The tensor has to go through memory instead, so ask the question of the
    boxes that are actually written.
    """
    src = _generate("partial_writes_read_whole", backend, arch).get_kernel()
    # each partial write goes out as it is produced ...
    assert src.count("store{r>s}") >= 3, (
        "the partial writes are not being staged out; the register image "
        "would hold only the last one's rows")
    # ... and the read that follows takes the union from there
    reads = re.findall(r"//\s*\w+ = \+\((s\d+) \* ", src)
    assert reads, "the final contraction does not read the staged tensor"


# ----------------------------------------------------------------------
# A lead window may span more than one register block
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_accumulator_is_sized_for_every_block_it_spans(backend, arch):
    """The result array holds as many slots as the store walks.

    With 32 lanes, `D[20:35, 12] += ...` covers lanes 20..31 of one register
    block and lanes 0..2 of the next, so the accumulator needs two slots per
    remaining index.  `_analyze` works that out and the store walks both;
    `_alloc_register_array` sized for one, because it added theta to a box
    that already carried it --- the bias image is staged in the tensor's own
    lead coordinates.  Order 4 hid it: every window fell inside one block, and
    the double count cancelled.

    The host interpreter does not enforce array bounds, so the emitted numbers
    do not give this away; the store's indices against the declared length do.
    """
    src = _generate("lead_window_spans_two_blocks", backend, arch).get_kernel()
    bad = _out_of_range_reads(src)
    assert not bad, f"store reads past its accumulator: {bad[:5]}"


@pytest.mark.parametrize("theta,blocks", [(0, 1), (4, 1), (20, 2), (30, 2)])
def test_accumulator_slot_count_follows_the_window(theta, blocks):
    """However the window falls, the array and the inner buffer agree.

    The inner buffer is sized from the range `_analyze` computed, the result
    array from the box; they describe the same thing and disagreeing is the
    defect.  Sweeping theta pins both the straddling case and the one-block
    case that used to cancel.
    """
    module = _load("lead_window_spans_two_blocks")
    descrs = module.descr_list()
    # move the window without rebuilding the case
    dest = descrs[-1].dest
    dest.offset[0] = theta
    for op in descrs[-1].ops:
        if getattr(op, "offset", None) and len(op.offset) == 3:
            op.offset[0] = theta

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    gen = Generator(descrs, ctx)
    gen.generate()
    src = gen.get_kernel()

    sizes = {m.group(1): int(m.group(2))
             for m in (re.search(r"float (i?r\d+)\[(\d+)\]", l)
                       for l in src.splitlines()) if m}
    inner = {k: v for k, v in sizes.items() if k.startswith("ir")}
    assert inner, "expected an inner accumulation buffer"
    for name, size in inner.items():
        outer = sizes.get(name[1:])
        assert outer == size, (
            f"{name[1:]} holds {outer} slots but {name} holds {size}; "
            f"the store walks {size // blocks} indices over {blocks} block(s)")
    assert not _out_of_range_reads(src)


# --------------------------------------------------------------------------- #
# Tensor.data is an ndarray of the tensor's shape
# --------------------------------------------------------------------------- #

def test_a_scalar_operand_reaches_the_kernel():
    """`alpha != 1` builds a synthetic `SCALAR` tensor to carry the constant.

    Its `data` is read back through `value()`, which indexes by coordinate
    tuple -- `()` for a rank-0 tensor.  A list answers that with a TypeError,
    and for a while nothing noticed: `value()` asked `realindex in self.data`
    first, which on a list tests the *elements* and never matches a coordinate,
    so every lookup fell through to `None`.  Asking the sparsity pattern
    instead reaches the access, and this case stopped generating on both
    backends.
    """
    import numpy as np

    from tensorforge.common.matrix.tensor import Tensor
    from tensorforge.common.basic_types import Addressing

    scalar = Tensor([], Addressing.SCALAR, data=np.array(13.0))
    assert scalar.value(()) == pytest.approx(13.0)

    source = _generate("csa_alpha").get_kernel()
    assert "13" in source, "the constant never reached the generated kernel"


@pytest.mark.parametrize("data,why", [
    ([13.0], "a list cannot be indexed by a coordinate tuple"),
    (13.0, "a bare float has no shape"),
])
def test_data_that_is_not_an_array_is_rejected_at_construction(data, why):
    """Checked, not coerced.

    An `np.asarray` in the constructor would accept these and leave the caller
    unfixed -- which is how the requirement came to have two homes.  The error
    names the tensor, so the caller is findable from the message alone.
    """
    from tensorforge.common.matrix.tensor import Tensor
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.exceptions import GenerationError

    with pytest.raises(GenerationError, match="must be an ndarray"):
        Tensor([], Addressing.SCALAR, data=data)


def test_data_of_the_wrong_shape_is_rejected_at_construction():
    """`np.array([alpha])` is shape `(1,)`; the tensor is `()`.  Close enough
    to pass an `isinstance`, and wrong at every index."""
    import numpy as np

    from tensorforge.common.matrix.tensor import Tensor
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.exceptions import GenerationError

    with pytest.raises(GenerationError, match="has shape"):
        Tensor([], Addressing.SCALAR, data=np.array([13.0]))
