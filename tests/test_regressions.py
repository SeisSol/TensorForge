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
import re
from pathlib import Path

import pytest

import kernel_eval
from tensorforge.backend.instructions.memory.load import GlbToRegLoader
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"


def _load(name):
    path = CASES / f"{name}.py"
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
    assert src.count("store{r>g}") == 4, (
        f"four writes were produced, {src.count('store{r>g}')} reach memory")


# ----------------------------------------------------------------------
# A sliced write covers its slice, and only its slice
# ----------------------------------------------------------------------

_STORE_HEAD = re.compile(r"//\s*(glb_\w+) = store\{r>g\}\((\w+)\);")
_FOR_BOUNDS = re.compile(r"for \(int32_t \w+ = (-?\d+); \w+ < (-?\d+);")
_ZERO_TO_GLOBAL = re.compile(r"\bglb_\w+\[\w+\] = 0;")


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
def test_sliced_write_lands_on_its_slice(backend, arch):
    """The slice is written where it belongs, and nothing else is touched."""
    src = _generate("sliced_write", backend, arch).get_kernel()

    assert not _ZERO_TO_GLOBAL.search(src), (
        "the columns outside the slice are being zero-filled; they belong to "
        "other descriptors or to the caller")

    stores = _store_loops(src)
    assert len(stores) == 1, f"expected one store, got {len(stores)}"
    _, bounds = stores[0]
    assert len(bounds) == 2, f"expected a lead loop and one counted loop: {bounds}"
    assert bounds[0] == (0, 1), (
        f"the destination spans one thread-block on the lead axis, "
        f"the store walks {bounds[0]}")
    assert bounds[1] == (6, 13), (
        f"the slice is columns 6..12, the store walks {bounds[1]}")


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_sliced_accumulation_stays_inside_its_half(backend, arch):
    """Each half writes one thread-block, not both.

    The destination is sliced on the *lead* axis here, so a store that took
    its extent from the tensor rather than from the descriptor walks two
    blocks and writes over the other half.
    """
    src = _generate("sliced_accumulate", backend, arch).get_kernel()
    assert not _ZERO_TO_GLOBAL.search(src)
    for reg, bounds in _store_loops(src):
        assert bounds and bounds[0] == (0, 1), (
            f"store of {reg} walks {bounds[0] if bounds else None} lead "
            f"blocks; each half is exactly one")
