# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The diagnostics still measure what they claim to.

A diagnostic that reaches into generator internals can stop measuring without
failing.  `tools/ir_opacity.py` wraps `pir.optimize` in `_counting_optimize`;
were the wrapper to name the parameters rather than forward them, an argument
added to `optimize` would raise one `TypeError` per case -- inside a `try`
whose whole purpose is to keep a case that does not generate from stopping
the sweep, so every case would be swallowed into a `did not generate:` list a
hundred entries long.

The output would stay plausible throughout: a table, a corpus line and a
percentage, the percentage of nothing.  A wrong number that looks like a
number is the failure mode worth guarding, because nobody re-derives a
diagnostic they have no reason to doubt.

Six tools reach into generator internals -- `ir_opacity` patches
`pir.optimize`, the five censuses wrap builder methods -- and all six can
break exactly this way when the thing they wrap changes shape.  So the check
is the same for all of them: run it, and insist it still saw the corpus.
These are smoke tests, not assertions about the numbers.  Pinning the counts
would mean re-recording them on every legitimate change, which is how a test
gets deleted.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

#: Each of these sweeps the whole corpus, so the file costs a couple of
#: minutes.  Marked slow so a working run is one deselect away, and left in
#: the default set because a diagnostic nobody runs is one nobody trusts.
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def _run(name, *args, timeout=900):
    proc = subprocess.run([sys.executable, str(TOOLS / name), *args],
                          capture_output=True, text=True, cwd=ROOT,
                          timeout=timeout)
    assert proc.returncode in (0, 1), (
        f"{name} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    return proc.stdout


#: The targets `tools/ir_opacity.py` sweeps, which is a subset of the four
#: `test_snapshots` records.
_OPACITY_BACKENDS = ("hip", "cuda")


def _snapshot_generates(backend: str) -> set:
    """Case names whose recorded snapshot on `backend` is generated source.

    The harness writes a snapshot for a case that refuses to generate too, and
    it starts with `FAILED:`.  Reading that back gives an expectation that
    calibrates itself: a case added to the corpus updates it by being recorded,
    and a case that stops generating shows up as a snapshot diff first, where
    it belongs, rather than here.
    """
    out = set()
    for path in (ROOT / "tests" / "snapshots").glob(f"*.{backend}.cpp"):
        if not path.read_text().startswith("FAILED:"):
            out.add(path.name[:-len(f".{backend}.cpp")])
    return out


def test_ir_opacity_still_generates_the_corpus():
    """The failure mode above: every case reported as not generating.

    What the tool reports as generated has to agree with what the corpus
    actually generates, which the snapshots already state.  Comparing against
    them rather than against a fraction of the corpus keeps the check exact
    while the set of cases that refuse to generate moves: a fixed bound tight
    enough to catch a broken wrapper is one a handful of deliberate refusals
    would trip.
    """
    out = _run("ir_opacity.py", "--cases")
    m = re.search(r"corpus: (\d+) cases x (\d+) targets, (\d+) generated, "
                  r"(\d+) failed", out)
    assert m, f"the corpus summary line is gone:\n{out[-1500:]}"
    cases, targets, generated, failed = (int(g) for g in m.groups())

    expected = sum(len(_snapshot_generates(b)) for b in _OPACITY_BACKENDS)
    assert generated == expected, (
        f"the tool reports {generated} of {cases * targets} generated, but "
        f"the snapshots record {expected} generating on {list(_OPACITY_BACKENDS)}. "
        f"If the generator is fine, the tool has stopped measuring -- check "
        f"that its wrapper still matches the signature it wraps.")


def test_ir_opacity_attributes_what_it_counts():
    """A site table of one row means the attribution collapsed, which has its
    own way of going wrong: `_site` walks out of the builder frames, and a
    module moving can leave every node attributed to the same place."""
    out = _run("ir_opacity.py", "--sites")
    section = out.split("REACH CODEGEN", 1)
    assert len(section) == 2, "the site table is gone"
    rows = [ln for ln in section[1].splitlines()
            if re.match(r"^\S+\.py:\w+\s", ln)]
    assert len(rows) >= 3, f"only {len(rows)} distinct sites:\n{section[1]}"


def test_no_site_label_is_ambiguous():
    """A bare `__init__.py:gen_ir` can name two different files at once.  A
    label that does not identify a file is not attribution."""
    out = _run("ir_opacity.py", "--sites")
    labels = re.findall(r"^(\S+\.py:\w+)\s", out, re.M)
    bare = [l for l in labels if l.startswith("__init__.py:")]
    assert not bare, f"ambiguous site labels: {sorted(set(bare))}"


@pytest.mark.parametrize("tool,marker", [
    ("layout_census.py", r"\bvalues\b"),
    ("operand_layouts.py", r"\bmfma\b|\buntracked\b"),
    ("slot_census.py", r"batch loops over \d+ cases"),
    ("buffer_spans.py", r"\bby kind\b|\bspan\b"),
    ("overlap_census.py", r"\d+ B of \d+ B to separate them"),
])
def test_the_censuses_still_see_something(tool, marker):
    """Each wraps a builder method and would report an empty corpus if the
    method it wraps were renamed, without any error to notice."""
    out = _run(tool)
    assert re.search(marker, out), f"{tool} produced nothing recognizable:\n{out[-800:]}"
    numbers = [int(n) for n in re.findall(r"^\s*(\d+)\s", out, re.M)]
    assert numbers and max(numbers) > 0, f"{tool} counted nothing:\n{out[-800:]}"


def test_the_runner_agrees_with_the_suite():
    """`tools/syntax_check.py` and `test_syntax.py` read one list.

    A runner that knew nothing of what the suite marks xfail would print those
    snapshots as failures every time.  A check with standing reds is a check
    nobody reads, which is the whole reason the xfail table exists on the test
    side.
    """
    from harness import syntax

    out = _run("syntax_check.py")
    m = re.search(r"(\d+) well-formed, (\d+) ill-formed, (\d+) known-bad", out)
    assert m, f"the summary line changed:\n{out[-600:]}"
    well_formed, ill_formed, known = (int(g) for g in m.groups())

    assert known == len(syntax.NOT_YET_ESIMD), (
        f"the runner counted {known} tracked failures, the list has "
        f"{len(syntax.NOT_YET_ESIMD)}")
    assert ill_formed == 0, (
        "a snapshot stopped compiling and is not in the tracked list")
    for name in syntax.NOT_YET_ESIMD:
        assert name in out, f"{name} is tracked but was not reported"


def test_every_mutation_still_applies():
    """A skipped mutation tests nothing, and reads almost like a pass.

    The harness prints `SKIPPED: the code has moved` when an anchor no longer
    matches, which is it working -- but in a list of a hundred and thirty the
    line goes by, and the check it stood for is quietly gone.

    Only that each anchor is still findable, not that the mutation is caught:
    the full harness takes minutes and a test that slow gets deselected, while
    `--dry-run` is a string search.  Finding the anchor is the part that rots.
    """
    out = _run("mutation_check.py", "--dry-run", timeout=300)
    stale = [ln for ln in out.splitlines()
             if "no longer testing anything" in ln]
    assert not stale, ("mutation anchors that no longer match:\n"
                       + "\n".join(stale))


def test_no_shared_access_costs_more_than_four_bank_cycles():
    """A swizzle that stops being applied is noticed nowhere else.

    If `_swizzle` asked whether the *source* symbol has a PIR buffer, as a
    proxy for "every write to this window goes through `store`", the proxy
    would agree with the real question for `GlbToShrLoader` and never for
    `StoreRegToShr`, whose source is a register and has no buffer by
    construction; a change to the loader's bindings would then quietly stop
    every macro window being permuted and send its accesses to 32-way --
    every lane in one bank with a different address.

    Nothing would fail.  The kernels stay correct, the snapshots re-record
    cleanly, and the only symptom is a number in a tool nobody has reason to
    run.  So the number is a test.

    A ceiling, not an exact count: the census moves whenever a case is added,
    and pinning it would mean re-recording on every unrelated change.  What
    must not happen is a *class* of conflict appearing.
    """
    import re

    out = _run("bank_conflicts.py")
    counts = {int(m.group(2)): int(m.group(1))
              for m in re.finditer(r'^\s*(\d+)\s+(\d+)-way\s*$', out, re.M)}
    assert counts, f"the census printed no histogram:\n{out[-800:]}"
    worst = max(counts)
    assert worst <= 4, (
        f"{counts[worst]} accesses cost {worst} bank cycles; the worst was 4. "
        f"A permutation is probably not being applied.\n{out[-600:]}")
