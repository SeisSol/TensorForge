# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Golden-output tests: the generated source, byte for byte.

Why this exists: every pass in ``backend.pir`` and ``backend.opt`` is a
source-to-source transform, and the only cheap statement about a refactor
that is worth anything is *the generated code did not change*.  Without a
frozen baseline that statement cannot be made, so a rework either ships
unverified or gets re-derived by hand each time.

These tests need no GPU and no toolchain.  They are deliberately dumb:
generate, compare to a file, print a diff.  A snapshot changing is not a
failure in itself --- it is a request to look at the diff and decide.  When
the change is intended::

    pytest --snapshot-update

and commit the diff *together with* the change that caused it, so review
sees both.

Failing cases get a snapshot too, of the exception.  A case that stops
generating --- or one that starts --- is exactly as interesting as one whose
output shifts, and silently dropping it from the corpus is how a suite ends
up green while covering less than it did.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# The pair is (backend, arch).  These are codegen targets, not devices: no
# hardware is involved, so the list is free to cover more than the machine
# running the tests has.
BACKENDS = (
    ("cuda", "sm_86"),
    ("hip", "gfx90a"),
    # SPMD SYCL, and the explicitly vectorised lowering of the same device.
    #
    # `oneapi` in SPMD mode used to sit here as a third SYCL entry and was
    # dropped: it differed from `acpp` in two lines out of several hundred --
    # `local_accessor` instead of `accessor`, and the kernel attributes -- so
    # 54 files bought almost nothing and every cross-cutting change had to be
    # reviewed four times instead of three.
    #
    # What that costs is named rather than waved away: nothing now generates
    # `[[intel::reqd_sub_group_size(16)]]`, since the `esimd` target takes the
    # other branch of the same `if`.  The `oneapi` half of `local_accessor`
    # stays covered through `esimd`.
    ("acpp", "pvc"),
    ("esimd", "pvc"),
)

# A mismatch on a 1000-line kernel should not bury the report.
_DIFF_LINES = 60


def _render(case, backend: str, arch: str) -> str:
    """The full generated surface for one case on one target, or the failure."""
    try:
        ctx = Context(arch=arch, backend=backend,
                      fp_type=getattr(case, "DTYPE", None))
        gen = Generator(case.descr_list(), ctx,
                        attrs=getattr(case, "ATTRS", None))
        gen.generate()
    except Exception as exc:                      # noqa: BLE001 -- recorded
        # `repr` of the args, not of the exception object: the latter can
        # carry an address for some exception types and would make the
        # snapshot unstable.
        msg = str(exc).strip()
        return f"FAILED: {type(exc).__name__}" + (f": {msg}" if msg else "") + "\n"

    parts = [
        f"// === base name ===\n{gen.get_base_name()}\n",
        f"// === header ===\n{gen.get_header() or ''}\n",
        f"// === launcher ===\n{gen.get_launcher() or ''}\n",
        f"// === kernel ===\n{gen.get_kernel() or ''}\n",
    ]
    return "\n".join(parts)


def _path_for(case, backend: str) -> Path:
    return SNAPSHOT_DIR / f"{case.NAME}.{backend}.cpp"


@pytest.mark.parametrize("backend,arch", BACKENDS, ids=[b for b, _ in BACKENDS])
def test_generated_source_matches_snapshot(snapshot_case, backend, arch,
                                           request):
    actual = _render(snapshot_case, backend, arch)
    path = _path_for(snapshot_case, backend)

    if request.config.getoption("--snapshot-update"):
        # A case that used to generate and now raises is a regression, never an
        # intended snapshot update.  Without this the harness happily replaces
        # 285 lines with a single `FAILED:` line and reports success, and the
        # broken output becomes the thing every later run is compared against.
        #
        # Not hypothetical: a killed `mutation_check.py` left a mutation in the
        # tree, the next `--snapshot-update` baked it into 57 files, and the
        # only visible symptom was that the diff was large -- which it is
        # during a migration anyway.
        if (actual.startswith("FAILED:") and path.exists()
                and not path.read_text().startswith("FAILED:")
                and not request.config.getoption("--snapshot-accept-failures")):
            pytest.fail(
                f"{snapshot_case.NAME} [{backend}] generated before and raises "
                f"now:\n  {actual.splitlines()[0]}\n"
                f"Refusing to record the failure over a working snapshot. Fix "
                f"the regression, or pass --snapshot-accept-failures if the "
                f"case is genuinely expected to stop generating.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
        return

    if not path.exists():
        pytest.fail(
            f"no snapshot for {snapshot_case.NAME} [{backend}].\n"
            f"If this case is new, run: pytest --snapshot-update\n"
            f"expected file: {path}")

    expected = path.read_text()
    if actual == expected:
        return

    diff = list(difflib.unified_diff(
        expected.splitlines(keepends=True), actual.splitlines(keepends=True),
        fromfile=f"{path.name} (recorded)", tofile=f"{path.name} (generated)",
        n=2))
    shown = "".join(diff[:_DIFF_LINES])
    more = ("\n... %d more diff lines\n" % (len(diff) - _DIFF_LINES)
            if len(diff) > _DIFF_LINES else "")
    pytest.fail(
        f"generated source for {snapshot_case.NAME} [{backend}] changed "
        f"({len(expected.splitlines())} -> {len(actual.splitlines())} lines).\n"
        f"If intended: pytest --snapshot-update, and commit the diff with "
        f"the change.\n\n{shown}{more}")


def test_no_orphaned_snapshots(request):
    """A snapshot with no case behind it means a case was renamed or lost.

    Four of the suite's stale failures came from exactly this --- a case
    moved, and nothing noticed that the old path stopped being covered.
    """
    if request.config.getoption("--snapshot-update"):
        pytest.skip("snapshots are being rewritten")
    if not SNAPSHOT_DIR.exists():
        pytest.skip("no snapshots recorded yet")

    from conftest import _discover_cases          # same directory as this file

    expected = {f"{c.NAME}.{b}.cpp"
                for c in _discover_cases() for b, _ in BACKENDS}
    found = {p.name for p in SNAPSHOT_DIR.glob("*.cpp")}
    orphans = sorted(found - expected)
    assert not orphans, (
        "snapshot files with no matching case+backend (rename? deletion?):\n  "
        + "\n  ".join(orphans)
        + "\nDelete them, or run pytest --snapshot-update.")


def test_generation_is_deterministic(snapshot_case):
    """Two generations in one process must agree, byte for byte.

    The whole harness rests on this.  It is not free: iteration over a
    ``set`` of IR nodes has silently reordered barrier placement here
    before, and that class of bug is invisible until a snapshot flaps.
    """
    backend, arch = BACKENDS[0]
    first = _render(snapshot_case, backend, arch)
    second = _render(snapshot_case, backend, arch)
    assert first == second, (
        f"{snapshot_case.NAME}: two generations in the same process differ "
        f"-- codegen is not deterministic, and no snapshot of it is "
        f"meaningful")
