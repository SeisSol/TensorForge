# SPDX-License-Identifier: MIT
"""Pytest wiring for the TensorForge end-to-end harness.

Discovery flow:

1. At session start, find GPUs (``gpu_detect``) and probe available
   toolchains (``toolchain.discover_targets``). The result is cached on
   pytest's config-cache so repeated runs skip the probe compile.
2. Walk ``tests_new/cases/**/*.py``; each file is one case module.
3. ``pytest_generate_tests`` parametrises every test taking a
   ``case``/``target`` fixture pair across the cross product.

The build cache lives under ``~/.cache/tensorforge-tests`` by default;
override with ``TF_TEST_CACHE``. Failing runs leave artifacts under
``<cache>/_failures/<case>-<target>/``.

When no GPU + toolchain combination is present, all GPU-bound tests
are skipped with a clear reason — ``pytest`` itself still runs (and
``test_layout``, ``test_reference`` exercise the host-only pieces).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import List

import pytest

from harness.gpu_detect import detect_all
from harness.toolchain import Target, discover_targets


HERE = Path(__file__).parent
CASES_ROOT = HERE / "cases"

# Default cache root: per-user, survives across sessions, easy to nuke.
_CACHE_ROOT = Path(os.environ.get(
    "TF_TEST_CACHE",
    str(Path.home() / ".cache" / "tensorforge-tests"),
))

# tensorforge_aux.{cu,cpp} sit here; the build needs the include dir.
TENSORFORGE_INCLUDE = HERE.parent / "tensorforge" / "include"


# ----------------------------------------------------------------------
# Case discovery
# ----------------------------------------------------------------------

def _load_case(path: Path):
    """Load a case file as a module without polluting ``sys.modules`` globally."""
    rel = path.relative_to(HERE).with_suffix("")
    mod_name = "tf_cases__" + ".".join(rel.parts)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load case {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _discover_cases() -> list:
    cases = []
    for path in sorted(CASES_ROOT.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = _load_case(path)
        if not hasattr(mod, "NAME") or not hasattr(mod, "descr_list"):
            continue       # not a case file
        cases.append(mod)
    return cases


# ----------------------------------------------------------------------
# Target discovery (session-cached)
# ----------------------------------------------------------------------

def _discover_targets_session(config) -> List[Target]:
    """Probe once per pytest session; cache on config for the run."""
    cached = getattr(config, "_tf_targets", None)
    if cached is not None:
        return cached
    gpus = detect_all()
    targets = discover_targets(gpus, _CACHE_ROOT / "probe")
    config._tf_targets = targets
    return targets


# ----------------------------------------------------------------------
# Pytest hooks
# ----------------------------------------------------------------------

def pytest_report_header(config):
    targets = _discover_targets_session(config)
    if not targets:
        return "tensorforge: no GPU + toolchain combination detected"
    lines = ["tensorforge targets:"]
    for t in targets:
        lines.append(f"  - {t.id}  ({t.device_name})")
    return "\n".join(lines)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = _discover_cases()
        metafunc.parametrize(
            "case", cases,
            ids=[c.NAME for c in cases],
        )
    if "target" in metafunc.fixturenames:
        targets = _discover_targets_session(metafunc.config)
        if not targets:
            # Parametrise with a single skip-marker so the test still
            # appears in the report rather than vanishing silently.
            metafunc.parametrize(
                "target",
                [pytest.param(None, marks=pytest.mark.skip(
                    reason="no usable GPU+toolchain target"))],
                ids=["no-target"],
            )
        else:
            metafunc.parametrize(
                "target", targets,
                ids=[t.id for t in targets],
            )


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="session")
def cache_root() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


@pytest.fixture(scope="session")
def tensorforge_include() -> Path:
    if not TENSORFORGE_INCLUDE.exists():
        pytest.skip(f"missing tensorforge include dir: {TENSORFORGE_INCLUDE}")
    return TENSORFORGE_INCLUDE
