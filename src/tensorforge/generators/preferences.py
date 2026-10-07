# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What was measured to be best, per device and per shape of kernel.

The static scorer ranks what a build says about itself; a measurement says
what the machine did.  Where there is one, it wins -- `tuning.autotune` takes a
matching preference as its pick and builds nothing else, and `autotune=prefer`
takes preferences alone and falls back to the default where none matches.

## Which device

A preference names a device as `arch`, `arch:variant` or a vendor.  The
architecture is what the generator targets and all it knows unasked; the
variant is the caller's to say (`Options.device`), because the instruction set
does not tell it: an MI300A and an MI300X are both gfx942, one an APU with the
host's memory and one a discrete part with its own, and a configuration that
is best on one need not be on the other.  Lookup goes from the most specific
name to the least -- `gfx942:mi300a`, then `gfx942`, then `amd` -- and within
one name the first matching entry wins, in file order, the caller's files
before the shipped one.

## Which kernels

`match` compares the kernel's features (`features`): the data type, the rows
of the lead dimension, the widest output (the columns a multiplication
writes) and the longest reduction.  A value matches itself, a two-element list
an inclusive range.  A feature a preference does not name does not constrain
it.

## What it prefers

`prefer` is what a `tuning.Candidate` is made of (`tuning.preferred`):
`lanes` and `width`, and any option.  `k_roll: auto` is the largest divisor
of the longest reduction up to 32, which is what the measured rolls were -- a
number there would only fit the one reduction length it was measured at.

## Where from

The shipped `preferences.yml` next to this module, and every file named in
`TF_PREFERENCES` (separated by the path separator) or `Options.preferences`,
which come first.  `evidence` is free text and should say which measurement a
preference comes from: without it nobody can tell whether it still holds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tensorforge.common.context import Context
from tensorforge.generators import lanes as lane_config
from tensorforge.generators.descriptions import contraction_lengths

SHIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'preferences.yml')


@dataclass(frozen=True)
class Preference:
    device: str
    match: Tuple[Tuple[str, Any], ...] = ()
    prefer: Tuple[Tuple[str, Any], ...] = ()
    evidence: str = ''
    source: str = ''

    def matches(self, feats: Dict[str, Any]) -> bool:
        for name, want in self.match:
            have = feats.get(name)
            if have is None:
                return False
            if isinstance(want, (list, tuple)) and len(want) == 2:
                lo, hi = want
                if (lo is not None and have < lo) or (hi is not None and have > hi):
                    return False
            elif have != want:
                return False
        return True


def _load_file(path: str) -> List[Preference]:
    with open(path) as f:
        data = yaml.safe_load(f) or []
    out = []
    for entry in data:
        out.append(Preference(
            device=str(entry['device']).lower(),
            match=tuple(sorted((entry.get('match') or {}).items())),
            prefer=tuple(sorted((entry.get('prefer') or {}).items())),
            evidence=str(entry.get('evidence', '')),
            source=path))
    return out


_CACHE: Dict[Tuple[str, ...], List[Preference]] = {}


def sources(context: Context) -> List[str]:
    """The files consulted, the caller's first."""
    named = []
    for text in (context.get_user_options().preferences,
                 os.environ.get('TF_PREFERENCES', '')):
        named += [p for p in (text or '').split(os.pathsep) if p]
    return named + ([SHIPPED] if os.path.exists(SHIPPED) else [])


def load(context: Context) -> List[Preference]:
    paths = tuple(sources(context))
    if paths not in _CACHE:
        prefs = []
        for path in paths:
            prefs += _load_file(path)
        _CACHE[paths] = prefs
    return _CACHE[paths]


def device_names(context: Context) -> List[str]:
    """The names this context's device answers to, most specific first."""
    hw = context.target.hw
    arch = str(hw.model).lower()
    variant = (context.get_user_options().device or '').lower()
    names = [f'{arch}:{variant}'] if variant else []
    return names + [arch, str(hw.vendor).lower()]


def features(descrs, context: Context) -> Dict[str, Any]:
    """What a preference can match on, read off the descriptors."""
    flat = [op for d in descrs for op in d.operations()]
    base = lane_config.deduce(flat, context)
    columns = 0
    for d in flat:
        dest = getattr(d, 'dest', None)
        box = getattr(dest, 'bbox', None)
        if box is None:
            continue
        n = 1
        for axis in range(1, box.rank()):
            n *= int(box.size(axis))
        columns = max(columns, n)
    depths = contraction_lengths(descrs)
    return {'dtype': str(context.fp_type.name).lower(),
            'rows': base.num_active_threads or base.num_threads,
            'columns': columns,
            'depth': max(depths) if depths else 0}


def lookup(descrs, context: Context) -> Optional[Preference]:
    """The preference for this kernel on this device, or None."""
    prefs = load(context)
    feats = features(descrs, context)
    for name in device_names(context):
        for pref in prefs:
            if pref.device == name and pref.matches(feats):
                return pref
    return None
