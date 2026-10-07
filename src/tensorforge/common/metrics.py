# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""What one build measured of what it laid down.

One record per build, made by the build and handed to what writes its
bodies: the emitter counts into it as it writes (`pir.emit`), the section
builder adds the register footprint of each body and what the wrap did.  A
caller that searches over configurations reads the figures of each build it
made off that build -- `lanes.search`, `tuning`, the merge decision -- and two
builds against one context never see each other's.

Every figure is None until something was counted: a build that measured
nothing has not measured zero.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class BuildMetrics:
    """The figures of one build."""

    def __init__(self, pressure: bool = False):
        #: Whether every emitted body reports its peak register footprint.
        #: Off unless asked: the figure costs a liveness walk per body, and
        #: only a caller searching over configurations reads it.
        self.pressure: bool = pressure

        #: The largest footprint of a body, in bytes per lane.  A maximum,
        #: since a register budget is per kernel and the widest body is what
        #: has to fit.
        self.peak_pressure: Optional[int] = None

        #: The same figure split by which register file holds it: the peak
        #: of the lane-varying values and the peak of the ones a whole wave
        #: agrees on.  They are taken at their own program points, so they do
        #: not add up to `peak_pressure`.
        #:
        #: Two files, because on AMD they are two: a wave-uniform value is an
        #: SGPR, of which a wave has about a hundred, and the scalar unit is a
        #: pipe of its own.  SeisSol's damage step fills both on gfx1150 --
        #: 107 SGPRs with 117 spilled beside 256 VGPRs -- and one figure
        #: against `max_reg_per_thread` shows neither.  On NVIDIA the uniform
        #: datapath carries integer and address arithmetic only, so a uniform
        #: float is a vector register there and the split is a diagnostic
        #: rather than a budget.
        self.peak_lane_pressure: Optional[int] = None
        self.peak_uniform_pressure: Optional[int] = None

        #: Arithmetic operations written out (`record_work`).
        self.emitted_work: Optional[int] = None

        #: Instructions laid down, in the units `pir.emit` counts
        #: (`record_code`).
        self.code_units: Optional[int] = None

        #: The same, had every merged run been written out
        #: (`record_written_code`).
        self.written_code_units: Optional[int] = None

        #: What the instructions occupy, by category (`record_mix`): category
        #: -> [issued per element and lane, copies in the code].
        self.issue_mix: Optional[Dict[str, List[int]]] = None

        #: Bytes a lane moves per element, by space and direction
        #: (`record_bytes`): 'global.read' -> bytes.
        self.memory_bytes: Optional[Dict[str, int]] = None

        #: The same two numbers per statement rather than summed
        #: (`record_hot`): (issued, copies) -> how many statements.
        #: `issue_mix` says how much a kernel runs and how much it occupies;
        #: this says how the two are spread, which is what tells a kernel that
        #: does not fit the instruction cache but spends its time in a part
        #: that does from one that does not (`analysis.icache.hot_set`).
        self.hot_profile: Optional[Dict[Tuple[int, int], int]] = None

        #: Statements between a load and the first statement reading it
        #: (`record_slack`): distance -> how many loads.  A load whose
        #: consumer is the next statement stalls on its latency; the
        #: scheduler's whole job is to make this number large.
        self.load_slack: Optional[Dict[int, int]] = None

        #: What `enable_wrap_loads` did with the transfers of the batch
        #: loops, one line per transfer as `pir.wrap.wrap_loads` reports it
        #: (`record_wrap`), or None where the pass did not run.
        self.wrap_report: Optional[List[str]] = None

    def record_pressure(self, value: int, lane: Optional[int] = None,
                        uniform: Optional[int] = None) -> None:
        if self.peak_pressure is None or value > self.peak_pressure:
            self.peak_pressure = value
        # Each file keeps its own maximum: the body with the widest register
        # image need not be the one holding the most uniform values, and a
        # budget is per file.
        if lane is not None and (self.peak_lane_pressure is None
                                 or lane > self.peak_lane_pressure):
            self.peak_lane_pressure = lane
        if uniform is not None and (self.peak_uniform_pressure is None
                                    or uniform > self.peak_uniform_pressure):
            self.peak_uniform_pressure = uniform

    def record_work(self, value: int = 1) -> None:
        """Count arithmetic the emitter wrote out.

        What a lane geometry changes and neither the register model nor
        blocks per SM can see: a packed FMA does two elements in one
        operation and a matrix instruction does a whole tile, so a geometry
        that reaches either issues fewer of these.  Counted per kernel, so a
        *rolled* reduction (`Options.k_roll`) counts its body once however
        many times it runs -- the same caveat the line estimate carries.
        """
        self.emitted_work = (self.emitted_work or 0) + value

    def record_code(self, value: int = 1) -> None:
        """Count instructions the emitter laid down (`code_units`).

        Not `record_work` again.  That one asks how often a statement runs,
        so a rolled reduction counts its trip count; this one asks how many
        copies of it the compiler writes into the kernel, so a rolled loop
        counts its body once and an unrolled one once per trip -- the figure
        an instruction cache holds.  Every statement is counted, not only the
        arithmetic: a load or an index calculation occupies the cache as much
        as an FMA does.
        """
        self.code_units = (self.code_units or 0) + value

    def record_written_code(self, value: int = 1) -> None:
        """Count what `record_code` counts, as a build that wrote every
        merged run out would lay it down: a rolled loop's body once per trip,
        and no counter for it.  A build that merged nothing counts the same
        as `record_code`."""
        self.written_code_units = (self.written_code_units or 0) + value

    def record_mix(self, category: str, issued: int, copies: int) -> None:
        """Count one statement by what it occupies (`analysis.pipeline`).

        `issued` is how often a lane runs it per element -- the trip counts
        of the loops around it, as `record_work` -- and `copies` how many
        times it is written into the code, as `record_code`.  The first is
        what a pipe has to get through, the second what the instruction
        cache holds.
        """
        if self.issue_mix is None:
            self.issue_mix = {}
        slot = self.issue_mix.setdefault(category, [0, 0])
        slot[0] += issued
        slot[1] += copies

    def record_bytes(self, key: str, value: int) -> None:
        """Bytes a lane moves per element through one space, one
        direction."""
        if self.memory_bytes is None:
            self.memory_bytes = {}
        self.memory_bytes[key] = self.memory_bytes.get(key, 0) + value

    def record_hot(self, issued: int, copies: int) -> None:
        """One statement, by how often it runs and how often it is written
        down.

        A histogram and not a list: a kernel lays down tens of thousands of
        statements and they take a handful of distinct weights, since the
        weights are the trip counts around them.
        """
        if self.hot_profile is None:
            self.hot_profile = {}
        key = (issued, copies)
        self.hot_profile[key] = self.hot_profile.get(key, 0) + 1

    def record_slack(self, distance: int) -> None:
        """One load, and how many statements stand between it and its first
        reader in the same body."""
        if self.load_slack is None:
            self.load_slack = {}
        self.load_slack[distance] = self.load_slack.get(distance, 0) + 1

    def record_wrap(self, lines: List[str]) -> None:
        """What the wrap pass reported for one body."""
        if self.wrap_report is None:
            self.wrap_report = []
        self.wrap_report.extend(lines)
