#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
#
# Everything one machine has to measure, in one command: `evaluate.py` into
# `$OUTDIR` (default `eval`) -- the ceilings, the timing runs over the ramp-up
# sweep, each kernel whole and in pieces, the build of every kernel, the
# plots, the comparison and the publication, with a manifest saying what ran
# and what did not.  Arguments pass straight through:
#
#   ./runall.sh                              all of it
#   ./runall.sh --profile                    and the vendor profiler
#   ./runall.sh --no-split --no-buildcost    the timing runs alone
#
# Across machines, `campaign.py` runs this on each of them and brings the
# trees back.
set -u
exec python3 "$(dirname "$0")/evaluate.py" --out "${OUTDIR:-eval}" "$@"
