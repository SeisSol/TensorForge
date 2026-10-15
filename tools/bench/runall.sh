#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
#
# Kept as the name people type.  What it used to do by hand -- a loop over the
# captures calling run, roofline and plot, each with its own idea of where the
# files are -- is `evaluate.py`, which also writes a manifest saying what ran
# and what did not.  Arguments pass straight through.
set -u
exec python3 "$(dirname "$0")/evaluate.py" --out "${OUTDIR:-eval}" "$@"
