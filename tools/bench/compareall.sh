#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
#
# The offline half of `runall.sh`: everything across the data you already
# have.  `runall.sh` measures one machine and plots it; this compares
# whatever has been measured -- several configurations, several orders,
# several machines -- and writes one PDF.
#
#   ./compareall.sh eval/                      one campaign
#   ./compareall.sh gb200/ mi300/ laptop/      three machines, rsynced back
#   OUTDIR=paper ./compareall.sh eval/         somewhere else
#
# Each argument is walked for `bench.json`, so pointing it at the directory
# `evaluate.py` wrote is enough; the ceilings beside them are picked up on
# the way.  Arguments after a `--` pass straight through to `compare.py`,
# which is where `--metric`, `--reference` and `--per-arch-order` live.
set -u
here="$(dirname "$0")"
out="${OUTDIR:-compare}"

paths=()
passthrough=()
seen_dashdash=0
for arg in "$@"; do
  if [ "$arg" = "--" ]; then seen_dashdash=1; continue; fi
  if [ "$seen_dashdash" -eq 1 ]; then passthrough+=("$arg"); else paths+=("$arg"); fi
done
if [ "${#paths[@]}" -eq 0 ]; then paths=(eval); fi

exec python3 "$here/compare.py" "${paths[@]}" --out "$out" --pdf \
     ${passthrough[@]+"${passthrough[@]}"}
