#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

BENCHDIR=data

for f in $BENCHDIR/*
do
    TF_BENCH_DUMP=$f python run.py suites/seissolbench.py --out $BENCHDIR/out-$f
    python roofline.py $BENCHDIR/out-$f/bench.json --out $BENCHDIR/out-$f
    python plot.py $BENCHDIR/out-$f/bench.json --roofline-json $BENCHDIR/out-$f/roofline.json --out $BENCHDIR/out-$f
done

zip -r outdata.zip $BENCHDIR
