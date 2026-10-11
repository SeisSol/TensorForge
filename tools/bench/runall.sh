#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

BENCHDIR=data

for f in $BENCHDIR/*
do
    TF_BENCH_DUMP=$f python3 run.py suites/seissolbench.py --out $BENCHDIR/out-$f --arch sm_120
    python3 roofline.py $BENCHDIR/out-$f/bench.json --out $BENCHDIR/out-$f
    python3 plot.py $BENCHDIR/out-$f/bench.json --roofline-json $BENCHDIR/out-$f/roofline.json --out $BENCHDIR/out-$f
done

zip -r outdata.zip $BENCHDIR
