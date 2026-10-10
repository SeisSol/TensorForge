#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

OUTDIR=bench/data

python seissol_export.py elastic linearck 3 s $OUTDIR/elastic-3-s-s1.json --mechanisms 0 --simcount 1
python seissol_export.py elastic linearck 3 d $OUTDIR/elastic-3-d-s1.json --mechanisms 0 --simcount 1
python seissol_export.py elastic linearck 4 s $OUTDIR/elastic-4-s-s1.json --mechanisms 0 --simcount 1
python seissol_export.py elastic linearck 4 d $OUTDIR/elastic-4-d-s1.json --mechanisms 0 --simcount 1
python seissol_export.py elastic linearck 6 s $OUTDIR/elastic-6-s-s1.json --mechanisms 0 --simcount 1
python seissol_export.py elastic linearck 6 d $OUTDIR/elastic-6-d-s1.json --mechanisms 0 --simcount 1

python seissol_export.py viscoelastic linearckanelastic 3 s $OUTDIR/viscoelastic-3-s-s1.json --mechanisms 3 --simcount 1
python seissol_export.py viscoelastic linearckanelastic 3 d $OUTDIR/viscoelastic-3-d-s1.json --mechanisms 3 --simcount 1
python seissol_export.py viscoelastic linearckanelastic 4 s $OUTDIR/viscoelastic-4-s-s1.json --mechanisms 3 --simcount 1
python seissol_export.py viscoelastic linearckanelastic 4 d $OUTDIR/viscoelastic-4-d-s1.json --mechanisms 3 --simcount 1
python seissol_export.py viscoelastic linearckanelastic 6 s $OUTDIR/viscoelastic-6-s-s1.json --mechanisms 3 --simcount 1
python seissol_export.py viscoelastic linearckanelastic 6 d $OUTDIR/viscoelastic-6-d-s1.json --mechanisms 3 --simcount 1

python seissol_export.py elastic linearck 3 s $OUTDIR/elastic-3-s-s8.json --mechanisms 0 --simcount 8
python seissol_export.py elastic linearck 3 d $OUTDIR/elastic-3-d-s8.json --mechanisms 0 --simcount 8
python seissol_export.py elastic linearck 4 s $OUTDIR/elastic-4-s-s8.json --mechanisms 0 --simcount 8
python seissol_export.py elastic linearck 4 d $OUTDIR/elastic-4-d-s8.json --mechanisms 0 --simcount 8
