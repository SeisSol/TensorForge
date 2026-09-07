# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The correctness corpus, run at batches that saturate the grid.

Not a performance corpus: the shapes are the ones the test cases pin, chosen to
exercise code paths rather than to resemble a solve.  What it is good for is a
regression signal -- these kernels exist on every branch, so a configuration
that got slower shows up here before anyone has a SeisSol capture to hand.

The batches are the point.  Every case declares `BATCH` two to four and the
launcher sizes its grid `min(occupancy_gridsize, numElements0)`, so at the
corpus's own batches the grid is a handful of blocks and the measurement is
launch overhead.  1024 is around where a mid-size device saturates, 65536 is
where per-element cost has stopped moving, and running both is what shows which
regime a number came from.
"""

from suite import CONFIGS, from_cases  # noqa: F401

NAME = 'corpus'
DESCRIPTION = 'the tests/cases corpus at saturating batches'

BATCHES = (1024, 8192, 65536)
CONFIGS = ['baseline', 'wave', 'pipeline', 'wrap1']


def workloads():
    return from_cases()
