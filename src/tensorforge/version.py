# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The version of this package, as `VERSION` states it.

A module of its own so that what stamps the version into a kernel needs
nothing else: `interop` hands it on to SeisSol beside the frontend.
"""

import os


def get_version():
    mydir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(mydir, 'VERSION')) as file:
        return file.read().strip()
