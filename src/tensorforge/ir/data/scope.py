# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from enum import Enum

class Scope(Enum):
    KERNEL = 0,
    BLOCK = 1,
    ENTRY = 2,
