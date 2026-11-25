// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
//
// `tensorforge::coreGrid` (`base.h`): the work-groups of a launch sized for a
// number of hardware threads an Xe-core (`Options.sycl_core_threads`).  Plain
// integer arithmetic, so the contract is checkable at compile time.
//
// The Max 1550's stack has 64 Xe-cores.  A work-group of sixteen hardware
// threads -- sixteen sub-groups of sixteen lanes, or sixteen work-items of the
// explicit vector -- and sixteen rows (multiplications) a group.

#include "tensorforge_device/base.h"

#include <cstddef>

using tensorforge::coreGrid;
using tensorforge::gcdOf;

static_assert(gcdOf(12, 18) == 6, "gcd");
static_assert(gcdOf(9, 0) == 9 && gcdOf(0, 9) == 9, "gcd with zero");
static_assert(gcdOf(512, 9) == 1, "powers of two are coprime to nine");

// 16, 32 and 64 threads an Xe-core: 64, 128 and 256 groups, in two parts.
static_assert(coreGrid(64, 16, 16, 16, 2) == 64, "16 an Xe-core");
static_assert(coreGrid(64, 32, 16, 16, 2) == 128, "32 an Xe-core");
static_assert(coreGrid(64, 64, 16, 16, 2) == 256, "64 an Xe-core");
// No parts (0) and one part are the same.
static_assert(coreGrid(64, 32, 16, 16, 0) == coreGrid(64, 32, 16, 16, 1),
              "parts");
// At least one group a part, however few the threads.
static_assert(coreGrid(1, 1, 16, 16, 2) == 2, "a group a part at least");
// A block of one changes nothing; nor does a block coprime to the rows.
static_assert(coreGrid(64, 24, 16, 16, 2, 1) == 96, "24 an Xe-core, no blocks");
static_assert(coreGrid(64, 32, 16, 16, 2, 9) == 128,
              "512 rows a part are coprime to 9");
// hvp's 24 an Xe-core with blocks of nine: 48 groups a part, 768 rows, a
// multiple of three -- 47 groups (752 rows) is the nearest below that is not.
static_assert(coreGrid(64, 24, 16, 16, 2, 9) == 94,
              "24 an Xe-core, blocks of nine");
static_assert(gcdOf(47 * 16, 9) == 1, "752 rows are coprime to 9");
// 48 an Xe-core: 96 groups a part, 1536 rows; 95 groups, 1520 rows.
static_assert(coreGrid(64, 48, 16, 16, 2, 9) == 190,
              "48 an Xe-core, blocks of nine");

int main() { return 0; }
