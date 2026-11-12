// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_

#include <sycl/sycl.hpp>

namespace tensorforge {

/// Lane `lane` of the work-item's own block of `Block` lanes, in a sub-group of
/// `Size`: the broadcast of a multiplication narrower than its sub-group
/// (`SyclLexic.broadcast`, `Options.sycl_split_broadcast`).
///
/// One `group_broadcast` per block, each from a lane the whole sub-group agrees
/// on, then a select by the block the work-item lies in.  IGC lowers a
/// broadcast from a uniform lane to a register region, and the select to one
/// predicated move; `select_from_group` at an index per work-item -- the other
/// spelling of the same read -- becomes an indirect register access, which on
/// the Max 1550 cost an order of magnitude at eight lanes a multiplication.
/// Every work-item executes every broadcast: a collective under a branch would
/// not be one.
template <int Block, int Size, typename T>
inline T split_broadcast(const sycl::sub_group &group, T value, int lane) {
  static_assert(Size % Block == 0, "the blocks tile the sub-group");
  T out = sycl::group_broadcast(group, value, lane);
  const int block = static_cast<int>(group.get_local_linear_id()) / Block;
#pragma unroll
  for (int b = 1; b < Size / Block; ++b) {
    const T other = sycl::group_broadcast(group, value, b * Block + lane);
    out = block == b ? other : out;
  }
  return out;
}

} // namespace tensorforge

#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_
