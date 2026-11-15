// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_

#include <sycl/sycl.hpp>

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__)
/// The two vector types the DPAS builtin below is declared over: OpenCL's
/// `float4` and `float8`, which is what IGC resolves the call by -- the name
/// is mangled with its argument types, and `sycl::vec` is a class.
typedef float tensorforgeDpasFloat4 __attribute__((ext_vector_type(4)));
typedef float tensorforgeDpasFloat8 __attribute__((ext_vector_type(8)));

/// `cl_intel_subgroup_matrix_multiply_accumulate_tf32`: one `dpas.8x8` over a
/// sub-group of 16, A 8x8 and B 8x16 in TF32, C and D 8x16 in F32.  Declared
/// rather than included: DPC++ has no SYCL spelling of it short of
/// `joint_matrix`, whose lane layout is opaque, and the declaration is what
/// IGC 2.x resolves.  IGC 1.0.15468 does not have it: it builds a kernel
/// calling it without a word and leaves the call an undefined symbol, which
/// the first launch stops at -- so the generator asks the `ocloc` on PATH
/// first (`intel_spmd.resolved`, `Options.sycl_dpas`).  Unused, it is no
/// reference at all, so a kernel that does not take the path is not affected.
SYCL_EXTERNAL tensorforgeDpasFloat8 intel_sub_group_tf32_tf32_matrix_mad_k8(
    tensorforgeDpasFloat4 a, tensorforgeDpasFloat8 b,
    tensorforgeDpasFloat8 acc);
#endif

namespace tensorforge {

/// `x` rounded to TF32: to nearest, ties away from zero, the low thirteen bits
/// of the significand cleared.  The upper half of an operand split for DPAS
/// (`primitives/intel_spmd.py`), and the rounding has to be done here: DPAS
/// truncates what it is handed, so an operand passed as it is loses up to a
/// whole TF32 ulp where the rounded one loses half.  The lower half, `x -
/// roundTF32(x)`, is exact in F32 and is then truncated by the instruction,
/// which costs it a bit of its own and nothing of `x`'s.  The same rounding
/// as the host's (`common/matrix/prepare.py`), so a prepared operator and one
/// split here are the same halves.
inline float roundTF32(float x) {
  const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(x);
  return sycl::bit_cast<float>((bits + 0x1000u) & 0xFFFFE000u);
}

/// The vectors the DPAS path holds its fragments in (`Options.sycl_dpas`,
/// `SyclLexic.get_fptype`): OpenCL's own under clang, which are what the
/// builtin takes and what IGC keeps in one block of registers.  `sycl::vec`
/// is a class over an array, which the front end takes apart into scalars;
/// put back together for every instruction, the order-6 SeisSol kernels
/// spilled 750 to 5 531 references at 256 registers that held none this
/// way, and the order-8 volume 16 677 (IGC 2.41.5).  `sycl::vec` where the
/// compiler is not clang -- a CPU stand-in -- since nothing else spells
/// the extension.
#if defined(__clang__)
typedef float NativeFloat4 __attribute__((ext_vector_type(4)));
typedef float NativeFloat8 __attribute__((ext_vector_type(8)));
#else
typedef sycl::vec<float, 4> NativeFloat4;
typedef sycl::vec<float, 8> NativeFloat8;
#endif

/// `acc + A B` by DPAS over the sub-group `group`, which has to be 16 wide and
/// converged: every work-item hands in its share of A, B and the accumulator
/// and gets its share of the result back (`Options.sycl_dpas`).
///
/// The shares, as measured on the Max 1550 (IGC 2.41.5) by placing distinct
/// values: work-item `l` holds A[2i + l / 8][l % 8] in element `i` of `a`,
/// B[k][l] in element `k` of `b`, and C[m][l], D[m][l] in element `m` of
/// `acc` and of the result.  So a work-item's accumulator is one column of
/// the 8 x 16 tile, which is what lets the SPMD lowering hold its output in
/// the tile as it is: lane `l` is a row of the product and the eight elements
/// are eight of its columns (`intel_spmd.d_fragment_bits`).
///
/// Off the device -- the host pass of the SYCL compiler, or a CPU stand-in
/// that executes it -- the same arithmetic out of sub-group broadcasts: A's
/// elements gathered from the work-items that hold them, both factors
/// truncated to TF32 as the instruction truncates them, the sum taken in
/// double and rounded once.  A reference for the layout, not for the
/// instruction's own rounding of the sum, which nothing here can state.
inline NativeFloat8 dpasTF32(const sycl::sub_group &group, NativeFloat4 a,
                             NativeFloat8 b, NativeFloat8 acc) {
#if defined(__SYCL_DEVICE_ONLY__)
  (void)group;
  return intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, acc);
#else
  const auto truncate = [](float x) {
    return sycl::bit_cast<float>(sycl::bit_cast<std::uint32_t>(x) &
                                 0xFFFFE000u);
  };
  // Every work-item takes part in every broadcast, in the same order.
  float shared[4][16];
  for (int i = 0; i < 4; ++i) {
    for (int lane = 0; lane < 16; ++lane) {
      shared[i][lane] = sycl::group_broadcast(group, a[i], lane);
    }
  }
  NativeFloat8 out;
  for (int m = 0; m < 8; ++m) {
    double sum = acc[m];
    for (int k = 0; k < 8; ++k) {
      // A[m][k] lies with work-item k + 8 (m % 2), in its element m / 2.
      sum += static_cast<double>(truncate(shared[m / 2][k + 8 * (m % 2)])) *
             static_cast<double>(truncate(b[k]));
    }
    out[m] = static_cast<float>(sum);
  }
  return out;
#endif
}

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
