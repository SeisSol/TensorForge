// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_SYCL_SPMD_H_

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

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

/// `cl_intel_subgroup_2d_block_io`: a 2D block of 32-bit data read by a
/// sub-group of 16, each work-item its column of a block of 16 columns, or
/// its row of a block of 16 rows where the read transposes -- the shapes
/// `tensorforge::subGroupBlock2d` takes, of those IGC 2.41 has.  Declared as
/// the DPAS builtin above is; unused, they are no reference.
typedef int tensorforgeInt2 __attribute__((ext_vector_type(2)));
#define TENSORFORGE_BLOCK2D_READ(NAME)                                         \
  SYCL_EXTERNAL void NAME(__attribute__((opencl_global)) void *base,           \
                          int width, int height, int pitch,                    \
                          tensorforgeInt2 coord,                               \
                          __attribute__((opencl_private)) std::uint32_t *out);
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_32b_1r16x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_32b_2r16x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_32b_4r16x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_32b_8r16x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_32b_16r16x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_transpose_32b_16r1x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_transpose_32b_16r2x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_transpose_32b_16r4x1c)
TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_transpose_32b_16r8x1c)
#undef TENSORFORGE_BLOCK2D_READ

/// Its writes: a block of 1 to 8 rows of 32-bit data, each work-item its
/// column of 16 (`tensorforge::subGroupBlock2dStore`).  Declared as the reads
/// are; unused, they are no reference.
#define TENSORFORGE_BLOCK2D_WRITE(NAME)                                        \
  SYCL_EXTERNAL void NAME(__attribute__((opencl_global)) void *base,           \
                          int width, int height, int pitch,                    \
                          tensorforgeInt2 coord,                               \
                          __attribute__((opencl_private)) std::uint32_t *in);
TENSORFORGE_BLOCK2D_WRITE(intel_sub_group_2d_block_write_32b_1r16x1c)
TENSORFORGE_BLOCK2D_WRITE(intel_sub_group_2d_block_write_32b_2r16x1c)
TENSORFORGE_BLOCK2D_WRITE(intel_sub_group_2d_block_write_32b_4r16x1c)
TENSORFORGE_BLOCK2D_WRITE(intel_sub_group_2d_block_write_32b_8r16x1c)
#undef TENSORFORGE_BLOCK2D_WRITE
#endif

namespace tensorforge {

/// `N` reads of a sub-group's lanes at consecutive offsets of global memory,
/// as one block read (`Options.spmd_block_runs`, `pir.block_reads`): element
/// `j` of what work-item `l` gets is `p[j * size + l]`, `size` the
/// sub-group's.  `p` has to point into global memory -- the cast asks no
/// question, and the block read reads there.  `sub_group::load`, which DPC++
/// lowers to `__spirv_SubgroupBlockReadINTEL` (one `load.ugm.d32xNt` on the
/// Max 1550) and marks deprecated in favor of `group_load`; the warning is
/// held off here rather than repeated for every kernel.
template <int N, typename T>
inline sycl::vec<T, N> subGroupRun(const sycl::sub_group &group, const T *p) {
  using namespace sycl::access;
  const auto global = sycl::ext::oneapi::experimental::static_address_cast<
      address_space::global_space>(
      sycl::multi_ptr<T, address_space::generic_space, decorated::yes>(
          const_cast<T *>(p)));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  return group.load<N>(global);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

/// What a work-item gets of a 2D block: `Count` elements, one as a number.
template <typename T, int Count>
using Block2dShare = std::conditional_t<Count == 1, T, sycl::vec<T, Count>>;

namespace detail {
/// Off the device, a 2D block message whose surface breaks a rule of
/// `cl_intel_subgroup_2d_block_io` stops the program, naming the rule: the
/// extension leaves such a message undefined, so the model has no value to
/// give it, and one that went on would let a missing check pass by luck.
[[noreturn]] inline void surfaceRuleBroken(const char *what) {
  std::fprintf(stderr, "tensorforge: a 2D block message %s\n", what);
  std::abort();
}

/// The extension's rules for a surface of `size`-byte elements: the base
/// 64-byte aligned, the width from 64 bytes and a whole number of elements,
/// at least one row, the pitch at least the width and a multiple of 16.
inline void checkSurface(const void *base, int width, int height, int pitch,
                         int size) {
  if (reinterpret_cast<std::uintptr_t>(base) % 64 != 0)
    surfaceRuleBroken("on a surface whose base is not 64-byte aligned");
  if (width < 64 || width % size != 0)
    surfaceRuleBroken("on a surface under 64 bytes wide or not a whole "
                      "number of elements");
  if (height < 1)
    surfaceRuleBroken("on a surface of no rows");
  if (pitch < width || pitch % 16 != 0)
    surfaceRuleBroken("on a surface whose pitch is under its width or not "
                      "a multiple of 16 bytes");
}
} // namespace detail

/// A 2D block of `Rows` x `Cols` 32-bit elements of the surface at `base` --
/// `width` bytes wide, `height` rows high, rows `pitch` bytes apart -- from
/// column `x` and row `y`, read by the sub-group `group`, which has to be 16
/// wide and full (`pir.block_reads.load_block2d`): work-item `l` gets column
/// `x + l`, its `Rows` elements, or transposed row `y + l` and its `Cols`.
///
/// Off the device -- the host pass, or a CPU stand-in that executes it --
/// the same elements read one at a time, and zero where they lie outside
/// the surface: the extension leaves their value unstated, so zero is this
/// model's choice, not a statement about the device.
template <typename T, int Rows, int Cols, bool Transposed>
inline Block2dShare<T, Transposed ? Cols : Rows>
subGroupBlock2d(const sycl::sub_group &group, const T *base, int width,
                int height, int pitch, int x, int y) {
  static_assert(sizeof(T) == 4, "a 2D block read of 32-bit data");
  static_assert(Transposed ? Rows == 16 : Cols == 16,
                "a column or a row a work-item of a sub-group of 16");
  constexpr int Count = Transposed ? Cols : Rows;
  std::uint32_t words[Count];
#if defined(__SYCL_DEVICE_ONLY__)
  (void)group;
  auto *surface =
      (__attribute__((opencl_global)) void *)(const_cast<T *>(base));
  auto *out = (__attribute__((opencl_private)) std::uint32_t *)(&words[0]);
  const tensorforgeInt2 at{x, y};
  if constexpr (Transposed) {
    static_assert(Cols == 1 || Cols == 2 || Cols == 4 || Cols == 8,
                  "a transposed block of 1, 2, 4 or 8 columns");
    if constexpr (Cols == 1)
      intel_sub_group_2d_block_read_transpose_32b_16r1x1c(
          surface, width, height, pitch, at, out);
    else if constexpr (Cols == 2)
      intel_sub_group_2d_block_read_transpose_32b_16r2x1c(
          surface, width, height, pitch, at, out);
    else if constexpr (Cols == 4)
      intel_sub_group_2d_block_read_transpose_32b_16r4x1c(
          surface, width, height, pitch, at, out);
    else
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
          surface, width, height, pitch, at, out);
  } else {
    static_assert(Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8 ||
                      Rows == 16,
                  "a block of 1, 2, 4, 8 or 16 rows: a work-item's share is "
                  "one sycl::vec");
    if constexpr (Rows == 1)
      intel_sub_group_2d_block_read_32b_1r16x1c(surface, width, height, pitch,
                                                at, out);
    else if constexpr (Rows == 2)
      intel_sub_group_2d_block_read_32b_2r16x1c(surface, width, height, pitch,
                                                at, out);
    else if constexpr (Rows == 4)
      intel_sub_group_2d_block_read_32b_4r16x1c(surface, width, height, pitch,
                                                at, out);
    else if constexpr (Rows == 8)
      intel_sub_group_2d_block_read_32b_8r16x1c(surface, width, height, pitch,
                                                at, out);
    else
      intel_sub_group_2d_block_read_32b_16r16x1c(surface, width, height, pitch,
                                                 at, out);
  }
#else
  detail::checkSurface(base, width, height, pitch, sizeof(T));
  const int lane = static_cast<int>(group.get_local_linear_id());
  const auto *bytes = reinterpret_cast<const unsigned char *>(base);
  for (int k = 0; k < Count; ++k) {
    const int col = x + (Transposed ? k : lane);
    const int row = y + (Transposed ? lane : k);
    const bool inside = col >= 0 && row >= 0 &&
                        col * static_cast<int>(sizeof(T)) < width &&
                        row < height;
    words[k] = 0u;
    if (inside) {
      std::memcpy(&words[k],
                  bytes + static_cast<std::size_t>(row) * pitch +
                      static_cast<std::size_t>(col) * sizeof(T),
                  sizeof(T));
    }
  }
#endif
  if constexpr (Count == 1) {
    return sycl::bit_cast<T>(words[0]);
  } else {
    sycl::vec<T, Count> share;
#pragma unroll
    for (int k = 0; k < Count; ++k) {
      share[k] = sycl::bit_cast<T>(words[k]);
    }
    return share;
  }
}

/// A 2D block of `Rows` x 16 32-bit elements written by the sub-group `group`
/// -- 16 wide and full -- to the surface at `base`, `width` bytes wide,
/// `height` rows high, rows `pitch` bytes apart, from column `x` and row `y`:
/// work-item `l` hands in column `x + l`, its `Rows` elements in order
/// (`pir.block_reads.store_block2d`).  Untransposed only, which is all the
/// hardware writes.
///
/// Off the device the elements are written one at a time, and a block that
/// does not lie wholly inside its surface stops the program: what the
/// device does with the part outside the extension leaves unstated, and no
/// writer here may rely on it.  `share` is the work-item's `Rows` elements
/// in any vector that indexes them -- `sycl::vec` or OpenCL's own, the one
/// the DPAS path holds its values in -- or a number where it is one.
template <typename T, int Rows, typename Share>
inline void subGroupBlock2dStore(const sycl::sub_group &group, T *base,
                                 int width, int height, int pitch, int x, int y,
                                 Share share) {
  static_assert(sizeof(T) == 4, "a 2D block write of 32-bit data");
  static_assert(Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8,
                "a block write of 1, 2, 4 or 8 rows");
  std::uint32_t words[Rows];
  if constexpr (Rows == 1) {
    words[0] = sycl::bit_cast<std::uint32_t>(static_cast<T>(share));
  } else {
#pragma unroll
    for (int k = 0; k < Rows; ++k) {
      words[k] = sycl::bit_cast<std::uint32_t>(static_cast<T>(share[k]));
    }
  }
#if defined(__SYCL_DEVICE_ONLY__)
  (void)group;
  auto *surface = (__attribute__((opencl_global)) void *)(base);
  auto *in = (__attribute__((opencl_private)) std::uint32_t *)(&words[0]);
  const tensorforgeInt2 at{x, y};
  if constexpr (Rows == 1)
    intel_sub_group_2d_block_write_32b_1r16x1c(surface, width, height, pitch,
                                               at, in);
  else if constexpr (Rows == 2)
    intel_sub_group_2d_block_write_32b_2r16x1c(surface, width, height, pitch,
                                               at, in);
  else if constexpr (Rows == 4)
    intel_sub_group_2d_block_write_32b_4r16x1c(surface, width, height, pitch,
                                               at, in);
  else
    intel_sub_group_2d_block_write_32b_8r16x1c(surface, width, height, pitch,
                                               at, in);
#else
  detail::checkSurface(base, width, height, pitch, sizeof(T));
  const int lane = static_cast<int>(group.get_local_linear_id());
  auto *bytes = reinterpret_cast<unsigned char *>(base);
  for (int k = 0; k < Rows; ++k) {
    const int col = x + lane;
    const int row = y + k;
    if (col < 0 || row < 0 || (col + 1) * static_cast<int>(sizeof(T)) > width ||
        row >= height)
      detail::surfaceRuleBroken("writing past its surface");
    std::memcpy(bytes + static_cast<std::size_t>(row) * pitch +
                    static_cast<std::size_t>(col) * sizeof(T),
                &words[k], sizeof(T));
  }
#endif
}

namespace detail {
/// Where an element's surface starts (`Options.sycl_element_blocks`): its
/// columns are `Pitch` bytes apart, and the 2D block messages want the
/// surface's base on a 64-byte boundary, which the element's own pointer
/// need not be.  It need only be aligned to `grain`, the largest power of
/// two dividing both the pitch and 64: then some `j` of the `period` pitches
/// before it is on the boundary, and the surface starts there, its rows from
/// `j` on the element's columns -- rows the messages read and write; the
/// `j` before them are only addressed.  `j` is the element's offset from the
/// boundary, in grains, times the inverse of `Pitch / grain` modulo
/// `period`: an odd number, its own inverse modulo 2 and 4.
template <int Pitch> struct ElementSurface {
  static_assert(Pitch > 0 && Pitch % 16 == 0,
                "a pitch of a whole number of 16 bytes");
  static constexpr int grain = (Pitch & -Pitch) < 64 ? (Pitch & -Pitch) : 64;
  static constexpr int period = 64 / grain;
  static constexpr int inverse = (Pitch / grain) % period;

  /// Whether an element at `address` has such a surface.
  static bool fits(std::uintptr_t address) {
    return (address & static_cast<std::uintptr_t>(grain - 1)) == 0;
  }

  /// The pitches its surface starts before it.
  static int shift(std::uintptr_t address) {
    return static_cast<int>((((address & 63) / grain) * inverse) &
                            (period - 1));
  }
};
} // namespace detail

/// Whether the element at `element` -- its columns `Pitch` bytes apart --
/// has a surface the 2D block messages can address
/// (`Options.sycl_element_blocks`): its address aligned to the grain of
/// `detail::ElementSurface`, 64 bytes for a pitch of whole cache lines, 32
/// for SeisSol's padding to 32 on PVC.  The sub-group's answer, from the
/// address of its first work-item, which holds the sub-group's element: a
/// branch on it is one the compiler takes for the sub-group.
template <int Pitch>
inline bool elementBlocks(const sycl::sub_group &group, const void *element) {
  const std::uintptr_t address = sycl::group_broadcast(
      group, reinterpret_cast<std::uintptr_t>(element), 0);
  return detail::ElementSurface<Pitch>::fits(address);
}

/// A 2D block of `Rows` rows of the element at `element`, read by the
/// sub-group `group` (`Options.sycl_element_blocks`): the element's storage
/// as a surface of its `Columns` columns, `Pitch` bytes each, every column a
/// row -- work-item `l` gets element `x + l` of columns `y` to `y + Rows`.
/// Only for an element `elementBlocks` answered for, and only for a block
/// inside its columns: the surface is the storage and nothing past it.
template <typename T, int Rows, int Pitch, int Columns>
inline Block2dShare<T, Rows> elementBlock2d(const sycl::sub_group &group,
                                            const T *element, int x, int y) {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(element);
  const int j = detail::ElementSurface<Pitch>::shift(address);
  const T *base = reinterpret_cast<const T *>(
      address - static_cast<std::uintptr_t>(j) * Pitch);
  return subGroupBlock2d<T, Rows, 16, false>(group, base, Pitch, Columns + j,
                                             Pitch, x, y + j);
}

/// The write of the same block: work-item `l` hands in element `x + l` of
/// columns `y` to `y + Rows`, 1 to 8 of them, as `subGroupBlock2dStore`
/// takes them.
template <typename T, int Rows, int Pitch, int Columns, typename Share>
inline void elementBlock2dStore(const sycl::sub_group &group, T *element, int x,
                                int y, Share share) {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(element);
  const int j = detail::ElementSurface<Pitch>::shift(address);
  T *base =
      reinterpret_cast<T *>(address - static_cast<std::uintptr_t>(j) * Pitch);
  subGroupBlock2dStore<T, Rows>(group, base, Pitch, Columns + j, Pitch, x,
                                y + j, share);
}

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
typedef float NativeFloat16 __attribute__((ext_vector_type(16)));
#else
typedef sycl::vec<float, 4> NativeFloat4;
typedef sycl::vec<float, 8> NativeFloat8;
typedef sycl::vec<float, 16> NativeFloat16;
#endif

/// One DPAS tile of an operator laid out as a 2D surface
/// (`Options.sycl_dpas_block2d`, `intel_spmd.surface_order`): sixteen rows of
/// sixteen floats from column `x` and row `y` of the surface at `base` --
/// `width` bytes wide, `height` rows high, rows `pitch` bytes apart -- read
/// by the sub-group `group`, 16 wide and full, in one 2D block message:
/// work-item `l` gets column `x + l`, the upper halves of its eight steps and
/// then the lower, in OpenCL's own vector, which the B fragments are taken out
/// of.  `subGroupBlock2d`'s read, written by the builtin straight into the
/// vector: through that helper's private array and `sycl::vec`, IGC 2.42.6
/// copied the halves out of every block and spilled SeisSol's order-6 volume
/// (14 kB, 9 214 references), and this way nothing.
inline NativeFloat16 dpasBlock2d(const sycl::sub_group &group,
                                 const float *base, int width, int height,
                                 int pitch, int x, int y) {
  NativeFloat16 share;
#if defined(__SYCL_DEVICE_ONLY__)
  (void)group;
  intel_sub_group_2d_block_read_32b_16r16x1c(
      (__attribute__((opencl_global)) void *)(const_cast<float *>(base)), width,
      height, pitch, tensorforgeInt2{x, y},
      (__attribute__((opencl_private)) std::uint32_t *)(&share));
#else
  const auto read = subGroupBlock2d<float, 16, 16, false>(group, base, width,
                                                          height, pitch, x, y);
  for (int k = 0; k < 16; ++k) {
    share[k] = read[k];
  }
#endif
  return share;
}

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
