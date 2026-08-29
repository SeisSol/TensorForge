// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_H_

#include <hip/hip_runtime.h>

#include <hip/hip_cooperative_groups.h>

#include <type_traits>
#include <utility>

#include "base.h"

namespace tensorforge {

#ifdef __gfx900__
static constexpr int AsmVersion = 9000;
#elif defined(__gfx906__)
static constexpr int AsmVersion = 9006;
#elif defined(__gfx908__)
static constexpr int AsmVersion = 9008;
#elif defined(__gfx90a__)
static constexpr int AsmVersion = 9010;
#elif defined(__gfx942__)
static constexpr int AsmVersion = 9402;
#elif defined(__gfx950__)
static constexpr int AsmVersion = 9500;
#elif defined(__GFX10__)
static constexpr int AsmVersion = 10000;
#elif defined(__GFX11__)
static constexpr int AsmVersion = 11000;
#elif defined(__GFX12__)
static constexpr int AsmVersion = 12000;
#elif defined(__GFX13__)
static constexpr int AsmVersion = 13000;
#else
static constexpr int AsmVersion = 0;
#endif

template <typename T> union IntType {
  static_assert(sizeof(T) % sizeof(int) == 0, "");
  T value;
  int ints[sizeof(T) / sizeof(int)];
  static constexpr std::size_t IntCount = sizeof(T) / sizeof(int);
};

template <typename T>
__device__ __forceinline__ auto readlane(T value, int lane) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_readlane(it.ints[i], lane);
  }
  return ot.value;
}

template <int Dpp1, int Dpp2, int Dpp3, bool Dpp4, typename T>
__device__ __forceinline__ auto dpp(T value) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_mov_dpp(it.ints[i], Dpp1, Dpp2, Dpp3, Dpp4);
  }
  return ot.value;
}

template <int Dpp1, int Dpp2, int Dpp3, bool Dpp4, typename T>
__device__ __forceinline__ auto dppUpdate(T value, T prev) -> T {
  IntType<T> it;
  IntType<T> pt;
  IntType<T> ot;

  it.value = value;
  pt.value = prev;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_update_dpp(pt.ints[i], it.ints[i], Dpp1, Dpp2,
                                             Dpp3, Dpp4);
  }
  return ot.value;
}

template <int MaskAnd, int MaskOr, int MaskXor, typename T>
__device__ __forceinline__ auto swizzle(T value) -> T {
  IntType<T> it;
  IntType<T> ot;

  constexpr int CompleteMask = MaskAnd | (MaskOr << 5) | (MaskXor << 10);

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_ds_swizzle(it.ints[i], CompleteMask);
  }
  return ot.value;
}

template <int Block, int Subblock, int Lane, typename T>
__device__ __forceinline__ auto swizzle_bcst(T value) -> T {

  constexpr int MaskAnd = (Subblock - 1) | (0x1f & ~(Block - 1));
  constexpr int MaskOr = (Lane * Subblock) & 0x1f;
  constexpr int MaskXor = 0;

  return swizzle<MaskAnd, MaskOr, MaskXor>(value);
}

template <typename T>
__device__ __forceinline__ auto bpermute(T value, int lane) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_ds_bpermute(4 * lane, it.ints[i]);
  }
  return ot.value;
}

template <bool fi, bool bctrl, typename T>
__device__ __forceinline__ auto permlanex16(T value, uint64_t p) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] =
        __builtin_amdgcn_permlanex16(0, it.ints[i], static_cast<uint32_t>(p),
                                     static_cast<uint32_t>(p >> 32), fi, bctrl);
  }
  return ot.value;
}

template <typename T>
__device__ __forceinline__ T broadcast_warp(T value, int lane) {
  return readlane(value, lane);
}

template <std::size_t Block, std::size_t Subblock, std::size_t Lane, typename T>
__device__ __forceinline__ T broadcast(T value) {
  static_assert(Block % Subblock == 0, "");
  static_assert(Lane * Subblock < Block, "");
  if constexpr (Block == Subblock) {
    return value;
  } else if constexpr (Block == 64 && Subblock == 1) {
    return broadcast_warp(value, Lane);
  } else if constexpr (Block == 32 && Subblock == 1 && AsmVersion >= 10000) {
    return broadcast_warp(value, Lane);
  } else if constexpr (Block == 32 && Subblock == 16 && AsmVersion >= 10000) {
    constexpr uint64_t LaneBcst = 0xfedcba9876543210_u64;
    return (__lane_id() / 16) != Lane
               ? permlanex16<true, false>(value, LaneBcst)
               : value;
  } else if constexpr (Block == 16 && Subblock == 1 && AsmVersion >= 9010 &&
                       AsmVersion < 10000) {
    return dpp<0x150 + Lane, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 16 && Subblock == 8 && false) {
    // TODO: row mask
    return dpp<0x128, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 8 && Subblock == 4 && false) {
    // TODO: row mask
    return dpp<0x124 + Lane * 8, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 4 && Subblock == 1) {
    return dpp<(Lane << 6) | (Lane << 4) | (Lane << 2) | (Lane << 0), 0xf, 0xf,
               true>(value);
  } else if constexpr (Block == 4 && Subblock == 2) {
    return dpp<((Lane + 1) << 6) | (Lane << 4) | ((Lane + 1) << 2) |
                   (Lane << 0),
               0xf, 0xf, true>(value);
  } else if constexpr (Block == 2 && Subblock == 1) {
    return dpp<((Lane + 2) << 6) | ((Lane + 2) << 4) | (Lane << 2) |
                   (Lane << 0),
               0xf, 0xf, true>(value);
  } else if constexpr (Block < 64) {
    return swizzle_bcst<Block, Subblock, Lane>(value);
  } else {
    const auto blockoffset = Block == 64 ? 0 : (__lane_id() / Block) * Block;
    const auto subblockvar = __lane_id() % Subblock;
    return bpermute(value, Lane * Subblock + subblockvar + blockoffset);
  }
  return value;
}

/*
#ifdef __gfx950__
template <typename T>
__device__ __forceinline__ auto permlane32(T value) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_permlane32_swap(it.ints[i], CompleteMask);
  }
  return ot.value;
}

template <typename T>
__device__ __forceinline__ auto permlane16(T value) -> T {
  IntType<T> it;
  IntType<T> ot;

  it.value = value;
#pragma unroll
  for (int i = 0; i < IntType<T>::IntCount; ++i) {
    ot.ints[i] = __builtin_amdgcn_permlane16_swap(it.ints[i], CompleteMask);
  }
  return ot.value;
}
#else
template <typename T>
__device__ __forceinline__ auto permlane32(T value) -> T {
  const auto blockvar = __lane_id() / 32;
  const auto subblockvar = __lane_id() % 32;
  return bpermute(value, subblockvar + (1 - blockvar) * 32);
}

template <typename T>
__device__ __forceinline__ auto permlane16(T value) -> T {
  constexpr auto Block = 32;
  constexpr auto AndMask = 0x1f;
  constexpr auto OrMask = 0x0;
  constexpr auto XorMask = Block - 1;
  return swizzle<AndMask, OrMask, XorMask>(value);
}
#endif
*/

/// Exchange the two halves of every `Block`-sized group of lanes.
///
/// Lane `i` reads lane `i ^ (Block / 2)`: one bit of the lane index toggled,
/// bit `log2(Block) - 1`.  That is the butterfly step, and it is what makes
/// the primitive compose --- a sequence of `swap`s is a permutation of the
/// lane index bits, which is how a register reordering into a matrix fragment
/// gets built.
///
/// The branches used to implement two different maps.  `Block` 4, 16 and 64
/// toggled one bit, as above; `Block` 8 and 32 read lane `i ^ (Block - 1)`,
/// the *mirror* lane, which the comment in `reduction` also described.  Both
/// are defensible readings of the name and nothing here could tell them
/// apart: the sole caller is a butterfly reduction, where each group is
/// already uniform, so any lane of the neighbouring group answers and the two
/// maps are indistinguishable.
///
/// They are not indistinguishable to anything that needs an exact
/// permutation.  One bit toggled is the useful one -- mirroring flips every
/// low bit at once, which does not compose into an arbitrary bit permutation
/// -- so that is what this is, for every `Block`.
template <std::size_t Block, typename T>
__device__ __forceinline__ T swap(T value) {
  if constexpr (Block == 1) {
    return value;
  } else if constexpr (Block == 64) {
    const auto blockvar = __lane_id() / 32;
    const auto subblockvar = __lane_id() % 32;
    return bpermute(value, subblockvar + (1 - blockvar) * 32);
  } else if constexpr (Block == 32 || Block == 8) {
    constexpr auto AndMask = 0x1f;
    constexpr auto OrMask = 0x0;
    constexpr auto XorMask = Block / 2;
    return swizzle<AndMask, OrMask, XorMask>(value);
  } else if constexpr (Block == 16) {
    return dpp<0x128, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 4) {
    return dpp<0b01001110, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 2) {
    return dpp<0b10110001, 0xf, 0xf, true>(value);
  }
  return value;
}

#ifdef __GFX9__
#define CMVCC ", vcc"
#define CMFI ""
#else
#define CMVCC ""
#define CMFI " fi:1 "
#endif

// FMAC_DPP inline assembly
// !! MAY DISREGARD WAIT STATES !!

#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__) ||          \
    defined(__GFX13__)
#define ROW_BCST16 "row_share"
#else
#define ROW_BCST16 "row_newbcast"
#endif

#define ISTRINGIFY(x) #x
#define STR(x) ISTRINGIFY(x)
#define FMADPP4(pos, c, a, b)                                                  \
  __asm("v_fmac_f32_dpp %0, %1, %2 quad_perm:[" STR(pos) "," STR(pos) "," STR( \
            pos) "," STR(pos) "] row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
        :)
#define FMADPP16(pos, c, a, b)                                                 \
  __asm("v_fmac_f32_dpp %0, %1, %2 " ROW_BCST16                                \
        ":" STR(pos) " row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI           \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
        :)
#define DMADPP16(pos, c, a, b)                                                 \
  __asm("v_fmac_f64_dpp %0, %1, %2 " ROW_BCST16                                \
        ":" STR(pos) " row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI           \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
        :)

#define MOV64DPP16(pos, c, a)                                                  \
  __asm("v_mov_b64_dpp %0, %1 " ROW_BCST16                                     \
        ":" STR(pos) " row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI           \
        : "+v"(c)                                                              \
        : "v"(a)                                                               \
        :)

// format:
// c: accumulator
// a: DPP-broadcasted register
// b: multiplicand (vector reg)

template <int Row>
__device__ __forceinline__ void fmacdpp4(float &c, float a, float b);

template <int Row>
__device__ __forceinline__ void fmacdpp16(float &c, float a, float b);

template <int Row>
__device__ __forceinline__ void fmacdpp16(double &c, double a, double b);

template <int Row> __device__ __forceinline__ float2 movdpp16(float2 a);

template <int Row> __device__ __forceinline__ float movdpp16(float a) {
  return dpp<0x150 + Row, 0xf, 0xf, true>(a);
}

#if !defined(__gfx900__)
constexpr bool HasFmacDpp4 = true;

template <>
__device__ __forceinline__ void fmacdpp4<0>(float &c, float a, float b) {
  FMADPP4(0, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp4<1>(float &c, float a, float b) {
  FMADPP4(1, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp4<2>(float &c, float a, float b) {
  FMADPP4(2, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp4<3>(float &c, float a, float b) {
  FMADPP4(3, c, a, b);
}
#else
constexpr bool HasFmacDpp4 = false;
#endif

#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) ||       \
    defined(__gfx942__) || defined(__gfx950__) || defined(__GFX10__) ||        \
    defined(__GFX11__) || defined(__GFX12__) || defined(__GFX13__)
constexpr bool HasFmacDpp16 = true;

template <>
__device__ __forceinline__ void fmacdpp16<0>(float &c, float a, float b) {
  FMADPP16(0x0, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<1>(float &c, float a, float b) {
  FMADPP16(0x1, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<2>(float &c, float a, float b) {
  FMADPP16(0x2, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<3>(float &c, float a, float b) {
  FMADPP16(0x3, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<4>(float &c, float a, float b) {
  FMADPP16(0x4, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<5>(float &c, float a, float b) {
  FMADPP16(0x5, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<6>(float &c, float a, float b) {
  FMADPP16(0x6, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<7>(float &c, float a, float b) {
  FMADPP16(0x7, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<8>(float &c, float a, float b) {
  FMADPP16(0x8, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<9>(float &c, float a, float b) {
  FMADPP16(0x9, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<10>(float &c, float a, float b) {
  FMADPP16(0xa, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<11>(float &c, float a, float b) {
  FMADPP16(0xb, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<12>(float &c, float a, float b) {
  FMADPP16(0xc, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<13>(float &c, float a, float b) {
  FMADPP16(0xd, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<14>(float &c, float a, float b) {
  FMADPP16(0xe, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<15>(float &c, float a, float b) {
  FMADPP16(0xf, c, a, b);
}

template <int row>
__device__ __forceinline__ void fmacdpp16(double &c, double a, double b);

template <>
__device__ __forceinline__ void fmacdpp16<0>(double &c, double a, double b) {
  DMADPP16(0x0, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<1>(double &c, double a, double b) {
  DMADPP16(0x1, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<2>(double &c, double a, double b) {
  DMADPP16(0x2, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<3>(double &c, double a, double b) {
  DMADPP16(0x3, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<4>(double &c, double a, double b) {
  DMADPP16(0x4, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<5>(double &c, double a, double b) {
  DMADPP16(0x5, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<6>(double &c, double a, double b) {
  DMADPP16(0x6, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<7>(double &c, double a, double b) {
  DMADPP16(0x7, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<8>(double &c, double a, double b) {
  DMADPP16(0x8, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<9>(double &c, double a, double b) {
  DMADPP16(0x9, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<10>(double &c, double a, double b) {
  DMADPP16(0xa, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<11>(double &c, double a, double b) {
  DMADPP16(0xb, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<12>(double &c, double a, double b) {
  DMADPP16(0xc, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<13>(double &c, double a, double b) {
  DMADPP16(0xd, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<14>(double &c, double a, double b) {
  DMADPP16(0xe, c, a, b);
}
template <>
__device__ __forceinline__ void fmacdpp16<15>(double &c, double a, double b) {
  DMADPP16(0xf, c, a, b);
}

template <int row> __device__ __forceinline__ float2 movdpp16(float2 a);

template <> __device__ __forceinline__ float2 movdpp16<0>(float2 a) {
  float2 c{};
  MOV64DPP16(0x0, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<1>(float2 a) {
  float2 c{};
  MOV64DPP16(0x1, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<2>(float2 a) {
  float2 c{};
  MOV64DPP16(0x2, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<3>(float2 a) {
  float2 c{};
  MOV64DPP16(0x3, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<4>(float2 a) {
  float2 c{};
  MOV64DPP16(0x4, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<5>(float2 a) {
  float2 c{};
  MOV64DPP16(0x5, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<6>(float2 a) {
  float2 c{};
  MOV64DPP16(0x6, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<7>(float2 a) {
  float2 c{};
  MOV64DPP16(0x7, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<8>(float2 a) {
  float2 c{};
  MOV64DPP16(0x8, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<9>(float2 a) {
  float2 c{};
  MOV64DPP16(0x9, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<10>(float2 a) {
  float2 c{};
  MOV64DPP16(0xa, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<11>(float2 a) {
  float2 c{};
  MOV64DPP16(0xb, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<12>(float2 a) {
  float2 c{};
  MOV64DPP16(0xc, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<13>(float2 a) {
  float2 c{};
  MOV64DPP16(0xd, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<14>(float2 a) {
  float2 c{};
  MOV64DPP16(0xe, c, a);
  return c;
}
template <> __device__ __forceinline__ float2 movdpp16<15>(float2 a) {
  float2 c{};
  MOV64DPP16(0xf, c, a);
  return c;
}
#else
constexpr bool HasFmacDpp16 = false;

template <int Row> __device__ __forceinline__ float2 movdpp16(float2 a) {
  return dpp<0x150 + Row, 0xf, 0xf, true>(a);
}
#endif

template <int Row>
__device__ __forceinline__ void fmacdpp16(float2 &c, float2 a, float2 b) {
  const auto aa = movdpp16<Row>(a);
  c += aa * b;
}

template <typename T, std::size_t End = 4>
__device__ __forceinline__ void fma4h(T &c, T a, T b) {
  if constexpr (HasFmacDpp4 && std::is_same_v<T, float>) {
    if constexpr (End >= 0) {
      fmacdpp4<0>(c, a, b);
    }
    if constexpr (End >= 1) {
      fmacdpp4<1>(c, a, b);
    }
    if constexpr (End >= 2) {
      fmacdpp4<2>(c, a, b);
    }
    if constexpr (End >= 3) {
      fmacdpp4<3>(c, a, b);
    }
  } else {
    if constexpr (End >= 0) {
      const T aa =
          dpp<(0 << 6) | (0 << 4) | (0 << 2) | (0 << 0), 0xf, 0xf, true>(a);
      c += aa * b;
    }
    if constexpr (End >= 1) {
      const T aa =
          dpp<(1 << 6) | (1 << 4) | (1 << 2) | (1 << 0), 0xf, 0xf, true>(a);
      c += aa * b;
    }
    if constexpr (End >= 2) {
      const T aa =
          dpp<(2 << 6) | (2 << 4) | (2 << 2) | (2 << 0), 0xf, 0xf, true>(a);
      c += aa * b;
    }
    if constexpr (End >= 3) {
      const T aa =
          dpp<(3 << 6) | (3 << 4) | (3 << 2) | (3 << 0), 0xf, 0xf, true>(a);
      c += aa * b;
    }
  }
}

template <typename T, std::size_t End = 16>
__device__ __forceinline__ void fma16h(T &c, T a, T b) {
  if constexpr (HasFmacDpp16) {
    if constexpr (End >= 0) {
      fmacdpp16<0>(c, a, b);
    }
    if constexpr (End >= 1) {
      fmacdpp16<1>(c, a, b);
    }
    if constexpr (End >= 2) {
      fmacdpp16<2>(c, a, b);
    }
    if constexpr (End >= 3) {
      fmacdpp16<3>(c, a, b);
    }
    if constexpr (End >= 4) {
      fmacdpp16<4>(c, a, b);
    }
    if constexpr (End >= 5) {
      fmacdpp16<5>(c, a, b);
    }
    if constexpr (End >= 6) {
      fmacdpp16<6>(c, a, b);
    }
    if constexpr (End >= 7) {
      fmacdpp16<7>(c, a, b);
    }
    if constexpr (End >= 8) {
      fmacdpp16<8>(c, a, b);
    }
    if constexpr (End >= 9) {
      fmacdpp16<9>(c, a, b);
    }
    if constexpr (End >= 10) {
      fmacdpp16<10>(c, a, b);
    }
    if constexpr (End >= 11) {
      fmacdpp16<11>(c, a, b);
    }
    if constexpr (End >= 12) {
      fmacdpp16<12>(c, a, b);
    }
    if constexpr (End >= 13) {
      fmacdpp16<13>(c, a, b);
    }
    if constexpr (End >= 14) {
      fmacdpp16<14>(c, a, b);
    }
    if constexpr (End >= 15) {
      fmacdpp16<15>(c, a, b);
    }
  } else if constexpr (HasFmacDpp4 && std::is_same_v<T, float>) {
    if constexpr (End >= 0) {
      const T aa = dpp<0x118, 0xc, 0xf, true>(a);
      {
        const T aaa = dpp<0x114, 0xa, 0xf, true>(a);
        fma4h<T, End>(c, aaa, b);
      }
      if constexpr (End >= 4) {
        const T aaa = dpp<0x104, 0x5, 0xf, true>(a);
        fma4h<T, End - 4>(c, aaa, b);
      }
    }
    if constexpr (End >= 8) {
      const T aa = dpp<0x108, 0x3, 0xf, true>(a);
      {
        const T aaa = dpp<0x114, 0xa, 0xf, true>(a);
        fma4h<T, End - 8>(c, aaa, b);
      }
      if constexpr (End >= 4) {
        const T aaa = dpp<0x104, 0x5, 0xf, true>(a);
        fma4h<T, End - 12>(c, aaa, b);
      }
    }
  } else {
    const auto blockbase = (__lane_id() / 16) * 16;
    const auto subblockpos = __lane_id() % 4;
    if constexpr (End >= 0) {
      const auto aa = bpermute(a, blockbase + subblockpos);
      fma4h<T, End>(c, aa, b);
    }
    if constexpr (End >= 4) {
      const auto aa = bpermute(a, blockbase + subblockpos + 4);
      fma4h<T, End - 4>(c, aa, b);
    }
    if constexpr (End >= 8) {
      const auto aa = bpermute(a, blockbase + subblockpos + 8);
      fma4h<T, End - 8>(c, aa, b);
    }
    if constexpr (End >= 12) {
      const auto aa = bpermute(a, blockbase + subblockpos + 12);
      fma4h<T, End - 12>(c, aa, b);
    }
  }
}

// Bits at 0, Subblock, 2*Subblock, ... below Block: the lanes that share a
// reduction with lane 0 of a block.  64-bit, because a wavefront is.
template <std::size_t Block, std::size_t Subblock>
constexpr unsigned long long groupMask() {
  unsigned long long mask = 0;
  for (std::size_t k = 0; k < Block; k += Subblock) {
    mask |= 1ull << k;
  }
  return mask;
}

template <typename Op, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ bool ballotReduction(bool value) {
  const auto ballot = __ballot(value ? 1 : 0);
  const auto thread = (threadIdx.x / Block) * Block;
  const auto subthread = Subblock == 1 ? 0 : (threadIdx.x % Subblock);

  // `(1 << subthread) << thread` named a single lane, so the And test reduced
  // to "is my own bit set" and returned this lane's input unchanged.  It was
  // also an `int`, which a wavefront's 64 lanes overflow from lane 31 on.
  const auto mask = groupMask<Block, Subblock>() << (thread + subthread);

  if constexpr (Op::Op == Operation::And) {
    return (mask & ballot) == mask;
  }
  if constexpr (Op::Op == Operation::Or) {
    return (mask & ballot) != 0;
  }
  if constexpr (Op::Op == Operation::Xor) {
    return (__popc((mask & ballot)) & 1) == 0;
  }
}

template <typename Op, std::size_t Block, std::size_t Subblock, typename T>
__device__ __forceinline__ T reduction(const T &value) {
  if constexpr (Block == Subblock) {
    return value;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::And &&
                       Block == 64 && Subblock == 1) {
    return __all(value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::Or &&
                       Block == 64 && Subblock == 1) {
    return __any(value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool>) {
    return ballotReduction<Op, Block, Subblock>(value);
  }

  // `swap<N>` exchanges the two halves of an N-sized group.  Once each
  // Subblock-sized group holds a uniform value, exchanging across
  // `2*Subblock` pairs each group with its neighbour, which is the butterfly
  // step -- so the width has to grow with the recursion.  `swap<Block>`
  // repeated the same full-width exchange at every level instead.
  //
  // This reads any lane of the neighbouring group, and the group is uniform,
  // so it cannot tell which lane it got. That is why `swap` could carry two
  // different maps for as long as this was its only caller.
  const auto other = swap<(Subblock << 1)>(value);
  const auto result = Op::applyOperation(value, other);
  // Argument order is <Op, Block, Subblock, T>: `T` was being passed where
  // `Block` is declared, which does not compile once this branch is reached.
  return reduction<Op, Block, (Subblock << 1), T>(result);
}

constexpr std::size_t GlobalMemspace = 1;
constexpr std::size_t ConstantMemspace = 4;

// The attribute has to sit on a typedef *inside* a class template, not on an
// alias template.  On an alias template with a dependent element type GCC
// drops it -- with a `-Wattributes` warning, not an error -- and `VectorT<T,
// N>` silently becomes plain `T`: size 4, alignment 4, and a
// `*(VectorT<float,4>*)&buf[i] = v` that moves one element instead of four.
// Clang applies it, so hipcc never saw this; `target_lexic` spells the same
// type for a host target, which does not.
template <typename T, std::size_t N> struct VectorOf {
  //: Naturally aligned: `N * sizeof(T)`.  This is the type a wide access uses
  //: when the base *proves* that much alignment.
  typedef T type __attribute__((__vector_size__(N * sizeof(T))));
  //: Same width, element alignment only.  For a base that proves less than
  //: the full width: the compiler then splits the access instead of emitting
  //: one the hardware requires to be aligned, which is a slower correct
  //: access rather than an undefined fast one.
  typedef T relaxed
      __attribute__((__vector_size__(N * sizeof(T)), __aligned__(sizeof(T))));
};

template <typename T, std::size_t N>
using VectorT = typename VectorOf<T, N>::type;

template <typename T, std::size_t N>
using VectorRelaxedT = typename VectorOf<T, N>::relaxed;

template <typename T, std::size_t Space>
using SpacePtr = __attribute__((address_space(Space))) T *;

template <typename T, std::size_t Space>
using SpacePtrRestrict = __attribute__((address_space(Space))) T *__restrict;

template <typename T>
__device__ __forceinline__ void transpose4x4b32(T &w1, T &w2, T &w3, T &w4,
                                                T v1, T v2, T v3, T v4);

template <typename T>
__device__ __forceinline__ void
transpose16x16b32(T &w1, T &w2, T &w3, T &w4, T &w5, T &w6, T &w7, T &w8, T &w9,
                  T &w10, T &w11, T &w12, T &w13, T &w14, T &w15, T &w16) {

  T v1, v2, v3, v4;
  T v5, v6, v7, v8;
  T v9, v10, v11, v12;
  T v13, v14, v15, v16;

  // transpose 4x4

  transpose4x4b32(v1, v2, v3, v4, w1, w2, w3, w4);
  transpose4x4b32(v5, v6, v7, v8, w5, w6, w7, w8);
  transpose4x4b32(v9, v10, v11, v12, w9, w10, w11, w12);
  transpose4x4b32(v13, v14, v15, v16, w13, w14, w15, w16);

  // from here on: DPP and row control suffice

  // transpose 8x8

  const T u1 = dppUpdate<0x124, 0b1111, 0b1010, true>(v5, v1);
  const T u2 = dppUpdate<0x124, 0b1111, 0b1010, true>(v6, v2);
  const T u3 = dppUpdate<0x124, 0b1111, 0b1010, true>(v7, v3);
  const T u4 = dppUpdate<0x124, 0b1111, 0b1010, true>(v8, v4);

  const T u5 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v1, v5);
  const T u6 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v2, v6);
  const T u7 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v3, v7);
  const T u8 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v4, v8);

  const T u9 = dppUpdate<0x124, 0b1111, 0b1010, true>(v13, v9);
  const T u10 = dppUpdate<0x124, 0b1111, 0b1010, true>(v14, v10);
  const T u11 = dppUpdate<0x124, 0b1111, 0b1010, true>(v15, v11);
  const T u12 = dppUpdate<0x124, 0b1111, 0b1010, true>(v16, v12);

  const T u13 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v9, v13);
  const T u14 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v10, v14);
  const T u15 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v11, v15);
  const T u16 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v12, v16);

  // transpose 16x16

  w1 = dppUpdate<0x128, 0b1111, 0b1100, true>(u9, u1);
  w2 = dppUpdate<0x128, 0b1111, 0b1100, true>(u10, u2);
  w3 = dppUpdate<0x128, 0b1111, 0b1100, true>(u11, u3);
  w4 = dppUpdate<0x128, 0b1111, 0b1100, true>(u12, u4);
  w5 = dppUpdate<0x128, 0b1111, 0b1100, true>(u13, u5);
  w6 = dppUpdate<0x128, 0b1111, 0b1100, true>(u14, u6);
  w7 = dppUpdate<0x128, 0b1111, 0b1100, true>(u15, u7);
  w8 = dppUpdate<0x128, 0b1111, 0b1100, true>(u16, u8);

  w9 = dppUpdate<0x128, 0b1111, 0b0011, true>(u1, u9);
  w10 = dppUpdate<0x128, 0b1111, 0b0011, true>(u2, u10);
  w11 = dppUpdate<0x128, 0b1111, 0b0011, true>(u3, u11);
  w12 = dppUpdate<0x128, 0b1111, 0b0011, true>(u4, u12);
  w13 = dppUpdate<0x128, 0b1111, 0b0011, true>(u5, u13);
  w14 = dppUpdate<0x128, 0b1111, 0b0011, true>(u6, u14);
  w15 = dppUpdate<0x128, 0b1111, 0b0011, true>(u7, u15);
  w16 = dppUpdate<0x128, 0b1111, 0b0011, true>(u8, u16);
}

template <typename T>
__device__ __forceinline__ void transpose16x2(T &w1, T &w2, T v1, T v2) {
  w1 = dppUpdate<0x128, 0b1111, 0b1100, true>(v2, v1);
  w2 = dppUpdate<0x128, 0b1111, 0b0011, true>(v1, v2);
}

template <typename T>
__device__ __forceinline__ void transpose16x4(T &w1, T &w2, T &w3, T &w4, T v1,
                                              T v2, T v3, T v4) {
  const T u1 = dppUpdate<0x124, 0b1111, 0b1010, true>(v2, v1);
  const T u2 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v1, v2);
  const T u3 = dppUpdate<0x124, 0b1111, 0b1010, true>(v4, v3);
  const T u4 = dppUpdate<0x12c, 0b1111, 0b0101, true>(v3, v4);

  transpose16x2(w1, w3, u1, u3);
  transpose16x2(w2, w4, u2, u4);
}

#define CM4STR(p1, p2, p3, p4, c, a, b)                                        \
  "v_cndmask_b32_dpp " c ", " a ", " b CMVCC                                   \
  " quad_perm:[" STR(p1) "," STR(p2) "," STR(p3) "," STR(                      \
      p4) "] row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI
#define CMRSTR(cnt, c, a, b)                                                   \
  "v_cndmask_b32_dpp " c ", " a ", " b CMVCC                                   \
  " row_ror:" STR(cnt) " row_mask:0xf bank_mask:0xf bound_ctrl:1" CMFI

template <typename T>
__device__ __forceinline__ void transpose4x4b32(T &w1, T &w2, T &w3, T &w4,
                                                T v1, T v2, T v3, T v4) {

  const uint64_t mask1a = 0x5555555555555555ULL;
  const uint64_t mask1b = 0xaaaaaaaaaaaaaaaaULL;
  const uint64_t mask2a = 0x3333333333333333ULL;
  const uint64_t mask2b = 0xccccccccccccccccULL;

  // T u1, u2, u3, u4;

  // 11 12 13 14
  // 21 22 23 24
  // 31 32 33 34
  // 41 42 43 44

  // 11 21 13 23 (DPP for row 2)
  // 12 22 14 24 (DPP for row 1)
  // 31 41 33 43 (DPP for row 4)
  // 32 42 34 44 (DPP for row 3)

  // 11 21 31 41 (DPP for row 3)
  // 12 22 32 42 (DPP for row 4)
  // 13 23 33 43 (DPP for row 1)
  // 14 24 34 44 (DPP for row 2)

  // clang-format off

  /*
  __asm("s_mov_b64 vcc, %[mask] \n\t"
  CM4STR(0, 0, 2, 2, "%[u1]", "%[v2]", "%[v1]") "\n\t"
  CM4STR(0, 0, 2, 2, "%[u3]", "%[v4]", "%[v3]")
  : [u1] "=&v" (u1), [u3] "=&v" (u3)
  : [mask] "s" (mask1a), [v1] "v" (v1), [v2] "v" (v2), [v3] "v" (v3), [v4] "v" (v4)
  : "vcc");
  __asm("s_mov_b64 vcc, %[mask] \n\t"
  CM4STR(1, 1, 3, 3, "%[u2]", "%[v1]", "%[v2]") "\n\t"
  CM4STR(1, 1, 3, 3, "%[u4]", "%[v3]", "%[v4]")
  : [u2] "=&v" (u2), [u4] "=&v" (u4)
  : [mask] "s" (mask1b), [v1] "v" (v1), [v2] "v" (v2), [v3] "v" (v3), [v4] "v" (v4)
  : "vcc");
  __asm("s_mov_b64 vcc, %[mask] \n\t"
  CM4STR(0, 1, 0, 1, "%[w1]", "%[u3]", "%[u1]") "\n\t"
  CM4STR(0, 1, 0, 1, "%[w2]", "%[u4]", "%[u2]")
  : [w1] "=&v" (w1), [w2] "=&v" (w2)
  : [mask] "s" (mask2a), [u1] "v" (u1), [u2] "v" (u2), [u3] "v" (u3), [u4] "v" (u4)
  : "vcc");
  __asm("s_mov_b64 vcc, %[mask] \n\t"
  CM4STR(2, 3, 2, 3, "%[w3]", "%[u1]", "%[u3]") "\n\t"
  CM4STR(2, 3, 2, 3, "%[w4]", "%[u2]", "%[u4]")
  : [w3] "=&v" (w3), [w4] "=&v" (w4)
  : [mask] "s" (mask2b), [u1] "v" (u1), [u2] "v" (u2), [u3] "v" (u3), [u4] "v" (u4)
  : "vcc");
  */

  // clang-format on

  // code w/o inline assembly (doesn't combine cndmask and dpp)

  const auto vv2 = dpp<0xa0, 0xf, 0xf, true>(v2);
  const auto vv4 = dpp<0xa0, 0xf, 0xf, true>(v4);
  const auto vv1 = dpp<0xf5, 0xf, 0xf, true>(v1);
  const auto vv3 = dpp<0xf5, 0xf, 0xf, true>(v3);

  const auto u1 = __lane_id() % 2 == 0 ? v1 : vv2;
  const auto u2 = __lane_id() % 2 == 1 ? v2 : vv1;
  const auto u3 = __lane_id() % 2 == 0 ? v3 : vv4;
  const auto u4 = __lane_id() % 2 == 1 ? v4 : vv3;

  const auto uu1 = dpp<0xee, 0xf, 0xf, true>(u1);
  const auto uu2 = dpp<0xee, 0xf, 0xf, true>(u2);
  const auto uu3 = dpp<0x44, 0xf, 0xf, true>(u3);
  const auto uu4 = dpp<0x44, 0xf, 0xf, true>(u4);

  w1 = __lane_id() % 4 < 2 ? u1 : uu3;
  w2 = __lane_id() % 4 < 2 ? u2 : uu4;
  w3 = __lane_id() % 4 >= 2 ? u3 : uu1;
  w4 = __lane_id() % 4 >= 2 ? u4 : uu2;
}

/*
// forward declares for structured buffer accessors
// (for "canonical" float4 loads)

using OldBufferRsrc = VectorT<std::int32_t, 4>;

extern "C" {

__device__ int
llvm_struct_buffer_load_i32(OldBufferRsrc, int, int, int,
                            int) __asm("llvm.amdgcn.struct.buffer.load.i32");
__device__ VectorT<int, 2> llvm_struct_buffer_load_v2i32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v2i32");
__device__ VectorT<int, 3> llvm_struct_buffer_load_v3i32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v3i32");
__device__ VectorT<int, 4> llvm_struct_buffer_load_v4i32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v4i32");

__device__ float
llvm_struct_buffer_load_f32(OldBufferRsrc, int, int, int,
                            int) __asm("llvm.amdgcn.struct.buffer.load.f32");
__device__ VectorT<float, 2> llvm_struct_buffer_load_v2f32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v2f32");
__device__ VectorT<float, 3> llvm_struct_buffer_load_v3f32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v3f32");
__device__ VectorT<float, 4> llvm_struct_buffer_load_v4f32(
    OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.load.v4f32");

__device__ int
llvm_struct_buffer_load_u8(OldBufferRsrc, int, int, int,
                           int) __asm("llvm.amdgcn.struct.buffer.load.i8");
__device__ int
llvm_struct_buffer_load_u16(OldBufferRsrc, int, int, int,
                            int) __asm("llvm.amdgcn.struct.buffer.load.i16");

__device__ void
llvm_struct_buffer_store_i32(int, OldBufferRsrc, int, int, int,
                             int) __asm("llvm.amdgcn.struct.buffer.store.i32");
__device__ void llvm_struct_buffer_store_v4i32(
    VectorT<int, 4>, OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.store.v4i32");
__device__ void
llvm_struct_buffer_store_f32(float, OldBufferRsrc, int, int, int,
                             int) __asm("llvm.amdgcn.struct.buffer.store.f32");
__device__ void llvm_struct_buffer_store_v4f32(
    VectorT<float, 4>, OldBufferRsrc, int, int, int,
    int) __asm("llvm.amdgcn.struct.buffer.store.v4f32");

} // extern "C"
*/
/*

enum BufferHints {

  Swizzled = 1 << 3
};

template<typename T>
class Buffer {
public:
    Buffer() {
        descriptor = __builtin_amdgcn_make_buffer_rsrc();
    }

    template<typename T, BufferHints Hints>
    T load(std::size_t offset) {
        if constexpr (sizeof(T) == 16) {
            __builtin_amdgcn_raw_buffer_store_b128
        }
    }

    template<typename T>
    store(std::size_t offset, const T& value) {

    }

    template<typename T>
    T atomic() {

    }
private:
    __amdgpu_buffer_rsrc_t descriptor;
};
*/

__device__ __forceinline__ std::tuple<short, short, short>
splitFloatBF16(float input) {
  const auto i1 = static_cast<__bf16>(input);
  const auto i1r = input - static_cast<float>(i1);
  const auto i2 = static_cast<__bf16>(i1r);
  const auto i2r = i1r - static_cast<float>(i2);
  const auto i3 = static_cast<__bf16>(i2r);
  const auto r1 = *reinterpret_cast<const short *>(&i1);
  const auto r2 = *reinterpret_cast<const short *>(&i2);
  const auto r3 = *reinterpret_cast<const short *>(&i3);
  return {r1, r2, r3};
}

__device__ __forceinline__
    std::tuple<VectorT<short, 4>, VectorT<short, 4>, VectorT<short, 4>>
    splitFloatx4BF16(float i1, float i2, float i3, float i4) {
  const auto [i1p0, i1p1, i1p2] = splitFloatBF16(i1);
  const auto [i2p0, i2p1, i2p2] = splitFloatBF16(i2);
  const auto [i3p0, i3p1, i3p2] = splitFloatBF16(i3);
  const auto [i4p0, i4p1, i4p2] = splitFloatBF16(i4);
  return {VectorT<short, 4>{i1p0, i2p0, i3p0, i4p0},
          VectorT<short, 4>{i1p1, i2p1, i3p1, i4p1},
          VectorT<short, 4>{i1p2, i2p2, i3p2, i4p2}};
}

__device__ __forceinline__ std::tuple<_Float16, _Float16>
splitFloatF16(float input) {
  const auto i1 = static_cast<_Float16>(input);
  const auto i1r = input - static_cast<float>(i1);
  const auto i2 = static_cast<_Float16>(i1r);
  return {i1, i2};
}

__device__
    __forceinline__ std::tuple<VectorT<_Float16, 4>, VectorT<_Float16, 4>>
    splitFloatx4F16(float i1, float i2, float i3, float i4) {
  const auto [i1p0, i1p1] = splitFloatF16(i1);
  const auto [i2p0, i2p1] = splitFloatF16(i2);
  const auto [i3p0, i3p1] = splitFloatF16(i3);
  const auto [i4p0, i4p1] = splitFloatF16(i4);
  return {VectorT<_Float16, 4>{i1p0, i2p0, i3p0, i4p0},
          VectorT<_Float16, 4>{i1p1, i2p1, i3p1, i4p1}};
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_H_
