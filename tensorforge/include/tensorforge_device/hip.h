#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>

#include "base.h"

namespace tensorforge {

#ifdef __gfx900__
static constexpr int AsmVersion = 9000;
#endif
#ifdef __gfx906__
static constexpr int AsmVersion = 9006;
#endif
#ifdef __gfx908__
static constexpr int AsmVersion = 9008;
#endif
#ifdef __gfx90a__
static constexpr int AsmVersion = 9010;
#endif
#ifdef __gfx942__
static constexpr int AsmVersion = 9402;
#endif
#ifdef __gfx950__
static constexpr int AsmVersion = 9500;
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
  } else if constexpr (Block == 16 && Subblock == 1 && AsmVersion >= 9010 &&
                       AsmVersion < 10000) {
    return dpp<0x150 + Lane, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 16 && Subblock == 8 && false) {
    // TODO: row mask
    return dpp<0x128, 0xf, 0xf, true>(value);
  } else if constexpr (Block == 4 && Subblock == 1) {
    return dpp<(Lane << 6) | (Lane << 4) | (Lane << 2) | (Lane << 0), 0xf, 0xf,
               true>(value);
  } else if constexpr (Block == 4 && Subblock == 2) {
    return dpp<((Lane + 1) << 6) | (Lane << 4) | ((Lane + 1) << 2) |
                   (Lane << 0),
               0xf, 0xf, true>(value);
  } else if constexpr (Block == 8 || Block == 16 || Block == 32) {
    return swizzle_bcst<Block, Subblock, Lane>(value);
  } else {
    const auto blockoffset = Block == 64 ? 0 : (__lane_id() / Block) * Block;
    const auto subblockvar = __lane_id() % Subblock;
    return bpermute(value, Lane * Subblock + subblockvar + blockoffset);
  }
  return value;
}

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
    constexpr auto XorMask = Block - 1;
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

// FMAC_DPP inline assembly
// !! MAY DISREGARD WAIT STATES !!

#define ISTRINGIFY(x) #x
#define STR(x) ISTRINGIFY(x)
#define FMADPP4(pos, c, a, b)                                                  \
  __asm("v_fmac_f32_dpp %0, %1, %2 quad_perm:[" STR(pos) "," STR(pos) "," STR( \
            pos) "," STR(pos) "] row_mask:0xf bank_mask:0xf bound_ctrl:1"      \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
        :)
#define FMADPP16(pos, c, a, b)                                                 \
  __asm("v_fmac_f32_dpp %0, %1, %2 row_newbcast:" STR(                         \
            pos) " row_mask:0xf bank_mask:0xf bound_ctrl:1"                    \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
        :)
#define DMADPP16(pos, c, a, b)                                                 \
  __asm("v_fmac_f64_dpp %0, %1, %2 row_newbcast:" STR(                         \
            pos) " row_mask:0xf bank_mask:0xf bound_ctrl:1"                    \
        : "+v"(c)                                                              \
        : "v"(a), "v"(b)                                                       \
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

#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) ||       \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||       \
    defined(__gfx950__)
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
    defined(__gfx942__)
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
#else
constexpr bool HasFmacDpp16 = false;
#endif

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

template <typename Op, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ bool ballotReduction(bool value) {
  const auto ballot = __ballot(warpSize, value ? 1 : 0);
  const auto thread = (threadIdx.x / Block) * Block;
  const auto subthread = Subblock == 1 ? 0 : (threadIdx.x % Subblock);
  constexpr auto basemask = 1;

  const auto mask = (basemask << subthread) << thread;

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
    return __all_sync(-1, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::Or &&
                       Block == 64 && Subblock == 1) {
    return __any_sync(-1, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool>) {
    return ballotReduction<Op, Block, Subblock>(value);
  }

  const auto other = swap<Block>(value);
  const auto result = Op::applyOperation(value, other);
  return reduction<Op, T, Block, (Subblock << 1)>(result);
}

/*
class Buffer {
public:
    Buffer() {
        descriptor = __builtin_amdgcn_make_buffer_rsrc();
    }

    template<typename T>
    T load(std::size_t offset) {
        if constexpr (sizeof(T) == 16) {
            __builtin_amdgcn_raw_buffer_store_b128
        }
    }

    template<typename T>
    store(std::size_t offset, const T& value) {

    }

    void copy() {

    }
private:
    __amdgpu_buffer_rsrc_t descriptor;
};

template<typename T>
class Tensor {

};

class Loader {
public:
    void start();
};
*/

} // namespace tensorforge
