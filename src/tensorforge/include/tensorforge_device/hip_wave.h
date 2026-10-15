// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_WAVE_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_WAVE_H_

// What every lane of a wave holds alike, read once for the wave.
//
// Nothing here needs the HIP runtime beyond its execution-space keywords, so
// that a compile for a device can check it on its own: only an instruction
// stream shows what the compiler made of it (`tests/test_wave_uniform.py`).

namespace tensorforge {

/// The width of a wave, as a constant `if constexpr` can test: 64 lanes on
/// gfx9 and earlier, 32 from gfx10 on.
///
/// The architecture decides it because nothing else is a constant to the
/// front end.  clang folds `__builtin_amdgcn_wavefrontsize()` in the backend,
/// and `__AMDGCN_WAVEFRONT_SIZE` is deprecated from clang 20 on and removed in
/// 22.  AMD's advice, for code that needs the width at compile time, is a
/// constant of its own from the architecture.  It is also all that HIP
/// builds: `hip_runtime.h` refuses wave64 on gfx10 and later.  The host pass
/// parses device code without running it, and 64 there only lets it parse.
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__) ||          \
    defined(__GFX13__)
inline constexpr int WaveSize = 32;
#else
inline constexpr int WaveSize = 64;
#endif

// A compile for another width -- `-mwavefrontsize64` on gfx10 and later --
// stops here where clang names its width in the macro and has no builtin for
// it yet (clang 18).  Where the builtin exists the macro is on its way out,
// and is left alone.
#if defined(__AMDGCN_WAVEFRONT_SIZE) && defined(__has_builtin)
#if !__has_builtin(__builtin_amdgcn_wavefrontsize)
static_assert(__AMDGCN_WAVEFRONT_SIZE == WaveSize,
              "a wave of another width than HIP builds on this architecture");
#endif
#endif

/// `value`, read as the first active lane's, where every lane of `Lanes`
/// consecutive ones holds it alike: the row of a block whose multiplications
/// are `Lanes` wide, and an address formed from it.  The compiler cannot see
/// that the row is the same across a wave, since it does not know the block;
/// told, it holds the value once, in a scalar register, computes with it on
/// the scalar unit, and lets a vector load take it as its base.
///
/// Only where `Lanes` is a whole number of waves: two multiplications in one
/// wave have two rows, and the first lane holds one of them.
///
/// In 32-bit pieces, the width `readfirstlane` takes: a pointer is two.
template <int Lanes, typename T>
__device__ __forceinline__ auto wave_uniform(T value) -> T {
  static_assert(sizeof(T) % sizeof(int) == 0,
                "a value read as the wave's is a whole number of 32-bit words");
  if constexpr (Lanes % WaveSize != 0) {
    return value;
  } else {
    union Words {
      T value;
      int words[sizeof(T) / sizeof(int)];
    };
    Words in;
    Words out;
    in.value = value;
#pragma unroll
    for (unsigned i = 0; i < sizeof(T) / sizeof(int); ++i) {
      out.words[i] = __builtin_amdgcn_readfirstlane(in.words[i]);
    }
    return out.value;
  }
}

} // namespace tensorforge

#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_HIP_WAVE_H_
