// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>
#include <sycl/sycl.hpp>

#include "base.h"

namespace tensorforge {
namespace intel_esimd = sycl::ext::intel::esimd;
namespace intel_xmx = intel_esimd::xmx;

using TF32 = sycl::ext::intel::experimental::esimd::tfloat32;

/// Split a vector of floats into the two TF32 halves a DPAS multiplies.
///
/// The same arrangement as `splitFloatTF32` in `cuda.h`, and it has to be:
/// both feed a three-term product whose error analysis is the split's, not
/// the instruction's.  TF32 keeps 11 mantissa bits against FP32's 24, so
/// `upper` holds the top 11 and `lower` the next 11 of what is left; the
/// remaining two bits fall below what the accumulator distinguishes.
///
/// Written as a conversion rather than as PTX because ESIMD has one: assigning
/// a `simd<float, N>` into a `simd<TF32, N>` rounds per element, which is what
/// `cvt.rna.tf32.f32` does one value at a time.
template <int N>
ESIMD_INLINE void splitFloatTF32(intel_esimd::simd<TF32, N> &upper,
                                 intel_esimd::simd<TF32, N> &lower,
                                 intel_esimd::simd<float, N> value) {
  // Explicit both ways.  The conversion constructor is implicit in ESIMD, but
  // spelling it out is what makes the two roundings visible: this is where the
  // 13 mantissa bits FP32 has over TF32 are dropped, and the next line is
  // where they are picked back up.
  upper = intel_esimd::simd<TF32, N>(value);
  // `upper` back to float first: the residual is what the rounding dropped,
  // and subtracting the *unrounded* value would give zero.
  const intel_esimd::simd<float, N> upperF = intel_esimd::simd<float, N>(upper);
  lower = intel_esimd::simd<TF32, N>(value - upperF);
}
} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
