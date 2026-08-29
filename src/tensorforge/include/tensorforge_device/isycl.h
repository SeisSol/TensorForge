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

/// The same 19-bit E8M10, and here it is a real type rather than a bit
/// pattern: `simd<float, N>` does not convert to `simd<tf32, N>` implicitly,
/// so a fragment staged with the wrong precision is a compile error.  On CUDA
/// the constraint letter forces a typedef; see `cuda.h`.
using tf32 = sycl::ext::intel::experimental::esimd::tfloat32;
/// Kept for the existing spelling in `isycl.h`'s own helpers.
using TF32 = tf32;

/// Split a float into the two TF32 halves a DPAS multiplies.
///
/// The same arrangement as `splitFloatTF32` in `cuda.h`, and it has to be:
/// both feed a three-term product whose error analysis is the split's, not
/// the instruction's.  TF32 keeps 11 mantissa bits against FP32's 24, so
/// `upper` holds the top 11 and `lower` the next 11 of what is left; the
/// remaining two fall below what the accumulator distinguishes.
///
/// The destinations are templates because a DPAS fragment is staged one
/// element at a time, so what arrives is `frag.select<1, 1>(slot)` -- a view
/// into a vector, not a vector.  A signature taking `simd<tf32, N>&` cannot
/// bind one, and taking the fragment plus an index instead would put the
/// layout arithmetic in two places.  Views are proxies and go by value.
template <typename UpperT, typename LowerT, typename ValueT>
ESIMD_INLINE void splitFloatTF32(UpperT upper, LowerT lower, ValueT value) {
  // Explicit both ways.  The conversion is implicit in ESIMD, but spelling it
  // out is what makes the two roundings visible: this is where the 13
  // mantissa bits FP32 has over TF32 are dropped, and the next line is where
  // they are picked back up.
  upper = static_cast<tf32>(value);
  const auto upperF = static_cast<float>(static_cast<tf32>(value));
  lower = static_cast<tf32>(value - upperF);
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
