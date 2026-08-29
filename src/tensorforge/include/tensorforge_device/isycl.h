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

/// Split a vector of floats into the two TF32 halves a DPAS multiplies.
///
/// The same arrangement as `splitFloatTF32` in `cuda.h`, and it has to be:
/// both feed a three-term product whose error analysis is the split's and not
/// the instruction's.  TF32 keeps 11 mantissa bits against FP32's 24, so
/// `upper` holds the top 11 and `lower` the next 11 of what is left; the
/// remaining two fall below what the accumulator distinguishes.
///
/// A whole vector at a time, and the destinations are templates, because a
/// DPAS fragment is filled by *runs*: `Src1[k * N + n]` is `A(n, k)`, so the
/// sixteen lanes an operand load already returns land in sixteen consecutive
/// slots.  What arrives here is therefore `frag.select<16, 1>(k * 16)` -- a
/// view into a vector -- and a signature taking `simd<tf32, N> &` cannot bind
/// one.  Views are proxies and go by value.
template <int N, typename UpperT, typename LowerT, typename ValueT>
ESIMD_INLINE void splitFloatTF32(UpperT upper, LowerT lower, ValueT value) {
  // `N` is explicit at the call site rather than deduced: all three operands
  // can be views into larger fragments -- `Src2` is filled a repeat row at a
  // time out of an operand that is itself a run -- and a view carries the
  // *parent's* length in its type, so nothing here could deduce the run
  // width from it.
  const intel_esimd::simd<float, N> v(value);
  const intel_esimd::simd<tf32, N> hi(v);
  const intel_esimd::simd<float, N> hiF(hi);
  upper = hi;
  lower = intel_esimd::simd<tf32, N>(v - hiF);
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
