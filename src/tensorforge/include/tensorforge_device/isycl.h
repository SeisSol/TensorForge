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
} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
