#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>
#include <sycl/sycl.hpp>

#include "base.h"

namespace tensorforge {
namespace intel_esimd = sycl::ext::intel::esimd;
namespace intel_xmx = iesimd::xmx;

using TF32 = sycl::ext::intel::experimental::esimd::tfloat32;
} // namespace tensorforge
