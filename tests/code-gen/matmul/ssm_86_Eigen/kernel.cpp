#include <cassert>
#include <cstring>
#include <cstdlib>
#include <limits>
#include "subroutine.h"
#include "kernel.h"
namespace yateto {
  constexpr unsigned long const kernel::matmulAB::NonZeroFlops;
  constexpr unsigned long const kernel::matmulAB::HardwareFlops;
  void kernel::matmulAB::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    assert(numElements != 0);
    assert(streamPtr != reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()));
    launcher_kernel_386f7503c1(C, extraOffset_C, const_cast<const float**>(A), extraOffset_A, const_cast<const float**>(B), extraOffset_B, numElements, flags, streamPtr);
    streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
    flags = nullptr;
  }
  constexpr unsigned long const kernel::matmulATB::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATB::HardwareFlops;
  void kernel::matmulATB::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    assert(numElements != 0);
    assert(streamPtr != reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()));
    launcher_kernel_9dfefadc54(const_cast<const float**>(A), extraOffset_A, const_cast<const float**>(B), extraOffset_B, C, extraOffset_C, numElements, flags, streamPtr);
    streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
    flags = nullptr;
  }
  constexpr unsigned long const kernel::matmulABT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulABT::HardwareFlops;
  void kernel::matmulABT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    assert(numElements != 0);
    assert(streamPtr != reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()));
    launcher_kernel_44a6a7e323(const_cast<const float**>(A), extraOffset_A, C, extraOffset_C, const_cast<const float**>(B), extraOffset_B, numElements, flags, streamPtr);
    streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
    flags = nullptr;
  }
  constexpr unsigned long const kernel::matmulATBT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATBT::HardwareFlops;
  void kernel::matmulATBT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    assert(numElements != 0);
    assert(streamPtr != reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()));
    float *_tmp0;
    float* _buffer0;
    _tmp0 = _buffer0;
    launcher_kernel_86dd44de64(const_cast<const float**>(A), extraOffset_A, C, extraOffset_C, const_cast<const float**>(B), extraOffset_B, numElements, flags, streamPtr);
    streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
    flags = nullptr;
  }
} // namespace yateto
