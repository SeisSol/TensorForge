#ifndef YATETO_KERNEL_H_
#define YATETO_KERNEL_H_
#include <cmath>
#include <limits>
#include "yateto.h"
#include "tensor.h"
namespace yateto {
  namespace kernel {
    struct matmulAB {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 0;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;
      yateto::LinearAllocatorT<float> linearAllocator;

      float const** A{};
      float const** B{};
      float** C{};

      unsigned numElements = 0;
      void *streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
      unsigned *flags = nullptr;
      int extraOffset_A{};
      int extraOffset_B{};
      int extraOffset_C{};

      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulATB {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 0;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;
      yateto::LinearAllocatorT<float> linearAllocator;

      float const** A{};
      float const** B{};
      float** C{};

      unsigned numElements = 0;
      void *streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
      unsigned *flags = nullptr;
      int extraOffset_A{};
      int extraOffset_B{};
      int extraOffset_C{};

      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulABT {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 0;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;
      yateto::LinearAllocatorT<float> linearAllocator;

      float const** A{};
      float const** B{};
      float** C{};

      unsigned numElements = 0;
      void *streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
      unsigned *flags = nullptr;
      int extraOffset_A{};
      int extraOffset_B{};
      int extraOffset_C{};

      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulATBT {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 0;
      constexpr static unsigned long const TmpMemRequiredInBytes = 4096;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 4096;
      yateto::LinearAllocatorT<float> linearAllocator;

      float const** A{};
      float const** B{};
      float** C{};

      unsigned numElements = 0;
      void *streamPtr = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());
      unsigned *flags = nullptr;
      int extraOffset_A{};
      int extraOffset_B{};
      int extraOffset_C{};

      void execute();
    };
  } // namespace kernel
} // namespace yateto
#endif
