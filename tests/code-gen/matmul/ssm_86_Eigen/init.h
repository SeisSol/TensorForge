#ifndef YATETO_INIT_H_
#define YATETO_INIT_H_
#include "tensor.h"
#include "yateto.h"
namespace yateto {
  namespace init {
    struct A : tensor::A {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        typedef ::yateto::DenseTensorView<2,float,unsigned> type;
        static inline type create(float* values) {
          return ::yateto::DenseTensorView<2,float,unsigned>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
    struct B : tensor::B {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        typedef ::yateto::DenseTensorView<2,float,unsigned> type;
        static inline type create(float* values) {
          return ::yateto::DenseTensorView<2,float,unsigned>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
    struct C : tensor::C {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        typedef ::yateto::DenseTensorView<2,float,unsigned> type;
        static inline type create(float* values) {
          return ::yateto::DenseTensorView<2,float,unsigned>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
  } // namespace init
} // namespace yateto
#endif
