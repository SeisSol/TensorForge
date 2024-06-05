#ifndef YATETO_TENSOR_H_
#define YATETO_TENSOR_H_
namespace yateto {
  namespace tensor {
    struct A {
      constexpr static unsigned const Shape[2] = {32, 32};
      constexpr static unsigned const Size = 1024;
      constexpr static unsigned size() {
        return Size;
      }
    };
    struct B {
      constexpr static unsigned const Shape[2] = {32, 32};
      constexpr static unsigned const Size = 1024;
      constexpr static unsigned size() {
        return Size;
      }
    };
    struct C {
      constexpr static unsigned const Shape[2] = {32, 32};
      constexpr static unsigned const Size = 1024;
      constexpr static unsigned size() {
        return Size;
      }
    };
  } // namespace tensor
  namespace tensor {
  } // namespace tensor
} // namespace yateto
#endif
