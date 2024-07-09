#ifndef YATETO_KERNELTEST_T_H_
#define YATETO_KERNELTEST_T_H_
#include "kernel.h"
#include "init.h"
#include "yateto.h"
#ifndef NDEBUG
#ifndef YATETO_TESTING_NO_FLOP_COUNTER
long long libxsmm_num_total_flops = 0;
long long pspamm_num_total_flops = 0;
#endif
#endif
#include <cxxtest/TestSuite.h>
namespace yateto {
  namespace unit_test {
    class KernelTestSuite;
  } // namespace unit_test
} // namespace yateto
class yateto::unit_test::KernelTestSuite : public CxxTest::TestSuite {
public:
  void testmatmulAB() {
    int numElements = 1;
    yateto::LinearAllocatorT<float> linearAllocator;
    float* A = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      A[i] = static_cast<float>((i + 0) % 512 + 1);
    }
    float* _ut_A = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_A(_ut_A, {32, 32}, {0, 0}, {32, 32});
    init::A::view::create(A).copyToView(_view__ut_A);

    float* B = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      B[i] = static_cast<float>((i + 1) % 512 + 1);
    }
    float* _ut_B = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_B(_ut_B, {32, 32}, {0, 0}, {32, 32});
    init::B::view::create(B).copyToView(_view__ut_B);

    float* C = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      C[i] = static_cast<float>((i + 2) % 512 + 1);
    }
    float* _ut_C = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_C(_ut_C, {32, 32}, {0, 0}, {32, 32});
    init::C::view::create(C).copyToView(_view__ut_C);

    kernel::matmulAB krnl;
    krnl.numElements = numElements;
    krnl.streamPtr = nullptr;
    const float* tempA = A;
    krnl.A = &tempA;
    const float* tempB = B;
    krnl.B = &tempB;
    krnl.C = C;
    krnl.execute();

    float *_tmp0;
    float* _buffer0 = linearAllocator.allocate(numElements * 1024);
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 1024 * sizeof(float));
    for (int _k = 0; _k < 32; ++_k) {
      for (int _j = 0; _j < 32; ++_j) {
        for (int _i = 0; _i < 32; ++_i) {
          _tmp0[1*_i + 32*_j] += _ut_A[1*_i + 32*_k] * _ut_B[1*_k + 32*_j];
        }
      }
    }
    for (int _b = 0; _b < 32; ++_b) {
      for (int _a = 0; _a < 32; ++_a) {
        _ut_C[1*_a + 32*_b] = _tmp0[1*_a + 32*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 32; ++_b) {
        for (int _a = 0; _a < 32; ++_a) {
          double ref = _ut_C[1*_a + 32*_b];
          double diff = ref - C[1*_a + 32*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 1.19e-05);
    }
    linearAllocator.free();
  }
  void testmatmulATB() {
    int numElements = 1;
    yateto::LinearAllocatorT<float> linearAllocator;
    float* A = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      A[i] = static_cast<float>((i + 0) % 512 + 1);
    }
    float* _ut_A = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_A(_ut_A, {32, 32}, {0, 0}, {32, 32});
    init::A::view::create(A).copyToView(_view__ut_A);

    float* B = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      B[i] = static_cast<float>((i + 1) % 512 + 1);
    }
    float* _ut_B = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_B(_ut_B, {32, 32}, {0, 0}, {32, 32});
    init::B::view::create(B).copyToView(_view__ut_B);

    float* C = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      C[i] = static_cast<float>((i + 2) % 512 + 1);
    }
    float* _ut_C = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_C(_ut_C, {32, 32}, {0, 0}, {32, 32});
    init::C::view::create(C).copyToView(_view__ut_C);

    kernel::matmulATB krnl;
    krnl.numElements = numElements;
    krnl.streamPtr = nullptr;
    const float* tempA = A;
    krnl.A = &tempA;
    const float* tempB = B;
    krnl.B = &tempB;
    krnl.C = C;
    krnl.execute();

    float *_tmp0;
    float* _buffer0 = linearAllocator.allocate(numElements * 1024);
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 1024 * sizeof(float));
    for (int _k = 0; _k < 32; ++_k) {
      for (int _j = 0; _j < 32; ++_j) {
        for (int _i = 0; _i < 32; ++_i) {
          _tmp0[1*_i + 32*_j] += _ut_A[1*_k + 32*_i] * _ut_B[1*_k + 32*_j];
        }
      }
    }
    for (int _b = 0; _b < 32; ++_b) {
      for (int _a = 0; _a < 32; ++_a) {
        _ut_C[1*_a + 32*_b] = _tmp0[1*_a + 32*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 32; ++_b) {
        for (int _a = 0; _a < 32; ++_a) {
          double ref = _ut_C[1*_a + 32*_b];
          double diff = ref - C[1*_a + 32*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 1.19e-05);
    }
    linearAllocator.free();
  }
  void testmatmulABT() {
    int numElements = 1;
    yateto::LinearAllocatorT<float> linearAllocator;
    float* A = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      A[i] = static_cast<float>((i + 0) % 512 + 1);
    }
    float* _ut_A = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_A(_ut_A, {32, 32}, {0, 0}, {32, 32});
    init::A::view::create(A).copyToView(_view__ut_A);

    float* B = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      B[i] = static_cast<float>((i + 1) % 512 + 1);
    }
    float* _ut_B = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_B(_ut_B, {32, 32}, {0, 0}, {32, 32});
    init::B::view::create(B).copyToView(_view__ut_B);

    float* C = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      C[i] = static_cast<float>((i + 2) % 512 + 1);
    }
    float* _ut_C = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_C(_ut_C, {32, 32}, {0, 0}, {32, 32});
    init::C::view::create(C).copyToView(_view__ut_C);

    kernel::matmulABT krnl;
    krnl.numElements = numElements;
    krnl.streamPtr = nullptr;
    const float* tempA = A;
    krnl.A = &tempA;
    const float* tempB = B;
    krnl.B = &tempB;
    krnl.C = C;
    krnl.execute();

    float *_tmp0;
    float* _buffer0 = linearAllocator.allocate(numElements * 1024);
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 1024 * sizeof(float));
    for (int _k = 0; _k < 32; ++_k) {
      for (int _j = 0; _j < 32; ++_j) {
        for (int _i = 0; _i < 32; ++_i) {
          _tmp0[1*_i + 32*_j] += _ut_A[1*_i + 32*_k] * _ut_B[1*_j + 32*_k];
        }
      }
    }
    for (int _b = 0; _b < 32; ++_b) {
      for (int _a = 0; _a < 32; ++_a) {
        _ut_C[1*_a + 32*_b] = _tmp0[1*_a + 32*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 32; ++_b) {
        for (int _a = 0; _a < 32; ++_a) {
          double ref = _ut_C[1*_a + 32*_b];
          double diff = ref - C[1*_a + 32*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 1.19e-05);
    }
    linearAllocator.free();
  }
  void testmatmulATBT() {
    int numElements = 1;
    yateto::LinearAllocatorT<float> linearAllocator;
    float* A = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      A[i] = static_cast<float>((i + 0) % 512 + 1);
    }
    float* _ut_A = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_A(_ut_A, {32, 32}, {0, 0}, {32, 32});
    init::A::view::create(A).copyToView(_view__ut_A);

    float* B = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      B[i] = static_cast<float>((i + 1) % 512 + 1);
    }
    float* _ut_B = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_B(_ut_B, {32, 32}, {0, 0}, {32, 32});
    init::B::view::create(B).copyToView(_view__ut_B);

    float* C = linearAllocator.allocate(numElements * 1024);
    for (int i = 0; i < 1024; ++i) {
      C[i] = static_cast<float>((i + 2) % 512 + 1);
    }
    float* _ut_C = linearAllocator.allocate(numElements * 1024);
    yateto::DenseTensorView<2,float,unsigned> _view__ut_C(_ut_C, {32, 32}, {0, 0}, {32, 32});
    init::C::view::create(C).copyToView(_view__ut_C);

    kernel::matmulATBT krnl;
    krnl.numElements = numElements;
    krnl.streamPtr = nullptr;
    const float* tempA = A;
    krnl.A = &tempA;
    const float* tempB = B;
    krnl.B = &tempB;
    krnl.C = C;
    krnl.execute();

    float *_tmp0;
    float* _buffer0 = linearAllocator.allocate(numElements * 1024);
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 1024 * sizeof(float));
    for (int _k = 0; _k < 32; ++_k) {
      for (int _j = 0; _j < 32; ++_j) {
        for (int _i = 0; _i < 32; ++_i) {
          _tmp0[1*_i + 32*_j] += _ut_A[1*_k + 32*_i] * _ut_B[1*_j + 32*_k];
        }
      }
    }
    for (int _b = 0; _b < 32; ++_b) {
      for (int _a = 0; _a < 32; ++_a) {
        _ut_C[1*_a + 32*_b] = _tmp0[1*_a + 32*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 32; ++_b) {
        for (int _a = 0; _a < 32; ++_a) {
          double ref = _ut_C[1*_a + 32*_b];
          double diff = ref - C[1*_a + 32*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 1.19e-05);
    }
    linearAllocator.free();
  }
};
#endif
