#include "Eigen/Eigen"
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
    {
      using Eigen::Matrix;
      using Eigen::Map;
      using Eigen::Stride;
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapA(const_cast<float*>(A));
      Map<Matrix<float,32,32>,Eigen::Unaligned,Stride<32,1>> _mapB(const_cast<float*>(B));
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapC(C);
      _mapC = _mapA*_mapB;
    }
        
  }
  constexpr unsigned long const kernel::matmulATB::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATB::HardwareFlops;
  void kernel::matmulATB::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    {
      using Eigen::Matrix;
      using Eigen::Map;
      using Eigen::Stride;
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapA(const_cast<float*>(A));
      Map<Matrix<float,32,32>,Eigen::Unaligned,Stride<32,1>> _mapB(const_cast<float*>(B));
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapC(C);
      _mapC = _mapA.transpose()*_mapB;
    }
        
  }
  constexpr unsigned long const kernel::matmulABT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulABT::HardwareFlops;
  void kernel::matmulABT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    {
      using Eigen::Matrix;
      using Eigen::Map;
      using Eigen::Stride;
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapA(const_cast<float*>(A));
      Map<Matrix<float,32,32>,Eigen::Unaligned,Stride<32,1>> _mapB(const_cast<float*>(B));
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapC(C);
      _mapC = _mapA*_mapB.transpose();
    }
        
  }
  constexpr unsigned long const kernel::matmulATBT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATBT::HardwareFlops;
  void kernel::matmulATBT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    float *_tmp0;
    alignas(32) float _buffer0[1024] ;
    _tmp0 = _buffer0;
    {
      using Eigen::Matrix;
      using Eigen::Map;
      using Eigen::Stride;
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapA(const_cast<float*>(B));
      Map<Matrix<float,32,32>,Eigen::Unaligned,Stride<32,1>> _mapB(const_cast<float*>(A));
      Map<Matrix<float,32,32>,Eigen::Aligned32,Stride<32,1>> _mapC(_tmp0);
      _mapC = _mapA*_mapB;
    }
        
    for (int _j = 0; _j < 32; ++_j) {
      #pragma omp simd
      for (int _i = 0; _i < 32; ++_i) {
        C[1*_i + 32*_j] = _tmp0[1*_j + 32*_i];
      }
    }
  }
} // namespace yateto
