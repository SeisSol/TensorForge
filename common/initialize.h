#ifndef GEMMFORGE_REFERENCE_INITIALIZE_H
#define GEMMFORGE_REFERENCE_INITIALIZE_H

#include "typedef.h"
#include <iostream>

namespace initgen {
  namespace reference {
    real *findData(real *Data, unsigned Stride, unsigned BlockId);
    void singleInit(int M, int N, real Alpha, real *A, int Ld);

    template<typename AT>
    void initialize( int M, int N,
              real Alpha, AT A,
              int Ld,
              unsigned Offset,
              unsigned NumElements) {

      for (unsigned Index = 0; Index < NumElements; ++Index) {
        real *MatrixA = findData(A, Offset, Index);
        singleInit(M, N, Alpha, MatrixA, Ld);
      }
    }
  }
}

#endif //GEMMFORGE_REFERENCE_GEMM_H