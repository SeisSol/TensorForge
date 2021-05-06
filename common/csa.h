#ifndef GEMMFORGE_REFERENCE_CSA_H
#define GEMMFORGE_REFERENCE_CSA_H

#include "typedef.h"
#include <iostream>

namespace csagen {
  namespace reference {
    void singleCsa(int M, int N,
                    real Alpha, real *A,
                    real Beta, real *B, int Ld);

    real *findData(real *Data, unsigned Stride, unsigned BlockId);

    template<typename AT, typename BT>
    void csa( int M, int N,
              real Alpha, AT A,
              real Beta, BT B,
              int Ld,
              unsigned Offset,
              unsigned NumElements) {

      for (unsigned Index = 0; Index < NumElements; ++Index) {
        real *MatrixA = findData(A, Offset, Index);
        real *MatrixB = findData(B, Offset, Index);

        singleCsa(M, N, Alpha, MatrixA, Beta, MatrixB, Ld);


      }
    }
  }
}

#endif //GEMMFORGE_REFERENCE_GEMM_H