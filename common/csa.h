#ifndef GEMMFORGE_REFERENCE_CSA_H
#define GEMMFORGE_REFERENCE_CSA_H

#include "typedef.h"
#include <iostream>

namespace csagen {
  namespace reference {
    void singleCsa(int M, int N,
                   real Alpha, real *A, int lda,
                   real Beta, real *B, int ldb);

    real *findData(real *Data, unsigned Stride, unsigned BlockId);

    template<typename AT, typename BT>
    void csa( int M, int N,
              real Alpha, AT A, int lda,
              real Beta, BT B, int ldb,
              unsigned OffsetA,
              unsigned OffsetB,
              unsigned NumElements) {

      for (unsigned Index = 0; Index < NumElements; ++Index) {
        real *MatrixA = findData(A, OffsetA, Index);
        real *MatrixB = findData(B, OffsetB, Index);

        singleCsa(M, N, Alpha, MatrixA, lda, Beta, MatrixB, ldb);


      }
    }
  }
}

#endif //GEMMFORGE_REFERENCE_GEMM_H