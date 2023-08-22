#ifndef GEMMFORGE_REFERENCE_GEMM_H
#define GEMMFORGE_REFERENCE_GEMM_H

#include "typedef.h"
#include <iostream>

#define GEMMFORGE 1
#define OPENBLAS 2

#define CPU_BACKEND CONCRETE_CPU_BACKEND

#if CPU_BACKEND == OPENBLAS
#include <cblas.h>
#endif

namespace gemmforge {
  namespace reference {
    enum class LayoutType {
      Trans, NoTrans
    };

    void singleGemm(LayoutType TypeA,
                    LayoutType TypeB,
                    int M, int N, int K,
                    real Alpha, real *A, int Lda,
                    real *B, int Ldb,
                    real Beta, real *C, int Ldc);


    real *findData(real *Data, unsigned Stride, unsigned BlockId);
    real *findData(real **Data, unsigned Stride, unsigned BlockId);

    template<typename AT, typename BT, typename CT>
    void gemm(LayoutType TypeA,
              LayoutType TypeB,
              int M, int N, int K,
              real Alpha, AT A, int Lda,
              BT B, int Ldb,
              real Beta, CT C, int Ldc,
              unsigned OffsetA,
              unsigned OffsetB,
              unsigned OffsetC,
              unsigned NumElements) {

      for (unsigned Index = 0; Index < NumElements; ++Index) {
        real *MatrixA = findData(A, OffsetA, Index);
        real *MatrixB = findData(B, OffsetB, Index);
        real *MatrixC = findData(C, OffsetC, Index);

#if CPU_BACKEND == GEMMFORGE
        singleGemm(TypeA, TypeB,
                   M, N, K,
                   Alpha, MatrixA, Lda,
                   MatrixB, Ldb,
                   Beta, MatrixC, Ldc);

#elif CPU_BACKEND == OPENBLAS
  CBLAS_LAYOUT Layout = CblasColMajor;
  CBLAS_TRANSPOSE TransA = TypeA == LayoutType::Trans ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = TypeB == LayoutType::Trans ? CblasTrans : CblasNoTrans;

#if REAL_SIZE == 4
  cblas_sgemm(Layout, TransA, TransB,
              M, N, K,
              Alpha, MatrixA, Lda,
              MatrixB, Ldb,
              Beta, MatrixC, Ldc);
#elif REAL_SIZE == 8
  cblas_dgemm(Layout, TransA, TransB,
              M, N, K,
              Alpha, MatrixA, Lda,
              MatrixB, Ldb,
              Beta, MatrixC, Ldc);
#endif

#else
#error "Chosen reference CPU-GEMM impl. is not supported"
#endif
      }
    }
  }
}

#endif //GEMMFORGE_REFERENCE_GEMM_H