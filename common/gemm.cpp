#include "gemm.h"

using namespace gemmforge::reference;

namespace gemmforge {
  namespace reference {

    void singleGemm(LayoutType TypeA,
                    LayoutType TypeB,
                    int M, int N, int K,
                    real Alpha, real *A, int Lda,
                    real *B, int Ldb,
                    real Beta, real *C, int Ldc) {

      int NumRowA{}, NumColA{}, NumRowB{};

      if (TypeA == LayoutType::NoTrans) {
        NumRowA = M;
        NumColA = K;
      } else {
        NumRowA = K;
        NumColA = M;
      }

      if (TypeB == LayoutType::NoTrans) {
        NumRowB = K;
      } else {
        NumRowB = N;
      }

      if (Alpha == 0.0) {
        for (int j = 0; j < N; ++j) {
          for (int i = 0; i < M; ++i) {
            C[i + j * Ldc] = Beta * C[i + j * Ldc];
          }
        }
        return;
      }

      if (TypeB == LayoutType::NoTrans) {
        if (TypeA == LayoutType::NoTrans) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[m + k * Lda] * B[k + n * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        } else {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[k + m * Lda] * B[k + n * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        }
      } else {
        if (TypeA == LayoutType::NoTrans) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[m + k * Lda] * B[n + k * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        } else {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[k + m * Lda] * B[n + k * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        }
      }
    }


    real *findData(real *Data, unsigned Stride, unsigned BlockId) {
      return &Data[BlockId * Stride];
    }

    real *findData(real **Data, unsigned Stride, unsigned BlockId) {
      return &(Data[BlockId][Stride]);
    }

  }
}