#include "csa.h"

using namespace csagen::reference;

namespace csagen {
  namespace reference {

    void singleCsa(int M, int N, real Alpha, real *A, int lda, real Beta, real *B, int ldb) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              B[m + n * ldb] = Alpha * A[m + n * lda] + Beta * B[m + n * ldb];
            }
          }
        }

        real *findData(real *Data, unsigned Stride, unsigned BlockId) {
          return &Data[BlockId * Stride];
        }
      }
    }



