#include "csa.h"

using namespace csagen::reference;

namespace csagen {
  namespace reference {

    void singleCsa(int M, int N, real Alpha, real *A, real Beta, real *B, int Ld) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              B[m + n * Ld] = Alpha * A[m + n * Ld] + Beta * B[m + n * Ld];
            }
          }
        }

        real *findData(real *Data, unsigned Stride, unsigned BlockId) {
          return &Data[BlockId * Stride];
        }
      }
    }



