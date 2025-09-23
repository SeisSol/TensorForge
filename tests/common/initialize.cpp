#include "initialize.h"

using namespace initgen::reference;

namespace initgen {
  namespace reference {

    void singleInit(int M, int N, real Alpha, real *A, int Ld) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              A[m + n * Ld] = Alpha;
            }
          }
        }

        real *findData(real *Data, unsigned Stride, unsigned BlockId) {
          return &Data[BlockId * Stride];
        }
      }
    }



