#ifndef GEMMGEN_AUX_H
#define GEMMGEN_AUX_H

#include "typedef.h"
#include <vector>

long long computeNumFlops(int M, int N, int K, real Alpha, real Beta);
std::vector<real*> shuffleMatrices(real* Matrices, int Size, int NumElements);

real getRandomNumber();


#endif //GEMMGEN_AUX_H
