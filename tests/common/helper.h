#ifndef TENSORFORGE_BENCHMARK_AUX_H
#define TENSORFORGE_BENCHMARK_AUX_H

#include "typedef.h"
#include <stddef.h>
#include <vector>

namespace cf {
namespace aux {
long long computeNumFlops(int m, int n, int k, real alpha, real beta);
std::vector<real *> shuffleMatrices(real *matrices, int size, int numElements);
void initMatrix(real *matrix, int size, size_t numElements);
real getRandomNumber();
bool compare(real *host, const real *device, unsigned size, size_t numElements,
             real eps);
} // namespace aux
} // namespace cf

#endif // TENSORFORGE_BENCHMARK_AUX_H
