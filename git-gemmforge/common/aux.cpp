#include "aux.h"
#include <algorithm>
#include <stdio.h>
#include <cstdlib>
#include <math.h>

long long computeNumFlops(int M, int N, int K, real Alpha, real Beta) {
  long long Flops = (K + (K - 1)) * M * N;

  if (Alpha != 1.0) {
    Flops += (M * N);
  }

  if (Beta != 0.0) {
    Flops += (M * N);
  }

  return Flops;
}


std::vector<real*> shuffleMatrices(real* Matrices, int Size, int NumElements) {
  std::vector<real*> Ptrs(NumElements, nullptr);
  for (int Index = 0; Index < NumElements; ++Index) {
    Ptrs.push_back(&Matrices[Index * Size]);
  }
  std::random_shuffle(Ptrs.begin(), Ptrs.end());
  return Ptrs;
}

real getRandomNumber() {
  return static_cast<real>(std::rand()) / RAND_MAX;
}