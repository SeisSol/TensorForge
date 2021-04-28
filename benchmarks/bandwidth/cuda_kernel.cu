#include "kernel.h"

__global__ void kernel_copyData(float *To, float *From, size_t NumElements) {
    int Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (Idx < NumElements) {
        To[Idx] = From[Idx];
    }
}

void copyData(float *To, float *From, size_t NumElements, size_t blocks, size_t threads, void *stream) {
  kernel_copyData<<<blocks, threads>>>(To, From, NumElements);
}
