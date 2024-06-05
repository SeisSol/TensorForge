#include "tensorforge_aux.h"
__global__ void 
__launch_bounds__(64)
 kernel_kernel_15b8c0b6fd(const float** m0, unsigned m0_extraOffset, float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags ) {
  // meta data:
  // m0 32×32(32×32) {0..32}×{0..32} pointer_based
  // m1 32×32(32×32) {0..32}×{0..32} pointer_based
  // m2 32×32(32×32) {0..32}×{0..32} pointer_based
  // <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae641bb0>[0, 1] = <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae641a60>[0, -1]×<tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae641af0>[-1, 1]
  unsigned batchId = threadIdx.y + blockDim.y * blockIdx.x;
  if (batchId < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchId]) : true;
    if (allowed) {
      const float * const __restrict__ glb_m0 = &m0[batchId][0 + m0_extraOffset];
      float * const __restrict__ glb_m1 = &m1[batchId][0 + m1_extraOffset];
      const float * const __restrict__ glb_m2 = &m2[batchId][0 + m2_extraOffset];
      float reg0[1] = {0.0f};
      __shared__  __align__(8) float totalShrMem[2048];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];
      float* __restrict__ s0 = &localShrMem0[0];
      {
        // s0 = load{g>s}(glb_m0[0, 1])
        #pragma unroll
        for (int i = 0; i < 32; i += 1) {
          s0[0 + 0 + 1 * threadIdx.x + i * 32] = __ldcg(&glb_m0[0 + 0 + 1 * threadIdx.x + i * 32]);
        }
      }
      float r0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __syncwarp();
      {
        // r0 = +(glb_m2 * s0) + None
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int k0 = 0; k0 < 32; ++k0) {
            #pragma unroll
            for (int n1 = 0; n1 < 32; ++n1) {
              float data0;
              data0 = glb_m2[(threadIdx.x - 0) * 1 + (k0 - 0) * 32];
              float prod0 = data0;
              float data1 = s0[(k0 - 0) * 1 + (n1 - 0) * 32];
              float prod1 = (prod0 * data1);
              float value = r0[n1 * 1];
              float newvalue = (value + prod1);
              r0[n1 * 1] = newvalue;
            }
          }
        }
        if (threadIdx.x < 32) {
        }
      }
      // glb_m1 = store{r>g}(r0);
      {
        #pragma unroll
        for (int i1 = 0; i1 < 32; ++i1) {
          float value = r0[i1 * 1];
          glb_m1[(threadIdx.x - 0) * 1 + (i1 - 0) * 32] = value;
        }
      }
    }
  }
}
void launcher_kernel_15b8c0b6fd(const float** m0, unsigned m0_extraOffset, float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags , void* streamPtr) {
  dim3 block (32, 2, 1);
  dim3 grid ((numElements + 2 - 1) / 2, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_15b8c0b6fd<<<grid,block,0,stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements,  flags );
  CHECK_ERR;
}
__global__ void 
__launch_bounds__(64)
 kernel_kernel_b0502b5c17(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags ) {
  // meta data:
  // m0 32×32(32×32) {0..32}×{0..32} pointer_based
  // m1 32×32(32×32) {0..32}×{0..32} pointer_based
  // m2 32×32(32×32) {0..32}×{0..32} pointer_based
  // <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae641760>[0, 1] = <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae641fa0>[-1, 0]×<tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ab040>[-1, 1]
  unsigned batchId = threadIdx.y + blockDim.y * blockIdx.x;
  if (batchId < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchId]) : true;
    if (allowed) {
      const float * const __restrict__ glb_m0 = &m0[batchId][0 + m0_extraOffset];
      const float * const __restrict__ glb_m1 = &m1[batchId][0 + m1_extraOffset];
      float * const __restrict__ glb_m2 = &m2[batchId][0 + m2_extraOffset];
      float reg0[1] = {0.0f};
      __shared__  __align__(8) float totalShrMem[2048];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];
      float* __restrict__ s0 = &localShrMem0[0];
      {
        // s0 = load{g>s}(glb_m0[0, 1])
        #pragma unroll
        for (int i = 0; i < 32; i += 1) {
          s0[0 + 0 + 1 * threadIdx.x + i * 32] = __ldcg(&glb_m0[0 + 0 + 1 * threadIdx.x + i * 32]);
        }
      }
      float r0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __syncwarp();
      {
        // r0 = +(glb_m1 * s0) + None
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int k0 = 0; k0 < 32; ++k0) {
            #pragma unroll
            for (int n1 = 0; n1 < 32; ++n1) {
              float data0;
              data0 = glb_m1[(k0 - 0) * 1 + (threadIdx.x - 0) * 32];
              float prod0 = data0;
              float data1 = s0[(k0 - 0) * 1 + (n1 - 0) * 32];
              float prod1 = (prod0 * data1);
              float value = r0[n1 * 1];
              float newvalue = (value + prod1);
              r0[n1 * 1] = newvalue;
            }
          }
        }
        if (threadIdx.x < 32) {
        }
      }
      // glb_m2 = store{r>g}(r0);
      {
        #pragma unroll
        for (int i1 = 0; i1 < 32; ++i1) {
          float value = r0[i1 * 1];
          glb_m2[(threadIdx.x - 0) * 1 + (i1 - 0) * 32] = value;
        }
      }
    }
  }
}
void launcher_kernel_b0502b5c17(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags , void* streamPtr) {
  dim3 block (32, 2, 1);
  dim3 grid ((numElements + 2 - 1) / 2, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_b0502b5c17<<<grid,block,0,stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements,  flags );
  CHECK_ERR;
}
__global__ void 
__launch_bounds__(64)
 kernel_kernel_06c3e4ee5d(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags ) {
  // meta data:
  // m0 32×32(32×32) {0..32}×{0..32} pointer_based
  // m1 32×32(32×32) {0..32}×{0..32} pointer_based
  // m2 32×32(32×32) {0..32}×{0..32} pointer_based
  // <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5c1370>[0, 1] = <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5c15b0>[0, -1]×<tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5c15e0>[1, -1]
  unsigned batchId = threadIdx.y + blockDim.y * blockIdx.x;
  if (batchId < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchId]) : true;
    if (allowed) {
      float * const __restrict__ glb_m0 = &m0[batchId][0 + m0_extraOffset];
      const float * const __restrict__ glb_m1 = &m1[batchId][0 + m1_extraOffset];
      const float * const __restrict__ glb_m2 = &m2[batchId][0 + m2_extraOffset];
      float reg0[1] = {0.0f};
      __shared__  __align__(8) float totalShrMem[2048];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];
      float* __restrict__ s0 = &localShrMem0[0];
      {
        // s0 = load{g>s}(glb_m1[0, 1])
        #pragma unroll
        for (int i = 0; i < 32; i += 1) {
          s0[0 + 0 + 1 * threadIdx.x + i * 32] = __ldcg(&glb_m1[0 + 0 + 1 * threadIdx.x + i * 32]);
        }
      }
      float r0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __syncwarp();
      {
        // r0 = +(glb_m2 * s0) + None
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int k0 = 0; k0 < 32; ++k0) {
            #pragma unroll
            for (int n1 = 0; n1 < 32; ++n1) {
              float data0;
              data0 = glb_m2[(threadIdx.x - 0) * 1 + (k0 - 0) * 32];
              float prod0 = data0;
              float data1 = s0[(n1 - 0) * 1 + (k0 - 0) * 32];
              float prod1 = (prod0 * data1);
              float value = r0[n1 * 1];
              float newvalue = (value + prod1);
              r0[n1 * 1] = newvalue;
            }
          }
        }
        if (threadIdx.x < 32) {
        }
      }
      // glb_m0 = store{r>g}(r0);
      {
        #pragma unroll
        for (int i1 = 0; i1 < 32; ++i1) {
          float value = r0[i1 * 1];
          glb_m0[(threadIdx.x - 0) * 1 + (i1 - 0) * 32] = value;
        }
      }
    }
  }
}
void launcher_kernel_06c3e4ee5d(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags , void* streamPtr) {
  dim3 block (32, 2, 1);
  dim3 grid ((numElements + 2 - 1) / 2, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_06c3e4ee5d<<<grid,block,0,stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements,  flags );
  CHECK_ERR;
}
__global__ void 
__launch_bounds__(64)
 kernel_kernel_0ac7dd0fbd(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags ) {
  // meta data:
  // m0 32×32(32×32) {0..32}×{0..32} pointer_based
  // m1 32×32(32×32) {0..32}×{0..32} pointer_based
  // m2 32×32(32×32) {0..32}×{0..32} pointer_based
  // <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ba490>[0, 1] = <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ba5b0>[0, -1]×<tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ba220>[-1, 1]
  // <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ba370>[0, 1] = <tensorforge.common.matrix.tensor.SubTensor object at 0x7f0dae5ba7c0>[1, 0]
  unsigned batchId = threadIdx.y + blockDim.y * blockIdx.x;
  if (batchId < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchId]) : true;
    if (allowed) {
      float * const __restrict__ glb_m0 = &m0[batchId][0 + m0_extraOffset];
      const float * const __restrict__ glb_m1 = &m1[batchId][0 + m1_extraOffset];
      const float * const __restrict__ glb_m2 = &m2[batchId][0 + m2_extraOffset];
      float reg0[1] = {0.0f};
      __shared__  __align__(8) float totalShrMem[2048];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];
      float* __restrict__ s0 = &localShrMem0[0];
      {
        // s0 = load{g>s}(glb_m2[0, 1])
        #pragma unroll
        for (int i = 0; i < 32; i += 1) {
          s0[0 + 0 + 1 * threadIdx.x + i * 32] = __ldcg(&glb_m2[0 + 0 + 1 * threadIdx.x + i * 32]);
        }
      }
      float r0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __syncwarp();
      {
        // r0 = +(glb_m1 * s0) + None
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int k0 = 0; k0 < 32; ++k0) {
            #pragma unroll
            for (int n1 = 0; n1 < 32; ++n1) {
              float data0;
              data0 = glb_m1[(threadIdx.x - 0) * 1 + (k0 - 0) * 32];
              float prod0 = data0;
              float data1 = s0[(k0 - 0) * 1 + (n1 - 0) * 32];
              float prod1 = (prod0 * data1);
              float value = r0[n1 * 1];
              float newvalue = (value + prod1);
              r0[n1 * 1] = newvalue;
            }
          }
        }
        if (threadIdx.x < 32) {
        }
      }
      __syncwarp();
      float* __restrict__ s1 = &localShrMem0[0];
      {
        // s1 = store{r>s}(localShrMem0, r0);
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int i1 = 0; i1 < 32; ++i1) {
            float value = r0[i1 * 1];
            s1[(threadIdx.x - 0) * 1 + (i1 - 0) * 32] = value;
          }
        }
      }
      float r1[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __syncwarp();
      {
        // r1 = +(s1) + None
        if (threadIdx.x < 32) {
          #pragma unroll
          for (int n1 = 0; n1 < 32; ++n1) {
            float data0 = s1[(n1 - 0) * 1 + (threadIdx.x - 0) * 32];
            float prod0 = data0;
            float value = r1[n1 * 1];
            float newvalue = (value + prod0);
            r1[n1 * 1] = newvalue;
          }
        }
        if (threadIdx.x < 32) {
        }
      }
      // glb_m0 = store{r>g}(r1);
      {
        #pragma unroll
        for (int i1 = 0; i1 < 32; ++i1) {
          float value = r1[i1 * 1];
          glb_m0[(threadIdx.x - 0) * 1 + (i1 - 0) * 32] = value;
        }
      }
    }
  }
}
void launcher_kernel_0ac7dd0fbd(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags , void* streamPtr) {
  dim3 block (32, 2, 1);
  dim3 grid ((numElements + 2 - 1) / 2, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0ac7dd0fbd<<<grid,block,0,stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements,  flags );
  CHECK_ERR;
}
