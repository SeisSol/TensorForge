
#include <iostream>
#include <cuda.h>
#include <cstring>

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

std::string PrevFile = "";
int PrevLine = 0;

void checkErr(const std::string &File, int Line) {
#ifndef NDEBUG
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::cout << std::endl << File
                  << ", line " << Line
                  << ": " << cudaGetErrorString(Error)
                  << " (" << Error << ")"
                  << std::endl;

        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = File;
      PrevLine = Line;
#endif
}

// Dense x Dense Kernel
__global__ void 
__launch_bounds__(64)
 kernel_sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_aacfffd(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 81 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
      float reg0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__  __align__(8) float totalShrMem[81];
      float * localShrMem0 = &totalShrMem[81 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
        shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
        if (threadIdx.x < 17) {
          shrRegion0[threadIdx.x + 64] = glb_B[threadIdx.x + 64];
        }
      }
      __syncthreads();
      if (threadIdx.x < 56) {
        float value;

        for (int k = 0; k < 9; ++k) {
          value = glb_A[threadIdx.x + k * 56];

          #pragma unroll
          for (int n = 0; n < 9; ++n) {
            reg0[n] += value * shrRegion0[k + 9 * n];
          }
        }
      }
      if (threadIdx.x < 56) {
        #pragma unroll
        for (int n = 0; n < 9; ++n) {
          glb_C[threadIdx.x + 56 * n] = reg0[n] + glb_C[threadIdx.x + 56 * n];
        }
      }
    }
  }
}


// Dense x Sparse Kernel
__global__ void 
__launch_bounds__(64)
 kernel_sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_21a8028(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 9 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
      float reg0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__  __align__(8) float totalShrMem[9];
      float * localShrMem0 = &totalShrMem[9 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
        if (threadIdx.x < 9) {
          shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
        }
      }
      __syncthreads();
      if (threadIdx.x < 56) {
        float value;

        value = glb_A[threadIdx.x + 0 * 56];

        value = glb_A[threadIdx.x + 1 * 56];

        // Mul begin col 1
        reg0[0] += value * shrRegion0[0];
        reg0[1] += value * shrRegion0[1];
        reg0[2] += value * shrRegion0[2];
        reg0[3] += value * shrRegion0[3];
        reg0[4] += value * shrRegion0[4];
        reg0[5] += value * shrRegion0[5];
        reg0[6] += value * shrRegion0[6];
        reg0[7] += value * shrRegion0[7];
        reg0[8] += value * shrRegion0[8];
        // Mul end col 8

        value = glb_A[threadIdx.x + 2 * 56];

        value = glb_A[threadIdx.x + 3 * 56];

        value = glb_A[threadIdx.x + 4 * 56];

        value = glb_A[threadIdx.x + 5 * 56];

        value = glb_A[threadIdx.x + 6 * 56];

        value = glb_A[threadIdx.x + 7 * 56];

        value = glb_A[threadIdx.x + 8 * 56];

      }
      if (threadIdx.x < 56) {
        #pragma unroll
        for (int n = 0; n < 9; ++n) {
          glb_C[threadIdx.x + 56 * n] = reg0[n] + glb_C[threadIdx.x + 56 * n];
        }
      }
    }
  }
}


// Dense x Dense Kernel Launcher
void sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_aacfffd(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(64, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_aacfffd<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}


// Dense x Sparse Kernel Launcher
void sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_21a8028(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(64, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_21a8028<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}



int main(){
  // Element Matrices
  std::cout << "Instantiating core matrices" << std::endl;
  float CoreA[56*9] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
  float CoreB_sparse[9] = {5., 5., 5., 5., 5., 5., 5., 5., 5.};
  float CoreB_dense[9 * 9] = {0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0.,
 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0.,
 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0.,
 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0.,
 0., 5., 0., 0., 0., 0., 0., 0., 0.};
  float CoreC[56*9] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  
  // Buffers 
  std::cout << "Instantiating buffer matrices" << std::endl;
  float* A = new float[56*9*479958];
  float* B_dense = new float[9*9*479958];
  float* B_sparse = new float[9*479958];
  float* C = new float[56*9*479958];
  float* R1 = new float[56*9*479958];
  float* R2 = new float[56*9*479958];

  // Copy the Element Matrices N times into Element Buffers
  std::cout << "Copying core matrices to buffers" << std::endl;
  for (int i = 0; i < 479958; i++){
    std::memcpy(A + 56 * 9 * i, CoreA, 56 * 9);
    std::memcpy(B_dense + 9 * 9 * i, CoreB_dense, 9 * 9);
    std::memcpy(B_sparse + 9 * i, CoreB_sparse, 9);
    std::memcpy(C + 56 * 9 * i, CoreC, 56 * 9);
  }

  float *A_dev = nullptr;
  float *B_sparse_dev = nullptr;
  float *B_dense_dev = nullptr;
  float *C1_dev = nullptr;
  float *C2_dev = nullptr;

  std::cout << "Allocating device memory" << std::endl;
  cudaMalloc((void **)&A_dev, sizeof(float) * 56 * 9 * 479958); CHECK_ERR;
  cudaMalloc((void **)&B_sparse_dev, sizeof(float) * 9 * 479958); CHECK_ERR;
  cudaMalloc((void **)&B_dense_dev, sizeof(float) * 9 * 9 * 479958); CHECK_ERR;
  cudaMalloc((void **)&C1_dev, sizeof(float) * 56 * 9 * 479958); CHECK_ERR;
  cudaMalloc((void **)&C2_dev, sizeof(float) * 56 * 9 * 479958); CHECK_ERR;

  std::cout << "Copying buffers to device" << std::endl;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 56 * 9 * 479958, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  9 * 479958, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  9 * 9 * 479958, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * 56 * 9 * 479958, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 56 * 9 * 479958, cudaMemcpyHostToDevice); CHECK_ERR;

  // Dense x Dense Matrix Mult
  std::cout << "Calling Dense x Dense kernel" << std::endl;
  float elapsedTime = 0.0; 
  cudaEvent_t startDD, stopDD;
  cudaEventCreate(&startDD);
  cudaEventCreate(&stopDD);
  cudaEventRecord(startDD);
   sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_aacfffd(A_dev, 0, B_dense_dev, 0, C1_dev, 0, 479958, nullptr, nullptr);
  cudaEventRecord(stopDD);
  cudaEventSynchronize(stopDD);
  cudaEventElapsedTime(&elapsedTime, startDD, stopDD);
  std::cout << "Dense x Dense kernel took " << elapsedTime << " ms" << std::endl; 
  cudaDeviceSynchronize();
  cudaMemcpy(R1, C1_dev, sizeof(float)*56 * 9 * 479958, cudaMemcpyDeviceToHost);

  // Dense x Sparse Matrix Mult
  std::cout << "Calling Dense x Sparse kernel" << std::endl;
  elapsedTime = 0.0;
  cudaEvent_t startDS, stopDS;
  cudaEventCreate(&startDS);
  cudaEventCreate(&stopDS);
  cudaEventRecord(startDS);
   sgemm_NT_NT_m56_n9_k9_lda56_ldb9_ldc56_alpha_1_beta_1_sss_21a8028(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, 479958, nullptr, nullptr);
  cudaEventRecord(stopDS);
  cudaEventSynchronize(stopDS);
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
  std::cout << "Dense x Sparse kernel took " << elapsedTime << " ms" << std::endl; 
  cudaDeviceSynchronize();
  cudaMemcpy(R2, C2_dev, sizeof(float)*56 * 9 * 479958, cudaMemcpyDeviceToHost);

  std::cout << "Freeing device memory" << std::endl;
  cudaFree(A_dev);
  cudaFree(B_sparse_dev);
  cudaFree(B_dense_dev);
  cudaFree(C1_dev);
  cudaFree(C2_dev);

  std::cout << "[";
  for (int ii = 0; ii < 56*9 -1; ii++){
    std::cout << R1[ii] << ", ";
  }
  std::cout << R1[56*9 -1] << "]" << std::endl;
  std::cout << "[";
  for (int ii = 0; ii < 56*9 - 1; ii++){
    std::cout << R2[ii] << ", ";
  }
  std::cout << R2[56*9 -1] << "]" << std::endl;
  for (int el = 0; el < 479958; el++) {
    for (int i = 0; i < 56; i++){
        for (int j = 0; j < 9; j++) {
        if (std::abs(R1[i*9 + j] - R2[i*9 + j]) > 0.001) {
            throw std::runtime_error(" Dense x  Dense and  Dense x  Sparse Matrix Mismatch in Multiplication at (" + std::to_string(i) +"," + std::to_string(j) + ")\n" + 
                std::to_string(R1[i*9 + j]) + " != " + std::to_string(R2[i*9 + j]));
        }
        }
    }
  }
  std::cout << "Results Match!" << std::endl;
}
