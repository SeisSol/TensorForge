// === base name ===
kernel_394cd56b0a266967

// === header ===
void launcher_kernel_394cd56b0a266967(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_394cd56b0a266967(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_394cd56b0a266967, block.x * block.y * block.z, 256 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (std::min(gridsize, numElements0), 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_394cd56b0a266967, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_394cd56b0a266967<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_394cd56b0a266967(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // TMP = abs(A)
    // C = neg(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v17_lead = threadIdx.x % 32;
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v19_k1 = 0; v19_k1 < 8; ++v19_k1) {
              float v27_data = glb_m0[(v17_lead + (v19_k1 * 8))];
              r0[v19_k1] = (fabsf(v27_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r0);
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
              float v36_data = r0[v34_i1];
              int32_t v43_a = v17_lead + (v34_i1 * 8);
              s0[(v43_a ^ ((v43_a >> 5) & 31))] = v36_data;
            }
          }
          __syncwarp();
          // glb_m1 = neg(s0)
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v51_k1 = 0; v51_k1 < 8; ++v51_k1) {
              int32_t v57_a = v51_k1 * 8;
              int32_t v58_a = v17_lead + v57_a;
              float v62_data = s0[(v58_a ^ ((v58_a >> 5) & 31))];
              glb_m1[(v17_lead + v57_a)] = ((-v62_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

