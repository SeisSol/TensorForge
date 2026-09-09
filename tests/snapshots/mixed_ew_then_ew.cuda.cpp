// === base name ===
kernel_17ba568fcb2e1a17

// === header ===
void launcher_kernel_17ba568fcb2e1a17(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_17ba568fcb2e1a17(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_17ba568fcb2e1a17, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_17ba568fcb2e1a17, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_17ba568fcb2e1a17<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_17ba568fcb2e1a17(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v13_lead = threadIdx.x % 32;
          if (v13_lead < 8) {
            #pragma unroll
            for (int32_t v15_k1 = 0; v15_k1 < 8; ++v15_k1) {
              float v23_data = glb_m0[(v13_lead + (v15_k1 * 8))];
              r0[v15_k1] = (fabsf(v23_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r0);
          if (v13_lead < 8) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
              float v32_data = r0[v30_i1];
              int32_t v39_a = v13_lead + (v30_i1 * 8);
              s0[(v39_a ^ ((v39_a >> 5) & 31))] = v32_data;
            }
          }
          __syncwarp();
          // glb_m1 = neg(s0)
          if (v13_lead < 8) {
            #pragma unroll
            for (int32_t v47_k1 = 0; v47_k1 < 8; ++v47_k1) {
              int32_t v53_a = v47_k1 * 8;
              int32_t v54_a = v13_lead + v53_a;
              float v58_data = s0[(v54_a ^ ((v54_a >> 5) & 31))];
              glb_m1[(v13_lead + v53_a)] = ((-v58_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

