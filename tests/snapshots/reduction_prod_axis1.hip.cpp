// === base name ===
kernel_06611af794

// === header ===
void launcher_kernel_06611af794(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_06611af794(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_06611af794, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_06611af794), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_06611af794, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_06611af794(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16(16) {0..16} strided
    // OUT = *(A, dims=[1])
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 16 + 0 + m1_extraOffset];
          // glb_m1 = *(glb_m0, dims=[1])
          int32_t v4_lead = threadIdx.x % 32;
          if (v4_lead < 16) {
            float v7_acc0 = 1.0f;
            #pragma unroll
            for (int32_t v6_r1 = 0; v6_r1 < 16; ++v6_r1) {
              int32_t v13_a = v6_r1 * 16;
              int32_t v14_a = v4_lead + v13_a;
              float v22_data = glb_m0[(v4_lead + v13_a)];
              v7_acc0 = (v7_acc0 * v22_data);
            }
            int32_t v29_lead = v4_lead + 0_i32;
            glb_m1[v29_lead] = v7_acc0;
          }
        }
      }
    }
  }
}

