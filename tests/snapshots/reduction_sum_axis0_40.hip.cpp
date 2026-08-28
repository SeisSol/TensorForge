// === base name ===
kernel_bb2d33c08e

// === header ===
void launcher_kernel_bb2d33c08e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_bb2d33c08e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_bb2d33c08e, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_bb2d33c08e), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_bb2d33c08e, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_bb2d33c08e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 40×3(40×3) {0..40}×{0..3} strided
    // m1 3(3) {0..3} strided
    // OUT = +(A, dims=[0])
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 120 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 3 + 0 + m1_extraOffset];
          // glb_m1 = +(glb_m0, dims=[0])
          int32_t v8_lead = threadIdx.x % 32;
          bool v24_own = v8_lead < 8;
          bool v44_w = v8_lead == 0;
          #pragma unroll
          for (int32_t v6_k1 = 0; v6_k1 < 3; ++v6_k1) {
            int32_t v14_a = v6_k1 * 40;
            int32_t v15_a = v8_lead + v14_a;
            float v23_data = glb_m0[(v8_lead + v14_a)];
            float v41_sel0;
            if (v24_own) {
              int32_t v31_a = (v8_lead + 32_i32) + v14_a;
              float v39_data = glb_m0[((v8_lead + 32_i32) + v14_a)];
              v41_sel0 = v39_data;
            }
            else {
              v41_sel0 = 0.0f;
            }
            float v43_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>((v23_data + v41_sel0));
            if (v44_w) {
              glb_m1[v6_k1] = v43_red;
            }
          }
        }
      }
    }
  }
}

