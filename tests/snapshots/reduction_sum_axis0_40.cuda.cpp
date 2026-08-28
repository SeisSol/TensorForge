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
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_bb2d33c08e, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_bb2d33c08e, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_bb2d33c08e<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
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
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 120 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 3 + 0 + m1_extraOffset];
          // glb_m1 = +(glb_m0, dims=[0])
          int32_t v5_lead = threadIdx.x % 32;
          bool v21_own = v5_lead < 8;
          bool v41_w = v5_lead == 0;
          #pragma unroll
          for (int32_t v3_k1 = 0; v3_k1 < 3; ++v3_k1) {
            int32_t v11_a = v3_k1 * 40;
            int32_t v12_a = v5_lead + v11_a;
            float v20_data = glb_m0[(v5_lead + v11_a)];
            float v38_sel0;
            if (v21_own) {
              int32_t v28_a = (v5_lead + 32_i32) + v11_a;
              float v36_data = glb_m0[((v5_lead + 32_i32) + v11_a)];
              v38_sel0 = v36_data;
            }
            else {
              v38_sel0 = 0.0f;
            }
            float v40_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>((v20_data + v38_sel0));
            if (v41_w) {
              glb_m1[v3_k1] = v40_red;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

