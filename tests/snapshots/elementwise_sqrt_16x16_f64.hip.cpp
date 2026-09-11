// === base name ===
kernel_6ad41261375845d5

// === header ===
void launcher_kernel_6ad41261375845d5(const double* m0, size_t m0_extraOffset, double* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_6ad41261375845d5(const double* m0, size_t m0_extraOffset, double* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6ad41261375845d5, block.x * block.y * block.z, 0 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6ad41261375845d5, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(double)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6ad41261375845d5), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_6ad41261375845d5, grid, block, 0 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6ad41261375845d5(const double* m0, size_t m0_extraOffset, double* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // B = sqrt(A)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 256 + 0 + m1_extraOffset];
          // glb_m1 = sqrt(glb_m0)
          int32_t v12_lead = threadIdx.x % 64;
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v14_k1 = 0; v14_k1 < 16; ++v14_k1) {
              int32_t v20_a = v14_k1 * 16;
              double v22_data = glb_m0[(v12_lead + v20_a)];
              glb_m1[(v12_lead + v20_a)] = (sqrt(v22_data));
            }
          }
        }
      }
    }
  }
}

