// === base name ===
kernel_9b7cf770a2d9cf8b

// === header ===
void launcher_kernel_9b7cf770a2d9cf8b(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9b7cf770a2d9cf8b(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9b7cf770a2d9cf8b, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_9b7cf770a2d9cf8b, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9b7cf770a2d9cf8b), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_9b7cf770a2d9cf8b, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9b7cf770a2d9cf8b(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16(16) {0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t v3_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v3_batchId0 < numElements0; v3_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v4_ahead1 = v3_batchId0 + (gridDim.x * blockDim.y);
        size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v3_batchId0 * 16 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 16);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v18_i1 * 16))]);
              r0[(v17_i0 + v18_i1)] = v26_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float v32_data = r0[0];
          float v33_data = r1[0];
          r1[0] = (v33_data + v32_data);
          float v38_data = r0[1];
          float v39_data = r1[0];
          r1[0] = (v39_data + v38_data);
          float v44_data = r0[2];
          float v45_data = r1[0];
          r1[0] = (v45_data + v44_data);
          float v50_data = r0[3];
          float v51_data = r1[0];
          r1[0] = (v51_data + v50_data);
          float v56_data = r0[4];
          float v57_data = r1[0];
          r1[0] = (v57_data + v56_data);
          float v62_data = r0[5];
          float v63_data = r1[0];
          r1[0] = (v63_data + v62_data);
          float v68_data = r0[6];
          float v69_data = r1[0];
          r1[0] = (v69_data + v68_data);
          float v74_data = r0[7];
          float v75_data = r1[0];
          r1[0] = (v75_data + v74_data);
          float v80_data = r0[8];
          float v81_data = r1[0];
          r1[0] = (v81_data + v80_data);
          float v86_data = r0[9];
          float v87_data = r1[0];
          r1[0] = (v87_data + v86_data);
          float v92_data = r0[10];
          float v93_data = r1[0];
          r1[0] = (v93_data + v92_data);
          float v98_data = r0[11];
          float v99_data = r1[0];
          r1[0] = (v99_data + v98_data);
          float v104_data = r0[12];
          float v105_data = r1[0];
          r1[0] = (v105_data + v104_data);
          float v110_data = r0[13];
          float v111_data = r1[0];
          r1[0] = (v111_data + v110_data);
          float v116_data = r0[14];
          float v117_data = r1[0];
          r1[0] = (v117_data + v116_data);
          float v122_data = r0[15];
          float v123_data = r1[0];
          r1[0] = (v123_data + v122_data);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v128_i0 = 0; v128_i0 < 1; ++v128_i0) {
            float v129_data = r1[v128_i0];
            glb_m0[(v16_lead + (v128_i0 * 16))] = v129_data;
          }
        }
      }
    }
  }
}

