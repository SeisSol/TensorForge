// === base name ===
kernel_a7d5d30824

// === header ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a7d5d30824, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a7d5d30824), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_a7d5d30824, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16(16) {0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 16;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v14_a = v8_i1 * 16;
              int32_t v15_a = v13_lead + v14_a;
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + v14_a)]);
              r0[(v7_i0 + v8_i1)] = v23_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float v29_data = r0[0];
          float v30_data = r1[0];
          r1[0] = (v30_data + v29_data);
          float v35_data = r0[1];
          float v36_data = r1[0];
          r1[0] = (v36_data + v35_data);
          float v41_data = r0[2];
          float v42_data = r1[0];
          r1[0] = (v42_data + v41_data);
          float v47_data = r0[3];
          float v48_data = r1[0];
          r1[0] = (v48_data + v47_data);
          float v53_data = r0[4];
          float v54_data = r1[0];
          r1[0] = (v54_data + v53_data);
          float v59_data = r0[5];
          float v60_data = r1[0];
          r1[0] = (v60_data + v59_data);
          float v65_data = r0[6];
          float v66_data = r1[0];
          r1[0] = (v66_data + v65_data);
          float v71_data = r0[7];
          float v72_data = r1[0];
          r1[0] = (v72_data + v71_data);
          float v77_data = r0[8];
          float v78_data = r1[0];
          r1[0] = (v78_data + v77_data);
          float v83_data = r0[9];
          float v84_data = r1[0];
          r1[0] = (v84_data + v83_data);
          float v89_data = r0[10];
          float v90_data = r1[0];
          r1[0] = (v90_data + v89_data);
          float v95_data = r0[11];
          float v96_data = r1[0];
          r1[0] = (v96_data + v95_data);
          float v101_data = r0[12];
          float v102_data = r1[0];
          r1[0] = (v102_data + v101_data);
          float v107_data = r0[13];
          float v108_data = r1[0];
          r1[0] = (v108_data + v107_data);
          float v113_data = r0[14];
          float v114_data = r1[0];
          r1[0] = (v114_data + v113_data);
          float v119_data = r0[15];
          float v120_data = r1[0];
          r1[0] = (v120_data + v119_data);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v125_i0 = 0; v125_i0 < 1; ++v125_i0) {
            float v126_data = r1[v125_i0];
            glb_m0[(v6_lead + (v125_i0 * 16))] = v126_data;
          }
        }
      }
    }
  }
}

