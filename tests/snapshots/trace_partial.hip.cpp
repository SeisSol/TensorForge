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
          int32_t v5_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
            int32_t v11_lead = v6_i0 * 16;
            int32_t v12_lead = v5_lead + v11_lead;
            int32_t v19_lead = v5_lead + v11_lead;
            #pragma unroll
            for (int32_t v7_i1 = 0; v7_i1 < 16; ++v7_i1) {
              int32_t v13_a = v7_i1 * 16;
              int32_t v14_a = v12_lead + v13_a;
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + v13_a)]);
              int32_t v23_a = v6_i0 + v7_i1;
              r0[v23_a] = v22_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float v28_data = r0[0];
          float v29_data = r1[0];
          r1[0] = (v29_data + v28_data);
          float v34_data = r0[1];
          float v35_data = r1[0];
          r1[0] = (v35_data + v34_data);
          float v40_data = r0[2];
          float v41_data = r1[0];
          r1[0] = (v41_data + v40_data);
          float v46_data = r0[3];
          float v47_data = r1[0];
          r1[0] = (v47_data + v46_data);
          float v52_data = r0[4];
          float v53_data = r1[0];
          r1[0] = (v53_data + v52_data);
          float v58_data = r0[5];
          float v59_data = r1[0];
          r1[0] = (v59_data + v58_data);
          float v64_data = r0[6];
          float v65_data = r1[0];
          r1[0] = (v65_data + v64_data);
          float v70_data = r0[7];
          float v71_data = r1[0];
          r1[0] = (v71_data + v70_data);
          float v76_data = r0[8];
          float v77_data = r1[0];
          r1[0] = (v77_data + v76_data);
          float v82_data = r0[9];
          float v83_data = r1[0];
          r1[0] = (v83_data + v82_data);
          float v88_data = r0[10];
          float v89_data = r1[0];
          r1[0] = (v89_data + v88_data);
          float v94_data = r0[11];
          float v95_data = r1[0];
          r1[0] = (v95_data + v94_data);
          float v100_data = r0[12];
          float v101_data = r1[0];
          r1[0] = (v101_data + v100_data);
          float v106_data = r0[13];
          float v107_data = r1[0];
          r1[0] = (v107_data + v106_data);
          float v112_data = r0[14];
          float v113_data = r1[0];
          r1[0] = (v113_data + v112_data);
          float v118_data = r0[15];
          float v119_data = r1[0];
          r1[0] = (v119_data + v118_data);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v124_i0 = 0; v124_i0 < 1; ++v124_i0) {
            float v125_data = r1[v124_i0];
            int32_t v130_lead = v5_lead + (v124_i0 * 16);
            glb_m0[v130_lead] = v125_data;
          }
        }
      }
    }
  }
}

