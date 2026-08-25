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
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 16;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 16;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          auto& ir1 = r1;
          float v24_data = r0[0];
          float v25_data = ir1[0];
          ir1[0] = (v25_data + v24_data);
          float v30_data = r0[1];
          float v31_data = ir1[0];
          ir1[0] = (v31_data + v30_data);
          float v36_data = r0[2];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + v36_data);
          float v42_data = r0[3];
          float v43_data = ir1[0];
          ir1[0] = (v43_data + v42_data);
          float v48_data = r0[4];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v54_data = r0[5];
          float v55_data = ir1[0];
          ir1[0] = (v55_data + v54_data);
          float v60_data = r0[6];
          float v61_data = ir1[0];
          ir1[0] = (v61_data + v60_data);
          float v66_data = r0[7];
          float v67_data = ir1[0];
          ir1[0] = (v67_data + v66_data);
          float v72_data = r0[8];
          float v73_data = ir1[0];
          ir1[0] = (v73_data + v72_data);
          float v78_data = r0[9];
          float v79_data = ir1[0];
          ir1[0] = (v79_data + v78_data);
          float v84_data = r0[10];
          float v85_data = ir1[0];
          ir1[0] = (v85_data + v84_data);
          float v90_data = r0[11];
          float v91_data = ir1[0];
          ir1[0] = (v91_data + v90_data);
          float v96_data = r0[12];
          float v97_data = ir1[0];
          ir1[0] = (v97_data + v96_data);
          float v102_data = r0[13];
          float v103_data = ir1[0];
          ir1[0] = (v103_data + v102_data);
          float v108_data = r0[14];
          float v109_data = ir1[0];
          ir1[0] = (v109_data + v108_data);
          float v114_data = r0[15];
          float v115_data = ir1[0];
          ir1[0] = (v115_data + v114_data);
          // glb_m0 = store{r>g}(r1);
          int32_t v119_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v120_i0 = 0; v120_i0 < 1; ++v120_i0) {
            float v121_data = r1[v120_i0];
            int32_t v126_lead = v119_lead + (v120_i0 * 16);
            glb_m0[v126_lead] = v121_data;
          }
          ;
        }
      }
    }
  }
}

