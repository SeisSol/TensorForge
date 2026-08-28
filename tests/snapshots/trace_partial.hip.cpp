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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v9_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v16_lead = v9_lead + (v10_i0 * 16);
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v11_i1 * 16))]);
              r0[(v10_i0 + v11_i1)] = v19_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float v25_data = r0[0];
          float v26_data = r1[0];
          r1[0] = (v26_data + v25_data);
          float v31_data = r0[1];
          float v32_data = r1[0];
          r1[0] = (v32_data + v31_data);
          float v37_data = r0[2];
          float v38_data = r1[0];
          r1[0] = (v38_data + v37_data);
          float v43_data = r0[3];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v49_data = r0[4];
          float v50_data = r1[0];
          r1[0] = (v50_data + v49_data);
          float v55_data = r0[5];
          float v56_data = r1[0];
          r1[0] = (v56_data + v55_data);
          float v61_data = r0[6];
          float v62_data = r1[0];
          r1[0] = (v62_data + v61_data);
          float v67_data = r0[7];
          float v68_data = r1[0];
          r1[0] = (v68_data + v67_data);
          float v73_data = r0[8];
          float v74_data = r1[0];
          r1[0] = (v74_data + v73_data);
          float v79_data = r0[9];
          float v80_data = r1[0];
          r1[0] = (v80_data + v79_data);
          float v85_data = r0[10];
          float v86_data = r1[0];
          r1[0] = (v86_data + v85_data);
          float v91_data = r0[11];
          float v92_data = r1[0];
          r1[0] = (v92_data + v91_data);
          float v97_data = r0[12];
          float v98_data = r1[0];
          r1[0] = (v98_data + v97_data);
          float v103_data = r0[13];
          float v104_data = r1[0];
          r1[0] = (v104_data + v103_data);
          float v109_data = r0[14];
          float v110_data = r1[0];
          r1[0] = (v110_data + v109_data);
          float v115_data = r0[15];
          float v116_data = r1[0];
          r1[0] = (v116_data + v115_data);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v121_i0 = 0; v121_i0 < 1; ++v121_i0) {
            float v122_data = r1[v121_i0];
            glb_m0[(v9_lead + (v121_i0 * 16))] = v122_data;
          }
        }
      }
    }
  }
}

