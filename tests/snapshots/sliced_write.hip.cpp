// === base name ===
kernel_49acf988a6

// === header ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_49acf988a6, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_49acf988a6), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_49acf988a6, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{6..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{6..13})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 32;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 10; v8_i1 < 13; ++v8_i1) {
              int32_t v14_a = v8_i1 * 32;
              int32_t v15_a = v13_lead + v14_a;
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + v14_a)]);
              int32_t v25_a = v7_i0 + (v8_i1 - 10);
              r0[v25_a] = v23_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v28_lin;
          float v29_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v29_lin;
          float v30_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v30_lin;
          float v31_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v31_lin;
          float v32_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[7]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float v34_data = r0[0];
          float v35_data = r0[1];
          float v36_data = r0[2];
          float v37_acc{};
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_lin = r1[0];
          float v45_bc = tensorforge::broadcast<32, 16, 0>(v44_lin);
          tensorforge::fmacdpp16<0>(v37_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<1>(v37_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<2>(v37_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<3>(v38_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<4>(v38_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<5>(v38_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<6>(v39_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<7>(v39_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<8>(v39_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<9>(v40_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<10>(v40_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<11>(v40_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<12>(v41_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<13>(v41_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<14>(v41_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<15>(v42_acc, v45_bc, v34_data);
          float v46_bc = tensorforge::broadcast<32, 16, 1>(v44_lin);
          tensorforge::fmacdpp16<0>(v42_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<1>(v42_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<2>(v43_acc, v46_bc, v34_data);
          tensorforge::fmacdpp16<3>(v43_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<4>(v43_acc, v46_bc, v36_data);
          r2[0] = v37_acc;
          r2[1] = v38_acc;
          r2[2] = v39_acc;
          r2[3] = v40_acc;
          r2[4] = v41_acc;
          r2[5] = v42_acc;
          r2[6] = v43_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v50_i0 = 0; v50_i0 < 1; ++v50_i0) {
            int32_t v55_lead = v50_i0 * 32;
            int32_t v57_a = (v6_lead + v55_lead) + 0;
            glb_m0[v57_a] = 0.0f;
            int32_t v64_a = (v6_lead + v55_lead) + 32;
            glb_m0[v64_a] = 0.0f;
            int32_t v71_a = (v6_lead + v55_lead) + 64;
            glb_m0[v71_a] = 0.0f;
            int32_t v78_a = (v6_lead + v55_lead) + 96;
            glb_m0[v78_a] = 0.0f;
            int32_t v85_a = (v6_lead + v55_lead) + 128;
            glb_m0[v85_a] = 0.0f;
            int32_t v92_a = (v6_lead + v55_lead) + 160;
            glb_m0[v92_a] = 0.0f;
            int32_t v93_a = v50_i0 + 0;
            float v95_data = r2[v50_i0];
            int32_t v101_a = (v6_lead + v55_lead) + 192;
            glb_m0[v101_a] = v95_data;
            int32_t v102_a = v50_i0 + 1;
            float v104_data = r2[(v50_i0 + 1)];
            int32_t v110_a = (v6_lead + v55_lead) + 224;
            glb_m0[v110_a] = v104_data;
            int32_t v111_a = v50_i0 + 2;
            float v113_data = r2[(v50_i0 + 2)];
            int32_t v119_a = (v6_lead + v55_lead) + 256;
            glb_m0[v119_a] = v113_data;
            int32_t v120_a = v50_i0 + 3;
            float v122_data = r2[(v50_i0 + 3)];
            int32_t v128_a = (v6_lead + v55_lead) + 288;
            glb_m0[v128_a] = v122_data;
            int32_t v129_a = v50_i0 + 4;
            float v131_data = r2[(v50_i0 + 4)];
            int32_t v137_a = (v6_lead + v55_lead) + 320;
            glb_m0[v137_a] = v131_data;
            int32_t v138_a = v50_i0 + 5;
            float v140_data = r2[(v50_i0 + 5)];
            int32_t v146_a = (v6_lead + v55_lead) + 352;
            glb_m0[v146_a] = v140_data;
            int32_t v147_a = v50_i0 + 6;
            float v149_data = r2[(v50_i0 + 6)];
            int32_t v155_a = (v6_lead + v55_lead) + 384;
            glb_m0[v155_a] = v149_data;
          }
        }
      }
    }
  }
}

