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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 10; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 - 10);
              r0[v22_a] = v20_data;
            }
          }
          float r1[13]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[4] = v128;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[5] = v160;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[7]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          auto& ir2 = r2;
          float v25_data = r0[0];
          float v26_data = r0[1];
          float v27_data = r0[2];
          float v28_acc{};
          float v29_acc{};
          float v30_acc{};
          float v31_acc{};
          float v32_acc{};
          float v33_acc{};
          float v34_acc{};
          float v35_lin = r1[0];
          float v36_bc = tensorforge::broadcast<32, 16, 0>(v35_lin);
          tensorforge::fmacdpp16<0>(v28_acc, v36_bc, v25_data);
          tensorforge::fmacdpp16<1>(v28_acc, v36_bc, v26_data);
          tensorforge::fmacdpp16<2>(v28_acc, v36_bc, v27_data);
          tensorforge::fmacdpp16<3>(v29_acc, v36_bc, v25_data);
          tensorforge::fmacdpp16<4>(v29_acc, v36_bc, v26_data);
          tensorforge::fmacdpp16<5>(v29_acc, v36_bc, v27_data);
          tensorforge::fmacdpp16<6>(v30_acc, v36_bc, v25_data);
          tensorforge::fmacdpp16<7>(v30_acc, v36_bc, v26_data);
          tensorforge::fmacdpp16<8>(v30_acc, v36_bc, v27_data);
          tensorforge::fmacdpp16<9>(v31_acc, v36_bc, v25_data);
          tensorforge::fmacdpp16<10>(v31_acc, v36_bc, v26_data);
          tensorforge::fmacdpp16<11>(v31_acc, v36_bc, v27_data);
          tensorforge::fmacdpp16<12>(v32_acc, v36_bc, v25_data);
          tensorforge::fmacdpp16<13>(v32_acc, v36_bc, v26_data);
          tensorforge::fmacdpp16<14>(v32_acc, v36_bc, v27_data);
          tensorforge::fmacdpp16<15>(v33_acc, v36_bc, v25_data);
          float v37_bc = tensorforge::broadcast<32, 16, 1>(v35_lin);
          tensorforge::fmacdpp16<0>(v33_acc, v37_bc, v26_data);
          tensorforge::fmacdpp16<1>(v33_acc, v37_bc, v27_data);
          tensorforge::fmacdpp16<2>(v34_acc, v37_bc, v25_data);
          tensorforge::fmacdpp16<3>(v34_acc, v37_bc, v26_data);
          tensorforge::fmacdpp16<4>(v34_acc, v37_bc, v27_data);
          ir2[0] = v28_acc;
          ir2[1] = v29_acc;
          ir2[2] = v30_acc;
          ir2[3] = v31_acc;
          ir2[4] = v32_acc;
          ir2[5] = v33_acc;
          ir2[6] = v34_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v41_i0 = 0; v41_i0 < 1; ++v41_i0) {
            int32_t v46_lead = v41_i0 * 32;
            int32_t v48_a = (v3_lead + v46_lead) + 0;
            glb_m0[v48_a] = 0.0f;
            int32_t v55_a = (v3_lead + v46_lead) + 32;
            glb_m0[v55_a] = 0.0f;
            int32_t v62_a = (v3_lead + v46_lead) + 64;
            glb_m0[v62_a] = 0.0f;
            int32_t v69_a = (v3_lead + v46_lead) + 96;
            glb_m0[v69_a] = 0.0f;
            int32_t v76_a = (v3_lead + v46_lead) + 128;
            glb_m0[v76_a] = 0.0f;
            int32_t v83_a = (v3_lead + v46_lead) + 160;
            glb_m0[v83_a] = 0.0f;
            int32_t v84_a = v41_i0 + 0;
            float v86_data = r2[v41_i0];
            int32_t v92_a = (v3_lead + v46_lead) + 192;
            glb_m0[v92_a] = v86_data;
            int32_t v93_a = v41_i0 + 1;
            float v95_data = r2[(v41_i0 + 1)];
            int32_t v101_a = (v3_lead + v46_lead) + 224;
            glb_m0[v101_a] = v95_data;
            int32_t v102_a = v41_i0 + 2;
            float v104_data = r2[(v41_i0 + 2)];
            int32_t v110_a = (v3_lead + v46_lead) + 256;
            glb_m0[v110_a] = v104_data;
            int32_t v111_a = v41_i0 + 3;
            float v113_data = r2[(v41_i0 + 3)];
            int32_t v119_a = (v3_lead + v46_lead) + 288;
            glb_m0[v119_a] = v113_data;
            int32_t v120_a = v41_i0 + 4;
            float v122_data = r2[(v41_i0 + 4)];
            int32_t v128_a = (v3_lead + v46_lead) + 320;
            glb_m0[v128_a] = v122_data;
            int32_t v129_a = v41_i0 + 5;
            float v131_data = r2[(v41_i0 + 5)];
            int32_t v137_a = (v3_lead + v46_lead) + 352;
            glb_m0[v137_a] = v131_data;
            int32_t v138_a = v41_i0 + 6;
            float v140_data = r2[(v41_i0 + 6)];
            int32_t v146_a = (v3_lead + v46_lead) + 384;
            glb_m0[v146_a] = v140_data;
          }
          ;
        }
      }
    }
  }
}

