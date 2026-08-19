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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 - 10);
              r0[v14_a] = v12_data;
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
          float v15_data = r0[0];
          float v16_data = r0[1];
          float v17_data = r0[2];
          float v18_acc{};
          float v19_acc{};
          float v20_acc{};
          float v21_acc{};
          float v22_acc{};
          float v23_acc{};
          float v24_acc{};
          float v25_lin = r1[0];
          float v26_bc = tensorforge::broadcast<32, 16, 0>(v25_lin);
          tensorforge::fmacdpp16<0>(v18_acc, v26_bc, v15_data);
          tensorforge::fmacdpp16<1>(v18_acc, v26_bc, v16_data);
          tensorforge::fmacdpp16<2>(v18_acc, v26_bc, v17_data);
          tensorforge::fmacdpp16<3>(v19_acc, v26_bc, v15_data);
          tensorforge::fmacdpp16<4>(v19_acc, v26_bc, v16_data);
          tensorforge::fmacdpp16<5>(v19_acc, v26_bc, v17_data);
          tensorforge::fmacdpp16<6>(v20_acc, v26_bc, v15_data);
          tensorforge::fmacdpp16<7>(v20_acc, v26_bc, v16_data);
          tensorforge::fmacdpp16<8>(v20_acc, v26_bc, v17_data);
          tensorforge::fmacdpp16<9>(v21_acc, v26_bc, v15_data);
          tensorforge::fmacdpp16<10>(v21_acc, v26_bc, v16_data);
          tensorforge::fmacdpp16<11>(v21_acc, v26_bc, v17_data);
          tensorforge::fmacdpp16<12>(v22_acc, v26_bc, v15_data);
          tensorforge::fmacdpp16<13>(v22_acc, v26_bc, v16_data);
          tensorforge::fmacdpp16<14>(v22_acc, v26_bc, v17_data);
          tensorforge::fmacdpp16<15>(v23_acc, v26_bc, v15_data);
          float v27_bc = tensorforge::broadcast<32, 16, 1>(v25_lin);
          tensorforge::fmacdpp16<0>(v23_acc, v27_bc, v16_data);
          tensorforge::fmacdpp16<1>(v23_acc, v27_bc, v17_data);
          tensorforge::fmacdpp16<2>(v24_acc, v27_bc, v15_data);
          tensorforge::fmacdpp16<3>(v24_acc, v27_bc, v16_data);
          tensorforge::fmacdpp16<4>(v24_acc, v27_bc, v17_data);
          ir2[0] = v18_acc;
          ir2[1] = v19_acc;
          ir2[2] = v20_acc;
          ir2[3] = v21_acc;
          ir2[4] = v22_acc;
          ir2[5] = v23_acc;
          ir2[6] = v24_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v30_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v36_lead = v31_i0 * 32;
            int32_t v38_a = (v30_lead + v36_lead) + 0;
            glb_m0[v38_a] = 0.0f;
            int32_t v45_a = (v30_lead + v36_lead) + 32;
            glb_m0[v45_a] = 0.0f;
            int32_t v52_a = (v30_lead + v36_lead) + 64;
            glb_m0[v52_a] = 0.0f;
            int32_t v59_a = (v30_lead + v36_lead) + 96;
            glb_m0[v59_a] = 0.0f;
            int32_t v66_a = (v30_lead + v36_lead) + 128;
            glb_m0[v66_a] = 0.0f;
            int32_t v73_a = (v30_lead + v36_lead) + 160;
            glb_m0[v73_a] = 0.0f;
            int32_t v74_a = v31_i0 + 0;
            float v76_data = r2[v31_i0];
            int32_t v82_a = (v30_lead + v36_lead) + 192;
            glb_m0[v82_a] = v76_data;
            int32_t v83_a = v31_i0 + 1;
            float v85_data = r2[(v31_i0 + 1)];
            int32_t v91_a = (v30_lead + v36_lead) + 224;
            glb_m0[v91_a] = v85_data;
            int32_t v92_a = v31_i0 + 2;
            float v94_data = r2[(v31_i0 + 2)];
            int32_t v100_a = (v30_lead + v36_lead) + 256;
            glb_m0[v100_a] = v94_data;
            int32_t v101_a = v31_i0 + 3;
            float v103_data = r2[(v31_i0 + 3)];
            int32_t v109_a = (v30_lead + v36_lead) + 288;
            glb_m0[v109_a] = v103_data;
            int32_t v110_a = v31_i0 + 4;
            float v112_data = r2[(v31_i0 + 4)];
            int32_t v118_a = (v30_lead + v36_lead) + 320;
            glb_m0[v118_a] = v112_data;
            int32_t v119_a = v31_i0 + 5;
            float v121_data = r2[(v31_i0 + 5)];
            int32_t v127_a = (v30_lead + v36_lead) + 352;
            glb_m0[v127_a] = v121_data;
            int32_t v128_a = v31_i0 + 6;
            float v130_data = r2[(v31_i0 + 6)];
            int32_t v136_a = (v30_lead + v36_lead) + 384;
            glb_m0[v136_a] = v130_data;
          }
          ;
        }
      }
    }
  }
}

