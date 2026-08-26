// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_30948bd44e, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_30948bd44e), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_30948bd44e, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 16;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[16]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v24_data = r0[0];
          float v25_data = r0[1];
          float v26_data = r0[2];
          float v27_data = r0[3];
          float v28_data = r0[4];
          float v29_data = r0[5];
          float v30_data = r0[6];
          float v31_data = r0[7];
          float v32_data = r0[8];
          float v33_data = r0[9];
          float v34_data = r0[10];
          float v35_data = r0[11];
          float v36_data = r0[12];
          float v37_data = r0[13];
          float v38_data = r0[14];
          float v39_data = r0[15];
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_acc{};
          float v46_acc{};
          float v47_acc{};
          float v48_acc{};
          float v49_acc{};
          float v50_acc{};
          float v51_acc{};
          float v52_acc{};
          float v53_acc{};
          float v54_acc{};
          float v55_acc{};
          float v56_lin = r1[0];
          tensorforge::fmacdpp16<0>(v40_acc, v56_lin, v24_data);
          tensorforge::fmacdpp16<1>(v40_acc, v56_lin, v25_data);
          tensorforge::fmacdpp16<2>(v41_acc, v56_lin, v24_data);
          tensorforge::fmacdpp16<3>(v41_acc, v56_lin, v25_data);
          tensorforge::fmacdpp16<4>(v41_acc, v56_lin, v26_data);
          tensorforge::fmacdpp16<5>(v42_acc, v56_lin, v25_data);
          tensorforge::fmacdpp16<6>(v42_acc, v56_lin, v26_data);
          tensorforge::fmacdpp16<7>(v42_acc, v56_lin, v27_data);
          tensorforge::fmacdpp16<8>(v43_acc, v56_lin, v26_data);
          tensorforge::fmacdpp16<9>(v43_acc, v56_lin, v27_data);
          tensorforge::fmacdpp16<10>(v43_acc, v56_lin, v28_data);
          tensorforge::fmacdpp16<11>(v44_acc, v56_lin, v27_data);
          tensorforge::fmacdpp16<12>(v44_acc, v56_lin, v28_data);
          tensorforge::fmacdpp16<13>(v44_acc, v56_lin, v29_data);
          tensorforge::fmacdpp16<14>(v45_acc, v56_lin, v28_data);
          tensorforge::fmacdpp16<15>(v45_acc, v56_lin, v29_data);
          float v57_lin = r1[1];
          tensorforge::fmacdpp16<0>(v45_acc, v57_lin, v30_data);
          tensorforge::fmacdpp16<1>(v46_acc, v57_lin, v29_data);
          tensorforge::fmacdpp16<2>(v46_acc, v57_lin, v30_data);
          tensorforge::fmacdpp16<3>(v46_acc, v57_lin, v31_data);
          tensorforge::fmacdpp16<4>(v47_acc, v57_lin, v30_data);
          tensorforge::fmacdpp16<5>(v47_acc, v57_lin, v31_data);
          tensorforge::fmacdpp16<6>(v47_acc, v57_lin, v32_data);
          tensorforge::fmacdpp16<7>(v48_acc, v57_lin, v31_data);
          tensorforge::fmacdpp16<8>(v48_acc, v57_lin, v32_data);
          tensorforge::fmacdpp16<9>(v48_acc, v57_lin, v33_data);
          tensorforge::fmacdpp16<10>(v49_acc, v57_lin, v32_data);
          tensorforge::fmacdpp16<11>(v49_acc, v57_lin, v33_data);
          tensorforge::fmacdpp16<12>(v49_acc, v57_lin, v34_data);
          tensorforge::fmacdpp16<13>(v50_acc, v57_lin, v33_data);
          tensorforge::fmacdpp16<14>(v50_acc, v57_lin, v34_data);
          tensorforge::fmacdpp16<15>(v50_acc, v57_lin, v35_data);
          float v58_lin = r1[2];
          tensorforge::fmacdpp16<0>(v51_acc, v58_lin, v34_data);
          tensorforge::fmacdpp16<1>(v51_acc, v58_lin, v35_data);
          tensorforge::fmacdpp16<2>(v51_acc, v58_lin, v36_data);
          tensorforge::fmacdpp16<3>(v52_acc, v58_lin, v35_data);
          tensorforge::fmacdpp16<4>(v52_acc, v58_lin, v36_data);
          tensorforge::fmacdpp16<5>(v52_acc, v58_lin, v37_data);
          tensorforge::fmacdpp16<6>(v53_acc, v58_lin, v36_data);
          tensorforge::fmacdpp16<7>(v53_acc, v58_lin, v37_data);
          tensorforge::fmacdpp16<8>(v53_acc, v58_lin, v38_data);
          tensorforge::fmacdpp16<9>(v54_acc, v58_lin, v37_data);
          tensorforge::fmacdpp16<10>(v54_acc, v58_lin, v38_data);
          tensorforge::fmacdpp16<11>(v54_acc, v58_lin, v39_data);
          tensorforge::fmacdpp16<12>(v55_acc, v58_lin, v38_data);
          tensorforge::fmacdpp16<13>(v55_acc, v58_lin, v39_data);
          ir2[0] = v40_acc;
          ir2[1] = v41_acc;
          ir2[2] = v42_acc;
          ir2[3] = v43_acc;
          ir2[4] = v44_acc;
          ir2[5] = v45_acc;
          ir2[6] = v46_acc;
          ir2[7] = v47_acc;
          ir2[8] = v48_acc;
          ir2[9] = v49_acc;
          ir2[10] = v50_acc;
          ir2[11] = v51_acc;
          ir2[12] = v52_acc;
          ir2[13] = v53_acc;
          ir2[14] = v54_acc;
          ir2[15] = v55_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
            int32_t v71_lead = v3_lead + (v62_i0 * 16);
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 16; ++v63_i1) {
              int32_t v64_a = v62_i0 + v63_i1;
              float v66_data = r2[(v62_i0 + v63_i1)];
              int32_t v73_a = v71_lead + (v63_i1 * 16);
              glb_m0[v73_a] = v66_data;
            }
          }
          ;
        }
      }
    }
  }
}

