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
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 10; v9_i1 < 13; ++v9_i1) {
              int32_t v15_a = v9_i1 * 32;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 - 10))] = v24_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v28_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          float v29_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v29_lin;
          float v30_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v30_lin;
          float v31_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v31_lin;
          float v32_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v32_lin;
          float v33_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v33_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[7]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float v35_data = r0[0];
          float v36_data = r0[1];
          float v37_data = r0[2];
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_lin = r1[0];
          float v46_bc = tensorforge::broadcast<32, 16, 0>(v45_lin);
          tensorforge::fmacdpp16<0>(v38_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<1>(v38_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<2>(v38_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<3>(v39_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<4>(v39_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<5>(v39_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<6>(v40_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<7>(v40_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<8>(v40_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<9>(v41_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<10>(v41_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<11>(v41_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<12>(v42_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<13>(v42_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<14>(v42_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<15>(v43_acc, v46_bc, v35_data);
          float v47_bc = tensorforge::broadcast<32, 16, 1>(v45_lin);
          tensorforge::fmacdpp16<0>(v43_acc, v47_bc, v36_data);
          tensorforge::fmacdpp16<1>(v43_acc, v47_bc, v37_data);
          tensorforge::fmacdpp16<2>(v44_acc, v47_bc, v35_data);
          tensorforge::fmacdpp16<3>(v44_acc, v47_bc, v36_data);
          tensorforge::fmacdpp16<4>(v44_acc, v47_bc, v37_data);
          r2[0] = v38_acc;
          r2[1] = v39_acc;
          r2[2] = v40_acc;
          r2[3] = v41_acc;
          r2[4] = v42_acc;
          r2[5] = v43_acc;
          r2[6] = v44_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v56_lead = v51_i0 * 32;
            glb_m0[(v7_lead + v56_lead)] = 0.0f;
            glb_m0[((v7_lead + v56_lead) + 32)] = 0.0f;
            glb_m0[((v7_lead + v56_lead) + 64)] = 0.0f;
            glb_m0[((v7_lead + v56_lead) + 96)] = 0.0f;
            glb_m0[((v7_lead + v56_lead) + 128)] = 0.0f;
            glb_m0[((v7_lead + v56_lead) + 160)] = 0.0f;
            int32_t v94_a = v51_i0 + 0;
            float v96_data = r2[v51_i0];
            glb_m0[((v7_lead + v56_lead) + 192)] = v96_data;
            int32_t v103_a = v51_i0 + 1;
            float v105_data = r2[(v51_i0 + 1)];
            glb_m0[((v7_lead + v56_lead) + 224)] = v105_data;
            int32_t v112_a = v51_i0 + 2;
            float v114_data = r2[(v51_i0 + 2)];
            glb_m0[((v7_lead + v56_lead) + 256)] = v114_data;
            int32_t v121_a = v51_i0 + 3;
            float v123_data = r2[(v51_i0 + 3)];
            glb_m0[((v7_lead + v56_lead) + 288)] = v123_data;
            int32_t v130_a = v51_i0 + 4;
            float v132_data = r2[(v51_i0 + 4)];
            glb_m0[((v7_lead + v56_lead) + 320)] = v132_data;
            int32_t v139_a = v51_i0 + 5;
            float v141_data = r2[(v51_i0 + 5)];
            glb_m0[((v7_lead + v56_lead) + 352)] = v141_data;
            int32_t v148_a = v51_i0 + 6;
            float v150_data = r2[(v51_i0 + 6)];
            glb_m0[((v7_lead + v56_lead) + 384)] = v150_data;
          }
        }
      }
    }
  }
}

