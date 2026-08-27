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
              int32_t v24_a = v7_i0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v26_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v26_lin;
          float v27_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v27_lin;
          float v28_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v28_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v30_data = r0[0];
          float v31_data = r0[1];
          float v32_data = r0[2];
          float v33_data = r0[3];
          float v34_data = r0[4];
          float v35_data = r0[5];
          float v36_data = r0[6];
          float v37_data = r0[7];
          float v38_data = r0[8];
          float v39_data = r0[9];
          float v40_data = r0[10];
          float v41_data = r0[11];
          float v42_data = r0[12];
          float v43_data = r0[13];
          float v44_data = r0[14];
          float v45_data = r0[15];
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
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_lin = r1[0];
          tensorforge::fmacdpp16<0>(v46_acc, v62_lin, v30_data);
          tensorforge::fmacdpp16<1>(v46_acc, v62_lin, v31_data);
          tensorforge::fmacdpp16<2>(v47_acc, v62_lin, v30_data);
          tensorforge::fmacdpp16<3>(v47_acc, v62_lin, v31_data);
          tensorforge::fmacdpp16<4>(v47_acc, v62_lin, v32_data);
          tensorforge::fmacdpp16<5>(v48_acc, v62_lin, v31_data);
          tensorforge::fmacdpp16<6>(v48_acc, v62_lin, v32_data);
          tensorforge::fmacdpp16<7>(v48_acc, v62_lin, v33_data);
          tensorforge::fmacdpp16<8>(v49_acc, v62_lin, v32_data);
          tensorforge::fmacdpp16<9>(v49_acc, v62_lin, v33_data);
          tensorforge::fmacdpp16<10>(v49_acc, v62_lin, v34_data);
          tensorforge::fmacdpp16<11>(v50_acc, v62_lin, v33_data);
          tensorforge::fmacdpp16<12>(v50_acc, v62_lin, v34_data);
          tensorforge::fmacdpp16<13>(v50_acc, v62_lin, v35_data);
          tensorforge::fmacdpp16<14>(v51_acc, v62_lin, v34_data);
          tensorforge::fmacdpp16<15>(v51_acc, v62_lin, v35_data);
          float v63_lin = r1[1];
          tensorforge::fmacdpp16<0>(v51_acc, v63_lin, v36_data);
          tensorforge::fmacdpp16<1>(v52_acc, v63_lin, v35_data);
          tensorforge::fmacdpp16<2>(v52_acc, v63_lin, v36_data);
          tensorforge::fmacdpp16<3>(v52_acc, v63_lin, v37_data);
          tensorforge::fmacdpp16<4>(v53_acc, v63_lin, v36_data);
          tensorforge::fmacdpp16<5>(v53_acc, v63_lin, v37_data);
          tensorforge::fmacdpp16<6>(v53_acc, v63_lin, v38_data);
          tensorforge::fmacdpp16<7>(v54_acc, v63_lin, v37_data);
          tensorforge::fmacdpp16<8>(v54_acc, v63_lin, v38_data);
          tensorforge::fmacdpp16<9>(v54_acc, v63_lin, v39_data);
          tensorforge::fmacdpp16<10>(v55_acc, v63_lin, v38_data);
          tensorforge::fmacdpp16<11>(v55_acc, v63_lin, v39_data);
          tensorforge::fmacdpp16<12>(v55_acc, v63_lin, v40_data);
          tensorforge::fmacdpp16<13>(v56_acc, v63_lin, v39_data);
          tensorforge::fmacdpp16<14>(v56_acc, v63_lin, v40_data);
          tensorforge::fmacdpp16<15>(v56_acc, v63_lin, v41_data);
          float v64_lin = r1[2];
          tensorforge::fmacdpp16<0>(v57_acc, v64_lin, v40_data);
          tensorforge::fmacdpp16<1>(v57_acc, v64_lin, v41_data);
          tensorforge::fmacdpp16<2>(v57_acc, v64_lin, v42_data);
          tensorforge::fmacdpp16<3>(v58_acc, v64_lin, v41_data);
          tensorforge::fmacdpp16<4>(v58_acc, v64_lin, v42_data);
          tensorforge::fmacdpp16<5>(v58_acc, v64_lin, v43_data);
          tensorforge::fmacdpp16<6>(v59_acc, v64_lin, v42_data);
          tensorforge::fmacdpp16<7>(v59_acc, v64_lin, v43_data);
          tensorforge::fmacdpp16<8>(v59_acc, v64_lin, v44_data);
          tensorforge::fmacdpp16<9>(v60_acc, v64_lin, v43_data);
          tensorforge::fmacdpp16<10>(v60_acc, v64_lin, v44_data);
          tensorforge::fmacdpp16<11>(v60_acc, v64_lin, v45_data);
          tensorforge::fmacdpp16<12>(v61_acc, v64_lin, v44_data);
          tensorforge::fmacdpp16<13>(v61_acc, v64_lin, v45_data);
          ir2[0] = v46_acc;
          ir2[1] = v47_acc;
          ir2[2] = v48_acc;
          ir2[3] = v49_acc;
          ir2[4] = v50_acc;
          ir2[5] = v51_acc;
          ir2[6] = v52_acc;
          ir2[7] = v53_acc;
          ir2[8] = v54_acc;
          ir2[9] = v55_acc;
          ir2[10] = v56_acc;
          ir2[11] = v57_acc;
          ir2[12] = v58_acc;
          ir2[13] = v59_acc;
          ir2[14] = v60_acc;
          ir2[15] = v61_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v77_lead = v6_lead + (v68_i0 * 16);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 16; ++v69_i1) {
              int32_t v70_a = v68_i0 + v69_i1;
              float v72_data = r2[(v68_i0 + v69_i1)];
              int32_t v79_a = v77_lead + (v69_i1 * 16);
              glb_m0[v79_a] = v72_data;
            }
          }
          ;
        }
      }
    }
  }
}

