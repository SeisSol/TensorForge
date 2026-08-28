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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v25_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v27_data = r0[0];
          float v28_data = r0[1];
          float v29_data = r0[2];
          float v30_data = r0[3];
          float v31_data = r0[4];
          float v32_data = r0[5];
          float v33_data = r0[6];
          float v34_data = r0[7];
          float v35_data = r0[8];
          float v36_data = r0[9];
          float v37_data = r0[10];
          float v38_data = r0[11];
          float v39_data = r0[12];
          float v40_data = r0[13];
          float v41_data = r0[14];
          float v42_data = r0[15];
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
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_lin = r1[0];
          tensorforge::fmacdpp16<0>(v43_acc, v59_lin, v27_data);
          tensorforge::fmacdpp16<1>(v43_acc, v59_lin, v28_data);
          tensorforge::fmacdpp16<2>(v44_acc, v59_lin, v27_data);
          tensorforge::fmacdpp16<3>(v44_acc, v59_lin, v28_data);
          tensorforge::fmacdpp16<4>(v44_acc, v59_lin, v29_data);
          tensorforge::fmacdpp16<5>(v45_acc, v59_lin, v28_data);
          tensorforge::fmacdpp16<6>(v45_acc, v59_lin, v29_data);
          tensorforge::fmacdpp16<7>(v45_acc, v59_lin, v30_data);
          tensorforge::fmacdpp16<8>(v46_acc, v59_lin, v29_data);
          tensorforge::fmacdpp16<9>(v46_acc, v59_lin, v30_data);
          tensorforge::fmacdpp16<10>(v46_acc, v59_lin, v31_data);
          tensorforge::fmacdpp16<11>(v47_acc, v59_lin, v30_data);
          tensorforge::fmacdpp16<12>(v47_acc, v59_lin, v31_data);
          tensorforge::fmacdpp16<13>(v47_acc, v59_lin, v32_data);
          tensorforge::fmacdpp16<14>(v48_acc, v59_lin, v31_data);
          tensorforge::fmacdpp16<15>(v48_acc, v59_lin, v32_data);
          float v60_lin = r1[1];
          tensorforge::fmacdpp16<0>(v48_acc, v60_lin, v33_data);
          tensorforge::fmacdpp16<1>(v49_acc, v60_lin, v32_data);
          tensorforge::fmacdpp16<2>(v49_acc, v60_lin, v33_data);
          tensorforge::fmacdpp16<3>(v49_acc, v60_lin, v34_data);
          tensorforge::fmacdpp16<4>(v50_acc, v60_lin, v33_data);
          tensorforge::fmacdpp16<5>(v50_acc, v60_lin, v34_data);
          tensorforge::fmacdpp16<6>(v50_acc, v60_lin, v35_data);
          tensorforge::fmacdpp16<7>(v51_acc, v60_lin, v34_data);
          tensorforge::fmacdpp16<8>(v51_acc, v60_lin, v35_data);
          tensorforge::fmacdpp16<9>(v51_acc, v60_lin, v36_data);
          tensorforge::fmacdpp16<10>(v52_acc, v60_lin, v35_data);
          tensorforge::fmacdpp16<11>(v52_acc, v60_lin, v36_data);
          tensorforge::fmacdpp16<12>(v52_acc, v60_lin, v37_data);
          tensorforge::fmacdpp16<13>(v53_acc, v60_lin, v36_data);
          tensorforge::fmacdpp16<14>(v53_acc, v60_lin, v37_data);
          tensorforge::fmacdpp16<15>(v53_acc, v60_lin, v38_data);
          float v61_lin = r1[2];
          tensorforge::fmacdpp16<0>(v54_acc, v61_lin, v37_data);
          tensorforge::fmacdpp16<1>(v54_acc, v61_lin, v38_data);
          tensorforge::fmacdpp16<2>(v54_acc, v61_lin, v39_data);
          tensorforge::fmacdpp16<3>(v55_acc, v61_lin, v38_data);
          tensorforge::fmacdpp16<4>(v55_acc, v61_lin, v39_data);
          tensorforge::fmacdpp16<5>(v55_acc, v61_lin, v40_data);
          tensorforge::fmacdpp16<6>(v56_acc, v61_lin, v39_data);
          tensorforge::fmacdpp16<7>(v56_acc, v61_lin, v40_data);
          tensorforge::fmacdpp16<8>(v56_acc, v61_lin, v41_data);
          tensorforge::fmacdpp16<9>(v57_acc, v61_lin, v40_data);
          tensorforge::fmacdpp16<10>(v57_acc, v61_lin, v41_data);
          tensorforge::fmacdpp16<11>(v57_acc, v61_lin, v42_data);
          tensorforge::fmacdpp16<12>(v58_acc, v61_lin, v41_data);
          tensorforge::fmacdpp16<13>(v58_acc, v61_lin, v42_data);
          r2[0] = v43_acc;
          r2[1] = v44_acc;
          r2[2] = v45_acc;
          r2[3] = v46_acc;
          r2[4] = v47_acc;
          r2[5] = v48_acc;
          r2[6] = v49_acc;
          r2[7] = v50_acc;
          r2[8] = v51_acc;
          r2[9] = v52_acc;
          r2[10] = v53_acc;
          r2[11] = v54_acc;
          r2[12] = v55_acc;
          r2[13] = v56_acc;
          r2[14] = v57_acc;
          r2[15] = v58_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
            int32_t v73_lead = v10_lead + (v65_i0 * 16);
            #pragma unroll
            for (int32_t v66_i1 = 0; v66_i1 < 16; ++v66_i1) {
              float v68_data = r2[(v65_i0 + v66_i1)];
              glb_m0[(v73_lead + (v66_i1 * 16))] = v68_data;
            }
          }
        }
      }
    }
  }
}

