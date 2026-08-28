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
            int32_t v16_lead = v11_i0 * 16;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v18_a = v12_i1 * 16;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + v12_i1)] = v27_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          float v31_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v31_lin;
          float v32_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v34_data = r0[0];
          float v35_data = r0[1];
          float v36_data = r0[2];
          float v37_data = r0[3];
          float v38_data = r0[4];
          float v39_data = r0[5];
          float v40_data = r0[6];
          float v41_data = r0[7];
          float v42_data = r0[8];
          float v43_data = r0[9];
          float v44_data = r0[10];
          float v45_data = r0[11];
          float v46_data = r0[12];
          float v47_data = r0[13];
          float v48_data = r0[14];
          float v49_data = r0[15];
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
          float v62_acc{};
          float v63_acc{};
          float v64_acc{};
          float v65_acc{};
          float v66_lin = r1[0];
          tensorforge::fmacdpp16<0>(v50_acc, v66_lin, v34_data);
          tensorforge::fmacdpp16<1>(v50_acc, v66_lin, v35_data);
          tensorforge::fmacdpp16<2>(v51_acc, v66_lin, v34_data);
          tensorforge::fmacdpp16<3>(v51_acc, v66_lin, v35_data);
          tensorforge::fmacdpp16<4>(v51_acc, v66_lin, v36_data);
          tensorforge::fmacdpp16<5>(v52_acc, v66_lin, v35_data);
          tensorforge::fmacdpp16<6>(v52_acc, v66_lin, v36_data);
          tensorforge::fmacdpp16<7>(v52_acc, v66_lin, v37_data);
          tensorforge::fmacdpp16<8>(v53_acc, v66_lin, v36_data);
          tensorforge::fmacdpp16<9>(v53_acc, v66_lin, v37_data);
          tensorforge::fmacdpp16<10>(v53_acc, v66_lin, v38_data);
          tensorforge::fmacdpp16<11>(v54_acc, v66_lin, v37_data);
          tensorforge::fmacdpp16<12>(v54_acc, v66_lin, v38_data);
          tensorforge::fmacdpp16<13>(v54_acc, v66_lin, v39_data);
          tensorforge::fmacdpp16<14>(v55_acc, v66_lin, v38_data);
          tensorforge::fmacdpp16<15>(v55_acc, v66_lin, v39_data);
          float v67_lin = r1[1];
          tensorforge::fmacdpp16<0>(v55_acc, v67_lin, v40_data);
          tensorforge::fmacdpp16<1>(v56_acc, v67_lin, v39_data);
          tensorforge::fmacdpp16<2>(v56_acc, v67_lin, v40_data);
          tensorforge::fmacdpp16<3>(v56_acc, v67_lin, v41_data);
          tensorforge::fmacdpp16<4>(v57_acc, v67_lin, v40_data);
          tensorforge::fmacdpp16<5>(v57_acc, v67_lin, v41_data);
          tensorforge::fmacdpp16<6>(v57_acc, v67_lin, v42_data);
          tensorforge::fmacdpp16<7>(v58_acc, v67_lin, v41_data);
          tensorforge::fmacdpp16<8>(v58_acc, v67_lin, v42_data);
          tensorforge::fmacdpp16<9>(v58_acc, v67_lin, v43_data);
          tensorforge::fmacdpp16<10>(v59_acc, v67_lin, v42_data);
          tensorforge::fmacdpp16<11>(v59_acc, v67_lin, v43_data);
          tensorforge::fmacdpp16<12>(v59_acc, v67_lin, v44_data);
          tensorforge::fmacdpp16<13>(v60_acc, v67_lin, v43_data);
          tensorforge::fmacdpp16<14>(v60_acc, v67_lin, v44_data);
          tensorforge::fmacdpp16<15>(v60_acc, v67_lin, v45_data);
          float v68_lin = r1[2];
          tensorforge::fmacdpp16<0>(v61_acc, v68_lin, v44_data);
          tensorforge::fmacdpp16<1>(v61_acc, v68_lin, v45_data);
          tensorforge::fmacdpp16<2>(v61_acc, v68_lin, v46_data);
          tensorforge::fmacdpp16<3>(v62_acc, v68_lin, v45_data);
          tensorforge::fmacdpp16<4>(v62_acc, v68_lin, v46_data);
          tensorforge::fmacdpp16<5>(v62_acc, v68_lin, v47_data);
          tensorforge::fmacdpp16<6>(v63_acc, v68_lin, v46_data);
          tensorforge::fmacdpp16<7>(v63_acc, v68_lin, v47_data);
          tensorforge::fmacdpp16<8>(v63_acc, v68_lin, v48_data);
          tensorforge::fmacdpp16<9>(v64_acc, v68_lin, v47_data);
          tensorforge::fmacdpp16<10>(v64_acc, v68_lin, v48_data);
          tensorforge::fmacdpp16<11>(v64_acc, v68_lin, v49_data);
          tensorforge::fmacdpp16<12>(v65_acc, v68_lin, v48_data);
          tensorforge::fmacdpp16<13>(v65_acc, v68_lin, v49_data);
          r2[0] = v50_acc;
          r2[1] = v51_acc;
          r2[2] = v52_acc;
          r2[3] = v53_acc;
          r2[4] = v54_acc;
          r2[5] = v55_acc;
          r2[6] = v56_acc;
          r2[7] = v57_acc;
          r2[8] = v58_acc;
          r2[9] = v59_acc;
          r2[10] = v60_acc;
          r2[11] = v61_acc;
          r2[12] = v62_acc;
          r2[13] = v63_acc;
          r2[14] = v64_acc;
          r2[15] = v65_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v72_i0 = 0; v72_i0 < 1; ++v72_i0) {
            int32_t v81_lead = v10_lead + (v72_i0 * 16);
            #pragma unroll
            for (int32_t v73_i1 = 0; v73_i1 < 16; ++v73_i1) {
              int32_t v74_a = v72_i0 + v73_i1;
              float v76_data = r2[(v72_i0 + v73_i1)];
              glb_m0[(v81_lead + (v73_i1 * 16))] = v76_data;
            }
          }
        }
      }
    }
  }
}

