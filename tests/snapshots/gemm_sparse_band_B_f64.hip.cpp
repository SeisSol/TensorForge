// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_417e1ddcc4, block.x * block.y * block.z, 256 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_417e1ddcc4), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_417e1ddcc4, grid, block, 256 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
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
              double v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          double r1[16]{};
          {
            // r1 = load{g>r}(glb_m2);
            double v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            double v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            double v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          double v21_data = r0[0];
          double v22_data = r0[1];
          double v23_data = r0[2];
          double v24_data = r0[3];
          double v25_data = r0[4];
          double v26_data = r0[5];
          double v27_data = r0[6];
          double v28_data = r0[7];
          double v29_data = r0[8];
          double v30_data = r0[9];
          double v31_data = r0[10];
          double v32_data = r0[11];
          double v33_data = r0[12];
          double v34_data = r0[13];
          double v35_data = r0[14];
          double v36_data = r0[15];
          double v37_acc{};
          double v38_acc{};
          double v39_acc{};
          double v40_acc{};
          double v41_acc{};
          double v42_acc{};
          double v43_acc{};
          double v44_acc{};
          double v45_acc{};
          double v46_acc{};
          double v47_acc{};
          double v48_acc{};
          double v49_acc{};
          double v50_acc{};
          double v51_acc{};
          double v52_acc{};
          double v53_lin = r1[0];
          tensorforge::fmacdpp16<0>(v37_acc, v53_lin, v21_data);
          tensorforge::fmacdpp16<1>(v37_acc, v53_lin, v22_data);
          tensorforge::fmacdpp16<2>(v38_acc, v53_lin, v21_data);
          tensorforge::fmacdpp16<3>(v38_acc, v53_lin, v22_data);
          tensorforge::fmacdpp16<4>(v38_acc, v53_lin, v23_data);
          tensorforge::fmacdpp16<5>(v39_acc, v53_lin, v22_data);
          tensorforge::fmacdpp16<6>(v39_acc, v53_lin, v23_data);
          tensorforge::fmacdpp16<7>(v39_acc, v53_lin, v24_data);
          tensorforge::fmacdpp16<8>(v40_acc, v53_lin, v23_data);
          tensorforge::fmacdpp16<9>(v40_acc, v53_lin, v24_data);
          tensorforge::fmacdpp16<10>(v40_acc, v53_lin, v25_data);
          tensorforge::fmacdpp16<11>(v41_acc, v53_lin, v24_data);
          tensorforge::fmacdpp16<12>(v41_acc, v53_lin, v25_data);
          tensorforge::fmacdpp16<13>(v41_acc, v53_lin, v26_data);
          tensorforge::fmacdpp16<14>(v42_acc, v53_lin, v25_data);
          tensorforge::fmacdpp16<15>(v42_acc, v53_lin, v26_data);
          double v54_lin = r1[1];
          tensorforge::fmacdpp16<0>(v42_acc, v54_lin, v27_data);
          tensorforge::fmacdpp16<1>(v43_acc, v54_lin, v26_data);
          tensorforge::fmacdpp16<2>(v43_acc, v54_lin, v27_data);
          tensorforge::fmacdpp16<3>(v43_acc, v54_lin, v28_data);
          tensorforge::fmacdpp16<4>(v44_acc, v54_lin, v27_data);
          tensorforge::fmacdpp16<5>(v44_acc, v54_lin, v28_data);
          tensorforge::fmacdpp16<6>(v44_acc, v54_lin, v29_data);
          tensorforge::fmacdpp16<7>(v45_acc, v54_lin, v28_data);
          tensorforge::fmacdpp16<8>(v45_acc, v54_lin, v29_data);
          tensorforge::fmacdpp16<9>(v45_acc, v54_lin, v30_data);
          tensorforge::fmacdpp16<10>(v46_acc, v54_lin, v29_data);
          tensorforge::fmacdpp16<11>(v46_acc, v54_lin, v30_data);
          tensorforge::fmacdpp16<12>(v46_acc, v54_lin, v31_data);
          tensorforge::fmacdpp16<13>(v47_acc, v54_lin, v30_data);
          tensorforge::fmacdpp16<14>(v47_acc, v54_lin, v31_data);
          tensorforge::fmacdpp16<15>(v47_acc, v54_lin, v32_data);
          double v55_lin = r1[2];
          tensorforge::fmacdpp16<0>(v48_acc, v55_lin, v31_data);
          tensorforge::fmacdpp16<1>(v48_acc, v55_lin, v32_data);
          tensorforge::fmacdpp16<2>(v48_acc, v55_lin, v33_data);
          tensorforge::fmacdpp16<3>(v49_acc, v55_lin, v32_data);
          tensorforge::fmacdpp16<4>(v49_acc, v55_lin, v33_data);
          tensorforge::fmacdpp16<5>(v49_acc, v55_lin, v34_data);
          tensorforge::fmacdpp16<6>(v50_acc, v55_lin, v33_data);
          tensorforge::fmacdpp16<7>(v50_acc, v55_lin, v34_data);
          tensorforge::fmacdpp16<8>(v50_acc, v55_lin, v35_data);
          tensorforge::fmacdpp16<9>(v51_acc, v55_lin, v34_data);
          tensorforge::fmacdpp16<10>(v51_acc, v55_lin, v35_data);
          tensorforge::fmacdpp16<11>(v51_acc, v55_lin, v36_data);
          tensorforge::fmacdpp16<12>(v52_acc, v55_lin, v35_data);
          tensorforge::fmacdpp16<13>(v52_acc, v55_lin, v36_data);
          ir2[0] = v37_acc;
          ir2[1] = v38_acc;
          ir2[2] = v39_acc;
          ir2[3] = v40_acc;
          ir2[4] = v41_acc;
          ir2[5] = v42_acc;
          ir2[6] = v43_acc;
          ir2[7] = v44_acc;
          ir2[8] = v45_acc;
          ir2[9] = v46_acc;
          ir2[10] = v47_acc;
          ir2[11] = v48_acc;
          ir2[12] = v49_acc;
          ir2[13] = v50_acc;
          ir2[14] = v51_acc;
          ir2[15] = v52_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v59_i0 = 0; v59_i0 < 1; ++v59_i0) {
            int32_t v68_lead = v2_lead + (v59_i0 * 16);
            #pragma unroll
            for (int32_t v60_i1 = 0; v60_i1 < 16; ++v60_i1) {
              int32_t v61_a = v59_i0 + v60_i1;
              double v63_data = r2[(v59_i0 + v60_i1)];
              int32_t v70_a = v68_lead + (v60_i1 * 16);
              glb_m0[v70_a] = v63_data;
            }
          }
          ;
        }
      }
    }
  }
}

