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
            int32_t v9_lead = v2_lead + (v3_i0 * 16);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 16);
              double v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
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
          double v14_data = r0[0];
          double v15_data = r0[1];
          double v16_data = r0[2];
          double v17_data = r0[3];
          double v18_data = r0[4];
          double v19_data = r0[5];
          double v20_data = r0[6];
          double v21_data = r0[7];
          double v22_data = r0[8];
          double v23_data = r0[9];
          double v24_data = r0[10];
          double v25_data = r0[11];
          double v26_data = r0[12];
          double v27_data = r0[13];
          double v28_data = r0[14];
          double v29_data = r0[15];
          double v30_acc{};
          double v31_acc{};
          double v32_acc{};
          double v33_acc{};
          double v34_acc{};
          double v35_acc{};
          double v36_acc{};
          double v37_acc{};
          double v38_acc{};
          double v39_acc{};
          double v40_acc{};
          double v41_acc{};
          double v42_acc{};
          double v43_acc{};
          double v44_acc{};
          double v45_acc{};
          double v46_lin = r1[0];
          tensorforge::fmacdpp16<0>(v30_acc, v46_lin, v14_data);
          tensorforge::fmacdpp16<1>(v30_acc, v46_lin, v15_data);
          tensorforge::fmacdpp16<2>(v31_acc, v46_lin, v14_data);
          tensorforge::fmacdpp16<3>(v31_acc, v46_lin, v15_data);
          tensorforge::fmacdpp16<4>(v31_acc, v46_lin, v16_data);
          tensorforge::fmacdpp16<5>(v32_acc, v46_lin, v15_data);
          tensorforge::fmacdpp16<6>(v32_acc, v46_lin, v16_data);
          tensorforge::fmacdpp16<7>(v32_acc, v46_lin, v17_data);
          tensorforge::fmacdpp16<8>(v33_acc, v46_lin, v16_data);
          tensorforge::fmacdpp16<9>(v33_acc, v46_lin, v17_data);
          tensorforge::fmacdpp16<10>(v33_acc, v46_lin, v18_data);
          tensorforge::fmacdpp16<11>(v34_acc, v46_lin, v17_data);
          tensorforge::fmacdpp16<12>(v34_acc, v46_lin, v18_data);
          tensorforge::fmacdpp16<13>(v34_acc, v46_lin, v19_data);
          tensorforge::fmacdpp16<14>(v35_acc, v46_lin, v18_data);
          tensorforge::fmacdpp16<15>(v35_acc, v46_lin, v19_data);
          double v47_lin = r1[1];
          tensorforge::fmacdpp16<0>(v35_acc, v47_lin, v20_data);
          tensorforge::fmacdpp16<1>(v36_acc, v47_lin, v19_data);
          tensorforge::fmacdpp16<2>(v36_acc, v47_lin, v20_data);
          tensorforge::fmacdpp16<3>(v36_acc, v47_lin, v21_data);
          tensorforge::fmacdpp16<4>(v37_acc, v47_lin, v20_data);
          tensorforge::fmacdpp16<5>(v37_acc, v47_lin, v21_data);
          tensorforge::fmacdpp16<6>(v37_acc, v47_lin, v22_data);
          tensorforge::fmacdpp16<7>(v38_acc, v47_lin, v21_data);
          tensorforge::fmacdpp16<8>(v38_acc, v47_lin, v22_data);
          tensorforge::fmacdpp16<9>(v38_acc, v47_lin, v23_data);
          tensorforge::fmacdpp16<10>(v39_acc, v47_lin, v22_data);
          tensorforge::fmacdpp16<11>(v39_acc, v47_lin, v23_data);
          tensorforge::fmacdpp16<12>(v39_acc, v47_lin, v24_data);
          tensorforge::fmacdpp16<13>(v40_acc, v47_lin, v23_data);
          tensorforge::fmacdpp16<14>(v40_acc, v47_lin, v24_data);
          tensorforge::fmacdpp16<15>(v40_acc, v47_lin, v25_data);
          double v48_lin = r1[2];
          tensorforge::fmacdpp16<0>(v41_acc, v48_lin, v24_data);
          tensorforge::fmacdpp16<1>(v41_acc, v48_lin, v25_data);
          tensorforge::fmacdpp16<2>(v41_acc, v48_lin, v26_data);
          tensorforge::fmacdpp16<3>(v42_acc, v48_lin, v25_data);
          tensorforge::fmacdpp16<4>(v42_acc, v48_lin, v26_data);
          tensorforge::fmacdpp16<5>(v42_acc, v48_lin, v27_data);
          tensorforge::fmacdpp16<6>(v43_acc, v48_lin, v26_data);
          tensorforge::fmacdpp16<7>(v43_acc, v48_lin, v27_data);
          tensorforge::fmacdpp16<8>(v43_acc, v48_lin, v28_data);
          tensorforge::fmacdpp16<9>(v44_acc, v48_lin, v27_data);
          tensorforge::fmacdpp16<10>(v44_acc, v48_lin, v28_data);
          tensorforge::fmacdpp16<11>(v44_acc, v48_lin, v29_data);
          tensorforge::fmacdpp16<12>(v45_acc, v48_lin, v28_data);
          tensorforge::fmacdpp16<13>(v45_acc, v48_lin, v29_data);
          ir2[0] = v30_acc;
          ir2[1] = v31_acc;
          ir2[2] = v32_acc;
          ir2[3] = v33_acc;
          ir2[4] = v34_acc;
          ir2[5] = v35_acc;
          ir2[6] = v36_acc;
          ir2[7] = v37_acc;
          ir2[8] = v38_acc;
          ir2[9] = v39_acc;
          ir2[10] = v40_acc;
          ir2[11] = v41_acc;
          ir2[12] = v42_acc;
          ir2[13] = v43_acc;
          ir2[14] = v44_acc;
          ir2[15] = v45_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v51_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v52_i0 = 0; v52_i0 < 1; ++v52_i0) {
            int32_t v60_lead = v51_lead + (v52_i0 * 16);
            #pragma unroll
            for (int32_t v53_i1 = 0; v53_i1 < 16; ++v53_i1) {
              int32_t v54_a = v52_i0 + v53_i1;
              double v55_data = r2[v54_a];
              int32_t v62_a = v60_lead + (v53_i1 * 16);
              glb_m0[v62_a] = v55_data;
            }
          }
          ;
        }
      }
    }
  }
}

