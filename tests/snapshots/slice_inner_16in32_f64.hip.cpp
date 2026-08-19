// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d37ccf0b0, block.x * block.y * block.z, 256 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3d37ccf0b0), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3d37ccf0b0, grid, block, 256 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v10_off = (v2_lead + (v3_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v12_a = v10_off + (v4_i1 * 32);
              double v13_data;
              {
                v13_data = __builtin_nontemporal_load(&glb_m1[v12_a]);
              }
              int32_t v15_a = v3_i0 + (v4_i1 - 8);
              r0[v15_a] = v13_data;
            }
          }
          double r1[8]{};
          {
            // r1 = load{g>r}(glb_m2);
            double v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            double v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            double v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
            double v48 = glb_m2[48 + threadIdx.x * 1];
            r1[3] = v48;
            double v64 = glb_m2[64 + threadIdx.x * 1];
            r1[4] = v64;
            double v80 = glb_m2[80 + threadIdx.x * 1];
            r1[5] = v80;
            double v96 = glb_m2[96 + threadIdx.x * 1];
            r1[6] = v96;
            double v112 = glb_m2[112 + threadIdx.x * 1];
            r1[7] = v112;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          auto& ir2 = r2;
          double v16_data = r0[0];
          double v17_data = r0[1];
          double v18_data = r0[2];
          double v19_data = r0[3];
          double v20_data = r0[4];
          double v21_data = r0[5];
          double v22_data = r0[6];
          double v23_data = r0[7];
          double v24_data = r0[8];
          double v25_data = r0[9];
          double v26_data = r0[10];
          double v27_data = r0[11];
          double v28_data = r0[12];
          double v29_data = r0[13];
          double v30_data = r0[14];
          double v31_data = r0[15];
          double v32_acc{};
          double v33_acc{};
          double v34_acc{};
          double v35_acc{};
          double v36_acc{};
          double v37_acc{};
          double v38_acc{};
          double v39_acc{};
          double v40_data = r1[0];
          double v41_data = r1[1];
          double v42_data = r1[2];
          double v43_data = r1[3];
          double v44_data = r1[4];
          double v45_data = r1[5];
          double v46_data = r1[6];
          double v47_data = r1[7];
          tensorforge::fmacdpp16<0>(v32_acc, v40_data, v16_data);
          tensorforge::fmacdpp16<1>(v32_acc, v40_data, v17_data);
          tensorforge::fmacdpp16<2>(v32_acc, v40_data, v18_data);
          tensorforge::fmacdpp16<3>(v32_acc, v40_data, v19_data);
          tensorforge::fmacdpp16<4>(v32_acc, v40_data, v20_data);
          tensorforge::fmacdpp16<5>(v32_acc, v40_data, v21_data);
          tensorforge::fmacdpp16<6>(v32_acc, v40_data, v22_data);
          tensorforge::fmacdpp16<7>(v32_acc, v40_data, v23_data);
          tensorforge::fmacdpp16<8>(v32_acc, v40_data, v24_data);
          tensorforge::fmacdpp16<9>(v32_acc, v40_data, v25_data);
          tensorforge::fmacdpp16<10>(v32_acc, v40_data, v26_data);
          tensorforge::fmacdpp16<11>(v32_acc, v40_data, v27_data);
          tensorforge::fmacdpp16<12>(v32_acc, v40_data, v28_data);
          tensorforge::fmacdpp16<13>(v32_acc, v40_data, v29_data);
          tensorforge::fmacdpp16<14>(v32_acc, v40_data, v30_data);
          tensorforge::fmacdpp16<15>(v32_acc, v40_data, v31_data);
          tensorforge::fmacdpp16<0>(v33_acc, v41_data, v16_data);
          tensorforge::fmacdpp16<1>(v33_acc, v41_data, v17_data);
          tensorforge::fmacdpp16<2>(v33_acc, v41_data, v18_data);
          tensorforge::fmacdpp16<3>(v33_acc, v41_data, v19_data);
          tensorforge::fmacdpp16<4>(v33_acc, v41_data, v20_data);
          tensorforge::fmacdpp16<5>(v33_acc, v41_data, v21_data);
          tensorforge::fmacdpp16<6>(v33_acc, v41_data, v22_data);
          tensorforge::fmacdpp16<7>(v33_acc, v41_data, v23_data);
          tensorforge::fmacdpp16<8>(v33_acc, v41_data, v24_data);
          tensorforge::fmacdpp16<9>(v33_acc, v41_data, v25_data);
          tensorforge::fmacdpp16<10>(v33_acc, v41_data, v26_data);
          tensorforge::fmacdpp16<11>(v33_acc, v41_data, v27_data);
          tensorforge::fmacdpp16<12>(v33_acc, v41_data, v28_data);
          tensorforge::fmacdpp16<13>(v33_acc, v41_data, v29_data);
          tensorforge::fmacdpp16<14>(v33_acc, v41_data, v30_data);
          tensorforge::fmacdpp16<15>(v33_acc, v41_data, v31_data);
          tensorforge::fmacdpp16<0>(v34_acc, v42_data, v16_data);
          tensorforge::fmacdpp16<1>(v34_acc, v42_data, v17_data);
          tensorforge::fmacdpp16<2>(v34_acc, v42_data, v18_data);
          tensorforge::fmacdpp16<3>(v34_acc, v42_data, v19_data);
          tensorforge::fmacdpp16<4>(v34_acc, v42_data, v20_data);
          tensorforge::fmacdpp16<5>(v34_acc, v42_data, v21_data);
          tensorforge::fmacdpp16<6>(v34_acc, v42_data, v22_data);
          tensorforge::fmacdpp16<7>(v34_acc, v42_data, v23_data);
          tensorforge::fmacdpp16<8>(v34_acc, v42_data, v24_data);
          tensorforge::fmacdpp16<9>(v34_acc, v42_data, v25_data);
          tensorforge::fmacdpp16<10>(v34_acc, v42_data, v26_data);
          tensorforge::fmacdpp16<11>(v34_acc, v42_data, v27_data);
          tensorforge::fmacdpp16<12>(v34_acc, v42_data, v28_data);
          tensorforge::fmacdpp16<13>(v34_acc, v42_data, v29_data);
          tensorforge::fmacdpp16<14>(v34_acc, v42_data, v30_data);
          tensorforge::fmacdpp16<15>(v34_acc, v42_data, v31_data);
          tensorforge::fmacdpp16<0>(v35_acc, v43_data, v16_data);
          tensorforge::fmacdpp16<1>(v35_acc, v43_data, v17_data);
          tensorforge::fmacdpp16<2>(v35_acc, v43_data, v18_data);
          tensorforge::fmacdpp16<3>(v35_acc, v43_data, v19_data);
          tensorforge::fmacdpp16<4>(v35_acc, v43_data, v20_data);
          tensorforge::fmacdpp16<5>(v35_acc, v43_data, v21_data);
          tensorforge::fmacdpp16<6>(v35_acc, v43_data, v22_data);
          tensorforge::fmacdpp16<7>(v35_acc, v43_data, v23_data);
          tensorforge::fmacdpp16<8>(v35_acc, v43_data, v24_data);
          tensorforge::fmacdpp16<9>(v35_acc, v43_data, v25_data);
          tensorforge::fmacdpp16<10>(v35_acc, v43_data, v26_data);
          tensorforge::fmacdpp16<11>(v35_acc, v43_data, v27_data);
          tensorforge::fmacdpp16<12>(v35_acc, v43_data, v28_data);
          tensorforge::fmacdpp16<13>(v35_acc, v43_data, v29_data);
          tensorforge::fmacdpp16<14>(v35_acc, v43_data, v30_data);
          tensorforge::fmacdpp16<15>(v35_acc, v43_data, v31_data);
          tensorforge::fmacdpp16<0>(v36_acc, v44_data, v16_data);
          tensorforge::fmacdpp16<1>(v36_acc, v44_data, v17_data);
          tensorforge::fmacdpp16<2>(v36_acc, v44_data, v18_data);
          tensorforge::fmacdpp16<3>(v36_acc, v44_data, v19_data);
          tensorforge::fmacdpp16<4>(v36_acc, v44_data, v20_data);
          tensorforge::fmacdpp16<5>(v36_acc, v44_data, v21_data);
          tensorforge::fmacdpp16<6>(v36_acc, v44_data, v22_data);
          tensorforge::fmacdpp16<7>(v36_acc, v44_data, v23_data);
          tensorforge::fmacdpp16<8>(v36_acc, v44_data, v24_data);
          tensorforge::fmacdpp16<9>(v36_acc, v44_data, v25_data);
          tensorforge::fmacdpp16<10>(v36_acc, v44_data, v26_data);
          tensorforge::fmacdpp16<11>(v36_acc, v44_data, v27_data);
          tensorforge::fmacdpp16<12>(v36_acc, v44_data, v28_data);
          tensorforge::fmacdpp16<13>(v36_acc, v44_data, v29_data);
          tensorforge::fmacdpp16<14>(v36_acc, v44_data, v30_data);
          tensorforge::fmacdpp16<15>(v36_acc, v44_data, v31_data);
          tensorforge::fmacdpp16<0>(v37_acc, v45_data, v16_data);
          tensorforge::fmacdpp16<1>(v37_acc, v45_data, v17_data);
          tensorforge::fmacdpp16<2>(v37_acc, v45_data, v18_data);
          tensorforge::fmacdpp16<3>(v37_acc, v45_data, v19_data);
          tensorforge::fmacdpp16<4>(v37_acc, v45_data, v20_data);
          tensorforge::fmacdpp16<5>(v37_acc, v45_data, v21_data);
          tensorforge::fmacdpp16<6>(v37_acc, v45_data, v22_data);
          tensorforge::fmacdpp16<7>(v37_acc, v45_data, v23_data);
          tensorforge::fmacdpp16<8>(v37_acc, v45_data, v24_data);
          tensorforge::fmacdpp16<9>(v37_acc, v45_data, v25_data);
          tensorforge::fmacdpp16<10>(v37_acc, v45_data, v26_data);
          tensorforge::fmacdpp16<11>(v37_acc, v45_data, v27_data);
          tensorforge::fmacdpp16<12>(v37_acc, v45_data, v28_data);
          tensorforge::fmacdpp16<13>(v37_acc, v45_data, v29_data);
          tensorforge::fmacdpp16<14>(v37_acc, v45_data, v30_data);
          tensorforge::fmacdpp16<15>(v37_acc, v45_data, v31_data);
          tensorforge::fmacdpp16<0>(v38_acc, v46_data, v16_data);
          tensorforge::fmacdpp16<1>(v38_acc, v46_data, v17_data);
          tensorforge::fmacdpp16<2>(v38_acc, v46_data, v18_data);
          tensorforge::fmacdpp16<3>(v38_acc, v46_data, v19_data);
          tensorforge::fmacdpp16<4>(v38_acc, v46_data, v20_data);
          tensorforge::fmacdpp16<5>(v38_acc, v46_data, v21_data);
          tensorforge::fmacdpp16<6>(v38_acc, v46_data, v22_data);
          tensorforge::fmacdpp16<7>(v38_acc, v46_data, v23_data);
          tensorforge::fmacdpp16<8>(v38_acc, v46_data, v24_data);
          tensorforge::fmacdpp16<9>(v38_acc, v46_data, v25_data);
          tensorforge::fmacdpp16<10>(v38_acc, v46_data, v26_data);
          tensorforge::fmacdpp16<11>(v38_acc, v46_data, v27_data);
          tensorforge::fmacdpp16<12>(v38_acc, v46_data, v28_data);
          tensorforge::fmacdpp16<13>(v38_acc, v46_data, v29_data);
          tensorforge::fmacdpp16<14>(v38_acc, v46_data, v30_data);
          tensorforge::fmacdpp16<15>(v38_acc, v46_data, v31_data);
          tensorforge::fmacdpp16<0>(v39_acc, v47_data, v16_data);
          tensorforge::fmacdpp16<1>(v39_acc, v47_data, v17_data);
          tensorforge::fmacdpp16<2>(v39_acc, v47_data, v18_data);
          tensorforge::fmacdpp16<3>(v39_acc, v47_data, v19_data);
          tensorforge::fmacdpp16<4>(v39_acc, v47_data, v20_data);
          tensorforge::fmacdpp16<5>(v39_acc, v47_data, v21_data);
          tensorforge::fmacdpp16<6>(v39_acc, v47_data, v22_data);
          tensorforge::fmacdpp16<7>(v39_acc, v47_data, v23_data);
          tensorforge::fmacdpp16<8>(v39_acc, v47_data, v24_data);
          tensorforge::fmacdpp16<9>(v39_acc, v47_data, v25_data);
          tensorforge::fmacdpp16<10>(v39_acc, v47_data, v26_data);
          tensorforge::fmacdpp16<11>(v39_acc, v47_data, v27_data);
          tensorforge::fmacdpp16<12>(v39_acc, v47_data, v28_data);
          tensorforge::fmacdpp16<13>(v39_acc, v47_data, v29_data);
          tensorforge::fmacdpp16<14>(v39_acc, v47_data, v30_data);
          tensorforge::fmacdpp16<15>(v39_acc, v47_data, v31_data);
          ir2[0] = v32_acc;
          ir2[1] = v33_acc;
          ir2[2] = v34_acc;
          ir2[3] = v35_acc;
          ir2[4] = v36_acc;
          ir2[5] = v37_acc;
          ir2[6] = v38_acc;
          ir2[7] = v39_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v50_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v59_lead = v50_lead + (v51_i0 * 16);
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 8; ++v52_i1) {
              int32_t v53_a = v51_i0 + v52_i1;
              double v54_data = r2[v53_a];
              int32_t v61_a = v59_lead + (v52_i1 * 16);
              glb_m0[v61_a] = v54_data;
            }
          }
          ;
        }
      }
    }
  }
}

