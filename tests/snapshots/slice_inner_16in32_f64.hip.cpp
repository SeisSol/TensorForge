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
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v11_off = (v3_lead + v9_lead) + 8;
            int32_t v19_off = (v3_lead + v9_lead) + 8;
            #pragma unroll
            for (int32_t v5_i1 = 8; v5_i1 < 24; ++v5_i1) {
              int32_t v12_a = v5_i1 * 32;
              int32_t v13_a = v11_off + v12_a;
              double v22_data = __builtin_nontemporal_load(&glb_m1[(v19_off + v12_a)]);
              int32_t v24_a = v4_i0 + (v5_i1 - 8);
              r0[v24_a] = v22_data;
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
          double v27_data = r0[0];
          double v28_data = r0[1];
          double v29_data = r0[2];
          double v30_data = r0[3];
          double v31_data = r0[4];
          double v32_data = r0[5];
          double v33_data = r0[6];
          double v34_data = r0[7];
          double v35_data = r0[8];
          double v36_data = r0[9];
          double v37_data = r0[10];
          double v38_data = r0[11];
          double v39_data = r0[12];
          double v40_data = r0[13];
          double v41_data = r0[14];
          double v42_data = r0[15];
          double v43_acc{};
          double v44_acc{};
          double v45_acc{};
          double v46_acc{};
          double v47_acc{};
          double v48_acc{};
          double v49_acc{};
          double v50_acc{};
          double v51_data = r1[0];
          double v52_data = r1[1];
          double v53_data = r1[2];
          double v54_data = r1[3];
          double v55_data = r1[4];
          double v56_data = r1[5];
          double v57_data = r1[6];
          double v58_data = r1[7];
          tensorforge::fmacdpp16<0>(v43_acc, v51_data, v27_data);
          tensorforge::fmacdpp16<1>(v43_acc, v51_data, v28_data);
          tensorforge::fmacdpp16<2>(v43_acc, v51_data, v29_data);
          tensorforge::fmacdpp16<3>(v43_acc, v51_data, v30_data);
          tensorforge::fmacdpp16<4>(v43_acc, v51_data, v31_data);
          tensorforge::fmacdpp16<5>(v43_acc, v51_data, v32_data);
          tensorforge::fmacdpp16<6>(v43_acc, v51_data, v33_data);
          tensorforge::fmacdpp16<7>(v43_acc, v51_data, v34_data);
          tensorforge::fmacdpp16<8>(v43_acc, v51_data, v35_data);
          tensorforge::fmacdpp16<9>(v43_acc, v51_data, v36_data);
          tensorforge::fmacdpp16<10>(v43_acc, v51_data, v37_data);
          tensorforge::fmacdpp16<11>(v43_acc, v51_data, v38_data);
          tensorforge::fmacdpp16<12>(v43_acc, v51_data, v39_data);
          tensorforge::fmacdpp16<13>(v43_acc, v51_data, v40_data);
          tensorforge::fmacdpp16<14>(v43_acc, v51_data, v41_data);
          tensorforge::fmacdpp16<15>(v43_acc, v51_data, v42_data);
          tensorforge::fmacdpp16<0>(v44_acc, v52_data, v27_data);
          tensorforge::fmacdpp16<1>(v44_acc, v52_data, v28_data);
          tensorforge::fmacdpp16<2>(v44_acc, v52_data, v29_data);
          tensorforge::fmacdpp16<3>(v44_acc, v52_data, v30_data);
          tensorforge::fmacdpp16<4>(v44_acc, v52_data, v31_data);
          tensorforge::fmacdpp16<5>(v44_acc, v52_data, v32_data);
          tensorforge::fmacdpp16<6>(v44_acc, v52_data, v33_data);
          tensorforge::fmacdpp16<7>(v44_acc, v52_data, v34_data);
          tensorforge::fmacdpp16<8>(v44_acc, v52_data, v35_data);
          tensorforge::fmacdpp16<9>(v44_acc, v52_data, v36_data);
          tensorforge::fmacdpp16<10>(v44_acc, v52_data, v37_data);
          tensorforge::fmacdpp16<11>(v44_acc, v52_data, v38_data);
          tensorforge::fmacdpp16<12>(v44_acc, v52_data, v39_data);
          tensorforge::fmacdpp16<13>(v44_acc, v52_data, v40_data);
          tensorforge::fmacdpp16<14>(v44_acc, v52_data, v41_data);
          tensorforge::fmacdpp16<15>(v44_acc, v52_data, v42_data);
          tensorforge::fmacdpp16<0>(v45_acc, v53_data, v27_data);
          tensorforge::fmacdpp16<1>(v45_acc, v53_data, v28_data);
          tensorforge::fmacdpp16<2>(v45_acc, v53_data, v29_data);
          tensorforge::fmacdpp16<3>(v45_acc, v53_data, v30_data);
          tensorforge::fmacdpp16<4>(v45_acc, v53_data, v31_data);
          tensorforge::fmacdpp16<5>(v45_acc, v53_data, v32_data);
          tensorforge::fmacdpp16<6>(v45_acc, v53_data, v33_data);
          tensorforge::fmacdpp16<7>(v45_acc, v53_data, v34_data);
          tensorforge::fmacdpp16<8>(v45_acc, v53_data, v35_data);
          tensorforge::fmacdpp16<9>(v45_acc, v53_data, v36_data);
          tensorforge::fmacdpp16<10>(v45_acc, v53_data, v37_data);
          tensorforge::fmacdpp16<11>(v45_acc, v53_data, v38_data);
          tensorforge::fmacdpp16<12>(v45_acc, v53_data, v39_data);
          tensorforge::fmacdpp16<13>(v45_acc, v53_data, v40_data);
          tensorforge::fmacdpp16<14>(v45_acc, v53_data, v41_data);
          tensorforge::fmacdpp16<15>(v45_acc, v53_data, v42_data);
          tensorforge::fmacdpp16<0>(v46_acc, v54_data, v27_data);
          tensorforge::fmacdpp16<1>(v46_acc, v54_data, v28_data);
          tensorforge::fmacdpp16<2>(v46_acc, v54_data, v29_data);
          tensorforge::fmacdpp16<3>(v46_acc, v54_data, v30_data);
          tensorforge::fmacdpp16<4>(v46_acc, v54_data, v31_data);
          tensorforge::fmacdpp16<5>(v46_acc, v54_data, v32_data);
          tensorforge::fmacdpp16<6>(v46_acc, v54_data, v33_data);
          tensorforge::fmacdpp16<7>(v46_acc, v54_data, v34_data);
          tensorforge::fmacdpp16<8>(v46_acc, v54_data, v35_data);
          tensorforge::fmacdpp16<9>(v46_acc, v54_data, v36_data);
          tensorforge::fmacdpp16<10>(v46_acc, v54_data, v37_data);
          tensorforge::fmacdpp16<11>(v46_acc, v54_data, v38_data);
          tensorforge::fmacdpp16<12>(v46_acc, v54_data, v39_data);
          tensorforge::fmacdpp16<13>(v46_acc, v54_data, v40_data);
          tensorforge::fmacdpp16<14>(v46_acc, v54_data, v41_data);
          tensorforge::fmacdpp16<15>(v46_acc, v54_data, v42_data);
          tensorforge::fmacdpp16<0>(v47_acc, v55_data, v27_data);
          tensorforge::fmacdpp16<1>(v47_acc, v55_data, v28_data);
          tensorforge::fmacdpp16<2>(v47_acc, v55_data, v29_data);
          tensorforge::fmacdpp16<3>(v47_acc, v55_data, v30_data);
          tensorforge::fmacdpp16<4>(v47_acc, v55_data, v31_data);
          tensorforge::fmacdpp16<5>(v47_acc, v55_data, v32_data);
          tensorforge::fmacdpp16<6>(v47_acc, v55_data, v33_data);
          tensorforge::fmacdpp16<7>(v47_acc, v55_data, v34_data);
          tensorforge::fmacdpp16<8>(v47_acc, v55_data, v35_data);
          tensorforge::fmacdpp16<9>(v47_acc, v55_data, v36_data);
          tensorforge::fmacdpp16<10>(v47_acc, v55_data, v37_data);
          tensorforge::fmacdpp16<11>(v47_acc, v55_data, v38_data);
          tensorforge::fmacdpp16<12>(v47_acc, v55_data, v39_data);
          tensorforge::fmacdpp16<13>(v47_acc, v55_data, v40_data);
          tensorforge::fmacdpp16<14>(v47_acc, v55_data, v41_data);
          tensorforge::fmacdpp16<15>(v47_acc, v55_data, v42_data);
          tensorforge::fmacdpp16<0>(v48_acc, v56_data, v27_data);
          tensorforge::fmacdpp16<1>(v48_acc, v56_data, v28_data);
          tensorforge::fmacdpp16<2>(v48_acc, v56_data, v29_data);
          tensorforge::fmacdpp16<3>(v48_acc, v56_data, v30_data);
          tensorforge::fmacdpp16<4>(v48_acc, v56_data, v31_data);
          tensorforge::fmacdpp16<5>(v48_acc, v56_data, v32_data);
          tensorforge::fmacdpp16<6>(v48_acc, v56_data, v33_data);
          tensorforge::fmacdpp16<7>(v48_acc, v56_data, v34_data);
          tensorforge::fmacdpp16<8>(v48_acc, v56_data, v35_data);
          tensorforge::fmacdpp16<9>(v48_acc, v56_data, v36_data);
          tensorforge::fmacdpp16<10>(v48_acc, v56_data, v37_data);
          tensorforge::fmacdpp16<11>(v48_acc, v56_data, v38_data);
          tensorforge::fmacdpp16<12>(v48_acc, v56_data, v39_data);
          tensorforge::fmacdpp16<13>(v48_acc, v56_data, v40_data);
          tensorforge::fmacdpp16<14>(v48_acc, v56_data, v41_data);
          tensorforge::fmacdpp16<15>(v48_acc, v56_data, v42_data);
          tensorforge::fmacdpp16<0>(v49_acc, v57_data, v27_data);
          tensorforge::fmacdpp16<1>(v49_acc, v57_data, v28_data);
          tensorforge::fmacdpp16<2>(v49_acc, v57_data, v29_data);
          tensorforge::fmacdpp16<3>(v49_acc, v57_data, v30_data);
          tensorforge::fmacdpp16<4>(v49_acc, v57_data, v31_data);
          tensorforge::fmacdpp16<5>(v49_acc, v57_data, v32_data);
          tensorforge::fmacdpp16<6>(v49_acc, v57_data, v33_data);
          tensorforge::fmacdpp16<7>(v49_acc, v57_data, v34_data);
          tensorforge::fmacdpp16<8>(v49_acc, v57_data, v35_data);
          tensorforge::fmacdpp16<9>(v49_acc, v57_data, v36_data);
          tensorforge::fmacdpp16<10>(v49_acc, v57_data, v37_data);
          tensorforge::fmacdpp16<11>(v49_acc, v57_data, v38_data);
          tensorforge::fmacdpp16<12>(v49_acc, v57_data, v39_data);
          tensorforge::fmacdpp16<13>(v49_acc, v57_data, v40_data);
          tensorforge::fmacdpp16<14>(v49_acc, v57_data, v41_data);
          tensorforge::fmacdpp16<15>(v49_acc, v57_data, v42_data);
          tensorforge::fmacdpp16<0>(v50_acc, v58_data, v27_data);
          tensorforge::fmacdpp16<1>(v50_acc, v58_data, v28_data);
          tensorforge::fmacdpp16<2>(v50_acc, v58_data, v29_data);
          tensorforge::fmacdpp16<3>(v50_acc, v58_data, v30_data);
          tensorforge::fmacdpp16<4>(v50_acc, v58_data, v31_data);
          tensorforge::fmacdpp16<5>(v50_acc, v58_data, v32_data);
          tensorforge::fmacdpp16<6>(v50_acc, v58_data, v33_data);
          tensorforge::fmacdpp16<7>(v50_acc, v58_data, v34_data);
          tensorforge::fmacdpp16<8>(v50_acc, v58_data, v35_data);
          tensorforge::fmacdpp16<9>(v50_acc, v58_data, v36_data);
          tensorforge::fmacdpp16<10>(v50_acc, v58_data, v37_data);
          tensorforge::fmacdpp16<11>(v50_acc, v58_data, v38_data);
          tensorforge::fmacdpp16<12>(v50_acc, v58_data, v39_data);
          tensorforge::fmacdpp16<13>(v50_acc, v58_data, v40_data);
          tensorforge::fmacdpp16<14>(v50_acc, v58_data, v41_data);
          tensorforge::fmacdpp16<15>(v50_acc, v58_data, v42_data);
          ir2[0] = v43_acc;
          ir2[1] = v44_acc;
          ir2[2] = v45_acc;
          ir2[3] = v46_acc;
          ir2[4] = v47_acc;
          ir2[5] = v48_acc;
          ir2[6] = v49_acc;
          ir2[7] = v50_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
            int32_t v71_lead = v3_lead + (v62_i0 * 16);
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 8; ++v63_i1) {
              int32_t v64_a = v62_i0 + v63_i1;
              double v66_data = r2[(v62_i0 + v63_i1)];
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

