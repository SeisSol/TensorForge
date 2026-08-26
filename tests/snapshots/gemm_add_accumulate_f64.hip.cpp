// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c847f49d, block.x * block.y * block.z, 256 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_16c847f49d), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_16c847f49d, grid, block, 256 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              double v19_data = __builtin_nontemporal_load(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          // [(0, 12), (0, 8)] [(0, 16)]
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
          double v45_data = r1[0];
          double v46_data = r1[1];
          double v47_data = r1[2];
          double v48_data = r1[3];
          double v49_data = r1[4];
          double v50_data = r1[5];
          double v51_data = r1[6];
          double v52_data = r1[7];
          tensorforge::fmacdpp16<0>(v37_acc, v45_data, v21_data);
          tensorforge::fmacdpp16<1>(v37_acc, v45_data, v22_data);
          tensorforge::fmacdpp16<2>(v37_acc, v45_data, v23_data);
          tensorforge::fmacdpp16<3>(v37_acc, v45_data, v24_data);
          tensorforge::fmacdpp16<4>(v37_acc, v45_data, v25_data);
          tensorforge::fmacdpp16<5>(v37_acc, v45_data, v26_data);
          tensorforge::fmacdpp16<6>(v37_acc, v45_data, v27_data);
          tensorforge::fmacdpp16<7>(v37_acc, v45_data, v28_data);
          tensorforge::fmacdpp16<8>(v37_acc, v45_data, v29_data);
          tensorforge::fmacdpp16<9>(v37_acc, v45_data, v30_data);
          tensorforge::fmacdpp16<10>(v37_acc, v45_data, v31_data);
          tensorforge::fmacdpp16<11>(v37_acc, v45_data, v32_data);
          tensorforge::fmacdpp16<12>(v37_acc, v45_data, v33_data);
          tensorforge::fmacdpp16<13>(v37_acc, v45_data, v34_data);
          tensorforge::fmacdpp16<14>(v37_acc, v45_data, v35_data);
          tensorforge::fmacdpp16<15>(v37_acc, v45_data, v36_data);
          tensorforge::fmacdpp16<0>(v38_acc, v46_data, v21_data);
          tensorforge::fmacdpp16<1>(v38_acc, v46_data, v22_data);
          tensorforge::fmacdpp16<2>(v38_acc, v46_data, v23_data);
          tensorforge::fmacdpp16<3>(v38_acc, v46_data, v24_data);
          tensorforge::fmacdpp16<4>(v38_acc, v46_data, v25_data);
          tensorforge::fmacdpp16<5>(v38_acc, v46_data, v26_data);
          tensorforge::fmacdpp16<6>(v38_acc, v46_data, v27_data);
          tensorforge::fmacdpp16<7>(v38_acc, v46_data, v28_data);
          tensorforge::fmacdpp16<8>(v38_acc, v46_data, v29_data);
          tensorforge::fmacdpp16<9>(v38_acc, v46_data, v30_data);
          tensorforge::fmacdpp16<10>(v38_acc, v46_data, v31_data);
          tensorforge::fmacdpp16<11>(v38_acc, v46_data, v32_data);
          tensorforge::fmacdpp16<12>(v38_acc, v46_data, v33_data);
          tensorforge::fmacdpp16<13>(v38_acc, v46_data, v34_data);
          tensorforge::fmacdpp16<14>(v38_acc, v46_data, v35_data);
          tensorforge::fmacdpp16<15>(v38_acc, v46_data, v36_data);
          tensorforge::fmacdpp16<0>(v39_acc, v47_data, v21_data);
          tensorforge::fmacdpp16<1>(v39_acc, v47_data, v22_data);
          tensorforge::fmacdpp16<2>(v39_acc, v47_data, v23_data);
          tensorforge::fmacdpp16<3>(v39_acc, v47_data, v24_data);
          tensorforge::fmacdpp16<4>(v39_acc, v47_data, v25_data);
          tensorforge::fmacdpp16<5>(v39_acc, v47_data, v26_data);
          tensorforge::fmacdpp16<6>(v39_acc, v47_data, v27_data);
          tensorforge::fmacdpp16<7>(v39_acc, v47_data, v28_data);
          tensorforge::fmacdpp16<8>(v39_acc, v47_data, v29_data);
          tensorforge::fmacdpp16<9>(v39_acc, v47_data, v30_data);
          tensorforge::fmacdpp16<10>(v39_acc, v47_data, v31_data);
          tensorforge::fmacdpp16<11>(v39_acc, v47_data, v32_data);
          tensorforge::fmacdpp16<12>(v39_acc, v47_data, v33_data);
          tensorforge::fmacdpp16<13>(v39_acc, v47_data, v34_data);
          tensorforge::fmacdpp16<14>(v39_acc, v47_data, v35_data);
          tensorforge::fmacdpp16<15>(v39_acc, v47_data, v36_data);
          tensorforge::fmacdpp16<0>(v40_acc, v48_data, v21_data);
          tensorforge::fmacdpp16<1>(v40_acc, v48_data, v22_data);
          tensorforge::fmacdpp16<2>(v40_acc, v48_data, v23_data);
          tensorforge::fmacdpp16<3>(v40_acc, v48_data, v24_data);
          tensorforge::fmacdpp16<4>(v40_acc, v48_data, v25_data);
          tensorforge::fmacdpp16<5>(v40_acc, v48_data, v26_data);
          tensorforge::fmacdpp16<6>(v40_acc, v48_data, v27_data);
          tensorforge::fmacdpp16<7>(v40_acc, v48_data, v28_data);
          tensorforge::fmacdpp16<8>(v40_acc, v48_data, v29_data);
          tensorforge::fmacdpp16<9>(v40_acc, v48_data, v30_data);
          tensorforge::fmacdpp16<10>(v40_acc, v48_data, v31_data);
          tensorforge::fmacdpp16<11>(v40_acc, v48_data, v32_data);
          tensorforge::fmacdpp16<12>(v40_acc, v48_data, v33_data);
          tensorforge::fmacdpp16<13>(v40_acc, v48_data, v34_data);
          tensorforge::fmacdpp16<14>(v40_acc, v48_data, v35_data);
          tensorforge::fmacdpp16<15>(v40_acc, v48_data, v36_data);
          tensorforge::fmacdpp16<0>(v41_acc, v49_data, v21_data);
          tensorforge::fmacdpp16<1>(v41_acc, v49_data, v22_data);
          tensorforge::fmacdpp16<2>(v41_acc, v49_data, v23_data);
          tensorforge::fmacdpp16<3>(v41_acc, v49_data, v24_data);
          tensorforge::fmacdpp16<4>(v41_acc, v49_data, v25_data);
          tensorforge::fmacdpp16<5>(v41_acc, v49_data, v26_data);
          tensorforge::fmacdpp16<6>(v41_acc, v49_data, v27_data);
          tensorforge::fmacdpp16<7>(v41_acc, v49_data, v28_data);
          tensorforge::fmacdpp16<8>(v41_acc, v49_data, v29_data);
          tensorforge::fmacdpp16<9>(v41_acc, v49_data, v30_data);
          tensorforge::fmacdpp16<10>(v41_acc, v49_data, v31_data);
          tensorforge::fmacdpp16<11>(v41_acc, v49_data, v32_data);
          tensorforge::fmacdpp16<12>(v41_acc, v49_data, v33_data);
          tensorforge::fmacdpp16<13>(v41_acc, v49_data, v34_data);
          tensorforge::fmacdpp16<14>(v41_acc, v49_data, v35_data);
          tensorforge::fmacdpp16<15>(v41_acc, v49_data, v36_data);
          tensorforge::fmacdpp16<0>(v42_acc, v50_data, v21_data);
          tensorforge::fmacdpp16<1>(v42_acc, v50_data, v22_data);
          tensorforge::fmacdpp16<2>(v42_acc, v50_data, v23_data);
          tensorforge::fmacdpp16<3>(v42_acc, v50_data, v24_data);
          tensorforge::fmacdpp16<4>(v42_acc, v50_data, v25_data);
          tensorforge::fmacdpp16<5>(v42_acc, v50_data, v26_data);
          tensorforge::fmacdpp16<6>(v42_acc, v50_data, v27_data);
          tensorforge::fmacdpp16<7>(v42_acc, v50_data, v28_data);
          tensorforge::fmacdpp16<8>(v42_acc, v50_data, v29_data);
          tensorforge::fmacdpp16<9>(v42_acc, v50_data, v30_data);
          tensorforge::fmacdpp16<10>(v42_acc, v50_data, v31_data);
          tensorforge::fmacdpp16<11>(v42_acc, v50_data, v32_data);
          tensorforge::fmacdpp16<12>(v42_acc, v50_data, v33_data);
          tensorforge::fmacdpp16<13>(v42_acc, v50_data, v34_data);
          tensorforge::fmacdpp16<14>(v42_acc, v50_data, v35_data);
          tensorforge::fmacdpp16<15>(v42_acc, v50_data, v36_data);
          tensorforge::fmacdpp16<0>(v43_acc, v51_data, v21_data);
          tensorforge::fmacdpp16<1>(v43_acc, v51_data, v22_data);
          tensorforge::fmacdpp16<2>(v43_acc, v51_data, v23_data);
          tensorforge::fmacdpp16<3>(v43_acc, v51_data, v24_data);
          tensorforge::fmacdpp16<4>(v43_acc, v51_data, v25_data);
          tensorforge::fmacdpp16<5>(v43_acc, v51_data, v26_data);
          tensorforge::fmacdpp16<6>(v43_acc, v51_data, v27_data);
          tensorforge::fmacdpp16<7>(v43_acc, v51_data, v28_data);
          tensorforge::fmacdpp16<8>(v43_acc, v51_data, v29_data);
          tensorforge::fmacdpp16<9>(v43_acc, v51_data, v30_data);
          tensorforge::fmacdpp16<10>(v43_acc, v51_data, v31_data);
          tensorforge::fmacdpp16<11>(v43_acc, v51_data, v32_data);
          tensorforge::fmacdpp16<12>(v43_acc, v51_data, v33_data);
          tensorforge::fmacdpp16<13>(v43_acc, v51_data, v34_data);
          tensorforge::fmacdpp16<14>(v43_acc, v51_data, v35_data);
          tensorforge::fmacdpp16<15>(v43_acc, v51_data, v36_data);
          tensorforge::fmacdpp16<0>(v44_acc, v52_data, v21_data);
          tensorforge::fmacdpp16<1>(v44_acc, v52_data, v22_data);
          tensorforge::fmacdpp16<2>(v44_acc, v52_data, v23_data);
          tensorforge::fmacdpp16<3>(v44_acc, v52_data, v24_data);
          tensorforge::fmacdpp16<4>(v44_acc, v52_data, v25_data);
          tensorforge::fmacdpp16<5>(v44_acc, v52_data, v26_data);
          tensorforge::fmacdpp16<6>(v44_acc, v52_data, v27_data);
          tensorforge::fmacdpp16<7>(v44_acc, v52_data, v28_data);
          tensorforge::fmacdpp16<8>(v44_acc, v52_data, v29_data);
          tensorforge::fmacdpp16<9>(v44_acc, v52_data, v30_data);
          tensorforge::fmacdpp16<10>(v44_acc, v52_data, v31_data);
          tensorforge::fmacdpp16<11>(v44_acc, v52_data, v32_data);
          tensorforge::fmacdpp16<12>(v44_acc, v52_data, v33_data);
          tensorforge::fmacdpp16<13>(v44_acc, v52_data, v34_data);
          tensorforge::fmacdpp16<14>(v44_acc, v52_data, v35_data);
          tensorforge::fmacdpp16<15>(v44_acc, v52_data, v36_data);
          ir2[0] = v37_acc;
          ir2[1] = v38_acc;
          ir2[2] = v39_acc;
          ir2[3] = v40_acc;
          ir2[4] = v41_acc;
          ir2[5] = v42_acc;
          ir2[6] = v43_acc;
          ir2[7] = v44_acc;
          // glb_m0 = store{r>g}(r2);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v57_i1 = 0; v57_i1 < 8; ++v57_i1) {
              int32_t v58_a = 0 + v57_i1;
              double v60_data = r2[v57_i1];
              int32_t v67_a = v2_lead + (v57_i1 * 12);
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m0[v67_a], v60_data);
            }
          }
          ;
        }
      }
    }
  }
}

