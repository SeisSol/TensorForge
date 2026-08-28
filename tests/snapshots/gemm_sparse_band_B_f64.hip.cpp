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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
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
              double v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + v12_i1)] = v27_data;
            }
          }
          double r1[16]{};
          // r1 = load{g>r}(glb_m2);
          double v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          double v31_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v31_lin;
          double v32_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v34_data = r0[0];
          double v35_data = r0[1];
          double v36_data = r0[2];
          double v37_data = r0[3];
          double v38_data = r0[4];
          double v39_data = r0[5];
          double v40_data = r0[6];
          double v41_data = r0[7];
          double v42_data = r0[8];
          double v43_data = r0[9];
          double v44_data = r0[10];
          double v45_data = r0[11];
          double v46_data = r0[12];
          double v47_data = r0[13];
          double v48_data = r0[14];
          double v49_data = r0[15];
          double v50_acc{};
          double v51_acc{};
          double v52_acc{};
          double v53_acc{};
          double v54_acc{};
          double v55_acc{};
          double v56_acc{};
          double v57_acc{};
          double v58_acc{};
          double v59_acc{};
          double v60_acc{};
          double v61_acc{};
          double v62_acc{};
          double v63_acc{};
          double v64_acc{};
          double v65_acc{};
          double v66_lin = r1[0];
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
          double v67_lin = r1[1];
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
          double v68_lin = r1[2];
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
              double v76_data = r2[(v72_i0 + v73_i1)];
              glb_m0[(v81_lead + (v73_i1 * 16))] = v76_data;
            }
          }
        }
      }
    }
  }
}

