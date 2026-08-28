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
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 12;
              int32_t v16_a = v7_lead + v15_a;
              double v24_data = __builtin_nontemporal_load(&glb_m1[(v7_lead + v15_a)]);
              r0[v9_i1] = v24_data;
            }
          }
          double r1[8]{};
          // r1 = load{g>r}(glb_m2);
          double v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          double v28_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v28_lin;
          double v29_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v29_lin;
          double v30_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v30_lin;
          double v31_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v31_lin;
          double v32_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v32_lin;
          double v33_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v33_lin;
          double v34_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v34_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          double v36_data = r0[0];
          double v37_data = r0[1];
          double v38_data = r0[2];
          double v39_data = r0[3];
          double v40_data = r0[4];
          double v41_data = r0[5];
          double v42_data = r0[6];
          double v43_data = r0[7];
          double v44_data = r0[8];
          double v45_data = r0[9];
          double v46_data = r0[10];
          double v47_data = r0[11];
          double v48_data = r0[12];
          double v49_data = r0[13];
          double v50_data = r0[14];
          double v51_data = r0[15];
          double v52_acc{};
          double v53_acc{};
          double v54_acc{};
          double v55_acc{};
          double v56_acc{};
          double v57_acc{};
          double v58_acc{};
          double v59_acc{};
          double v60_data = r1[0];
          double v61_data = r1[1];
          double v62_data = r1[2];
          double v63_data = r1[3];
          double v64_data = r1[4];
          double v65_data = r1[5];
          double v66_data = r1[6];
          double v67_data = r1[7];
          tensorforge::fmacdpp16<0>(v52_acc, v60_data, v36_data);
          tensorforge::fmacdpp16<1>(v52_acc, v60_data, v37_data);
          tensorforge::fmacdpp16<2>(v52_acc, v60_data, v38_data);
          tensorforge::fmacdpp16<3>(v52_acc, v60_data, v39_data);
          tensorforge::fmacdpp16<4>(v52_acc, v60_data, v40_data);
          tensorforge::fmacdpp16<5>(v52_acc, v60_data, v41_data);
          tensorforge::fmacdpp16<6>(v52_acc, v60_data, v42_data);
          tensorforge::fmacdpp16<7>(v52_acc, v60_data, v43_data);
          tensorforge::fmacdpp16<8>(v52_acc, v60_data, v44_data);
          tensorforge::fmacdpp16<9>(v52_acc, v60_data, v45_data);
          tensorforge::fmacdpp16<10>(v52_acc, v60_data, v46_data);
          tensorforge::fmacdpp16<11>(v52_acc, v60_data, v47_data);
          tensorforge::fmacdpp16<12>(v52_acc, v60_data, v48_data);
          tensorforge::fmacdpp16<13>(v52_acc, v60_data, v49_data);
          tensorforge::fmacdpp16<14>(v52_acc, v60_data, v50_data);
          tensorforge::fmacdpp16<15>(v52_acc, v60_data, v51_data);
          tensorforge::fmacdpp16<0>(v53_acc, v61_data, v36_data);
          tensorforge::fmacdpp16<1>(v53_acc, v61_data, v37_data);
          tensorforge::fmacdpp16<2>(v53_acc, v61_data, v38_data);
          tensorforge::fmacdpp16<3>(v53_acc, v61_data, v39_data);
          tensorforge::fmacdpp16<4>(v53_acc, v61_data, v40_data);
          tensorforge::fmacdpp16<5>(v53_acc, v61_data, v41_data);
          tensorforge::fmacdpp16<6>(v53_acc, v61_data, v42_data);
          tensorforge::fmacdpp16<7>(v53_acc, v61_data, v43_data);
          tensorforge::fmacdpp16<8>(v53_acc, v61_data, v44_data);
          tensorforge::fmacdpp16<9>(v53_acc, v61_data, v45_data);
          tensorforge::fmacdpp16<10>(v53_acc, v61_data, v46_data);
          tensorforge::fmacdpp16<11>(v53_acc, v61_data, v47_data);
          tensorforge::fmacdpp16<12>(v53_acc, v61_data, v48_data);
          tensorforge::fmacdpp16<13>(v53_acc, v61_data, v49_data);
          tensorforge::fmacdpp16<14>(v53_acc, v61_data, v50_data);
          tensorforge::fmacdpp16<15>(v53_acc, v61_data, v51_data);
          tensorforge::fmacdpp16<0>(v54_acc, v62_data, v36_data);
          tensorforge::fmacdpp16<1>(v54_acc, v62_data, v37_data);
          tensorforge::fmacdpp16<2>(v54_acc, v62_data, v38_data);
          tensorforge::fmacdpp16<3>(v54_acc, v62_data, v39_data);
          tensorforge::fmacdpp16<4>(v54_acc, v62_data, v40_data);
          tensorforge::fmacdpp16<5>(v54_acc, v62_data, v41_data);
          tensorforge::fmacdpp16<6>(v54_acc, v62_data, v42_data);
          tensorforge::fmacdpp16<7>(v54_acc, v62_data, v43_data);
          tensorforge::fmacdpp16<8>(v54_acc, v62_data, v44_data);
          tensorforge::fmacdpp16<9>(v54_acc, v62_data, v45_data);
          tensorforge::fmacdpp16<10>(v54_acc, v62_data, v46_data);
          tensorforge::fmacdpp16<11>(v54_acc, v62_data, v47_data);
          tensorforge::fmacdpp16<12>(v54_acc, v62_data, v48_data);
          tensorforge::fmacdpp16<13>(v54_acc, v62_data, v49_data);
          tensorforge::fmacdpp16<14>(v54_acc, v62_data, v50_data);
          tensorforge::fmacdpp16<15>(v54_acc, v62_data, v51_data);
          tensorforge::fmacdpp16<0>(v55_acc, v63_data, v36_data);
          tensorforge::fmacdpp16<1>(v55_acc, v63_data, v37_data);
          tensorforge::fmacdpp16<2>(v55_acc, v63_data, v38_data);
          tensorforge::fmacdpp16<3>(v55_acc, v63_data, v39_data);
          tensorforge::fmacdpp16<4>(v55_acc, v63_data, v40_data);
          tensorforge::fmacdpp16<5>(v55_acc, v63_data, v41_data);
          tensorforge::fmacdpp16<6>(v55_acc, v63_data, v42_data);
          tensorforge::fmacdpp16<7>(v55_acc, v63_data, v43_data);
          tensorforge::fmacdpp16<8>(v55_acc, v63_data, v44_data);
          tensorforge::fmacdpp16<9>(v55_acc, v63_data, v45_data);
          tensorforge::fmacdpp16<10>(v55_acc, v63_data, v46_data);
          tensorforge::fmacdpp16<11>(v55_acc, v63_data, v47_data);
          tensorforge::fmacdpp16<12>(v55_acc, v63_data, v48_data);
          tensorforge::fmacdpp16<13>(v55_acc, v63_data, v49_data);
          tensorforge::fmacdpp16<14>(v55_acc, v63_data, v50_data);
          tensorforge::fmacdpp16<15>(v55_acc, v63_data, v51_data);
          tensorforge::fmacdpp16<0>(v56_acc, v64_data, v36_data);
          tensorforge::fmacdpp16<1>(v56_acc, v64_data, v37_data);
          tensorforge::fmacdpp16<2>(v56_acc, v64_data, v38_data);
          tensorforge::fmacdpp16<3>(v56_acc, v64_data, v39_data);
          tensorforge::fmacdpp16<4>(v56_acc, v64_data, v40_data);
          tensorforge::fmacdpp16<5>(v56_acc, v64_data, v41_data);
          tensorforge::fmacdpp16<6>(v56_acc, v64_data, v42_data);
          tensorforge::fmacdpp16<7>(v56_acc, v64_data, v43_data);
          tensorforge::fmacdpp16<8>(v56_acc, v64_data, v44_data);
          tensorforge::fmacdpp16<9>(v56_acc, v64_data, v45_data);
          tensorforge::fmacdpp16<10>(v56_acc, v64_data, v46_data);
          tensorforge::fmacdpp16<11>(v56_acc, v64_data, v47_data);
          tensorforge::fmacdpp16<12>(v56_acc, v64_data, v48_data);
          tensorforge::fmacdpp16<13>(v56_acc, v64_data, v49_data);
          tensorforge::fmacdpp16<14>(v56_acc, v64_data, v50_data);
          tensorforge::fmacdpp16<15>(v56_acc, v64_data, v51_data);
          tensorforge::fmacdpp16<0>(v57_acc, v65_data, v36_data);
          tensorforge::fmacdpp16<1>(v57_acc, v65_data, v37_data);
          tensorforge::fmacdpp16<2>(v57_acc, v65_data, v38_data);
          tensorforge::fmacdpp16<3>(v57_acc, v65_data, v39_data);
          tensorforge::fmacdpp16<4>(v57_acc, v65_data, v40_data);
          tensorforge::fmacdpp16<5>(v57_acc, v65_data, v41_data);
          tensorforge::fmacdpp16<6>(v57_acc, v65_data, v42_data);
          tensorforge::fmacdpp16<7>(v57_acc, v65_data, v43_data);
          tensorforge::fmacdpp16<8>(v57_acc, v65_data, v44_data);
          tensorforge::fmacdpp16<9>(v57_acc, v65_data, v45_data);
          tensorforge::fmacdpp16<10>(v57_acc, v65_data, v46_data);
          tensorforge::fmacdpp16<11>(v57_acc, v65_data, v47_data);
          tensorforge::fmacdpp16<12>(v57_acc, v65_data, v48_data);
          tensorforge::fmacdpp16<13>(v57_acc, v65_data, v49_data);
          tensorforge::fmacdpp16<14>(v57_acc, v65_data, v50_data);
          tensorforge::fmacdpp16<15>(v57_acc, v65_data, v51_data);
          tensorforge::fmacdpp16<0>(v58_acc, v66_data, v36_data);
          tensorforge::fmacdpp16<1>(v58_acc, v66_data, v37_data);
          tensorforge::fmacdpp16<2>(v58_acc, v66_data, v38_data);
          tensorforge::fmacdpp16<3>(v58_acc, v66_data, v39_data);
          tensorforge::fmacdpp16<4>(v58_acc, v66_data, v40_data);
          tensorforge::fmacdpp16<5>(v58_acc, v66_data, v41_data);
          tensorforge::fmacdpp16<6>(v58_acc, v66_data, v42_data);
          tensorforge::fmacdpp16<7>(v58_acc, v66_data, v43_data);
          tensorforge::fmacdpp16<8>(v58_acc, v66_data, v44_data);
          tensorforge::fmacdpp16<9>(v58_acc, v66_data, v45_data);
          tensorforge::fmacdpp16<10>(v58_acc, v66_data, v46_data);
          tensorforge::fmacdpp16<11>(v58_acc, v66_data, v47_data);
          tensorforge::fmacdpp16<12>(v58_acc, v66_data, v48_data);
          tensorforge::fmacdpp16<13>(v58_acc, v66_data, v49_data);
          tensorforge::fmacdpp16<14>(v58_acc, v66_data, v50_data);
          tensorforge::fmacdpp16<15>(v58_acc, v66_data, v51_data);
          tensorforge::fmacdpp16<0>(v59_acc, v67_data, v36_data);
          tensorforge::fmacdpp16<1>(v59_acc, v67_data, v37_data);
          tensorforge::fmacdpp16<2>(v59_acc, v67_data, v38_data);
          tensorforge::fmacdpp16<3>(v59_acc, v67_data, v39_data);
          tensorforge::fmacdpp16<4>(v59_acc, v67_data, v40_data);
          tensorforge::fmacdpp16<5>(v59_acc, v67_data, v41_data);
          tensorforge::fmacdpp16<6>(v59_acc, v67_data, v42_data);
          tensorforge::fmacdpp16<7>(v59_acc, v67_data, v43_data);
          tensorforge::fmacdpp16<8>(v59_acc, v67_data, v44_data);
          tensorforge::fmacdpp16<9>(v59_acc, v67_data, v45_data);
          tensorforge::fmacdpp16<10>(v59_acc, v67_data, v46_data);
          tensorforge::fmacdpp16<11>(v59_acc, v67_data, v47_data);
          tensorforge::fmacdpp16<12>(v59_acc, v67_data, v48_data);
          tensorforge::fmacdpp16<13>(v59_acc, v67_data, v49_data);
          tensorforge::fmacdpp16<14>(v59_acc, v67_data, v50_data);
          tensorforge::fmacdpp16<15>(v59_acc, v67_data, v51_data);
          r2[0] = v52_acc;
          r2[1] = v53_acc;
          r2[2] = v54_acc;
          r2[3] = v55_acc;
          r2[4] = v56_acc;
          r2[5] = v57_acc;
          r2[6] = v58_acc;
          r2[7] = v59_acc;
          // glb_m0 = store{r>g}(r2);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v72_i1 = 0; v72_i1 < 8; ++v72_i1) {
              int32_t v73_a = 0 + v72_i1;
              double v75_data = r2[v72_i1];
              int32_t v82_a = v7_lead + (v72_i1 * 12);
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m0[v82_a], v75_data);
            }
          }
        }
      }
    }
  }
}

