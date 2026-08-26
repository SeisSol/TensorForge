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
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 16;
              int32_t v12_a = v10_lead + v11_a;
              double v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
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
            double v128 = glb_m2[128 + threadIdx.x * 1];
            r1[8] = v128;
            double v144 = glb_m2[144 + threadIdx.x * 1];
            r1[9] = v144;
            double v160 = glb_m2[160 + threadIdx.x * 1];
            r1[10] = v160;
            double v176 = glb_m2[176 + threadIdx.x * 1];
            r1[11] = v176;
            double v192 = glb_m2[192 + threadIdx.x * 1];
            r1[12] = v192;
            double v208 = glb_m2[208 + threadIdx.x * 1];
            r1[13] = v208;
            double v224 = glb_m2[224 + threadIdx.x * 1];
            r1[14] = v224;
            double v240 = glb_m2[240 + threadIdx.x * 1];
            r1[15] = v240;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          double v24_data = r0[0];
          double v25_data = r0[1];
          double v26_data = r0[2];
          double v27_data = r0[3];
          double v28_data = r0[4];
          double v29_data = r0[5];
          double v30_data = r0[6];
          double v31_data = r0[7];
          double v32_data = r0[8];
          double v33_data = r0[9];
          double v34_data = r0[10];
          double v35_data = r0[11];
          double v36_data = r0[12];
          double v37_data = r0[13];
          double v38_data = r0[14];
          double v39_data = r0[15];
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
          double v53_acc{};
          double v54_acc{};
          double v55_acc{};
          double v56_data = r1[0];
          double v57_data = r1[1];
          double v58_data = r1[2];
          double v59_data = r1[3];
          double v60_data = r1[4];
          double v61_data = r1[5];
          double v62_data = r1[6];
          double v63_data = r1[7];
          double v64_data = r1[8];
          double v65_data = r1[9];
          double v66_data = r1[10];
          double v67_data = r1[11];
          double v68_data = r1[12];
          double v69_data = r1[13];
          double v70_data = r1[14];
          double v71_data = r1[15];
          tensorforge::fmacdpp16<0>(v40_acc, v56_data, v24_data);
          tensorforge::fmacdpp16<1>(v40_acc, v56_data, v25_data);
          tensorforge::fmacdpp16<2>(v40_acc, v56_data, v26_data);
          tensorforge::fmacdpp16<3>(v40_acc, v56_data, v27_data);
          tensorforge::fmacdpp16<4>(v40_acc, v56_data, v28_data);
          tensorforge::fmacdpp16<5>(v40_acc, v56_data, v29_data);
          tensorforge::fmacdpp16<6>(v40_acc, v56_data, v30_data);
          tensorforge::fmacdpp16<7>(v40_acc, v56_data, v31_data);
          tensorforge::fmacdpp16<8>(v40_acc, v56_data, v32_data);
          tensorforge::fmacdpp16<9>(v40_acc, v56_data, v33_data);
          tensorforge::fmacdpp16<10>(v40_acc, v56_data, v34_data);
          tensorforge::fmacdpp16<11>(v40_acc, v56_data, v35_data);
          tensorforge::fmacdpp16<12>(v40_acc, v56_data, v36_data);
          tensorforge::fmacdpp16<13>(v40_acc, v56_data, v37_data);
          tensorforge::fmacdpp16<14>(v40_acc, v56_data, v38_data);
          tensorforge::fmacdpp16<15>(v40_acc, v56_data, v39_data);
          tensorforge::fmacdpp16<0>(v41_acc, v57_data, v24_data);
          tensorforge::fmacdpp16<1>(v41_acc, v57_data, v25_data);
          tensorforge::fmacdpp16<2>(v41_acc, v57_data, v26_data);
          tensorforge::fmacdpp16<3>(v41_acc, v57_data, v27_data);
          tensorforge::fmacdpp16<4>(v41_acc, v57_data, v28_data);
          tensorforge::fmacdpp16<5>(v41_acc, v57_data, v29_data);
          tensorforge::fmacdpp16<6>(v41_acc, v57_data, v30_data);
          tensorforge::fmacdpp16<7>(v41_acc, v57_data, v31_data);
          tensorforge::fmacdpp16<8>(v41_acc, v57_data, v32_data);
          tensorforge::fmacdpp16<9>(v41_acc, v57_data, v33_data);
          tensorforge::fmacdpp16<10>(v41_acc, v57_data, v34_data);
          tensorforge::fmacdpp16<11>(v41_acc, v57_data, v35_data);
          tensorforge::fmacdpp16<12>(v41_acc, v57_data, v36_data);
          tensorforge::fmacdpp16<13>(v41_acc, v57_data, v37_data);
          tensorforge::fmacdpp16<14>(v41_acc, v57_data, v38_data);
          tensorforge::fmacdpp16<15>(v41_acc, v57_data, v39_data);
          tensorforge::fmacdpp16<0>(v42_acc, v58_data, v24_data);
          tensorforge::fmacdpp16<1>(v42_acc, v58_data, v25_data);
          tensorforge::fmacdpp16<2>(v42_acc, v58_data, v26_data);
          tensorforge::fmacdpp16<3>(v42_acc, v58_data, v27_data);
          tensorforge::fmacdpp16<4>(v42_acc, v58_data, v28_data);
          tensorforge::fmacdpp16<5>(v42_acc, v58_data, v29_data);
          tensorforge::fmacdpp16<6>(v42_acc, v58_data, v30_data);
          tensorforge::fmacdpp16<7>(v42_acc, v58_data, v31_data);
          tensorforge::fmacdpp16<8>(v42_acc, v58_data, v32_data);
          tensorforge::fmacdpp16<9>(v42_acc, v58_data, v33_data);
          tensorforge::fmacdpp16<10>(v42_acc, v58_data, v34_data);
          tensorforge::fmacdpp16<11>(v42_acc, v58_data, v35_data);
          tensorforge::fmacdpp16<12>(v42_acc, v58_data, v36_data);
          tensorforge::fmacdpp16<13>(v42_acc, v58_data, v37_data);
          tensorforge::fmacdpp16<14>(v42_acc, v58_data, v38_data);
          tensorforge::fmacdpp16<15>(v42_acc, v58_data, v39_data);
          tensorforge::fmacdpp16<0>(v43_acc, v59_data, v24_data);
          tensorforge::fmacdpp16<1>(v43_acc, v59_data, v25_data);
          tensorforge::fmacdpp16<2>(v43_acc, v59_data, v26_data);
          tensorforge::fmacdpp16<3>(v43_acc, v59_data, v27_data);
          tensorforge::fmacdpp16<4>(v43_acc, v59_data, v28_data);
          tensorforge::fmacdpp16<5>(v43_acc, v59_data, v29_data);
          tensorforge::fmacdpp16<6>(v43_acc, v59_data, v30_data);
          tensorforge::fmacdpp16<7>(v43_acc, v59_data, v31_data);
          tensorforge::fmacdpp16<8>(v43_acc, v59_data, v32_data);
          tensorforge::fmacdpp16<9>(v43_acc, v59_data, v33_data);
          tensorforge::fmacdpp16<10>(v43_acc, v59_data, v34_data);
          tensorforge::fmacdpp16<11>(v43_acc, v59_data, v35_data);
          tensorforge::fmacdpp16<12>(v43_acc, v59_data, v36_data);
          tensorforge::fmacdpp16<13>(v43_acc, v59_data, v37_data);
          tensorforge::fmacdpp16<14>(v43_acc, v59_data, v38_data);
          tensorforge::fmacdpp16<15>(v43_acc, v59_data, v39_data);
          tensorforge::fmacdpp16<0>(v44_acc, v60_data, v24_data);
          tensorforge::fmacdpp16<1>(v44_acc, v60_data, v25_data);
          tensorforge::fmacdpp16<2>(v44_acc, v60_data, v26_data);
          tensorforge::fmacdpp16<3>(v44_acc, v60_data, v27_data);
          tensorforge::fmacdpp16<4>(v44_acc, v60_data, v28_data);
          tensorforge::fmacdpp16<5>(v44_acc, v60_data, v29_data);
          tensorforge::fmacdpp16<6>(v44_acc, v60_data, v30_data);
          tensorforge::fmacdpp16<7>(v44_acc, v60_data, v31_data);
          tensorforge::fmacdpp16<8>(v44_acc, v60_data, v32_data);
          tensorforge::fmacdpp16<9>(v44_acc, v60_data, v33_data);
          tensorforge::fmacdpp16<10>(v44_acc, v60_data, v34_data);
          tensorforge::fmacdpp16<11>(v44_acc, v60_data, v35_data);
          tensorforge::fmacdpp16<12>(v44_acc, v60_data, v36_data);
          tensorforge::fmacdpp16<13>(v44_acc, v60_data, v37_data);
          tensorforge::fmacdpp16<14>(v44_acc, v60_data, v38_data);
          tensorforge::fmacdpp16<15>(v44_acc, v60_data, v39_data);
          tensorforge::fmacdpp16<0>(v45_acc, v61_data, v24_data);
          tensorforge::fmacdpp16<1>(v45_acc, v61_data, v25_data);
          tensorforge::fmacdpp16<2>(v45_acc, v61_data, v26_data);
          tensorforge::fmacdpp16<3>(v45_acc, v61_data, v27_data);
          tensorforge::fmacdpp16<4>(v45_acc, v61_data, v28_data);
          tensorforge::fmacdpp16<5>(v45_acc, v61_data, v29_data);
          tensorforge::fmacdpp16<6>(v45_acc, v61_data, v30_data);
          tensorforge::fmacdpp16<7>(v45_acc, v61_data, v31_data);
          tensorforge::fmacdpp16<8>(v45_acc, v61_data, v32_data);
          tensorforge::fmacdpp16<9>(v45_acc, v61_data, v33_data);
          tensorforge::fmacdpp16<10>(v45_acc, v61_data, v34_data);
          tensorforge::fmacdpp16<11>(v45_acc, v61_data, v35_data);
          tensorforge::fmacdpp16<12>(v45_acc, v61_data, v36_data);
          tensorforge::fmacdpp16<13>(v45_acc, v61_data, v37_data);
          tensorforge::fmacdpp16<14>(v45_acc, v61_data, v38_data);
          tensorforge::fmacdpp16<15>(v45_acc, v61_data, v39_data);
          tensorforge::fmacdpp16<0>(v46_acc, v62_data, v24_data);
          tensorforge::fmacdpp16<1>(v46_acc, v62_data, v25_data);
          tensorforge::fmacdpp16<2>(v46_acc, v62_data, v26_data);
          tensorforge::fmacdpp16<3>(v46_acc, v62_data, v27_data);
          tensorforge::fmacdpp16<4>(v46_acc, v62_data, v28_data);
          tensorforge::fmacdpp16<5>(v46_acc, v62_data, v29_data);
          tensorforge::fmacdpp16<6>(v46_acc, v62_data, v30_data);
          tensorforge::fmacdpp16<7>(v46_acc, v62_data, v31_data);
          tensorforge::fmacdpp16<8>(v46_acc, v62_data, v32_data);
          tensorforge::fmacdpp16<9>(v46_acc, v62_data, v33_data);
          tensorforge::fmacdpp16<10>(v46_acc, v62_data, v34_data);
          tensorforge::fmacdpp16<11>(v46_acc, v62_data, v35_data);
          tensorforge::fmacdpp16<12>(v46_acc, v62_data, v36_data);
          tensorforge::fmacdpp16<13>(v46_acc, v62_data, v37_data);
          tensorforge::fmacdpp16<14>(v46_acc, v62_data, v38_data);
          tensorforge::fmacdpp16<15>(v46_acc, v62_data, v39_data);
          tensorforge::fmacdpp16<0>(v47_acc, v63_data, v24_data);
          tensorforge::fmacdpp16<1>(v47_acc, v63_data, v25_data);
          tensorforge::fmacdpp16<2>(v47_acc, v63_data, v26_data);
          tensorforge::fmacdpp16<3>(v47_acc, v63_data, v27_data);
          tensorforge::fmacdpp16<4>(v47_acc, v63_data, v28_data);
          tensorforge::fmacdpp16<5>(v47_acc, v63_data, v29_data);
          tensorforge::fmacdpp16<6>(v47_acc, v63_data, v30_data);
          tensorforge::fmacdpp16<7>(v47_acc, v63_data, v31_data);
          tensorforge::fmacdpp16<8>(v47_acc, v63_data, v32_data);
          tensorforge::fmacdpp16<9>(v47_acc, v63_data, v33_data);
          tensorforge::fmacdpp16<10>(v47_acc, v63_data, v34_data);
          tensorforge::fmacdpp16<11>(v47_acc, v63_data, v35_data);
          tensorforge::fmacdpp16<12>(v47_acc, v63_data, v36_data);
          tensorforge::fmacdpp16<13>(v47_acc, v63_data, v37_data);
          tensorforge::fmacdpp16<14>(v47_acc, v63_data, v38_data);
          tensorforge::fmacdpp16<15>(v47_acc, v63_data, v39_data);
          tensorforge::fmacdpp16<0>(v48_acc, v64_data, v24_data);
          tensorforge::fmacdpp16<1>(v48_acc, v64_data, v25_data);
          tensorforge::fmacdpp16<2>(v48_acc, v64_data, v26_data);
          tensorforge::fmacdpp16<3>(v48_acc, v64_data, v27_data);
          tensorforge::fmacdpp16<4>(v48_acc, v64_data, v28_data);
          tensorforge::fmacdpp16<5>(v48_acc, v64_data, v29_data);
          tensorforge::fmacdpp16<6>(v48_acc, v64_data, v30_data);
          tensorforge::fmacdpp16<7>(v48_acc, v64_data, v31_data);
          tensorforge::fmacdpp16<8>(v48_acc, v64_data, v32_data);
          tensorforge::fmacdpp16<9>(v48_acc, v64_data, v33_data);
          tensorforge::fmacdpp16<10>(v48_acc, v64_data, v34_data);
          tensorforge::fmacdpp16<11>(v48_acc, v64_data, v35_data);
          tensorforge::fmacdpp16<12>(v48_acc, v64_data, v36_data);
          tensorforge::fmacdpp16<13>(v48_acc, v64_data, v37_data);
          tensorforge::fmacdpp16<14>(v48_acc, v64_data, v38_data);
          tensorforge::fmacdpp16<15>(v48_acc, v64_data, v39_data);
          tensorforge::fmacdpp16<0>(v49_acc, v65_data, v24_data);
          tensorforge::fmacdpp16<1>(v49_acc, v65_data, v25_data);
          tensorforge::fmacdpp16<2>(v49_acc, v65_data, v26_data);
          tensorforge::fmacdpp16<3>(v49_acc, v65_data, v27_data);
          tensorforge::fmacdpp16<4>(v49_acc, v65_data, v28_data);
          tensorforge::fmacdpp16<5>(v49_acc, v65_data, v29_data);
          tensorforge::fmacdpp16<6>(v49_acc, v65_data, v30_data);
          tensorforge::fmacdpp16<7>(v49_acc, v65_data, v31_data);
          tensorforge::fmacdpp16<8>(v49_acc, v65_data, v32_data);
          tensorforge::fmacdpp16<9>(v49_acc, v65_data, v33_data);
          tensorforge::fmacdpp16<10>(v49_acc, v65_data, v34_data);
          tensorforge::fmacdpp16<11>(v49_acc, v65_data, v35_data);
          tensorforge::fmacdpp16<12>(v49_acc, v65_data, v36_data);
          tensorforge::fmacdpp16<13>(v49_acc, v65_data, v37_data);
          tensorforge::fmacdpp16<14>(v49_acc, v65_data, v38_data);
          tensorforge::fmacdpp16<15>(v49_acc, v65_data, v39_data);
          tensorforge::fmacdpp16<0>(v50_acc, v66_data, v24_data);
          tensorforge::fmacdpp16<1>(v50_acc, v66_data, v25_data);
          tensorforge::fmacdpp16<2>(v50_acc, v66_data, v26_data);
          tensorforge::fmacdpp16<3>(v50_acc, v66_data, v27_data);
          tensorforge::fmacdpp16<4>(v50_acc, v66_data, v28_data);
          tensorforge::fmacdpp16<5>(v50_acc, v66_data, v29_data);
          tensorforge::fmacdpp16<6>(v50_acc, v66_data, v30_data);
          tensorforge::fmacdpp16<7>(v50_acc, v66_data, v31_data);
          tensorforge::fmacdpp16<8>(v50_acc, v66_data, v32_data);
          tensorforge::fmacdpp16<9>(v50_acc, v66_data, v33_data);
          tensorforge::fmacdpp16<10>(v50_acc, v66_data, v34_data);
          tensorforge::fmacdpp16<11>(v50_acc, v66_data, v35_data);
          tensorforge::fmacdpp16<12>(v50_acc, v66_data, v36_data);
          tensorforge::fmacdpp16<13>(v50_acc, v66_data, v37_data);
          tensorforge::fmacdpp16<14>(v50_acc, v66_data, v38_data);
          tensorforge::fmacdpp16<15>(v50_acc, v66_data, v39_data);
          tensorforge::fmacdpp16<0>(v51_acc, v67_data, v24_data);
          tensorforge::fmacdpp16<1>(v51_acc, v67_data, v25_data);
          tensorforge::fmacdpp16<2>(v51_acc, v67_data, v26_data);
          tensorforge::fmacdpp16<3>(v51_acc, v67_data, v27_data);
          tensorforge::fmacdpp16<4>(v51_acc, v67_data, v28_data);
          tensorforge::fmacdpp16<5>(v51_acc, v67_data, v29_data);
          tensorforge::fmacdpp16<6>(v51_acc, v67_data, v30_data);
          tensorforge::fmacdpp16<7>(v51_acc, v67_data, v31_data);
          tensorforge::fmacdpp16<8>(v51_acc, v67_data, v32_data);
          tensorforge::fmacdpp16<9>(v51_acc, v67_data, v33_data);
          tensorforge::fmacdpp16<10>(v51_acc, v67_data, v34_data);
          tensorforge::fmacdpp16<11>(v51_acc, v67_data, v35_data);
          tensorforge::fmacdpp16<12>(v51_acc, v67_data, v36_data);
          tensorforge::fmacdpp16<13>(v51_acc, v67_data, v37_data);
          tensorforge::fmacdpp16<14>(v51_acc, v67_data, v38_data);
          tensorforge::fmacdpp16<15>(v51_acc, v67_data, v39_data);
          tensorforge::fmacdpp16<0>(v52_acc, v68_data, v24_data);
          tensorforge::fmacdpp16<1>(v52_acc, v68_data, v25_data);
          tensorforge::fmacdpp16<2>(v52_acc, v68_data, v26_data);
          tensorforge::fmacdpp16<3>(v52_acc, v68_data, v27_data);
          tensorforge::fmacdpp16<4>(v52_acc, v68_data, v28_data);
          tensorforge::fmacdpp16<5>(v52_acc, v68_data, v29_data);
          tensorforge::fmacdpp16<6>(v52_acc, v68_data, v30_data);
          tensorforge::fmacdpp16<7>(v52_acc, v68_data, v31_data);
          tensorforge::fmacdpp16<8>(v52_acc, v68_data, v32_data);
          tensorforge::fmacdpp16<9>(v52_acc, v68_data, v33_data);
          tensorforge::fmacdpp16<10>(v52_acc, v68_data, v34_data);
          tensorforge::fmacdpp16<11>(v52_acc, v68_data, v35_data);
          tensorforge::fmacdpp16<12>(v52_acc, v68_data, v36_data);
          tensorforge::fmacdpp16<13>(v52_acc, v68_data, v37_data);
          tensorforge::fmacdpp16<14>(v52_acc, v68_data, v38_data);
          tensorforge::fmacdpp16<15>(v52_acc, v68_data, v39_data);
          tensorforge::fmacdpp16<0>(v53_acc, v69_data, v24_data);
          tensorforge::fmacdpp16<1>(v53_acc, v69_data, v25_data);
          tensorforge::fmacdpp16<2>(v53_acc, v69_data, v26_data);
          tensorforge::fmacdpp16<3>(v53_acc, v69_data, v27_data);
          tensorforge::fmacdpp16<4>(v53_acc, v69_data, v28_data);
          tensorforge::fmacdpp16<5>(v53_acc, v69_data, v29_data);
          tensorforge::fmacdpp16<6>(v53_acc, v69_data, v30_data);
          tensorforge::fmacdpp16<7>(v53_acc, v69_data, v31_data);
          tensorforge::fmacdpp16<8>(v53_acc, v69_data, v32_data);
          tensorforge::fmacdpp16<9>(v53_acc, v69_data, v33_data);
          tensorforge::fmacdpp16<10>(v53_acc, v69_data, v34_data);
          tensorforge::fmacdpp16<11>(v53_acc, v69_data, v35_data);
          tensorforge::fmacdpp16<12>(v53_acc, v69_data, v36_data);
          tensorforge::fmacdpp16<13>(v53_acc, v69_data, v37_data);
          tensorforge::fmacdpp16<14>(v53_acc, v69_data, v38_data);
          tensorforge::fmacdpp16<15>(v53_acc, v69_data, v39_data);
          tensorforge::fmacdpp16<0>(v54_acc, v70_data, v24_data);
          tensorforge::fmacdpp16<1>(v54_acc, v70_data, v25_data);
          tensorforge::fmacdpp16<2>(v54_acc, v70_data, v26_data);
          tensorforge::fmacdpp16<3>(v54_acc, v70_data, v27_data);
          tensorforge::fmacdpp16<4>(v54_acc, v70_data, v28_data);
          tensorforge::fmacdpp16<5>(v54_acc, v70_data, v29_data);
          tensorforge::fmacdpp16<6>(v54_acc, v70_data, v30_data);
          tensorforge::fmacdpp16<7>(v54_acc, v70_data, v31_data);
          tensorforge::fmacdpp16<8>(v54_acc, v70_data, v32_data);
          tensorforge::fmacdpp16<9>(v54_acc, v70_data, v33_data);
          tensorforge::fmacdpp16<10>(v54_acc, v70_data, v34_data);
          tensorforge::fmacdpp16<11>(v54_acc, v70_data, v35_data);
          tensorforge::fmacdpp16<12>(v54_acc, v70_data, v36_data);
          tensorforge::fmacdpp16<13>(v54_acc, v70_data, v37_data);
          tensorforge::fmacdpp16<14>(v54_acc, v70_data, v38_data);
          tensorforge::fmacdpp16<15>(v54_acc, v70_data, v39_data);
          tensorforge::fmacdpp16<0>(v55_acc, v71_data, v24_data);
          tensorforge::fmacdpp16<1>(v55_acc, v71_data, v25_data);
          tensorforge::fmacdpp16<2>(v55_acc, v71_data, v26_data);
          tensorforge::fmacdpp16<3>(v55_acc, v71_data, v27_data);
          tensorforge::fmacdpp16<4>(v55_acc, v71_data, v28_data);
          tensorforge::fmacdpp16<5>(v55_acc, v71_data, v29_data);
          tensorforge::fmacdpp16<6>(v55_acc, v71_data, v30_data);
          tensorforge::fmacdpp16<7>(v55_acc, v71_data, v31_data);
          tensorforge::fmacdpp16<8>(v55_acc, v71_data, v32_data);
          tensorforge::fmacdpp16<9>(v55_acc, v71_data, v33_data);
          tensorforge::fmacdpp16<10>(v55_acc, v71_data, v34_data);
          tensorforge::fmacdpp16<11>(v55_acc, v71_data, v35_data);
          tensorforge::fmacdpp16<12>(v55_acc, v71_data, v36_data);
          tensorforge::fmacdpp16<13>(v55_acc, v71_data, v37_data);
          tensorforge::fmacdpp16<14>(v55_acc, v71_data, v38_data);
          tensorforge::fmacdpp16<15>(v55_acc, v71_data, v39_data);
          ir2[0] = v40_acc;
          ir2[1] = v41_acc;
          ir2[2] = v42_acc;
          ir2[3] = v43_acc;
          ir2[4] = v44_acc;
          ir2[5] = v45_acc;
          ir2[6] = v46_acc;
          ir2[7] = v47_acc;
          ir2[8] = v48_acc;
          ir2[9] = v49_acc;
          ir2[10] = v50_acc;
          ir2[11] = v51_acc;
          ir2[12] = v52_acc;
          ir2[13] = v53_acc;
          ir2[14] = v54_acc;
          ir2[15] = v55_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v75_i0 = 0; v75_i0 < 1; ++v75_i0) {
            int32_t v84_lead = v3_lead + (v75_i0 * 16);
            #pragma unroll
            for (int32_t v76_i1 = 0; v76_i1 < 16; ++v76_i1) {
              int32_t v77_a = v75_i0 + v76_i1;
              double v79_data = r2[(v75_i0 + v76_i1)];
              int32_t v86_a = v84_lead + (v76_i1 * 16);
              glb_m0[v86_a] = v79_data;
            }
          }
          ;
        }
      }
    }
  }
}

