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
          double v53_data = r1[0];
          double v54_data = r1[1];
          double v55_data = r1[2];
          double v56_data = r1[3];
          double v57_data = r1[4];
          double v58_data = r1[5];
          double v59_data = r1[6];
          double v60_data = r1[7];
          double v61_data = r1[8];
          double v62_data = r1[9];
          double v63_data = r1[10];
          double v64_data = r1[11];
          double v65_data = r1[12];
          double v66_data = r1[13];
          double v67_data = r1[14];
          double v68_data = r1[15];
          tensorforge::fmacdpp16<0>(v37_acc, v53_data, v21_data);
          tensorforge::fmacdpp16<1>(v37_acc, v53_data, v22_data);
          tensorforge::fmacdpp16<2>(v37_acc, v53_data, v23_data);
          tensorforge::fmacdpp16<3>(v37_acc, v53_data, v24_data);
          tensorforge::fmacdpp16<4>(v37_acc, v53_data, v25_data);
          tensorforge::fmacdpp16<5>(v37_acc, v53_data, v26_data);
          tensorforge::fmacdpp16<6>(v37_acc, v53_data, v27_data);
          tensorforge::fmacdpp16<7>(v37_acc, v53_data, v28_data);
          tensorforge::fmacdpp16<8>(v37_acc, v53_data, v29_data);
          tensorforge::fmacdpp16<9>(v37_acc, v53_data, v30_data);
          tensorforge::fmacdpp16<10>(v37_acc, v53_data, v31_data);
          tensorforge::fmacdpp16<11>(v37_acc, v53_data, v32_data);
          tensorforge::fmacdpp16<12>(v37_acc, v53_data, v33_data);
          tensorforge::fmacdpp16<13>(v37_acc, v53_data, v34_data);
          tensorforge::fmacdpp16<14>(v37_acc, v53_data, v35_data);
          tensorforge::fmacdpp16<15>(v37_acc, v53_data, v36_data);
          tensorforge::fmacdpp16<0>(v38_acc, v54_data, v21_data);
          tensorforge::fmacdpp16<1>(v38_acc, v54_data, v22_data);
          tensorforge::fmacdpp16<2>(v38_acc, v54_data, v23_data);
          tensorforge::fmacdpp16<3>(v38_acc, v54_data, v24_data);
          tensorforge::fmacdpp16<4>(v38_acc, v54_data, v25_data);
          tensorforge::fmacdpp16<5>(v38_acc, v54_data, v26_data);
          tensorforge::fmacdpp16<6>(v38_acc, v54_data, v27_data);
          tensorforge::fmacdpp16<7>(v38_acc, v54_data, v28_data);
          tensorforge::fmacdpp16<8>(v38_acc, v54_data, v29_data);
          tensorforge::fmacdpp16<9>(v38_acc, v54_data, v30_data);
          tensorforge::fmacdpp16<10>(v38_acc, v54_data, v31_data);
          tensorforge::fmacdpp16<11>(v38_acc, v54_data, v32_data);
          tensorforge::fmacdpp16<12>(v38_acc, v54_data, v33_data);
          tensorforge::fmacdpp16<13>(v38_acc, v54_data, v34_data);
          tensorforge::fmacdpp16<14>(v38_acc, v54_data, v35_data);
          tensorforge::fmacdpp16<15>(v38_acc, v54_data, v36_data);
          tensorforge::fmacdpp16<0>(v39_acc, v55_data, v21_data);
          tensorforge::fmacdpp16<1>(v39_acc, v55_data, v22_data);
          tensorforge::fmacdpp16<2>(v39_acc, v55_data, v23_data);
          tensorforge::fmacdpp16<3>(v39_acc, v55_data, v24_data);
          tensorforge::fmacdpp16<4>(v39_acc, v55_data, v25_data);
          tensorforge::fmacdpp16<5>(v39_acc, v55_data, v26_data);
          tensorforge::fmacdpp16<6>(v39_acc, v55_data, v27_data);
          tensorforge::fmacdpp16<7>(v39_acc, v55_data, v28_data);
          tensorforge::fmacdpp16<8>(v39_acc, v55_data, v29_data);
          tensorforge::fmacdpp16<9>(v39_acc, v55_data, v30_data);
          tensorforge::fmacdpp16<10>(v39_acc, v55_data, v31_data);
          tensorforge::fmacdpp16<11>(v39_acc, v55_data, v32_data);
          tensorforge::fmacdpp16<12>(v39_acc, v55_data, v33_data);
          tensorforge::fmacdpp16<13>(v39_acc, v55_data, v34_data);
          tensorforge::fmacdpp16<14>(v39_acc, v55_data, v35_data);
          tensorforge::fmacdpp16<15>(v39_acc, v55_data, v36_data);
          tensorforge::fmacdpp16<0>(v40_acc, v56_data, v21_data);
          tensorforge::fmacdpp16<1>(v40_acc, v56_data, v22_data);
          tensorforge::fmacdpp16<2>(v40_acc, v56_data, v23_data);
          tensorforge::fmacdpp16<3>(v40_acc, v56_data, v24_data);
          tensorforge::fmacdpp16<4>(v40_acc, v56_data, v25_data);
          tensorforge::fmacdpp16<5>(v40_acc, v56_data, v26_data);
          tensorforge::fmacdpp16<6>(v40_acc, v56_data, v27_data);
          tensorforge::fmacdpp16<7>(v40_acc, v56_data, v28_data);
          tensorforge::fmacdpp16<8>(v40_acc, v56_data, v29_data);
          tensorforge::fmacdpp16<9>(v40_acc, v56_data, v30_data);
          tensorforge::fmacdpp16<10>(v40_acc, v56_data, v31_data);
          tensorforge::fmacdpp16<11>(v40_acc, v56_data, v32_data);
          tensorforge::fmacdpp16<12>(v40_acc, v56_data, v33_data);
          tensorforge::fmacdpp16<13>(v40_acc, v56_data, v34_data);
          tensorforge::fmacdpp16<14>(v40_acc, v56_data, v35_data);
          tensorforge::fmacdpp16<15>(v40_acc, v56_data, v36_data);
          tensorforge::fmacdpp16<0>(v41_acc, v57_data, v21_data);
          tensorforge::fmacdpp16<1>(v41_acc, v57_data, v22_data);
          tensorforge::fmacdpp16<2>(v41_acc, v57_data, v23_data);
          tensorforge::fmacdpp16<3>(v41_acc, v57_data, v24_data);
          tensorforge::fmacdpp16<4>(v41_acc, v57_data, v25_data);
          tensorforge::fmacdpp16<5>(v41_acc, v57_data, v26_data);
          tensorforge::fmacdpp16<6>(v41_acc, v57_data, v27_data);
          tensorforge::fmacdpp16<7>(v41_acc, v57_data, v28_data);
          tensorforge::fmacdpp16<8>(v41_acc, v57_data, v29_data);
          tensorforge::fmacdpp16<9>(v41_acc, v57_data, v30_data);
          tensorforge::fmacdpp16<10>(v41_acc, v57_data, v31_data);
          tensorforge::fmacdpp16<11>(v41_acc, v57_data, v32_data);
          tensorforge::fmacdpp16<12>(v41_acc, v57_data, v33_data);
          tensorforge::fmacdpp16<13>(v41_acc, v57_data, v34_data);
          tensorforge::fmacdpp16<14>(v41_acc, v57_data, v35_data);
          tensorforge::fmacdpp16<15>(v41_acc, v57_data, v36_data);
          tensorforge::fmacdpp16<0>(v42_acc, v58_data, v21_data);
          tensorforge::fmacdpp16<1>(v42_acc, v58_data, v22_data);
          tensorforge::fmacdpp16<2>(v42_acc, v58_data, v23_data);
          tensorforge::fmacdpp16<3>(v42_acc, v58_data, v24_data);
          tensorforge::fmacdpp16<4>(v42_acc, v58_data, v25_data);
          tensorforge::fmacdpp16<5>(v42_acc, v58_data, v26_data);
          tensorforge::fmacdpp16<6>(v42_acc, v58_data, v27_data);
          tensorforge::fmacdpp16<7>(v42_acc, v58_data, v28_data);
          tensorforge::fmacdpp16<8>(v42_acc, v58_data, v29_data);
          tensorforge::fmacdpp16<9>(v42_acc, v58_data, v30_data);
          tensorforge::fmacdpp16<10>(v42_acc, v58_data, v31_data);
          tensorforge::fmacdpp16<11>(v42_acc, v58_data, v32_data);
          tensorforge::fmacdpp16<12>(v42_acc, v58_data, v33_data);
          tensorforge::fmacdpp16<13>(v42_acc, v58_data, v34_data);
          tensorforge::fmacdpp16<14>(v42_acc, v58_data, v35_data);
          tensorforge::fmacdpp16<15>(v42_acc, v58_data, v36_data);
          tensorforge::fmacdpp16<0>(v43_acc, v59_data, v21_data);
          tensorforge::fmacdpp16<1>(v43_acc, v59_data, v22_data);
          tensorforge::fmacdpp16<2>(v43_acc, v59_data, v23_data);
          tensorforge::fmacdpp16<3>(v43_acc, v59_data, v24_data);
          tensorforge::fmacdpp16<4>(v43_acc, v59_data, v25_data);
          tensorforge::fmacdpp16<5>(v43_acc, v59_data, v26_data);
          tensorforge::fmacdpp16<6>(v43_acc, v59_data, v27_data);
          tensorforge::fmacdpp16<7>(v43_acc, v59_data, v28_data);
          tensorforge::fmacdpp16<8>(v43_acc, v59_data, v29_data);
          tensorforge::fmacdpp16<9>(v43_acc, v59_data, v30_data);
          tensorforge::fmacdpp16<10>(v43_acc, v59_data, v31_data);
          tensorforge::fmacdpp16<11>(v43_acc, v59_data, v32_data);
          tensorforge::fmacdpp16<12>(v43_acc, v59_data, v33_data);
          tensorforge::fmacdpp16<13>(v43_acc, v59_data, v34_data);
          tensorforge::fmacdpp16<14>(v43_acc, v59_data, v35_data);
          tensorforge::fmacdpp16<15>(v43_acc, v59_data, v36_data);
          tensorforge::fmacdpp16<0>(v44_acc, v60_data, v21_data);
          tensorforge::fmacdpp16<1>(v44_acc, v60_data, v22_data);
          tensorforge::fmacdpp16<2>(v44_acc, v60_data, v23_data);
          tensorforge::fmacdpp16<3>(v44_acc, v60_data, v24_data);
          tensorforge::fmacdpp16<4>(v44_acc, v60_data, v25_data);
          tensorforge::fmacdpp16<5>(v44_acc, v60_data, v26_data);
          tensorforge::fmacdpp16<6>(v44_acc, v60_data, v27_data);
          tensorforge::fmacdpp16<7>(v44_acc, v60_data, v28_data);
          tensorforge::fmacdpp16<8>(v44_acc, v60_data, v29_data);
          tensorforge::fmacdpp16<9>(v44_acc, v60_data, v30_data);
          tensorforge::fmacdpp16<10>(v44_acc, v60_data, v31_data);
          tensorforge::fmacdpp16<11>(v44_acc, v60_data, v32_data);
          tensorforge::fmacdpp16<12>(v44_acc, v60_data, v33_data);
          tensorforge::fmacdpp16<13>(v44_acc, v60_data, v34_data);
          tensorforge::fmacdpp16<14>(v44_acc, v60_data, v35_data);
          tensorforge::fmacdpp16<15>(v44_acc, v60_data, v36_data);
          tensorforge::fmacdpp16<0>(v45_acc, v61_data, v21_data);
          tensorforge::fmacdpp16<1>(v45_acc, v61_data, v22_data);
          tensorforge::fmacdpp16<2>(v45_acc, v61_data, v23_data);
          tensorforge::fmacdpp16<3>(v45_acc, v61_data, v24_data);
          tensorforge::fmacdpp16<4>(v45_acc, v61_data, v25_data);
          tensorforge::fmacdpp16<5>(v45_acc, v61_data, v26_data);
          tensorforge::fmacdpp16<6>(v45_acc, v61_data, v27_data);
          tensorforge::fmacdpp16<7>(v45_acc, v61_data, v28_data);
          tensorforge::fmacdpp16<8>(v45_acc, v61_data, v29_data);
          tensorforge::fmacdpp16<9>(v45_acc, v61_data, v30_data);
          tensorforge::fmacdpp16<10>(v45_acc, v61_data, v31_data);
          tensorforge::fmacdpp16<11>(v45_acc, v61_data, v32_data);
          tensorforge::fmacdpp16<12>(v45_acc, v61_data, v33_data);
          tensorforge::fmacdpp16<13>(v45_acc, v61_data, v34_data);
          tensorforge::fmacdpp16<14>(v45_acc, v61_data, v35_data);
          tensorforge::fmacdpp16<15>(v45_acc, v61_data, v36_data);
          tensorforge::fmacdpp16<0>(v46_acc, v62_data, v21_data);
          tensorforge::fmacdpp16<1>(v46_acc, v62_data, v22_data);
          tensorforge::fmacdpp16<2>(v46_acc, v62_data, v23_data);
          tensorforge::fmacdpp16<3>(v46_acc, v62_data, v24_data);
          tensorforge::fmacdpp16<4>(v46_acc, v62_data, v25_data);
          tensorforge::fmacdpp16<5>(v46_acc, v62_data, v26_data);
          tensorforge::fmacdpp16<6>(v46_acc, v62_data, v27_data);
          tensorforge::fmacdpp16<7>(v46_acc, v62_data, v28_data);
          tensorforge::fmacdpp16<8>(v46_acc, v62_data, v29_data);
          tensorforge::fmacdpp16<9>(v46_acc, v62_data, v30_data);
          tensorforge::fmacdpp16<10>(v46_acc, v62_data, v31_data);
          tensorforge::fmacdpp16<11>(v46_acc, v62_data, v32_data);
          tensorforge::fmacdpp16<12>(v46_acc, v62_data, v33_data);
          tensorforge::fmacdpp16<13>(v46_acc, v62_data, v34_data);
          tensorforge::fmacdpp16<14>(v46_acc, v62_data, v35_data);
          tensorforge::fmacdpp16<15>(v46_acc, v62_data, v36_data);
          tensorforge::fmacdpp16<0>(v47_acc, v63_data, v21_data);
          tensorforge::fmacdpp16<1>(v47_acc, v63_data, v22_data);
          tensorforge::fmacdpp16<2>(v47_acc, v63_data, v23_data);
          tensorforge::fmacdpp16<3>(v47_acc, v63_data, v24_data);
          tensorforge::fmacdpp16<4>(v47_acc, v63_data, v25_data);
          tensorforge::fmacdpp16<5>(v47_acc, v63_data, v26_data);
          tensorforge::fmacdpp16<6>(v47_acc, v63_data, v27_data);
          tensorforge::fmacdpp16<7>(v47_acc, v63_data, v28_data);
          tensorforge::fmacdpp16<8>(v47_acc, v63_data, v29_data);
          tensorforge::fmacdpp16<9>(v47_acc, v63_data, v30_data);
          tensorforge::fmacdpp16<10>(v47_acc, v63_data, v31_data);
          tensorforge::fmacdpp16<11>(v47_acc, v63_data, v32_data);
          tensorforge::fmacdpp16<12>(v47_acc, v63_data, v33_data);
          tensorforge::fmacdpp16<13>(v47_acc, v63_data, v34_data);
          tensorforge::fmacdpp16<14>(v47_acc, v63_data, v35_data);
          tensorforge::fmacdpp16<15>(v47_acc, v63_data, v36_data);
          tensorforge::fmacdpp16<0>(v48_acc, v64_data, v21_data);
          tensorforge::fmacdpp16<1>(v48_acc, v64_data, v22_data);
          tensorforge::fmacdpp16<2>(v48_acc, v64_data, v23_data);
          tensorforge::fmacdpp16<3>(v48_acc, v64_data, v24_data);
          tensorforge::fmacdpp16<4>(v48_acc, v64_data, v25_data);
          tensorforge::fmacdpp16<5>(v48_acc, v64_data, v26_data);
          tensorforge::fmacdpp16<6>(v48_acc, v64_data, v27_data);
          tensorforge::fmacdpp16<7>(v48_acc, v64_data, v28_data);
          tensorforge::fmacdpp16<8>(v48_acc, v64_data, v29_data);
          tensorforge::fmacdpp16<9>(v48_acc, v64_data, v30_data);
          tensorforge::fmacdpp16<10>(v48_acc, v64_data, v31_data);
          tensorforge::fmacdpp16<11>(v48_acc, v64_data, v32_data);
          tensorforge::fmacdpp16<12>(v48_acc, v64_data, v33_data);
          tensorforge::fmacdpp16<13>(v48_acc, v64_data, v34_data);
          tensorforge::fmacdpp16<14>(v48_acc, v64_data, v35_data);
          tensorforge::fmacdpp16<15>(v48_acc, v64_data, v36_data);
          tensorforge::fmacdpp16<0>(v49_acc, v65_data, v21_data);
          tensorforge::fmacdpp16<1>(v49_acc, v65_data, v22_data);
          tensorforge::fmacdpp16<2>(v49_acc, v65_data, v23_data);
          tensorforge::fmacdpp16<3>(v49_acc, v65_data, v24_data);
          tensorforge::fmacdpp16<4>(v49_acc, v65_data, v25_data);
          tensorforge::fmacdpp16<5>(v49_acc, v65_data, v26_data);
          tensorforge::fmacdpp16<6>(v49_acc, v65_data, v27_data);
          tensorforge::fmacdpp16<7>(v49_acc, v65_data, v28_data);
          tensorforge::fmacdpp16<8>(v49_acc, v65_data, v29_data);
          tensorforge::fmacdpp16<9>(v49_acc, v65_data, v30_data);
          tensorforge::fmacdpp16<10>(v49_acc, v65_data, v31_data);
          tensorforge::fmacdpp16<11>(v49_acc, v65_data, v32_data);
          tensorforge::fmacdpp16<12>(v49_acc, v65_data, v33_data);
          tensorforge::fmacdpp16<13>(v49_acc, v65_data, v34_data);
          tensorforge::fmacdpp16<14>(v49_acc, v65_data, v35_data);
          tensorforge::fmacdpp16<15>(v49_acc, v65_data, v36_data);
          tensorforge::fmacdpp16<0>(v50_acc, v66_data, v21_data);
          tensorforge::fmacdpp16<1>(v50_acc, v66_data, v22_data);
          tensorforge::fmacdpp16<2>(v50_acc, v66_data, v23_data);
          tensorforge::fmacdpp16<3>(v50_acc, v66_data, v24_data);
          tensorforge::fmacdpp16<4>(v50_acc, v66_data, v25_data);
          tensorforge::fmacdpp16<5>(v50_acc, v66_data, v26_data);
          tensorforge::fmacdpp16<6>(v50_acc, v66_data, v27_data);
          tensorforge::fmacdpp16<7>(v50_acc, v66_data, v28_data);
          tensorforge::fmacdpp16<8>(v50_acc, v66_data, v29_data);
          tensorforge::fmacdpp16<9>(v50_acc, v66_data, v30_data);
          tensorforge::fmacdpp16<10>(v50_acc, v66_data, v31_data);
          tensorforge::fmacdpp16<11>(v50_acc, v66_data, v32_data);
          tensorforge::fmacdpp16<12>(v50_acc, v66_data, v33_data);
          tensorforge::fmacdpp16<13>(v50_acc, v66_data, v34_data);
          tensorforge::fmacdpp16<14>(v50_acc, v66_data, v35_data);
          tensorforge::fmacdpp16<15>(v50_acc, v66_data, v36_data);
          tensorforge::fmacdpp16<0>(v51_acc, v67_data, v21_data);
          tensorforge::fmacdpp16<1>(v51_acc, v67_data, v22_data);
          tensorforge::fmacdpp16<2>(v51_acc, v67_data, v23_data);
          tensorforge::fmacdpp16<3>(v51_acc, v67_data, v24_data);
          tensorforge::fmacdpp16<4>(v51_acc, v67_data, v25_data);
          tensorforge::fmacdpp16<5>(v51_acc, v67_data, v26_data);
          tensorforge::fmacdpp16<6>(v51_acc, v67_data, v27_data);
          tensorforge::fmacdpp16<7>(v51_acc, v67_data, v28_data);
          tensorforge::fmacdpp16<8>(v51_acc, v67_data, v29_data);
          tensorforge::fmacdpp16<9>(v51_acc, v67_data, v30_data);
          tensorforge::fmacdpp16<10>(v51_acc, v67_data, v31_data);
          tensorforge::fmacdpp16<11>(v51_acc, v67_data, v32_data);
          tensorforge::fmacdpp16<12>(v51_acc, v67_data, v33_data);
          tensorforge::fmacdpp16<13>(v51_acc, v67_data, v34_data);
          tensorforge::fmacdpp16<14>(v51_acc, v67_data, v35_data);
          tensorforge::fmacdpp16<15>(v51_acc, v67_data, v36_data);
          tensorforge::fmacdpp16<0>(v52_acc, v68_data, v21_data);
          tensorforge::fmacdpp16<1>(v52_acc, v68_data, v22_data);
          tensorforge::fmacdpp16<2>(v52_acc, v68_data, v23_data);
          tensorforge::fmacdpp16<3>(v52_acc, v68_data, v24_data);
          tensorforge::fmacdpp16<4>(v52_acc, v68_data, v25_data);
          tensorforge::fmacdpp16<5>(v52_acc, v68_data, v26_data);
          tensorforge::fmacdpp16<6>(v52_acc, v68_data, v27_data);
          tensorforge::fmacdpp16<7>(v52_acc, v68_data, v28_data);
          tensorforge::fmacdpp16<8>(v52_acc, v68_data, v29_data);
          tensorforge::fmacdpp16<9>(v52_acc, v68_data, v30_data);
          tensorforge::fmacdpp16<10>(v52_acc, v68_data, v31_data);
          tensorforge::fmacdpp16<11>(v52_acc, v68_data, v32_data);
          tensorforge::fmacdpp16<12>(v52_acc, v68_data, v33_data);
          tensorforge::fmacdpp16<13>(v52_acc, v68_data, v34_data);
          tensorforge::fmacdpp16<14>(v52_acc, v68_data, v35_data);
          tensorforge::fmacdpp16<15>(v52_acc, v68_data, v36_data);
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
          for (int32_t v72_i0 = 0; v72_i0 < 1; ++v72_i0) {
            int32_t v81_lead = v2_lead + (v72_i0 * 16);
            #pragma unroll
            for (int32_t v73_i1 = 0; v73_i1 < 16; ++v73_i1) {
              int32_t v74_a = v72_i0 + v73_i1;
              double v76_data = r2[(v72_i0 + v73_i1)];
              int32_t v83_a = v81_lead + (v73_i1 * 16);
              glb_m0[v83_a] = v76_data;
            }
          }
          ;
        }
      }
    }
  }
}

