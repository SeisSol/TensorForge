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
          int32_t v13_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
            int32_t v20_lead = v13_lead + (v14_i0 * 16);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              double v23_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v15_i1 * 16))]);
              r0[(v14_i0 + v15_i1)] = v23_data;
            }
          }
          double r1[16]{};
          // r1 = load{g>r}(glb_m2);
          double v26_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v26_lin;
          double v27_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v27_lin;
          double v28_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v28_lin;
          double v29_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v29_lin;
          double v30_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v30_lin;
          double v31_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v31_lin;
          double v32_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v32_lin;
          double v33_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v33_lin;
          double v34_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v34_lin;
          double v35_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v35_lin;
          double v36_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v36_lin;
          double v37_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v37_lin;
          double v38_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v38_lin;
          double v39_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v39_lin;
          double v40_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v40_lin;
          double v41_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v41_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v43_data = r0[0];
          double v44_data = r0[1];
          double v45_data = r0[2];
          double v46_data = r0[3];
          double v47_data = r0[4];
          double v48_data = r0[5];
          double v49_data = r0[6];
          double v50_data = r0[7];
          double v51_data = r0[8];
          double v52_data = r0[9];
          double v53_data = r0[10];
          double v54_data = r0[11];
          double v55_data = r0[12];
          double v56_data = r0[13];
          double v57_data = r0[14];
          double v58_data = r0[15];
          double v59_acc{};
          double v60_acc{};
          double v61_acc{};
          double v62_acc{};
          double v63_acc{};
          double v64_acc{};
          double v65_acc{};
          double v66_acc{};
          double v67_acc{};
          double v68_acc{};
          double v69_acc{};
          double v70_acc{};
          double v71_acc{};
          double v72_acc{};
          double v73_acc{};
          double v74_acc{};
          double v75_data = r1[0];
          double v76_data = r1[1];
          double v77_data = r1[2];
          double v78_data = r1[3];
          double v79_data = r1[4];
          double v80_data = r1[5];
          double v81_data = r1[6];
          double v82_data = r1[7];
          double v83_data = r1[8];
          double v84_data = r1[9];
          double v85_data = r1[10];
          double v86_data = r1[11];
          double v87_data = r1[12];
          double v88_data = r1[13];
          double v89_data = r1[14];
          double v90_data = r1[15];
          tensorforge::fmacdpp16<0>(v59_acc, v75_data, v43_data);
          tensorforge::fmacdpp16<1>(v59_acc, v75_data, v44_data);
          tensorforge::fmacdpp16<2>(v59_acc, v75_data, v45_data);
          tensorforge::fmacdpp16<3>(v59_acc, v75_data, v46_data);
          tensorforge::fmacdpp16<4>(v59_acc, v75_data, v47_data);
          tensorforge::fmacdpp16<5>(v59_acc, v75_data, v48_data);
          tensorforge::fmacdpp16<6>(v59_acc, v75_data, v49_data);
          tensorforge::fmacdpp16<7>(v59_acc, v75_data, v50_data);
          tensorforge::fmacdpp16<8>(v59_acc, v75_data, v51_data);
          tensorforge::fmacdpp16<9>(v59_acc, v75_data, v52_data);
          tensorforge::fmacdpp16<10>(v59_acc, v75_data, v53_data);
          tensorforge::fmacdpp16<11>(v59_acc, v75_data, v54_data);
          tensorforge::fmacdpp16<12>(v59_acc, v75_data, v55_data);
          tensorforge::fmacdpp16<13>(v59_acc, v75_data, v56_data);
          tensorforge::fmacdpp16<14>(v59_acc, v75_data, v57_data);
          tensorforge::fmacdpp16<15>(v59_acc, v75_data, v58_data);
          tensorforge::fmacdpp16<0>(v60_acc, v76_data, v43_data);
          tensorforge::fmacdpp16<1>(v60_acc, v76_data, v44_data);
          tensorforge::fmacdpp16<2>(v60_acc, v76_data, v45_data);
          tensorforge::fmacdpp16<3>(v60_acc, v76_data, v46_data);
          tensorforge::fmacdpp16<4>(v60_acc, v76_data, v47_data);
          tensorforge::fmacdpp16<5>(v60_acc, v76_data, v48_data);
          tensorforge::fmacdpp16<6>(v60_acc, v76_data, v49_data);
          tensorforge::fmacdpp16<7>(v60_acc, v76_data, v50_data);
          tensorforge::fmacdpp16<8>(v60_acc, v76_data, v51_data);
          tensorforge::fmacdpp16<9>(v60_acc, v76_data, v52_data);
          tensorforge::fmacdpp16<10>(v60_acc, v76_data, v53_data);
          tensorforge::fmacdpp16<11>(v60_acc, v76_data, v54_data);
          tensorforge::fmacdpp16<12>(v60_acc, v76_data, v55_data);
          tensorforge::fmacdpp16<13>(v60_acc, v76_data, v56_data);
          tensorforge::fmacdpp16<14>(v60_acc, v76_data, v57_data);
          tensorforge::fmacdpp16<15>(v60_acc, v76_data, v58_data);
          tensorforge::fmacdpp16<0>(v61_acc, v77_data, v43_data);
          tensorforge::fmacdpp16<1>(v61_acc, v77_data, v44_data);
          tensorforge::fmacdpp16<2>(v61_acc, v77_data, v45_data);
          tensorforge::fmacdpp16<3>(v61_acc, v77_data, v46_data);
          tensorforge::fmacdpp16<4>(v61_acc, v77_data, v47_data);
          tensorforge::fmacdpp16<5>(v61_acc, v77_data, v48_data);
          tensorforge::fmacdpp16<6>(v61_acc, v77_data, v49_data);
          tensorforge::fmacdpp16<7>(v61_acc, v77_data, v50_data);
          tensorforge::fmacdpp16<8>(v61_acc, v77_data, v51_data);
          tensorforge::fmacdpp16<9>(v61_acc, v77_data, v52_data);
          tensorforge::fmacdpp16<10>(v61_acc, v77_data, v53_data);
          tensorforge::fmacdpp16<11>(v61_acc, v77_data, v54_data);
          tensorforge::fmacdpp16<12>(v61_acc, v77_data, v55_data);
          tensorforge::fmacdpp16<13>(v61_acc, v77_data, v56_data);
          tensorforge::fmacdpp16<14>(v61_acc, v77_data, v57_data);
          tensorforge::fmacdpp16<15>(v61_acc, v77_data, v58_data);
          tensorforge::fmacdpp16<0>(v62_acc, v78_data, v43_data);
          tensorforge::fmacdpp16<1>(v62_acc, v78_data, v44_data);
          tensorforge::fmacdpp16<2>(v62_acc, v78_data, v45_data);
          tensorforge::fmacdpp16<3>(v62_acc, v78_data, v46_data);
          tensorforge::fmacdpp16<4>(v62_acc, v78_data, v47_data);
          tensorforge::fmacdpp16<5>(v62_acc, v78_data, v48_data);
          tensorforge::fmacdpp16<6>(v62_acc, v78_data, v49_data);
          tensorforge::fmacdpp16<7>(v62_acc, v78_data, v50_data);
          tensorforge::fmacdpp16<8>(v62_acc, v78_data, v51_data);
          tensorforge::fmacdpp16<9>(v62_acc, v78_data, v52_data);
          tensorforge::fmacdpp16<10>(v62_acc, v78_data, v53_data);
          tensorforge::fmacdpp16<11>(v62_acc, v78_data, v54_data);
          tensorforge::fmacdpp16<12>(v62_acc, v78_data, v55_data);
          tensorforge::fmacdpp16<13>(v62_acc, v78_data, v56_data);
          tensorforge::fmacdpp16<14>(v62_acc, v78_data, v57_data);
          tensorforge::fmacdpp16<15>(v62_acc, v78_data, v58_data);
          tensorforge::fmacdpp16<0>(v63_acc, v79_data, v43_data);
          tensorforge::fmacdpp16<1>(v63_acc, v79_data, v44_data);
          tensorforge::fmacdpp16<2>(v63_acc, v79_data, v45_data);
          tensorforge::fmacdpp16<3>(v63_acc, v79_data, v46_data);
          tensorforge::fmacdpp16<4>(v63_acc, v79_data, v47_data);
          tensorforge::fmacdpp16<5>(v63_acc, v79_data, v48_data);
          tensorforge::fmacdpp16<6>(v63_acc, v79_data, v49_data);
          tensorforge::fmacdpp16<7>(v63_acc, v79_data, v50_data);
          tensorforge::fmacdpp16<8>(v63_acc, v79_data, v51_data);
          tensorforge::fmacdpp16<9>(v63_acc, v79_data, v52_data);
          tensorforge::fmacdpp16<10>(v63_acc, v79_data, v53_data);
          tensorforge::fmacdpp16<11>(v63_acc, v79_data, v54_data);
          tensorforge::fmacdpp16<12>(v63_acc, v79_data, v55_data);
          tensorforge::fmacdpp16<13>(v63_acc, v79_data, v56_data);
          tensorforge::fmacdpp16<14>(v63_acc, v79_data, v57_data);
          tensorforge::fmacdpp16<15>(v63_acc, v79_data, v58_data);
          tensorforge::fmacdpp16<0>(v64_acc, v80_data, v43_data);
          tensorforge::fmacdpp16<1>(v64_acc, v80_data, v44_data);
          tensorforge::fmacdpp16<2>(v64_acc, v80_data, v45_data);
          tensorforge::fmacdpp16<3>(v64_acc, v80_data, v46_data);
          tensorforge::fmacdpp16<4>(v64_acc, v80_data, v47_data);
          tensorforge::fmacdpp16<5>(v64_acc, v80_data, v48_data);
          tensorforge::fmacdpp16<6>(v64_acc, v80_data, v49_data);
          tensorforge::fmacdpp16<7>(v64_acc, v80_data, v50_data);
          tensorforge::fmacdpp16<8>(v64_acc, v80_data, v51_data);
          tensorforge::fmacdpp16<9>(v64_acc, v80_data, v52_data);
          tensorforge::fmacdpp16<10>(v64_acc, v80_data, v53_data);
          tensorforge::fmacdpp16<11>(v64_acc, v80_data, v54_data);
          tensorforge::fmacdpp16<12>(v64_acc, v80_data, v55_data);
          tensorforge::fmacdpp16<13>(v64_acc, v80_data, v56_data);
          tensorforge::fmacdpp16<14>(v64_acc, v80_data, v57_data);
          tensorforge::fmacdpp16<15>(v64_acc, v80_data, v58_data);
          tensorforge::fmacdpp16<0>(v65_acc, v81_data, v43_data);
          tensorforge::fmacdpp16<1>(v65_acc, v81_data, v44_data);
          tensorforge::fmacdpp16<2>(v65_acc, v81_data, v45_data);
          tensorforge::fmacdpp16<3>(v65_acc, v81_data, v46_data);
          tensorforge::fmacdpp16<4>(v65_acc, v81_data, v47_data);
          tensorforge::fmacdpp16<5>(v65_acc, v81_data, v48_data);
          tensorforge::fmacdpp16<6>(v65_acc, v81_data, v49_data);
          tensorforge::fmacdpp16<7>(v65_acc, v81_data, v50_data);
          tensorforge::fmacdpp16<8>(v65_acc, v81_data, v51_data);
          tensorforge::fmacdpp16<9>(v65_acc, v81_data, v52_data);
          tensorforge::fmacdpp16<10>(v65_acc, v81_data, v53_data);
          tensorforge::fmacdpp16<11>(v65_acc, v81_data, v54_data);
          tensorforge::fmacdpp16<12>(v65_acc, v81_data, v55_data);
          tensorforge::fmacdpp16<13>(v65_acc, v81_data, v56_data);
          tensorforge::fmacdpp16<14>(v65_acc, v81_data, v57_data);
          tensorforge::fmacdpp16<15>(v65_acc, v81_data, v58_data);
          tensorforge::fmacdpp16<0>(v66_acc, v82_data, v43_data);
          tensorforge::fmacdpp16<1>(v66_acc, v82_data, v44_data);
          tensorforge::fmacdpp16<2>(v66_acc, v82_data, v45_data);
          tensorforge::fmacdpp16<3>(v66_acc, v82_data, v46_data);
          tensorforge::fmacdpp16<4>(v66_acc, v82_data, v47_data);
          tensorforge::fmacdpp16<5>(v66_acc, v82_data, v48_data);
          tensorforge::fmacdpp16<6>(v66_acc, v82_data, v49_data);
          tensorforge::fmacdpp16<7>(v66_acc, v82_data, v50_data);
          tensorforge::fmacdpp16<8>(v66_acc, v82_data, v51_data);
          tensorforge::fmacdpp16<9>(v66_acc, v82_data, v52_data);
          tensorforge::fmacdpp16<10>(v66_acc, v82_data, v53_data);
          tensorforge::fmacdpp16<11>(v66_acc, v82_data, v54_data);
          tensorforge::fmacdpp16<12>(v66_acc, v82_data, v55_data);
          tensorforge::fmacdpp16<13>(v66_acc, v82_data, v56_data);
          tensorforge::fmacdpp16<14>(v66_acc, v82_data, v57_data);
          tensorforge::fmacdpp16<15>(v66_acc, v82_data, v58_data);
          tensorforge::fmacdpp16<0>(v67_acc, v83_data, v43_data);
          tensorforge::fmacdpp16<1>(v67_acc, v83_data, v44_data);
          tensorforge::fmacdpp16<2>(v67_acc, v83_data, v45_data);
          tensorforge::fmacdpp16<3>(v67_acc, v83_data, v46_data);
          tensorforge::fmacdpp16<4>(v67_acc, v83_data, v47_data);
          tensorforge::fmacdpp16<5>(v67_acc, v83_data, v48_data);
          tensorforge::fmacdpp16<6>(v67_acc, v83_data, v49_data);
          tensorforge::fmacdpp16<7>(v67_acc, v83_data, v50_data);
          tensorforge::fmacdpp16<8>(v67_acc, v83_data, v51_data);
          tensorforge::fmacdpp16<9>(v67_acc, v83_data, v52_data);
          tensorforge::fmacdpp16<10>(v67_acc, v83_data, v53_data);
          tensorforge::fmacdpp16<11>(v67_acc, v83_data, v54_data);
          tensorforge::fmacdpp16<12>(v67_acc, v83_data, v55_data);
          tensorforge::fmacdpp16<13>(v67_acc, v83_data, v56_data);
          tensorforge::fmacdpp16<14>(v67_acc, v83_data, v57_data);
          tensorforge::fmacdpp16<15>(v67_acc, v83_data, v58_data);
          tensorforge::fmacdpp16<0>(v68_acc, v84_data, v43_data);
          tensorforge::fmacdpp16<1>(v68_acc, v84_data, v44_data);
          tensorforge::fmacdpp16<2>(v68_acc, v84_data, v45_data);
          tensorforge::fmacdpp16<3>(v68_acc, v84_data, v46_data);
          tensorforge::fmacdpp16<4>(v68_acc, v84_data, v47_data);
          tensorforge::fmacdpp16<5>(v68_acc, v84_data, v48_data);
          tensorforge::fmacdpp16<6>(v68_acc, v84_data, v49_data);
          tensorforge::fmacdpp16<7>(v68_acc, v84_data, v50_data);
          tensorforge::fmacdpp16<8>(v68_acc, v84_data, v51_data);
          tensorforge::fmacdpp16<9>(v68_acc, v84_data, v52_data);
          tensorforge::fmacdpp16<10>(v68_acc, v84_data, v53_data);
          tensorforge::fmacdpp16<11>(v68_acc, v84_data, v54_data);
          tensorforge::fmacdpp16<12>(v68_acc, v84_data, v55_data);
          tensorforge::fmacdpp16<13>(v68_acc, v84_data, v56_data);
          tensorforge::fmacdpp16<14>(v68_acc, v84_data, v57_data);
          tensorforge::fmacdpp16<15>(v68_acc, v84_data, v58_data);
          tensorforge::fmacdpp16<0>(v69_acc, v85_data, v43_data);
          tensorforge::fmacdpp16<1>(v69_acc, v85_data, v44_data);
          tensorforge::fmacdpp16<2>(v69_acc, v85_data, v45_data);
          tensorforge::fmacdpp16<3>(v69_acc, v85_data, v46_data);
          tensorforge::fmacdpp16<4>(v69_acc, v85_data, v47_data);
          tensorforge::fmacdpp16<5>(v69_acc, v85_data, v48_data);
          tensorforge::fmacdpp16<6>(v69_acc, v85_data, v49_data);
          tensorforge::fmacdpp16<7>(v69_acc, v85_data, v50_data);
          tensorforge::fmacdpp16<8>(v69_acc, v85_data, v51_data);
          tensorforge::fmacdpp16<9>(v69_acc, v85_data, v52_data);
          tensorforge::fmacdpp16<10>(v69_acc, v85_data, v53_data);
          tensorforge::fmacdpp16<11>(v69_acc, v85_data, v54_data);
          tensorforge::fmacdpp16<12>(v69_acc, v85_data, v55_data);
          tensorforge::fmacdpp16<13>(v69_acc, v85_data, v56_data);
          tensorforge::fmacdpp16<14>(v69_acc, v85_data, v57_data);
          tensorforge::fmacdpp16<15>(v69_acc, v85_data, v58_data);
          tensorforge::fmacdpp16<0>(v70_acc, v86_data, v43_data);
          tensorforge::fmacdpp16<1>(v70_acc, v86_data, v44_data);
          tensorforge::fmacdpp16<2>(v70_acc, v86_data, v45_data);
          tensorforge::fmacdpp16<3>(v70_acc, v86_data, v46_data);
          tensorforge::fmacdpp16<4>(v70_acc, v86_data, v47_data);
          tensorforge::fmacdpp16<5>(v70_acc, v86_data, v48_data);
          tensorforge::fmacdpp16<6>(v70_acc, v86_data, v49_data);
          tensorforge::fmacdpp16<7>(v70_acc, v86_data, v50_data);
          tensorforge::fmacdpp16<8>(v70_acc, v86_data, v51_data);
          tensorforge::fmacdpp16<9>(v70_acc, v86_data, v52_data);
          tensorforge::fmacdpp16<10>(v70_acc, v86_data, v53_data);
          tensorforge::fmacdpp16<11>(v70_acc, v86_data, v54_data);
          tensorforge::fmacdpp16<12>(v70_acc, v86_data, v55_data);
          tensorforge::fmacdpp16<13>(v70_acc, v86_data, v56_data);
          tensorforge::fmacdpp16<14>(v70_acc, v86_data, v57_data);
          tensorforge::fmacdpp16<15>(v70_acc, v86_data, v58_data);
          tensorforge::fmacdpp16<0>(v71_acc, v87_data, v43_data);
          tensorforge::fmacdpp16<1>(v71_acc, v87_data, v44_data);
          tensorforge::fmacdpp16<2>(v71_acc, v87_data, v45_data);
          tensorforge::fmacdpp16<3>(v71_acc, v87_data, v46_data);
          tensorforge::fmacdpp16<4>(v71_acc, v87_data, v47_data);
          tensorforge::fmacdpp16<5>(v71_acc, v87_data, v48_data);
          tensorforge::fmacdpp16<6>(v71_acc, v87_data, v49_data);
          tensorforge::fmacdpp16<7>(v71_acc, v87_data, v50_data);
          tensorforge::fmacdpp16<8>(v71_acc, v87_data, v51_data);
          tensorforge::fmacdpp16<9>(v71_acc, v87_data, v52_data);
          tensorforge::fmacdpp16<10>(v71_acc, v87_data, v53_data);
          tensorforge::fmacdpp16<11>(v71_acc, v87_data, v54_data);
          tensorforge::fmacdpp16<12>(v71_acc, v87_data, v55_data);
          tensorforge::fmacdpp16<13>(v71_acc, v87_data, v56_data);
          tensorforge::fmacdpp16<14>(v71_acc, v87_data, v57_data);
          tensorforge::fmacdpp16<15>(v71_acc, v87_data, v58_data);
          tensorforge::fmacdpp16<0>(v72_acc, v88_data, v43_data);
          tensorforge::fmacdpp16<1>(v72_acc, v88_data, v44_data);
          tensorforge::fmacdpp16<2>(v72_acc, v88_data, v45_data);
          tensorforge::fmacdpp16<3>(v72_acc, v88_data, v46_data);
          tensorforge::fmacdpp16<4>(v72_acc, v88_data, v47_data);
          tensorforge::fmacdpp16<5>(v72_acc, v88_data, v48_data);
          tensorforge::fmacdpp16<6>(v72_acc, v88_data, v49_data);
          tensorforge::fmacdpp16<7>(v72_acc, v88_data, v50_data);
          tensorforge::fmacdpp16<8>(v72_acc, v88_data, v51_data);
          tensorforge::fmacdpp16<9>(v72_acc, v88_data, v52_data);
          tensorforge::fmacdpp16<10>(v72_acc, v88_data, v53_data);
          tensorforge::fmacdpp16<11>(v72_acc, v88_data, v54_data);
          tensorforge::fmacdpp16<12>(v72_acc, v88_data, v55_data);
          tensorforge::fmacdpp16<13>(v72_acc, v88_data, v56_data);
          tensorforge::fmacdpp16<14>(v72_acc, v88_data, v57_data);
          tensorforge::fmacdpp16<15>(v72_acc, v88_data, v58_data);
          tensorforge::fmacdpp16<0>(v73_acc, v89_data, v43_data);
          tensorforge::fmacdpp16<1>(v73_acc, v89_data, v44_data);
          tensorforge::fmacdpp16<2>(v73_acc, v89_data, v45_data);
          tensorforge::fmacdpp16<3>(v73_acc, v89_data, v46_data);
          tensorforge::fmacdpp16<4>(v73_acc, v89_data, v47_data);
          tensorforge::fmacdpp16<5>(v73_acc, v89_data, v48_data);
          tensorforge::fmacdpp16<6>(v73_acc, v89_data, v49_data);
          tensorforge::fmacdpp16<7>(v73_acc, v89_data, v50_data);
          tensorforge::fmacdpp16<8>(v73_acc, v89_data, v51_data);
          tensorforge::fmacdpp16<9>(v73_acc, v89_data, v52_data);
          tensorforge::fmacdpp16<10>(v73_acc, v89_data, v53_data);
          tensorforge::fmacdpp16<11>(v73_acc, v89_data, v54_data);
          tensorforge::fmacdpp16<12>(v73_acc, v89_data, v55_data);
          tensorforge::fmacdpp16<13>(v73_acc, v89_data, v56_data);
          tensorforge::fmacdpp16<14>(v73_acc, v89_data, v57_data);
          tensorforge::fmacdpp16<15>(v73_acc, v89_data, v58_data);
          tensorforge::fmacdpp16<0>(v74_acc, v90_data, v43_data);
          tensorforge::fmacdpp16<1>(v74_acc, v90_data, v44_data);
          tensorforge::fmacdpp16<2>(v74_acc, v90_data, v45_data);
          tensorforge::fmacdpp16<3>(v74_acc, v90_data, v46_data);
          tensorforge::fmacdpp16<4>(v74_acc, v90_data, v47_data);
          tensorforge::fmacdpp16<5>(v74_acc, v90_data, v48_data);
          tensorforge::fmacdpp16<6>(v74_acc, v90_data, v49_data);
          tensorforge::fmacdpp16<7>(v74_acc, v90_data, v50_data);
          tensorforge::fmacdpp16<8>(v74_acc, v90_data, v51_data);
          tensorforge::fmacdpp16<9>(v74_acc, v90_data, v52_data);
          tensorforge::fmacdpp16<10>(v74_acc, v90_data, v53_data);
          tensorforge::fmacdpp16<11>(v74_acc, v90_data, v54_data);
          tensorforge::fmacdpp16<12>(v74_acc, v90_data, v55_data);
          tensorforge::fmacdpp16<13>(v74_acc, v90_data, v56_data);
          tensorforge::fmacdpp16<14>(v74_acc, v90_data, v57_data);
          tensorforge::fmacdpp16<15>(v74_acc, v90_data, v58_data);
          r2[0] = v59_acc;
          r2[1] = v60_acc;
          r2[2] = v61_acc;
          r2[3] = v62_acc;
          r2[4] = v63_acc;
          r2[5] = v64_acc;
          r2[6] = v65_acc;
          r2[7] = v66_acc;
          r2[8] = v67_acc;
          r2[9] = v68_acc;
          r2[10] = v69_acc;
          r2[11] = v70_acc;
          r2[12] = v71_acc;
          r2[13] = v72_acc;
          r2[14] = v73_acc;
          r2[15] = v74_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v94_i0 = 0; v94_i0 < 1; ++v94_i0) {
            int32_t v102_lead = v13_lead + (v94_i0 * 16);
            #pragma unroll
            for (int32_t v95_i1 = 0; v95_i1 < 16; ++v95_i1) {
              double v97_data = r2[(v94_i0 + v95_i1)];
              glb_m0[(v102_lead + (v95_i1 * 16))] = v97_data;
            }
          }
        }
      }
    }
  }
}

