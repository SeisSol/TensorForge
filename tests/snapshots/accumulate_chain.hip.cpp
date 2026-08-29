// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a03a3cd0d, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8a03a3cd0d), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_8a03a3cd0d, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
              float v29_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v21_i1 * 12))]);
              r0[v21_i1] = v29_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v32_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v32_lin;
          float v33_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v33_lin;
          float v34_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v34_lin;
          float v35_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v35_lin;
          float v36_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v36_lin;
          float v37_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v37_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
              float v51_data = __builtin_nontemporal_load(&glb_m3[(v19_lead + (v43_i1 * 12))]);
              r3[v43_i1] = v51_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v54_data = r0[0];
          float v55_data = r0[1];
          float v56_data = r0[2];
          float v57_data = r0[3];
          float v58_data = r0[4];
          float v59_data = r0[5];
          float v60_data = r0[6];
          float v61_data = r0[7];
          float v62_data = r0[8];
          float v63_data = r0[9];
          float v64_data = r0[10];
          float v65_data = r0[11];
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_lin = r1[0];
          tensorforge::fmacdpp16<0>(v66_acc, v74_lin, v54_data);
          tensorforge::fmacdpp16<1>(v66_acc, v74_lin, v55_data);
          tensorforge::fmacdpp16<2>(v66_acc, v74_lin, v56_data);
          tensorforge::fmacdpp16<3>(v66_acc, v74_lin, v57_data);
          tensorforge::fmacdpp16<4>(v66_acc, v74_lin, v58_data);
          tensorforge::fmacdpp16<5>(v66_acc, v74_lin, v59_data);
          tensorforge::fmacdpp16<6>(v66_acc, v74_lin, v60_data);
          tensorforge::fmacdpp16<7>(v66_acc, v74_lin, v61_data);
          tensorforge::fmacdpp16<8>(v66_acc, v74_lin, v62_data);
          tensorforge::fmacdpp16<9>(v66_acc, v74_lin, v63_data);
          tensorforge::fmacdpp16<10>(v66_acc, v74_lin, v64_data);
          tensorforge::fmacdpp16<11>(v66_acc, v74_lin, v65_data);
          tensorforge::fmacdpp16<12>(v67_acc, v74_lin, v54_data);
          tensorforge::fmacdpp16<13>(v67_acc, v74_lin, v55_data);
          tensorforge::fmacdpp16<14>(v67_acc, v74_lin, v56_data);
          tensorforge::fmacdpp16<15>(v67_acc, v74_lin, v57_data);
          float v75_lin = r1[1];
          tensorforge::fmacdpp16<0>(v67_acc, v75_lin, v58_data);
          tensorforge::fmacdpp16<1>(v67_acc, v75_lin, v59_data);
          tensorforge::fmacdpp16<2>(v67_acc, v75_lin, v60_data);
          tensorforge::fmacdpp16<3>(v67_acc, v75_lin, v61_data);
          tensorforge::fmacdpp16<4>(v67_acc, v75_lin, v62_data);
          tensorforge::fmacdpp16<5>(v67_acc, v75_lin, v63_data);
          tensorforge::fmacdpp16<6>(v67_acc, v75_lin, v64_data);
          tensorforge::fmacdpp16<7>(v67_acc, v75_lin, v65_data);
          tensorforge::fmacdpp16<8>(v68_acc, v75_lin, v54_data);
          tensorforge::fmacdpp16<9>(v68_acc, v75_lin, v55_data);
          tensorforge::fmacdpp16<10>(v68_acc, v75_lin, v56_data);
          tensorforge::fmacdpp16<11>(v68_acc, v75_lin, v57_data);
          tensorforge::fmacdpp16<12>(v68_acc, v75_lin, v58_data);
          tensorforge::fmacdpp16<13>(v68_acc, v75_lin, v59_data);
          tensorforge::fmacdpp16<14>(v68_acc, v75_lin, v60_data);
          tensorforge::fmacdpp16<15>(v68_acc, v75_lin, v61_data);
          float v76_lin = r1[2];
          tensorforge::fmacdpp16<0>(v68_acc, v76_lin, v62_data);
          tensorforge::fmacdpp16<1>(v68_acc, v76_lin, v63_data);
          tensorforge::fmacdpp16<2>(v68_acc, v76_lin, v64_data);
          tensorforge::fmacdpp16<3>(v68_acc, v76_lin, v65_data);
          tensorforge::fmacdpp16<4>(v69_acc, v76_lin, v54_data);
          tensorforge::fmacdpp16<5>(v69_acc, v76_lin, v55_data);
          tensorforge::fmacdpp16<6>(v69_acc, v76_lin, v56_data);
          tensorforge::fmacdpp16<7>(v69_acc, v76_lin, v57_data);
          tensorforge::fmacdpp16<8>(v69_acc, v76_lin, v58_data);
          tensorforge::fmacdpp16<9>(v69_acc, v76_lin, v59_data);
          tensorforge::fmacdpp16<10>(v69_acc, v76_lin, v60_data);
          tensorforge::fmacdpp16<11>(v69_acc, v76_lin, v61_data);
          tensorforge::fmacdpp16<12>(v69_acc, v76_lin, v62_data);
          tensorforge::fmacdpp16<13>(v69_acc, v76_lin, v63_data);
          tensorforge::fmacdpp16<14>(v69_acc, v76_lin, v64_data);
          tensorforge::fmacdpp16<15>(v69_acc, v76_lin, v65_data);
          float v77_lin = r1[3];
          tensorforge::fmacdpp16<0>(v70_acc, v77_lin, v54_data);
          tensorforge::fmacdpp16<1>(v70_acc, v77_lin, v55_data);
          tensorforge::fmacdpp16<2>(v70_acc, v77_lin, v56_data);
          tensorforge::fmacdpp16<3>(v70_acc, v77_lin, v57_data);
          tensorforge::fmacdpp16<4>(v70_acc, v77_lin, v58_data);
          tensorforge::fmacdpp16<5>(v70_acc, v77_lin, v59_data);
          tensorforge::fmacdpp16<6>(v70_acc, v77_lin, v60_data);
          tensorforge::fmacdpp16<7>(v70_acc, v77_lin, v61_data);
          tensorforge::fmacdpp16<8>(v70_acc, v77_lin, v62_data);
          tensorforge::fmacdpp16<9>(v70_acc, v77_lin, v63_data);
          tensorforge::fmacdpp16<10>(v70_acc, v77_lin, v64_data);
          tensorforge::fmacdpp16<11>(v70_acc, v77_lin, v65_data);
          tensorforge::fmacdpp16<12>(v71_acc, v77_lin, v54_data);
          tensorforge::fmacdpp16<13>(v71_acc, v77_lin, v55_data);
          tensorforge::fmacdpp16<14>(v71_acc, v77_lin, v56_data);
          tensorforge::fmacdpp16<15>(v71_acc, v77_lin, v57_data);
          float v78_lin = r1[4];
          tensorforge::fmacdpp16<0>(v71_acc, v78_lin, v58_data);
          tensorforge::fmacdpp16<1>(v71_acc, v78_lin, v59_data);
          tensorforge::fmacdpp16<2>(v71_acc, v78_lin, v60_data);
          tensorforge::fmacdpp16<3>(v71_acc, v78_lin, v61_data);
          tensorforge::fmacdpp16<4>(v71_acc, v78_lin, v62_data);
          tensorforge::fmacdpp16<5>(v71_acc, v78_lin, v63_data);
          tensorforge::fmacdpp16<6>(v71_acc, v78_lin, v64_data);
          tensorforge::fmacdpp16<7>(v71_acc, v78_lin, v65_data);
          tensorforge::fmacdpp16<8>(v72_acc, v78_lin, v54_data);
          tensorforge::fmacdpp16<9>(v72_acc, v78_lin, v55_data);
          tensorforge::fmacdpp16<10>(v72_acc, v78_lin, v56_data);
          tensorforge::fmacdpp16<11>(v72_acc, v78_lin, v57_data);
          tensorforge::fmacdpp16<12>(v72_acc, v78_lin, v58_data);
          tensorforge::fmacdpp16<13>(v72_acc, v78_lin, v59_data);
          tensorforge::fmacdpp16<14>(v72_acc, v78_lin, v60_data);
          tensorforge::fmacdpp16<15>(v72_acc, v78_lin, v61_data);
          float v79_lin = r1[5];
          tensorforge::fmacdpp16<0>(v72_acc, v79_lin, v62_data);
          tensorforge::fmacdpp16<1>(v72_acc, v79_lin, v63_data);
          tensorforge::fmacdpp16<2>(v72_acc, v79_lin, v64_data);
          tensorforge::fmacdpp16<3>(v72_acc, v79_lin, v65_data);
          tensorforge::fmacdpp16<4>(v73_acc, v79_lin, v54_data);
          tensorforge::fmacdpp16<5>(v73_acc, v79_lin, v55_data);
          tensorforge::fmacdpp16<6>(v73_acc, v79_lin, v56_data);
          tensorforge::fmacdpp16<7>(v73_acc, v79_lin, v57_data);
          tensorforge::fmacdpp16<8>(v73_acc, v79_lin, v58_data);
          tensorforge::fmacdpp16<9>(v73_acc, v79_lin, v59_data);
          tensorforge::fmacdpp16<10>(v73_acc, v79_lin, v60_data);
          tensorforge::fmacdpp16<11>(v73_acc, v79_lin, v61_data);
          tensorforge::fmacdpp16<12>(v73_acc, v79_lin, v62_data);
          tensorforge::fmacdpp16<13>(v73_acc, v79_lin, v63_data);
          tensorforge::fmacdpp16<14>(v73_acc, v79_lin, v64_data);
          tensorforge::fmacdpp16<15>(v73_acc, v79_lin, v65_data);
          r2[0] = v66_acc;
          r2[1] = v67_acc;
          r2[2] = v68_acc;
          r2[3] = v69_acc;
          r2[4] = v70_acc;
          r2[5] = v71_acc;
          r2[6] = v72_acc;
          r2[7] = v73_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v81_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v81_lin;
          float v82_lin = glb_m4[16 + threadIdx.x * 1];
          r4[1] = v82_lin;
          float v83_lin = glb_m4[32 + threadIdx.x * 1];
          r4[2] = v83_lin;
          float v84_lin = glb_m4[48 + threadIdx.x * 1];
          r4[3] = v84_lin;
          float v85_lin = glb_m4[64 + threadIdx.x * 1];
          r4[4] = v85_lin;
          float v86_lin = glb_m4[80 + threadIdx.x * 1];
          r4[5] = v86_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v92_i1 = 0; v92_i1 < 12; ++v92_i1) {
              float v100_data = __builtin_nontemporal_load(&glb_m5[(v19_lead + (v92_i1 * 12))]);
              r6[v92_i1] = v100_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v104_data = r3[0];
          float v105_data = r3[1];
          float v106_data = r3[2];
          float v107_data = r3[3];
          float v108_data = r3[4];
          float v109_data = r3[5];
          float v110_data = r3[6];
          float v111_data = r3[7];
          float v112_data = r3[8];
          float v113_data = r3[9];
          float v114_data = r3[10];
          float v115_data = r3[11];
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_acc{};
          float v122_acc{};
          float v123_acc{};
          float v124_lin = r4[0];
          tensorforge::fmacdpp16<0>(v116_acc, v124_lin, v104_data);
          tensorforge::fmacdpp16<1>(v116_acc, v124_lin, v105_data);
          tensorforge::fmacdpp16<2>(v116_acc, v124_lin, v106_data);
          tensorforge::fmacdpp16<3>(v116_acc, v124_lin, v107_data);
          tensorforge::fmacdpp16<4>(v116_acc, v124_lin, v108_data);
          tensorforge::fmacdpp16<5>(v116_acc, v124_lin, v109_data);
          tensorforge::fmacdpp16<6>(v116_acc, v124_lin, v110_data);
          tensorforge::fmacdpp16<7>(v116_acc, v124_lin, v111_data);
          tensorforge::fmacdpp16<8>(v116_acc, v124_lin, v112_data);
          tensorforge::fmacdpp16<9>(v116_acc, v124_lin, v113_data);
          tensorforge::fmacdpp16<10>(v116_acc, v124_lin, v114_data);
          tensorforge::fmacdpp16<11>(v116_acc, v124_lin, v115_data);
          tensorforge::fmacdpp16<12>(v117_acc, v124_lin, v104_data);
          tensorforge::fmacdpp16<13>(v117_acc, v124_lin, v105_data);
          tensorforge::fmacdpp16<14>(v117_acc, v124_lin, v106_data);
          tensorforge::fmacdpp16<15>(v117_acc, v124_lin, v107_data);
          float v125_lin = r4[1];
          tensorforge::fmacdpp16<0>(v117_acc, v125_lin, v108_data);
          tensorforge::fmacdpp16<1>(v117_acc, v125_lin, v109_data);
          tensorforge::fmacdpp16<2>(v117_acc, v125_lin, v110_data);
          tensorforge::fmacdpp16<3>(v117_acc, v125_lin, v111_data);
          tensorforge::fmacdpp16<4>(v117_acc, v125_lin, v112_data);
          tensorforge::fmacdpp16<5>(v117_acc, v125_lin, v113_data);
          tensorforge::fmacdpp16<6>(v117_acc, v125_lin, v114_data);
          tensorforge::fmacdpp16<7>(v117_acc, v125_lin, v115_data);
          tensorforge::fmacdpp16<8>(v118_acc, v125_lin, v104_data);
          tensorforge::fmacdpp16<9>(v118_acc, v125_lin, v105_data);
          tensorforge::fmacdpp16<10>(v118_acc, v125_lin, v106_data);
          tensorforge::fmacdpp16<11>(v118_acc, v125_lin, v107_data);
          tensorforge::fmacdpp16<12>(v118_acc, v125_lin, v108_data);
          tensorforge::fmacdpp16<13>(v118_acc, v125_lin, v109_data);
          tensorforge::fmacdpp16<14>(v118_acc, v125_lin, v110_data);
          tensorforge::fmacdpp16<15>(v118_acc, v125_lin, v111_data);
          float v126_lin = r4[2];
          tensorforge::fmacdpp16<0>(v118_acc, v126_lin, v112_data);
          tensorforge::fmacdpp16<1>(v118_acc, v126_lin, v113_data);
          tensorforge::fmacdpp16<2>(v118_acc, v126_lin, v114_data);
          tensorforge::fmacdpp16<3>(v118_acc, v126_lin, v115_data);
          tensorforge::fmacdpp16<4>(v119_acc, v126_lin, v104_data);
          tensorforge::fmacdpp16<5>(v119_acc, v126_lin, v105_data);
          tensorforge::fmacdpp16<6>(v119_acc, v126_lin, v106_data);
          tensorforge::fmacdpp16<7>(v119_acc, v126_lin, v107_data);
          tensorforge::fmacdpp16<8>(v119_acc, v126_lin, v108_data);
          tensorforge::fmacdpp16<9>(v119_acc, v126_lin, v109_data);
          tensorforge::fmacdpp16<10>(v119_acc, v126_lin, v110_data);
          tensorforge::fmacdpp16<11>(v119_acc, v126_lin, v111_data);
          tensorforge::fmacdpp16<12>(v119_acc, v126_lin, v112_data);
          tensorforge::fmacdpp16<13>(v119_acc, v126_lin, v113_data);
          tensorforge::fmacdpp16<14>(v119_acc, v126_lin, v114_data);
          tensorforge::fmacdpp16<15>(v119_acc, v126_lin, v115_data);
          float v127_lin = r4[3];
          tensorforge::fmacdpp16<0>(v120_acc, v127_lin, v104_data);
          tensorforge::fmacdpp16<1>(v120_acc, v127_lin, v105_data);
          tensorforge::fmacdpp16<2>(v120_acc, v127_lin, v106_data);
          tensorforge::fmacdpp16<3>(v120_acc, v127_lin, v107_data);
          tensorforge::fmacdpp16<4>(v120_acc, v127_lin, v108_data);
          tensorforge::fmacdpp16<5>(v120_acc, v127_lin, v109_data);
          tensorforge::fmacdpp16<6>(v120_acc, v127_lin, v110_data);
          tensorforge::fmacdpp16<7>(v120_acc, v127_lin, v111_data);
          tensorforge::fmacdpp16<8>(v120_acc, v127_lin, v112_data);
          tensorforge::fmacdpp16<9>(v120_acc, v127_lin, v113_data);
          tensorforge::fmacdpp16<10>(v120_acc, v127_lin, v114_data);
          tensorforge::fmacdpp16<11>(v120_acc, v127_lin, v115_data);
          tensorforge::fmacdpp16<12>(v121_acc, v127_lin, v104_data);
          tensorforge::fmacdpp16<13>(v121_acc, v127_lin, v105_data);
          tensorforge::fmacdpp16<14>(v121_acc, v127_lin, v106_data);
          tensorforge::fmacdpp16<15>(v121_acc, v127_lin, v107_data);
          float v128_lin = r4[4];
          tensorforge::fmacdpp16<0>(v121_acc, v128_lin, v108_data);
          tensorforge::fmacdpp16<1>(v121_acc, v128_lin, v109_data);
          tensorforge::fmacdpp16<2>(v121_acc, v128_lin, v110_data);
          tensorforge::fmacdpp16<3>(v121_acc, v128_lin, v111_data);
          tensorforge::fmacdpp16<4>(v121_acc, v128_lin, v112_data);
          tensorforge::fmacdpp16<5>(v121_acc, v128_lin, v113_data);
          tensorforge::fmacdpp16<6>(v121_acc, v128_lin, v114_data);
          tensorforge::fmacdpp16<7>(v121_acc, v128_lin, v115_data);
          tensorforge::fmacdpp16<8>(v122_acc, v128_lin, v104_data);
          tensorforge::fmacdpp16<9>(v122_acc, v128_lin, v105_data);
          tensorforge::fmacdpp16<10>(v122_acc, v128_lin, v106_data);
          tensorforge::fmacdpp16<11>(v122_acc, v128_lin, v107_data);
          tensorforge::fmacdpp16<12>(v122_acc, v128_lin, v108_data);
          tensorforge::fmacdpp16<13>(v122_acc, v128_lin, v109_data);
          tensorforge::fmacdpp16<14>(v122_acc, v128_lin, v110_data);
          tensorforge::fmacdpp16<15>(v122_acc, v128_lin, v111_data);
          float v129_lin = r4[5];
          tensorforge::fmacdpp16<0>(v122_acc, v129_lin, v112_data);
          tensorforge::fmacdpp16<1>(v122_acc, v129_lin, v113_data);
          tensorforge::fmacdpp16<2>(v122_acc, v129_lin, v114_data);
          tensorforge::fmacdpp16<3>(v122_acc, v129_lin, v115_data);
          tensorforge::fmacdpp16<4>(v123_acc, v129_lin, v104_data);
          tensorforge::fmacdpp16<5>(v123_acc, v129_lin, v105_data);
          tensorforge::fmacdpp16<6>(v123_acc, v129_lin, v106_data);
          tensorforge::fmacdpp16<7>(v123_acc, v129_lin, v107_data);
          tensorforge::fmacdpp16<8>(v123_acc, v129_lin, v108_data);
          tensorforge::fmacdpp16<9>(v123_acc, v129_lin, v109_data);
          tensorforge::fmacdpp16<10>(v123_acc, v129_lin, v110_data);
          tensorforge::fmacdpp16<11>(v123_acc, v129_lin, v111_data);
          tensorforge::fmacdpp16<12>(v123_acc, v129_lin, v112_data);
          tensorforge::fmacdpp16<13>(v123_acc, v129_lin, v113_data);
          tensorforge::fmacdpp16<14>(v123_acc, v129_lin, v114_data);
          tensorforge::fmacdpp16<15>(v123_acc, v129_lin, v115_data);
          ir5[0] = v116_acc;
          ir5[1] = v117_acc;
          ir5[2] = v118_acc;
          ir5[3] = v119_acc;
          ir5[4] = v120_acc;
          ir5[5] = v121_acc;
          ir5[6] = v122_acc;
          ir5[7] = v123_acc;
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v134_n1 = 0; v134_n1 < 8; ++v134_n1) {
              float v136_data = ir5[v134_n1];
              float v138_data = r2[v134_n1];
              r5[v134_n1] = (v138_data + v136_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v142_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v142_lin;
          float v143_lin = glb_m6[16 + threadIdx.x * 1];
          r7[1] = v143_lin;
          float v144_lin = glb_m6[32 + threadIdx.x * 1];
          r7[2] = v144_lin;
          float v145_lin = glb_m6[48 + threadIdx.x * 1];
          r7[3] = v145_lin;
          float v146_lin = glb_m6[64 + threadIdx.x * 1];
          r7[4] = v146_lin;
          float v147_lin = glb_m6[80 + threadIdx.x * 1];
          r7[5] = v147_lin;
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v153_i1 = 0; v153_i1 < 12; ++v153_i1) {
              float v161_data = __builtin_nontemporal_load(&glb_m7[(v19_lead + (v153_i1 * 12))]);
              r9[v153_i1] = v161_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v165_data = r6[0];
          float v166_data = r6[1];
          float v167_data = r6[2];
          float v168_data = r6[3];
          float v169_data = r6[4];
          float v170_data = r6[5];
          float v171_data = r6[6];
          float v172_data = r6[7];
          float v173_data = r6[8];
          float v174_data = r6[9];
          float v175_data = r6[10];
          float v176_data = r6[11];
          float v177_acc{};
          float v178_acc{};
          float v179_acc{};
          float v180_acc{};
          float v181_acc{};
          float v182_acc{};
          float v183_acc{};
          float v184_acc{};
          float v185_lin = r7[0];
          tensorforge::fmacdpp16<0>(v177_acc, v185_lin, v165_data);
          tensorforge::fmacdpp16<1>(v177_acc, v185_lin, v166_data);
          tensorforge::fmacdpp16<2>(v177_acc, v185_lin, v167_data);
          tensorforge::fmacdpp16<3>(v177_acc, v185_lin, v168_data);
          tensorforge::fmacdpp16<4>(v177_acc, v185_lin, v169_data);
          tensorforge::fmacdpp16<5>(v177_acc, v185_lin, v170_data);
          tensorforge::fmacdpp16<6>(v177_acc, v185_lin, v171_data);
          tensorforge::fmacdpp16<7>(v177_acc, v185_lin, v172_data);
          tensorforge::fmacdpp16<8>(v177_acc, v185_lin, v173_data);
          tensorforge::fmacdpp16<9>(v177_acc, v185_lin, v174_data);
          tensorforge::fmacdpp16<10>(v177_acc, v185_lin, v175_data);
          tensorforge::fmacdpp16<11>(v177_acc, v185_lin, v176_data);
          tensorforge::fmacdpp16<12>(v178_acc, v185_lin, v165_data);
          tensorforge::fmacdpp16<13>(v178_acc, v185_lin, v166_data);
          tensorforge::fmacdpp16<14>(v178_acc, v185_lin, v167_data);
          tensorforge::fmacdpp16<15>(v178_acc, v185_lin, v168_data);
          float v186_lin = r7[1];
          tensorforge::fmacdpp16<0>(v178_acc, v186_lin, v169_data);
          tensorforge::fmacdpp16<1>(v178_acc, v186_lin, v170_data);
          tensorforge::fmacdpp16<2>(v178_acc, v186_lin, v171_data);
          tensorforge::fmacdpp16<3>(v178_acc, v186_lin, v172_data);
          tensorforge::fmacdpp16<4>(v178_acc, v186_lin, v173_data);
          tensorforge::fmacdpp16<5>(v178_acc, v186_lin, v174_data);
          tensorforge::fmacdpp16<6>(v178_acc, v186_lin, v175_data);
          tensorforge::fmacdpp16<7>(v178_acc, v186_lin, v176_data);
          tensorforge::fmacdpp16<8>(v179_acc, v186_lin, v165_data);
          tensorforge::fmacdpp16<9>(v179_acc, v186_lin, v166_data);
          tensorforge::fmacdpp16<10>(v179_acc, v186_lin, v167_data);
          tensorforge::fmacdpp16<11>(v179_acc, v186_lin, v168_data);
          tensorforge::fmacdpp16<12>(v179_acc, v186_lin, v169_data);
          tensorforge::fmacdpp16<13>(v179_acc, v186_lin, v170_data);
          tensorforge::fmacdpp16<14>(v179_acc, v186_lin, v171_data);
          tensorforge::fmacdpp16<15>(v179_acc, v186_lin, v172_data);
          float v187_lin = r7[2];
          tensorforge::fmacdpp16<0>(v179_acc, v187_lin, v173_data);
          tensorforge::fmacdpp16<1>(v179_acc, v187_lin, v174_data);
          tensorforge::fmacdpp16<2>(v179_acc, v187_lin, v175_data);
          tensorforge::fmacdpp16<3>(v179_acc, v187_lin, v176_data);
          tensorforge::fmacdpp16<4>(v180_acc, v187_lin, v165_data);
          tensorforge::fmacdpp16<5>(v180_acc, v187_lin, v166_data);
          tensorforge::fmacdpp16<6>(v180_acc, v187_lin, v167_data);
          tensorforge::fmacdpp16<7>(v180_acc, v187_lin, v168_data);
          tensorforge::fmacdpp16<8>(v180_acc, v187_lin, v169_data);
          tensorforge::fmacdpp16<9>(v180_acc, v187_lin, v170_data);
          tensorforge::fmacdpp16<10>(v180_acc, v187_lin, v171_data);
          tensorforge::fmacdpp16<11>(v180_acc, v187_lin, v172_data);
          tensorforge::fmacdpp16<12>(v180_acc, v187_lin, v173_data);
          tensorforge::fmacdpp16<13>(v180_acc, v187_lin, v174_data);
          tensorforge::fmacdpp16<14>(v180_acc, v187_lin, v175_data);
          tensorforge::fmacdpp16<15>(v180_acc, v187_lin, v176_data);
          float v188_lin = r7[3];
          tensorforge::fmacdpp16<0>(v181_acc, v188_lin, v165_data);
          tensorforge::fmacdpp16<1>(v181_acc, v188_lin, v166_data);
          tensorforge::fmacdpp16<2>(v181_acc, v188_lin, v167_data);
          tensorforge::fmacdpp16<3>(v181_acc, v188_lin, v168_data);
          tensorforge::fmacdpp16<4>(v181_acc, v188_lin, v169_data);
          tensorforge::fmacdpp16<5>(v181_acc, v188_lin, v170_data);
          tensorforge::fmacdpp16<6>(v181_acc, v188_lin, v171_data);
          tensorforge::fmacdpp16<7>(v181_acc, v188_lin, v172_data);
          tensorforge::fmacdpp16<8>(v181_acc, v188_lin, v173_data);
          tensorforge::fmacdpp16<9>(v181_acc, v188_lin, v174_data);
          tensorforge::fmacdpp16<10>(v181_acc, v188_lin, v175_data);
          tensorforge::fmacdpp16<11>(v181_acc, v188_lin, v176_data);
          tensorforge::fmacdpp16<12>(v182_acc, v188_lin, v165_data);
          tensorforge::fmacdpp16<13>(v182_acc, v188_lin, v166_data);
          tensorforge::fmacdpp16<14>(v182_acc, v188_lin, v167_data);
          tensorforge::fmacdpp16<15>(v182_acc, v188_lin, v168_data);
          float v189_lin = r7[4];
          tensorforge::fmacdpp16<0>(v182_acc, v189_lin, v169_data);
          tensorforge::fmacdpp16<1>(v182_acc, v189_lin, v170_data);
          tensorforge::fmacdpp16<2>(v182_acc, v189_lin, v171_data);
          tensorforge::fmacdpp16<3>(v182_acc, v189_lin, v172_data);
          tensorforge::fmacdpp16<4>(v182_acc, v189_lin, v173_data);
          tensorforge::fmacdpp16<5>(v182_acc, v189_lin, v174_data);
          tensorforge::fmacdpp16<6>(v182_acc, v189_lin, v175_data);
          tensorforge::fmacdpp16<7>(v182_acc, v189_lin, v176_data);
          tensorforge::fmacdpp16<8>(v183_acc, v189_lin, v165_data);
          tensorforge::fmacdpp16<9>(v183_acc, v189_lin, v166_data);
          tensorforge::fmacdpp16<10>(v183_acc, v189_lin, v167_data);
          tensorforge::fmacdpp16<11>(v183_acc, v189_lin, v168_data);
          tensorforge::fmacdpp16<12>(v183_acc, v189_lin, v169_data);
          tensorforge::fmacdpp16<13>(v183_acc, v189_lin, v170_data);
          tensorforge::fmacdpp16<14>(v183_acc, v189_lin, v171_data);
          tensorforge::fmacdpp16<15>(v183_acc, v189_lin, v172_data);
          float v190_lin = r7[5];
          tensorforge::fmacdpp16<0>(v183_acc, v190_lin, v173_data);
          tensorforge::fmacdpp16<1>(v183_acc, v190_lin, v174_data);
          tensorforge::fmacdpp16<2>(v183_acc, v190_lin, v175_data);
          tensorforge::fmacdpp16<3>(v183_acc, v190_lin, v176_data);
          tensorforge::fmacdpp16<4>(v184_acc, v190_lin, v165_data);
          tensorforge::fmacdpp16<5>(v184_acc, v190_lin, v166_data);
          tensorforge::fmacdpp16<6>(v184_acc, v190_lin, v167_data);
          tensorforge::fmacdpp16<7>(v184_acc, v190_lin, v168_data);
          tensorforge::fmacdpp16<8>(v184_acc, v190_lin, v169_data);
          tensorforge::fmacdpp16<9>(v184_acc, v190_lin, v170_data);
          tensorforge::fmacdpp16<10>(v184_acc, v190_lin, v171_data);
          tensorforge::fmacdpp16<11>(v184_acc, v190_lin, v172_data);
          tensorforge::fmacdpp16<12>(v184_acc, v190_lin, v173_data);
          tensorforge::fmacdpp16<13>(v184_acc, v190_lin, v174_data);
          tensorforge::fmacdpp16<14>(v184_acc, v190_lin, v175_data);
          tensorforge::fmacdpp16<15>(v184_acc, v190_lin, v176_data);
          ir8[0] = v177_acc;
          ir8[1] = v178_acc;
          ir8[2] = v179_acc;
          ir8[3] = v180_acc;
          ir8[4] = v181_acc;
          ir8[5] = v182_acc;
          ir8[6] = v183_acc;
          ir8[7] = v184_acc;
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v195_n1 = 0; v195_n1 < 8; ++v195_n1) {
              float v197_data = ir8[v195_n1];
              float v199_data = r5[v195_n1];
              r8[v195_n1] = (v199_data + v197_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          float v203_lin = glb_m8[0 + threadIdx.x * 1];
          r10[0] = v203_lin;
          float v204_lin = glb_m8[16 + threadIdx.x * 1];
          r10[1] = v204_lin;
          float v205_lin = glb_m8[32 + threadIdx.x * 1];
          r10[2] = v205_lin;
          float v206_lin = glb_m8[48 + threadIdx.x * 1];
          r10[3] = v206_lin;
          float v207_lin = glb_m8[64 + threadIdx.x * 1];
          r10[4] = v207_lin;
          float v208_lin = glb_m8[80 + threadIdx.x * 1];
          r10[5] = v208_lin;
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v211_data = r9[0];
          float v212_data = r9[1];
          float v213_data = r9[2];
          float v214_data = r9[3];
          float v215_data = r9[4];
          float v216_data = r9[5];
          float v217_data = r9[6];
          float v218_data = r9[7];
          float v219_data = r9[8];
          float v220_data = r9[9];
          float v221_data = r9[10];
          float v222_data = r9[11];
          float v223_acc{};
          float v224_acc{};
          float v225_acc{};
          float v226_acc{};
          float v227_acc{};
          float v228_acc{};
          float v229_acc{};
          float v230_acc{};
          float v231_lin = r10[0];
          tensorforge::fmacdpp16<0>(v223_acc, v231_lin, v211_data);
          tensorforge::fmacdpp16<1>(v223_acc, v231_lin, v212_data);
          tensorforge::fmacdpp16<2>(v223_acc, v231_lin, v213_data);
          tensorforge::fmacdpp16<3>(v223_acc, v231_lin, v214_data);
          tensorforge::fmacdpp16<4>(v223_acc, v231_lin, v215_data);
          tensorforge::fmacdpp16<5>(v223_acc, v231_lin, v216_data);
          tensorforge::fmacdpp16<6>(v223_acc, v231_lin, v217_data);
          tensorforge::fmacdpp16<7>(v223_acc, v231_lin, v218_data);
          tensorforge::fmacdpp16<8>(v223_acc, v231_lin, v219_data);
          tensorforge::fmacdpp16<9>(v223_acc, v231_lin, v220_data);
          tensorforge::fmacdpp16<10>(v223_acc, v231_lin, v221_data);
          tensorforge::fmacdpp16<11>(v223_acc, v231_lin, v222_data);
          tensorforge::fmacdpp16<12>(v224_acc, v231_lin, v211_data);
          tensorforge::fmacdpp16<13>(v224_acc, v231_lin, v212_data);
          tensorforge::fmacdpp16<14>(v224_acc, v231_lin, v213_data);
          tensorforge::fmacdpp16<15>(v224_acc, v231_lin, v214_data);
          float v232_lin = r10[1];
          tensorforge::fmacdpp16<0>(v224_acc, v232_lin, v215_data);
          tensorforge::fmacdpp16<1>(v224_acc, v232_lin, v216_data);
          tensorforge::fmacdpp16<2>(v224_acc, v232_lin, v217_data);
          tensorforge::fmacdpp16<3>(v224_acc, v232_lin, v218_data);
          tensorforge::fmacdpp16<4>(v224_acc, v232_lin, v219_data);
          tensorforge::fmacdpp16<5>(v224_acc, v232_lin, v220_data);
          tensorforge::fmacdpp16<6>(v224_acc, v232_lin, v221_data);
          tensorforge::fmacdpp16<7>(v224_acc, v232_lin, v222_data);
          tensorforge::fmacdpp16<8>(v225_acc, v232_lin, v211_data);
          tensorforge::fmacdpp16<9>(v225_acc, v232_lin, v212_data);
          tensorforge::fmacdpp16<10>(v225_acc, v232_lin, v213_data);
          tensorforge::fmacdpp16<11>(v225_acc, v232_lin, v214_data);
          tensorforge::fmacdpp16<12>(v225_acc, v232_lin, v215_data);
          tensorforge::fmacdpp16<13>(v225_acc, v232_lin, v216_data);
          tensorforge::fmacdpp16<14>(v225_acc, v232_lin, v217_data);
          tensorforge::fmacdpp16<15>(v225_acc, v232_lin, v218_data);
          float v233_lin = r10[2];
          tensorforge::fmacdpp16<0>(v225_acc, v233_lin, v219_data);
          tensorforge::fmacdpp16<1>(v225_acc, v233_lin, v220_data);
          tensorforge::fmacdpp16<2>(v225_acc, v233_lin, v221_data);
          tensorforge::fmacdpp16<3>(v225_acc, v233_lin, v222_data);
          tensorforge::fmacdpp16<4>(v226_acc, v233_lin, v211_data);
          tensorforge::fmacdpp16<5>(v226_acc, v233_lin, v212_data);
          tensorforge::fmacdpp16<6>(v226_acc, v233_lin, v213_data);
          tensorforge::fmacdpp16<7>(v226_acc, v233_lin, v214_data);
          tensorforge::fmacdpp16<8>(v226_acc, v233_lin, v215_data);
          tensorforge::fmacdpp16<9>(v226_acc, v233_lin, v216_data);
          tensorforge::fmacdpp16<10>(v226_acc, v233_lin, v217_data);
          tensorforge::fmacdpp16<11>(v226_acc, v233_lin, v218_data);
          tensorforge::fmacdpp16<12>(v226_acc, v233_lin, v219_data);
          tensorforge::fmacdpp16<13>(v226_acc, v233_lin, v220_data);
          tensorforge::fmacdpp16<14>(v226_acc, v233_lin, v221_data);
          tensorforge::fmacdpp16<15>(v226_acc, v233_lin, v222_data);
          float v234_lin = r10[3];
          tensorforge::fmacdpp16<0>(v227_acc, v234_lin, v211_data);
          tensorforge::fmacdpp16<1>(v227_acc, v234_lin, v212_data);
          tensorforge::fmacdpp16<2>(v227_acc, v234_lin, v213_data);
          tensorforge::fmacdpp16<3>(v227_acc, v234_lin, v214_data);
          tensorforge::fmacdpp16<4>(v227_acc, v234_lin, v215_data);
          tensorforge::fmacdpp16<5>(v227_acc, v234_lin, v216_data);
          tensorforge::fmacdpp16<6>(v227_acc, v234_lin, v217_data);
          tensorforge::fmacdpp16<7>(v227_acc, v234_lin, v218_data);
          tensorforge::fmacdpp16<8>(v227_acc, v234_lin, v219_data);
          tensorforge::fmacdpp16<9>(v227_acc, v234_lin, v220_data);
          tensorforge::fmacdpp16<10>(v227_acc, v234_lin, v221_data);
          tensorforge::fmacdpp16<11>(v227_acc, v234_lin, v222_data);
          tensorforge::fmacdpp16<12>(v228_acc, v234_lin, v211_data);
          tensorforge::fmacdpp16<13>(v228_acc, v234_lin, v212_data);
          tensorforge::fmacdpp16<14>(v228_acc, v234_lin, v213_data);
          tensorforge::fmacdpp16<15>(v228_acc, v234_lin, v214_data);
          float v235_lin = r10[4];
          tensorforge::fmacdpp16<0>(v228_acc, v235_lin, v215_data);
          tensorforge::fmacdpp16<1>(v228_acc, v235_lin, v216_data);
          tensorforge::fmacdpp16<2>(v228_acc, v235_lin, v217_data);
          tensorforge::fmacdpp16<3>(v228_acc, v235_lin, v218_data);
          tensorforge::fmacdpp16<4>(v228_acc, v235_lin, v219_data);
          tensorforge::fmacdpp16<5>(v228_acc, v235_lin, v220_data);
          tensorforge::fmacdpp16<6>(v228_acc, v235_lin, v221_data);
          tensorforge::fmacdpp16<7>(v228_acc, v235_lin, v222_data);
          tensorforge::fmacdpp16<8>(v229_acc, v235_lin, v211_data);
          tensorforge::fmacdpp16<9>(v229_acc, v235_lin, v212_data);
          tensorforge::fmacdpp16<10>(v229_acc, v235_lin, v213_data);
          tensorforge::fmacdpp16<11>(v229_acc, v235_lin, v214_data);
          tensorforge::fmacdpp16<12>(v229_acc, v235_lin, v215_data);
          tensorforge::fmacdpp16<13>(v229_acc, v235_lin, v216_data);
          tensorforge::fmacdpp16<14>(v229_acc, v235_lin, v217_data);
          tensorforge::fmacdpp16<15>(v229_acc, v235_lin, v218_data);
          float v236_lin = r10[5];
          tensorforge::fmacdpp16<0>(v229_acc, v236_lin, v219_data);
          tensorforge::fmacdpp16<1>(v229_acc, v236_lin, v220_data);
          tensorforge::fmacdpp16<2>(v229_acc, v236_lin, v221_data);
          tensorforge::fmacdpp16<3>(v229_acc, v236_lin, v222_data);
          tensorforge::fmacdpp16<4>(v230_acc, v236_lin, v211_data);
          tensorforge::fmacdpp16<5>(v230_acc, v236_lin, v212_data);
          tensorforge::fmacdpp16<6>(v230_acc, v236_lin, v213_data);
          tensorforge::fmacdpp16<7>(v230_acc, v236_lin, v214_data);
          tensorforge::fmacdpp16<8>(v230_acc, v236_lin, v215_data);
          tensorforge::fmacdpp16<9>(v230_acc, v236_lin, v216_data);
          tensorforge::fmacdpp16<10>(v230_acc, v236_lin, v217_data);
          tensorforge::fmacdpp16<11>(v230_acc, v236_lin, v218_data);
          tensorforge::fmacdpp16<12>(v230_acc, v236_lin, v219_data);
          tensorforge::fmacdpp16<13>(v230_acc, v236_lin, v220_data);
          tensorforge::fmacdpp16<14>(v230_acc, v236_lin, v221_data);
          tensorforge::fmacdpp16<15>(v230_acc, v236_lin, v222_data);
          ir11[0] = v223_acc;
          ir11[1] = v224_acc;
          ir11[2] = v225_acc;
          ir11[3] = v226_acc;
          ir11[4] = v227_acc;
          ir11[5] = v228_acc;
          ir11[6] = v229_acc;
          ir11[7] = v230_acc;
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v241_n1 = 0; v241_n1 < 8; ++v241_n1) {
              float v243_data = ir11[v241_n1];
              float v245_data = r8[v241_n1];
              r11[v241_n1] = (v245_data + v243_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v252_i1 = 0; v252_i1 < 8; ++v252_i1) {
              float v254_data = r11[v252_i1];
              glb_m0[(v19_lead + (v252_i1 * 12))] = v254_data;
            }
          }
        }
      }
    }
  }
}

