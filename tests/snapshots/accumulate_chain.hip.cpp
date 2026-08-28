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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v18_i1 * 12))]);
              r0[v18_i1] = v26_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v29_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          float v30_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v30_lin;
          float v31_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v31_lin;
          float v32_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v32_lin;
          float v33_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v33_lin;
          float v34_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v34_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m3[(v16_lead + (v40_i1 * 12))]);
              r3[v40_i1] = v48_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v51_data = r0[0];
          float v52_data = r0[1];
          float v53_data = r0[2];
          float v54_data = r0[3];
          float v55_data = r0[4];
          float v56_data = r0[5];
          float v57_data = r0[6];
          float v58_data = r0[7];
          float v59_data = r0[8];
          float v60_data = r0[9];
          float v61_data = r0[10];
          float v62_data = r0[11];
          float v63_acc{};
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_lin = r1[0];
          tensorforge::fmacdpp16<0>(v63_acc, v71_lin, v51_data);
          tensorforge::fmacdpp16<1>(v63_acc, v71_lin, v52_data);
          tensorforge::fmacdpp16<2>(v63_acc, v71_lin, v53_data);
          tensorforge::fmacdpp16<3>(v63_acc, v71_lin, v54_data);
          tensorforge::fmacdpp16<4>(v63_acc, v71_lin, v55_data);
          tensorforge::fmacdpp16<5>(v63_acc, v71_lin, v56_data);
          tensorforge::fmacdpp16<6>(v63_acc, v71_lin, v57_data);
          tensorforge::fmacdpp16<7>(v63_acc, v71_lin, v58_data);
          tensorforge::fmacdpp16<8>(v63_acc, v71_lin, v59_data);
          tensorforge::fmacdpp16<9>(v63_acc, v71_lin, v60_data);
          tensorforge::fmacdpp16<10>(v63_acc, v71_lin, v61_data);
          tensorforge::fmacdpp16<11>(v63_acc, v71_lin, v62_data);
          tensorforge::fmacdpp16<12>(v64_acc, v71_lin, v51_data);
          tensorforge::fmacdpp16<13>(v64_acc, v71_lin, v52_data);
          tensorforge::fmacdpp16<14>(v64_acc, v71_lin, v53_data);
          tensorforge::fmacdpp16<15>(v64_acc, v71_lin, v54_data);
          float v72_lin = r1[1];
          tensorforge::fmacdpp16<0>(v64_acc, v72_lin, v55_data);
          tensorforge::fmacdpp16<1>(v64_acc, v72_lin, v56_data);
          tensorforge::fmacdpp16<2>(v64_acc, v72_lin, v57_data);
          tensorforge::fmacdpp16<3>(v64_acc, v72_lin, v58_data);
          tensorforge::fmacdpp16<4>(v64_acc, v72_lin, v59_data);
          tensorforge::fmacdpp16<5>(v64_acc, v72_lin, v60_data);
          tensorforge::fmacdpp16<6>(v64_acc, v72_lin, v61_data);
          tensorforge::fmacdpp16<7>(v64_acc, v72_lin, v62_data);
          tensorforge::fmacdpp16<8>(v65_acc, v72_lin, v51_data);
          tensorforge::fmacdpp16<9>(v65_acc, v72_lin, v52_data);
          tensorforge::fmacdpp16<10>(v65_acc, v72_lin, v53_data);
          tensorforge::fmacdpp16<11>(v65_acc, v72_lin, v54_data);
          tensorforge::fmacdpp16<12>(v65_acc, v72_lin, v55_data);
          tensorforge::fmacdpp16<13>(v65_acc, v72_lin, v56_data);
          tensorforge::fmacdpp16<14>(v65_acc, v72_lin, v57_data);
          tensorforge::fmacdpp16<15>(v65_acc, v72_lin, v58_data);
          float v73_lin = r1[2];
          tensorforge::fmacdpp16<0>(v65_acc, v73_lin, v59_data);
          tensorforge::fmacdpp16<1>(v65_acc, v73_lin, v60_data);
          tensorforge::fmacdpp16<2>(v65_acc, v73_lin, v61_data);
          tensorforge::fmacdpp16<3>(v65_acc, v73_lin, v62_data);
          tensorforge::fmacdpp16<4>(v66_acc, v73_lin, v51_data);
          tensorforge::fmacdpp16<5>(v66_acc, v73_lin, v52_data);
          tensorforge::fmacdpp16<6>(v66_acc, v73_lin, v53_data);
          tensorforge::fmacdpp16<7>(v66_acc, v73_lin, v54_data);
          tensorforge::fmacdpp16<8>(v66_acc, v73_lin, v55_data);
          tensorforge::fmacdpp16<9>(v66_acc, v73_lin, v56_data);
          tensorforge::fmacdpp16<10>(v66_acc, v73_lin, v57_data);
          tensorforge::fmacdpp16<11>(v66_acc, v73_lin, v58_data);
          tensorforge::fmacdpp16<12>(v66_acc, v73_lin, v59_data);
          tensorforge::fmacdpp16<13>(v66_acc, v73_lin, v60_data);
          tensorforge::fmacdpp16<14>(v66_acc, v73_lin, v61_data);
          tensorforge::fmacdpp16<15>(v66_acc, v73_lin, v62_data);
          float v74_lin = r1[3];
          tensorforge::fmacdpp16<0>(v67_acc, v74_lin, v51_data);
          tensorforge::fmacdpp16<1>(v67_acc, v74_lin, v52_data);
          tensorforge::fmacdpp16<2>(v67_acc, v74_lin, v53_data);
          tensorforge::fmacdpp16<3>(v67_acc, v74_lin, v54_data);
          tensorforge::fmacdpp16<4>(v67_acc, v74_lin, v55_data);
          tensorforge::fmacdpp16<5>(v67_acc, v74_lin, v56_data);
          tensorforge::fmacdpp16<6>(v67_acc, v74_lin, v57_data);
          tensorforge::fmacdpp16<7>(v67_acc, v74_lin, v58_data);
          tensorforge::fmacdpp16<8>(v67_acc, v74_lin, v59_data);
          tensorforge::fmacdpp16<9>(v67_acc, v74_lin, v60_data);
          tensorforge::fmacdpp16<10>(v67_acc, v74_lin, v61_data);
          tensorforge::fmacdpp16<11>(v67_acc, v74_lin, v62_data);
          tensorforge::fmacdpp16<12>(v68_acc, v74_lin, v51_data);
          tensorforge::fmacdpp16<13>(v68_acc, v74_lin, v52_data);
          tensorforge::fmacdpp16<14>(v68_acc, v74_lin, v53_data);
          tensorforge::fmacdpp16<15>(v68_acc, v74_lin, v54_data);
          float v75_lin = r1[4];
          tensorforge::fmacdpp16<0>(v68_acc, v75_lin, v55_data);
          tensorforge::fmacdpp16<1>(v68_acc, v75_lin, v56_data);
          tensorforge::fmacdpp16<2>(v68_acc, v75_lin, v57_data);
          tensorforge::fmacdpp16<3>(v68_acc, v75_lin, v58_data);
          tensorforge::fmacdpp16<4>(v68_acc, v75_lin, v59_data);
          tensorforge::fmacdpp16<5>(v68_acc, v75_lin, v60_data);
          tensorforge::fmacdpp16<6>(v68_acc, v75_lin, v61_data);
          tensorforge::fmacdpp16<7>(v68_acc, v75_lin, v62_data);
          tensorforge::fmacdpp16<8>(v69_acc, v75_lin, v51_data);
          tensorforge::fmacdpp16<9>(v69_acc, v75_lin, v52_data);
          tensorforge::fmacdpp16<10>(v69_acc, v75_lin, v53_data);
          tensorforge::fmacdpp16<11>(v69_acc, v75_lin, v54_data);
          tensorforge::fmacdpp16<12>(v69_acc, v75_lin, v55_data);
          tensorforge::fmacdpp16<13>(v69_acc, v75_lin, v56_data);
          tensorforge::fmacdpp16<14>(v69_acc, v75_lin, v57_data);
          tensorforge::fmacdpp16<15>(v69_acc, v75_lin, v58_data);
          float v76_lin = r1[5];
          tensorforge::fmacdpp16<0>(v69_acc, v76_lin, v59_data);
          tensorforge::fmacdpp16<1>(v69_acc, v76_lin, v60_data);
          tensorforge::fmacdpp16<2>(v69_acc, v76_lin, v61_data);
          tensorforge::fmacdpp16<3>(v69_acc, v76_lin, v62_data);
          tensorforge::fmacdpp16<4>(v70_acc, v76_lin, v51_data);
          tensorforge::fmacdpp16<5>(v70_acc, v76_lin, v52_data);
          tensorforge::fmacdpp16<6>(v70_acc, v76_lin, v53_data);
          tensorforge::fmacdpp16<7>(v70_acc, v76_lin, v54_data);
          tensorforge::fmacdpp16<8>(v70_acc, v76_lin, v55_data);
          tensorforge::fmacdpp16<9>(v70_acc, v76_lin, v56_data);
          tensorforge::fmacdpp16<10>(v70_acc, v76_lin, v57_data);
          tensorforge::fmacdpp16<11>(v70_acc, v76_lin, v58_data);
          tensorforge::fmacdpp16<12>(v70_acc, v76_lin, v59_data);
          tensorforge::fmacdpp16<13>(v70_acc, v76_lin, v60_data);
          tensorforge::fmacdpp16<14>(v70_acc, v76_lin, v61_data);
          tensorforge::fmacdpp16<15>(v70_acc, v76_lin, v62_data);
          r2[0] = v63_acc;
          r2[1] = v64_acc;
          r2[2] = v65_acc;
          r2[3] = v66_acc;
          r2[4] = v67_acc;
          r2[5] = v68_acc;
          r2[6] = v69_acc;
          r2[7] = v70_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v78_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v78_lin;
          float v79_lin = glb_m4[16 + threadIdx.x * 1];
          r4[1] = v79_lin;
          float v80_lin = glb_m4[32 + threadIdx.x * 1];
          r4[2] = v80_lin;
          float v81_lin = glb_m4[48 + threadIdx.x * 1];
          r4[3] = v81_lin;
          float v82_lin = glb_m4[64 + threadIdx.x * 1];
          r4[4] = v82_lin;
          float v83_lin = glb_m4[80 + threadIdx.x * 1];
          r4[5] = v83_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v89_i1 = 0; v89_i1 < 12; ++v89_i1) {
              float v97_data = __builtin_nontemporal_load(&glb_m5[(v16_lead + (v89_i1 * 12))]);
              r6[v89_i1] = v97_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v101_data = r3[0];
          float v102_data = r3[1];
          float v103_data = r3[2];
          float v104_data = r3[3];
          float v105_data = r3[4];
          float v106_data = r3[5];
          float v107_data = r3[6];
          float v108_data = r3[7];
          float v109_data = r3[8];
          float v110_data = r3[9];
          float v111_data = r3[10];
          float v112_data = r3[11];
          float v113_acc{};
          float v114_acc{};
          float v115_acc{};
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_lin = r4[0];
          tensorforge::fmacdpp16<0>(v113_acc, v121_lin, v101_data);
          tensorforge::fmacdpp16<1>(v113_acc, v121_lin, v102_data);
          tensorforge::fmacdpp16<2>(v113_acc, v121_lin, v103_data);
          tensorforge::fmacdpp16<3>(v113_acc, v121_lin, v104_data);
          tensorforge::fmacdpp16<4>(v113_acc, v121_lin, v105_data);
          tensorforge::fmacdpp16<5>(v113_acc, v121_lin, v106_data);
          tensorforge::fmacdpp16<6>(v113_acc, v121_lin, v107_data);
          tensorforge::fmacdpp16<7>(v113_acc, v121_lin, v108_data);
          tensorforge::fmacdpp16<8>(v113_acc, v121_lin, v109_data);
          tensorforge::fmacdpp16<9>(v113_acc, v121_lin, v110_data);
          tensorforge::fmacdpp16<10>(v113_acc, v121_lin, v111_data);
          tensorforge::fmacdpp16<11>(v113_acc, v121_lin, v112_data);
          tensorforge::fmacdpp16<12>(v114_acc, v121_lin, v101_data);
          tensorforge::fmacdpp16<13>(v114_acc, v121_lin, v102_data);
          tensorforge::fmacdpp16<14>(v114_acc, v121_lin, v103_data);
          tensorforge::fmacdpp16<15>(v114_acc, v121_lin, v104_data);
          float v122_lin = r4[1];
          tensorforge::fmacdpp16<0>(v114_acc, v122_lin, v105_data);
          tensorforge::fmacdpp16<1>(v114_acc, v122_lin, v106_data);
          tensorforge::fmacdpp16<2>(v114_acc, v122_lin, v107_data);
          tensorforge::fmacdpp16<3>(v114_acc, v122_lin, v108_data);
          tensorforge::fmacdpp16<4>(v114_acc, v122_lin, v109_data);
          tensorforge::fmacdpp16<5>(v114_acc, v122_lin, v110_data);
          tensorforge::fmacdpp16<6>(v114_acc, v122_lin, v111_data);
          tensorforge::fmacdpp16<7>(v114_acc, v122_lin, v112_data);
          tensorforge::fmacdpp16<8>(v115_acc, v122_lin, v101_data);
          tensorforge::fmacdpp16<9>(v115_acc, v122_lin, v102_data);
          tensorforge::fmacdpp16<10>(v115_acc, v122_lin, v103_data);
          tensorforge::fmacdpp16<11>(v115_acc, v122_lin, v104_data);
          tensorforge::fmacdpp16<12>(v115_acc, v122_lin, v105_data);
          tensorforge::fmacdpp16<13>(v115_acc, v122_lin, v106_data);
          tensorforge::fmacdpp16<14>(v115_acc, v122_lin, v107_data);
          tensorforge::fmacdpp16<15>(v115_acc, v122_lin, v108_data);
          float v123_lin = r4[2];
          tensorforge::fmacdpp16<0>(v115_acc, v123_lin, v109_data);
          tensorforge::fmacdpp16<1>(v115_acc, v123_lin, v110_data);
          tensorforge::fmacdpp16<2>(v115_acc, v123_lin, v111_data);
          tensorforge::fmacdpp16<3>(v115_acc, v123_lin, v112_data);
          tensorforge::fmacdpp16<4>(v116_acc, v123_lin, v101_data);
          tensorforge::fmacdpp16<5>(v116_acc, v123_lin, v102_data);
          tensorforge::fmacdpp16<6>(v116_acc, v123_lin, v103_data);
          tensorforge::fmacdpp16<7>(v116_acc, v123_lin, v104_data);
          tensorforge::fmacdpp16<8>(v116_acc, v123_lin, v105_data);
          tensorforge::fmacdpp16<9>(v116_acc, v123_lin, v106_data);
          tensorforge::fmacdpp16<10>(v116_acc, v123_lin, v107_data);
          tensorforge::fmacdpp16<11>(v116_acc, v123_lin, v108_data);
          tensorforge::fmacdpp16<12>(v116_acc, v123_lin, v109_data);
          tensorforge::fmacdpp16<13>(v116_acc, v123_lin, v110_data);
          tensorforge::fmacdpp16<14>(v116_acc, v123_lin, v111_data);
          tensorforge::fmacdpp16<15>(v116_acc, v123_lin, v112_data);
          float v124_lin = r4[3];
          tensorforge::fmacdpp16<0>(v117_acc, v124_lin, v101_data);
          tensorforge::fmacdpp16<1>(v117_acc, v124_lin, v102_data);
          tensorforge::fmacdpp16<2>(v117_acc, v124_lin, v103_data);
          tensorforge::fmacdpp16<3>(v117_acc, v124_lin, v104_data);
          tensorforge::fmacdpp16<4>(v117_acc, v124_lin, v105_data);
          tensorforge::fmacdpp16<5>(v117_acc, v124_lin, v106_data);
          tensorforge::fmacdpp16<6>(v117_acc, v124_lin, v107_data);
          tensorforge::fmacdpp16<7>(v117_acc, v124_lin, v108_data);
          tensorforge::fmacdpp16<8>(v117_acc, v124_lin, v109_data);
          tensorforge::fmacdpp16<9>(v117_acc, v124_lin, v110_data);
          tensorforge::fmacdpp16<10>(v117_acc, v124_lin, v111_data);
          tensorforge::fmacdpp16<11>(v117_acc, v124_lin, v112_data);
          tensorforge::fmacdpp16<12>(v118_acc, v124_lin, v101_data);
          tensorforge::fmacdpp16<13>(v118_acc, v124_lin, v102_data);
          tensorforge::fmacdpp16<14>(v118_acc, v124_lin, v103_data);
          tensorforge::fmacdpp16<15>(v118_acc, v124_lin, v104_data);
          float v125_lin = r4[4];
          tensorforge::fmacdpp16<0>(v118_acc, v125_lin, v105_data);
          tensorforge::fmacdpp16<1>(v118_acc, v125_lin, v106_data);
          tensorforge::fmacdpp16<2>(v118_acc, v125_lin, v107_data);
          tensorforge::fmacdpp16<3>(v118_acc, v125_lin, v108_data);
          tensorforge::fmacdpp16<4>(v118_acc, v125_lin, v109_data);
          tensorforge::fmacdpp16<5>(v118_acc, v125_lin, v110_data);
          tensorforge::fmacdpp16<6>(v118_acc, v125_lin, v111_data);
          tensorforge::fmacdpp16<7>(v118_acc, v125_lin, v112_data);
          tensorforge::fmacdpp16<8>(v119_acc, v125_lin, v101_data);
          tensorforge::fmacdpp16<9>(v119_acc, v125_lin, v102_data);
          tensorforge::fmacdpp16<10>(v119_acc, v125_lin, v103_data);
          tensorforge::fmacdpp16<11>(v119_acc, v125_lin, v104_data);
          tensorforge::fmacdpp16<12>(v119_acc, v125_lin, v105_data);
          tensorforge::fmacdpp16<13>(v119_acc, v125_lin, v106_data);
          tensorforge::fmacdpp16<14>(v119_acc, v125_lin, v107_data);
          tensorforge::fmacdpp16<15>(v119_acc, v125_lin, v108_data);
          float v126_lin = r4[5];
          tensorforge::fmacdpp16<0>(v119_acc, v126_lin, v109_data);
          tensorforge::fmacdpp16<1>(v119_acc, v126_lin, v110_data);
          tensorforge::fmacdpp16<2>(v119_acc, v126_lin, v111_data);
          tensorforge::fmacdpp16<3>(v119_acc, v126_lin, v112_data);
          tensorforge::fmacdpp16<4>(v120_acc, v126_lin, v101_data);
          tensorforge::fmacdpp16<5>(v120_acc, v126_lin, v102_data);
          tensorforge::fmacdpp16<6>(v120_acc, v126_lin, v103_data);
          tensorforge::fmacdpp16<7>(v120_acc, v126_lin, v104_data);
          tensorforge::fmacdpp16<8>(v120_acc, v126_lin, v105_data);
          tensorforge::fmacdpp16<9>(v120_acc, v126_lin, v106_data);
          tensorforge::fmacdpp16<10>(v120_acc, v126_lin, v107_data);
          tensorforge::fmacdpp16<11>(v120_acc, v126_lin, v108_data);
          tensorforge::fmacdpp16<12>(v120_acc, v126_lin, v109_data);
          tensorforge::fmacdpp16<13>(v120_acc, v126_lin, v110_data);
          tensorforge::fmacdpp16<14>(v120_acc, v126_lin, v111_data);
          tensorforge::fmacdpp16<15>(v120_acc, v126_lin, v112_data);
          ir5[0] = v113_acc;
          ir5[1] = v114_acc;
          ir5[2] = v115_acc;
          ir5[3] = v116_acc;
          ir5[4] = v117_acc;
          ir5[5] = v118_acc;
          ir5[6] = v119_acc;
          ir5[7] = v120_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v131_n1 = 0; v131_n1 < 8; ++v131_n1) {
              float v133_data = ir5[v131_n1];
              float v135_data = r2[v131_n1];
              r5[v131_n1] = (v135_data + v133_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v139_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v139_lin;
          float v140_lin = glb_m6[16 + threadIdx.x * 1];
          r7[1] = v140_lin;
          float v141_lin = glb_m6[32 + threadIdx.x * 1];
          r7[2] = v141_lin;
          float v142_lin = glb_m6[48 + threadIdx.x * 1];
          r7[3] = v142_lin;
          float v143_lin = glb_m6[64 + threadIdx.x * 1];
          r7[4] = v143_lin;
          float v144_lin = glb_m6[80 + threadIdx.x * 1];
          r7[5] = v144_lin;
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v150_i1 = 0; v150_i1 < 12; ++v150_i1) {
              float v158_data = __builtin_nontemporal_load(&glb_m7[(v16_lead + (v150_i1 * 12))]);
              r9[v150_i1] = v158_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v162_data = r6[0];
          float v163_data = r6[1];
          float v164_data = r6[2];
          float v165_data = r6[3];
          float v166_data = r6[4];
          float v167_data = r6[5];
          float v168_data = r6[6];
          float v169_data = r6[7];
          float v170_data = r6[8];
          float v171_data = r6[9];
          float v172_data = r6[10];
          float v173_data = r6[11];
          float v174_acc{};
          float v175_acc{};
          float v176_acc{};
          float v177_acc{};
          float v178_acc{};
          float v179_acc{};
          float v180_acc{};
          float v181_acc{};
          float v182_lin = r7[0];
          tensorforge::fmacdpp16<0>(v174_acc, v182_lin, v162_data);
          tensorforge::fmacdpp16<1>(v174_acc, v182_lin, v163_data);
          tensorforge::fmacdpp16<2>(v174_acc, v182_lin, v164_data);
          tensorforge::fmacdpp16<3>(v174_acc, v182_lin, v165_data);
          tensorforge::fmacdpp16<4>(v174_acc, v182_lin, v166_data);
          tensorforge::fmacdpp16<5>(v174_acc, v182_lin, v167_data);
          tensorforge::fmacdpp16<6>(v174_acc, v182_lin, v168_data);
          tensorforge::fmacdpp16<7>(v174_acc, v182_lin, v169_data);
          tensorforge::fmacdpp16<8>(v174_acc, v182_lin, v170_data);
          tensorforge::fmacdpp16<9>(v174_acc, v182_lin, v171_data);
          tensorforge::fmacdpp16<10>(v174_acc, v182_lin, v172_data);
          tensorforge::fmacdpp16<11>(v174_acc, v182_lin, v173_data);
          tensorforge::fmacdpp16<12>(v175_acc, v182_lin, v162_data);
          tensorforge::fmacdpp16<13>(v175_acc, v182_lin, v163_data);
          tensorforge::fmacdpp16<14>(v175_acc, v182_lin, v164_data);
          tensorforge::fmacdpp16<15>(v175_acc, v182_lin, v165_data);
          float v183_lin = r7[1];
          tensorforge::fmacdpp16<0>(v175_acc, v183_lin, v166_data);
          tensorforge::fmacdpp16<1>(v175_acc, v183_lin, v167_data);
          tensorforge::fmacdpp16<2>(v175_acc, v183_lin, v168_data);
          tensorforge::fmacdpp16<3>(v175_acc, v183_lin, v169_data);
          tensorforge::fmacdpp16<4>(v175_acc, v183_lin, v170_data);
          tensorforge::fmacdpp16<5>(v175_acc, v183_lin, v171_data);
          tensorforge::fmacdpp16<6>(v175_acc, v183_lin, v172_data);
          tensorforge::fmacdpp16<7>(v175_acc, v183_lin, v173_data);
          tensorforge::fmacdpp16<8>(v176_acc, v183_lin, v162_data);
          tensorforge::fmacdpp16<9>(v176_acc, v183_lin, v163_data);
          tensorforge::fmacdpp16<10>(v176_acc, v183_lin, v164_data);
          tensorforge::fmacdpp16<11>(v176_acc, v183_lin, v165_data);
          tensorforge::fmacdpp16<12>(v176_acc, v183_lin, v166_data);
          tensorforge::fmacdpp16<13>(v176_acc, v183_lin, v167_data);
          tensorforge::fmacdpp16<14>(v176_acc, v183_lin, v168_data);
          tensorforge::fmacdpp16<15>(v176_acc, v183_lin, v169_data);
          float v184_lin = r7[2];
          tensorforge::fmacdpp16<0>(v176_acc, v184_lin, v170_data);
          tensorforge::fmacdpp16<1>(v176_acc, v184_lin, v171_data);
          tensorforge::fmacdpp16<2>(v176_acc, v184_lin, v172_data);
          tensorforge::fmacdpp16<3>(v176_acc, v184_lin, v173_data);
          tensorforge::fmacdpp16<4>(v177_acc, v184_lin, v162_data);
          tensorforge::fmacdpp16<5>(v177_acc, v184_lin, v163_data);
          tensorforge::fmacdpp16<6>(v177_acc, v184_lin, v164_data);
          tensorforge::fmacdpp16<7>(v177_acc, v184_lin, v165_data);
          tensorforge::fmacdpp16<8>(v177_acc, v184_lin, v166_data);
          tensorforge::fmacdpp16<9>(v177_acc, v184_lin, v167_data);
          tensorforge::fmacdpp16<10>(v177_acc, v184_lin, v168_data);
          tensorforge::fmacdpp16<11>(v177_acc, v184_lin, v169_data);
          tensorforge::fmacdpp16<12>(v177_acc, v184_lin, v170_data);
          tensorforge::fmacdpp16<13>(v177_acc, v184_lin, v171_data);
          tensorforge::fmacdpp16<14>(v177_acc, v184_lin, v172_data);
          tensorforge::fmacdpp16<15>(v177_acc, v184_lin, v173_data);
          float v185_lin = r7[3];
          tensorforge::fmacdpp16<0>(v178_acc, v185_lin, v162_data);
          tensorforge::fmacdpp16<1>(v178_acc, v185_lin, v163_data);
          tensorforge::fmacdpp16<2>(v178_acc, v185_lin, v164_data);
          tensorforge::fmacdpp16<3>(v178_acc, v185_lin, v165_data);
          tensorforge::fmacdpp16<4>(v178_acc, v185_lin, v166_data);
          tensorforge::fmacdpp16<5>(v178_acc, v185_lin, v167_data);
          tensorforge::fmacdpp16<6>(v178_acc, v185_lin, v168_data);
          tensorforge::fmacdpp16<7>(v178_acc, v185_lin, v169_data);
          tensorforge::fmacdpp16<8>(v178_acc, v185_lin, v170_data);
          tensorforge::fmacdpp16<9>(v178_acc, v185_lin, v171_data);
          tensorforge::fmacdpp16<10>(v178_acc, v185_lin, v172_data);
          tensorforge::fmacdpp16<11>(v178_acc, v185_lin, v173_data);
          tensorforge::fmacdpp16<12>(v179_acc, v185_lin, v162_data);
          tensorforge::fmacdpp16<13>(v179_acc, v185_lin, v163_data);
          tensorforge::fmacdpp16<14>(v179_acc, v185_lin, v164_data);
          tensorforge::fmacdpp16<15>(v179_acc, v185_lin, v165_data);
          float v186_lin = r7[4];
          tensorforge::fmacdpp16<0>(v179_acc, v186_lin, v166_data);
          tensorforge::fmacdpp16<1>(v179_acc, v186_lin, v167_data);
          tensorforge::fmacdpp16<2>(v179_acc, v186_lin, v168_data);
          tensorforge::fmacdpp16<3>(v179_acc, v186_lin, v169_data);
          tensorforge::fmacdpp16<4>(v179_acc, v186_lin, v170_data);
          tensorforge::fmacdpp16<5>(v179_acc, v186_lin, v171_data);
          tensorforge::fmacdpp16<6>(v179_acc, v186_lin, v172_data);
          tensorforge::fmacdpp16<7>(v179_acc, v186_lin, v173_data);
          tensorforge::fmacdpp16<8>(v180_acc, v186_lin, v162_data);
          tensorforge::fmacdpp16<9>(v180_acc, v186_lin, v163_data);
          tensorforge::fmacdpp16<10>(v180_acc, v186_lin, v164_data);
          tensorforge::fmacdpp16<11>(v180_acc, v186_lin, v165_data);
          tensorforge::fmacdpp16<12>(v180_acc, v186_lin, v166_data);
          tensorforge::fmacdpp16<13>(v180_acc, v186_lin, v167_data);
          tensorforge::fmacdpp16<14>(v180_acc, v186_lin, v168_data);
          tensorforge::fmacdpp16<15>(v180_acc, v186_lin, v169_data);
          float v187_lin = r7[5];
          tensorforge::fmacdpp16<0>(v180_acc, v187_lin, v170_data);
          tensorforge::fmacdpp16<1>(v180_acc, v187_lin, v171_data);
          tensorforge::fmacdpp16<2>(v180_acc, v187_lin, v172_data);
          tensorforge::fmacdpp16<3>(v180_acc, v187_lin, v173_data);
          tensorforge::fmacdpp16<4>(v181_acc, v187_lin, v162_data);
          tensorforge::fmacdpp16<5>(v181_acc, v187_lin, v163_data);
          tensorforge::fmacdpp16<6>(v181_acc, v187_lin, v164_data);
          tensorforge::fmacdpp16<7>(v181_acc, v187_lin, v165_data);
          tensorforge::fmacdpp16<8>(v181_acc, v187_lin, v166_data);
          tensorforge::fmacdpp16<9>(v181_acc, v187_lin, v167_data);
          tensorforge::fmacdpp16<10>(v181_acc, v187_lin, v168_data);
          tensorforge::fmacdpp16<11>(v181_acc, v187_lin, v169_data);
          tensorforge::fmacdpp16<12>(v181_acc, v187_lin, v170_data);
          tensorforge::fmacdpp16<13>(v181_acc, v187_lin, v171_data);
          tensorforge::fmacdpp16<14>(v181_acc, v187_lin, v172_data);
          tensorforge::fmacdpp16<15>(v181_acc, v187_lin, v173_data);
          ir8[0] = v174_acc;
          ir8[1] = v175_acc;
          ir8[2] = v176_acc;
          ir8[3] = v177_acc;
          ir8[4] = v178_acc;
          ir8[5] = v179_acc;
          ir8[6] = v180_acc;
          ir8[7] = v181_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v192_n1 = 0; v192_n1 < 8; ++v192_n1) {
              float v194_data = ir8[v192_n1];
              float v196_data = r5[v192_n1];
              r8[v192_n1] = (v196_data + v194_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          float v200_lin = glb_m8[0 + threadIdx.x * 1];
          r10[0] = v200_lin;
          float v201_lin = glb_m8[16 + threadIdx.x * 1];
          r10[1] = v201_lin;
          float v202_lin = glb_m8[32 + threadIdx.x * 1];
          r10[2] = v202_lin;
          float v203_lin = glb_m8[48 + threadIdx.x * 1];
          r10[3] = v203_lin;
          float v204_lin = glb_m8[64 + threadIdx.x * 1];
          r10[4] = v204_lin;
          float v205_lin = glb_m8[80 + threadIdx.x * 1];
          r10[5] = v205_lin;
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v208_data = r9[0];
          float v209_data = r9[1];
          float v210_data = r9[2];
          float v211_data = r9[3];
          float v212_data = r9[4];
          float v213_data = r9[5];
          float v214_data = r9[6];
          float v215_data = r9[7];
          float v216_data = r9[8];
          float v217_data = r9[9];
          float v218_data = r9[10];
          float v219_data = r9[11];
          float v220_acc{};
          float v221_acc{};
          float v222_acc{};
          float v223_acc{};
          float v224_acc{};
          float v225_acc{};
          float v226_acc{};
          float v227_acc{};
          float v228_lin = r10[0];
          tensorforge::fmacdpp16<0>(v220_acc, v228_lin, v208_data);
          tensorforge::fmacdpp16<1>(v220_acc, v228_lin, v209_data);
          tensorforge::fmacdpp16<2>(v220_acc, v228_lin, v210_data);
          tensorforge::fmacdpp16<3>(v220_acc, v228_lin, v211_data);
          tensorforge::fmacdpp16<4>(v220_acc, v228_lin, v212_data);
          tensorforge::fmacdpp16<5>(v220_acc, v228_lin, v213_data);
          tensorforge::fmacdpp16<6>(v220_acc, v228_lin, v214_data);
          tensorforge::fmacdpp16<7>(v220_acc, v228_lin, v215_data);
          tensorforge::fmacdpp16<8>(v220_acc, v228_lin, v216_data);
          tensorforge::fmacdpp16<9>(v220_acc, v228_lin, v217_data);
          tensorforge::fmacdpp16<10>(v220_acc, v228_lin, v218_data);
          tensorforge::fmacdpp16<11>(v220_acc, v228_lin, v219_data);
          tensorforge::fmacdpp16<12>(v221_acc, v228_lin, v208_data);
          tensorforge::fmacdpp16<13>(v221_acc, v228_lin, v209_data);
          tensorforge::fmacdpp16<14>(v221_acc, v228_lin, v210_data);
          tensorforge::fmacdpp16<15>(v221_acc, v228_lin, v211_data);
          float v229_lin = r10[1];
          tensorforge::fmacdpp16<0>(v221_acc, v229_lin, v212_data);
          tensorforge::fmacdpp16<1>(v221_acc, v229_lin, v213_data);
          tensorforge::fmacdpp16<2>(v221_acc, v229_lin, v214_data);
          tensorforge::fmacdpp16<3>(v221_acc, v229_lin, v215_data);
          tensorforge::fmacdpp16<4>(v221_acc, v229_lin, v216_data);
          tensorforge::fmacdpp16<5>(v221_acc, v229_lin, v217_data);
          tensorforge::fmacdpp16<6>(v221_acc, v229_lin, v218_data);
          tensorforge::fmacdpp16<7>(v221_acc, v229_lin, v219_data);
          tensorforge::fmacdpp16<8>(v222_acc, v229_lin, v208_data);
          tensorforge::fmacdpp16<9>(v222_acc, v229_lin, v209_data);
          tensorforge::fmacdpp16<10>(v222_acc, v229_lin, v210_data);
          tensorforge::fmacdpp16<11>(v222_acc, v229_lin, v211_data);
          tensorforge::fmacdpp16<12>(v222_acc, v229_lin, v212_data);
          tensorforge::fmacdpp16<13>(v222_acc, v229_lin, v213_data);
          tensorforge::fmacdpp16<14>(v222_acc, v229_lin, v214_data);
          tensorforge::fmacdpp16<15>(v222_acc, v229_lin, v215_data);
          float v230_lin = r10[2];
          tensorforge::fmacdpp16<0>(v222_acc, v230_lin, v216_data);
          tensorforge::fmacdpp16<1>(v222_acc, v230_lin, v217_data);
          tensorforge::fmacdpp16<2>(v222_acc, v230_lin, v218_data);
          tensorforge::fmacdpp16<3>(v222_acc, v230_lin, v219_data);
          tensorforge::fmacdpp16<4>(v223_acc, v230_lin, v208_data);
          tensorforge::fmacdpp16<5>(v223_acc, v230_lin, v209_data);
          tensorforge::fmacdpp16<6>(v223_acc, v230_lin, v210_data);
          tensorforge::fmacdpp16<7>(v223_acc, v230_lin, v211_data);
          tensorforge::fmacdpp16<8>(v223_acc, v230_lin, v212_data);
          tensorforge::fmacdpp16<9>(v223_acc, v230_lin, v213_data);
          tensorforge::fmacdpp16<10>(v223_acc, v230_lin, v214_data);
          tensorforge::fmacdpp16<11>(v223_acc, v230_lin, v215_data);
          tensorforge::fmacdpp16<12>(v223_acc, v230_lin, v216_data);
          tensorforge::fmacdpp16<13>(v223_acc, v230_lin, v217_data);
          tensorforge::fmacdpp16<14>(v223_acc, v230_lin, v218_data);
          tensorforge::fmacdpp16<15>(v223_acc, v230_lin, v219_data);
          float v231_lin = r10[3];
          tensorforge::fmacdpp16<0>(v224_acc, v231_lin, v208_data);
          tensorforge::fmacdpp16<1>(v224_acc, v231_lin, v209_data);
          tensorforge::fmacdpp16<2>(v224_acc, v231_lin, v210_data);
          tensorforge::fmacdpp16<3>(v224_acc, v231_lin, v211_data);
          tensorforge::fmacdpp16<4>(v224_acc, v231_lin, v212_data);
          tensorforge::fmacdpp16<5>(v224_acc, v231_lin, v213_data);
          tensorforge::fmacdpp16<6>(v224_acc, v231_lin, v214_data);
          tensorforge::fmacdpp16<7>(v224_acc, v231_lin, v215_data);
          tensorforge::fmacdpp16<8>(v224_acc, v231_lin, v216_data);
          tensorforge::fmacdpp16<9>(v224_acc, v231_lin, v217_data);
          tensorforge::fmacdpp16<10>(v224_acc, v231_lin, v218_data);
          tensorforge::fmacdpp16<11>(v224_acc, v231_lin, v219_data);
          tensorforge::fmacdpp16<12>(v225_acc, v231_lin, v208_data);
          tensorforge::fmacdpp16<13>(v225_acc, v231_lin, v209_data);
          tensorforge::fmacdpp16<14>(v225_acc, v231_lin, v210_data);
          tensorforge::fmacdpp16<15>(v225_acc, v231_lin, v211_data);
          float v232_lin = r10[4];
          tensorforge::fmacdpp16<0>(v225_acc, v232_lin, v212_data);
          tensorforge::fmacdpp16<1>(v225_acc, v232_lin, v213_data);
          tensorforge::fmacdpp16<2>(v225_acc, v232_lin, v214_data);
          tensorforge::fmacdpp16<3>(v225_acc, v232_lin, v215_data);
          tensorforge::fmacdpp16<4>(v225_acc, v232_lin, v216_data);
          tensorforge::fmacdpp16<5>(v225_acc, v232_lin, v217_data);
          tensorforge::fmacdpp16<6>(v225_acc, v232_lin, v218_data);
          tensorforge::fmacdpp16<7>(v225_acc, v232_lin, v219_data);
          tensorforge::fmacdpp16<8>(v226_acc, v232_lin, v208_data);
          tensorforge::fmacdpp16<9>(v226_acc, v232_lin, v209_data);
          tensorforge::fmacdpp16<10>(v226_acc, v232_lin, v210_data);
          tensorforge::fmacdpp16<11>(v226_acc, v232_lin, v211_data);
          tensorforge::fmacdpp16<12>(v226_acc, v232_lin, v212_data);
          tensorforge::fmacdpp16<13>(v226_acc, v232_lin, v213_data);
          tensorforge::fmacdpp16<14>(v226_acc, v232_lin, v214_data);
          tensorforge::fmacdpp16<15>(v226_acc, v232_lin, v215_data);
          float v233_lin = r10[5];
          tensorforge::fmacdpp16<0>(v226_acc, v233_lin, v216_data);
          tensorforge::fmacdpp16<1>(v226_acc, v233_lin, v217_data);
          tensorforge::fmacdpp16<2>(v226_acc, v233_lin, v218_data);
          tensorforge::fmacdpp16<3>(v226_acc, v233_lin, v219_data);
          tensorforge::fmacdpp16<4>(v227_acc, v233_lin, v208_data);
          tensorforge::fmacdpp16<5>(v227_acc, v233_lin, v209_data);
          tensorforge::fmacdpp16<6>(v227_acc, v233_lin, v210_data);
          tensorforge::fmacdpp16<7>(v227_acc, v233_lin, v211_data);
          tensorforge::fmacdpp16<8>(v227_acc, v233_lin, v212_data);
          tensorforge::fmacdpp16<9>(v227_acc, v233_lin, v213_data);
          tensorforge::fmacdpp16<10>(v227_acc, v233_lin, v214_data);
          tensorforge::fmacdpp16<11>(v227_acc, v233_lin, v215_data);
          tensorforge::fmacdpp16<12>(v227_acc, v233_lin, v216_data);
          tensorforge::fmacdpp16<13>(v227_acc, v233_lin, v217_data);
          tensorforge::fmacdpp16<14>(v227_acc, v233_lin, v218_data);
          tensorforge::fmacdpp16<15>(v227_acc, v233_lin, v219_data);
          ir11[0] = v220_acc;
          ir11[1] = v221_acc;
          ir11[2] = v222_acc;
          ir11[3] = v223_acc;
          ir11[4] = v224_acc;
          ir11[5] = v225_acc;
          ir11[6] = v226_acc;
          ir11[7] = v227_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v238_n1 = 0; v238_n1 < 8; ++v238_n1) {
              float v240_data = ir11[v238_n1];
              float v242_data = r8[v238_n1];
              r11[v238_n1] = (v242_data + v240_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v249_i1 = 0; v249_i1 < 8; ++v249_i1) {
              float v251_data = r11[v249_i1];
              glb_m0[(v16_lead + (v249_i1 * 12))] = v251_data;
            }
          }
        }
      }
    }
  }
}

