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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v28_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              int32_t v40_a = v34_i1 * 12;
              int32_t v41_a = v3_lead + v40_a;
              float v49_data = __builtin_nontemporal_load(&glb_m3[(v3_lead + v40_a)]);
              int32_t v50_a = 0 + v34_i1;
              r3[v50_a] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          auto& ir2 = r2;
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          float v60_data = r0[8];
          float v61_data = r0[9];
          float v62_data = r0[10];
          float v63_data = r0[11];
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_lin = r1[0];
          tensorforge::fmacdpp16<0>(v64_acc, v72_lin, v52_data);
          tensorforge::fmacdpp16<1>(v64_acc, v72_lin, v53_data);
          tensorforge::fmacdpp16<2>(v64_acc, v72_lin, v54_data);
          tensorforge::fmacdpp16<3>(v64_acc, v72_lin, v55_data);
          tensorforge::fmacdpp16<4>(v64_acc, v72_lin, v56_data);
          tensorforge::fmacdpp16<5>(v64_acc, v72_lin, v57_data);
          tensorforge::fmacdpp16<6>(v64_acc, v72_lin, v58_data);
          tensorforge::fmacdpp16<7>(v64_acc, v72_lin, v59_data);
          tensorforge::fmacdpp16<8>(v64_acc, v72_lin, v60_data);
          tensorforge::fmacdpp16<9>(v64_acc, v72_lin, v61_data);
          tensorforge::fmacdpp16<10>(v64_acc, v72_lin, v62_data);
          tensorforge::fmacdpp16<11>(v64_acc, v72_lin, v63_data);
          tensorforge::fmacdpp16<12>(v65_acc, v72_lin, v52_data);
          tensorforge::fmacdpp16<13>(v65_acc, v72_lin, v53_data);
          tensorforge::fmacdpp16<14>(v65_acc, v72_lin, v54_data);
          tensorforge::fmacdpp16<15>(v65_acc, v72_lin, v55_data);
          float v73_lin = r1[1];
          tensorforge::fmacdpp16<0>(v65_acc, v73_lin, v56_data);
          tensorforge::fmacdpp16<1>(v65_acc, v73_lin, v57_data);
          tensorforge::fmacdpp16<2>(v65_acc, v73_lin, v58_data);
          tensorforge::fmacdpp16<3>(v65_acc, v73_lin, v59_data);
          tensorforge::fmacdpp16<4>(v65_acc, v73_lin, v60_data);
          tensorforge::fmacdpp16<5>(v65_acc, v73_lin, v61_data);
          tensorforge::fmacdpp16<6>(v65_acc, v73_lin, v62_data);
          tensorforge::fmacdpp16<7>(v65_acc, v73_lin, v63_data);
          tensorforge::fmacdpp16<8>(v66_acc, v73_lin, v52_data);
          tensorforge::fmacdpp16<9>(v66_acc, v73_lin, v53_data);
          tensorforge::fmacdpp16<10>(v66_acc, v73_lin, v54_data);
          tensorforge::fmacdpp16<11>(v66_acc, v73_lin, v55_data);
          tensorforge::fmacdpp16<12>(v66_acc, v73_lin, v56_data);
          tensorforge::fmacdpp16<13>(v66_acc, v73_lin, v57_data);
          tensorforge::fmacdpp16<14>(v66_acc, v73_lin, v58_data);
          tensorforge::fmacdpp16<15>(v66_acc, v73_lin, v59_data);
          float v74_lin = r1[2];
          tensorforge::fmacdpp16<0>(v66_acc, v74_lin, v60_data);
          tensorforge::fmacdpp16<1>(v66_acc, v74_lin, v61_data);
          tensorforge::fmacdpp16<2>(v66_acc, v74_lin, v62_data);
          tensorforge::fmacdpp16<3>(v66_acc, v74_lin, v63_data);
          tensorforge::fmacdpp16<4>(v67_acc, v74_lin, v52_data);
          tensorforge::fmacdpp16<5>(v67_acc, v74_lin, v53_data);
          tensorforge::fmacdpp16<6>(v67_acc, v74_lin, v54_data);
          tensorforge::fmacdpp16<7>(v67_acc, v74_lin, v55_data);
          tensorforge::fmacdpp16<8>(v67_acc, v74_lin, v56_data);
          tensorforge::fmacdpp16<9>(v67_acc, v74_lin, v57_data);
          tensorforge::fmacdpp16<10>(v67_acc, v74_lin, v58_data);
          tensorforge::fmacdpp16<11>(v67_acc, v74_lin, v59_data);
          tensorforge::fmacdpp16<12>(v67_acc, v74_lin, v60_data);
          tensorforge::fmacdpp16<13>(v67_acc, v74_lin, v61_data);
          tensorforge::fmacdpp16<14>(v67_acc, v74_lin, v62_data);
          tensorforge::fmacdpp16<15>(v67_acc, v74_lin, v63_data);
          float v75_lin = r1[3];
          tensorforge::fmacdpp16<0>(v68_acc, v75_lin, v52_data);
          tensorforge::fmacdpp16<1>(v68_acc, v75_lin, v53_data);
          tensorforge::fmacdpp16<2>(v68_acc, v75_lin, v54_data);
          tensorforge::fmacdpp16<3>(v68_acc, v75_lin, v55_data);
          tensorforge::fmacdpp16<4>(v68_acc, v75_lin, v56_data);
          tensorforge::fmacdpp16<5>(v68_acc, v75_lin, v57_data);
          tensorforge::fmacdpp16<6>(v68_acc, v75_lin, v58_data);
          tensorforge::fmacdpp16<7>(v68_acc, v75_lin, v59_data);
          tensorforge::fmacdpp16<8>(v68_acc, v75_lin, v60_data);
          tensorforge::fmacdpp16<9>(v68_acc, v75_lin, v61_data);
          tensorforge::fmacdpp16<10>(v68_acc, v75_lin, v62_data);
          tensorforge::fmacdpp16<11>(v68_acc, v75_lin, v63_data);
          tensorforge::fmacdpp16<12>(v69_acc, v75_lin, v52_data);
          tensorforge::fmacdpp16<13>(v69_acc, v75_lin, v53_data);
          tensorforge::fmacdpp16<14>(v69_acc, v75_lin, v54_data);
          tensorforge::fmacdpp16<15>(v69_acc, v75_lin, v55_data);
          float v76_lin = r1[4];
          tensorforge::fmacdpp16<0>(v69_acc, v76_lin, v56_data);
          tensorforge::fmacdpp16<1>(v69_acc, v76_lin, v57_data);
          tensorforge::fmacdpp16<2>(v69_acc, v76_lin, v58_data);
          tensorforge::fmacdpp16<3>(v69_acc, v76_lin, v59_data);
          tensorforge::fmacdpp16<4>(v69_acc, v76_lin, v60_data);
          tensorforge::fmacdpp16<5>(v69_acc, v76_lin, v61_data);
          tensorforge::fmacdpp16<6>(v69_acc, v76_lin, v62_data);
          tensorforge::fmacdpp16<7>(v69_acc, v76_lin, v63_data);
          tensorforge::fmacdpp16<8>(v70_acc, v76_lin, v52_data);
          tensorforge::fmacdpp16<9>(v70_acc, v76_lin, v53_data);
          tensorforge::fmacdpp16<10>(v70_acc, v76_lin, v54_data);
          tensorforge::fmacdpp16<11>(v70_acc, v76_lin, v55_data);
          tensorforge::fmacdpp16<12>(v70_acc, v76_lin, v56_data);
          tensorforge::fmacdpp16<13>(v70_acc, v76_lin, v57_data);
          tensorforge::fmacdpp16<14>(v70_acc, v76_lin, v58_data);
          tensorforge::fmacdpp16<15>(v70_acc, v76_lin, v59_data);
          float v77_lin = r1[5];
          tensorforge::fmacdpp16<0>(v70_acc, v77_lin, v60_data);
          tensorforge::fmacdpp16<1>(v70_acc, v77_lin, v61_data);
          tensorforge::fmacdpp16<2>(v70_acc, v77_lin, v62_data);
          tensorforge::fmacdpp16<3>(v70_acc, v77_lin, v63_data);
          tensorforge::fmacdpp16<4>(v71_acc, v77_lin, v52_data);
          tensorforge::fmacdpp16<5>(v71_acc, v77_lin, v53_data);
          tensorforge::fmacdpp16<6>(v71_acc, v77_lin, v54_data);
          tensorforge::fmacdpp16<7>(v71_acc, v77_lin, v55_data);
          tensorforge::fmacdpp16<8>(v71_acc, v77_lin, v56_data);
          tensorforge::fmacdpp16<9>(v71_acc, v77_lin, v57_data);
          tensorforge::fmacdpp16<10>(v71_acc, v77_lin, v58_data);
          tensorforge::fmacdpp16<11>(v71_acc, v77_lin, v59_data);
          tensorforge::fmacdpp16<12>(v71_acc, v77_lin, v60_data);
          tensorforge::fmacdpp16<13>(v71_acc, v77_lin, v61_data);
          tensorforge::fmacdpp16<14>(v71_acc, v77_lin, v62_data);
          tensorforge::fmacdpp16<15>(v71_acc, v77_lin, v63_data);
          ir2[0] = v64_acc;
          ir2[1] = v65_acc;
          ir2[2] = v66_acc;
          ir2[3] = v67_acc;
          ir2[4] = v68_acc;
          ir2[5] = v69_acc;
          ir2[6] = v70_acc;
          ir2[7] = v71_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v79_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v79_lin;
          float v80_lin = glb_m4[16 + threadIdx.x * 1];
          r4[1] = v80_lin;
          float v81_lin = glb_m4[32 + threadIdx.x * 1];
          r4[2] = v81_lin;
          float v82_lin = glb_m4[48 + threadIdx.x * 1];
          r4[3] = v82_lin;
          float v83_lin = glb_m4[64 + threadIdx.x * 1];
          r4[4] = v83_lin;
          float v84_lin = glb_m4[80 + threadIdx.x * 1];
          r4[5] = v84_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v90_i1 = 0; v90_i1 < 12; ++v90_i1) {
              int32_t v96_a = v90_i1 * 12;
              int32_t v97_a = v3_lead + v96_a;
              float v105_data = __builtin_nontemporal_load(&glb_m5[(v3_lead + v96_a)]);
              int32_t v106_a = 0 + v90_i1;
              r6[v106_a] = v105_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          {
            // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            float v108_data = r3[0];
            float v109_data = r3[1];
            float v110_data = r3[2];
            float v111_data = r3[3];
            float v112_data = r3[4];
            float v113_data = r3[5];
            float v114_data = r3[6];
            float v115_data = r3[7];
            float v116_data = r3[8];
            float v117_data = r3[9];
            float v118_data = r3[10];
            float v119_data = r3[11];
            float v120_acc{};
            float v121_acc{};
            float v122_acc{};
            float v123_acc{};
            float v124_acc{};
            float v125_acc{};
            float v126_acc{};
            float v127_acc{};
            float v128_lin = r4[0];
            tensorforge::fmacdpp16<0>(v120_acc, v128_lin, v108_data);
            tensorforge::fmacdpp16<1>(v120_acc, v128_lin, v109_data);
            tensorforge::fmacdpp16<2>(v120_acc, v128_lin, v110_data);
            tensorforge::fmacdpp16<3>(v120_acc, v128_lin, v111_data);
            tensorforge::fmacdpp16<4>(v120_acc, v128_lin, v112_data);
            tensorforge::fmacdpp16<5>(v120_acc, v128_lin, v113_data);
            tensorforge::fmacdpp16<6>(v120_acc, v128_lin, v114_data);
            tensorforge::fmacdpp16<7>(v120_acc, v128_lin, v115_data);
            tensorforge::fmacdpp16<8>(v120_acc, v128_lin, v116_data);
            tensorforge::fmacdpp16<9>(v120_acc, v128_lin, v117_data);
            tensorforge::fmacdpp16<10>(v120_acc, v128_lin, v118_data);
            tensorforge::fmacdpp16<11>(v120_acc, v128_lin, v119_data);
            tensorforge::fmacdpp16<12>(v121_acc, v128_lin, v108_data);
            tensorforge::fmacdpp16<13>(v121_acc, v128_lin, v109_data);
            tensorforge::fmacdpp16<14>(v121_acc, v128_lin, v110_data);
            tensorforge::fmacdpp16<15>(v121_acc, v128_lin, v111_data);
            float v129_lin = r4[1];
            tensorforge::fmacdpp16<0>(v121_acc, v129_lin, v112_data);
            tensorforge::fmacdpp16<1>(v121_acc, v129_lin, v113_data);
            tensorforge::fmacdpp16<2>(v121_acc, v129_lin, v114_data);
            tensorforge::fmacdpp16<3>(v121_acc, v129_lin, v115_data);
            tensorforge::fmacdpp16<4>(v121_acc, v129_lin, v116_data);
            tensorforge::fmacdpp16<5>(v121_acc, v129_lin, v117_data);
            tensorforge::fmacdpp16<6>(v121_acc, v129_lin, v118_data);
            tensorforge::fmacdpp16<7>(v121_acc, v129_lin, v119_data);
            tensorforge::fmacdpp16<8>(v122_acc, v129_lin, v108_data);
            tensorforge::fmacdpp16<9>(v122_acc, v129_lin, v109_data);
            tensorforge::fmacdpp16<10>(v122_acc, v129_lin, v110_data);
            tensorforge::fmacdpp16<11>(v122_acc, v129_lin, v111_data);
            tensorforge::fmacdpp16<12>(v122_acc, v129_lin, v112_data);
            tensorforge::fmacdpp16<13>(v122_acc, v129_lin, v113_data);
            tensorforge::fmacdpp16<14>(v122_acc, v129_lin, v114_data);
            tensorforge::fmacdpp16<15>(v122_acc, v129_lin, v115_data);
            float v130_lin = r4[2];
            tensorforge::fmacdpp16<0>(v122_acc, v130_lin, v116_data);
            tensorforge::fmacdpp16<1>(v122_acc, v130_lin, v117_data);
            tensorforge::fmacdpp16<2>(v122_acc, v130_lin, v118_data);
            tensorforge::fmacdpp16<3>(v122_acc, v130_lin, v119_data);
            tensorforge::fmacdpp16<4>(v123_acc, v130_lin, v108_data);
            tensorforge::fmacdpp16<5>(v123_acc, v130_lin, v109_data);
            tensorforge::fmacdpp16<6>(v123_acc, v130_lin, v110_data);
            tensorforge::fmacdpp16<7>(v123_acc, v130_lin, v111_data);
            tensorforge::fmacdpp16<8>(v123_acc, v130_lin, v112_data);
            tensorforge::fmacdpp16<9>(v123_acc, v130_lin, v113_data);
            tensorforge::fmacdpp16<10>(v123_acc, v130_lin, v114_data);
            tensorforge::fmacdpp16<11>(v123_acc, v130_lin, v115_data);
            tensorforge::fmacdpp16<12>(v123_acc, v130_lin, v116_data);
            tensorforge::fmacdpp16<13>(v123_acc, v130_lin, v117_data);
            tensorforge::fmacdpp16<14>(v123_acc, v130_lin, v118_data);
            tensorforge::fmacdpp16<15>(v123_acc, v130_lin, v119_data);
            float v131_lin = r4[3];
            tensorforge::fmacdpp16<0>(v124_acc, v131_lin, v108_data);
            tensorforge::fmacdpp16<1>(v124_acc, v131_lin, v109_data);
            tensorforge::fmacdpp16<2>(v124_acc, v131_lin, v110_data);
            tensorforge::fmacdpp16<3>(v124_acc, v131_lin, v111_data);
            tensorforge::fmacdpp16<4>(v124_acc, v131_lin, v112_data);
            tensorforge::fmacdpp16<5>(v124_acc, v131_lin, v113_data);
            tensorforge::fmacdpp16<6>(v124_acc, v131_lin, v114_data);
            tensorforge::fmacdpp16<7>(v124_acc, v131_lin, v115_data);
            tensorforge::fmacdpp16<8>(v124_acc, v131_lin, v116_data);
            tensorforge::fmacdpp16<9>(v124_acc, v131_lin, v117_data);
            tensorforge::fmacdpp16<10>(v124_acc, v131_lin, v118_data);
            tensorforge::fmacdpp16<11>(v124_acc, v131_lin, v119_data);
            tensorforge::fmacdpp16<12>(v125_acc, v131_lin, v108_data);
            tensorforge::fmacdpp16<13>(v125_acc, v131_lin, v109_data);
            tensorforge::fmacdpp16<14>(v125_acc, v131_lin, v110_data);
            tensorforge::fmacdpp16<15>(v125_acc, v131_lin, v111_data);
            float v132_lin = r4[4];
            tensorforge::fmacdpp16<0>(v125_acc, v132_lin, v112_data);
            tensorforge::fmacdpp16<1>(v125_acc, v132_lin, v113_data);
            tensorforge::fmacdpp16<2>(v125_acc, v132_lin, v114_data);
            tensorforge::fmacdpp16<3>(v125_acc, v132_lin, v115_data);
            tensorforge::fmacdpp16<4>(v125_acc, v132_lin, v116_data);
            tensorforge::fmacdpp16<5>(v125_acc, v132_lin, v117_data);
            tensorforge::fmacdpp16<6>(v125_acc, v132_lin, v118_data);
            tensorforge::fmacdpp16<7>(v125_acc, v132_lin, v119_data);
            tensorforge::fmacdpp16<8>(v126_acc, v132_lin, v108_data);
            tensorforge::fmacdpp16<9>(v126_acc, v132_lin, v109_data);
            tensorforge::fmacdpp16<10>(v126_acc, v132_lin, v110_data);
            tensorforge::fmacdpp16<11>(v126_acc, v132_lin, v111_data);
            tensorforge::fmacdpp16<12>(v126_acc, v132_lin, v112_data);
            tensorforge::fmacdpp16<13>(v126_acc, v132_lin, v113_data);
            tensorforge::fmacdpp16<14>(v126_acc, v132_lin, v114_data);
            tensorforge::fmacdpp16<15>(v126_acc, v132_lin, v115_data);
            float v133_lin = r4[5];
            tensorforge::fmacdpp16<0>(v126_acc, v133_lin, v116_data);
            tensorforge::fmacdpp16<1>(v126_acc, v133_lin, v117_data);
            tensorforge::fmacdpp16<2>(v126_acc, v133_lin, v118_data);
            tensorforge::fmacdpp16<3>(v126_acc, v133_lin, v119_data);
            tensorforge::fmacdpp16<4>(v127_acc, v133_lin, v108_data);
            tensorforge::fmacdpp16<5>(v127_acc, v133_lin, v109_data);
            tensorforge::fmacdpp16<6>(v127_acc, v133_lin, v110_data);
            tensorforge::fmacdpp16<7>(v127_acc, v133_lin, v111_data);
            tensorforge::fmacdpp16<8>(v127_acc, v133_lin, v112_data);
            tensorforge::fmacdpp16<9>(v127_acc, v133_lin, v113_data);
            tensorforge::fmacdpp16<10>(v127_acc, v133_lin, v114_data);
            tensorforge::fmacdpp16<11>(v127_acc, v133_lin, v115_data);
            tensorforge::fmacdpp16<12>(v127_acc, v133_lin, v116_data);
            tensorforge::fmacdpp16<13>(v127_acc, v133_lin, v117_data);
            tensorforge::fmacdpp16<14>(v127_acc, v133_lin, v118_data);
            tensorforge::fmacdpp16<15>(v127_acc, v133_lin, v119_data);
            ir5[0] = v120_acc;
            ir5[1] = v121_acc;
            ir5[2] = v122_acc;
            ir5[3] = v123_acc;
            ir5[4] = v124_acc;
            ir5[5] = v125_acc;
            ir5[6] = v126_acc;
            ir5[7] = v127_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v138_n1 = 0; v138_n1 < 8; ++v138_n1) {
                int32_t v139_a = 0 + v138_n1;
                float v141_data = ir5[v138_n1];
                int32_t v142_a = 0 + v138_n1;
                float v144_data = r2[v138_n1];
                int32_t v146_a = 0 + v138_n1;
                r5[v138_n1] = (v144_data + v141_data);
              }
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v149_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v149_lin;
          float v150_lin = glb_m6[16 + threadIdx.x * 1];
          r7[1] = v150_lin;
          float v151_lin = glb_m6[32 + threadIdx.x * 1];
          r7[2] = v151_lin;
          float v152_lin = glb_m6[48 + threadIdx.x * 1];
          r7[3] = v152_lin;
          float v153_lin = glb_m6[64 + threadIdx.x * 1];
          r7[4] = v153_lin;
          float v154_lin = glb_m6[80 + threadIdx.x * 1];
          r7[5] = v154_lin;
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 12; ++v160_i1) {
              int32_t v166_a = v160_i1 * 12;
              int32_t v167_a = v3_lead + v166_a;
              float v175_data = __builtin_nontemporal_load(&glb_m7[(v3_lead + v166_a)]);
              int32_t v176_a = 0 + v160_i1;
              r9[v176_a] = v175_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          {
            // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir8[8]{};
            float v178_data = r6[0];
            float v179_data = r6[1];
            float v180_data = r6[2];
            float v181_data = r6[3];
            float v182_data = r6[4];
            float v183_data = r6[5];
            float v184_data = r6[6];
            float v185_data = r6[7];
            float v186_data = r6[8];
            float v187_data = r6[9];
            float v188_data = r6[10];
            float v189_data = r6[11];
            float v190_acc{};
            float v191_acc{};
            float v192_acc{};
            float v193_acc{};
            float v194_acc{};
            float v195_acc{};
            float v196_acc{};
            float v197_acc{};
            float v198_lin = r7[0];
            tensorforge::fmacdpp16<0>(v190_acc, v198_lin, v178_data);
            tensorforge::fmacdpp16<1>(v190_acc, v198_lin, v179_data);
            tensorforge::fmacdpp16<2>(v190_acc, v198_lin, v180_data);
            tensorforge::fmacdpp16<3>(v190_acc, v198_lin, v181_data);
            tensorforge::fmacdpp16<4>(v190_acc, v198_lin, v182_data);
            tensorforge::fmacdpp16<5>(v190_acc, v198_lin, v183_data);
            tensorforge::fmacdpp16<6>(v190_acc, v198_lin, v184_data);
            tensorforge::fmacdpp16<7>(v190_acc, v198_lin, v185_data);
            tensorforge::fmacdpp16<8>(v190_acc, v198_lin, v186_data);
            tensorforge::fmacdpp16<9>(v190_acc, v198_lin, v187_data);
            tensorforge::fmacdpp16<10>(v190_acc, v198_lin, v188_data);
            tensorforge::fmacdpp16<11>(v190_acc, v198_lin, v189_data);
            tensorforge::fmacdpp16<12>(v191_acc, v198_lin, v178_data);
            tensorforge::fmacdpp16<13>(v191_acc, v198_lin, v179_data);
            tensorforge::fmacdpp16<14>(v191_acc, v198_lin, v180_data);
            tensorforge::fmacdpp16<15>(v191_acc, v198_lin, v181_data);
            float v199_lin = r7[1];
            tensorforge::fmacdpp16<0>(v191_acc, v199_lin, v182_data);
            tensorforge::fmacdpp16<1>(v191_acc, v199_lin, v183_data);
            tensorforge::fmacdpp16<2>(v191_acc, v199_lin, v184_data);
            tensorforge::fmacdpp16<3>(v191_acc, v199_lin, v185_data);
            tensorforge::fmacdpp16<4>(v191_acc, v199_lin, v186_data);
            tensorforge::fmacdpp16<5>(v191_acc, v199_lin, v187_data);
            tensorforge::fmacdpp16<6>(v191_acc, v199_lin, v188_data);
            tensorforge::fmacdpp16<7>(v191_acc, v199_lin, v189_data);
            tensorforge::fmacdpp16<8>(v192_acc, v199_lin, v178_data);
            tensorforge::fmacdpp16<9>(v192_acc, v199_lin, v179_data);
            tensorforge::fmacdpp16<10>(v192_acc, v199_lin, v180_data);
            tensorforge::fmacdpp16<11>(v192_acc, v199_lin, v181_data);
            tensorforge::fmacdpp16<12>(v192_acc, v199_lin, v182_data);
            tensorforge::fmacdpp16<13>(v192_acc, v199_lin, v183_data);
            tensorforge::fmacdpp16<14>(v192_acc, v199_lin, v184_data);
            tensorforge::fmacdpp16<15>(v192_acc, v199_lin, v185_data);
            float v200_lin = r7[2];
            tensorforge::fmacdpp16<0>(v192_acc, v200_lin, v186_data);
            tensorforge::fmacdpp16<1>(v192_acc, v200_lin, v187_data);
            tensorforge::fmacdpp16<2>(v192_acc, v200_lin, v188_data);
            tensorforge::fmacdpp16<3>(v192_acc, v200_lin, v189_data);
            tensorforge::fmacdpp16<4>(v193_acc, v200_lin, v178_data);
            tensorforge::fmacdpp16<5>(v193_acc, v200_lin, v179_data);
            tensorforge::fmacdpp16<6>(v193_acc, v200_lin, v180_data);
            tensorforge::fmacdpp16<7>(v193_acc, v200_lin, v181_data);
            tensorforge::fmacdpp16<8>(v193_acc, v200_lin, v182_data);
            tensorforge::fmacdpp16<9>(v193_acc, v200_lin, v183_data);
            tensorforge::fmacdpp16<10>(v193_acc, v200_lin, v184_data);
            tensorforge::fmacdpp16<11>(v193_acc, v200_lin, v185_data);
            tensorforge::fmacdpp16<12>(v193_acc, v200_lin, v186_data);
            tensorforge::fmacdpp16<13>(v193_acc, v200_lin, v187_data);
            tensorforge::fmacdpp16<14>(v193_acc, v200_lin, v188_data);
            tensorforge::fmacdpp16<15>(v193_acc, v200_lin, v189_data);
            float v201_lin = r7[3];
            tensorforge::fmacdpp16<0>(v194_acc, v201_lin, v178_data);
            tensorforge::fmacdpp16<1>(v194_acc, v201_lin, v179_data);
            tensorforge::fmacdpp16<2>(v194_acc, v201_lin, v180_data);
            tensorforge::fmacdpp16<3>(v194_acc, v201_lin, v181_data);
            tensorforge::fmacdpp16<4>(v194_acc, v201_lin, v182_data);
            tensorforge::fmacdpp16<5>(v194_acc, v201_lin, v183_data);
            tensorforge::fmacdpp16<6>(v194_acc, v201_lin, v184_data);
            tensorforge::fmacdpp16<7>(v194_acc, v201_lin, v185_data);
            tensorforge::fmacdpp16<8>(v194_acc, v201_lin, v186_data);
            tensorforge::fmacdpp16<9>(v194_acc, v201_lin, v187_data);
            tensorforge::fmacdpp16<10>(v194_acc, v201_lin, v188_data);
            tensorforge::fmacdpp16<11>(v194_acc, v201_lin, v189_data);
            tensorforge::fmacdpp16<12>(v195_acc, v201_lin, v178_data);
            tensorforge::fmacdpp16<13>(v195_acc, v201_lin, v179_data);
            tensorforge::fmacdpp16<14>(v195_acc, v201_lin, v180_data);
            tensorforge::fmacdpp16<15>(v195_acc, v201_lin, v181_data);
            float v202_lin = r7[4];
            tensorforge::fmacdpp16<0>(v195_acc, v202_lin, v182_data);
            tensorforge::fmacdpp16<1>(v195_acc, v202_lin, v183_data);
            tensorforge::fmacdpp16<2>(v195_acc, v202_lin, v184_data);
            tensorforge::fmacdpp16<3>(v195_acc, v202_lin, v185_data);
            tensorforge::fmacdpp16<4>(v195_acc, v202_lin, v186_data);
            tensorforge::fmacdpp16<5>(v195_acc, v202_lin, v187_data);
            tensorforge::fmacdpp16<6>(v195_acc, v202_lin, v188_data);
            tensorforge::fmacdpp16<7>(v195_acc, v202_lin, v189_data);
            tensorforge::fmacdpp16<8>(v196_acc, v202_lin, v178_data);
            tensorforge::fmacdpp16<9>(v196_acc, v202_lin, v179_data);
            tensorforge::fmacdpp16<10>(v196_acc, v202_lin, v180_data);
            tensorforge::fmacdpp16<11>(v196_acc, v202_lin, v181_data);
            tensorforge::fmacdpp16<12>(v196_acc, v202_lin, v182_data);
            tensorforge::fmacdpp16<13>(v196_acc, v202_lin, v183_data);
            tensorforge::fmacdpp16<14>(v196_acc, v202_lin, v184_data);
            tensorforge::fmacdpp16<15>(v196_acc, v202_lin, v185_data);
            float v203_lin = r7[5];
            tensorforge::fmacdpp16<0>(v196_acc, v203_lin, v186_data);
            tensorforge::fmacdpp16<1>(v196_acc, v203_lin, v187_data);
            tensorforge::fmacdpp16<2>(v196_acc, v203_lin, v188_data);
            tensorforge::fmacdpp16<3>(v196_acc, v203_lin, v189_data);
            tensorforge::fmacdpp16<4>(v197_acc, v203_lin, v178_data);
            tensorforge::fmacdpp16<5>(v197_acc, v203_lin, v179_data);
            tensorforge::fmacdpp16<6>(v197_acc, v203_lin, v180_data);
            tensorforge::fmacdpp16<7>(v197_acc, v203_lin, v181_data);
            tensorforge::fmacdpp16<8>(v197_acc, v203_lin, v182_data);
            tensorforge::fmacdpp16<9>(v197_acc, v203_lin, v183_data);
            tensorforge::fmacdpp16<10>(v197_acc, v203_lin, v184_data);
            tensorforge::fmacdpp16<11>(v197_acc, v203_lin, v185_data);
            tensorforge::fmacdpp16<12>(v197_acc, v203_lin, v186_data);
            tensorforge::fmacdpp16<13>(v197_acc, v203_lin, v187_data);
            tensorforge::fmacdpp16<14>(v197_acc, v203_lin, v188_data);
            tensorforge::fmacdpp16<15>(v197_acc, v203_lin, v189_data);
            ir8[0] = v190_acc;
            ir8[1] = v191_acc;
            ir8[2] = v192_acc;
            ir8[3] = v193_acc;
            ir8[4] = v194_acc;
            ir8[5] = v195_acc;
            ir8[6] = v196_acc;
            ir8[7] = v197_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v208_n1 = 0; v208_n1 < 8; ++v208_n1) {
                int32_t v209_a = 0 + v208_n1;
                float v211_data = ir8[v208_n1];
                int32_t v212_a = 0 + v208_n1;
                float v214_data = r5[v208_n1];
                int32_t v216_a = 0 + v208_n1;
                r8[v208_n1] = (v214_data + v211_data);
              }
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          float v219_lin = glb_m8[0 + threadIdx.x * 1];
          r10[0] = v219_lin;
          float v220_lin = glb_m8[16 + threadIdx.x * 1];
          r10[1] = v220_lin;
          float v221_lin = glb_m8[32 + threadIdx.x * 1];
          r10[2] = v221_lin;
          float v222_lin = glb_m8[48 + threadIdx.x * 1];
          r10[3] = v222_lin;
          float v223_lin = glb_m8[64 + threadIdx.x * 1];
          r10[4] = v223_lin;
          float v224_lin = glb_m8[80 + threadIdx.x * 1];
          r10[5] = v224_lin;
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          {
            // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir11[8]{};
            float v226_data = r9[0];
            float v227_data = r9[1];
            float v228_data = r9[2];
            float v229_data = r9[3];
            float v230_data = r9[4];
            float v231_data = r9[5];
            float v232_data = r9[6];
            float v233_data = r9[7];
            float v234_data = r9[8];
            float v235_data = r9[9];
            float v236_data = r9[10];
            float v237_data = r9[11];
            float v238_acc{};
            float v239_acc{};
            float v240_acc{};
            float v241_acc{};
            float v242_acc{};
            float v243_acc{};
            float v244_acc{};
            float v245_acc{};
            float v246_lin = r10[0];
            tensorforge::fmacdpp16<0>(v238_acc, v246_lin, v226_data);
            tensorforge::fmacdpp16<1>(v238_acc, v246_lin, v227_data);
            tensorforge::fmacdpp16<2>(v238_acc, v246_lin, v228_data);
            tensorforge::fmacdpp16<3>(v238_acc, v246_lin, v229_data);
            tensorforge::fmacdpp16<4>(v238_acc, v246_lin, v230_data);
            tensorforge::fmacdpp16<5>(v238_acc, v246_lin, v231_data);
            tensorforge::fmacdpp16<6>(v238_acc, v246_lin, v232_data);
            tensorforge::fmacdpp16<7>(v238_acc, v246_lin, v233_data);
            tensorforge::fmacdpp16<8>(v238_acc, v246_lin, v234_data);
            tensorforge::fmacdpp16<9>(v238_acc, v246_lin, v235_data);
            tensorforge::fmacdpp16<10>(v238_acc, v246_lin, v236_data);
            tensorforge::fmacdpp16<11>(v238_acc, v246_lin, v237_data);
            tensorforge::fmacdpp16<12>(v239_acc, v246_lin, v226_data);
            tensorforge::fmacdpp16<13>(v239_acc, v246_lin, v227_data);
            tensorforge::fmacdpp16<14>(v239_acc, v246_lin, v228_data);
            tensorforge::fmacdpp16<15>(v239_acc, v246_lin, v229_data);
            float v247_lin = r10[1];
            tensorforge::fmacdpp16<0>(v239_acc, v247_lin, v230_data);
            tensorforge::fmacdpp16<1>(v239_acc, v247_lin, v231_data);
            tensorforge::fmacdpp16<2>(v239_acc, v247_lin, v232_data);
            tensorforge::fmacdpp16<3>(v239_acc, v247_lin, v233_data);
            tensorforge::fmacdpp16<4>(v239_acc, v247_lin, v234_data);
            tensorforge::fmacdpp16<5>(v239_acc, v247_lin, v235_data);
            tensorforge::fmacdpp16<6>(v239_acc, v247_lin, v236_data);
            tensorforge::fmacdpp16<7>(v239_acc, v247_lin, v237_data);
            tensorforge::fmacdpp16<8>(v240_acc, v247_lin, v226_data);
            tensorforge::fmacdpp16<9>(v240_acc, v247_lin, v227_data);
            tensorforge::fmacdpp16<10>(v240_acc, v247_lin, v228_data);
            tensorforge::fmacdpp16<11>(v240_acc, v247_lin, v229_data);
            tensorforge::fmacdpp16<12>(v240_acc, v247_lin, v230_data);
            tensorforge::fmacdpp16<13>(v240_acc, v247_lin, v231_data);
            tensorforge::fmacdpp16<14>(v240_acc, v247_lin, v232_data);
            tensorforge::fmacdpp16<15>(v240_acc, v247_lin, v233_data);
            float v248_lin = r10[2];
            tensorforge::fmacdpp16<0>(v240_acc, v248_lin, v234_data);
            tensorforge::fmacdpp16<1>(v240_acc, v248_lin, v235_data);
            tensorforge::fmacdpp16<2>(v240_acc, v248_lin, v236_data);
            tensorforge::fmacdpp16<3>(v240_acc, v248_lin, v237_data);
            tensorforge::fmacdpp16<4>(v241_acc, v248_lin, v226_data);
            tensorforge::fmacdpp16<5>(v241_acc, v248_lin, v227_data);
            tensorforge::fmacdpp16<6>(v241_acc, v248_lin, v228_data);
            tensorforge::fmacdpp16<7>(v241_acc, v248_lin, v229_data);
            tensorforge::fmacdpp16<8>(v241_acc, v248_lin, v230_data);
            tensorforge::fmacdpp16<9>(v241_acc, v248_lin, v231_data);
            tensorforge::fmacdpp16<10>(v241_acc, v248_lin, v232_data);
            tensorforge::fmacdpp16<11>(v241_acc, v248_lin, v233_data);
            tensorforge::fmacdpp16<12>(v241_acc, v248_lin, v234_data);
            tensorforge::fmacdpp16<13>(v241_acc, v248_lin, v235_data);
            tensorforge::fmacdpp16<14>(v241_acc, v248_lin, v236_data);
            tensorforge::fmacdpp16<15>(v241_acc, v248_lin, v237_data);
            float v249_lin = r10[3];
            tensorforge::fmacdpp16<0>(v242_acc, v249_lin, v226_data);
            tensorforge::fmacdpp16<1>(v242_acc, v249_lin, v227_data);
            tensorforge::fmacdpp16<2>(v242_acc, v249_lin, v228_data);
            tensorforge::fmacdpp16<3>(v242_acc, v249_lin, v229_data);
            tensorforge::fmacdpp16<4>(v242_acc, v249_lin, v230_data);
            tensorforge::fmacdpp16<5>(v242_acc, v249_lin, v231_data);
            tensorforge::fmacdpp16<6>(v242_acc, v249_lin, v232_data);
            tensorforge::fmacdpp16<7>(v242_acc, v249_lin, v233_data);
            tensorforge::fmacdpp16<8>(v242_acc, v249_lin, v234_data);
            tensorforge::fmacdpp16<9>(v242_acc, v249_lin, v235_data);
            tensorforge::fmacdpp16<10>(v242_acc, v249_lin, v236_data);
            tensorforge::fmacdpp16<11>(v242_acc, v249_lin, v237_data);
            tensorforge::fmacdpp16<12>(v243_acc, v249_lin, v226_data);
            tensorforge::fmacdpp16<13>(v243_acc, v249_lin, v227_data);
            tensorforge::fmacdpp16<14>(v243_acc, v249_lin, v228_data);
            tensorforge::fmacdpp16<15>(v243_acc, v249_lin, v229_data);
            float v250_lin = r10[4];
            tensorforge::fmacdpp16<0>(v243_acc, v250_lin, v230_data);
            tensorforge::fmacdpp16<1>(v243_acc, v250_lin, v231_data);
            tensorforge::fmacdpp16<2>(v243_acc, v250_lin, v232_data);
            tensorforge::fmacdpp16<3>(v243_acc, v250_lin, v233_data);
            tensorforge::fmacdpp16<4>(v243_acc, v250_lin, v234_data);
            tensorforge::fmacdpp16<5>(v243_acc, v250_lin, v235_data);
            tensorforge::fmacdpp16<6>(v243_acc, v250_lin, v236_data);
            tensorforge::fmacdpp16<7>(v243_acc, v250_lin, v237_data);
            tensorforge::fmacdpp16<8>(v244_acc, v250_lin, v226_data);
            tensorforge::fmacdpp16<9>(v244_acc, v250_lin, v227_data);
            tensorforge::fmacdpp16<10>(v244_acc, v250_lin, v228_data);
            tensorforge::fmacdpp16<11>(v244_acc, v250_lin, v229_data);
            tensorforge::fmacdpp16<12>(v244_acc, v250_lin, v230_data);
            tensorforge::fmacdpp16<13>(v244_acc, v250_lin, v231_data);
            tensorforge::fmacdpp16<14>(v244_acc, v250_lin, v232_data);
            tensorforge::fmacdpp16<15>(v244_acc, v250_lin, v233_data);
            float v251_lin = r10[5];
            tensorforge::fmacdpp16<0>(v244_acc, v251_lin, v234_data);
            tensorforge::fmacdpp16<1>(v244_acc, v251_lin, v235_data);
            tensorforge::fmacdpp16<2>(v244_acc, v251_lin, v236_data);
            tensorforge::fmacdpp16<3>(v244_acc, v251_lin, v237_data);
            tensorforge::fmacdpp16<4>(v245_acc, v251_lin, v226_data);
            tensorforge::fmacdpp16<5>(v245_acc, v251_lin, v227_data);
            tensorforge::fmacdpp16<6>(v245_acc, v251_lin, v228_data);
            tensorforge::fmacdpp16<7>(v245_acc, v251_lin, v229_data);
            tensorforge::fmacdpp16<8>(v245_acc, v251_lin, v230_data);
            tensorforge::fmacdpp16<9>(v245_acc, v251_lin, v231_data);
            tensorforge::fmacdpp16<10>(v245_acc, v251_lin, v232_data);
            tensorforge::fmacdpp16<11>(v245_acc, v251_lin, v233_data);
            tensorforge::fmacdpp16<12>(v245_acc, v251_lin, v234_data);
            tensorforge::fmacdpp16<13>(v245_acc, v251_lin, v235_data);
            tensorforge::fmacdpp16<14>(v245_acc, v251_lin, v236_data);
            tensorforge::fmacdpp16<15>(v245_acc, v251_lin, v237_data);
            ir11[0] = v238_acc;
            ir11[1] = v239_acc;
            ir11[2] = v240_acc;
            ir11[3] = v241_acc;
            ir11[4] = v242_acc;
            ir11[5] = v243_acc;
            ir11[6] = v244_acc;
            ir11[7] = v245_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v256_n1 = 0; v256_n1 < 8; ++v256_n1) {
                int32_t v257_a = 0 + v256_n1;
                float v259_data = ir11[v256_n1];
                int32_t v260_a = 0 + v256_n1;
                float v262_data = r8[v256_n1];
                int32_t v264_a = 0 + v256_n1;
                r11[v256_n1] = (v262_data + v259_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v270_i1 = 0; v270_i1 < 8; ++v270_i1) {
              int32_t v271_a = 0 + v270_i1;
              float v273_data = r11[v270_i1];
              int32_t v280_a = v3_lead + (v270_i1 * 12);
              glb_m0[v280_a] = v273_data;
            }
          }
          ;
        }
      }
    }
  }
}

