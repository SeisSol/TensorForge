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
              int32_t v24_a = v18_i1 * 12;
              int32_t v25_a = v16_lead + v24_a;
              float v33_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v24_a)]);
              r0[v18_i1] = v33_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v36_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v36_lin;
          float v37_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v37_lin;
          float v38_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v38_lin;
          float v39_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v39_lin;
          float v40_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v40_lin;
          float v41_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v41_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 12; ++v47_i1) {
              int32_t v53_a = v47_i1 * 12;
              int32_t v54_a = v16_lead + v53_a;
              float v62_data = __builtin_nontemporal_load(&glb_m3[(v16_lead + v53_a)]);
              r3[v47_i1] = v62_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          float v69_data = r0[4];
          float v70_data = r0[5];
          float v71_data = r0[6];
          float v72_data = r0[7];
          float v73_data = r0[8];
          float v74_data = r0[9];
          float v75_data = r0[10];
          float v76_data = r0[11];
          float v77_acc{};
          float v78_acc{};
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_acc{};
          float v83_acc{};
          float v84_acc{};
          float v85_lin = r1[0];
          tensorforge::fmacdpp16<0>(v77_acc, v85_lin, v65_data);
          tensorforge::fmacdpp16<1>(v77_acc, v85_lin, v66_data);
          tensorforge::fmacdpp16<2>(v77_acc, v85_lin, v67_data);
          tensorforge::fmacdpp16<3>(v77_acc, v85_lin, v68_data);
          tensorforge::fmacdpp16<4>(v77_acc, v85_lin, v69_data);
          tensorforge::fmacdpp16<5>(v77_acc, v85_lin, v70_data);
          tensorforge::fmacdpp16<6>(v77_acc, v85_lin, v71_data);
          tensorforge::fmacdpp16<7>(v77_acc, v85_lin, v72_data);
          tensorforge::fmacdpp16<8>(v77_acc, v85_lin, v73_data);
          tensorforge::fmacdpp16<9>(v77_acc, v85_lin, v74_data);
          tensorforge::fmacdpp16<10>(v77_acc, v85_lin, v75_data);
          tensorforge::fmacdpp16<11>(v77_acc, v85_lin, v76_data);
          tensorforge::fmacdpp16<12>(v78_acc, v85_lin, v65_data);
          tensorforge::fmacdpp16<13>(v78_acc, v85_lin, v66_data);
          tensorforge::fmacdpp16<14>(v78_acc, v85_lin, v67_data);
          tensorforge::fmacdpp16<15>(v78_acc, v85_lin, v68_data);
          float v86_lin = r1[1];
          tensorforge::fmacdpp16<0>(v78_acc, v86_lin, v69_data);
          tensorforge::fmacdpp16<1>(v78_acc, v86_lin, v70_data);
          tensorforge::fmacdpp16<2>(v78_acc, v86_lin, v71_data);
          tensorforge::fmacdpp16<3>(v78_acc, v86_lin, v72_data);
          tensorforge::fmacdpp16<4>(v78_acc, v86_lin, v73_data);
          tensorforge::fmacdpp16<5>(v78_acc, v86_lin, v74_data);
          tensorforge::fmacdpp16<6>(v78_acc, v86_lin, v75_data);
          tensorforge::fmacdpp16<7>(v78_acc, v86_lin, v76_data);
          tensorforge::fmacdpp16<8>(v79_acc, v86_lin, v65_data);
          tensorforge::fmacdpp16<9>(v79_acc, v86_lin, v66_data);
          tensorforge::fmacdpp16<10>(v79_acc, v86_lin, v67_data);
          tensorforge::fmacdpp16<11>(v79_acc, v86_lin, v68_data);
          tensorforge::fmacdpp16<12>(v79_acc, v86_lin, v69_data);
          tensorforge::fmacdpp16<13>(v79_acc, v86_lin, v70_data);
          tensorforge::fmacdpp16<14>(v79_acc, v86_lin, v71_data);
          tensorforge::fmacdpp16<15>(v79_acc, v86_lin, v72_data);
          float v87_lin = r1[2];
          tensorforge::fmacdpp16<0>(v79_acc, v87_lin, v73_data);
          tensorforge::fmacdpp16<1>(v79_acc, v87_lin, v74_data);
          tensorforge::fmacdpp16<2>(v79_acc, v87_lin, v75_data);
          tensorforge::fmacdpp16<3>(v79_acc, v87_lin, v76_data);
          tensorforge::fmacdpp16<4>(v80_acc, v87_lin, v65_data);
          tensorforge::fmacdpp16<5>(v80_acc, v87_lin, v66_data);
          tensorforge::fmacdpp16<6>(v80_acc, v87_lin, v67_data);
          tensorforge::fmacdpp16<7>(v80_acc, v87_lin, v68_data);
          tensorforge::fmacdpp16<8>(v80_acc, v87_lin, v69_data);
          tensorforge::fmacdpp16<9>(v80_acc, v87_lin, v70_data);
          tensorforge::fmacdpp16<10>(v80_acc, v87_lin, v71_data);
          tensorforge::fmacdpp16<11>(v80_acc, v87_lin, v72_data);
          tensorforge::fmacdpp16<12>(v80_acc, v87_lin, v73_data);
          tensorforge::fmacdpp16<13>(v80_acc, v87_lin, v74_data);
          tensorforge::fmacdpp16<14>(v80_acc, v87_lin, v75_data);
          tensorforge::fmacdpp16<15>(v80_acc, v87_lin, v76_data);
          float v88_lin = r1[3];
          tensorforge::fmacdpp16<0>(v81_acc, v88_lin, v65_data);
          tensorforge::fmacdpp16<1>(v81_acc, v88_lin, v66_data);
          tensorforge::fmacdpp16<2>(v81_acc, v88_lin, v67_data);
          tensorforge::fmacdpp16<3>(v81_acc, v88_lin, v68_data);
          tensorforge::fmacdpp16<4>(v81_acc, v88_lin, v69_data);
          tensorforge::fmacdpp16<5>(v81_acc, v88_lin, v70_data);
          tensorforge::fmacdpp16<6>(v81_acc, v88_lin, v71_data);
          tensorforge::fmacdpp16<7>(v81_acc, v88_lin, v72_data);
          tensorforge::fmacdpp16<8>(v81_acc, v88_lin, v73_data);
          tensorforge::fmacdpp16<9>(v81_acc, v88_lin, v74_data);
          tensorforge::fmacdpp16<10>(v81_acc, v88_lin, v75_data);
          tensorforge::fmacdpp16<11>(v81_acc, v88_lin, v76_data);
          tensorforge::fmacdpp16<12>(v82_acc, v88_lin, v65_data);
          tensorforge::fmacdpp16<13>(v82_acc, v88_lin, v66_data);
          tensorforge::fmacdpp16<14>(v82_acc, v88_lin, v67_data);
          tensorforge::fmacdpp16<15>(v82_acc, v88_lin, v68_data);
          float v89_lin = r1[4];
          tensorforge::fmacdpp16<0>(v82_acc, v89_lin, v69_data);
          tensorforge::fmacdpp16<1>(v82_acc, v89_lin, v70_data);
          tensorforge::fmacdpp16<2>(v82_acc, v89_lin, v71_data);
          tensorforge::fmacdpp16<3>(v82_acc, v89_lin, v72_data);
          tensorforge::fmacdpp16<4>(v82_acc, v89_lin, v73_data);
          tensorforge::fmacdpp16<5>(v82_acc, v89_lin, v74_data);
          tensorforge::fmacdpp16<6>(v82_acc, v89_lin, v75_data);
          tensorforge::fmacdpp16<7>(v82_acc, v89_lin, v76_data);
          tensorforge::fmacdpp16<8>(v83_acc, v89_lin, v65_data);
          tensorforge::fmacdpp16<9>(v83_acc, v89_lin, v66_data);
          tensorforge::fmacdpp16<10>(v83_acc, v89_lin, v67_data);
          tensorforge::fmacdpp16<11>(v83_acc, v89_lin, v68_data);
          tensorforge::fmacdpp16<12>(v83_acc, v89_lin, v69_data);
          tensorforge::fmacdpp16<13>(v83_acc, v89_lin, v70_data);
          tensorforge::fmacdpp16<14>(v83_acc, v89_lin, v71_data);
          tensorforge::fmacdpp16<15>(v83_acc, v89_lin, v72_data);
          float v90_lin = r1[5];
          tensorforge::fmacdpp16<0>(v83_acc, v90_lin, v73_data);
          tensorforge::fmacdpp16<1>(v83_acc, v90_lin, v74_data);
          tensorforge::fmacdpp16<2>(v83_acc, v90_lin, v75_data);
          tensorforge::fmacdpp16<3>(v83_acc, v90_lin, v76_data);
          tensorforge::fmacdpp16<4>(v84_acc, v90_lin, v65_data);
          tensorforge::fmacdpp16<5>(v84_acc, v90_lin, v66_data);
          tensorforge::fmacdpp16<6>(v84_acc, v90_lin, v67_data);
          tensorforge::fmacdpp16<7>(v84_acc, v90_lin, v68_data);
          tensorforge::fmacdpp16<8>(v84_acc, v90_lin, v69_data);
          tensorforge::fmacdpp16<9>(v84_acc, v90_lin, v70_data);
          tensorforge::fmacdpp16<10>(v84_acc, v90_lin, v71_data);
          tensorforge::fmacdpp16<11>(v84_acc, v90_lin, v72_data);
          tensorforge::fmacdpp16<12>(v84_acc, v90_lin, v73_data);
          tensorforge::fmacdpp16<13>(v84_acc, v90_lin, v74_data);
          tensorforge::fmacdpp16<14>(v84_acc, v90_lin, v75_data);
          tensorforge::fmacdpp16<15>(v84_acc, v90_lin, v76_data);
          r2[0] = v77_acc;
          r2[1] = v78_acc;
          r2[2] = v79_acc;
          r2[3] = v80_acc;
          r2[4] = v81_acc;
          r2[5] = v82_acc;
          r2[6] = v83_acc;
          r2[7] = v84_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v92_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v92_lin;
          float v93_lin = glb_m4[16 + threadIdx.x * 1];
          r4[1] = v93_lin;
          float v94_lin = glb_m4[32 + threadIdx.x * 1];
          r4[2] = v94_lin;
          float v95_lin = glb_m4[48 + threadIdx.x * 1];
          r4[3] = v95_lin;
          float v96_lin = glb_m4[64 + threadIdx.x * 1];
          r4[4] = v96_lin;
          float v97_lin = glb_m4[80 + threadIdx.x * 1];
          r4[5] = v97_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v103_i1 = 0; v103_i1 < 12; ++v103_i1) {
              int32_t v109_a = v103_i1 * 12;
              int32_t v110_a = v16_lead + v109_a;
              float v118_data = __builtin_nontemporal_load(&glb_m5[(v16_lead + v109_a)]);
              r6[v103_i1] = v118_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v122_data = r3[0];
          float v123_data = r3[1];
          float v124_data = r3[2];
          float v125_data = r3[3];
          float v126_data = r3[4];
          float v127_data = r3[5];
          float v128_data = r3[6];
          float v129_data = r3[7];
          float v130_data = r3[8];
          float v131_data = r3[9];
          float v132_data = r3[10];
          float v133_data = r3[11];
          float v134_acc{};
          float v135_acc{};
          float v136_acc{};
          float v137_acc{};
          float v138_acc{};
          float v139_acc{};
          float v140_acc{};
          float v141_acc{};
          float v142_lin = r4[0];
          tensorforge::fmacdpp16<0>(v134_acc, v142_lin, v122_data);
          tensorforge::fmacdpp16<1>(v134_acc, v142_lin, v123_data);
          tensorforge::fmacdpp16<2>(v134_acc, v142_lin, v124_data);
          tensorforge::fmacdpp16<3>(v134_acc, v142_lin, v125_data);
          tensorforge::fmacdpp16<4>(v134_acc, v142_lin, v126_data);
          tensorforge::fmacdpp16<5>(v134_acc, v142_lin, v127_data);
          tensorforge::fmacdpp16<6>(v134_acc, v142_lin, v128_data);
          tensorforge::fmacdpp16<7>(v134_acc, v142_lin, v129_data);
          tensorforge::fmacdpp16<8>(v134_acc, v142_lin, v130_data);
          tensorforge::fmacdpp16<9>(v134_acc, v142_lin, v131_data);
          tensorforge::fmacdpp16<10>(v134_acc, v142_lin, v132_data);
          tensorforge::fmacdpp16<11>(v134_acc, v142_lin, v133_data);
          tensorforge::fmacdpp16<12>(v135_acc, v142_lin, v122_data);
          tensorforge::fmacdpp16<13>(v135_acc, v142_lin, v123_data);
          tensorforge::fmacdpp16<14>(v135_acc, v142_lin, v124_data);
          tensorforge::fmacdpp16<15>(v135_acc, v142_lin, v125_data);
          float v143_lin = r4[1];
          tensorforge::fmacdpp16<0>(v135_acc, v143_lin, v126_data);
          tensorforge::fmacdpp16<1>(v135_acc, v143_lin, v127_data);
          tensorforge::fmacdpp16<2>(v135_acc, v143_lin, v128_data);
          tensorforge::fmacdpp16<3>(v135_acc, v143_lin, v129_data);
          tensorforge::fmacdpp16<4>(v135_acc, v143_lin, v130_data);
          tensorforge::fmacdpp16<5>(v135_acc, v143_lin, v131_data);
          tensorforge::fmacdpp16<6>(v135_acc, v143_lin, v132_data);
          tensorforge::fmacdpp16<7>(v135_acc, v143_lin, v133_data);
          tensorforge::fmacdpp16<8>(v136_acc, v143_lin, v122_data);
          tensorforge::fmacdpp16<9>(v136_acc, v143_lin, v123_data);
          tensorforge::fmacdpp16<10>(v136_acc, v143_lin, v124_data);
          tensorforge::fmacdpp16<11>(v136_acc, v143_lin, v125_data);
          tensorforge::fmacdpp16<12>(v136_acc, v143_lin, v126_data);
          tensorforge::fmacdpp16<13>(v136_acc, v143_lin, v127_data);
          tensorforge::fmacdpp16<14>(v136_acc, v143_lin, v128_data);
          tensorforge::fmacdpp16<15>(v136_acc, v143_lin, v129_data);
          float v144_lin = r4[2];
          tensorforge::fmacdpp16<0>(v136_acc, v144_lin, v130_data);
          tensorforge::fmacdpp16<1>(v136_acc, v144_lin, v131_data);
          tensorforge::fmacdpp16<2>(v136_acc, v144_lin, v132_data);
          tensorforge::fmacdpp16<3>(v136_acc, v144_lin, v133_data);
          tensorforge::fmacdpp16<4>(v137_acc, v144_lin, v122_data);
          tensorforge::fmacdpp16<5>(v137_acc, v144_lin, v123_data);
          tensorforge::fmacdpp16<6>(v137_acc, v144_lin, v124_data);
          tensorforge::fmacdpp16<7>(v137_acc, v144_lin, v125_data);
          tensorforge::fmacdpp16<8>(v137_acc, v144_lin, v126_data);
          tensorforge::fmacdpp16<9>(v137_acc, v144_lin, v127_data);
          tensorforge::fmacdpp16<10>(v137_acc, v144_lin, v128_data);
          tensorforge::fmacdpp16<11>(v137_acc, v144_lin, v129_data);
          tensorforge::fmacdpp16<12>(v137_acc, v144_lin, v130_data);
          tensorforge::fmacdpp16<13>(v137_acc, v144_lin, v131_data);
          tensorforge::fmacdpp16<14>(v137_acc, v144_lin, v132_data);
          tensorforge::fmacdpp16<15>(v137_acc, v144_lin, v133_data);
          float v145_lin = r4[3];
          tensorforge::fmacdpp16<0>(v138_acc, v145_lin, v122_data);
          tensorforge::fmacdpp16<1>(v138_acc, v145_lin, v123_data);
          tensorforge::fmacdpp16<2>(v138_acc, v145_lin, v124_data);
          tensorforge::fmacdpp16<3>(v138_acc, v145_lin, v125_data);
          tensorforge::fmacdpp16<4>(v138_acc, v145_lin, v126_data);
          tensorforge::fmacdpp16<5>(v138_acc, v145_lin, v127_data);
          tensorforge::fmacdpp16<6>(v138_acc, v145_lin, v128_data);
          tensorforge::fmacdpp16<7>(v138_acc, v145_lin, v129_data);
          tensorforge::fmacdpp16<8>(v138_acc, v145_lin, v130_data);
          tensorforge::fmacdpp16<9>(v138_acc, v145_lin, v131_data);
          tensorforge::fmacdpp16<10>(v138_acc, v145_lin, v132_data);
          tensorforge::fmacdpp16<11>(v138_acc, v145_lin, v133_data);
          tensorforge::fmacdpp16<12>(v139_acc, v145_lin, v122_data);
          tensorforge::fmacdpp16<13>(v139_acc, v145_lin, v123_data);
          tensorforge::fmacdpp16<14>(v139_acc, v145_lin, v124_data);
          tensorforge::fmacdpp16<15>(v139_acc, v145_lin, v125_data);
          float v146_lin = r4[4];
          tensorforge::fmacdpp16<0>(v139_acc, v146_lin, v126_data);
          tensorforge::fmacdpp16<1>(v139_acc, v146_lin, v127_data);
          tensorforge::fmacdpp16<2>(v139_acc, v146_lin, v128_data);
          tensorforge::fmacdpp16<3>(v139_acc, v146_lin, v129_data);
          tensorforge::fmacdpp16<4>(v139_acc, v146_lin, v130_data);
          tensorforge::fmacdpp16<5>(v139_acc, v146_lin, v131_data);
          tensorforge::fmacdpp16<6>(v139_acc, v146_lin, v132_data);
          tensorforge::fmacdpp16<7>(v139_acc, v146_lin, v133_data);
          tensorforge::fmacdpp16<8>(v140_acc, v146_lin, v122_data);
          tensorforge::fmacdpp16<9>(v140_acc, v146_lin, v123_data);
          tensorforge::fmacdpp16<10>(v140_acc, v146_lin, v124_data);
          tensorforge::fmacdpp16<11>(v140_acc, v146_lin, v125_data);
          tensorforge::fmacdpp16<12>(v140_acc, v146_lin, v126_data);
          tensorforge::fmacdpp16<13>(v140_acc, v146_lin, v127_data);
          tensorforge::fmacdpp16<14>(v140_acc, v146_lin, v128_data);
          tensorforge::fmacdpp16<15>(v140_acc, v146_lin, v129_data);
          float v147_lin = r4[5];
          tensorforge::fmacdpp16<0>(v140_acc, v147_lin, v130_data);
          tensorforge::fmacdpp16<1>(v140_acc, v147_lin, v131_data);
          tensorforge::fmacdpp16<2>(v140_acc, v147_lin, v132_data);
          tensorforge::fmacdpp16<3>(v140_acc, v147_lin, v133_data);
          tensorforge::fmacdpp16<4>(v141_acc, v147_lin, v122_data);
          tensorforge::fmacdpp16<5>(v141_acc, v147_lin, v123_data);
          tensorforge::fmacdpp16<6>(v141_acc, v147_lin, v124_data);
          tensorforge::fmacdpp16<7>(v141_acc, v147_lin, v125_data);
          tensorforge::fmacdpp16<8>(v141_acc, v147_lin, v126_data);
          tensorforge::fmacdpp16<9>(v141_acc, v147_lin, v127_data);
          tensorforge::fmacdpp16<10>(v141_acc, v147_lin, v128_data);
          tensorforge::fmacdpp16<11>(v141_acc, v147_lin, v129_data);
          tensorforge::fmacdpp16<12>(v141_acc, v147_lin, v130_data);
          tensorforge::fmacdpp16<13>(v141_acc, v147_lin, v131_data);
          tensorforge::fmacdpp16<14>(v141_acc, v147_lin, v132_data);
          tensorforge::fmacdpp16<15>(v141_acc, v147_lin, v133_data);
          ir5[0] = v134_acc;
          ir5[1] = v135_acc;
          ir5[2] = v136_acc;
          ir5[3] = v137_acc;
          ir5[4] = v138_acc;
          ir5[5] = v139_acc;
          ir5[6] = v140_acc;
          ir5[7] = v141_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v152_n1 = 0; v152_n1 < 8; ++v152_n1) {
              int32_t v153_a = 0 + v152_n1;
              float v155_data = ir5[v152_n1];
              int32_t v156_a = 0 + v152_n1;
              float v158_data = r2[v152_n1];
              r5[v152_n1] = (v158_data + v155_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v162_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v162_lin;
          float v163_lin = glb_m6[16 + threadIdx.x * 1];
          r7[1] = v163_lin;
          float v164_lin = glb_m6[32 + threadIdx.x * 1];
          r7[2] = v164_lin;
          float v165_lin = glb_m6[48 + threadIdx.x * 1];
          r7[3] = v165_lin;
          float v166_lin = glb_m6[64 + threadIdx.x * 1];
          r7[4] = v166_lin;
          float v167_lin = glb_m6[80 + threadIdx.x * 1];
          r7[5] = v167_lin;
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v173_i1 = 0; v173_i1 < 12; ++v173_i1) {
              int32_t v179_a = v173_i1 * 12;
              int32_t v180_a = v16_lead + v179_a;
              float v188_data = __builtin_nontemporal_load(&glb_m7[(v16_lead + v179_a)]);
              r9[v173_i1] = v188_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v192_data = r6[0];
          float v193_data = r6[1];
          float v194_data = r6[2];
          float v195_data = r6[3];
          float v196_data = r6[4];
          float v197_data = r6[5];
          float v198_data = r6[6];
          float v199_data = r6[7];
          float v200_data = r6[8];
          float v201_data = r6[9];
          float v202_data = r6[10];
          float v203_data = r6[11];
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_acc{};
          float v208_acc{};
          float v209_acc{};
          float v210_acc{};
          float v211_acc{};
          float v212_lin = r7[0];
          tensorforge::fmacdpp16<0>(v204_acc, v212_lin, v192_data);
          tensorforge::fmacdpp16<1>(v204_acc, v212_lin, v193_data);
          tensorforge::fmacdpp16<2>(v204_acc, v212_lin, v194_data);
          tensorforge::fmacdpp16<3>(v204_acc, v212_lin, v195_data);
          tensorforge::fmacdpp16<4>(v204_acc, v212_lin, v196_data);
          tensorforge::fmacdpp16<5>(v204_acc, v212_lin, v197_data);
          tensorforge::fmacdpp16<6>(v204_acc, v212_lin, v198_data);
          tensorforge::fmacdpp16<7>(v204_acc, v212_lin, v199_data);
          tensorforge::fmacdpp16<8>(v204_acc, v212_lin, v200_data);
          tensorforge::fmacdpp16<9>(v204_acc, v212_lin, v201_data);
          tensorforge::fmacdpp16<10>(v204_acc, v212_lin, v202_data);
          tensorforge::fmacdpp16<11>(v204_acc, v212_lin, v203_data);
          tensorforge::fmacdpp16<12>(v205_acc, v212_lin, v192_data);
          tensorforge::fmacdpp16<13>(v205_acc, v212_lin, v193_data);
          tensorforge::fmacdpp16<14>(v205_acc, v212_lin, v194_data);
          tensorforge::fmacdpp16<15>(v205_acc, v212_lin, v195_data);
          float v213_lin = r7[1];
          tensorforge::fmacdpp16<0>(v205_acc, v213_lin, v196_data);
          tensorforge::fmacdpp16<1>(v205_acc, v213_lin, v197_data);
          tensorforge::fmacdpp16<2>(v205_acc, v213_lin, v198_data);
          tensorforge::fmacdpp16<3>(v205_acc, v213_lin, v199_data);
          tensorforge::fmacdpp16<4>(v205_acc, v213_lin, v200_data);
          tensorforge::fmacdpp16<5>(v205_acc, v213_lin, v201_data);
          tensorforge::fmacdpp16<6>(v205_acc, v213_lin, v202_data);
          tensorforge::fmacdpp16<7>(v205_acc, v213_lin, v203_data);
          tensorforge::fmacdpp16<8>(v206_acc, v213_lin, v192_data);
          tensorforge::fmacdpp16<9>(v206_acc, v213_lin, v193_data);
          tensorforge::fmacdpp16<10>(v206_acc, v213_lin, v194_data);
          tensorforge::fmacdpp16<11>(v206_acc, v213_lin, v195_data);
          tensorforge::fmacdpp16<12>(v206_acc, v213_lin, v196_data);
          tensorforge::fmacdpp16<13>(v206_acc, v213_lin, v197_data);
          tensorforge::fmacdpp16<14>(v206_acc, v213_lin, v198_data);
          tensorforge::fmacdpp16<15>(v206_acc, v213_lin, v199_data);
          float v214_lin = r7[2];
          tensorforge::fmacdpp16<0>(v206_acc, v214_lin, v200_data);
          tensorforge::fmacdpp16<1>(v206_acc, v214_lin, v201_data);
          tensorforge::fmacdpp16<2>(v206_acc, v214_lin, v202_data);
          tensorforge::fmacdpp16<3>(v206_acc, v214_lin, v203_data);
          tensorforge::fmacdpp16<4>(v207_acc, v214_lin, v192_data);
          tensorforge::fmacdpp16<5>(v207_acc, v214_lin, v193_data);
          tensorforge::fmacdpp16<6>(v207_acc, v214_lin, v194_data);
          tensorforge::fmacdpp16<7>(v207_acc, v214_lin, v195_data);
          tensorforge::fmacdpp16<8>(v207_acc, v214_lin, v196_data);
          tensorforge::fmacdpp16<9>(v207_acc, v214_lin, v197_data);
          tensorforge::fmacdpp16<10>(v207_acc, v214_lin, v198_data);
          tensorforge::fmacdpp16<11>(v207_acc, v214_lin, v199_data);
          tensorforge::fmacdpp16<12>(v207_acc, v214_lin, v200_data);
          tensorforge::fmacdpp16<13>(v207_acc, v214_lin, v201_data);
          tensorforge::fmacdpp16<14>(v207_acc, v214_lin, v202_data);
          tensorforge::fmacdpp16<15>(v207_acc, v214_lin, v203_data);
          float v215_lin = r7[3];
          tensorforge::fmacdpp16<0>(v208_acc, v215_lin, v192_data);
          tensorforge::fmacdpp16<1>(v208_acc, v215_lin, v193_data);
          tensorforge::fmacdpp16<2>(v208_acc, v215_lin, v194_data);
          tensorforge::fmacdpp16<3>(v208_acc, v215_lin, v195_data);
          tensorforge::fmacdpp16<4>(v208_acc, v215_lin, v196_data);
          tensorforge::fmacdpp16<5>(v208_acc, v215_lin, v197_data);
          tensorforge::fmacdpp16<6>(v208_acc, v215_lin, v198_data);
          tensorforge::fmacdpp16<7>(v208_acc, v215_lin, v199_data);
          tensorforge::fmacdpp16<8>(v208_acc, v215_lin, v200_data);
          tensorforge::fmacdpp16<9>(v208_acc, v215_lin, v201_data);
          tensorforge::fmacdpp16<10>(v208_acc, v215_lin, v202_data);
          tensorforge::fmacdpp16<11>(v208_acc, v215_lin, v203_data);
          tensorforge::fmacdpp16<12>(v209_acc, v215_lin, v192_data);
          tensorforge::fmacdpp16<13>(v209_acc, v215_lin, v193_data);
          tensorforge::fmacdpp16<14>(v209_acc, v215_lin, v194_data);
          tensorforge::fmacdpp16<15>(v209_acc, v215_lin, v195_data);
          float v216_lin = r7[4];
          tensorforge::fmacdpp16<0>(v209_acc, v216_lin, v196_data);
          tensorforge::fmacdpp16<1>(v209_acc, v216_lin, v197_data);
          tensorforge::fmacdpp16<2>(v209_acc, v216_lin, v198_data);
          tensorforge::fmacdpp16<3>(v209_acc, v216_lin, v199_data);
          tensorforge::fmacdpp16<4>(v209_acc, v216_lin, v200_data);
          tensorforge::fmacdpp16<5>(v209_acc, v216_lin, v201_data);
          tensorforge::fmacdpp16<6>(v209_acc, v216_lin, v202_data);
          tensorforge::fmacdpp16<7>(v209_acc, v216_lin, v203_data);
          tensorforge::fmacdpp16<8>(v210_acc, v216_lin, v192_data);
          tensorforge::fmacdpp16<9>(v210_acc, v216_lin, v193_data);
          tensorforge::fmacdpp16<10>(v210_acc, v216_lin, v194_data);
          tensorforge::fmacdpp16<11>(v210_acc, v216_lin, v195_data);
          tensorforge::fmacdpp16<12>(v210_acc, v216_lin, v196_data);
          tensorforge::fmacdpp16<13>(v210_acc, v216_lin, v197_data);
          tensorforge::fmacdpp16<14>(v210_acc, v216_lin, v198_data);
          tensorforge::fmacdpp16<15>(v210_acc, v216_lin, v199_data);
          float v217_lin = r7[5];
          tensorforge::fmacdpp16<0>(v210_acc, v217_lin, v200_data);
          tensorforge::fmacdpp16<1>(v210_acc, v217_lin, v201_data);
          tensorforge::fmacdpp16<2>(v210_acc, v217_lin, v202_data);
          tensorforge::fmacdpp16<3>(v210_acc, v217_lin, v203_data);
          tensorforge::fmacdpp16<4>(v211_acc, v217_lin, v192_data);
          tensorforge::fmacdpp16<5>(v211_acc, v217_lin, v193_data);
          tensorforge::fmacdpp16<6>(v211_acc, v217_lin, v194_data);
          tensorforge::fmacdpp16<7>(v211_acc, v217_lin, v195_data);
          tensorforge::fmacdpp16<8>(v211_acc, v217_lin, v196_data);
          tensorforge::fmacdpp16<9>(v211_acc, v217_lin, v197_data);
          tensorforge::fmacdpp16<10>(v211_acc, v217_lin, v198_data);
          tensorforge::fmacdpp16<11>(v211_acc, v217_lin, v199_data);
          tensorforge::fmacdpp16<12>(v211_acc, v217_lin, v200_data);
          tensorforge::fmacdpp16<13>(v211_acc, v217_lin, v201_data);
          tensorforge::fmacdpp16<14>(v211_acc, v217_lin, v202_data);
          tensorforge::fmacdpp16<15>(v211_acc, v217_lin, v203_data);
          ir8[0] = v204_acc;
          ir8[1] = v205_acc;
          ir8[2] = v206_acc;
          ir8[3] = v207_acc;
          ir8[4] = v208_acc;
          ir8[5] = v209_acc;
          ir8[6] = v210_acc;
          ir8[7] = v211_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v222_n1 = 0; v222_n1 < 8; ++v222_n1) {
              int32_t v223_a = 0 + v222_n1;
              float v225_data = ir8[v222_n1];
              int32_t v226_a = 0 + v222_n1;
              float v228_data = r5[v222_n1];
              r8[v222_n1] = (v228_data + v225_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          float v232_lin = glb_m8[0 + threadIdx.x * 1];
          r10[0] = v232_lin;
          float v233_lin = glb_m8[16 + threadIdx.x * 1];
          r10[1] = v233_lin;
          float v234_lin = glb_m8[32 + threadIdx.x * 1];
          r10[2] = v234_lin;
          float v235_lin = glb_m8[48 + threadIdx.x * 1];
          r10[3] = v235_lin;
          float v236_lin = glb_m8[64 + threadIdx.x * 1];
          r10[4] = v236_lin;
          float v237_lin = glb_m8[80 + threadIdx.x * 1];
          r10[5] = v237_lin;
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v240_data = r9[0];
          float v241_data = r9[1];
          float v242_data = r9[2];
          float v243_data = r9[3];
          float v244_data = r9[4];
          float v245_data = r9[5];
          float v246_data = r9[6];
          float v247_data = r9[7];
          float v248_data = r9[8];
          float v249_data = r9[9];
          float v250_data = r9[10];
          float v251_data = r9[11];
          float v252_acc{};
          float v253_acc{};
          float v254_acc{};
          float v255_acc{};
          float v256_acc{};
          float v257_acc{};
          float v258_acc{};
          float v259_acc{};
          float v260_lin = r10[0];
          tensorforge::fmacdpp16<0>(v252_acc, v260_lin, v240_data);
          tensorforge::fmacdpp16<1>(v252_acc, v260_lin, v241_data);
          tensorforge::fmacdpp16<2>(v252_acc, v260_lin, v242_data);
          tensorforge::fmacdpp16<3>(v252_acc, v260_lin, v243_data);
          tensorforge::fmacdpp16<4>(v252_acc, v260_lin, v244_data);
          tensorforge::fmacdpp16<5>(v252_acc, v260_lin, v245_data);
          tensorforge::fmacdpp16<6>(v252_acc, v260_lin, v246_data);
          tensorforge::fmacdpp16<7>(v252_acc, v260_lin, v247_data);
          tensorforge::fmacdpp16<8>(v252_acc, v260_lin, v248_data);
          tensorforge::fmacdpp16<9>(v252_acc, v260_lin, v249_data);
          tensorforge::fmacdpp16<10>(v252_acc, v260_lin, v250_data);
          tensorforge::fmacdpp16<11>(v252_acc, v260_lin, v251_data);
          tensorforge::fmacdpp16<12>(v253_acc, v260_lin, v240_data);
          tensorforge::fmacdpp16<13>(v253_acc, v260_lin, v241_data);
          tensorforge::fmacdpp16<14>(v253_acc, v260_lin, v242_data);
          tensorforge::fmacdpp16<15>(v253_acc, v260_lin, v243_data);
          float v261_lin = r10[1];
          tensorforge::fmacdpp16<0>(v253_acc, v261_lin, v244_data);
          tensorforge::fmacdpp16<1>(v253_acc, v261_lin, v245_data);
          tensorforge::fmacdpp16<2>(v253_acc, v261_lin, v246_data);
          tensorforge::fmacdpp16<3>(v253_acc, v261_lin, v247_data);
          tensorforge::fmacdpp16<4>(v253_acc, v261_lin, v248_data);
          tensorforge::fmacdpp16<5>(v253_acc, v261_lin, v249_data);
          tensorforge::fmacdpp16<6>(v253_acc, v261_lin, v250_data);
          tensorforge::fmacdpp16<7>(v253_acc, v261_lin, v251_data);
          tensorforge::fmacdpp16<8>(v254_acc, v261_lin, v240_data);
          tensorforge::fmacdpp16<9>(v254_acc, v261_lin, v241_data);
          tensorforge::fmacdpp16<10>(v254_acc, v261_lin, v242_data);
          tensorforge::fmacdpp16<11>(v254_acc, v261_lin, v243_data);
          tensorforge::fmacdpp16<12>(v254_acc, v261_lin, v244_data);
          tensorforge::fmacdpp16<13>(v254_acc, v261_lin, v245_data);
          tensorforge::fmacdpp16<14>(v254_acc, v261_lin, v246_data);
          tensorforge::fmacdpp16<15>(v254_acc, v261_lin, v247_data);
          float v262_lin = r10[2];
          tensorforge::fmacdpp16<0>(v254_acc, v262_lin, v248_data);
          tensorforge::fmacdpp16<1>(v254_acc, v262_lin, v249_data);
          tensorforge::fmacdpp16<2>(v254_acc, v262_lin, v250_data);
          tensorforge::fmacdpp16<3>(v254_acc, v262_lin, v251_data);
          tensorforge::fmacdpp16<4>(v255_acc, v262_lin, v240_data);
          tensorforge::fmacdpp16<5>(v255_acc, v262_lin, v241_data);
          tensorforge::fmacdpp16<6>(v255_acc, v262_lin, v242_data);
          tensorforge::fmacdpp16<7>(v255_acc, v262_lin, v243_data);
          tensorforge::fmacdpp16<8>(v255_acc, v262_lin, v244_data);
          tensorforge::fmacdpp16<9>(v255_acc, v262_lin, v245_data);
          tensorforge::fmacdpp16<10>(v255_acc, v262_lin, v246_data);
          tensorforge::fmacdpp16<11>(v255_acc, v262_lin, v247_data);
          tensorforge::fmacdpp16<12>(v255_acc, v262_lin, v248_data);
          tensorforge::fmacdpp16<13>(v255_acc, v262_lin, v249_data);
          tensorforge::fmacdpp16<14>(v255_acc, v262_lin, v250_data);
          tensorforge::fmacdpp16<15>(v255_acc, v262_lin, v251_data);
          float v263_lin = r10[3];
          tensorforge::fmacdpp16<0>(v256_acc, v263_lin, v240_data);
          tensorforge::fmacdpp16<1>(v256_acc, v263_lin, v241_data);
          tensorforge::fmacdpp16<2>(v256_acc, v263_lin, v242_data);
          tensorforge::fmacdpp16<3>(v256_acc, v263_lin, v243_data);
          tensorforge::fmacdpp16<4>(v256_acc, v263_lin, v244_data);
          tensorforge::fmacdpp16<5>(v256_acc, v263_lin, v245_data);
          tensorforge::fmacdpp16<6>(v256_acc, v263_lin, v246_data);
          tensorforge::fmacdpp16<7>(v256_acc, v263_lin, v247_data);
          tensorforge::fmacdpp16<8>(v256_acc, v263_lin, v248_data);
          tensorforge::fmacdpp16<9>(v256_acc, v263_lin, v249_data);
          tensorforge::fmacdpp16<10>(v256_acc, v263_lin, v250_data);
          tensorforge::fmacdpp16<11>(v256_acc, v263_lin, v251_data);
          tensorforge::fmacdpp16<12>(v257_acc, v263_lin, v240_data);
          tensorforge::fmacdpp16<13>(v257_acc, v263_lin, v241_data);
          tensorforge::fmacdpp16<14>(v257_acc, v263_lin, v242_data);
          tensorforge::fmacdpp16<15>(v257_acc, v263_lin, v243_data);
          float v264_lin = r10[4];
          tensorforge::fmacdpp16<0>(v257_acc, v264_lin, v244_data);
          tensorforge::fmacdpp16<1>(v257_acc, v264_lin, v245_data);
          tensorforge::fmacdpp16<2>(v257_acc, v264_lin, v246_data);
          tensorforge::fmacdpp16<3>(v257_acc, v264_lin, v247_data);
          tensorforge::fmacdpp16<4>(v257_acc, v264_lin, v248_data);
          tensorforge::fmacdpp16<5>(v257_acc, v264_lin, v249_data);
          tensorforge::fmacdpp16<6>(v257_acc, v264_lin, v250_data);
          tensorforge::fmacdpp16<7>(v257_acc, v264_lin, v251_data);
          tensorforge::fmacdpp16<8>(v258_acc, v264_lin, v240_data);
          tensorforge::fmacdpp16<9>(v258_acc, v264_lin, v241_data);
          tensorforge::fmacdpp16<10>(v258_acc, v264_lin, v242_data);
          tensorforge::fmacdpp16<11>(v258_acc, v264_lin, v243_data);
          tensorforge::fmacdpp16<12>(v258_acc, v264_lin, v244_data);
          tensorforge::fmacdpp16<13>(v258_acc, v264_lin, v245_data);
          tensorforge::fmacdpp16<14>(v258_acc, v264_lin, v246_data);
          tensorforge::fmacdpp16<15>(v258_acc, v264_lin, v247_data);
          float v265_lin = r10[5];
          tensorforge::fmacdpp16<0>(v258_acc, v265_lin, v248_data);
          tensorforge::fmacdpp16<1>(v258_acc, v265_lin, v249_data);
          tensorforge::fmacdpp16<2>(v258_acc, v265_lin, v250_data);
          tensorforge::fmacdpp16<3>(v258_acc, v265_lin, v251_data);
          tensorforge::fmacdpp16<4>(v259_acc, v265_lin, v240_data);
          tensorforge::fmacdpp16<5>(v259_acc, v265_lin, v241_data);
          tensorforge::fmacdpp16<6>(v259_acc, v265_lin, v242_data);
          tensorforge::fmacdpp16<7>(v259_acc, v265_lin, v243_data);
          tensorforge::fmacdpp16<8>(v259_acc, v265_lin, v244_data);
          tensorforge::fmacdpp16<9>(v259_acc, v265_lin, v245_data);
          tensorforge::fmacdpp16<10>(v259_acc, v265_lin, v246_data);
          tensorforge::fmacdpp16<11>(v259_acc, v265_lin, v247_data);
          tensorforge::fmacdpp16<12>(v259_acc, v265_lin, v248_data);
          tensorforge::fmacdpp16<13>(v259_acc, v265_lin, v249_data);
          tensorforge::fmacdpp16<14>(v259_acc, v265_lin, v250_data);
          tensorforge::fmacdpp16<15>(v259_acc, v265_lin, v251_data);
          ir11[0] = v252_acc;
          ir11[1] = v253_acc;
          ir11[2] = v254_acc;
          ir11[3] = v255_acc;
          ir11[4] = v256_acc;
          ir11[5] = v257_acc;
          ir11[6] = v258_acc;
          ir11[7] = v259_acc;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v270_n1 = 0; v270_n1 < 8; ++v270_n1) {
              int32_t v271_a = 0 + v270_n1;
              float v273_data = ir11[v270_n1];
              int32_t v274_a = 0 + v270_n1;
              float v276_data = r8[v270_n1];
              r11[v270_n1] = (v276_data + v273_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v283_i1 = 0; v283_i1 < 8; ++v283_i1) {
              int32_t v284_a = 0 + v283_i1;
              float v286_data = r11[v283_i1];
              glb_m0[(v16_lead + (v283_i1 * 12))] = v286_data;
            }
          }
        }
      }
    }
  }
}

