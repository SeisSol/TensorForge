// === base name ===
kernel_10f9ca73b2110da7

// === header ===
void launcher_kernel_10f9ca73b2110da7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_10f9ca73b2110da7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_10f9ca73b2110da7, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_10f9ca73b2110da7), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_10f9ca73b2110da7, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_10f9ca73b2110da7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m2[(v19_lead + (v36_i1 * 12))]);
              r1[v36_i1] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 12; ++v51_i1) {
              float v59_data = __builtin_nontemporal_load(&glb_m3[(v19_lead + (v51_i1 * 12))]);
              r3[v51_i1] = v59_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v62_data = r1[0];
          float v63_data = r1[1];
          float v64_data = r1[2];
          float v65_data = r1[3];
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          tensorforge::transpose4x4b32(v66_tp, v67_tp, v68_tp, v69_tp, v62_data, v63_data, v64_data, v65_data);
          tensorforge::VectorT<float, 4> v70_acc{};
          float v71_data = r0[0];
          float v72_data = r0[1];
          float v73_data = r0[2];
          float v74_data = r0[3];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v75_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 2, 0, 0);
          float v79_data = r0[4];
          float v80_data = r0[5];
          float v81_data = r0[6];
          float v82_data = r0[7];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v83_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 2, 1, 0);
          float v87_data = r0[8];
          float v88_data = r0[9];
          float v89_data = r0[10];
          float v90_data = r0[11];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v91_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v89_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v90_data, v93_acc, 2, 2, 0);
          r2[0] = (v94_acc[0]);
          r2[1] = (v94_acc[1]);
          r2[2] = (v94_acc[2]);
          r2[3] = (v94_acc[3]);
          float v99_data = r1[4];
          float v100_data = r1[5];
          float v101_data = r1[6];
          float v102_data = r1[7];
          float v103_tp{};
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          tensorforge::transpose4x4b32(v103_tp, v104_tp, v105_tp, v106_tp, v99_data, v100_data, v101_data, v102_data);
          tensorforge::VectorT<float, 4> v107_acc{};
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v71_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v72_data, v112_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v73_data, v113_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v74_data, v114_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v79_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v80_data, v120_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v81_data, v121_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v82_data, v122_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v87_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v88_data, v128_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v89_data, v129_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v90_data, v130_acc, 2, 2, 0);
          r2[4] = (v131_acc[0]);
          r2[5] = (v131_acc[1]);
          r2[6] = (v131_acc[2]);
          r2[7] = (v131_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v141_i1 = 0; v141_i1 < 8; ++v141_i1) {
              float v149_data = __builtin_nontemporal_load(&glb_m4[(v19_lead + (v141_i1 * 12))]);
              r4[v141_i1] = v149_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v156_i1 = 0; v156_i1 < 12; ++v156_i1) {
              float v164_data = __builtin_nontemporal_load(&glb_m5[(v19_lead + (v156_i1 * 12))]);
              r6[v156_i1] = v164_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v168_data = r4[0];
          float v169_data = r4[1];
          float v170_data = r4[2];
          float v171_data = r4[3];
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          tensorforge::transpose4x4b32(v172_tp, v173_tp, v174_tp, v175_tp, v168_data, v169_data, v170_data, v171_data);
          tensorforge::VectorT<float, 4> v176_acc{};
          float v177_data = r3[0];
          float v178_data = r3[1];
          float v179_data = r3[2];
          float v180_data = r3[3];
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v177_data, v176_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v178_data, v181_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v179_data, v182_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v180_data, v183_acc, 2, 0, 0);
          float v185_data = r3[4];
          float v186_data = r3[5];
          float v187_data = r3[6];
          float v188_data = r3[7];
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v185_data, v184_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v186_data, v189_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v187_data, v190_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v188_data, v191_acc, 2, 1, 0);
          float v193_data = r3[8];
          float v194_data = r3[9];
          float v195_data = r3[10];
          float v196_data = r3[11];
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v193_data, v192_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v194_data, v197_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v195_data, v198_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v196_data, v199_acc, 2, 2, 0);
          ir5[0] = (v200_acc[0]);
          ir5[1] = (v200_acc[1]);
          ir5[2] = (v200_acc[2]);
          ir5[3] = (v200_acc[3]);
          float v205_data = r4[4];
          float v206_data = r4[5];
          float v207_data = r4[6];
          float v208_data = r4[7];
          float v209_tp{};
          float v210_tp{};
          float v211_tp{};
          float v212_tp{};
          tensorforge::transpose4x4b32(v209_tp, v210_tp, v211_tp, v212_tp, v205_data, v206_data, v207_data, v208_data);
          tensorforge::VectorT<float, 4> v213_acc{};
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v177_data, v213_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v178_data, v218_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v179_data, v219_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v180_data, v220_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v185_data, v221_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v186_data, v226_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v187_data, v227_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v188_data, v228_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v193_data, v229_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v194_data, v234_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v195_data, v235_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v196_data, v236_acc, 2, 2, 0);
          ir5[4] = (v237_acc[0]);
          ir5[5] = (v237_acc[1]);
          ir5[6] = (v237_acc[2]);
          ir5[7] = (v237_acc[3]);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v246_n1 = 0; v246_n1 < 8; ++v246_n1) {
              float v248_data = ir5[v246_n1];
              float v250_data = r2[v246_n1];
              r5[v246_n1] = (v250_data + v248_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v258_i1 = 0; v258_i1 < 8; ++v258_i1) {
              float v266_data = __builtin_nontemporal_load(&glb_m6[(v19_lead + (v258_i1 * 12))]);
              r7[v258_i1] = v266_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v273_i1 = 0; v273_i1 < 12; ++v273_i1) {
              float v281_data = __builtin_nontemporal_load(&glb_m7[(v19_lead + (v273_i1 * 12))]);
              r9[v273_i1] = v281_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v285_data = r7[0];
          float v286_data = r7[1];
          float v287_data = r7[2];
          float v288_data = r7[3];
          float v289_tp{};
          float v290_tp{};
          float v291_tp{};
          float v292_tp{};
          tensorforge::transpose4x4b32(v289_tp, v290_tp, v291_tp, v292_tp, v285_data, v286_data, v287_data, v288_data);
          tensorforge::VectorT<float, 4> v293_acc{};
          float v294_data = r6[0];
          float v295_data = r6[1];
          float v296_data = r6[2];
          float v297_data = r6[3];
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v294_data, v293_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v295_data, v298_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v296_data, v299_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v297_data, v300_acc, 2, 0, 0);
          float v302_data = r6[4];
          float v303_data = r6[5];
          float v304_data = r6[6];
          float v305_data = r6[7];
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v302_data, v301_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v303_data, v306_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v304_data, v307_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v305_data, v308_acc, 2, 1, 0);
          float v310_data = r6[8];
          float v311_data = r6[9];
          float v312_data = r6[10];
          float v313_data = r6[11];
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v310_data, v309_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v311_data, v314_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v312_data, v315_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v313_data, v316_acc, 2, 2, 0);
          ir8[0] = (v317_acc[0]);
          ir8[1] = (v317_acc[1]);
          ir8[2] = (v317_acc[2]);
          ir8[3] = (v317_acc[3]);
          float v322_data = r7[4];
          float v323_data = r7[5];
          float v324_data = r7[6];
          float v325_data = r7[7];
          float v326_tp{};
          float v327_tp{};
          float v328_tp{};
          float v329_tp{};
          tensorforge::transpose4x4b32(v326_tp, v327_tp, v328_tp, v329_tp, v322_data, v323_data, v324_data, v325_data);
          tensorforge::VectorT<float, 4> v330_acc{};
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v294_data, v330_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v295_data, v335_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v296_data, v336_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v297_data, v337_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v302_data, v338_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v303_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v304_data, v344_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v305_data, v345_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v310_data, v346_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v311_data, v351_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v312_data, v352_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v313_data, v353_acc, 2, 2, 0);
          ir8[4] = (v354_acc[0]);
          ir8[5] = (v354_acc[1]);
          ir8[6] = (v354_acc[2]);
          ir8[7] = (v354_acc[3]);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v363_n1 = 0; v363_n1 < 8; ++v363_n1) {
              float v365_data = ir8[v363_n1];
              float v367_data = r5[v363_n1];
              r8[v363_n1] = (v367_data + v365_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v375_i1 = 0; v375_i1 < 8; ++v375_i1) {
              float v383_data = __builtin_nontemporal_load(&glb_m8[(v19_lead + (v375_i1 * 12))]);
              r10[v375_i1] = v383_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v387_data = r10[0];
          float v388_data = r10[1];
          float v389_data = r10[2];
          float v390_data = r10[3];
          float v391_tp{};
          float v392_tp{};
          float v393_tp{};
          float v394_tp{};
          tensorforge::transpose4x4b32(v391_tp, v392_tp, v393_tp, v394_tp, v387_data, v388_data, v389_data, v390_data);
          tensorforge::VectorT<float, 4> v395_acc{};
          float v396_data = r9[0];
          float v397_data = r9[1];
          float v398_data = r9[2];
          float v399_data = r9[3];
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v391_tp, v396_data, v395_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v392_tp, v397_data, v400_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v393_tp, v398_data, v401_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v399_data, v402_acc, 2, 0, 0);
          float v404_data = r9[4];
          float v405_data = r9[5];
          float v406_data = r9[6];
          float v407_data = r9[7];
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v391_tp, v404_data, v403_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v392_tp, v405_data, v408_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v393_tp, v406_data, v409_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v407_data, v410_acc, 2, 1, 0);
          float v412_data = r9[8];
          float v413_data = r9[9];
          float v414_data = r9[10];
          float v415_data = r9[11];
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v391_tp, v412_data, v411_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v392_tp, v413_data, v416_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v393_tp, v414_data, v417_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v415_data, v418_acc, 2, 2, 0);
          ir11[0] = (v419_acc[0]);
          ir11[1] = (v419_acc[1]);
          ir11[2] = (v419_acc[2]);
          ir11[3] = (v419_acc[3]);
          float v424_data = r10[4];
          float v425_data = r10[5];
          float v426_data = r10[6];
          float v427_data = r10[7];
          float v428_tp{};
          float v429_tp{};
          float v430_tp{};
          float v431_tp{};
          tensorforge::transpose4x4b32(v428_tp, v429_tp, v430_tp, v431_tp, v424_data, v425_data, v426_data, v427_data);
          tensorforge::VectorT<float, 4> v432_acc{};
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v396_data, v432_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v397_data, v437_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v398_data, v438_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v399_data, v439_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v404_data, v440_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v405_data, v445_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v406_data, v446_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v407_data, v447_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v412_data, v448_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v413_data, v453_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v414_data, v454_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v415_data, v455_acc, 2, 2, 0);
          ir11[4] = (v456_acc[0]);
          ir11[5] = (v456_acc[1]);
          ir11[6] = (v456_acc[2]);
          ir11[7] = (v456_acc[3]);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v465_n1 = 0; v465_n1 < 8; ++v465_n1) {
              float v467_data = ir11[v465_n1];
              float v469_data = r8[v465_n1];
              r11[v465_n1] = (v469_data + v467_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v476_i1 = 0; v476_i1 < 8; ++v476_i1) {
              float v478_data = r11[v476_i1];
              glb_m0[(v19_lead + (v476_i1 * 12))] = v478_data;
            }
          }
        }
      }
    }
  }
}

