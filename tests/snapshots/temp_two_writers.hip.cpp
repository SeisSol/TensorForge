// === base name ===
kernel_3014f56b8a7c347c

// === header ===
void launcher_kernel_3014f56b8a7c347c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3014f56b8a7c347c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3014f56b8a7c347c, block.x * block.y * block.z, 3328 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3014f56b8a7c347c), hipFuncAttributeMaxDynamicSharedMemorySize, 3328 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3014f56b8a7c347c, grid, block, 3328 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3014f56b8a7c347c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×32(6×12) {0..6}×{0..12} strided
    // m1 32×32(12×12) {0..12}×{0..12} strided
    // m2 32×32(6×12) {0..6}×{0..12} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // m4 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      __syncthreads();
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 6) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v16_lead + (v18_i1 * 6))]);
              r0[v18_i1] = v26_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v33_i1 * 12))]);
              r1[v33_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v16_lead < 6) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 12; ++v48_i1) {
              float v56_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v48_i1 * 6))]);
              r3[v48_i1] = v56_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v59_data = r1[0];
          float v60_data = r1[1];
          float v61_data = r1[2];
          float v62_data = r1[3];
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[1];
          float v70_data = r0[2];
          float v71_data = r0[3];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 2, 0, 0);
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 2, 1, 0);
          float v84_data = r0[8];
          float v85_data = r0[9];
          float v86_data = r0[10];
          float v87_data = r0[11];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v85_data, v88_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v89_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v90_acc, 2, 2, 0);
          r2[0] = (v91_acc[0]);
          r2[1] = (v91_acc[1]);
          r2[2] = (v91_acc[2]);
          r2[3] = (v91_acc[3]);
          float v96_data = r1[4];
          float v97_data = r1[5];
          float v98_data = r1[6];
          float v99_data = r1[7];
          float v100_tp{};
          float v101_tp{};
          float v102_tp{};
          float v103_tp{};
          tensorforge::transpose4x4b32(v100_tp, v101_tp, v102_tp, v103_tp, v96_data, v97_data, v98_data, v99_data);
          tensorforge::VectorT<float, 4> v104_acc{};
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v69_data, v109_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v70_data, v110_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v71_data, v111_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v77_data, v117_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v78_data, v118_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v79_data, v119_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v85_data, v125_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v86_data, v126_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v87_data, v127_acc, 2, 2, 0);
          r2[4] = (v128_acc[0]);
          r2[5] = (v128_acc[1]);
          r2[6] = (v128_acc[2]);
          r2[7] = (v128_acc[3]);
          float v133_data = r1[8];
          float v134_data = r1[9];
          float v135_data = r1[10];
          float v136_data = r1[11];
          float v137_tp{};
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          tensorforge::transpose4x4b32(v137_tp, v138_tp, v139_tp, v140_tp, v133_data, v134_data, v135_data, v136_data);
          tensorforge::VectorT<float, 4> v141_acc{};
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v146_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v148_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v154_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v155_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v156_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v84_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v85_data, v162_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v86_data, v163_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v87_data, v164_acc, 2, 2, 0);
          r2[8] = (v165_acc[0]);
          r2[9] = (v165_acc[1]);
          r2[10] = (v165_acc[2]);
          r2[11] = (v165_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v16_lead < 6) {
            #pragma unroll
            for (int32_t v174_i1 = 0; v174_i1 < 12; ++v174_i1) {
              float v176_data = r2[v174_i1];
              int32_t v183_a = v16_lead + (v174_i1 * 12);
              s0[(v183_a ^ ((v183_a >> 4) & 15))] = v176_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v192_i1 = 0; v192_i1 < 12; ++v192_i1) {
              float v200_data = __builtin_nontemporal_load(&glb_m4[(v16_lead + (v192_i1 * 12))]);
              r5[v192_i1] = v200_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v207_tp{};
          float v208_tp{};
          float v209_tp{};
          float v210_tp{};
          tensorforge::transpose4x4b32(v207_tp, v208_tp, v209_tp, v210_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v211_acc{};
          float v212_data = r3[0];
          float v213_data = r3[1];
          float v214_data = r3[2];
          float v215_data = r3[3];
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v212_data, v211_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v213_data, v216_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v214_data, v217_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v215_data, v218_acc, 2, 0, 0);
          float v220_data = r3[4];
          float v221_data = r3[5];
          float v222_data = r3[6];
          float v223_data = r3[7];
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v220_data, v219_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v221_data, v224_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v222_data, v225_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v223_data, v226_acc, 2, 1, 0);
          float v228_data = r3[8];
          float v229_data = r3[9];
          float v230_data = r3[10];
          float v231_data = r3[11];
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v228_data, v227_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v229_data, v232_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v230_data, v233_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v231_data, v234_acc, 2, 2, 0);
          r4[0] = (v235_acc[0]);
          r4[1] = (v235_acc[1]);
          r4[2] = (v235_acc[2]);
          r4[3] = (v235_acc[3]);
          float v244_tp{};
          float v245_tp{};
          float v246_tp{};
          float v247_tp{};
          tensorforge::transpose4x4b32(v244_tp, v245_tp, v246_tp, v247_tp, v96_data, v97_data, v98_data, v99_data);
          tensorforge::VectorT<float, 4> v248_acc{};
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v212_data, v248_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v213_data, v253_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v214_data, v254_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v215_data, v255_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v220_data, v256_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v221_data, v261_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v222_data, v262_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v223_data, v263_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v228_data, v264_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v229_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v230_data, v270_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v231_data, v271_acc, 2, 2, 0);
          r4[4] = (v272_acc[0]);
          r4[5] = (v272_acc[1]);
          r4[6] = (v272_acc[2]);
          r4[7] = (v272_acc[3]);
          float v281_tp{};
          float v282_tp{};
          float v283_tp{};
          float v284_tp{};
          tensorforge::transpose4x4b32(v281_tp, v282_tp, v283_tp, v284_tp, v133_data, v134_data, v135_data, v136_data);
          tensorforge::VectorT<float, 4> v285_acc{};
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v212_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v213_data, v290_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v214_data, v291_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v215_data, v292_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v220_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v221_data, v298_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v222_data, v299_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v223_data, v300_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v228_data, v301_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v229_data, v306_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v230_data, v307_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v231_data, v308_acc, 2, 2, 0);
          r4[8] = (v309_acc[0]);
          r4[9] = (v309_acc[1]);
          r4[10] = (v309_acc[2]);
          r4[11] = (v309_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v16_lead < 6) {
            int32_t v326_off = v16_lead + 6;
            #pragma unroll
            for (int32_t v318_i1 = 0; v318_i1 < 12; ++v318_i1) {
              float v320_data = r4[v318_i1];
              int32_t v328_a = v326_off + (v318_i1 * 12);
              s0[(v328_a ^ ((v328_a >> 4) & 15))] = v320_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v342_data = s0[(v16_lead ^ ((v16_lead >> 4) & 15))];
          int32_t v348_a = v16_lead + 12;
          float v352_data = s0[(v348_a ^ ((v348_a >> 4) & 15))];
          int32_t v358_a = v16_lead + 24;
          float v362_data = s0[(v358_a ^ ((v358_a >> 4) & 15))];
          int32_t v368_a = v16_lead + 36;
          float v372_data = s0[(v368_a ^ ((v368_a >> 4) & 15))];
          float v373_tp{};
          float v374_tp{};
          float v375_tp{};
          float v376_tp{};
          tensorforge::transpose4x4b32(v373_tp, v374_tp, v375_tp, v376_tp, v342_data, v352_data, v362_data, v372_data);
          tensorforge::VectorT<float, 4> v377_acc{};
          float v378_data = r5[0];
          float v379_data = r5[1];
          float v380_data = r5[2];
          float v381_data = r5[3];
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v378_data, v377_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v379_data, v382_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v380_data, v383_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v381_data, v384_acc, 2, 0, 0);
          float v386_data = r5[4];
          float v387_data = r5[5];
          float v388_data = r5[6];
          float v389_data = r5[7];
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v386_data, v385_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v387_data, v390_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v388_data, v391_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v389_data, v392_acc, 2, 1, 0);
          float v394_data = r5[8];
          float v395_data = r5[9];
          float v396_data = r5[10];
          float v397_data = r5[11];
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v394_data, v393_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v395_data, v398_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v396_data, v399_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v397_data, v400_acc, 2, 2, 0);
          r6[0] = (v401_acc[0]);
          r6[1] = (v401_acc[1]);
          r6[2] = (v401_acc[2]);
          r6[3] = (v401_acc[3]);
          int32_t v411_a = v16_lead + 48;
          float v415_data = s0[(v411_a ^ ((v411_a >> 4) & 15))];
          int32_t v421_a = v16_lead + 60;
          float v425_data = s0[(v421_a ^ ((v421_a >> 4) & 15))];
          int32_t v431_a = v16_lead + 72;
          float v435_data = s0[(v431_a ^ ((v431_a >> 4) & 15))];
          int32_t v441_a = v16_lead + 84;
          float v445_data = s0[(v441_a ^ ((v441_a >> 4) & 15))];
          float v446_tp{};
          float v447_tp{};
          float v448_tp{};
          float v449_tp{};
          tensorforge::transpose4x4b32(v446_tp, v447_tp, v448_tp, v449_tp, v415_data, v425_data, v435_data, v445_data);
          tensorforge::VectorT<float, 4> v450_acc{};
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v378_data, v450_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v379_data, v455_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v380_data, v456_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v381_data, v457_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v386_data, v458_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v387_data, v463_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v388_data, v464_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v389_data, v465_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v394_data, v466_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v395_data, v471_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v396_data, v472_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v397_data, v473_acc, 2, 2, 0);
          r6[4] = (v474_acc[0]);
          r6[5] = (v474_acc[1]);
          r6[6] = (v474_acc[2]);
          r6[7] = (v474_acc[3]);
          int32_t v484_a = v16_lead + 96;
          float v488_data = s0[(v484_a ^ ((v484_a >> 4) & 15))];
          int32_t v494_a = v16_lead + 108;
          float v498_data = s0[(v494_a ^ ((v494_a >> 4) & 15))];
          int32_t v504_a = v16_lead + 120;
          float v508_data = s0[(v504_a ^ ((v504_a >> 4) & 15))];
          int32_t v514_a = v16_lead + 132;
          float v518_data = s0[(v514_a ^ ((v514_a >> 4) & 15))];
          float v519_tp{};
          float v520_tp{};
          float v521_tp{};
          float v522_tp{};
          tensorforge::transpose4x4b32(v519_tp, v520_tp, v521_tp, v522_tp, v488_data, v498_data, v508_data, v518_data);
          tensorforge::VectorT<float, 4> v523_acc{};
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v519_tp, v378_data, v523_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v520_tp, v379_data, v528_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v521_tp, v380_data, v529_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v522_tp, v381_data, v530_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v519_tp, v386_data, v531_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v520_tp, v387_data, v536_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v521_tp, v388_data, v537_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v522_tp, v389_data, v538_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v519_tp, v394_data, v539_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v520_tp, v395_data, v544_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v521_tp, v396_data, v545_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v522_tp, v397_data, v546_acc, 2, 2, 0);
          r6[8] = (v547_acc[0]);
          r6[9] = (v547_acc[1]);
          r6[10] = (v547_acc[2]);
          r6[11] = (v547_acc[3]);
          // glb_m3 = store{r>g}(r6);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v556_i1 = 0; v556_i1 < 12; ++v556_i1) {
              float v558_data = r6[v556_i1];
              glb_m3[(v16_lead + (v556_i1 * 12))] = v558_data;
            }
          }
        }
      }
    }
  }
}

