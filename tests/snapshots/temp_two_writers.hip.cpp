// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3e24e7feaf, block.x * block.y * block.z, 3328 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3e24e7feaf), hipFuncAttributeMaxDynamicSharedMemorySize, 3328 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3e24e7feaf, grid, block, 3328 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
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
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v8_lead = threadIdx.x % 16;
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 12; ++v10_i1) {
              int32_t v16_a = v10_i1 * 6;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v8_lead + v16_a)]);
              int32_t v26_a = 0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          float v28_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          float v29_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v29_lin;
          float v30_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v30_lin;
          float v31_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v31_lin;
          float v32_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v32_lin;
          float v33_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v33_lin;
          float v34_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v34_lin;
          float v35_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v35_lin;
          float v36_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v36_lin;
          float v37_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v37_lin;
          float v38_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v38_lin;
          float v39_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v39_lin;
          float v40_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v40_lin;
          float v41_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v41_lin;
          float v42_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v42_lin;
          float v43_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v43_lin;
          float v44_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v44_lin;
          float v45_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v45_lin;
          float v46_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v46_lin;
          float v47_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v47_lin;
          float v48_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v48_lin;
          float v49_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v49_lin;
          float v50_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v50_lin;
          float v51_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v51_lin;
          float v52_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v52_lin;
          float v53_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v53_lin;
          float v54_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v54_lin;
          float v55_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v55_lin;
          float v56_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v56_lin;
          float v57_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v57_lin;
          float v58_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v58_lin;
          float v59_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v59_lin;
          float v60_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v60_lin;
          float v61_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v61_lin;
          float v62_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v62_lin;
          float v63_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v63_lin;
          float v64_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v64_lin;
          float v65_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v65_lin;
          float v66_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v66_lin;
          float v67_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v67_lin;
          float v68_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v68_lin;
          float v69_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v69_lin;
          float v70_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v70_lin;
          float v71_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v71_lin;
          float v72_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v72_lin;
          float v73_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v73_lin;
          float v74_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v74_lin;
          float v75_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v75_lin;
          float v76_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v76_lin;
          float v77_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v77_lin;
          float v78_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v78_lin;
          float v79_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v79_lin;
          float v80_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v80_lin;
          float v81_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v81_lin;
          float v82_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v82_lin;
          float v83_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v83_lin;
          float v84_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v84_lin;
          float v85_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v85_lin;
          float v86_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v86_lin;
          float v87_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v87_lin;
          float v88_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v88_lin;
          float v89_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v89_lin;
          float v90_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v90_lin;
          float v91_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v91_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 12; ++v97_i1) {
              int32_t v103_a = v97_i1 * 6;
              int32_t v104_a = v8_lead + v103_a;
              float v112_data = __builtin_nontemporal_load(&glb_m2[(v8_lead + v103_a)]);
              int32_t v113_a = 0 + v97_i1;
              r3[v113_a] = v112_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir2 = r2;
          float v115_data = r1[0];
          float v116_data = r1[1];
          float v117_data = r1[2];
          float v118_data = r1[3];
          float v119_tp{};
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          tensorforge::transpose4x4b32(v119_tp, v120_tp, v121_tp, v122_tp, v115_data, v116_data, v117_data, v118_data);
          tensorforge::VectorT<float, 4> v123_acc{};
          float v124_data = r0[0];
          float v125_data = r0[1];
          float v126_data = r0[2];
          float v127_data = r0[3];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v124_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v125_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v126_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v127_data, v130_acc, 2, 0, 0);
          float v132_data = r0[4];
          float v133_data = r0[5];
          float v134_data = r0[6];
          float v135_data = r0[7];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v132_data, v131_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v133_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v134_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v135_data, v138_acc, 2, 1, 0);
          float v140_data = r0[8];
          float v141_data = r0[9];
          float v142_data = r0[10];
          float v143_data = r0[11];
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v140_data, v139_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v141_data, v144_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v142_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v143_data, v146_acc, 2, 2, 0);
          ir2[0] = (v147_acc[0]);
          ir2[1] = (v147_acc[1]);
          ir2[2] = (v147_acc[2]);
          ir2[3] = (v147_acc[3]);
          float v152_data = r1[4];
          float v153_data = r1[5];
          float v154_data = r1[6];
          float v155_data = r1[7];
          float v156_tp{};
          float v157_tp{};
          float v158_tp{};
          float v159_tp{};
          tensorforge::transpose4x4b32(v156_tp, v157_tp, v158_tp, v159_tp, v152_data, v153_data, v154_data, v155_data);
          tensorforge::VectorT<float, 4> v160_acc{};
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v124_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v125_data, v165_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v126_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v127_data, v167_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v132_data, v168_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v133_data, v173_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v134_data, v174_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v135_data, v175_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v140_data, v176_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v141_data, v181_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v142_data, v182_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v143_data, v183_acc, 2, 2, 0);
          ir2[4] = (v184_acc[0]);
          ir2[5] = (v184_acc[1]);
          ir2[6] = (v184_acc[2]);
          ir2[7] = (v184_acc[3]);
          float v189_data = r1[8];
          float v190_data = r1[9];
          float v191_data = r1[10];
          float v192_data = r1[11];
          float v193_tp{};
          float v194_tp{};
          float v195_tp{};
          float v196_tp{};
          tensorforge::transpose4x4b32(v193_tp, v194_tp, v195_tp, v196_tp, v189_data, v190_data, v191_data, v192_data);
          tensorforge::VectorT<float, 4> v197_acc{};
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v124_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v125_data, v202_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v126_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v127_data, v204_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v132_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v133_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v134_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v135_data, v212_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v140_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v141_data, v218_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v142_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v143_data, v220_acc, 2, 2, 0);
          ir2[8] = (v221_acc[0]);
          ir2[9] = (v221_acc[1]);
          ir2[10] = (v221_acc[2]);
          ir2[11] = (v221_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v231_i1 = 0; v231_i1 < 12; ++v231_i1) {
              int32_t v232_a = 0 + v231_i1;
              float v234_data = r2[v231_i1];
              int32_t v241_a = v8_lead + (v231_i1 * 12);
              s0[v241_a] = v234_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v247_i1 = 0; v247_i1 < 12; ++v247_i1) {
              int32_t v253_a = v247_i1 * 12;
              int32_t v254_a = v8_lead + v253_a;
              float v262_data = __builtin_nontemporal_load(&glb_m4[(v8_lead + v253_a)]);
              int32_t v263_a = 0 + v247_i1;
              r5[v263_a] = v262_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir4 = r4;
          float v265_data = r1[0];
          float v266_data = r1[1];
          float v267_data = r1[2];
          float v268_data = r1[3];
          float v269_tp{};
          float v270_tp{};
          float v271_tp{};
          float v272_tp{};
          tensorforge::transpose4x4b32(v269_tp, v270_tp, v271_tp, v272_tp, v265_data, v266_data, v267_data, v268_data);
          tensorforge::VectorT<float, 4> v273_acc{};
          float v274_data = r3[0];
          float v275_data = r3[1];
          float v276_data = r3[2];
          float v277_data = r3[3];
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v274_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v275_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v276_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v277_data, v280_acc, 2, 0, 0);
          float v282_data = r3[4];
          float v283_data = r3[5];
          float v284_data = r3[6];
          float v285_data = r3[7];
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v282_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v283_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v284_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v285_data, v288_acc, 2, 1, 0);
          float v290_data = r3[8];
          float v291_data = r3[9];
          float v292_data = r3[10];
          float v293_data = r3[11];
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v290_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v291_data, v294_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v292_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v293_data, v296_acc, 2, 2, 0);
          ir4[0] = (v297_acc[0]);
          ir4[1] = (v297_acc[1]);
          ir4[2] = (v297_acc[2]);
          ir4[3] = (v297_acc[3]);
          float v302_data = r1[4];
          float v303_data = r1[5];
          float v304_data = r1[6];
          float v305_data = r1[7];
          float v306_tp{};
          float v307_tp{};
          float v308_tp{};
          float v309_tp{};
          tensorforge::transpose4x4b32(v306_tp, v307_tp, v308_tp, v309_tp, v302_data, v303_data, v304_data, v305_data);
          tensorforge::VectorT<float, 4> v310_acc{};
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v306_tp, v274_data, v310_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v275_data, v315_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v276_data, v316_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v277_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v306_tp, v282_data, v318_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v283_data, v323_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v284_data, v324_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v285_data, v325_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v306_tp, v290_data, v326_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v291_data, v331_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v292_data, v332_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v293_data, v333_acc, 2, 2, 0);
          ir4[4] = (v334_acc[0]);
          ir4[5] = (v334_acc[1]);
          ir4[6] = (v334_acc[2]);
          ir4[7] = (v334_acc[3]);
          float v339_data = r1[8];
          float v340_data = r1[9];
          float v341_data = r1[10];
          float v342_data = r1[11];
          float v343_tp{};
          float v344_tp{};
          float v345_tp{};
          float v346_tp{};
          tensorforge::transpose4x4b32(v343_tp, v344_tp, v345_tp, v346_tp, v339_data, v340_data, v341_data, v342_data);
          tensorforge::VectorT<float, 4> v347_acc{};
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v274_data, v347_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v275_data, v352_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v276_data, v353_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v277_data, v354_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v282_data, v355_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v283_data, v360_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v284_data, v361_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v285_data, v362_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v290_data, v363_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v291_data, v368_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v292_data, v369_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v293_data, v370_acc, 2, 2, 0);
          ir4[8] = (v371_acc[0]);
          ir4[9] = (v371_acc[1]);
          ir4[10] = (v371_acc[2]);
          ir4[11] = (v371_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v8_lead < 6) {
            int32_t v389_off = v8_lead + 6;
            #pragma unroll
            for (int32_t v380_i1 = 0; v380_i1 < 12; ++v380_i1) {
              int32_t v381_a = 0 + v380_i1;
              float v383_data = r4[v380_i1];
              int32_t v391_a = v389_off + (v380_i1 * 12);
              s0[v391_a] = v383_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          ;
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          auto& ir6 = r6;
          float v393_data = r5[0];
          float v394_data = r5[1];
          float v395_data = r5[2];
          float v396_data = r5[3];
          float v397_data = r5[4];
          float v398_data = r5[5];
          float v399_data = r5[6];
          float v400_data = r5[7];
          float v401_data = r5[8];
          float v402_data = r5[9];
          float v403_data = r5[10];
          float v404_data = r5[11];
          float v405_acc{};
          float v406_acc{};
          float v407_acc{};
          float v408_acc{};
          float v409_acc{};
          float v410_acc{};
          float v411_acc{};
          float v412_acc{};
          float v413_acc{};
          float v414_acc{};
          float v415_acc{};
          float v416_acc{};
          float v417_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v405_acc, v417_lin, v393_data);
          tensorforge::fmacdpp16<1>(v405_acc, v417_lin, v394_data);
          tensorforge::fmacdpp16<2>(v405_acc, v417_lin, v395_data);
          tensorforge::fmacdpp16<3>(v405_acc, v417_lin, v396_data);
          tensorforge::fmacdpp16<4>(v405_acc, v417_lin, v397_data);
          tensorforge::fmacdpp16<5>(v405_acc, v417_lin, v398_data);
          tensorforge::fmacdpp16<6>(v405_acc, v417_lin, v399_data);
          tensorforge::fmacdpp16<7>(v405_acc, v417_lin, v400_data);
          tensorforge::fmacdpp16<8>(v405_acc, v417_lin, v401_data);
          tensorforge::fmacdpp16<9>(v405_acc, v417_lin, v402_data);
          tensorforge::fmacdpp16<10>(v405_acc, v417_lin, v403_data);
          tensorforge::fmacdpp16<11>(v405_acc, v417_lin, v404_data);
          tensorforge::fmacdpp16<12>(v406_acc, v417_lin, v393_data);
          tensorforge::fmacdpp16<13>(v406_acc, v417_lin, v394_data);
          tensorforge::fmacdpp16<14>(v406_acc, v417_lin, v395_data);
          tensorforge::fmacdpp16<15>(v406_acc, v417_lin, v396_data);
          float v418_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v406_acc, v418_lin, v397_data);
          tensorforge::fmacdpp16<1>(v406_acc, v418_lin, v398_data);
          tensorforge::fmacdpp16<2>(v406_acc, v418_lin, v399_data);
          tensorforge::fmacdpp16<3>(v406_acc, v418_lin, v400_data);
          tensorforge::fmacdpp16<4>(v406_acc, v418_lin, v401_data);
          tensorforge::fmacdpp16<5>(v406_acc, v418_lin, v402_data);
          tensorforge::fmacdpp16<6>(v406_acc, v418_lin, v403_data);
          tensorforge::fmacdpp16<7>(v406_acc, v418_lin, v404_data);
          tensorforge::fmacdpp16<8>(v407_acc, v418_lin, v393_data);
          tensorforge::fmacdpp16<9>(v407_acc, v418_lin, v394_data);
          tensorforge::fmacdpp16<10>(v407_acc, v418_lin, v395_data);
          tensorforge::fmacdpp16<11>(v407_acc, v418_lin, v396_data);
          tensorforge::fmacdpp16<12>(v407_acc, v418_lin, v397_data);
          tensorforge::fmacdpp16<13>(v407_acc, v418_lin, v398_data);
          tensorforge::fmacdpp16<14>(v407_acc, v418_lin, v399_data);
          tensorforge::fmacdpp16<15>(v407_acc, v418_lin, v400_data);
          float v419_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v407_acc, v419_lin, v401_data);
          tensorforge::fmacdpp16<1>(v407_acc, v419_lin, v402_data);
          tensorforge::fmacdpp16<2>(v407_acc, v419_lin, v403_data);
          tensorforge::fmacdpp16<3>(v407_acc, v419_lin, v404_data);
          tensorforge::fmacdpp16<4>(v408_acc, v419_lin, v393_data);
          tensorforge::fmacdpp16<5>(v408_acc, v419_lin, v394_data);
          tensorforge::fmacdpp16<6>(v408_acc, v419_lin, v395_data);
          tensorforge::fmacdpp16<7>(v408_acc, v419_lin, v396_data);
          tensorforge::fmacdpp16<8>(v408_acc, v419_lin, v397_data);
          tensorforge::fmacdpp16<9>(v408_acc, v419_lin, v398_data);
          tensorforge::fmacdpp16<10>(v408_acc, v419_lin, v399_data);
          tensorforge::fmacdpp16<11>(v408_acc, v419_lin, v400_data);
          tensorforge::fmacdpp16<12>(v408_acc, v419_lin, v401_data);
          tensorforge::fmacdpp16<13>(v408_acc, v419_lin, v402_data);
          tensorforge::fmacdpp16<14>(v408_acc, v419_lin, v403_data);
          tensorforge::fmacdpp16<15>(v408_acc, v419_lin, v404_data);
          float v420_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v409_acc, v420_lin, v393_data);
          tensorforge::fmacdpp16<1>(v409_acc, v420_lin, v394_data);
          tensorforge::fmacdpp16<2>(v409_acc, v420_lin, v395_data);
          tensorforge::fmacdpp16<3>(v409_acc, v420_lin, v396_data);
          tensorforge::fmacdpp16<4>(v409_acc, v420_lin, v397_data);
          tensorforge::fmacdpp16<5>(v409_acc, v420_lin, v398_data);
          tensorforge::fmacdpp16<6>(v409_acc, v420_lin, v399_data);
          tensorforge::fmacdpp16<7>(v409_acc, v420_lin, v400_data);
          tensorforge::fmacdpp16<8>(v409_acc, v420_lin, v401_data);
          tensorforge::fmacdpp16<9>(v409_acc, v420_lin, v402_data);
          tensorforge::fmacdpp16<10>(v409_acc, v420_lin, v403_data);
          tensorforge::fmacdpp16<11>(v409_acc, v420_lin, v404_data);
          tensorforge::fmacdpp16<12>(v410_acc, v420_lin, v393_data);
          tensorforge::fmacdpp16<13>(v410_acc, v420_lin, v394_data);
          tensorforge::fmacdpp16<14>(v410_acc, v420_lin, v395_data);
          tensorforge::fmacdpp16<15>(v410_acc, v420_lin, v396_data);
          float v421_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v410_acc, v421_lin, v397_data);
          tensorforge::fmacdpp16<1>(v410_acc, v421_lin, v398_data);
          tensorforge::fmacdpp16<2>(v410_acc, v421_lin, v399_data);
          tensorforge::fmacdpp16<3>(v410_acc, v421_lin, v400_data);
          tensorforge::fmacdpp16<4>(v410_acc, v421_lin, v401_data);
          tensorforge::fmacdpp16<5>(v410_acc, v421_lin, v402_data);
          tensorforge::fmacdpp16<6>(v410_acc, v421_lin, v403_data);
          tensorforge::fmacdpp16<7>(v410_acc, v421_lin, v404_data);
          tensorforge::fmacdpp16<8>(v411_acc, v421_lin, v393_data);
          tensorforge::fmacdpp16<9>(v411_acc, v421_lin, v394_data);
          tensorforge::fmacdpp16<10>(v411_acc, v421_lin, v395_data);
          tensorforge::fmacdpp16<11>(v411_acc, v421_lin, v396_data);
          tensorforge::fmacdpp16<12>(v411_acc, v421_lin, v397_data);
          tensorforge::fmacdpp16<13>(v411_acc, v421_lin, v398_data);
          tensorforge::fmacdpp16<14>(v411_acc, v421_lin, v399_data);
          tensorforge::fmacdpp16<15>(v411_acc, v421_lin, v400_data);
          float v422_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v411_acc, v422_lin, v401_data);
          tensorforge::fmacdpp16<1>(v411_acc, v422_lin, v402_data);
          tensorforge::fmacdpp16<2>(v411_acc, v422_lin, v403_data);
          tensorforge::fmacdpp16<3>(v411_acc, v422_lin, v404_data);
          tensorforge::fmacdpp16<4>(v412_acc, v422_lin, v393_data);
          tensorforge::fmacdpp16<5>(v412_acc, v422_lin, v394_data);
          tensorforge::fmacdpp16<6>(v412_acc, v422_lin, v395_data);
          tensorforge::fmacdpp16<7>(v412_acc, v422_lin, v396_data);
          tensorforge::fmacdpp16<8>(v412_acc, v422_lin, v397_data);
          tensorforge::fmacdpp16<9>(v412_acc, v422_lin, v398_data);
          tensorforge::fmacdpp16<10>(v412_acc, v422_lin, v399_data);
          tensorforge::fmacdpp16<11>(v412_acc, v422_lin, v400_data);
          tensorforge::fmacdpp16<12>(v412_acc, v422_lin, v401_data);
          tensorforge::fmacdpp16<13>(v412_acc, v422_lin, v402_data);
          tensorforge::fmacdpp16<14>(v412_acc, v422_lin, v403_data);
          tensorforge::fmacdpp16<15>(v412_acc, v422_lin, v404_data);
          float v423_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v413_acc, v423_lin, v393_data);
          tensorforge::fmacdpp16<1>(v413_acc, v423_lin, v394_data);
          tensorforge::fmacdpp16<2>(v413_acc, v423_lin, v395_data);
          tensorforge::fmacdpp16<3>(v413_acc, v423_lin, v396_data);
          tensorforge::fmacdpp16<4>(v413_acc, v423_lin, v397_data);
          tensorforge::fmacdpp16<5>(v413_acc, v423_lin, v398_data);
          tensorforge::fmacdpp16<6>(v413_acc, v423_lin, v399_data);
          tensorforge::fmacdpp16<7>(v413_acc, v423_lin, v400_data);
          tensorforge::fmacdpp16<8>(v413_acc, v423_lin, v401_data);
          tensorforge::fmacdpp16<9>(v413_acc, v423_lin, v402_data);
          tensorforge::fmacdpp16<10>(v413_acc, v423_lin, v403_data);
          tensorforge::fmacdpp16<11>(v413_acc, v423_lin, v404_data);
          tensorforge::fmacdpp16<12>(v414_acc, v423_lin, v393_data);
          tensorforge::fmacdpp16<13>(v414_acc, v423_lin, v394_data);
          tensorforge::fmacdpp16<14>(v414_acc, v423_lin, v395_data);
          tensorforge::fmacdpp16<15>(v414_acc, v423_lin, v396_data);
          float v424_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v414_acc, v424_lin, v397_data);
          tensorforge::fmacdpp16<1>(v414_acc, v424_lin, v398_data);
          tensorforge::fmacdpp16<2>(v414_acc, v424_lin, v399_data);
          tensorforge::fmacdpp16<3>(v414_acc, v424_lin, v400_data);
          tensorforge::fmacdpp16<4>(v414_acc, v424_lin, v401_data);
          tensorforge::fmacdpp16<5>(v414_acc, v424_lin, v402_data);
          tensorforge::fmacdpp16<6>(v414_acc, v424_lin, v403_data);
          tensorforge::fmacdpp16<7>(v414_acc, v424_lin, v404_data);
          tensorforge::fmacdpp16<8>(v415_acc, v424_lin, v393_data);
          tensorforge::fmacdpp16<9>(v415_acc, v424_lin, v394_data);
          tensorforge::fmacdpp16<10>(v415_acc, v424_lin, v395_data);
          tensorforge::fmacdpp16<11>(v415_acc, v424_lin, v396_data);
          tensorforge::fmacdpp16<12>(v415_acc, v424_lin, v397_data);
          tensorforge::fmacdpp16<13>(v415_acc, v424_lin, v398_data);
          tensorforge::fmacdpp16<14>(v415_acc, v424_lin, v399_data);
          tensorforge::fmacdpp16<15>(v415_acc, v424_lin, v400_data);
          float v425_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v415_acc, v425_lin, v401_data);
          tensorforge::fmacdpp16<1>(v415_acc, v425_lin, v402_data);
          tensorforge::fmacdpp16<2>(v415_acc, v425_lin, v403_data);
          tensorforge::fmacdpp16<3>(v415_acc, v425_lin, v404_data);
          tensorforge::fmacdpp16<4>(v416_acc, v425_lin, v393_data);
          tensorforge::fmacdpp16<5>(v416_acc, v425_lin, v394_data);
          tensorforge::fmacdpp16<6>(v416_acc, v425_lin, v395_data);
          tensorforge::fmacdpp16<7>(v416_acc, v425_lin, v396_data);
          tensorforge::fmacdpp16<8>(v416_acc, v425_lin, v397_data);
          tensorforge::fmacdpp16<9>(v416_acc, v425_lin, v398_data);
          tensorforge::fmacdpp16<10>(v416_acc, v425_lin, v399_data);
          tensorforge::fmacdpp16<11>(v416_acc, v425_lin, v400_data);
          tensorforge::fmacdpp16<12>(v416_acc, v425_lin, v401_data);
          tensorforge::fmacdpp16<13>(v416_acc, v425_lin, v402_data);
          tensorforge::fmacdpp16<14>(v416_acc, v425_lin, v403_data);
          tensorforge::fmacdpp16<15>(v416_acc, v425_lin, v404_data);
          ir6[0] = v405_acc;
          ir6[1] = v406_acc;
          ir6[2] = v407_acc;
          ir6[3] = v408_acc;
          ir6[4] = v409_acc;
          ir6[5] = v410_acc;
          ir6[6] = v411_acc;
          ir6[7] = v412_acc;
          ir6[8] = v413_acc;
          ir6[9] = v414_acc;
          ir6[10] = v415_acc;
          ir6[11] = v416_acc;
          // glb_m3 = store{r>g}(r6);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v430_i1 = 0; v430_i1 < 12; ++v430_i1) {
              int32_t v431_a = 0 + v430_i1;
              float v433_data = r6[v430_i1];
              int32_t v440_a = v8_lead + (v430_i1 * 12);
              glb_m3[v440_a] = v433_data;
            }
          }
          ;
        }
      }
    }
  }
}

