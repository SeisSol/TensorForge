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
          int32_t v9_lead = threadIdx.x % 16;
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 12; ++v11_i1) {
              int32_t v17_a = v11_i1 * 6;
              int32_t v18_a = v9_lead + v17_a;
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v9_lead + v17_a)]);
              r0[v11_i1] = v26_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          float v29_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          float v30_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v30_lin;
          float v31_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v31_lin;
          float v32_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v32_lin;
          float v33_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v33_lin;
          float v34_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v34_lin;
          float v35_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v35_lin;
          float v36_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v36_lin;
          float v37_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v37_lin;
          float v38_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v38_lin;
          float v39_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v39_lin;
          float v40_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v40_lin;
          float v41_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v41_lin;
          float v42_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v42_lin;
          float v43_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v43_lin;
          float v44_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v44_lin;
          float v45_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v45_lin;
          float v46_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v46_lin;
          float v47_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v47_lin;
          float v48_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v48_lin;
          float v49_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v49_lin;
          float v50_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v50_lin;
          float v51_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v51_lin;
          float v52_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v52_lin;
          float v53_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v53_lin;
          float v54_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v54_lin;
          float v55_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v55_lin;
          float v56_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v56_lin;
          float v57_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v57_lin;
          float v58_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v58_lin;
          float v59_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v59_lin;
          float v60_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v60_lin;
          float v61_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v61_lin;
          float v62_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v62_lin;
          float v63_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v63_lin;
          float v64_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v64_lin;
          float v65_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v65_lin;
          float v66_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v66_lin;
          float v67_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v67_lin;
          float v68_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v68_lin;
          float v69_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v69_lin;
          float v70_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v70_lin;
          float v71_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v71_lin;
          float v72_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v72_lin;
          float v73_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v73_lin;
          float v74_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v74_lin;
          float v75_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v75_lin;
          float v76_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v76_lin;
          float v77_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v77_lin;
          float v78_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v78_lin;
          float v79_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v79_lin;
          float v80_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v80_lin;
          float v81_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v81_lin;
          float v82_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v82_lin;
          float v83_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v83_lin;
          float v84_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v84_lin;
          float v85_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v85_lin;
          float v86_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v86_lin;
          float v87_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v87_lin;
          float v88_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v88_lin;
          float v89_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v89_lin;
          float v90_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v90_lin;
          float v91_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v91_lin;
          float v92_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v92_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v98_i1 = 0; v98_i1 < 12; ++v98_i1) {
              int32_t v104_a = v98_i1 * 6;
              int32_t v105_a = v9_lead + v104_a;
              float v113_data = __builtin_nontemporal_load(&glb_m2[(v9_lead + v104_a)]);
              r3[v98_i1] = v113_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v116_data = r1[0];
          float v117_data = r1[1];
          float v118_data = r1[2];
          float v119_data = r1[3];
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          tensorforge::transpose4x4b32(v120_tp, v121_tp, v122_tp, v123_tp, v116_data, v117_data, v118_data, v119_data);
          tensorforge::VectorT<float, 4> v124_acc{};
          float v125_data = r0[0];
          float v126_data = r0[1];
          float v127_data = r0[2];
          float v128_data = r0[3];
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v125_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v126_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v127_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v128_data, v131_acc, 2, 0, 0);
          float v133_data = r0[4];
          float v134_data = r0[5];
          float v135_data = r0[6];
          float v136_data = r0[7];
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v133_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v134_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v135_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v136_data, v139_acc, 2, 1, 0);
          float v141_data = r0[8];
          float v142_data = r0[9];
          float v143_data = r0[10];
          float v144_data = r0[11];
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v141_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v142_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v143_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v144_data, v147_acc, 2, 2, 0);
          r2[0] = (v148_acc[0]);
          r2[1] = (v148_acc[1]);
          r2[2] = (v148_acc[2]);
          r2[3] = (v148_acc[3]);
          float v153_data = r1[4];
          float v154_data = r1[5];
          float v155_data = r1[6];
          float v156_data = r1[7];
          float v157_tp{};
          float v158_tp{};
          float v159_tp{};
          float v160_tp{};
          tensorforge::transpose4x4b32(v157_tp, v158_tp, v159_tp, v160_tp, v153_data, v154_data, v155_data, v156_data);
          tensorforge::VectorT<float, 4> v161_acc{};
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v125_data, v161_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v126_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v127_data, v167_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v128_data, v168_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v133_data, v169_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v134_data, v174_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v135_data, v175_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v136_data, v176_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v141_data, v177_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v142_data, v182_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v143_data, v183_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v144_data, v184_acc, 2, 2, 0);
          r2[4] = (v185_acc[0]);
          r2[5] = (v185_acc[1]);
          r2[6] = (v185_acc[2]);
          r2[7] = (v185_acc[3]);
          float v190_data = r1[8];
          float v191_data = r1[9];
          float v192_data = r1[10];
          float v193_data = r1[11];
          float v194_tp{};
          float v195_tp{};
          float v196_tp{};
          float v197_tp{};
          tensorforge::transpose4x4b32(v194_tp, v195_tp, v196_tp, v197_tp, v190_data, v191_data, v192_data, v193_data);
          tensorforge::VectorT<float, 4> v198_acc{};
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v125_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v126_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v127_data, v204_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v128_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v133_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v134_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v135_data, v212_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v136_data, v213_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v141_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v142_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v143_data, v220_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v144_data, v221_acc, 2, 2, 0);
          r2[8] = (v222_acc[0]);
          r2[9] = (v222_acc[1]);
          r2[10] = (v222_acc[2]);
          r2[11] = (v222_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v232_i1 = 0; v232_i1 < 12; ++v232_i1) {
              int32_t v233_a = 0 + v232_i1;
              float v235_data = r2[v232_i1];
              s0[(v9_lead + (v232_i1 * 12))] = v235_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v9_lead < 12) {
            #pragma unroll
            for (int32_t v248_i1 = 0; v248_i1 < 12; ++v248_i1) {
              int32_t v254_a = v248_i1 * 12;
              int32_t v255_a = v9_lead + v254_a;
              float v263_data = __builtin_nontemporal_load(&glb_m4[(v9_lead + v254_a)]);
              r5[v248_i1] = v263_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v270_tp{};
          float v271_tp{};
          float v272_tp{};
          float v273_tp{};
          tensorforge::transpose4x4b32(v270_tp, v271_tp, v272_tp, v273_tp, v116_data, v117_data, v118_data, v119_data);
          tensorforge::VectorT<float, 4> v274_acc{};
          float v275_data = r3[0];
          float v276_data = r3[1];
          float v277_data = r3[2];
          float v278_data = r3[3];
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v275_data, v274_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v276_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v277_data, v280_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v278_data, v281_acc, 2, 0, 0);
          float v283_data = r3[4];
          float v284_data = r3[5];
          float v285_data = r3[6];
          float v286_data = r3[7];
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v283_data, v282_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v284_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v285_data, v288_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v286_data, v289_acc, 2, 1, 0);
          float v291_data = r3[8];
          float v292_data = r3[9];
          float v293_data = r3[10];
          float v294_data = r3[11];
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v291_data, v290_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v292_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v293_data, v296_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v294_data, v297_acc, 2, 2, 0);
          r4[0] = (v298_acc[0]);
          r4[1] = (v298_acc[1]);
          r4[2] = (v298_acc[2]);
          r4[3] = (v298_acc[3]);
          float v307_tp{};
          float v308_tp{};
          float v309_tp{};
          float v310_tp{};
          tensorforge::transpose4x4b32(v307_tp, v308_tp, v309_tp, v310_tp, v153_data, v154_data, v155_data, v156_data);
          tensorforge::VectorT<float, 4> v311_acc{};
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v275_data, v311_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v276_data, v316_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v277_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v278_data, v318_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v283_data, v319_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v284_data, v324_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v285_data, v325_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v286_data, v326_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v307_tp, v291_data, v327_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v292_data, v332_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v293_data, v333_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v294_data, v334_acc, 2, 2, 0);
          r4[4] = (v335_acc[0]);
          r4[5] = (v335_acc[1]);
          r4[6] = (v335_acc[2]);
          r4[7] = (v335_acc[3]);
          float v344_tp{};
          float v345_tp{};
          float v346_tp{};
          float v347_tp{};
          tensorforge::transpose4x4b32(v344_tp, v345_tp, v346_tp, v347_tp, v190_data, v191_data, v192_data, v193_data);
          tensorforge::VectorT<float, 4> v348_acc{};
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v275_data, v348_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v276_data, v353_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v277_data, v354_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v278_data, v355_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v283_data, v356_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v284_data, v361_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v285_data, v362_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v286_data, v363_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v291_data, v364_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v292_data, v369_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v293_data, v370_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v294_data, v371_acc, 2, 2, 0);
          r4[8] = (v372_acc[0]);
          r4[9] = (v372_acc[1]);
          r4[10] = (v372_acc[2]);
          r4[11] = (v372_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v9_lead < 6) {
            int32_t v390_off = v9_lead + 6;
            #pragma unroll
            for (int32_t v381_i1 = 0; v381_i1 < 12; ++v381_i1) {
              int32_t v382_a = 0 + v381_i1;
              float v384_data = r4[v381_i1];
              s0[(v390_off + (v381_i1 * 12))] = v384_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v394_data = r5[0];
          float v395_data = r5[1];
          float v396_data = r5[2];
          float v397_data = r5[3];
          float v398_data = r5[4];
          float v399_data = r5[5];
          float v400_data = r5[6];
          float v401_data = r5[7];
          float v402_data = r5[8];
          float v403_data = r5[9];
          float v404_data = r5[10];
          float v405_data = r5[11];
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
          float v417_acc{};
          float v418_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v406_acc, v418_lin, v394_data);
          tensorforge::fmacdpp16<1>(v406_acc, v418_lin, v395_data);
          tensorforge::fmacdpp16<2>(v406_acc, v418_lin, v396_data);
          tensorforge::fmacdpp16<3>(v406_acc, v418_lin, v397_data);
          tensorforge::fmacdpp16<4>(v406_acc, v418_lin, v398_data);
          tensorforge::fmacdpp16<5>(v406_acc, v418_lin, v399_data);
          tensorforge::fmacdpp16<6>(v406_acc, v418_lin, v400_data);
          tensorforge::fmacdpp16<7>(v406_acc, v418_lin, v401_data);
          tensorforge::fmacdpp16<8>(v406_acc, v418_lin, v402_data);
          tensorforge::fmacdpp16<9>(v406_acc, v418_lin, v403_data);
          tensorforge::fmacdpp16<10>(v406_acc, v418_lin, v404_data);
          tensorforge::fmacdpp16<11>(v406_acc, v418_lin, v405_data);
          tensorforge::fmacdpp16<12>(v407_acc, v418_lin, v394_data);
          tensorforge::fmacdpp16<13>(v407_acc, v418_lin, v395_data);
          tensorforge::fmacdpp16<14>(v407_acc, v418_lin, v396_data);
          tensorforge::fmacdpp16<15>(v407_acc, v418_lin, v397_data);
          float v419_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v407_acc, v419_lin, v398_data);
          tensorforge::fmacdpp16<1>(v407_acc, v419_lin, v399_data);
          tensorforge::fmacdpp16<2>(v407_acc, v419_lin, v400_data);
          tensorforge::fmacdpp16<3>(v407_acc, v419_lin, v401_data);
          tensorforge::fmacdpp16<4>(v407_acc, v419_lin, v402_data);
          tensorforge::fmacdpp16<5>(v407_acc, v419_lin, v403_data);
          tensorforge::fmacdpp16<6>(v407_acc, v419_lin, v404_data);
          tensorforge::fmacdpp16<7>(v407_acc, v419_lin, v405_data);
          tensorforge::fmacdpp16<8>(v408_acc, v419_lin, v394_data);
          tensorforge::fmacdpp16<9>(v408_acc, v419_lin, v395_data);
          tensorforge::fmacdpp16<10>(v408_acc, v419_lin, v396_data);
          tensorforge::fmacdpp16<11>(v408_acc, v419_lin, v397_data);
          tensorforge::fmacdpp16<12>(v408_acc, v419_lin, v398_data);
          tensorforge::fmacdpp16<13>(v408_acc, v419_lin, v399_data);
          tensorforge::fmacdpp16<14>(v408_acc, v419_lin, v400_data);
          tensorforge::fmacdpp16<15>(v408_acc, v419_lin, v401_data);
          float v420_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v408_acc, v420_lin, v402_data);
          tensorforge::fmacdpp16<1>(v408_acc, v420_lin, v403_data);
          tensorforge::fmacdpp16<2>(v408_acc, v420_lin, v404_data);
          tensorforge::fmacdpp16<3>(v408_acc, v420_lin, v405_data);
          tensorforge::fmacdpp16<4>(v409_acc, v420_lin, v394_data);
          tensorforge::fmacdpp16<5>(v409_acc, v420_lin, v395_data);
          tensorforge::fmacdpp16<6>(v409_acc, v420_lin, v396_data);
          tensorforge::fmacdpp16<7>(v409_acc, v420_lin, v397_data);
          tensorforge::fmacdpp16<8>(v409_acc, v420_lin, v398_data);
          tensorforge::fmacdpp16<9>(v409_acc, v420_lin, v399_data);
          tensorforge::fmacdpp16<10>(v409_acc, v420_lin, v400_data);
          tensorforge::fmacdpp16<11>(v409_acc, v420_lin, v401_data);
          tensorforge::fmacdpp16<12>(v409_acc, v420_lin, v402_data);
          tensorforge::fmacdpp16<13>(v409_acc, v420_lin, v403_data);
          tensorforge::fmacdpp16<14>(v409_acc, v420_lin, v404_data);
          tensorforge::fmacdpp16<15>(v409_acc, v420_lin, v405_data);
          float v421_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v410_acc, v421_lin, v394_data);
          tensorforge::fmacdpp16<1>(v410_acc, v421_lin, v395_data);
          tensorforge::fmacdpp16<2>(v410_acc, v421_lin, v396_data);
          tensorforge::fmacdpp16<3>(v410_acc, v421_lin, v397_data);
          tensorforge::fmacdpp16<4>(v410_acc, v421_lin, v398_data);
          tensorforge::fmacdpp16<5>(v410_acc, v421_lin, v399_data);
          tensorforge::fmacdpp16<6>(v410_acc, v421_lin, v400_data);
          tensorforge::fmacdpp16<7>(v410_acc, v421_lin, v401_data);
          tensorforge::fmacdpp16<8>(v410_acc, v421_lin, v402_data);
          tensorforge::fmacdpp16<9>(v410_acc, v421_lin, v403_data);
          tensorforge::fmacdpp16<10>(v410_acc, v421_lin, v404_data);
          tensorforge::fmacdpp16<11>(v410_acc, v421_lin, v405_data);
          tensorforge::fmacdpp16<12>(v411_acc, v421_lin, v394_data);
          tensorforge::fmacdpp16<13>(v411_acc, v421_lin, v395_data);
          tensorforge::fmacdpp16<14>(v411_acc, v421_lin, v396_data);
          tensorforge::fmacdpp16<15>(v411_acc, v421_lin, v397_data);
          float v422_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v411_acc, v422_lin, v398_data);
          tensorforge::fmacdpp16<1>(v411_acc, v422_lin, v399_data);
          tensorforge::fmacdpp16<2>(v411_acc, v422_lin, v400_data);
          tensorforge::fmacdpp16<3>(v411_acc, v422_lin, v401_data);
          tensorforge::fmacdpp16<4>(v411_acc, v422_lin, v402_data);
          tensorforge::fmacdpp16<5>(v411_acc, v422_lin, v403_data);
          tensorforge::fmacdpp16<6>(v411_acc, v422_lin, v404_data);
          tensorforge::fmacdpp16<7>(v411_acc, v422_lin, v405_data);
          tensorforge::fmacdpp16<8>(v412_acc, v422_lin, v394_data);
          tensorforge::fmacdpp16<9>(v412_acc, v422_lin, v395_data);
          tensorforge::fmacdpp16<10>(v412_acc, v422_lin, v396_data);
          tensorforge::fmacdpp16<11>(v412_acc, v422_lin, v397_data);
          tensorforge::fmacdpp16<12>(v412_acc, v422_lin, v398_data);
          tensorforge::fmacdpp16<13>(v412_acc, v422_lin, v399_data);
          tensorforge::fmacdpp16<14>(v412_acc, v422_lin, v400_data);
          tensorforge::fmacdpp16<15>(v412_acc, v422_lin, v401_data);
          float v423_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v412_acc, v423_lin, v402_data);
          tensorforge::fmacdpp16<1>(v412_acc, v423_lin, v403_data);
          tensorforge::fmacdpp16<2>(v412_acc, v423_lin, v404_data);
          tensorforge::fmacdpp16<3>(v412_acc, v423_lin, v405_data);
          tensorforge::fmacdpp16<4>(v413_acc, v423_lin, v394_data);
          tensorforge::fmacdpp16<5>(v413_acc, v423_lin, v395_data);
          tensorforge::fmacdpp16<6>(v413_acc, v423_lin, v396_data);
          tensorforge::fmacdpp16<7>(v413_acc, v423_lin, v397_data);
          tensorforge::fmacdpp16<8>(v413_acc, v423_lin, v398_data);
          tensorforge::fmacdpp16<9>(v413_acc, v423_lin, v399_data);
          tensorforge::fmacdpp16<10>(v413_acc, v423_lin, v400_data);
          tensorforge::fmacdpp16<11>(v413_acc, v423_lin, v401_data);
          tensorforge::fmacdpp16<12>(v413_acc, v423_lin, v402_data);
          tensorforge::fmacdpp16<13>(v413_acc, v423_lin, v403_data);
          tensorforge::fmacdpp16<14>(v413_acc, v423_lin, v404_data);
          tensorforge::fmacdpp16<15>(v413_acc, v423_lin, v405_data);
          float v424_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v414_acc, v424_lin, v394_data);
          tensorforge::fmacdpp16<1>(v414_acc, v424_lin, v395_data);
          tensorforge::fmacdpp16<2>(v414_acc, v424_lin, v396_data);
          tensorforge::fmacdpp16<3>(v414_acc, v424_lin, v397_data);
          tensorforge::fmacdpp16<4>(v414_acc, v424_lin, v398_data);
          tensorforge::fmacdpp16<5>(v414_acc, v424_lin, v399_data);
          tensorforge::fmacdpp16<6>(v414_acc, v424_lin, v400_data);
          tensorforge::fmacdpp16<7>(v414_acc, v424_lin, v401_data);
          tensorforge::fmacdpp16<8>(v414_acc, v424_lin, v402_data);
          tensorforge::fmacdpp16<9>(v414_acc, v424_lin, v403_data);
          tensorforge::fmacdpp16<10>(v414_acc, v424_lin, v404_data);
          tensorforge::fmacdpp16<11>(v414_acc, v424_lin, v405_data);
          tensorforge::fmacdpp16<12>(v415_acc, v424_lin, v394_data);
          tensorforge::fmacdpp16<13>(v415_acc, v424_lin, v395_data);
          tensorforge::fmacdpp16<14>(v415_acc, v424_lin, v396_data);
          tensorforge::fmacdpp16<15>(v415_acc, v424_lin, v397_data);
          float v425_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v415_acc, v425_lin, v398_data);
          tensorforge::fmacdpp16<1>(v415_acc, v425_lin, v399_data);
          tensorforge::fmacdpp16<2>(v415_acc, v425_lin, v400_data);
          tensorforge::fmacdpp16<3>(v415_acc, v425_lin, v401_data);
          tensorforge::fmacdpp16<4>(v415_acc, v425_lin, v402_data);
          tensorforge::fmacdpp16<5>(v415_acc, v425_lin, v403_data);
          tensorforge::fmacdpp16<6>(v415_acc, v425_lin, v404_data);
          tensorforge::fmacdpp16<7>(v415_acc, v425_lin, v405_data);
          tensorforge::fmacdpp16<8>(v416_acc, v425_lin, v394_data);
          tensorforge::fmacdpp16<9>(v416_acc, v425_lin, v395_data);
          tensorforge::fmacdpp16<10>(v416_acc, v425_lin, v396_data);
          tensorforge::fmacdpp16<11>(v416_acc, v425_lin, v397_data);
          tensorforge::fmacdpp16<12>(v416_acc, v425_lin, v398_data);
          tensorforge::fmacdpp16<13>(v416_acc, v425_lin, v399_data);
          tensorforge::fmacdpp16<14>(v416_acc, v425_lin, v400_data);
          tensorforge::fmacdpp16<15>(v416_acc, v425_lin, v401_data);
          float v426_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v416_acc, v426_lin, v402_data);
          tensorforge::fmacdpp16<1>(v416_acc, v426_lin, v403_data);
          tensorforge::fmacdpp16<2>(v416_acc, v426_lin, v404_data);
          tensorforge::fmacdpp16<3>(v416_acc, v426_lin, v405_data);
          tensorforge::fmacdpp16<4>(v417_acc, v426_lin, v394_data);
          tensorforge::fmacdpp16<5>(v417_acc, v426_lin, v395_data);
          tensorforge::fmacdpp16<6>(v417_acc, v426_lin, v396_data);
          tensorforge::fmacdpp16<7>(v417_acc, v426_lin, v397_data);
          tensorforge::fmacdpp16<8>(v417_acc, v426_lin, v398_data);
          tensorforge::fmacdpp16<9>(v417_acc, v426_lin, v399_data);
          tensorforge::fmacdpp16<10>(v417_acc, v426_lin, v400_data);
          tensorforge::fmacdpp16<11>(v417_acc, v426_lin, v401_data);
          tensorforge::fmacdpp16<12>(v417_acc, v426_lin, v402_data);
          tensorforge::fmacdpp16<13>(v417_acc, v426_lin, v403_data);
          tensorforge::fmacdpp16<14>(v417_acc, v426_lin, v404_data);
          tensorforge::fmacdpp16<15>(v417_acc, v426_lin, v405_data);
          r6[0] = v406_acc;
          r6[1] = v407_acc;
          r6[2] = v408_acc;
          r6[3] = v409_acc;
          r6[4] = v410_acc;
          r6[5] = v411_acc;
          r6[6] = v412_acc;
          r6[7] = v413_acc;
          r6[8] = v414_acc;
          r6[9] = v415_acc;
          r6[10] = v416_acc;
          r6[11] = v417_acc;
          // glb_m3 = store{r>g}(r6);
          if (v9_lead < 12) {
            #pragma unroll
            for (int32_t v431_i1 = 0; v431_i1 < 12; ++v431_i1) {
              int32_t v432_a = 0 + v431_i1;
              float v434_data = r6[v431_i1];
              glb_m3[(v9_lead + (v431_i1 * 12))] = v434_data;
            }
          }
        }
      }
    }
  }
}

