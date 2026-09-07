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
          if (v16_lead < 6) {
            #pragma unroll
            for (int32_t v98_i1 = 0; v98_i1 < 12; ++v98_i1) {
              float v106_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v98_i1 * 6))]);
              r3[v98_i1] = v106_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v109_data = r1[0];
          float v110_data = r1[1];
          float v111_data = r1[2];
          float v112_data = r1[3];
          float v113_tp{};
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          tensorforge::transpose4x4b32(v113_tp, v114_tp, v115_tp, v116_tp, v109_data, v110_data, v111_data, v112_data);
          tensorforge::VectorT<float, 4> v117_acc{};
          float v118_data = r0[0];
          float v119_data = r0[1];
          float v120_data = r0[2];
          float v121_data = r0[3];
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v118_data, v117_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v122_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v120_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v121_data, v124_acc, 2, 0, 0);
          float v126_data = r0[4];
          float v127_data = r0[5];
          float v128_data = r0[6];
          float v129_data = r0[7];
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v126_data, v125_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v127_data, v130_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v128_data, v131_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v129_data, v132_acc, 2, 1, 0);
          float v134_data = r0[8];
          float v135_data = r0[9];
          float v136_data = r0[10];
          float v137_data = r0[11];
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v134_data, v133_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v135_data, v138_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v136_data, v139_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v137_data, v140_acc, 2, 2, 0);
          r2[0] = (v141_acc[0]);
          r2[1] = (v141_acc[1]);
          r2[2] = (v141_acc[2]);
          r2[3] = (v141_acc[3]);
          float v146_data = r1[4];
          float v147_data = r1[5];
          float v148_data = r1[6];
          float v149_data = r1[7];
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          float v153_tp{};
          tensorforge::transpose4x4b32(v150_tp, v151_tp, v152_tp, v153_tp, v146_data, v147_data, v148_data, v149_data);
          tensorforge::VectorT<float, 4> v154_acc{};
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v118_data, v154_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v119_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v120_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v121_data, v161_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v126_data, v162_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v127_data, v167_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v128_data, v168_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v129_data, v169_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v134_data, v170_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v135_data, v175_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v136_data, v176_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v137_data, v177_acc, 2, 2, 0);
          r2[4] = (v178_acc[0]);
          r2[5] = (v178_acc[1]);
          r2[6] = (v178_acc[2]);
          r2[7] = (v178_acc[3]);
          float v183_data = r1[8];
          float v184_data = r1[9];
          float v185_data = r1[10];
          float v186_data = r1[11];
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          tensorforge::transpose4x4b32(v187_tp, v188_tp, v189_tp, v190_tp, v183_data, v184_data, v185_data, v186_data);
          tensorforge::VectorT<float, 4> v191_acc{};
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v118_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v119_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v120_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v121_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v126_data, v199_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v127_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v128_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v129_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v134_data, v207_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v135_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v136_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v137_data, v214_acc, 2, 2, 0);
          r2[8] = (v215_acc[0]);
          r2[9] = (v215_acc[1]);
          r2[10] = (v215_acc[2]);
          r2[11] = (v215_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v16_lead < 6) {
            #pragma unroll
            for (int32_t v224_i1 = 0; v224_i1 < 12; ++v224_i1) {
              float v226_data = r2[v224_i1];
              s0[(v16_lead + (v224_i1 * 12))] = v226_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v239_i1 = 0; v239_i1 < 12; ++v239_i1) {
              float v247_data = __builtin_nontemporal_load(&glb_m4[(v16_lead + (v239_i1 * 12))]);
              r5[v239_i1] = v247_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v254_tp{};
          float v255_tp{};
          float v256_tp{};
          float v257_tp{};
          tensorforge::transpose4x4b32(v254_tp, v255_tp, v256_tp, v257_tp, v109_data, v110_data, v111_data, v112_data);
          tensorforge::VectorT<float, 4> v258_acc{};
          float v259_data = r3[0];
          float v260_data = r3[1];
          float v261_data = r3[2];
          float v262_data = r3[3];
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v259_data, v258_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v260_data, v263_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v261_data, v264_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v262_data, v265_acc, 2, 0, 0);
          float v267_data = r3[4];
          float v268_data = r3[5];
          float v269_data = r3[6];
          float v270_data = r3[7];
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v267_data, v266_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v268_data, v271_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v269_data, v272_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v270_data, v273_acc, 2, 1, 0);
          float v275_data = r3[8];
          float v276_data = r3[9];
          float v277_data = r3[10];
          float v278_data = r3[11];
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v275_data, v274_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v276_data, v279_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v277_data, v280_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v278_data, v281_acc, 2, 2, 0);
          r4[0] = (v282_acc[0]);
          r4[1] = (v282_acc[1]);
          r4[2] = (v282_acc[2]);
          r4[3] = (v282_acc[3]);
          float v291_tp{};
          float v292_tp{};
          float v293_tp{};
          float v294_tp{};
          tensorforge::transpose4x4b32(v291_tp, v292_tp, v293_tp, v294_tp, v146_data, v147_data, v148_data, v149_data);
          tensorforge::VectorT<float, 4> v295_acc{};
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v259_data, v295_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v260_data, v300_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v261_data, v301_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v262_data, v302_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v267_data, v303_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v268_data, v308_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v269_data, v309_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v270_data, v310_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v275_data, v311_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v276_data, v316_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v277_data, v317_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v278_data, v318_acc, 2, 2, 0);
          r4[4] = (v319_acc[0]);
          r4[5] = (v319_acc[1]);
          r4[6] = (v319_acc[2]);
          r4[7] = (v319_acc[3]);
          float v328_tp{};
          float v329_tp{};
          float v330_tp{};
          float v331_tp{};
          tensorforge::transpose4x4b32(v328_tp, v329_tp, v330_tp, v331_tp, v183_data, v184_data, v185_data, v186_data);
          tensorforge::VectorT<float, 4> v332_acc{};
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v259_data, v332_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v260_data, v337_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v261_data, v338_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v262_data, v339_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v267_data, v340_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v268_data, v345_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v269_data, v346_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v270_data, v347_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v275_data, v348_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v276_data, v353_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v277_data, v354_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v278_data, v355_acc, 2, 2, 0);
          r4[8] = (v356_acc[0]);
          r4[9] = (v356_acc[1]);
          r4[10] = (v356_acc[2]);
          r4[11] = (v356_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v16_lead < 6) {
            int32_t v373_off = v16_lead + 6;
            #pragma unroll
            for (int32_t v365_i1 = 0; v365_i1 < 12; ++v365_i1) {
              float v367_data = r4[v365_i1];
              s0[(v373_off + (v365_i1 * 12))] = v367_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v377_data = r5[0];
          float v378_data = r5[1];
          float v379_data = r5[2];
          float v380_data = r5[3];
          float v381_data = r5[4];
          float v382_data = r5[5];
          float v383_data = r5[6];
          float v384_data = r5[7];
          float v385_data = r5[8];
          float v386_data = r5[9];
          float v387_data = r5[10];
          float v388_data = r5[11];
          float v389_acc{};
          float v390_acc{};
          float v391_acc{};
          float v392_acc{};
          float v393_acc{};
          float v394_acc{};
          float v395_acc{};
          float v396_acc{};
          float v397_acc{};
          float v398_acc{};
          float v399_acc{};
          float v400_acc{};
          float v401_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v389_acc, v401_lin, v377_data);
          tensorforge::fmacdpp16<1>(v389_acc, v401_lin, v378_data);
          tensorforge::fmacdpp16<2>(v389_acc, v401_lin, v379_data);
          tensorforge::fmacdpp16<3>(v389_acc, v401_lin, v380_data);
          tensorforge::fmacdpp16<4>(v389_acc, v401_lin, v381_data);
          tensorforge::fmacdpp16<5>(v389_acc, v401_lin, v382_data);
          tensorforge::fmacdpp16<6>(v389_acc, v401_lin, v383_data);
          tensorforge::fmacdpp16<7>(v389_acc, v401_lin, v384_data);
          tensorforge::fmacdpp16<8>(v389_acc, v401_lin, v385_data);
          tensorforge::fmacdpp16<9>(v389_acc, v401_lin, v386_data);
          tensorforge::fmacdpp16<10>(v389_acc, v401_lin, v387_data);
          tensorforge::fmacdpp16<11>(v389_acc, v401_lin, v388_data);
          tensorforge::fmacdpp16<12>(v390_acc, v401_lin, v377_data);
          tensorforge::fmacdpp16<13>(v390_acc, v401_lin, v378_data);
          tensorforge::fmacdpp16<14>(v390_acc, v401_lin, v379_data);
          tensorforge::fmacdpp16<15>(v390_acc, v401_lin, v380_data);
          float v402_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v390_acc, v402_lin, v381_data);
          tensorforge::fmacdpp16<1>(v390_acc, v402_lin, v382_data);
          tensorforge::fmacdpp16<2>(v390_acc, v402_lin, v383_data);
          tensorforge::fmacdpp16<3>(v390_acc, v402_lin, v384_data);
          tensorforge::fmacdpp16<4>(v390_acc, v402_lin, v385_data);
          tensorforge::fmacdpp16<5>(v390_acc, v402_lin, v386_data);
          tensorforge::fmacdpp16<6>(v390_acc, v402_lin, v387_data);
          tensorforge::fmacdpp16<7>(v390_acc, v402_lin, v388_data);
          tensorforge::fmacdpp16<8>(v391_acc, v402_lin, v377_data);
          tensorforge::fmacdpp16<9>(v391_acc, v402_lin, v378_data);
          tensorforge::fmacdpp16<10>(v391_acc, v402_lin, v379_data);
          tensorforge::fmacdpp16<11>(v391_acc, v402_lin, v380_data);
          tensorforge::fmacdpp16<12>(v391_acc, v402_lin, v381_data);
          tensorforge::fmacdpp16<13>(v391_acc, v402_lin, v382_data);
          tensorforge::fmacdpp16<14>(v391_acc, v402_lin, v383_data);
          tensorforge::fmacdpp16<15>(v391_acc, v402_lin, v384_data);
          float v403_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v391_acc, v403_lin, v385_data);
          tensorforge::fmacdpp16<1>(v391_acc, v403_lin, v386_data);
          tensorforge::fmacdpp16<2>(v391_acc, v403_lin, v387_data);
          tensorforge::fmacdpp16<3>(v391_acc, v403_lin, v388_data);
          tensorforge::fmacdpp16<4>(v392_acc, v403_lin, v377_data);
          tensorforge::fmacdpp16<5>(v392_acc, v403_lin, v378_data);
          tensorforge::fmacdpp16<6>(v392_acc, v403_lin, v379_data);
          tensorforge::fmacdpp16<7>(v392_acc, v403_lin, v380_data);
          tensorforge::fmacdpp16<8>(v392_acc, v403_lin, v381_data);
          tensorforge::fmacdpp16<9>(v392_acc, v403_lin, v382_data);
          tensorforge::fmacdpp16<10>(v392_acc, v403_lin, v383_data);
          tensorforge::fmacdpp16<11>(v392_acc, v403_lin, v384_data);
          tensorforge::fmacdpp16<12>(v392_acc, v403_lin, v385_data);
          tensorforge::fmacdpp16<13>(v392_acc, v403_lin, v386_data);
          tensorforge::fmacdpp16<14>(v392_acc, v403_lin, v387_data);
          tensorforge::fmacdpp16<15>(v392_acc, v403_lin, v388_data);
          float v404_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v393_acc, v404_lin, v377_data);
          tensorforge::fmacdpp16<1>(v393_acc, v404_lin, v378_data);
          tensorforge::fmacdpp16<2>(v393_acc, v404_lin, v379_data);
          tensorforge::fmacdpp16<3>(v393_acc, v404_lin, v380_data);
          tensorforge::fmacdpp16<4>(v393_acc, v404_lin, v381_data);
          tensorforge::fmacdpp16<5>(v393_acc, v404_lin, v382_data);
          tensorforge::fmacdpp16<6>(v393_acc, v404_lin, v383_data);
          tensorforge::fmacdpp16<7>(v393_acc, v404_lin, v384_data);
          tensorforge::fmacdpp16<8>(v393_acc, v404_lin, v385_data);
          tensorforge::fmacdpp16<9>(v393_acc, v404_lin, v386_data);
          tensorforge::fmacdpp16<10>(v393_acc, v404_lin, v387_data);
          tensorforge::fmacdpp16<11>(v393_acc, v404_lin, v388_data);
          tensorforge::fmacdpp16<12>(v394_acc, v404_lin, v377_data);
          tensorforge::fmacdpp16<13>(v394_acc, v404_lin, v378_data);
          tensorforge::fmacdpp16<14>(v394_acc, v404_lin, v379_data);
          tensorforge::fmacdpp16<15>(v394_acc, v404_lin, v380_data);
          float v405_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v394_acc, v405_lin, v381_data);
          tensorforge::fmacdpp16<1>(v394_acc, v405_lin, v382_data);
          tensorforge::fmacdpp16<2>(v394_acc, v405_lin, v383_data);
          tensorforge::fmacdpp16<3>(v394_acc, v405_lin, v384_data);
          tensorforge::fmacdpp16<4>(v394_acc, v405_lin, v385_data);
          tensorforge::fmacdpp16<5>(v394_acc, v405_lin, v386_data);
          tensorforge::fmacdpp16<6>(v394_acc, v405_lin, v387_data);
          tensorforge::fmacdpp16<7>(v394_acc, v405_lin, v388_data);
          tensorforge::fmacdpp16<8>(v395_acc, v405_lin, v377_data);
          tensorforge::fmacdpp16<9>(v395_acc, v405_lin, v378_data);
          tensorforge::fmacdpp16<10>(v395_acc, v405_lin, v379_data);
          tensorforge::fmacdpp16<11>(v395_acc, v405_lin, v380_data);
          tensorforge::fmacdpp16<12>(v395_acc, v405_lin, v381_data);
          tensorforge::fmacdpp16<13>(v395_acc, v405_lin, v382_data);
          tensorforge::fmacdpp16<14>(v395_acc, v405_lin, v383_data);
          tensorforge::fmacdpp16<15>(v395_acc, v405_lin, v384_data);
          float v406_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v395_acc, v406_lin, v385_data);
          tensorforge::fmacdpp16<1>(v395_acc, v406_lin, v386_data);
          tensorforge::fmacdpp16<2>(v395_acc, v406_lin, v387_data);
          tensorforge::fmacdpp16<3>(v395_acc, v406_lin, v388_data);
          tensorforge::fmacdpp16<4>(v396_acc, v406_lin, v377_data);
          tensorforge::fmacdpp16<5>(v396_acc, v406_lin, v378_data);
          tensorforge::fmacdpp16<6>(v396_acc, v406_lin, v379_data);
          tensorforge::fmacdpp16<7>(v396_acc, v406_lin, v380_data);
          tensorforge::fmacdpp16<8>(v396_acc, v406_lin, v381_data);
          tensorforge::fmacdpp16<9>(v396_acc, v406_lin, v382_data);
          tensorforge::fmacdpp16<10>(v396_acc, v406_lin, v383_data);
          tensorforge::fmacdpp16<11>(v396_acc, v406_lin, v384_data);
          tensorforge::fmacdpp16<12>(v396_acc, v406_lin, v385_data);
          tensorforge::fmacdpp16<13>(v396_acc, v406_lin, v386_data);
          tensorforge::fmacdpp16<14>(v396_acc, v406_lin, v387_data);
          tensorforge::fmacdpp16<15>(v396_acc, v406_lin, v388_data);
          float v407_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v397_acc, v407_lin, v377_data);
          tensorforge::fmacdpp16<1>(v397_acc, v407_lin, v378_data);
          tensorforge::fmacdpp16<2>(v397_acc, v407_lin, v379_data);
          tensorforge::fmacdpp16<3>(v397_acc, v407_lin, v380_data);
          tensorforge::fmacdpp16<4>(v397_acc, v407_lin, v381_data);
          tensorforge::fmacdpp16<5>(v397_acc, v407_lin, v382_data);
          tensorforge::fmacdpp16<6>(v397_acc, v407_lin, v383_data);
          tensorforge::fmacdpp16<7>(v397_acc, v407_lin, v384_data);
          tensorforge::fmacdpp16<8>(v397_acc, v407_lin, v385_data);
          tensorforge::fmacdpp16<9>(v397_acc, v407_lin, v386_data);
          tensorforge::fmacdpp16<10>(v397_acc, v407_lin, v387_data);
          tensorforge::fmacdpp16<11>(v397_acc, v407_lin, v388_data);
          tensorforge::fmacdpp16<12>(v398_acc, v407_lin, v377_data);
          tensorforge::fmacdpp16<13>(v398_acc, v407_lin, v378_data);
          tensorforge::fmacdpp16<14>(v398_acc, v407_lin, v379_data);
          tensorforge::fmacdpp16<15>(v398_acc, v407_lin, v380_data);
          float v408_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v398_acc, v408_lin, v381_data);
          tensorforge::fmacdpp16<1>(v398_acc, v408_lin, v382_data);
          tensorforge::fmacdpp16<2>(v398_acc, v408_lin, v383_data);
          tensorforge::fmacdpp16<3>(v398_acc, v408_lin, v384_data);
          tensorforge::fmacdpp16<4>(v398_acc, v408_lin, v385_data);
          tensorforge::fmacdpp16<5>(v398_acc, v408_lin, v386_data);
          tensorforge::fmacdpp16<6>(v398_acc, v408_lin, v387_data);
          tensorforge::fmacdpp16<7>(v398_acc, v408_lin, v388_data);
          tensorforge::fmacdpp16<8>(v399_acc, v408_lin, v377_data);
          tensorforge::fmacdpp16<9>(v399_acc, v408_lin, v378_data);
          tensorforge::fmacdpp16<10>(v399_acc, v408_lin, v379_data);
          tensorforge::fmacdpp16<11>(v399_acc, v408_lin, v380_data);
          tensorforge::fmacdpp16<12>(v399_acc, v408_lin, v381_data);
          tensorforge::fmacdpp16<13>(v399_acc, v408_lin, v382_data);
          tensorforge::fmacdpp16<14>(v399_acc, v408_lin, v383_data);
          tensorforge::fmacdpp16<15>(v399_acc, v408_lin, v384_data);
          float v409_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v399_acc, v409_lin, v385_data);
          tensorforge::fmacdpp16<1>(v399_acc, v409_lin, v386_data);
          tensorforge::fmacdpp16<2>(v399_acc, v409_lin, v387_data);
          tensorforge::fmacdpp16<3>(v399_acc, v409_lin, v388_data);
          tensorforge::fmacdpp16<4>(v400_acc, v409_lin, v377_data);
          tensorforge::fmacdpp16<5>(v400_acc, v409_lin, v378_data);
          tensorforge::fmacdpp16<6>(v400_acc, v409_lin, v379_data);
          tensorforge::fmacdpp16<7>(v400_acc, v409_lin, v380_data);
          tensorforge::fmacdpp16<8>(v400_acc, v409_lin, v381_data);
          tensorforge::fmacdpp16<9>(v400_acc, v409_lin, v382_data);
          tensorforge::fmacdpp16<10>(v400_acc, v409_lin, v383_data);
          tensorforge::fmacdpp16<11>(v400_acc, v409_lin, v384_data);
          tensorforge::fmacdpp16<12>(v400_acc, v409_lin, v385_data);
          tensorforge::fmacdpp16<13>(v400_acc, v409_lin, v386_data);
          tensorforge::fmacdpp16<14>(v400_acc, v409_lin, v387_data);
          tensorforge::fmacdpp16<15>(v400_acc, v409_lin, v388_data);
          r6[0] = v389_acc;
          r6[1] = v390_acc;
          r6[2] = v391_acc;
          r6[3] = v392_acc;
          r6[4] = v393_acc;
          r6[5] = v394_acc;
          r6[6] = v395_acc;
          r6[7] = v396_acc;
          r6[8] = v397_acc;
          r6[9] = v398_acc;
          r6[10] = v399_acc;
          r6[11] = v400_acc;
          // glb_m3 = store{r>g}(r6);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v414_i1 = 0; v414_i1 < 12; ++v414_i1) {
              float v416_data = r6[v414_i1];
              glb_m3[(v16_lead + (v414_i1 * 12))] = v416_data;
            }
          }
        }
      }
    }
  }
}

