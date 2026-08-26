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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 6;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m0[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          float v23_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v28_lin;
          float v29_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v29_lin;
          float v30_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v30_lin;
          float v31_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v31_lin;
          float v32_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v32_lin;
          float v33_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v33_lin;
          float v34_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v34_lin;
          float v35_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v35_lin;
          float v36_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v36_lin;
          float v37_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v37_lin;
          float v38_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v38_lin;
          float v39_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v39_lin;
          float v40_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v40_lin;
          float v41_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v41_lin;
          float v42_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v42_lin;
          float v43_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v43_lin;
          float v44_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v44_lin;
          float v45_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v45_lin;
          float v46_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v46_lin;
          float v47_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v47_lin;
          float v48_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v48_lin;
          float v49_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v49_lin;
          float v50_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v50_lin;
          float v51_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v51_lin;
          float v52_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v52_lin;
          float v53_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v53_lin;
          float v54_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v54_lin;
          float v55_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v55_lin;
          float v56_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v56_lin;
          float v57_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v57_lin;
          float v58_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v58_lin;
          float v59_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v59_lin;
          float v60_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v60_lin;
          float v61_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v61_lin;
          float v62_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v62_lin;
          float v63_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v63_lin;
          float v64_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v64_lin;
          float v65_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v65_lin;
          float v66_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v66_lin;
          float v67_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v67_lin;
          float v68_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v68_lin;
          float v69_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v69_lin;
          float v70_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v70_lin;
          float v71_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v71_lin;
          float v72_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v72_lin;
          float v73_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v73_lin;
          float v74_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v74_lin;
          float v75_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v75_lin;
          float v76_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v76_lin;
          float v77_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v77_lin;
          float v78_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v78_lin;
          float v79_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v79_lin;
          float v80_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v80_lin;
          float v81_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v81_lin;
          float v82_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v82_lin;
          float v83_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v83_lin;
          float v84_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v84_lin;
          float v85_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v85_lin;
          float v86_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v86_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v92_i1 = 0; v92_i1 < 12; ++v92_i1) {
              int32_t v98_a = v92_i1 * 6;
              int32_t v99_a = v3_lead + v98_a;
              float v107_data = __builtin_nontemporal_load(&glb_m2[(v3_lead + v98_a)]);
              int32_t v108_a = 0 + v92_i1;
              r3[v108_a] = v107_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir2 = r2;
          float v110_data = r1[0];
          float v111_data = r1[1];
          float v112_data = r1[2];
          float v113_data = r1[3];
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          float v117_tp{};
          tensorforge::transpose4x4b32(v114_tp, v115_tp, v116_tp, v117_tp, v110_data, v111_data, v112_data, v113_data);
          tensorforge::VectorT<float, 4> v118_acc{};
          float v119_data = r0[0];
          float v120_data = r0[1];
          float v121_data = r0[2];
          float v122_data = r0[3];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v120_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v121_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v122_data, v125_acc, 2, 0, 0);
          float v127_data = r0[4];
          float v128_data = r0[5];
          float v129_data = r0[6];
          float v130_data = r0[7];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v127_data, v126_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v128_data, v131_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v129_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v130_data, v133_acc, 2, 1, 0);
          float v135_data = r0[8];
          float v136_data = r0[9];
          float v137_data = r0[10];
          float v138_data = r0[11];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v135_data, v134_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v136_data, v139_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v137_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v138_data, v141_acc, 2, 2, 0);
          ir2[0] = (v142_acc[0]);
          ir2[1] = (v142_acc[1]);
          ir2[2] = (v142_acc[2]);
          ir2[3] = (v142_acc[3]);
          float v147_data = r1[4];
          float v148_data = r1[5];
          float v149_data = r1[6];
          float v150_data = r1[7];
          float v151_tp{};
          float v152_tp{};
          float v153_tp{};
          float v154_tp{};
          tensorforge::transpose4x4b32(v151_tp, v152_tp, v153_tp, v154_tp, v147_data, v148_data, v149_data, v150_data);
          tensorforge::VectorT<float, 4> v155_acc{};
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v119_data, v155_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v120_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v121_data, v161_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v122_data, v162_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v127_data, v163_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v128_data, v168_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v129_data, v169_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v130_data, v170_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v135_data, v171_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v136_data, v176_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v137_data, v177_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v138_data, v178_acc, 2, 2, 0);
          ir2[4] = (v179_acc[0]);
          ir2[5] = (v179_acc[1]);
          ir2[6] = (v179_acc[2]);
          ir2[7] = (v179_acc[3]);
          float v184_data = r1[8];
          float v185_data = r1[9];
          float v186_data = r1[10];
          float v187_data = r1[11];
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          tensorforge::transpose4x4b32(v188_tp, v189_tp, v190_tp, v191_tp, v184_data, v185_data, v186_data, v187_data);
          tensorforge::VectorT<float, 4> v192_acc{};
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v119_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v120_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v121_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v122_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v127_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v128_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v129_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v130_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v135_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v136_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v137_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v138_data, v215_acc, 2, 2, 0);
          ir2[8] = (v216_acc[0]);
          ir2[9] = (v216_acc[1]);
          ir2[10] = (v216_acc[2]);
          ir2[11] = (v216_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v225_i1 = 0; v225_i1 < 12; ++v225_i1) {
              int32_t v226_a = 0 + v225_i1;
              float v228_data = r2[v225_i1];
              int32_t v235_a = v3_lead + (v225_i1 * 12);
              s0[v235_a] = v228_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v241_i1 = 0; v241_i1 < 12; ++v241_i1) {
              int32_t v247_a = v241_i1 * 12;
              int32_t v248_a = v3_lead + v247_a;
              float v256_data = __builtin_nontemporal_load(&glb_m4[(v3_lead + v247_a)]);
              int32_t v257_a = 0 + v241_i1;
              r5[v257_a] = v256_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir4 = r4;
          float v259_data = r1[0];
          float v260_data = r1[1];
          float v261_data = r1[2];
          float v262_data = r1[3];
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          tensorforge::transpose4x4b32(v263_tp, v264_tp, v265_tp, v266_tp, v259_data, v260_data, v261_data, v262_data);
          tensorforge::VectorT<float, 4> v267_acc{};
          float v268_data = r3[0];
          float v269_data = r3[1];
          float v270_data = r3[2];
          float v271_data = r3[3];
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v268_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v269_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v270_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v271_data, v274_acc, 2, 0, 0);
          float v276_data = r3[4];
          float v277_data = r3[5];
          float v278_data = r3[6];
          float v279_data = r3[7];
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v276_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v277_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v278_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v279_data, v282_acc, 2, 1, 0);
          float v284_data = r3[8];
          float v285_data = r3[9];
          float v286_data = r3[10];
          float v287_data = r3[11];
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v284_data, v283_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v285_data, v288_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v286_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v287_data, v290_acc, 2, 2, 0);
          ir4[0] = (v291_acc[0]);
          ir4[1] = (v291_acc[1]);
          ir4[2] = (v291_acc[2]);
          ir4[3] = (v291_acc[3]);
          float v296_data = r1[4];
          float v297_data = r1[5];
          float v298_data = r1[6];
          float v299_data = r1[7];
          float v300_tp{};
          float v301_tp{};
          float v302_tp{};
          float v303_tp{};
          tensorforge::transpose4x4b32(v300_tp, v301_tp, v302_tp, v303_tp, v296_data, v297_data, v298_data, v299_data);
          tensorforge::VectorT<float, 4> v304_acc{};
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v268_data, v304_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v301_tp, v269_data, v309_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v302_tp, v270_data, v310_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v303_tp, v271_data, v311_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v276_data, v312_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v301_tp, v277_data, v317_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v302_tp, v278_data, v318_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v303_tp, v279_data, v319_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v284_data, v320_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v301_tp, v285_data, v325_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v302_tp, v286_data, v326_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v303_tp, v287_data, v327_acc, 2, 2, 0);
          ir4[4] = (v328_acc[0]);
          ir4[5] = (v328_acc[1]);
          ir4[6] = (v328_acc[2]);
          ir4[7] = (v328_acc[3]);
          float v333_data = r1[8];
          float v334_data = r1[9];
          float v335_data = r1[10];
          float v336_data = r1[11];
          float v337_tp{};
          float v338_tp{};
          float v339_tp{};
          float v340_tp{};
          tensorforge::transpose4x4b32(v337_tp, v338_tp, v339_tp, v340_tp, v333_data, v334_data, v335_data, v336_data);
          tensorforge::VectorT<float, 4> v341_acc{};
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v268_data, v341_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v269_data, v346_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v270_data, v347_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v271_data, v348_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v276_data, v349_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v277_data, v354_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v278_data, v355_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v279_data, v356_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v284_data, v357_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v285_data, v362_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v286_data, v363_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v287_data, v364_acc, 2, 2, 0);
          ir4[8] = (v365_acc[0]);
          ir4[9] = (v365_acc[1]);
          ir4[10] = (v365_acc[2]);
          ir4[11] = (v365_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v3_lead < 6) {
            int32_t v383_off = v3_lead + 6;
            #pragma unroll
            for (int32_t v374_i1 = 0; v374_i1 < 12; ++v374_i1) {
              int32_t v375_a = 0 + v374_i1;
              float v377_data = r4[v374_i1];
              int32_t v385_a = v383_off + (v374_i1 * 12);
              s0[v385_a] = v377_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          ;
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          auto& ir6 = r6;
          float v387_data = r5[0];
          float v388_data = r5[1];
          float v389_data = r5[2];
          float v390_data = r5[3];
          float v391_data = r5[4];
          float v392_data = r5[5];
          float v393_data = r5[6];
          float v394_data = r5[7];
          float v395_data = r5[8];
          float v396_data = r5[9];
          float v397_data = r5[10];
          float v398_data = r5[11];
          float v399_acc{};
          float v400_acc{};
          float v401_acc{};
          float v402_acc{};
          float v403_acc{};
          float v404_acc{};
          float v405_acc{};
          float v406_acc{};
          float v407_acc{};
          float v408_acc{};
          float v409_acc{};
          float v410_acc{};
          float v411_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v399_acc, v411_lin, v387_data);
          tensorforge::fmacdpp16<1>(v399_acc, v411_lin, v388_data);
          tensorforge::fmacdpp16<2>(v399_acc, v411_lin, v389_data);
          tensorforge::fmacdpp16<3>(v399_acc, v411_lin, v390_data);
          tensorforge::fmacdpp16<4>(v399_acc, v411_lin, v391_data);
          tensorforge::fmacdpp16<5>(v399_acc, v411_lin, v392_data);
          tensorforge::fmacdpp16<6>(v399_acc, v411_lin, v393_data);
          tensorforge::fmacdpp16<7>(v399_acc, v411_lin, v394_data);
          tensorforge::fmacdpp16<8>(v399_acc, v411_lin, v395_data);
          tensorforge::fmacdpp16<9>(v399_acc, v411_lin, v396_data);
          tensorforge::fmacdpp16<10>(v399_acc, v411_lin, v397_data);
          tensorforge::fmacdpp16<11>(v399_acc, v411_lin, v398_data);
          tensorforge::fmacdpp16<12>(v400_acc, v411_lin, v387_data);
          tensorforge::fmacdpp16<13>(v400_acc, v411_lin, v388_data);
          tensorforge::fmacdpp16<14>(v400_acc, v411_lin, v389_data);
          tensorforge::fmacdpp16<15>(v400_acc, v411_lin, v390_data);
          float v412_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v400_acc, v412_lin, v391_data);
          tensorforge::fmacdpp16<1>(v400_acc, v412_lin, v392_data);
          tensorforge::fmacdpp16<2>(v400_acc, v412_lin, v393_data);
          tensorforge::fmacdpp16<3>(v400_acc, v412_lin, v394_data);
          tensorforge::fmacdpp16<4>(v400_acc, v412_lin, v395_data);
          tensorforge::fmacdpp16<5>(v400_acc, v412_lin, v396_data);
          tensorforge::fmacdpp16<6>(v400_acc, v412_lin, v397_data);
          tensorforge::fmacdpp16<7>(v400_acc, v412_lin, v398_data);
          tensorforge::fmacdpp16<8>(v401_acc, v412_lin, v387_data);
          tensorforge::fmacdpp16<9>(v401_acc, v412_lin, v388_data);
          tensorforge::fmacdpp16<10>(v401_acc, v412_lin, v389_data);
          tensorforge::fmacdpp16<11>(v401_acc, v412_lin, v390_data);
          tensorforge::fmacdpp16<12>(v401_acc, v412_lin, v391_data);
          tensorforge::fmacdpp16<13>(v401_acc, v412_lin, v392_data);
          tensorforge::fmacdpp16<14>(v401_acc, v412_lin, v393_data);
          tensorforge::fmacdpp16<15>(v401_acc, v412_lin, v394_data);
          float v413_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v401_acc, v413_lin, v395_data);
          tensorforge::fmacdpp16<1>(v401_acc, v413_lin, v396_data);
          tensorforge::fmacdpp16<2>(v401_acc, v413_lin, v397_data);
          tensorforge::fmacdpp16<3>(v401_acc, v413_lin, v398_data);
          tensorforge::fmacdpp16<4>(v402_acc, v413_lin, v387_data);
          tensorforge::fmacdpp16<5>(v402_acc, v413_lin, v388_data);
          tensorforge::fmacdpp16<6>(v402_acc, v413_lin, v389_data);
          tensorforge::fmacdpp16<7>(v402_acc, v413_lin, v390_data);
          tensorforge::fmacdpp16<8>(v402_acc, v413_lin, v391_data);
          tensorforge::fmacdpp16<9>(v402_acc, v413_lin, v392_data);
          tensorforge::fmacdpp16<10>(v402_acc, v413_lin, v393_data);
          tensorforge::fmacdpp16<11>(v402_acc, v413_lin, v394_data);
          tensorforge::fmacdpp16<12>(v402_acc, v413_lin, v395_data);
          tensorforge::fmacdpp16<13>(v402_acc, v413_lin, v396_data);
          tensorforge::fmacdpp16<14>(v402_acc, v413_lin, v397_data);
          tensorforge::fmacdpp16<15>(v402_acc, v413_lin, v398_data);
          float v414_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v403_acc, v414_lin, v387_data);
          tensorforge::fmacdpp16<1>(v403_acc, v414_lin, v388_data);
          tensorforge::fmacdpp16<2>(v403_acc, v414_lin, v389_data);
          tensorforge::fmacdpp16<3>(v403_acc, v414_lin, v390_data);
          tensorforge::fmacdpp16<4>(v403_acc, v414_lin, v391_data);
          tensorforge::fmacdpp16<5>(v403_acc, v414_lin, v392_data);
          tensorforge::fmacdpp16<6>(v403_acc, v414_lin, v393_data);
          tensorforge::fmacdpp16<7>(v403_acc, v414_lin, v394_data);
          tensorforge::fmacdpp16<8>(v403_acc, v414_lin, v395_data);
          tensorforge::fmacdpp16<9>(v403_acc, v414_lin, v396_data);
          tensorforge::fmacdpp16<10>(v403_acc, v414_lin, v397_data);
          tensorforge::fmacdpp16<11>(v403_acc, v414_lin, v398_data);
          tensorforge::fmacdpp16<12>(v404_acc, v414_lin, v387_data);
          tensorforge::fmacdpp16<13>(v404_acc, v414_lin, v388_data);
          tensorforge::fmacdpp16<14>(v404_acc, v414_lin, v389_data);
          tensorforge::fmacdpp16<15>(v404_acc, v414_lin, v390_data);
          float v415_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v404_acc, v415_lin, v391_data);
          tensorforge::fmacdpp16<1>(v404_acc, v415_lin, v392_data);
          tensorforge::fmacdpp16<2>(v404_acc, v415_lin, v393_data);
          tensorforge::fmacdpp16<3>(v404_acc, v415_lin, v394_data);
          tensorforge::fmacdpp16<4>(v404_acc, v415_lin, v395_data);
          tensorforge::fmacdpp16<5>(v404_acc, v415_lin, v396_data);
          tensorforge::fmacdpp16<6>(v404_acc, v415_lin, v397_data);
          tensorforge::fmacdpp16<7>(v404_acc, v415_lin, v398_data);
          tensorforge::fmacdpp16<8>(v405_acc, v415_lin, v387_data);
          tensorforge::fmacdpp16<9>(v405_acc, v415_lin, v388_data);
          tensorforge::fmacdpp16<10>(v405_acc, v415_lin, v389_data);
          tensorforge::fmacdpp16<11>(v405_acc, v415_lin, v390_data);
          tensorforge::fmacdpp16<12>(v405_acc, v415_lin, v391_data);
          tensorforge::fmacdpp16<13>(v405_acc, v415_lin, v392_data);
          tensorforge::fmacdpp16<14>(v405_acc, v415_lin, v393_data);
          tensorforge::fmacdpp16<15>(v405_acc, v415_lin, v394_data);
          float v416_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v405_acc, v416_lin, v395_data);
          tensorforge::fmacdpp16<1>(v405_acc, v416_lin, v396_data);
          tensorforge::fmacdpp16<2>(v405_acc, v416_lin, v397_data);
          tensorforge::fmacdpp16<3>(v405_acc, v416_lin, v398_data);
          tensorforge::fmacdpp16<4>(v406_acc, v416_lin, v387_data);
          tensorforge::fmacdpp16<5>(v406_acc, v416_lin, v388_data);
          tensorforge::fmacdpp16<6>(v406_acc, v416_lin, v389_data);
          tensorforge::fmacdpp16<7>(v406_acc, v416_lin, v390_data);
          tensorforge::fmacdpp16<8>(v406_acc, v416_lin, v391_data);
          tensorforge::fmacdpp16<9>(v406_acc, v416_lin, v392_data);
          tensorforge::fmacdpp16<10>(v406_acc, v416_lin, v393_data);
          tensorforge::fmacdpp16<11>(v406_acc, v416_lin, v394_data);
          tensorforge::fmacdpp16<12>(v406_acc, v416_lin, v395_data);
          tensorforge::fmacdpp16<13>(v406_acc, v416_lin, v396_data);
          tensorforge::fmacdpp16<14>(v406_acc, v416_lin, v397_data);
          tensorforge::fmacdpp16<15>(v406_acc, v416_lin, v398_data);
          float v417_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v407_acc, v417_lin, v387_data);
          tensorforge::fmacdpp16<1>(v407_acc, v417_lin, v388_data);
          tensorforge::fmacdpp16<2>(v407_acc, v417_lin, v389_data);
          tensorforge::fmacdpp16<3>(v407_acc, v417_lin, v390_data);
          tensorforge::fmacdpp16<4>(v407_acc, v417_lin, v391_data);
          tensorforge::fmacdpp16<5>(v407_acc, v417_lin, v392_data);
          tensorforge::fmacdpp16<6>(v407_acc, v417_lin, v393_data);
          tensorforge::fmacdpp16<7>(v407_acc, v417_lin, v394_data);
          tensorforge::fmacdpp16<8>(v407_acc, v417_lin, v395_data);
          tensorforge::fmacdpp16<9>(v407_acc, v417_lin, v396_data);
          tensorforge::fmacdpp16<10>(v407_acc, v417_lin, v397_data);
          tensorforge::fmacdpp16<11>(v407_acc, v417_lin, v398_data);
          tensorforge::fmacdpp16<12>(v408_acc, v417_lin, v387_data);
          tensorforge::fmacdpp16<13>(v408_acc, v417_lin, v388_data);
          tensorforge::fmacdpp16<14>(v408_acc, v417_lin, v389_data);
          tensorforge::fmacdpp16<15>(v408_acc, v417_lin, v390_data);
          float v418_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v408_acc, v418_lin, v391_data);
          tensorforge::fmacdpp16<1>(v408_acc, v418_lin, v392_data);
          tensorforge::fmacdpp16<2>(v408_acc, v418_lin, v393_data);
          tensorforge::fmacdpp16<3>(v408_acc, v418_lin, v394_data);
          tensorforge::fmacdpp16<4>(v408_acc, v418_lin, v395_data);
          tensorforge::fmacdpp16<5>(v408_acc, v418_lin, v396_data);
          tensorforge::fmacdpp16<6>(v408_acc, v418_lin, v397_data);
          tensorforge::fmacdpp16<7>(v408_acc, v418_lin, v398_data);
          tensorforge::fmacdpp16<8>(v409_acc, v418_lin, v387_data);
          tensorforge::fmacdpp16<9>(v409_acc, v418_lin, v388_data);
          tensorforge::fmacdpp16<10>(v409_acc, v418_lin, v389_data);
          tensorforge::fmacdpp16<11>(v409_acc, v418_lin, v390_data);
          tensorforge::fmacdpp16<12>(v409_acc, v418_lin, v391_data);
          tensorforge::fmacdpp16<13>(v409_acc, v418_lin, v392_data);
          tensorforge::fmacdpp16<14>(v409_acc, v418_lin, v393_data);
          tensorforge::fmacdpp16<15>(v409_acc, v418_lin, v394_data);
          float v419_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v409_acc, v419_lin, v395_data);
          tensorforge::fmacdpp16<1>(v409_acc, v419_lin, v396_data);
          tensorforge::fmacdpp16<2>(v409_acc, v419_lin, v397_data);
          tensorforge::fmacdpp16<3>(v409_acc, v419_lin, v398_data);
          tensorforge::fmacdpp16<4>(v410_acc, v419_lin, v387_data);
          tensorforge::fmacdpp16<5>(v410_acc, v419_lin, v388_data);
          tensorforge::fmacdpp16<6>(v410_acc, v419_lin, v389_data);
          tensorforge::fmacdpp16<7>(v410_acc, v419_lin, v390_data);
          tensorforge::fmacdpp16<8>(v410_acc, v419_lin, v391_data);
          tensorforge::fmacdpp16<9>(v410_acc, v419_lin, v392_data);
          tensorforge::fmacdpp16<10>(v410_acc, v419_lin, v393_data);
          tensorforge::fmacdpp16<11>(v410_acc, v419_lin, v394_data);
          tensorforge::fmacdpp16<12>(v410_acc, v419_lin, v395_data);
          tensorforge::fmacdpp16<13>(v410_acc, v419_lin, v396_data);
          tensorforge::fmacdpp16<14>(v410_acc, v419_lin, v397_data);
          tensorforge::fmacdpp16<15>(v410_acc, v419_lin, v398_data);
          ir6[0] = v399_acc;
          ir6[1] = v400_acc;
          ir6[2] = v401_acc;
          ir6[3] = v402_acc;
          ir6[4] = v403_acc;
          ir6[5] = v404_acc;
          ir6[6] = v405_acc;
          ir6[7] = v406_acc;
          ir6[8] = v407_acc;
          ir6[9] = v408_acc;
          ir6[10] = v409_acc;
          ir6[11] = v410_acc;
          // glb_m3 = store{r>g}(r6);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v424_i1 = 0; v424_i1 < 12; ++v424_i1) {
              int32_t v425_a = 0 + v424_i1;
              float v427_data = r6[v424_i1];
              int32_t v434_a = v3_lead + (v424_i1 * 12);
              glb_m3[v434_a] = v427_data;
            }
          }
          ;
        }
      }
    }
  }
}

