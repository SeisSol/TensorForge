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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 16;
          if (v15_lead < 6) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 12; ++v17_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v15_lead + (v17_i1 * 6))]);
              r0[v17_i1] = v25_data;
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
          if (v15_lead < 6) {
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 12; ++v97_i1) {
              float v105_data = __builtin_nontemporal_load(&glb_m2[(v15_lead + (v97_i1 * 6))]);
              r3[v97_i1] = v105_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v108_data = r1[0];
          float v109_data = r1[1];
          float v110_data = r1[2];
          float v111_data = r1[3];
          float v112_tp{};
          float v113_tp{};
          float v114_tp{};
          float v115_tp{};
          tensorforge::transpose4x4b32(v112_tp, v113_tp, v114_tp, v115_tp, v108_data, v109_data, v110_data, v111_data);
          tensorforge::VectorT<float, 4> v116_acc{};
          float v117_data = r0[0];
          float v118_data = r0[1];
          float v119_data = r0[2];
          float v120_data = r0[3];
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v117_data, v116_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v118_data, v121_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v122_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v120_data, v123_acc, 2, 0, 0);
          float v125_data = r0[4];
          float v126_data = r0[5];
          float v127_data = r0[6];
          float v128_data = r0[7];
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v125_data, v124_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v126_data, v129_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v127_data, v130_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v128_data, v131_acc, 2, 1, 0);
          float v133_data = r0[8];
          float v134_data = r0[9];
          float v135_data = r0[10];
          float v136_data = r0[11];
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v133_data, v132_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v134_data, v137_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v135_data, v138_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v136_data, v139_acc, 2, 2, 0);
          r2[0] = (v140_acc[0]);
          r2[1] = (v140_acc[1]);
          r2[2] = (v140_acc[2]);
          r2[3] = (v140_acc[3]);
          float v145_data = r1[4];
          float v146_data = r1[5];
          float v147_data = r1[6];
          float v148_data = r1[7];
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          tensorforge::transpose4x4b32(v149_tp, v150_tp, v151_tp, v152_tp, v145_data, v146_data, v147_data, v148_data);
          tensorforge::VectorT<float, 4> v153_acc{};
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v117_data, v153_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v118_data, v158_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v119_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v120_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v125_data, v161_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v126_data, v166_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v127_data, v167_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v128_data, v168_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v133_data, v169_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v134_data, v174_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v135_data, v175_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v136_data, v176_acc, 2, 2, 0);
          r2[4] = (v177_acc[0]);
          r2[5] = (v177_acc[1]);
          r2[6] = (v177_acc[2]);
          r2[7] = (v177_acc[3]);
          float v182_data = r1[8];
          float v183_data = r1[9];
          float v184_data = r1[10];
          float v185_data = r1[11];
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          tensorforge::transpose4x4b32(v186_tp, v187_tp, v188_tp, v189_tp, v182_data, v183_data, v184_data, v185_data);
          tensorforge::VectorT<float, 4> v190_acc{};
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v117_data, v190_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v118_data, v195_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v119_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v120_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v125_data, v198_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v126_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v127_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v128_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v133_data, v206_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v134_data, v211_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v135_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v136_data, v213_acc, 2, 2, 0);
          r2[8] = (v214_acc[0]);
          r2[9] = (v214_acc[1]);
          r2[10] = (v214_acc[2]);
          r2[11] = (v214_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v15_lead < 6) {
            #pragma unroll
            for (int32_t v224_i1 = 0; v224_i1 < 12; ++v224_i1) {
              float v226_data = r2[v224_i1];
              int32_t v233_a = v15_lead + (v224_i1 * 12);
              s0[(v233_a ^ ((v233_a >> 4) & 15))] = v226_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v15_lead < 12) {
            #pragma unroll
            for (int32_t v242_i1 = 0; v242_i1 < 12; ++v242_i1) {
              float v250_data = __builtin_nontemporal_load(&glb_m4[(v15_lead + (v242_i1 * 12))]);
              r5[v242_i1] = v250_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v257_tp{};
          float v258_tp{};
          float v259_tp{};
          float v260_tp{};
          tensorforge::transpose4x4b32(v257_tp, v258_tp, v259_tp, v260_tp, v108_data, v109_data, v110_data, v111_data);
          tensorforge::VectorT<float, 4> v261_acc{};
          float v262_data = r3[0];
          float v263_data = r3[1];
          float v264_data = r3[2];
          float v265_data = r3[3];
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v262_data, v261_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v263_data, v266_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v264_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v265_data, v268_acc, 2, 0, 0);
          float v270_data = r3[4];
          float v271_data = r3[5];
          float v272_data = r3[6];
          float v273_data = r3[7];
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v270_data, v269_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v271_data, v274_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v272_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v273_data, v276_acc, 2, 1, 0);
          float v278_data = r3[8];
          float v279_data = r3[9];
          float v280_data = r3[10];
          float v281_data = r3[11];
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v278_data, v277_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v279_data, v282_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v280_data, v283_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v281_data, v284_acc, 2, 2, 0);
          r4[0] = (v285_acc[0]);
          r4[1] = (v285_acc[1]);
          r4[2] = (v285_acc[2]);
          r4[3] = (v285_acc[3]);
          float v294_tp{};
          float v295_tp{};
          float v296_tp{};
          float v297_tp{};
          tensorforge::transpose4x4b32(v294_tp, v295_tp, v296_tp, v297_tp, v145_data, v146_data, v147_data, v148_data);
          tensorforge::VectorT<float, 4> v298_acc{};
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v262_data, v298_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v263_data, v303_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v264_data, v304_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v265_data, v305_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v270_data, v306_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v271_data, v311_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v272_data, v312_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v273_data, v313_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v278_data, v314_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v279_data, v319_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v280_data, v320_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v281_data, v321_acc, 2, 2, 0);
          r4[4] = (v322_acc[0]);
          r4[5] = (v322_acc[1]);
          r4[6] = (v322_acc[2]);
          r4[7] = (v322_acc[3]);
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          float v334_tp{};
          tensorforge::transpose4x4b32(v331_tp, v332_tp, v333_tp, v334_tp, v182_data, v183_data, v184_data, v185_data);
          tensorforge::VectorT<float, 4> v335_acc{};
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v262_data, v335_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v263_data, v340_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v264_data, v341_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v265_data, v342_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v270_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v271_data, v348_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v272_data, v349_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v273_data, v350_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v278_data, v351_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v279_data, v356_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v280_data, v357_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v281_data, v358_acc, 2, 2, 0);
          r4[8] = (v359_acc[0]);
          r4[9] = (v359_acc[1]);
          r4[10] = (v359_acc[2]);
          r4[11] = (v359_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v15_lead < 6) {
            int32_t v376_off = v15_lead + 6;
            #pragma unroll
            for (int32_t v368_i1 = 0; v368_i1 < 12; ++v368_i1) {
              float v370_data = r4[v368_i1];
              int32_t v378_a = v376_off + (v368_i1 * 12);
              s0[(v378_a ^ ((v378_a >> 4) & 15))] = v370_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v383_data = r5[0];
          float v384_data = r5[1];
          float v385_data = r5[2];
          float v386_data = r5[3];
          float v387_data = r5[4];
          float v388_data = r5[5];
          float v389_data = r5[6];
          float v390_data = r5[7];
          float v391_data = r5[8];
          float v392_data = r5[9];
          float v393_data = r5[10];
          float v394_data = r5[11];
          float v395_acc{};
          float v396_acc{};
          float v397_acc{};
          float v398_acc{};
          float v399_acc{};
          float v400_acc{};
          float v401_acc{};
          float v402_acc{};
          float v403_acc{};
          float v404_acc{};
          float v405_acc{};
          float v406_acc{};
          float v410_lin = s0[(0 + threadIdx.x * 1 ^ ((0 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v395_acc, v410_lin, v383_data);
          tensorforge::fmacdpp16<1>(v395_acc, v410_lin, v384_data);
          tensorforge::fmacdpp16<2>(v395_acc, v410_lin, v385_data);
          tensorforge::fmacdpp16<3>(v395_acc, v410_lin, v386_data);
          tensorforge::fmacdpp16<4>(v395_acc, v410_lin, v387_data);
          tensorforge::fmacdpp16<5>(v395_acc, v410_lin, v388_data);
          tensorforge::fmacdpp16<6>(v395_acc, v410_lin, v389_data);
          tensorforge::fmacdpp16<7>(v395_acc, v410_lin, v390_data);
          tensorforge::fmacdpp16<8>(v395_acc, v410_lin, v391_data);
          tensorforge::fmacdpp16<9>(v395_acc, v410_lin, v392_data);
          tensorforge::fmacdpp16<10>(v395_acc, v410_lin, v393_data);
          tensorforge::fmacdpp16<11>(v395_acc, v410_lin, v394_data);
          tensorforge::fmacdpp16<12>(v396_acc, v410_lin, v383_data);
          tensorforge::fmacdpp16<13>(v396_acc, v410_lin, v384_data);
          tensorforge::fmacdpp16<14>(v396_acc, v410_lin, v385_data);
          tensorforge::fmacdpp16<15>(v396_acc, v410_lin, v386_data);
          float v414_lin = s0[(16 + threadIdx.x * 1 ^ ((16 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v396_acc, v414_lin, v387_data);
          tensorforge::fmacdpp16<1>(v396_acc, v414_lin, v388_data);
          tensorforge::fmacdpp16<2>(v396_acc, v414_lin, v389_data);
          tensorforge::fmacdpp16<3>(v396_acc, v414_lin, v390_data);
          tensorforge::fmacdpp16<4>(v396_acc, v414_lin, v391_data);
          tensorforge::fmacdpp16<5>(v396_acc, v414_lin, v392_data);
          tensorforge::fmacdpp16<6>(v396_acc, v414_lin, v393_data);
          tensorforge::fmacdpp16<7>(v396_acc, v414_lin, v394_data);
          tensorforge::fmacdpp16<8>(v397_acc, v414_lin, v383_data);
          tensorforge::fmacdpp16<9>(v397_acc, v414_lin, v384_data);
          tensorforge::fmacdpp16<10>(v397_acc, v414_lin, v385_data);
          tensorforge::fmacdpp16<11>(v397_acc, v414_lin, v386_data);
          tensorforge::fmacdpp16<12>(v397_acc, v414_lin, v387_data);
          tensorforge::fmacdpp16<13>(v397_acc, v414_lin, v388_data);
          tensorforge::fmacdpp16<14>(v397_acc, v414_lin, v389_data);
          tensorforge::fmacdpp16<15>(v397_acc, v414_lin, v390_data);
          float v418_lin = s0[(32 + threadIdx.x * 1 ^ ((32 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v397_acc, v418_lin, v391_data);
          tensorforge::fmacdpp16<1>(v397_acc, v418_lin, v392_data);
          tensorforge::fmacdpp16<2>(v397_acc, v418_lin, v393_data);
          tensorforge::fmacdpp16<3>(v397_acc, v418_lin, v394_data);
          tensorforge::fmacdpp16<4>(v398_acc, v418_lin, v383_data);
          tensorforge::fmacdpp16<5>(v398_acc, v418_lin, v384_data);
          tensorforge::fmacdpp16<6>(v398_acc, v418_lin, v385_data);
          tensorforge::fmacdpp16<7>(v398_acc, v418_lin, v386_data);
          tensorforge::fmacdpp16<8>(v398_acc, v418_lin, v387_data);
          tensorforge::fmacdpp16<9>(v398_acc, v418_lin, v388_data);
          tensorforge::fmacdpp16<10>(v398_acc, v418_lin, v389_data);
          tensorforge::fmacdpp16<11>(v398_acc, v418_lin, v390_data);
          tensorforge::fmacdpp16<12>(v398_acc, v418_lin, v391_data);
          tensorforge::fmacdpp16<13>(v398_acc, v418_lin, v392_data);
          tensorforge::fmacdpp16<14>(v398_acc, v418_lin, v393_data);
          tensorforge::fmacdpp16<15>(v398_acc, v418_lin, v394_data);
          float v422_lin = s0[(48 + threadIdx.x * 1 ^ ((48 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v399_acc, v422_lin, v383_data);
          tensorforge::fmacdpp16<1>(v399_acc, v422_lin, v384_data);
          tensorforge::fmacdpp16<2>(v399_acc, v422_lin, v385_data);
          tensorforge::fmacdpp16<3>(v399_acc, v422_lin, v386_data);
          tensorforge::fmacdpp16<4>(v399_acc, v422_lin, v387_data);
          tensorforge::fmacdpp16<5>(v399_acc, v422_lin, v388_data);
          tensorforge::fmacdpp16<6>(v399_acc, v422_lin, v389_data);
          tensorforge::fmacdpp16<7>(v399_acc, v422_lin, v390_data);
          tensorforge::fmacdpp16<8>(v399_acc, v422_lin, v391_data);
          tensorforge::fmacdpp16<9>(v399_acc, v422_lin, v392_data);
          tensorforge::fmacdpp16<10>(v399_acc, v422_lin, v393_data);
          tensorforge::fmacdpp16<11>(v399_acc, v422_lin, v394_data);
          tensorforge::fmacdpp16<12>(v400_acc, v422_lin, v383_data);
          tensorforge::fmacdpp16<13>(v400_acc, v422_lin, v384_data);
          tensorforge::fmacdpp16<14>(v400_acc, v422_lin, v385_data);
          tensorforge::fmacdpp16<15>(v400_acc, v422_lin, v386_data);
          float v426_lin = s0[(64 + threadIdx.x * 1 ^ ((64 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v400_acc, v426_lin, v387_data);
          tensorforge::fmacdpp16<1>(v400_acc, v426_lin, v388_data);
          tensorforge::fmacdpp16<2>(v400_acc, v426_lin, v389_data);
          tensorforge::fmacdpp16<3>(v400_acc, v426_lin, v390_data);
          tensorforge::fmacdpp16<4>(v400_acc, v426_lin, v391_data);
          tensorforge::fmacdpp16<5>(v400_acc, v426_lin, v392_data);
          tensorforge::fmacdpp16<6>(v400_acc, v426_lin, v393_data);
          tensorforge::fmacdpp16<7>(v400_acc, v426_lin, v394_data);
          tensorforge::fmacdpp16<8>(v401_acc, v426_lin, v383_data);
          tensorforge::fmacdpp16<9>(v401_acc, v426_lin, v384_data);
          tensorforge::fmacdpp16<10>(v401_acc, v426_lin, v385_data);
          tensorforge::fmacdpp16<11>(v401_acc, v426_lin, v386_data);
          tensorforge::fmacdpp16<12>(v401_acc, v426_lin, v387_data);
          tensorforge::fmacdpp16<13>(v401_acc, v426_lin, v388_data);
          tensorforge::fmacdpp16<14>(v401_acc, v426_lin, v389_data);
          tensorforge::fmacdpp16<15>(v401_acc, v426_lin, v390_data);
          float v430_lin = s0[(80 + threadIdx.x * 1 ^ ((80 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v401_acc, v430_lin, v391_data);
          tensorforge::fmacdpp16<1>(v401_acc, v430_lin, v392_data);
          tensorforge::fmacdpp16<2>(v401_acc, v430_lin, v393_data);
          tensorforge::fmacdpp16<3>(v401_acc, v430_lin, v394_data);
          tensorforge::fmacdpp16<4>(v402_acc, v430_lin, v383_data);
          tensorforge::fmacdpp16<5>(v402_acc, v430_lin, v384_data);
          tensorforge::fmacdpp16<6>(v402_acc, v430_lin, v385_data);
          tensorforge::fmacdpp16<7>(v402_acc, v430_lin, v386_data);
          tensorforge::fmacdpp16<8>(v402_acc, v430_lin, v387_data);
          tensorforge::fmacdpp16<9>(v402_acc, v430_lin, v388_data);
          tensorforge::fmacdpp16<10>(v402_acc, v430_lin, v389_data);
          tensorforge::fmacdpp16<11>(v402_acc, v430_lin, v390_data);
          tensorforge::fmacdpp16<12>(v402_acc, v430_lin, v391_data);
          tensorforge::fmacdpp16<13>(v402_acc, v430_lin, v392_data);
          tensorforge::fmacdpp16<14>(v402_acc, v430_lin, v393_data);
          tensorforge::fmacdpp16<15>(v402_acc, v430_lin, v394_data);
          float v434_lin = s0[(96 + threadIdx.x * 1 ^ ((96 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v403_acc, v434_lin, v383_data);
          tensorforge::fmacdpp16<1>(v403_acc, v434_lin, v384_data);
          tensorforge::fmacdpp16<2>(v403_acc, v434_lin, v385_data);
          tensorforge::fmacdpp16<3>(v403_acc, v434_lin, v386_data);
          tensorforge::fmacdpp16<4>(v403_acc, v434_lin, v387_data);
          tensorforge::fmacdpp16<5>(v403_acc, v434_lin, v388_data);
          tensorforge::fmacdpp16<6>(v403_acc, v434_lin, v389_data);
          tensorforge::fmacdpp16<7>(v403_acc, v434_lin, v390_data);
          tensorforge::fmacdpp16<8>(v403_acc, v434_lin, v391_data);
          tensorforge::fmacdpp16<9>(v403_acc, v434_lin, v392_data);
          tensorforge::fmacdpp16<10>(v403_acc, v434_lin, v393_data);
          tensorforge::fmacdpp16<11>(v403_acc, v434_lin, v394_data);
          tensorforge::fmacdpp16<12>(v404_acc, v434_lin, v383_data);
          tensorforge::fmacdpp16<13>(v404_acc, v434_lin, v384_data);
          tensorforge::fmacdpp16<14>(v404_acc, v434_lin, v385_data);
          tensorforge::fmacdpp16<15>(v404_acc, v434_lin, v386_data);
          float v438_lin = s0[(112 + threadIdx.x * 1 ^ ((112 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v404_acc, v438_lin, v387_data);
          tensorforge::fmacdpp16<1>(v404_acc, v438_lin, v388_data);
          tensorforge::fmacdpp16<2>(v404_acc, v438_lin, v389_data);
          tensorforge::fmacdpp16<3>(v404_acc, v438_lin, v390_data);
          tensorforge::fmacdpp16<4>(v404_acc, v438_lin, v391_data);
          tensorforge::fmacdpp16<5>(v404_acc, v438_lin, v392_data);
          tensorforge::fmacdpp16<6>(v404_acc, v438_lin, v393_data);
          tensorforge::fmacdpp16<7>(v404_acc, v438_lin, v394_data);
          tensorforge::fmacdpp16<8>(v405_acc, v438_lin, v383_data);
          tensorforge::fmacdpp16<9>(v405_acc, v438_lin, v384_data);
          tensorforge::fmacdpp16<10>(v405_acc, v438_lin, v385_data);
          tensorforge::fmacdpp16<11>(v405_acc, v438_lin, v386_data);
          tensorforge::fmacdpp16<12>(v405_acc, v438_lin, v387_data);
          tensorforge::fmacdpp16<13>(v405_acc, v438_lin, v388_data);
          tensorforge::fmacdpp16<14>(v405_acc, v438_lin, v389_data);
          tensorforge::fmacdpp16<15>(v405_acc, v438_lin, v390_data);
          float v442_lin = s0[(128 + threadIdx.x * 1 ^ ((128 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v405_acc, v442_lin, v391_data);
          tensorforge::fmacdpp16<1>(v405_acc, v442_lin, v392_data);
          tensorforge::fmacdpp16<2>(v405_acc, v442_lin, v393_data);
          tensorforge::fmacdpp16<3>(v405_acc, v442_lin, v394_data);
          tensorforge::fmacdpp16<4>(v406_acc, v442_lin, v383_data);
          tensorforge::fmacdpp16<5>(v406_acc, v442_lin, v384_data);
          tensorforge::fmacdpp16<6>(v406_acc, v442_lin, v385_data);
          tensorforge::fmacdpp16<7>(v406_acc, v442_lin, v386_data);
          tensorforge::fmacdpp16<8>(v406_acc, v442_lin, v387_data);
          tensorforge::fmacdpp16<9>(v406_acc, v442_lin, v388_data);
          tensorforge::fmacdpp16<10>(v406_acc, v442_lin, v389_data);
          tensorforge::fmacdpp16<11>(v406_acc, v442_lin, v390_data);
          tensorforge::fmacdpp16<12>(v406_acc, v442_lin, v391_data);
          tensorforge::fmacdpp16<13>(v406_acc, v442_lin, v392_data);
          tensorforge::fmacdpp16<14>(v406_acc, v442_lin, v393_data);
          tensorforge::fmacdpp16<15>(v406_acc, v442_lin, v394_data);
          r6[0] = v395_acc;
          r6[1] = v396_acc;
          r6[2] = v397_acc;
          r6[3] = v398_acc;
          r6[4] = v399_acc;
          r6[5] = v400_acc;
          r6[6] = v401_acc;
          r6[7] = v402_acc;
          r6[8] = v403_acc;
          r6[9] = v404_acc;
          r6[10] = v405_acc;
          r6[11] = v406_acc;
          // glb_m3 = store{r>g}(r6);
          if (v15_lead < 12) {
            #pragma unroll
            for (int32_t v447_i1 = 0; v447_i1 < 12; ++v447_i1) {
              float v449_data = r6[v447_i1];
              glb_m3[(v15_lead + (v447_i1 * 12))] = v449_data;
            }
          }
        }
      }
    }
  }
}

