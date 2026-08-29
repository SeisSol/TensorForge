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
          int32_t v12_lead = threadIdx.x % 16;
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 12; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m0[(v12_lead + (v14_i1 * 6))]);
              r0[v14_i1] = v22_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          float v25_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v25_lin;
          float v26_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v26_lin;
          float v27_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v27_lin;
          float v28_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v28_lin;
          float v29_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v29_lin;
          float v30_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v30_lin;
          float v31_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v31_lin;
          float v32_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v32_lin;
          float v33_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v33_lin;
          float v34_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v34_lin;
          float v35_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v35_lin;
          float v36_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v36_lin;
          float v37_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v37_lin;
          float v38_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v38_lin;
          float v39_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v39_lin;
          float v40_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v40_lin;
          float v41_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v41_lin;
          float v42_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v42_lin;
          float v43_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v43_lin;
          float v44_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v44_lin;
          float v45_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v45_lin;
          float v46_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v46_lin;
          float v47_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v47_lin;
          float v48_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v48_lin;
          float v49_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v49_lin;
          float v50_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v50_lin;
          float v51_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v51_lin;
          float v52_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v52_lin;
          float v53_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v53_lin;
          float v54_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v54_lin;
          float v55_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v55_lin;
          float v56_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v56_lin;
          float v57_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v57_lin;
          float v58_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v58_lin;
          float v59_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v59_lin;
          float v60_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v60_lin;
          float v61_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v61_lin;
          float v62_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v62_lin;
          float v63_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v63_lin;
          float v64_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v64_lin;
          float v65_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v65_lin;
          float v66_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v66_lin;
          float v67_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v67_lin;
          float v68_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v68_lin;
          float v69_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v69_lin;
          float v70_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v70_lin;
          float v71_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v71_lin;
          float v72_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v72_lin;
          float v73_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v73_lin;
          float v74_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v74_lin;
          float v75_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v75_lin;
          float v76_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v76_lin;
          float v77_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v77_lin;
          float v78_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v78_lin;
          float v79_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v79_lin;
          float v80_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v80_lin;
          float v81_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v81_lin;
          float v82_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v82_lin;
          float v83_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v83_lin;
          float v84_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v84_lin;
          float v85_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v85_lin;
          float v86_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v86_lin;
          float v87_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v87_lin;
          float v88_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v88_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v94_i1 = 0; v94_i1 < 12; ++v94_i1) {
              float v102_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v94_i1 * 6))]);
              r3[v94_i1] = v102_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v105_data = r1[0];
          float v106_data = r1[1];
          float v107_data = r1[2];
          float v108_data = r1[3];
          float v109_tp{};
          float v110_tp{};
          float v111_tp{};
          float v112_tp{};
          tensorforge::transpose4x4b32(v109_tp, v110_tp, v111_tp, v112_tp, v105_data, v106_data, v107_data, v108_data);
          tensorforge::VectorT<float, 4> v113_acc{};
          float v114_data = r0[0];
          float v115_data = r0[1];
          float v116_data = r0[2];
          float v117_data = r0[3];
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v114_data, v113_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v115_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v116_data, v119_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v117_data, v120_acc, 2, 0, 0);
          float v122_data = r0[4];
          float v123_data = r0[5];
          float v124_data = r0[6];
          float v125_data = r0[7];
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v122_data, v121_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v123_data, v126_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v124_data, v127_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v125_data, v128_acc, 2, 1, 0);
          float v130_data = r0[8];
          float v131_data = r0[9];
          float v132_data = r0[10];
          float v133_data = r0[11];
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v130_data, v129_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v131_data, v134_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v132_data, v135_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v133_data, v136_acc, 2, 2, 0);
          r2[0] = (v137_acc[0]);
          r2[1] = (v137_acc[1]);
          r2[2] = (v137_acc[2]);
          r2[3] = (v137_acc[3]);
          float v142_data = r1[4];
          float v143_data = r1[5];
          float v144_data = r1[6];
          float v145_data = r1[7];
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          float v149_tp{};
          tensorforge::transpose4x4b32(v146_tp, v147_tp, v148_tp, v149_tp, v142_data, v143_data, v144_data, v145_data);
          tensorforge::VectorT<float, 4> v150_acc{};
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v114_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v115_data, v155_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v116_data, v156_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v117_data, v157_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v122_data, v158_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v123_data, v163_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v124_data, v164_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v125_data, v165_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v130_data, v166_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v131_data, v171_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v132_data, v172_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v133_data, v173_acc, 2, 2, 0);
          r2[4] = (v174_acc[0]);
          r2[5] = (v174_acc[1]);
          r2[6] = (v174_acc[2]);
          r2[7] = (v174_acc[3]);
          float v179_data = r1[8];
          float v180_data = r1[9];
          float v181_data = r1[10];
          float v182_data = r1[11];
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          tensorforge::transpose4x4b32(v183_tp, v184_tp, v185_tp, v186_tp, v179_data, v180_data, v181_data, v182_data);
          tensorforge::VectorT<float, 4> v187_acc{};
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v114_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v115_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v116_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v117_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v122_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v123_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v124_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v125_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v130_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v131_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v132_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v133_data, v210_acc, 2, 2, 0);
          r2[8] = (v211_acc[0]);
          r2[9] = (v211_acc[1]);
          r2[10] = (v211_acc[2]);
          r2[11] = (v211_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v221_i1 = 0; v221_i1 < 12; ++v221_i1) {
              float v223_data = r2[v221_i1];
              int32_t v230_a = v12_lead + (v221_i1 * 12);
              s0[(v230_a ^ ((v230_a >> 4) & 15))] = v223_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v239_i1 = 0; v239_i1 < 12; ++v239_i1) {
              float v247_data = __builtin_nontemporal_load(&glb_m4[(v12_lead + (v239_i1 * 12))]);
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
          tensorforge::transpose4x4b32(v254_tp, v255_tp, v256_tp, v257_tp, v105_data, v106_data, v107_data, v108_data);
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
          tensorforge::transpose4x4b32(v291_tp, v292_tp, v293_tp, v294_tp, v142_data, v143_data, v144_data, v145_data);
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
          tensorforge::transpose4x4b32(v328_tp, v329_tp, v330_tp, v331_tp, v179_data, v180_data, v181_data, v182_data);
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
          if (v12_lead < 6) {
            int32_t v373_off = v12_lead + 6;
            #pragma unroll
            for (int32_t v365_i1 = 0; v365_i1 < 12; ++v365_i1) {
              float v367_data = r4[v365_i1];
              int32_t v375_a = v373_off + (v365_i1 * 12);
              s0[(v375_a ^ ((v375_a >> 4) & 15))] = v367_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v380_data = r5[0];
          float v381_data = r5[1];
          float v382_data = r5[2];
          float v383_data = r5[3];
          float v384_data = r5[4];
          float v385_data = r5[5];
          float v386_data = r5[6];
          float v387_data = r5[7];
          float v388_data = r5[8];
          float v389_data = r5[9];
          float v390_data = r5[10];
          float v391_data = r5[11];
          float v392_acc{};
          float v393_acc{};
          float v394_acc{};
          float v395_acc{};
          float v396_acc{};
          float v397_acc{};
          float v398_acc{};
          float v399_acc{};
          float v400_acc{};
          float v401_acc{};
          float v402_acc{};
          float v403_acc{};
          float v407_lin = s0[(0 + threadIdx.x * 1 ^ ((0 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v392_acc, v407_lin, v380_data);
          tensorforge::fmacdpp16<1>(v392_acc, v407_lin, v381_data);
          tensorforge::fmacdpp16<2>(v392_acc, v407_lin, v382_data);
          tensorforge::fmacdpp16<3>(v392_acc, v407_lin, v383_data);
          tensorforge::fmacdpp16<4>(v392_acc, v407_lin, v384_data);
          tensorforge::fmacdpp16<5>(v392_acc, v407_lin, v385_data);
          tensorforge::fmacdpp16<6>(v392_acc, v407_lin, v386_data);
          tensorforge::fmacdpp16<7>(v392_acc, v407_lin, v387_data);
          tensorforge::fmacdpp16<8>(v392_acc, v407_lin, v388_data);
          tensorforge::fmacdpp16<9>(v392_acc, v407_lin, v389_data);
          tensorforge::fmacdpp16<10>(v392_acc, v407_lin, v390_data);
          tensorforge::fmacdpp16<11>(v392_acc, v407_lin, v391_data);
          tensorforge::fmacdpp16<12>(v393_acc, v407_lin, v380_data);
          tensorforge::fmacdpp16<13>(v393_acc, v407_lin, v381_data);
          tensorforge::fmacdpp16<14>(v393_acc, v407_lin, v382_data);
          tensorforge::fmacdpp16<15>(v393_acc, v407_lin, v383_data);
          float v411_lin = s0[(16 + threadIdx.x * 1 ^ ((16 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v393_acc, v411_lin, v384_data);
          tensorforge::fmacdpp16<1>(v393_acc, v411_lin, v385_data);
          tensorforge::fmacdpp16<2>(v393_acc, v411_lin, v386_data);
          tensorforge::fmacdpp16<3>(v393_acc, v411_lin, v387_data);
          tensorforge::fmacdpp16<4>(v393_acc, v411_lin, v388_data);
          tensorforge::fmacdpp16<5>(v393_acc, v411_lin, v389_data);
          tensorforge::fmacdpp16<6>(v393_acc, v411_lin, v390_data);
          tensorforge::fmacdpp16<7>(v393_acc, v411_lin, v391_data);
          tensorforge::fmacdpp16<8>(v394_acc, v411_lin, v380_data);
          tensorforge::fmacdpp16<9>(v394_acc, v411_lin, v381_data);
          tensorforge::fmacdpp16<10>(v394_acc, v411_lin, v382_data);
          tensorforge::fmacdpp16<11>(v394_acc, v411_lin, v383_data);
          tensorforge::fmacdpp16<12>(v394_acc, v411_lin, v384_data);
          tensorforge::fmacdpp16<13>(v394_acc, v411_lin, v385_data);
          tensorforge::fmacdpp16<14>(v394_acc, v411_lin, v386_data);
          tensorforge::fmacdpp16<15>(v394_acc, v411_lin, v387_data);
          float v415_lin = s0[(32 + threadIdx.x * 1 ^ ((32 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v394_acc, v415_lin, v388_data);
          tensorforge::fmacdpp16<1>(v394_acc, v415_lin, v389_data);
          tensorforge::fmacdpp16<2>(v394_acc, v415_lin, v390_data);
          tensorforge::fmacdpp16<3>(v394_acc, v415_lin, v391_data);
          tensorforge::fmacdpp16<4>(v395_acc, v415_lin, v380_data);
          tensorforge::fmacdpp16<5>(v395_acc, v415_lin, v381_data);
          tensorforge::fmacdpp16<6>(v395_acc, v415_lin, v382_data);
          tensorforge::fmacdpp16<7>(v395_acc, v415_lin, v383_data);
          tensorforge::fmacdpp16<8>(v395_acc, v415_lin, v384_data);
          tensorforge::fmacdpp16<9>(v395_acc, v415_lin, v385_data);
          tensorforge::fmacdpp16<10>(v395_acc, v415_lin, v386_data);
          tensorforge::fmacdpp16<11>(v395_acc, v415_lin, v387_data);
          tensorforge::fmacdpp16<12>(v395_acc, v415_lin, v388_data);
          tensorforge::fmacdpp16<13>(v395_acc, v415_lin, v389_data);
          tensorforge::fmacdpp16<14>(v395_acc, v415_lin, v390_data);
          tensorforge::fmacdpp16<15>(v395_acc, v415_lin, v391_data);
          float v419_lin = s0[(48 + threadIdx.x * 1 ^ ((48 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v396_acc, v419_lin, v380_data);
          tensorforge::fmacdpp16<1>(v396_acc, v419_lin, v381_data);
          tensorforge::fmacdpp16<2>(v396_acc, v419_lin, v382_data);
          tensorforge::fmacdpp16<3>(v396_acc, v419_lin, v383_data);
          tensorforge::fmacdpp16<4>(v396_acc, v419_lin, v384_data);
          tensorforge::fmacdpp16<5>(v396_acc, v419_lin, v385_data);
          tensorforge::fmacdpp16<6>(v396_acc, v419_lin, v386_data);
          tensorforge::fmacdpp16<7>(v396_acc, v419_lin, v387_data);
          tensorforge::fmacdpp16<8>(v396_acc, v419_lin, v388_data);
          tensorforge::fmacdpp16<9>(v396_acc, v419_lin, v389_data);
          tensorforge::fmacdpp16<10>(v396_acc, v419_lin, v390_data);
          tensorforge::fmacdpp16<11>(v396_acc, v419_lin, v391_data);
          tensorforge::fmacdpp16<12>(v397_acc, v419_lin, v380_data);
          tensorforge::fmacdpp16<13>(v397_acc, v419_lin, v381_data);
          tensorforge::fmacdpp16<14>(v397_acc, v419_lin, v382_data);
          tensorforge::fmacdpp16<15>(v397_acc, v419_lin, v383_data);
          float v423_lin = s0[(64 + threadIdx.x * 1 ^ ((64 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v397_acc, v423_lin, v384_data);
          tensorforge::fmacdpp16<1>(v397_acc, v423_lin, v385_data);
          tensorforge::fmacdpp16<2>(v397_acc, v423_lin, v386_data);
          tensorforge::fmacdpp16<3>(v397_acc, v423_lin, v387_data);
          tensorforge::fmacdpp16<4>(v397_acc, v423_lin, v388_data);
          tensorforge::fmacdpp16<5>(v397_acc, v423_lin, v389_data);
          tensorforge::fmacdpp16<6>(v397_acc, v423_lin, v390_data);
          tensorforge::fmacdpp16<7>(v397_acc, v423_lin, v391_data);
          tensorforge::fmacdpp16<8>(v398_acc, v423_lin, v380_data);
          tensorforge::fmacdpp16<9>(v398_acc, v423_lin, v381_data);
          tensorforge::fmacdpp16<10>(v398_acc, v423_lin, v382_data);
          tensorforge::fmacdpp16<11>(v398_acc, v423_lin, v383_data);
          tensorforge::fmacdpp16<12>(v398_acc, v423_lin, v384_data);
          tensorforge::fmacdpp16<13>(v398_acc, v423_lin, v385_data);
          tensorforge::fmacdpp16<14>(v398_acc, v423_lin, v386_data);
          tensorforge::fmacdpp16<15>(v398_acc, v423_lin, v387_data);
          float v427_lin = s0[(80 + threadIdx.x * 1 ^ ((80 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v398_acc, v427_lin, v388_data);
          tensorforge::fmacdpp16<1>(v398_acc, v427_lin, v389_data);
          tensorforge::fmacdpp16<2>(v398_acc, v427_lin, v390_data);
          tensorforge::fmacdpp16<3>(v398_acc, v427_lin, v391_data);
          tensorforge::fmacdpp16<4>(v399_acc, v427_lin, v380_data);
          tensorforge::fmacdpp16<5>(v399_acc, v427_lin, v381_data);
          tensorforge::fmacdpp16<6>(v399_acc, v427_lin, v382_data);
          tensorforge::fmacdpp16<7>(v399_acc, v427_lin, v383_data);
          tensorforge::fmacdpp16<8>(v399_acc, v427_lin, v384_data);
          tensorforge::fmacdpp16<9>(v399_acc, v427_lin, v385_data);
          tensorforge::fmacdpp16<10>(v399_acc, v427_lin, v386_data);
          tensorforge::fmacdpp16<11>(v399_acc, v427_lin, v387_data);
          tensorforge::fmacdpp16<12>(v399_acc, v427_lin, v388_data);
          tensorforge::fmacdpp16<13>(v399_acc, v427_lin, v389_data);
          tensorforge::fmacdpp16<14>(v399_acc, v427_lin, v390_data);
          tensorforge::fmacdpp16<15>(v399_acc, v427_lin, v391_data);
          float v431_lin = s0[(96 + threadIdx.x * 1 ^ ((96 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v400_acc, v431_lin, v380_data);
          tensorforge::fmacdpp16<1>(v400_acc, v431_lin, v381_data);
          tensorforge::fmacdpp16<2>(v400_acc, v431_lin, v382_data);
          tensorforge::fmacdpp16<3>(v400_acc, v431_lin, v383_data);
          tensorforge::fmacdpp16<4>(v400_acc, v431_lin, v384_data);
          tensorforge::fmacdpp16<5>(v400_acc, v431_lin, v385_data);
          tensorforge::fmacdpp16<6>(v400_acc, v431_lin, v386_data);
          tensorforge::fmacdpp16<7>(v400_acc, v431_lin, v387_data);
          tensorforge::fmacdpp16<8>(v400_acc, v431_lin, v388_data);
          tensorforge::fmacdpp16<9>(v400_acc, v431_lin, v389_data);
          tensorforge::fmacdpp16<10>(v400_acc, v431_lin, v390_data);
          tensorforge::fmacdpp16<11>(v400_acc, v431_lin, v391_data);
          tensorforge::fmacdpp16<12>(v401_acc, v431_lin, v380_data);
          tensorforge::fmacdpp16<13>(v401_acc, v431_lin, v381_data);
          tensorforge::fmacdpp16<14>(v401_acc, v431_lin, v382_data);
          tensorforge::fmacdpp16<15>(v401_acc, v431_lin, v383_data);
          float v435_lin = s0[(112 + threadIdx.x * 1 ^ ((112 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v401_acc, v435_lin, v384_data);
          tensorforge::fmacdpp16<1>(v401_acc, v435_lin, v385_data);
          tensorforge::fmacdpp16<2>(v401_acc, v435_lin, v386_data);
          tensorforge::fmacdpp16<3>(v401_acc, v435_lin, v387_data);
          tensorforge::fmacdpp16<4>(v401_acc, v435_lin, v388_data);
          tensorforge::fmacdpp16<5>(v401_acc, v435_lin, v389_data);
          tensorforge::fmacdpp16<6>(v401_acc, v435_lin, v390_data);
          tensorforge::fmacdpp16<7>(v401_acc, v435_lin, v391_data);
          tensorforge::fmacdpp16<8>(v402_acc, v435_lin, v380_data);
          tensorforge::fmacdpp16<9>(v402_acc, v435_lin, v381_data);
          tensorforge::fmacdpp16<10>(v402_acc, v435_lin, v382_data);
          tensorforge::fmacdpp16<11>(v402_acc, v435_lin, v383_data);
          tensorforge::fmacdpp16<12>(v402_acc, v435_lin, v384_data);
          tensorforge::fmacdpp16<13>(v402_acc, v435_lin, v385_data);
          tensorforge::fmacdpp16<14>(v402_acc, v435_lin, v386_data);
          tensorforge::fmacdpp16<15>(v402_acc, v435_lin, v387_data);
          float v439_lin = s0[(128 + threadIdx.x * 1 ^ ((128 + threadIdx.x * 1 >> 4) & 15))];
          tensorforge::fmacdpp16<0>(v402_acc, v439_lin, v388_data);
          tensorforge::fmacdpp16<1>(v402_acc, v439_lin, v389_data);
          tensorforge::fmacdpp16<2>(v402_acc, v439_lin, v390_data);
          tensorforge::fmacdpp16<3>(v402_acc, v439_lin, v391_data);
          tensorforge::fmacdpp16<4>(v403_acc, v439_lin, v380_data);
          tensorforge::fmacdpp16<5>(v403_acc, v439_lin, v381_data);
          tensorforge::fmacdpp16<6>(v403_acc, v439_lin, v382_data);
          tensorforge::fmacdpp16<7>(v403_acc, v439_lin, v383_data);
          tensorforge::fmacdpp16<8>(v403_acc, v439_lin, v384_data);
          tensorforge::fmacdpp16<9>(v403_acc, v439_lin, v385_data);
          tensorforge::fmacdpp16<10>(v403_acc, v439_lin, v386_data);
          tensorforge::fmacdpp16<11>(v403_acc, v439_lin, v387_data);
          tensorforge::fmacdpp16<12>(v403_acc, v439_lin, v388_data);
          tensorforge::fmacdpp16<13>(v403_acc, v439_lin, v389_data);
          tensorforge::fmacdpp16<14>(v403_acc, v439_lin, v390_data);
          tensorforge::fmacdpp16<15>(v403_acc, v439_lin, v391_data);
          r6[0] = v392_acc;
          r6[1] = v393_acc;
          r6[2] = v394_acc;
          r6[3] = v395_acc;
          r6[4] = v396_acc;
          r6[5] = v397_acc;
          r6[6] = v398_acc;
          r6[7] = v399_acc;
          r6[8] = v400_acc;
          r6[9] = v401_acc;
          r6[10] = v402_acc;
          r6[11] = v403_acc;
          // glb_m3 = store{r>g}(r6);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v444_i1 = 0; v444_i1 < 12; ++v444_i1) {
              float v446_data = r6[v444_i1];
              glb_m3[(v12_lead + (v444_i1 * 12))] = v446_data;
            }
          }
        }
      }
    }
  }
}

