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
              int32_t v20_a = v14_i1 * 6;
              int32_t v21_a = v12_lead + v20_a;
              float v29_data = __builtin_nontemporal_load(&glb_m0[(v12_lead + v20_a)]);
              r0[v14_i1] = v29_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          float v32_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v32_lin;
          float v33_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v33_lin;
          float v34_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v34_lin;
          float v35_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v35_lin;
          float v36_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v36_lin;
          float v37_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v37_lin;
          float v38_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v38_lin;
          float v39_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v39_lin;
          float v40_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v40_lin;
          float v41_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v41_lin;
          float v42_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v42_lin;
          float v43_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v43_lin;
          float v44_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v44_lin;
          float v45_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v45_lin;
          float v46_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v46_lin;
          float v47_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v47_lin;
          float v48_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v48_lin;
          float v49_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v49_lin;
          float v50_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v50_lin;
          float v51_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v51_lin;
          float v52_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v52_lin;
          float v53_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v53_lin;
          float v54_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v54_lin;
          float v55_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v55_lin;
          float v56_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v56_lin;
          float v57_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v57_lin;
          float v58_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v58_lin;
          float v59_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v59_lin;
          float v60_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v60_lin;
          float v61_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v61_lin;
          float v62_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v62_lin;
          float v63_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v63_lin;
          float v64_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v64_lin;
          float v65_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v65_lin;
          float v66_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v66_lin;
          float v67_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v67_lin;
          float v68_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v68_lin;
          float v69_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v69_lin;
          float v70_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v70_lin;
          float v71_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v71_lin;
          float v72_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v72_lin;
          float v73_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v73_lin;
          float v74_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v74_lin;
          float v75_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v75_lin;
          float v76_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v76_lin;
          float v77_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v77_lin;
          float v78_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v78_lin;
          float v79_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v79_lin;
          float v80_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v80_lin;
          float v81_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v81_lin;
          float v82_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v82_lin;
          float v83_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v83_lin;
          float v84_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v84_lin;
          float v85_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v85_lin;
          float v86_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v86_lin;
          float v87_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v87_lin;
          float v88_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v88_lin;
          float v89_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v89_lin;
          float v90_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v90_lin;
          float v91_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v91_lin;
          float v92_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v92_lin;
          float v93_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v93_lin;
          float v94_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v94_lin;
          float v95_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v95_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 12; ++v101_i1) {
              int32_t v107_a = v101_i1 * 6;
              int32_t v108_a = v12_lead + v107_a;
              float v116_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + v107_a)]);
              r3[v101_i1] = v116_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v119_data = r1[0];
          float v120_data = r1[1];
          float v121_data = r1[2];
          float v122_data = r1[3];
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          tensorforge::transpose4x4b32(v123_tp, v124_tp, v125_tp, v126_tp, v119_data, v120_data, v121_data, v122_data);
          tensorforge::VectorT<float, 4> v127_acc{};
          float v128_data = r0[0];
          float v129_data = r0[1];
          float v130_data = r0[2];
          float v131_data = r0[3];
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v128_data, v127_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v129_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v130_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v131_data, v134_acc, 2, 0, 0);
          float v136_data = r0[4];
          float v137_data = r0[5];
          float v138_data = r0[6];
          float v139_data = r0[7];
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v136_data, v135_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v137_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v138_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v139_data, v142_acc, 2, 1, 0);
          float v144_data = r0[8];
          float v145_data = r0[9];
          float v146_data = r0[10];
          float v147_data = r0[11];
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v144_data, v143_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v145_data, v148_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v146_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v147_data, v150_acc, 2, 2, 0);
          r2[0] = (v151_acc[0]);
          r2[1] = (v151_acc[1]);
          r2[2] = (v151_acc[2]);
          r2[3] = (v151_acc[3]);
          float v156_data = r1[4];
          float v157_data = r1[5];
          float v158_data = r1[6];
          float v159_data = r1[7];
          float v160_tp{};
          float v161_tp{};
          float v162_tp{};
          float v163_tp{};
          tensorforge::transpose4x4b32(v160_tp, v161_tp, v162_tp, v163_tp, v156_data, v157_data, v158_data, v159_data);
          tensorforge::VectorT<float, 4> v164_acc{};
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v128_data, v164_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v129_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v130_data, v170_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v131_data, v171_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v136_data, v172_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v137_data, v177_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v138_data, v178_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v139_data, v179_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v144_data, v180_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v145_data, v185_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v146_data, v186_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v147_data, v187_acc, 2, 2, 0);
          r2[4] = (v188_acc[0]);
          r2[5] = (v188_acc[1]);
          r2[6] = (v188_acc[2]);
          r2[7] = (v188_acc[3]);
          float v193_data = r1[8];
          float v194_data = r1[9];
          float v195_data = r1[10];
          float v196_data = r1[11];
          float v197_tp{};
          float v198_tp{};
          float v199_tp{};
          float v200_tp{};
          tensorforge::transpose4x4b32(v197_tp, v198_tp, v199_tp, v200_tp, v193_data, v194_data, v195_data, v196_data);
          tensorforge::VectorT<float, 4> v201_acc{};
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v128_data, v201_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v129_data, v206_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v130_data, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v131_data, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v136_data, v209_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v137_data, v214_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v138_data, v215_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v139_data, v216_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v144_data, v217_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v145_data, v222_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v146_data, v223_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v147_data, v224_acc, 2, 2, 0);
          r2[8] = (v225_acc[0]);
          r2[9] = (v225_acc[1]);
          r2[10] = (v225_acc[2]);
          r2[11] = (v225_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v235_i1 = 0; v235_i1 < 12; ++v235_i1) {
              int32_t v236_a = 0 + v235_i1;
              float v238_data = r2[v235_i1];
              s0[(v12_lead + (v235_i1 * 12))] = v238_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v251_i1 = 0; v251_i1 < 12; ++v251_i1) {
              int32_t v257_a = v251_i1 * 12;
              int32_t v258_a = v12_lead + v257_a;
              float v266_data = __builtin_nontemporal_load(&glb_m4[(v12_lead + v257_a)]);
              r5[v251_i1] = v266_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          float v276_tp{};
          tensorforge::transpose4x4b32(v273_tp, v274_tp, v275_tp, v276_tp, v119_data, v120_data, v121_data, v122_data);
          tensorforge::VectorT<float, 4> v277_acc{};
          float v278_data = r3[0];
          float v279_data = r3[1];
          float v280_data = r3[2];
          float v281_data = r3[3];
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v278_data, v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v279_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v280_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v281_data, v284_acc, 2, 0, 0);
          float v286_data = r3[4];
          float v287_data = r3[5];
          float v288_data = r3[6];
          float v289_data = r3[7];
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v286_data, v285_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v287_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v288_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v289_data, v292_acc, 2, 1, 0);
          float v294_data = r3[8];
          float v295_data = r3[9];
          float v296_data = r3[10];
          float v297_data = r3[11];
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v294_data, v293_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v295_data, v298_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v296_data, v299_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v297_data, v300_acc, 2, 2, 0);
          r4[0] = (v301_acc[0]);
          r4[1] = (v301_acc[1]);
          r4[2] = (v301_acc[2]);
          r4[3] = (v301_acc[3]);
          float v310_tp{};
          float v311_tp{};
          float v312_tp{};
          float v313_tp{};
          tensorforge::transpose4x4b32(v310_tp, v311_tp, v312_tp, v313_tp, v156_data, v157_data, v158_data, v159_data);
          tensorforge::VectorT<float, 4> v314_acc{};
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v278_data, v314_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v279_data, v319_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v280_data, v320_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v281_data, v321_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v286_data, v322_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v287_data, v327_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v288_data, v328_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v289_data, v329_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v294_data, v330_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v295_data, v335_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v296_data, v336_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v297_data, v337_acc, 2, 2, 0);
          r4[4] = (v338_acc[0]);
          r4[5] = (v338_acc[1]);
          r4[6] = (v338_acc[2]);
          r4[7] = (v338_acc[3]);
          float v347_tp{};
          float v348_tp{};
          float v349_tp{};
          float v350_tp{};
          tensorforge::transpose4x4b32(v347_tp, v348_tp, v349_tp, v350_tp, v193_data, v194_data, v195_data, v196_data);
          tensorforge::VectorT<float, 4> v351_acc{};
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v278_data, v351_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v279_data, v356_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v280_data, v357_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v281_data, v358_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v286_data, v359_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v287_data, v364_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v288_data, v365_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v289_data, v366_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v294_data, v367_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v295_data, v372_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v296_data, v373_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v297_data, v374_acc, 2, 2, 0);
          r4[8] = (v375_acc[0]);
          r4[9] = (v375_acc[1]);
          r4[10] = (v375_acc[2]);
          r4[11] = (v375_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v12_lead < 6) {
            int32_t v393_off = v12_lead + 6;
            #pragma unroll
            for (int32_t v384_i1 = 0; v384_i1 < 12; ++v384_i1) {
              int32_t v385_a = 0 + v384_i1;
              float v387_data = r4[v384_i1];
              s0[(v393_off + (v384_i1 * 12))] = v387_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v397_data = r5[0];
          float v398_data = r5[1];
          float v399_data = r5[2];
          float v400_data = r5[3];
          float v401_data = r5[4];
          float v402_data = r5[5];
          float v403_data = r5[6];
          float v404_data = r5[7];
          float v405_data = r5[8];
          float v406_data = r5[9];
          float v407_data = r5[10];
          float v408_data = r5[11];
          float v409_acc{};
          float v410_acc{};
          float v411_acc{};
          float v412_acc{};
          float v413_acc{};
          float v414_acc{};
          float v415_acc{};
          float v416_acc{};
          float v417_acc{};
          float v418_acc{};
          float v419_acc{};
          float v420_acc{};
          float v421_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v409_acc, v421_lin, v397_data);
          tensorforge::fmacdpp16<1>(v409_acc, v421_lin, v398_data);
          tensorforge::fmacdpp16<2>(v409_acc, v421_lin, v399_data);
          tensorforge::fmacdpp16<3>(v409_acc, v421_lin, v400_data);
          tensorforge::fmacdpp16<4>(v409_acc, v421_lin, v401_data);
          tensorforge::fmacdpp16<5>(v409_acc, v421_lin, v402_data);
          tensorforge::fmacdpp16<6>(v409_acc, v421_lin, v403_data);
          tensorforge::fmacdpp16<7>(v409_acc, v421_lin, v404_data);
          tensorforge::fmacdpp16<8>(v409_acc, v421_lin, v405_data);
          tensorforge::fmacdpp16<9>(v409_acc, v421_lin, v406_data);
          tensorforge::fmacdpp16<10>(v409_acc, v421_lin, v407_data);
          tensorforge::fmacdpp16<11>(v409_acc, v421_lin, v408_data);
          tensorforge::fmacdpp16<12>(v410_acc, v421_lin, v397_data);
          tensorforge::fmacdpp16<13>(v410_acc, v421_lin, v398_data);
          tensorforge::fmacdpp16<14>(v410_acc, v421_lin, v399_data);
          tensorforge::fmacdpp16<15>(v410_acc, v421_lin, v400_data);
          float v422_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v410_acc, v422_lin, v401_data);
          tensorforge::fmacdpp16<1>(v410_acc, v422_lin, v402_data);
          tensorforge::fmacdpp16<2>(v410_acc, v422_lin, v403_data);
          tensorforge::fmacdpp16<3>(v410_acc, v422_lin, v404_data);
          tensorforge::fmacdpp16<4>(v410_acc, v422_lin, v405_data);
          tensorforge::fmacdpp16<5>(v410_acc, v422_lin, v406_data);
          tensorforge::fmacdpp16<6>(v410_acc, v422_lin, v407_data);
          tensorforge::fmacdpp16<7>(v410_acc, v422_lin, v408_data);
          tensorforge::fmacdpp16<8>(v411_acc, v422_lin, v397_data);
          tensorforge::fmacdpp16<9>(v411_acc, v422_lin, v398_data);
          tensorforge::fmacdpp16<10>(v411_acc, v422_lin, v399_data);
          tensorforge::fmacdpp16<11>(v411_acc, v422_lin, v400_data);
          tensorforge::fmacdpp16<12>(v411_acc, v422_lin, v401_data);
          tensorforge::fmacdpp16<13>(v411_acc, v422_lin, v402_data);
          tensorforge::fmacdpp16<14>(v411_acc, v422_lin, v403_data);
          tensorforge::fmacdpp16<15>(v411_acc, v422_lin, v404_data);
          float v423_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v411_acc, v423_lin, v405_data);
          tensorforge::fmacdpp16<1>(v411_acc, v423_lin, v406_data);
          tensorforge::fmacdpp16<2>(v411_acc, v423_lin, v407_data);
          tensorforge::fmacdpp16<3>(v411_acc, v423_lin, v408_data);
          tensorforge::fmacdpp16<4>(v412_acc, v423_lin, v397_data);
          tensorforge::fmacdpp16<5>(v412_acc, v423_lin, v398_data);
          tensorforge::fmacdpp16<6>(v412_acc, v423_lin, v399_data);
          tensorforge::fmacdpp16<7>(v412_acc, v423_lin, v400_data);
          tensorforge::fmacdpp16<8>(v412_acc, v423_lin, v401_data);
          tensorforge::fmacdpp16<9>(v412_acc, v423_lin, v402_data);
          tensorforge::fmacdpp16<10>(v412_acc, v423_lin, v403_data);
          tensorforge::fmacdpp16<11>(v412_acc, v423_lin, v404_data);
          tensorforge::fmacdpp16<12>(v412_acc, v423_lin, v405_data);
          tensorforge::fmacdpp16<13>(v412_acc, v423_lin, v406_data);
          tensorforge::fmacdpp16<14>(v412_acc, v423_lin, v407_data);
          tensorforge::fmacdpp16<15>(v412_acc, v423_lin, v408_data);
          float v424_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v413_acc, v424_lin, v397_data);
          tensorforge::fmacdpp16<1>(v413_acc, v424_lin, v398_data);
          tensorforge::fmacdpp16<2>(v413_acc, v424_lin, v399_data);
          tensorforge::fmacdpp16<3>(v413_acc, v424_lin, v400_data);
          tensorforge::fmacdpp16<4>(v413_acc, v424_lin, v401_data);
          tensorforge::fmacdpp16<5>(v413_acc, v424_lin, v402_data);
          tensorforge::fmacdpp16<6>(v413_acc, v424_lin, v403_data);
          tensorforge::fmacdpp16<7>(v413_acc, v424_lin, v404_data);
          tensorforge::fmacdpp16<8>(v413_acc, v424_lin, v405_data);
          tensorforge::fmacdpp16<9>(v413_acc, v424_lin, v406_data);
          tensorforge::fmacdpp16<10>(v413_acc, v424_lin, v407_data);
          tensorforge::fmacdpp16<11>(v413_acc, v424_lin, v408_data);
          tensorforge::fmacdpp16<12>(v414_acc, v424_lin, v397_data);
          tensorforge::fmacdpp16<13>(v414_acc, v424_lin, v398_data);
          tensorforge::fmacdpp16<14>(v414_acc, v424_lin, v399_data);
          tensorforge::fmacdpp16<15>(v414_acc, v424_lin, v400_data);
          float v425_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v414_acc, v425_lin, v401_data);
          tensorforge::fmacdpp16<1>(v414_acc, v425_lin, v402_data);
          tensorforge::fmacdpp16<2>(v414_acc, v425_lin, v403_data);
          tensorforge::fmacdpp16<3>(v414_acc, v425_lin, v404_data);
          tensorforge::fmacdpp16<4>(v414_acc, v425_lin, v405_data);
          tensorforge::fmacdpp16<5>(v414_acc, v425_lin, v406_data);
          tensorforge::fmacdpp16<6>(v414_acc, v425_lin, v407_data);
          tensorforge::fmacdpp16<7>(v414_acc, v425_lin, v408_data);
          tensorforge::fmacdpp16<8>(v415_acc, v425_lin, v397_data);
          tensorforge::fmacdpp16<9>(v415_acc, v425_lin, v398_data);
          tensorforge::fmacdpp16<10>(v415_acc, v425_lin, v399_data);
          tensorforge::fmacdpp16<11>(v415_acc, v425_lin, v400_data);
          tensorforge::fmacdpp16<12>(v415_acc, v425_lin, v401_data);
          tensorforge::fmacdpp16<13>(v415_acc, v425_lin, v402_data);
          tensorforge::fmacdpp16<14>(v415_acc, v425_lin, v403_data);
          tensorforge::fmacdpp16<15>(v415_acc, v425_lin, v404_data);
          float v426_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v415_acc, v426_lin, v405_data);
          tensorforge::fmacdpp16<1>(v415_acc, v426_lin, v406_data);
          tensorforge::fmacdpp16<2>(v415_acc, v426_lin, v407_data);
          tensorforge::fmacdpp16<3>(v415_acc, v426_lin, v408_data);
          tensorforge::fmacdpp16<4>(v416_acc, v426_lin, v397_data);
          tensorforge::fmacdpp16<5>(v416_acc, v426_lin, v398_data);
          tensorforge::fmacdpp16<6>(v416_acc, v426_lin, v399_data);
          tensorforge::fmacdpp16<7>(v416_acc, v426_lin, v400_data);
          tensorforge::fmacdpp16<8>(v416_acc, v426_lin, v401_data);
          tensorforge::fmacdpp16<9>(v416_acc, v426_lin, v402_data);
          tensorforge::fmacdpp16<10>(v416_acc, v426_lin, v403_data);
          tensorforge::fmacdpp16<11>(v416_acc, v426_lin, v404_data);
          tensorforge::fmacdpp16<12>(v416_acc, v426_lin, v405_data);
          tensorforge::fmacdpp16<13>(v416_acc, v426_lin, v406_data);
          tensorforge::fmacdpp16<14>(v416_acc, v426_lin, v407_data);
          tensorforge::fmacdpp16<15>(v416_acc, v426_lin, v408_data);
          float v427_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v417_acc, v427_lin, v397_data);
          tensorforge::fmacdpp16<1>(v417_acc, v427_lin, v398_data);
          tensorforge::fmacdpp16<2>(v417_acc, v427_lin, v399_data);
          tensorforge::fmacdpp16<3>(v417_acc, v427_lin, v400_data);
          tensorforge::fmacdpp16<4>(v417_acc, v427_lin, v401_data);
          tensorforge::fmacdpp16<5>(v417_acc, v427_lin, v402_data);
          tensorforge::fmacdpp16<6>(v417_acc, v427_lin, v403_data);
          tensorforge::fmacdpp16<7>(v417_acc, v427_lin, v404_data);
          tensorforge::fmacdpp16<8>(v417_acc, v427_lin, v405_data);
          tensorforge::fmacdpp16<9>(v417_acc, v427_lin, v406_data);
          tensorforge::fmacdpp16<10>(v417_acc, v427_lin, v407_data);
          tensorforge::fmacdpp16<11>(v417_acc, v427_lin, v408_data);
          tensorforge::fmacdpp16<12>(v418_acc, v427_lin, v397_data);
          tensorforge::fmacdpp16<13>(v418_acc, v427_lin, v398_data);
          tensorforge::fmacdpp16<14>(v418_acc, v427_lin, v399_data);
          tensorforge::fmacdpp16<15>(v418_acc, v427_lin, v400_data);
          float v428_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v418_acc, v428_lin, v401_data);
          tensorforge::fmacdpp16<1>(v418_acc, v428_lin, v402_data);
          tensorforge::fmacdpp16<2>(v418_acc, v428_lin, v403_data);
          tensorforge::fmacdpp16<3>(v418_acc, v428_lin, v404_data);
          tensorforge::fmacdpp16<4>(v418_acc, v428_lin, v405_data);
          tensorforge::fmacdpp16<5>(v418_acc, v428_lin, v406_data);
          tensorforge::fmacdpp16<6>(v418_acc, v428_lin, v407_data);
          tensorforge::fmacdpp16<7>(v418_acc, v428_lin, v408_data);
          tensorforge::fmacdpp16<8>(v419_acc, v428_lin, v397_data);
          tensorforge::fmacdpp16<9>(v419_acc, v428_lin, v398_data);
          tensorforge::fmacdpp16<10>(v419_acc, v428_lin, v399_data);
          tensorforge::fmacdpp16<11>(v419_acc, v428_lin, v400_data);
          tensorforge::fmacdpp16<12>(v419_acc, v428_lin, v401_data);
          tensorforge::fmacdpp16<13>(v419_acc, v428_lin, v402_data);
          tensorforge::fmacdpp16<14>(v419_acc, v428_lin, v403_data);
          tensorforge::fmacdpp16<15>(v419_acc, v428_lin, v404_data);
          float v429_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v419_acc, v429_lin, v405_data);
          tensorforge::fmacdpp16<1>(v419_acc, v429_lin, v406_data);
          tensorforge::fmacdpp16<2>(v419_acc, v429_lin, v407_data);
          tensorforge::fmacdpp16<3>(v419_acc, v429_lin, v408_data);
          tensorforge::fmacdpp16<4>(v420_acc, v429_lin, v397_data);
          tensorforge::fmacdpp16<5>(v420_acc, v429_lin, v398_data);
          tensorforge::fmacdpp16<6>(v420_acc, v429_lin, v399_data);
          tensorforge::fmacdpp16<7>(v420_acc, v429_lin, v400_data);
          tensorforge::fmacdpp16<8>(v420_acc, v429_lin, v401_data);
          tensorforge::fmacdpp16<9>(v420_acc, v429_lin, v402_data);
          tensorforge::fmacdpp16<10>(v420_acc, v429_lin, v403_data);
          tensorforge::fmacdpp16<11>(v420_acc, v429_lin, v404_data);
          tensorforge::fmacdpp16<12>(v420_acc, v429_lin, v405_data);
          tensorforge::fmacdpp16<13>(v420_acc, v429_lin, v406_data);
          tensorforge::fmacdpp16<14>(v420_acc, v429_lin, v407_data);
          tensorforge::fmacdpp16<15>(v420_acc, v429_lin, v408_data);
          r6[0] = v409_acc;
          r6[1] = v410_acc;
          r6[2] = v411_acc;
          r6[3] = v412_acc;
          r6[4] = v413_acc;
          r6[5] = v414_acc;
          r6[6] = v415_acc;
          r6[7] = v416_acc;
          r6[8] = v417_acc;
          r6[9] = v418_acc;
          r6[10] = v419_acc;
          r6[11] = v420_acc;
          // glb_m3 = store{r>g}(r6);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v434_i1 = 0; v434_i1 < 12; ++v434_i1) {
              int32_t v435_a = 0 + v434_i1;
              float v437_data = r6[v434_i1];
              glb_m3[(v12_lead + (v434_i1 * 12))] = v437_data;
            }
          }
        }
      }
    }
  }
}

