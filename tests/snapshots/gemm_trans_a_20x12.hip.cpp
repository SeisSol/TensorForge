// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f94e030d8c, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_f94e030d8c), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_f94e030d8c, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×16(12×16) {0..12}×{0..16} strided
    // m1 20×12(20×12) {0..20}×{0..12} strided
    // m2 20×16(20×16) {0..20}×{0..16} strided
    // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v4_lead = threadIdx.x % 16;
          bool v5_g = v4_lead < 12;
          #pragma unroll
          for (int32_t v1_i0 = 0; v1_i0 < 20; ++v1_i0) {
            if (v5_g) {
              int32_t v12_a = v1_i0 + (v4_lead * 20);
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v1_i0 + (v4_lead * 20))]);
              int32_t v21_a = v1_i0 + 0;
              r0[v21_a] = v20_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          float v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v28_lin;
          float v29_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v29_lin;
          float v30_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v30_lin;
          float v31_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v31_lin;
          float v32_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v32_lin;
          float v33_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v33_lin;
          float v34_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v34_lin;
          float v35_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v35_lin;
          float v36_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v36_lin;
          float v37_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v37_lin;
          float v38_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v38_lin;
          float v39_lin = glb_m2[256 + threadIdx.x * 1];
          r1[16] = v39_lin;
          float v40_lin = glb_m2[272 + threadIdx.x * 1];
          r1[17] = v40_lin;
          float v41_lin = glb_m2[288 + threadIdx.x * 1];
          r1[18] = v41_lin;
          float v42_lin = glb_m2[304 + threadIdx.x * 1];
          r1[19] = v42_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          auto& ir2 = r2;
          float v44_data = r1[0];
          float v45_data = r1[2];
          float v46_data = r1[4];
          float v47_data = r1[6];
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          tensorforge::transpose4x4b32(v48_tp, v49_tp, v50_tp, v51_tp, v44_data, v45_data, v46_data, v47_data);
          float v52_data = r1[1];
          float v53_data = r1[3];
          float v54_data = r1[5];
          float v55_data = r1[7];
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          tensorforge::transpose4x4b32(v56_tp, v57_tp, v58_tp, v59_tp, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 4> v60_acc{};
          float v61_data = r0[0];
          float v62_data = r0[1];
          float v63_data = r0[2];
          float v64_data = r0[3];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 0, 0);
          float v69_data = r0[4];
          float v70_data = r0[5];
          float v71_data = r0[6];
          float v72_data = r0[7];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 1, 0);
          float v77_data = r0[8];
          float v78_data = r0[9];
          float v79_data = r0[10];
          float v80_data = r0[11];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v83_acc, 2, 2, 0);
          float v85_data = r0[12];
          float v86_data = r0[13];
          float v87_data = r0[14];
          float v88_data = r0[15];
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v85_data, v84_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v86_data, v89_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v87_data, v90_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v88_data, v91_acc, 2, 3, 0);
          float v93_data = r0[16];
          float v94_data = r0[17];
          float v95_data = r0[18];
          float v96_data = r0[19];
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v94_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v96_data, v99_acc, 2, 0, 0);
          ir2[0] = (v100_acc[0]);
          ir2[1] = (v100_acc[1]);
          ir2[2] = (v100_acc[2]);
          ir2[3] = (v100_acc[3]);
          float v105_data = r1[8];
          float v106_data = r1[10];
          float v107_data = r1[12];
          float v108_data = r1[14];
          float v109_tp{};
          float v110_tp{};
          float v111_tp{};
          float v112_tp{};
          tensorforge::transpose4x4b32(v109_tp, v110_tp, v111_tp, v112_tp, v105_data, v106_data, v107_data, v108_data);
          float v113_data = r1[9];
          float v114_data = r1[11];
          float v115_data = r1[13];
          float v116_data = r1[15];
          float v117_tp{};
          float v118_tp{};
          float v119_tp{};
          float v120_tp{};
          tensorforge::transpose4x4b32(v117_tp, v118_tp, v119_tp, v120_tp, v113_data, v114_data, v115_data, v116_data);
          tensorforge::VectorT<float, 4> v121_acc{};
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v61_data, v121_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v62_data, v126_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v63_data, v127_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v64_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v69_data, v129_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v70_data, v134_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v71_data, v135_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v72_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v77_data, v137_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v78_data, v142_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v79_data, v143_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v80_data, v144_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v85_data, v145_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v86_data, v150_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v87_data, v151_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v88_data, v152_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v93_data, v153_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v94_data, v158_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v95_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v96_data, v160_acc, 2, 0, 0);
          ir2[4] = (v161_acc[0]);
          ir2[5] = (v161_acc[1]);
          ir2[6] = (v161_acc[2]);
          ir2[7] = (v161_acc[3]);
          float v166_data = r1[16];
          float v167_data = r1[18];
          float v168_data = r1[20];
          float v169_data = r1[22];
          float v170_tp{};
          float v171_tp{};
          float v172_tp{};
          float v173_tp{};
          tensorforge::transpose4x4b32(v170_tp, v171_tp, v172_tp, v173_tp, v166_data, v167_data, v168_data, v169_data);
          float v174_data = r1[17];
          float v175_data = r1[19];
          float v176_data = r1[21];
          float v177_data = r1[23];
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          tensorforge::transpose4x4b32(v178_tp, v179_tp, v180_tp, v181_tp, v174_data, v175_data, v176_data, v177_data);
          tensorforge::VectorT<float, 4> v182_acc{};
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v61_data, v182_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v62_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v63_data, v188_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v64_data, v189_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v69_data, v190_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v70_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v71_data, v196_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v72_data, v197_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v77_data, v198_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v78_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v79_data, v204_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v80_data, v205_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v85_data, v206_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v86_data, v211_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v87_data, v212_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v88_data, v213_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v93_data, v214_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v94_data, v219_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v95_data, v220_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v96_data, v221_acc, 2, 0, 0);
          ir2[8] = (v222_acc[0]);
          ir2[9] = (v222_acc[1]);
          ir2[10] = (v222_acc[2]);
          ir2[11] = (v222_acc[3]);
          float v227_data = r1[24];
          float v228_data = r1[26];
          float v229_data = r1[28];
          float v230_data = r1[30];
          float v231_tp{};
          float v232_tp{};
          float v233_tp{};
          float v234_tp{};
          tensorforge::transpose4x4b32(v231_tp, v232_tp, v233_tp, v234_tp, v227_data, v228_data, v229_data, v230_data);
          float v235_data = r1[25];
          float v236_data = r1[27];
          float v237_data = r1[29];
          float v238_data = r1[31];
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          float v242_tp{};
          tensorforge::transpose4x4b32(v239_tp, v240_tp, v241_tp, v242_tp, v235_data, v236_data, v237_data, v238_data);
          tensorforge::VectorT<float, 4> v243_acc{};
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v61_data, v243_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v62_data, v248_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v63_data, v249_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v64_data, v250_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v69_data, v251_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v70_data, v256_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v71_data, v257_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v72_data, v258_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v77_data, v259_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v78_data, v264_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v79_data, v265_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v80_data, v266_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v85_data, v267_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v86_data, v272_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v87_data, v273_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v88_data, v274_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v93_data, v275_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v94_data, v280_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v95_data, v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v96_data, v282_acc, 2, 0, 0);
          ir2[12] = (v283_acc[0]);
          ir2[13] = (v283_acc[1]);
          ir2[14] = (v283_acc[2]);
          ir2[15] = (v283_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v290_lead = threadIdx.x % 16;
          if (v290_lead < 12) {
            #pragma unroll
            for (int32_t v292_i1 = 0; v292_i1 < 16; ++v292_i1) {
              int32_t v293_a = 0 + v292_i1;
              float v295_data = r2[v292_i1];
              int32_t v302_a = v290_lead + (v292_i1 * 12);
              glb_m0[v302_a] = v295_data;
            }
          }
          ;
        }
      }
    }
  }
}

