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
          int32_t v8_lead = threadIdx.x % 16;
          bool v9_g = v8_lead < 12;
          #pragma unroll
          for (int32_t v5_i0 = 0; v5_i0 < 20; ++v5_i0) {
            if (v9_g) {
              int32_t v16_a = v5_i0 + (v8_lead * 20);
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v5_i0 + (v8_lead * 20))]);
              r0[v5_i0] = v24_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          float v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v28_lin;
          float v29_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v29_lin;
          float v30_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v30_lin;
          float v31_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v31_lin;
          float v32_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v32_lin;
          float v33_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v33_lin;
          float v34_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v34_lin;
          float v35_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v35_lin;
          float v36_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v36_lin;
          float v37_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v37_lin;
          float v38_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v38_lin;
          float v39_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v39_lin;
          float v40_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v40_lin;
          float v41_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v41_lin;
          float v42_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v42_lin;
          float v43_lin = glb_m2[256 + threadIdx.x * 1];
          r1[16] = v43_lin;
          float v44_lin = glb_m2[272 + threadIdx.x * 1];
          r1[17] = v44_lin;
          float v45_lin = glb_m2[288 + threadIdx.x * 1];
          r1[18] = v45_lin;
          float v46_lin = glb_m2[304 + threadIdx.x * 1];
          r1[19] = v46_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v48_data = r1[0];
          float v49_data = r1[2];
          float v50_data = r1[4];
          float v51_data = r1[6];
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          tensorforge::transpose4x4b32(v52_tp, v53_tp, v54_tp, v55_tp, v48_data, v49_data, v50_data, v51_data);
          float v56_data = r1[1];
          float v57_data = r1[3];
          float v58_data = r1[5];
          float v59_data = r1[7];
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
          tensorforge::VectorT<float, 4> v64_acc{};
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 2, 0, 0);
          float v73_data = r0[4];
          float v74_data = r0[5];
          float v75_data = r0[6];
          float v76_data = r0[7];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v72_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v79_acc, 2, 1, 0);
          float v81_data = r0[8];
          float v82_data = r0[9];
          float v83_data = r0[10];
          float v84_data = r0[11];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v81_data, v80_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v83_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v87_acc, 2, 2, 0);
          float v89_data = r0[12];
          float v90_data = r0[13];
          float v91_data = r0[14];
          float v92_data = r0[15];
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v89_data, v88_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v90_data, v93_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v91_data, v94_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v95_acc, 2, 3, 0);
          float v97_data = r0[16];
          float v98_data = r0[17];
          float v99_data = r0[18];
          float v100_data = r0[19];
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v97_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v98_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v99_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v103_acc, 2, 0, 0);
          r2[0] = (v104_acc[0]);
          r2[1] = (v104_acc[1]);
          r2[2] = (v104_acc[2]);
          r2[3] = (v104_acc[3]);
          float v109_data = r1[8];
          float v110_data = r1[10];
          float v111_data = r1[12];
          float v112_data = r1[14];
          float v113_tp{};
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          tensorforge::transpose4x4b32(v113_tp, v114_tp, v115_tp, v116_tp, v109_data, v110_data, v111_data, v112_data);
          float v117_data = r1[9];
          float v118_data = r1[11];
          float v119_data = r1[13];
          float v120_data = r1[15];
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          tensorforge::transpose4x4b32(v121_tp, v122_tp, v123_tp, v124_tp, v117_data, v118_data, v119_data, v120_data);
          tensorforge::VectorT<float, 4> v125_acc{};
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v65_data, v125_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v66_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v67_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v68_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v73_data, v133_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v74_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v75_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v76_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v81_data, v141_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v82_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v83_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v84_data, v148_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v89_data, v149_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v90_data, v154_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v91_data, v155_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v92_data, v156_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v97_data, v157_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v98_data, v162_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v99_data, v163_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v100_data, v164_acc, 2, 0, 0);
          r2[4] = (v165_acc[0]);
          r2[5] = (v165_acc[1]);
          r2[6] = (v165_acc[2]);
          r2[7] = (v165_acc[3]);
          float v170_data = r1[16];
          float v171_data = r1[18];
          float v172_data = r1[20];
          float v173_data = r1[22];
          float v174_tp{};
          float v175_tp{};
          float v176_tp{};
          float v177_tp{};
          tensorforge::transpose4x4b32(v174_tp, v175_tp, v176_tp, v177_tp, v170_data, v171_data, v172_data, v173_data);
          float v178_data = r1[17];
          float v179_data = r1[19];
          float v180_data = r1[21];
          float v181_data = r1[23];
          float v182_tp{};
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          tensorforge::transpose4x4b32(v182_tp, v183_tp, v184_tp, v185_tp, v178_data, v179_data, v180_data, v181_data);
          tensorforge::VectorT<float, 4> v186_acc{};
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v65_data, v186_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v66_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v67_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v68_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v73_data, v194_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v74_data, v199_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v75_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v76_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v81_data, v202_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v82_data, v207_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v83_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v84_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v89_data, v210_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v90_data, v215_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v91_data, v216_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v92_data, v217_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v97_data, v218_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v98_data, v223_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v99_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v100_data, v225_acc, 2, 0, 0);
          r2[8] = (v226_acc[0]);
          r2[9] = (v226_acc[1]);
          r2[10] = (v226_acc[2]);
          r2[11] = (v226_acc[3]);
          float v231_data = r1[24];
          float v232_data = r1[26];
          float v233_data = r1[28];
          float v234_data = r1[30];
          float v235_tp{};
          float v236_tp{};
          float v237_tp{};
          float v238_tp{};
          tensorforge::transpose4x4b32(v235_tp, v236_tp, v237_tp, v238_tp, v231_data, v232_data, v233_data, v234_data);
          float v239_data = r1[25];
          float v240_data = r1[27];
          float v241_data = r1[29];
          float v242_data = r1[31];
          float v243_tp{};
          float v244_tp{};
          float v245_tp{};
          float v246_tp{};
          tensorforge::transpose4x4b32(v243_tp, v244_tp, v245_tp, v246_tp, v239_data, v240_data, v241_data, v242_data);
          tensorforge::VectorT<float, 4> v247_acc{};
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v65_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v66_data, v252_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v67_data, v253_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v68_data, v254_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v73_data, v255_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v74_data, v260_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v75_data, v261_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v76_data, v262_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v81_data, v263_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v82_data, v268_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v83_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v84_data, v270_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v89_data, v271_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v90_data, v276_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v91_data, v277_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v92_data, v278_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v97_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v98_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v99_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v100_data, v286_acc, 2, 0, 0);
          r2[12] = (v287_acc[0]);
          r2[13] = (v287_acc[1]);
          r2[14] = (v287_acc[2]);
          r2[15] = (v287_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v294_lead = threadIdx.x % 16;
          if (v294_lead < 12) {
            #pragma unroll
            for (int32_t v296_i1 = 0; v296_i1 < 16; ++v296_i1) {
              int32_t v297_a = 0 + v296_i1;
              float v299_data = r2[v296_i1];
              glb_m0[(v294_lead + (v296_i1 * 12))] = v299_data;
            }
          }
        }
      }
    }
  }
}

