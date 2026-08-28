// === base name ===
kernel_d08f36e369

// === header ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d08f36e369, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d08f36e369), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_d08f36e369, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 56×18(56×18) {0..56}×{0..18} strided
    // m1 56×18(56×18) {0..56}×{0..18} strided
    // m2 18×18(18×18) {0..18}×{0..18} strided
    // m0 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, 1] = m1 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, -1]×m2 18×18(18×18) {0..18}×{0..18} strided({0..18}×{0..18})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 1008 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1008 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 18; ++v9_i1) {
              int32_t v15_a = v9_i1 * 56;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 * 2))] = v24_data;
            }
          }
          if (v7_lead < 24) {
            int32_t v33_lead = v7_lead + 32_i32;
            int32_t v40_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 18; ++v28_i1) {
              int32_t v34_a = v28_i1 * 56;
              int32_t v35_a = v33_lead + v34_a;
              float v43_data = __builtin_nontemporal_load(&glb_m1[(v40_lead + v34_a)]);
              r0[(1 + (v28_i1 * 2))] = v43_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          float v47_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v47_lin;
          float v48_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v48_lin;
          float v49_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v49_lin;
          float v50_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v50_lin;
          float v51_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v51_lin;
          float v52_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v52_lin;
          float v53_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v53_lin;
          float v54_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v54_lin;
          float v55_lin = glb_m2[256 + threadIdx.x * 1];
          r1[8] = v55_lin;
          float v56_lin = glb_m2[288 + threadIdx.x * 1];
          r1[9] = v56_lin;
          float v57_lin = glb_m2[320 + threadIdx.x * 1];
          r1[10] = v57_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
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
          float v69_data = r0[2];
          float v70_data = r0[4];
          float v71_data = r0[6];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v67_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 3, 0, 0);
          float v76_data = r0[8];
          float v77_data = r0[10];
          float v78_data = r0[12];
          float v79_data = r0[14];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v75_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 3, 1, 0);
          float v84_data = r0[16];
          float v85_data = r0[18];
          float v86_data = r0[20];
          float v87_data = r0[22];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v83_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v85_data, v88_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v89_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v90_acc, 3, 2, 0);
          float v92_data = r0[24];
          float v93_data = r0[26];
          float v94_data = r0[28];
          float v95_data = r0[30];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v92_data, v91_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v93_data, v96_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v94_data, v97_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v95_data, v98_acc, 3, 3, 0);
          float v100_data = r0[32];
          float v101_data = r0[34];
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v99_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v101_data, v104_acc, 3, 4, 0);
          r2[0] = (v105_acc[0]);
          r2[2] = (v105_acc[1]);
          r2[4] = (v105_acc[2]);
          r2[6] = (v105_acc[3]);
          tensorforge::VectorT<float, 4> v110_acc{};
          float v111_data = r0[1];
          float v112_data = r0[3];
          float v113_data = r0[5];
          float v114_data = r0[7];
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v111_data, v110_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v112_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v113_data, v116_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v114_data, v117_acc, 3, 0, 0);
          float v119_data = r0[9];
          float v120_data = r0[11];
          float v121_data = r0[13];
          float v122_data = r0[15];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v119_data, v118_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v120_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v121_data, v124_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v122_data, v125_acc, 3, 1, 0);
          float v127_data = r0[17];
          float v128_data = r0[19];
          float v129_data = r0[21];
          float v130_data = r0[23];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v127_data, v126_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v128_data, v131_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v129_data, v132_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v130_data, v133_acc, 3, 2, 0);
          float v135_data = r0[25];
          float v136_data = r0[27];
          float v137_data = r0[29];
          float v138_data = r0[31];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v135_data, v134_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v136_data, v139_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v137_data, v140_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v138_data, v141_acc, 3, 3, 0);
          float v143_data = r0[33];
          float v144_data = r0[35];
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v143_data, v142_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v144_data, v147_acc, 3, 4, 0);
          r2[1] = (v148_acc[0]);
          r2[3] = (v148_acc[1]);
          r2[5] = (v148_acc[2]);
          r2[7] = (v148_acc[3]);
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
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v68_data, v161_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v69_data, v166_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v70_data, v167_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v71_data, v168_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v76_data, v169_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v77_data, v174_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v78_data, v175_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v79_data, v176_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v84_data, v177_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v85_data, v182_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v86_data, v183_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v87_data, v184_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v92_data, v185_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v93_data, v190_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v94_data, v191_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v95_data, v192_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v100_data, v193_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v101_data, v198_acc, 3, 4, 0);
          r2[8] = (v199_acc[0]);
          r2[10] = (v199_acc[1]);
          r2[12] = (v199_acc[2]);
          r2[14] = (v199_acc[3]);
          tensorforge::VectorT<float, 4> v204_acc{};
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v111_data, v204_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v112_data, v209_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v113_data, v210_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v114_data, v211_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v119_data, v212_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v120_data, v217_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v121_data, v218_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v122_data, v219_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v127_data, v220_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v128_data, v225_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v129_data, v226_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v130_data, v227_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v135_data, v228_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v136_data, v233_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v137_data, v234_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v138_data, v235_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v143_data, v236_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v144_data, v241_acc, 3, 4, 0);
          r2[9] = (v242_acc[0]);
          r2[11] = (v242_acc[1]);
          r2[13] = (v242_acc[2]);
          r2[15] = (v242_acc[3]);
          float v247_data = r1[8];
          float v248_data = r1[9];
          float v249_data = r1[10];
          float v250_data = r1[11];
          float v251_tp{};
          float v252_tp{};
          float v253_tp{};
          float v254_tp{};
          tensorforge::transpose4x4b32(v251_tp, v252_tp, v253_tp, v254_tp, v247_data, v248_data, v249_data, v250_data);
          tensorforge::VectorT<float, 4> v255_acc{};
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v68_data, v255_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v69_data, v260_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v70_data, v261_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v71_data, v262_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v76_data, v263_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v77_data, v268_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v78_data, v269_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v79_data, v270_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v84_data, v271_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v85_data, v276_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v86_data, v277_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v87_data, v278_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v92_data, v279_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v93_data, v284_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v94_data, v285_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v95_data, v286_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v100_data, v287_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v101_data, v292_acc, 3, 4, 0);
          r2[16] = (v293_acc[0]);
          r2[18] = (v293_acc[1]);
          r2[20] = (v293_acc[2]);
          r2[22] = (v293_acc[3]);
          tensorforge::VectorT<float, 4> v298_acc{};
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v111_data, v298_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v112_data, v303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v113_data, v304_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v114_data, v305_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v119_data, v306_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v120_data, v311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v121_data, v312_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v122_data, v313_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v127_data, v314_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v128_data, v319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v129_data, v320_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v130_data, v321_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v135_data, v322_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v136_data, v327_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v137_data, v328_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v138_data, v329_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v143_data, v330_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v144_data, v335_acc, 3, 4, 0);
          r2[17] = (v336_acc[0]);
          r2[19] = (v336_acc[1]);
          r2[21] = (v336_acc[2]);
          r2[23] = (v336_acc[3]);
          float v341_data = r1[12];
          float v342_data = r1[13];
          float v343_data = r1[14];
          float v344_data = r1[15];
          float v345_tp{};
          float v346_tp{};
          float v347_tp{};
          float v348_tp{};
          tensorforge::transpose4x4b32(v345_tp, v346_tp, v347_tp, v348_tp, v341_data, v342_data, v343_data, v344_data);
          tensorforge::VectorT<float, 4> v349_acc{};
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v68_data, v349_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v69_data, v354_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v70_data, v355_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v71_data, v356_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v76_data, v357_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v77_data, v362_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v78_data, v363_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v79_data, v364_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v84_data, v365_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v85_data, v370_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v86_data, v371_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v87_data, v372_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v92_data, v373_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v93_data, v378_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v94_data, v379_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v95_data, v380_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v100_data, v381_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v101_data, v386_acc, 3, 4, 0);
          r2[24] = (v387_acc[0]);
          r2[26] = (v387_acc[1]);
          r2[28] = (v387_acc[2]);
          r2[30] = (v387_acc[3]);
          tensorforge::VectorT<float, 4> v392_acc{};
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v111_data, v392_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v112_data, v397_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v113_data, v398_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v114_data, v399_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v119_data, v400_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v120_data, v405_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v121_data, v406_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v122_data, v407_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v127_data, v408_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v128_data, v413_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v129_data, v414_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v130_data, v415_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v135_data, v416_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v136_data, v421_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v137_data, v422_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v138_data, v423_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v143_data, v424_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v144_data, v429_acc, 3, 4, 0);
          r2[25] = (v430_acc[0]);
          r2[27] = (v430_acc[1]);
          r2[29] = (v430_acc[2]);
          r2[31] = (v430_acc[3]);
          float v435_data = r1[16];
          float v436_data = r1[17];
          float v439_tp{};
          float v440_tp{};
          float v441_tp{};
          float v442_tp{};
          tensorforge::transpose4x4b32(v439_tp, v440_tp, v441_tp, v442_tp, v435_data, v436_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v443_acc{};
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v68_data, v443_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v69_data, v448_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v70_data, v449_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v71_data, v450_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v76_data, v451_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v77_data, v456_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v78_data, v457_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v79_data, v458_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v84_data, v459_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v85_data, v464_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v86_data, v465_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v87_data, v466_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v92_data, v467_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v93_data, v472_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v94_data, v473_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v95_data, v474_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v100_data, v475_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v101_data, v480_acc, 3, 4, 0);
          r2[32] = (v481_acc[0]);
          r2[34] = (v481_acc[1]);
          tensorforge::VectorT<float, 4> v484_acc{};
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v111_data, v484_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v112_data, v489_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v113_data, v490_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v114_data, v491_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v119_data, v492_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v120_data, v497_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v121_data, v498_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v122_data, v499_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v127_data, v500_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v128_data, v505_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v129_data, v506_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v130_data, v507_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v135_data, v508_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v136_data, v513_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v137_data, v514_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v138_data, v515_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v143_data, v516_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v144_data, v521_acc, 3, 4, 0);
          r2[33] = (v522_acc[0]);
          r2[35] = (v522_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v528_i0 = 0; v528_i0 < 1; ++v528_i0) {
            int32_t v539_lead = v7_lead + (v528_i0 * 32);
            #pragma unroll
            for (int32_t v529_i1 = 0; v529_i1 < 18; ++v529_i1) {
              int32_t v530_a = v529_i1 * 2;
              int32_t v531_a = v528_i0 + v530_a;
              float v534_data = r2[(v528_i0 + v530_a)];
              glb_m0[(v539_lead + (v529_i1 * 56))] = v534_data;
            }
          }
          if (v7_lead < 24) {
            int32_t v553_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v543_i1 = 0; v543_i1 < 18; ++v543_i1) {
              int32_t v544_a = v543_i1 * 2;
              int32_t v545_a = 1 + v544_a;
              float v548_data = r2[(1 + v544_a)];
              glb_m0[(v553_lead + (v543_i1 * 56))] = v548_data;
            }
          }
        }
      }
    }
  }
}

