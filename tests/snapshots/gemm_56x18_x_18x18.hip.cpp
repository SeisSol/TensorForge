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
          int32_t v6_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 32;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 18; ++v8_i1) {
              int32_t v14_a = v8_i1 * 56;
              int32_t v15_a = v13_lead + v14_a;
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + v14_a)]);
              int32_t v25_a = v7_i0 + (v8_i1 * 2);
              r0[v25_a] = v23_data;
            }
          }
          if (v6_lead < 24) {
            int32_t v32_lead = v6_lead + 32_i32;
            int32_t v39_lead = v6_lead + 32_i32;
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 18; ++v27_i1) {
              int32_t v33_a = v27_i1 * 56;
              int32_t v34_a = v32_lead + v33_a;
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v39_lead + v33_a)]);
              int32_t v44_a = 1 + (v27_i1 * 2);
              r0[v44_a] = v42_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          float v46_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v46_lin;
          float v47_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v47_lin;
          float v48_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v48_lin;
          float v49_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v49_lin;
          float v50_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v50_lin;
          float v51_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v51_lin;
          float v52_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v52_lin;
          float v53_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v53_lin;
          float v54_lin = glb_m2[256 + threadIdx.x * 1];
          r1[8] = v54_lin;
          float v55_lin = glb_m2[288 + threadIdx.x * 1];
          r1[9] = v55_lin;
          float v56_lin = glb_m2[320 + threadIdx.x * 1];
          r1[10] = v56_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          auto& ir2 = r2;
          float v58_data = r1[0];
          float v59_data = r1[1];
          float v60_data = r1[2];
          float v61_data = r1[3];
          float v62_tp{};
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          tensorforge::transpose4x4b32(v62_tp, v63_tp, v64_tp, v65_tp, v58_data, v59_data, v60_data, v61_data);
          tensorforge::VectorT<float, 4> v66_acc{};
          float v67_data = r0[0];
          float v68_data = r0[2];
          float v69_data = r0[4];
          float v70_data = r0[6];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v66_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 3, 0, 0);
          float v75_data = r0[8];
          float v76_data = r0[10];
          float v77_data = r0[12];
          float v78_data = r0[14];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v74_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 3, 1, 0);
          float v83_data = r0[16];
          float v84_data = r0[18];
          float v85_data = r0[20];
          float v86_data = r0[22];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v82_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v87_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v85_data, v88_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v89_acc, 3, 2, 0);
          float v91_data = r0[24];
          float v92_data = r0[26];
          float v93_data = r0[28];
          float v94_data = r0[30];
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v91_data, v90_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v92_data, v95_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v93_data, v96_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v94_data, v97_acc, 3, 3, 0);
          float v99_data = r0[32];
          float v100_data = r0[34];
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v99_data, v98_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v103_acc, 3, 4, 0);
          ir2[0] = (v104_acc[0]);
          ir2[2] = (v104_acc[1]);
          ir2[4] = (v104_acc[2]);
          ir2[6] = (v104_acc[3]);
          tensorforge::VectorT<float, 4> v109_acc{};
          float v110_data = r0[1];
          float v111_data = r0[3];
          float v112_data = r0[5];
          float v113_data = r0[7];
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v110_data, v109_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v111_data, v114_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v112_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v113_data, v116_acc, 3, 0, 0);
          float v118_data = r0[9];
          float v119_data = r0[11];
          float v120_data = r0[13];
          float v121_data = r0[15];
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v118_data, v117_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v119_data, v122_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v120_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v121_data, v124_acc, 3, 1, 0);
          float v126_data = r0[17];
          float v127_data = r0[19];
          float v128_data = r0[21];
          float v129_data = r0[23];
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v126_data, v125_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v127_data, v130_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v128_data, v131_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v129_data, v132_acc, 3, 2, 0);
          float v134_data = r0[25];
          float v135_data = r0[27];
          float v136_data = r0[29];
          float v137_data = r0[31];
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v134_data, v133_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v135_data, v138_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v136_data, v139_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v137_data, v140_acc, 3, 3, 0);
          float v142_data = r0[33];
          float v143_data = r0[35];
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v142_data, v141_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v143_data, v146_acc, 3, 4, 0);
          ir2[1] = (v147_acc[0]);
          ir2[3] = (v147_acc[1]);
          ir2[5] = (v147_acc[2]);
          ir2[7] = (v147_acc[3]);
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
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v67_data, v160_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v68_data, v165_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v69_data, v166_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v70_data, v167_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v75_data, v168_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v76_data, v173_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v77_data, v174_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v78_data, v175_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v83_data, v176_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v84_data, v181_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v85_data, v182_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v86_data, v183_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v91_data, v184_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v92_data, v189_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v93_data, v190_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v94_data, v191_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v99_data, v192_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v100_data, v197_acc, 3, 4, 0);
          ir2[8] = (v198_acc[0]);
          ir2[10] = (v198_acc[1]);
          ir2[12] = (v198_acc[2]);
          ir2[14] = (v198_acc[3]);
          tensorforge::VectorT<float, 4> v203_acc{};
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v110_data, v203_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v111_data, v208_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v112_data, v209_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v113_data, v210_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v118_data, v211_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v119_data, v216_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v120_data, v217_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v121_data, v218_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v126_data, v219_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v127_data, v224_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v128_data, v225_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v129_data, v226_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v134_data, v227_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v135_data, v232_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v136_data, v233_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v137_data, v234_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v142_data, v235_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v157_tp, v143_data, v240_acc, 3, 4, 0);
          ir2[9] = (v241_acc[0]);
          ir2[11] = (v241_acc[1]);
          ir2[13] = (v241_acc[2]);
          ir2[15] = (v241_acc[3]);
          float v246_data = r1[8];
          float v247_data = r1[9];
          float v248_data = r1[10];
          float v249_data = r1[11];
          float v250_tp{};
          float v251_tp{};
          float v252_tp{};
          float v253_tp{};
          tensorforge::transpose4x4b32(v250_tp, v251_tp, v252_tp, v253_tp, v246_data, v247_data, v248_data, v249_data);
          tensorforge::VectorT<float, 4> v254_acc{};
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v67_data, v254_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v68_data, v259_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v69_data, v260_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v70_data, v261_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v75_data, v262_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v76_data, v267_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v77_data, v268_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v78_data, v269_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v83_data, v270_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v84_data, v275_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v85_data, v276_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v86_data, v277_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v91_data, v278_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v92_data, v283_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v93_data, v284_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v94_data, v285_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v99_data, v286_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v100_data, v291_acc, 3, 4, 0);
          ir2[16] = (v292_acc[0]);
          ir2[18] = (v292_acc[1]);
          ir2[20] = (v292_acc[2]);
          ir2[22] = (v292_acc[3]);
          tensorforge::VectorT<float, 4> v297_acc{};
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v110_data, v297_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v111_data, v302_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v112_data, v303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v113_data, v304_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v118_data, v305_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v119_data, v310_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v120_data, v311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v121_data, v312_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v126_data, v313_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v127_data, v318_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v128_data, v319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v129_data, v320_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v134_data, v321_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v135_data, v326_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v136_data, v327_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v137_data, v328_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v142_data, v329_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v143_data, v334_acc, 3, 4, 0);
          ir2[17] = (v335_acc[0]);
          ir2[19] = (v335_acc[1]);
          ir2[21] = (v335_acc[2]);
          ir2[23] = (v335_acc[3]);
          float v340_data = r1[12];
          float v341_data = r1[13];
          float v342_data = r1[14];
          float v343_data = r1[15];
          float v344_tp{};
          float v345_tp{};
          float v346_tp{};
          float v347_tp{};
          tensorforge::transpose4x4b32(v344_tp, v345_tp, v346_tp, v347_tp, v340_data, v341_data, v342_data, v343_data);
          tensorforge::VectorT<float, 4> v348_acc{};
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v67_data, v348_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v68_data, v353_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v69_data, v354_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v70_data, v355_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v75_data, v356_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v76_data, v361_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v77_data, v362_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v78_data, v363_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v83_data, v364_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v84_data, v369_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v85_data, v370_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v86_data, v371_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v91_data, v372_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v92_data, v377_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v93_data, v378_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v94_data, v379_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v99_data, v380_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v100_data, v385_acc, 3, 4, 0);
          ir2[24] = (v386_acc[0]);
          ir2[26] = (v386_acc[1]);
          ir2[28] = (v386_acc[2]);
          ir2[30] = (v386_acc[3]);
          tensorforge::VectorT<float, 4> v391_acc{};
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v110_data, v391_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v111_data, v396_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v112_data, v397_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v113_data, v398_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v118_data, v399_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v119_data, v404_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v120_data, v405_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v121_data, v406_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v126_data, v407_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v127_data, v412_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v128_data, v413_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v129_data, v414_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v134_data, v415_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v135_data, v420_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v136_data, v421_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v137_data, v422_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v142_data, v423_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v143_data, v428_acc, 3, 4, 0);
          ir2[25] = (v429_acc[0]);
          ir2[27] = (v429_acc[1]);
          ir2[29] = (v429_acc[2]);
          ir2[31] = (v429_acc[3]);
          float v434_data = r1[16];
          float v435_data = r1[17];
          float v438_tp{};
          float v439_tp{};
          float v440_tp{};
          float v441_tp{};
          tensorforge::transpose4x4b32(v438_tp, v439_tp, v440_tp, v441_tp, v434_data, v435_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v442_acc{};
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v67_data, v442_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v68_data, v447_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v69_data, v448_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v70_data, v449_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v75_data, v450_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v76_data, v455_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v77_data, v456_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v78_data, v457_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v83_data, v458_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v84_data, v463_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v85_data, v464_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v86_data, v465_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v91_data, v466_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v92_data, v471_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v93_data, v472_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v94_data, v473_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v99_data, v474_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v100_data, v479_acc, 3, 4, 0);
          ir2[32] = (v480_acc[0]);
          ir2[34] = (v480_acc[1]);
          tensorforge::VectorT<float, 4> v483_acc{};
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v110_data, v483_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v111_data, v488_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v112_data, v489_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v113_data, v490_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v118_data, v491_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v119_data, v496_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v120_data, v497_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v121_data, v498_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v126_data, v499_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v127_data, v504_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v128_data, v505_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v129_data, v506_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v134_data, v507_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v135_data, v512_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v136_data, v513_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v137_data, v514_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v142_data, v515_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v143_data, v520_acc, 3, 4, 0);
          ir2[33] = (v521_acc[0]);
          ir2[35] = (v521_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v527_i0 = 0; v527_i0 < 1; ++v527_i0) {
            int32_t v538_lead = v6_lead + (v527_i0 * 32);
            #pragma unroll
            for (int32_t v528_i1 = 0; v528_i1 < 18; ++v528_i1) {
              int32_t v529_a = v528_i1 * 2;
              int32_t v530_a = v527_i0 + v529_a;
              float v533_data = r2[(v527_i0 + v529_a)];
              int32_t v540_a = v538_lead + (v528_i1 * 56);
              glb_m0[v540_a] = v533_data;
            }
          }
          if (v6_lead < 24) {
            int32_t v552_lead = v6_lead + 32_i32;
            #pragma unroll
            for (int32_t v542_i1 = 0; v542_i1 < 18; ++v542_i1) {
              int32_t v543_a = v542_i1 * 2;
              int32_t v544_a = 1 + v543_a;
              float v547_data = r2[(1 + v543_a)];
              int32_t v554_a = v552_lead + (v542_i1 * 56);
              glb_m0[v554_a] = v547_data;
            }
          }
          ;
        }
      }
    }
  }
}

