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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 18; ++v5_i1) {
              int32_t v11_a = v5_i1 * 56;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 * 2);
              r0[v22_a] = v20_data;
            }
          }
          if (v3_lead < 24) {
            int32_t v29_lead = v3_lead + 32_i32;
            int32_t v36_lead = v3_lead + 32_i32;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 18; ++v24_i1) {
              int32_t v30_a = v24_i1 * 56;
              int32_t v31_a = v29_lead + v30_a;
              float v39_data = __builtin_nontemporal_load(&glb_m1[(v36_lead + v30_a)]);
              int32_t v41_a = 1 + (v24_i1 * 2);
              r0[v41_a] = v39_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          float v43_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v43_lin;
          float v44_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v44_lin;
          float v45_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v45_lin;
          float v46_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v46_lin;
          float v47_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v47_lin;
          float v48_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v48_lin;
          float v49_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v49_lin;
          float v50_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v50_lin;
          float v51_lin = glb_m2[256 + threadIdx.x * 1];
          r1[8] = v51_lin;
          float v52_lin = glb_m2[288 + threadIdx.x * 1];
          r1[9] = v52_lin;
          float v53_lin = glb_m2[320 + threadIdx.x * 1];
          r1[10] = v53_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          auto& ir2 = r2;
          float v55_data = r1[0];
          float v56_data = r1[1];
          float v57_data = r1[2];
          float v58_data = r1[3];
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v63_acc{};
          float v64_data = r0[0];
          float v65_data = r0[2];
          float v66_data = r0[4];
          float v67_data = r0[6];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v63_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 3, 0, 0);
          float v72_data = r0[8];
          float v73_data = r0[10];
          float v74_data = r0[12];
          float v75_data = r0[14];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v71_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 3, 1, 0);
          float v80_data = r0[16];
          float v81_data = r0[18];
          float v82_data = r0[20];
          float v83_data = r0[22];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v79_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 3, 2, 0);
          float v88_data = r0[24];
          float v89_data = r0[26];
          float v90_data = r0[28];
          float v91_data = r0[30];
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v87_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v89_data, v92_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v90_data, v93_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v91_data, v94_acc, 3, 3, 0);
          float v96_data = r0[32];
          float v97_data = r0[34];
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v96_data, v95_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v97_data, v100_acc, 3, 4, 0);
          ir2[0] = (v101_acc[0]);
          ir2[2] = (v101_acc[1]);
          ir2[4] = (v101_acc[2]);
          ir2[6] = (v101_acc[3]);
          tensorforge::VectorT<float, 4> v106_acc{};
          float v107_data = r0[1];
          float v108_data = r0[3];
          float v109_data = r0[5];
          float v110_data = r0[7];
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v107_data, v106_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v108_data, v111_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v109_data, v112_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v110_data, v113_acc, 3, 0, 0);
          float v115_data = r0[9];
          float v116_data = r0[11];
          float v117_data = r0[13];
          float v118_data = r0[15];
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v115_data, v114_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v116_data, v119_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v117_data, v120_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v118_data, v121_acc, 3, 1, 0);
          float v123_data = r0[17];
          float v124_data = r0[19];
          float v125_data = r0[21];
          float v126_data = r0[23];
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v123_data, v122_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v124_data, v127_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v125_data, v128_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v126_data, v129_acc, 3, 2, 0);
          float v131_data = r0[25];
          float v132_data = r0[27];
          float v133_data = r0[29];
          float v134_data = r0[31];
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v131_data, v130_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v132_data, v135_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v133_data, v136_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v134_data, v137_acc, 3, 3, 0);
          float v139_data = r0[33];
          float v140_data = r0[35];
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v139_data, v138_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v140_data, v143_acc, 3, 4, 0);
          ir2[1] = (v144_acc[0]);
          ir2[3] = (v144_acc[1]);
          ir2[5] = (v144_acc[2]);
          ir2[7] = (v144_acc[3]);
          float v149_data = r1[4];
          float v150_data = r1[5];
          float v151_data = r1[6];
          float v152_data = r1[7];
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          float v156_tp{};
          tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v149_data, v150_data, v151_data, v152_data);
          tensorforge::VectorT<float, 4> v157_acc{};
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v64_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v65_data, v162_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v66_data, v163_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v67_data, v164_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v72_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v73_data, v170_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v74_data, v171_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v75_data, v172_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v80_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v81_data, v178_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v82_data, v179_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v83_data, v180_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v88_data, v181_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v89_data, v186_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v90_data, v187_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v91_data, v188_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v96_data, v189_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v97_data, v194_acc, 3, 4, 0);
          ir2[8] = (v195_acc[0]);
          ir2[10] = (v195_acc[1]);
          ir2[12] = (v195_acc[2]);
          ir2[14] = (v195_acc[3]);
          tensorforge::VectorT<float, 4> v200_acc{};
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v107_data, v200_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v108_data, v205_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v109_data, v206_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v110_data, v207_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v115_data, v208_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v116_data, v213_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v117_data, v214_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v118_data, v215_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v123_data, v216_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v124_data, v221_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v125_data, v222_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v126_data, v223_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v131_data, v224_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v132_data, v229_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v133_data, v230_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v134_data, v231_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v139_data, v232_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v140_data, v237_acc, 3, 4, 0);
          ir2[9] = (v238_acc[0]);
          ir2[11] = (v238_acc[1]);
          ir2[13] = (v238_acc[2]);
          ir2[15] = (v238_acc[3]);
          float v243_data = r1[8];
          float v244_data = r1[9];
          float v245_data = r1[10];
          float v246_data = r1[11];
          float v247_tp{};
          float v248_tp{};
          float v249_tp{};
          float v250_tp{};
          tensorforge::transpose4x4b32(v247_tp, v248_tp, v249_tp, v250_tp, v243_data, v244_data, v245_data, v246_data);
          tensorforge::VectorT<float, 4> v251_acc{};
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v64_data, v251_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v65_data, v256_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v66_data, v257_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v67_data, v258_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v72_data, v259_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v73_data, v264_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v74_data, v265_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v75_data, v266_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v80_data, v267_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v81_data, v272_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v82_data, v273_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v83_data, v274_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v88_data, v275_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v89_data, v280_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v90_data, v281_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v91_data, v282_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v96_data, v283_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v97_data, v288_acc, 3, 4, 0);
          ir2[16] = (v289_acc[0]);
          ir2[18] = (v289_acc[1]);
          ir2[20] = (v289_acc[2]);
          ir2[22] = (v289_acc[3]);
          tensorforge::VectorT<float, 4> v294_acc{};
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v107_data, v294_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v108_data, v299_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v109_data, v300_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v110_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v115_data, v302_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v116_data, v307_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v117_data, v308_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v118_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v123_data, v310_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v124_data, v315_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v125_data, v316_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v126_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v131_data, v318_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v132_data, v323_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v133_data, v324_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v134_data, v325_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v139_data, v326_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v140_data, v331_acc, 3, 4, 0);
          ir2[17] = (v332_acc[0]);
          ir2[19] = (v332_acc[1]);
          ir2[21] = (v332_acc[2]);
          ir2[23] = (v332_acc[3]);
          float v337_data = r1[12];
          float v338_data = r1[13];
          float v339_data = r1[14];
          float v340_data = r1[15];
          float v341_tp{};
          float v342_tp{};
          float v343_tp{};
          float v344_tp{};
          tensorforge::transpose4x4b32(v341_tp, v342_tp, v343_tp, v344_tp, v337_data, v338_data, v339_data, v340_data);
          tensorforge::VectorT<float, 4> v345_acc{};
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v64_data, v345_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v65_data, v350_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v66_data, v351_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v67_data, v352_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v72_data, v353_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v73_data, v358_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v74_data, v359_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v75_data, v360_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v80_data, v361_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v81_data, v366_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v82_data, v367_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v83_data, v368_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v88_data, v369_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v89_data, v374_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v90_data, v375_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v91_data, v376_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v96_data, v377_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v97_data, v382_acc, 3, 4, 0);
          ir2[24] = (v383_acc[0]);
          ir2[26] = (v383_acc[1]);
          ir2[28] = (v383_acc[2]);
          ir2[30] = (v383_acc[3]);
          tensorforge::VectorT<float, 4> v388_acc{};
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v107_data, v388_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v108_data, v393_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v109_data, v394_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v110_data, v395_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v115_data, v396_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v116_data, v401_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v117_data, v402_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v118_data, v403_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v123_data, v404_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v124_data, v409_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v125_data, v410_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v126_data, v411_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v131_data, v412_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v132_data, v417_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v133_data, v418_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v134_data, v419_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v139_data, v420_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v140_data, v425_acc, 3, 4, 0);
          ir2[25] = (v426_acc[0]);
          ir2[27] = (v426_acc[1]);
          ir2[29] = (v426_acc[2]);
          ir2[31] = (v426_acc[3]);
          float v431_data = r1[16];
          float v432_data = r1[17];
          float v435_tp{};
          float v436_tp{};
          float v437_tp{};
          float v438_tp{};
          tensorforge::transpose4x4b32(v435_tp, v436_tp, v437_tp, v438_tp, v431_data, v432_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v439_acc{};
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v64_data, v439_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v65_data, v444_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v66_data, v445_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v67_data, v446_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v72_data, v447_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v73_data, v452_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v74_data, v453_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v75_data, v454_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v80_data, v455_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v81_data, v460_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v82_data, v461_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v83_data, v462_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v88_data, v463_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v89_data, v468_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v90_data, v469_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v91_data, v470_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v96_data, v471_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v97_data, v476_acc, 3, 4, 0);
          ir2[32] = (v477_acc[0]);
          ir2[34] = (v477_acc[1]);
          tensorforge::VectorT<float, 4> v480_acc{};
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v107_data, v480_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v108_data, v485_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v109_data, v486_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v110_data, v487_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v115_data, v488_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v116_data, v493_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v117_data, v494_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v118_data, v495_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v123_data, v496_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v124_data, v501_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v125_data, v502_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v126_data, v503_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v131_data, v504_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v132_data, v509_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v133_data, v510_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v134_data, v511_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v139_data, v512_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v140_data, v517_acc, 3, 4, 0);
          ir2[33] = (v518_acc[0]);
          ir2[35] = (v518_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v524_i0 = 0; v524_i0 < 1; ++v524_i0) {
            int32_t v535_lead = v3_lead + (v524_i0 * 32);
            #pragma unroll
            for (int32_t v525_i1 = 0; v525_i1 < 18; ++v525_i1) {
              int32_t v526_a = v525_i1 * 2;
              int32_t v527_a = v524_i0 + v526_a;
              float v530_data = r2[(v524_i0 + v526_a)];
              int32_t v537_a = v535_lead + (v525_i1 * 56);
              glb_m0[v537_a] = v530_data;
            }
          }
          if (v3_lead < 24) {
            int32_t v549_lead = v3_lead + 32_i32;
            #pragma unroll
            for (int32_t v539_i1 = 0; v539_i1 < 18; ++v539_i1) {
              int32_t v540_a = v539_i1 * 2;
              int32_t v541_a = 1 + v540_a;
              float v544_data = r2[(1 + v540_a)];
              int32_t v551_a = v549_lead + (v539_i1 * 56);
              glb_m0[v551_a] = v544_data;
            }
          }
          ;
        }
      }
    }
  }
}

