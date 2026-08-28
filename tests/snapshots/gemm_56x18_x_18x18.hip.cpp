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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 1008 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1008 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 18; ++v12_i1) {
              int32_t v18_a = v12_i1 * 56;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + (v12_i1 * 2))] = v27_data;
            }
          }
          if (v10_lead < 24) {
            int32_t v36_lead = v10_lead + 32_i32;
            int32_t v43_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 18; ++v31_i1) {
              int32_t v37_a = v31_i1 * 56;
              int32_t v38_a = v36_lead + v37_a;
              float v46_data = __builtin_nontemporal_load(&glb_m1[(v43_lead + v37_a)]);
              r0[(1 + (v31_i1 * 2))] = v46_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          float v50_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v50_lin;
          float v51_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v51_lin;
          float v52_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v52_lin;
          float v53_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v53_lin;
          float v54_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v54_lin;
          float v55_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v55_lin;
          float v56_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v56_lin;
          float v57_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v57_lin;
          float v58_lin = glb_m2[256 + threadIdx.x * 1];
          r1[8] = v58_lin;
          float v59_lin = glb_m2[288 + threadIdx.x * 1];
          r1[9] = v59_lin;
          float v60_lin = glb_m2[320 + threadIdx.x * 1];
          r1[10] = v60_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          float v62_data = r1[0];
          float v63_data = r1[1];
          float v64_data = r1[2];
          float v65_data = r1[3];
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          tensorforge::transpose4x4b32(v66_tp, v67_tp, v68_tp, v69_tp, v62_data, v63_data, v64_data, v65_data);
          tensorforge::VectorT<float, 4> v70_acc{};
          float v71_data = r0[0];
          float v72_data = r0[2];
          float v73_data = r0[4];
          float v74_data = r0[6];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v70_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v75_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 3, 0, 0);
          float v79_data = r0[8];
          float v80_data = r0[10];
          float v81_data = r0[12];
          float v82_data = r0[14];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v78_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v83_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 3, 1, 0);
          float v87_data = r0[16];
          float v88_data = r0[18];
          float v89_data = r0[20];
          float v90_data = r0[22];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v86_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v91_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v89_data, v92_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v90_data, v93_acc, 3, 2, 0);
          float v95_data = r0[24];
          float v96_data = r0[26];
          float v97_data = r0[28];
          float v98_data = r0[30];
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v95_data, v94_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v96_data, v99_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v97_data, v100_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v98_data, v101_acc, 3, 3, 0);
          float v103_data = r0[32];
          float v104_data = r0[34];
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v102_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v107_acc, 3, 4, 0);
          r2[0] = (v108_acc[0]);
          r2[2] = (v108_acc[1]);
          r2[4] = (v108_acc[2]);
          r2[6] = (v108_acc[3]);
          tensorforge::VectorT<float, 4> v113_acc{};
          float v114_data = r0[1];
          float v115_data = r0[3];
          float v116_data = r0[5];
          float v117_data = r0[7];
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v114_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v115_data, v118_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v116_data, v119_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v117_data, v120_acc, 3, 0, 0);
          float v122_data = r0[9];
          float v123_data = r0[11];
          float v124_data = r0[13];
          float v125_data = r0[15];
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v122_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v123_data, v126_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v124_data, v127_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v125_data, v128_acc, 3, 1, 0);
          float v130_data = r0[17];
          float v131_data = r0[19];
          float v132_data = r0[21];
          float v133_data = r0[23];
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v130_data, v129_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v131_data, v134_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v132_data, v135_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v133_data, v136_acc, 3, 2, 0);
          float v138_data = r0[25];
          float v139_data = r0[27];
          float v140_data = r0[29];
          float v141_data = r0[31];
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v138_data, v137_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v139_data, v142_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v140_data, v143_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v141_data, v144_acc, 3, 3, 0);
          float v146_data = r0[33];
          float v147_data = r0[35];
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v146_data, v145_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v147_data, v150_acc, 3, 4, 0);
          r2[1] = (v151_acc[0]);
          r2[3] = (v151_acc[1]);
          r2[5] = (v151_acc[2]);
          r2[7] = (v151_acc[3]);
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
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v71_data, v164_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v72_data, v169_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v73_data, v170_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v74_data, v171_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v79_data, v172_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v80_data, v177_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v81_data, v178_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v82_data, v179_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v87_data, v180_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v88_data, v185_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v89_data, v186_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v90_data, v187_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v95_data, v188_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v96_data, v193_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v97_data, v194_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v98_data, v195_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v103_data, v196_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v104_data, v201_acc, 3, 4, 0);
          r2[8] = (v202_acc[0]);
          r2[10] = (v202_acc[1]);
          r2[12] = (v202_acc[2]);
          r2[14] = (v202_acc[3]);
          tensorforge::VectorT<float, 4> v207_acc{};
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v114_data, v207_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v115_data, v212_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v116_data, v213_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v117_data, v214_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v122_data, v215_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v123_data, v220_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v124_data, v221_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v125_data, v222_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v130_data, v223_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v131_data, v228_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v132_data, v229_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v133_data, v230_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v138_data, v231_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v139_data, v236_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v140_data, v237_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v141_data, v238_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v146_data, v239_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v147_data, v244_acc, 3, 4, 0);
          r2[9] = (v245_acc[0]);
          r2[11] = (v245_acc[1]);
          r2[13] = (v245_acc[2]);
          r2[15] = (v245_acc[3]);
          float v250_data = r1[8];
          float v251_data = r1[9];
          float v252_data = r1[10];
          float v253_data = r1[11];
          float v254_tp{};
          float v255_tp{};
          float v256_tp{};
          float v257_tp{};
          tensorforge::transpose4x4b32(v254_tp, v255_tp, v256_tp, v257_tp, v250_data, v251_data, v252_data, v253_data);
          tensorforge::VectorT<float, 4> v258_acc{};
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v71_data, v258_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v72_data, v263_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v73_data, v264_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v74_data, v265_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v79_data, v266_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v80_data, v271_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v81_data, v272_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v82_data, v273_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v87_data, v274_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v88_data, v279_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v89_data, v280_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v90_data, v281_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v95_data, v282_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v96_data, v287_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v97_data, v288_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v98_data, v289_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v103_data, v290_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v104_data, v295_acc, 3, 4, 0);
          r2[16] = (v296_acc[0]);
          r2[18] = (v296_acc[1]);
          r2[20] = (v296_acc[2]);
          r2[22] = (v296_acc[3]);
          tensorforge::VectorT<float, 4> v301_acc{};
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v114_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v115_data, v306_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v116_data, v307_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v117_data, v308_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v122_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v123_data, v314_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v124_data, v315_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v125_data, v316_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v130_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v131_data, v322_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v132_data, v323_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v133_data, v324_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v138_data, v325_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v139_data, v330_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v140_data, v331_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v141_data, v332_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v146_data, v333_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v147_data, v338_acc, 3, 4, 0);
          r2[17] = (v339_acc[0]);
          r2[19] = (v339_acc[1]);
          r2[21] = (v339_acc[2]);
          r2[23] = (v339_acc[3]);
          float v344_data = r1[12];
          float v345_data = r1[13];
          float v346_data = r1[14];
          float v347_data = r1[15];
          float v348_tp{};
          float v349_tp{};
          float v350_tp{};
          float v351_tp{};
          tensorforge::transpose4x4b32(v348_tp, v349_tp, v350_tp, v351_tp, v344_data, v345_data, v346_data, v347_data);
          tensorforge::VectorT<float, 4> v352_acc{};
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v71_data, v352_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v72_data, v357_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v73_data, v358_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v74_data, v359_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v79_data, v360_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v80_data, v365_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v81_data, v366_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v82_data, v367_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v87_data, v368_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v88_data, v373_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v89_data, v374_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v90_data, v375_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v95_data, v376_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v96_data, v381_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v97_data, v382_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v98_data, v383_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v103_data, v384_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v104_data, v389_acc, 3, 4, 0);
          r2[24] = (v390_acc[0]);
          r2[26] = (v390_acc[1]);
          r2[28] = (v390_acc[2]);
          r2[30] = (v390_acc[3]);
          tensorforge::VectorT<float, 4> v395_acc{};
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v114_data, v395_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v115_data, v400_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v116_data, v401_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v117_data, v402_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v122_data, v403_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v123_data, v408_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v124_data, v409_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v125_data, v410_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v130_data, v411_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v131_data, v416_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v132_data, v417_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v133_data, v418_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v138_data, v419_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v139_data, v424_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v140_data, v425_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v141_data, v426_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v146_data, v427_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v147_data, v432_acc, 3, 4, 0);
          r2[25] = (v433_acc[0]);
          r2[27] = (v433_acc[1]);
          r2[29] = (v433_acc[2]);
          r2[31] = (v433_acc[3]);
          float v438_data = r1[16];
          float v439_data = r1[17];
          float v442_tp{};
          float v443_tp{};
          float v444_tp{};
          float v445_tp{};
          tensorforge::transpose4x4b32(v442_tp, v443_tp, v444_tp, v445_tp, v438_data, v439_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v446_acc{};
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v71_data, v446_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v72_data, v451_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v73_data, v452_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v74_data, v453_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v79_data, v454_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v80_data, v459_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v81_data, v460_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v82_data, v461_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v87_data, v462_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v88_data, v467_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v89_data, v468_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v90_data, v469_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v95_data, v470_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v96_data, v475_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v97_data, v476_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v98_data, v477_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v103_data, v478_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v104_data, v483_acc, 3, 4, 0);
          r2[32] = (v484_acc[0]);
          r2[34] = (v484_acc[1]);
          tensorforge::VectorT<float, 4> v487_acc{};
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v114_data, v487_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v115_data, v492_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v116_data, v493_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v117_data, v494_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v122_data, v495_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v123_data, v500_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v124_data, v501_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v125_data, v502_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v130_data, v503_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v131_data, v508_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v132_data, v509_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v133_data, v510_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v138_data, v511_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v139_data, v516_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v140_data, v517_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v141_data, v518_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v146_data, v519_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v147_data, v524_acc, 3, 4, 0);
          r2[33] = (v525_acc[0]);
          r2[35] = (v525_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v531_i0 = 0; v531_i0 < 1; ++v531_i0) {
            int32_t v542_lead = v10_lead + (v531_i0 * 32);
            #pragma unroll
            for (int32_t v532_i1 = 0; v532_i1 < 18; ++v532_i1) {
              int32_t v533_a = v532_i1 * 2;
              int32_t v534_a = v531_i0 + v533_a;
              float v537_data = r2[(v531_i0 + v533_a)];
              glb_m0[(v542_lead + (v532_i1 * 56))] = v537_data;
            }
          }
          if (v10_lead < 24) {
            int32_t v556_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v546_i1 = 0; v546_i1 < 18; ++v546_i1) {
              int32_t v547_a = v546_i1 * 2;
              int32_t v548_a = 1 + v547_a;
              float v551_data = r2[(1 + v547_a)];
              glb_m0[(v556_lead + (v546_i1 * 56))] = v551_data;
            }
          }
        }
      }
    }
  }
}

