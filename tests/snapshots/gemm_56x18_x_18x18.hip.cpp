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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 18; ++v4_i1) {
              int32_t v10_a = v4_i1 * 56;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 * 2);
              r0[v21_a] = v19_data;
            }
          }
          if (v2_lead < 24) {
            int32_t v28_lead = v2_lead + 32_i32;
            int32_t v35_lead = v2_lead + 32_i32;
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 18; ++v23_i1) {
              int32_t v29_a = v23_i1 * 56;
              int32_t v30_a = v28_lead + v29_a;
              float v38_data = __builtin_nontemporal_load(&glb_m1[(v35_lead + v29_a)]);
              int32_t v40_a = 1 + (v23_i1 * 2);
              r0[v40_a] = v38_data;
            }
          }
          float r1[18]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[4] = v128;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[5] = v160;
            float v192 = glb_m2[192 + threadIdx.x * 1];
            r1[6] = v192;
            float v224 = glb_m2[224 + threadIdx.x * 1];
            r1[7] = v224;
            float v256 = glb_m2[256 + threadIdx.x * 1];
            r1[8] = v256;
            float v288 = glb_m2[288 + threadIdx.x * 1];
            r1[9] = v288;
            float v320 = glb_m2[320 + threadIdx.x * 1];
            r1[10] = v320;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          auto& ir2 = r2;
          float v41_data = r1[0];
          float v42_data = r1[1];
          float v43_data = r1[2];
          float v44_data = r1[3];
          float v45_tp{};
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          tensorforge::transpose4x4b32(v45_tp, v46_tp, v47_tp, v48_tp, v41_data, v42_data, v43_data, v44_data);
          tensorforge::VectorT<float, 4> v49_acc{};
          float v50_data = r0[0];
          float v51_data = r0[2];
          float v52_data = r0[4];
          float v53_data = r0[6];
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v49_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 3, 0, 0);
          float v58_data = r0[8];
          float v59_data = r0[10];
          float v60_data = r0[12];
          float v61_data = r0[14];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v57_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 3, 1, 0);
          float v66_data = r0[16];
          float v67_data = r0[18];
          float v68_data = r0[20];
          float v69_data = r0[22];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v65_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 3, 2, 0);
          float v74_data = r0[24];
          float v75_data = r0[26];
          float v76_data = r0[28];
          float v77_data = r0[30];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v73_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v78_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v79_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v80_acc, 3, 3, 0);
          float v82_data = r0[32];
          float v83_data = r0[34];
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v82_data, v81_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v83_data, v86_acc, 3, 4, 0);
          ir2[0] = (v87_acc[0]);
          ir2[2] = (v87_acc[1]);
          ir2[4] = (v87_acc[2]);
          ir2[6] = (v87_acc[3]);
          tensorforge::VectorT<float, 4> v92_acc{};
          float v93_data = r0[1];
          float v94_data = r0[3];
          float v95_data = r0[5];
          float v96_data = r0[7];
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v93_data, v92_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v94_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v95_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v96_data, v99_acc, 3, 0, 0);
          float v101_data = r0[9];
          float v102_data = r0[11];
          float v103_data = r0[13];
          float v104_data = r0[15];
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v101_data, v100_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v102_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v103_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v104_data, v107_acc, 3, 1, 0);
          float v109_data = r0[17];
          float v110_data = r0[19];
          float v111_data = r0[21];
          float v112_data = r0[23];
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v109_data, v108_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v110_data, v113_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v111_data, v114_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v112_data, v115_acc, 3, 2, 0);
          float v117_data = r0[25];
          float v118_data = r0[27];
          float v119_data = r0[29];
          float v120_data = r0[31];
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v117_data, v116_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v118_data, v121_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v119_data, v122_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v120_data, v123_acc, 3, 3, 0);
          float v125_data = r0[33];
          float v126_data = r0[35];
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v125_data, v124_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v126_data, v129_acc, 3, 4, 0);
          ir2[1] = (v130_acc[0]);
          ir2[3] = (v130_acc[1]);
          ir2[5] = (v130_acc[2]);
          ir2[7] = (v130_acc[3]);
          float v135_data = r1[4];
          float v136_data = r1[5];
          float v137_data = r1[6];
          float v138_data = r1[7];
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          float v142_tp{};
          tensorforge::transpose4x4b32(v139_tp, v140_tp, v141_tp, v142_tp, v135_data, v136_data, v137_data, v138_data);
          tensorforge::VectorT<float, 4> v143_acc{};
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v50_data, v143_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v51_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v52_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v53_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v58_data, v151_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v59_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v60_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v61_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v66_data, v159_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v67_data, v164_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v68_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v69_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v74_data, v167_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v75_data, v172_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v76_data, v173_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v77_data, v174_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v82_data, v175_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v83_data, v180_acc, 3, 4, 0);
          ir2[8] = (v181_acc[0]);
          ir2[10] = (v181_acc[1]);
          ir2[12] = (v181_acc[2]);
          ir2[14] = (v181_acc[3]);
          tensorforge::VectorT<float, 4> v186_acc{};
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v93_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v94_data, v191_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v95_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v96_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v101_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v102_data, v199_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v103_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v104_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v109_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v110_data, v207_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v111_data, v208_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v112_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v117_data, v210_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v118_data, v215_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v119_data, v216_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v120_data, v217_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v125_data, v218_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v126_data, v223_acc, 3, 4, 0);
          ir2[9] = (v224_acc[0]);
          ir2[11] = (v224_acc[1]);
          ir2[13] = (v224_acc[2]);
          ir2[15] = (v224_acc[3]);
          float v229_data = r1[8];
          float v230_data = r1[9];
          float v231_data = r1[10];
          float v232_data = r1[11];
          float v233_tp{};
          float v234_tp{};
          float v235_tp{};
          float v236_tp{};
          tensorforge::transpose4x4b32(v233_tp, v234_tp, v235_tp, v236_tp, v229_data, v230_data, v231_data, v232_data);
          tensorforge::VectorT<float, 4> v237_acc{};
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v50_data, v237_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v51_data, v242_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v52_data, v243_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v53_data, v244_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v58_data, v245_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v59_data, v250_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v60_data, v251_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v61_data, v252_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v66_data, v253_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v67_data, v258_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v68_data, v259_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v69_data, v260_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v74_data, v261_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v75_data, v266_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v76_data, v267_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v77_data, v268_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v82_data, v269_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v83_data, v274_acc, 3, 4, 0);
          ir2[16] = (v275_acc[0]);
          ir2[18] = (v275_acc[1]);
          ir2[20] = (v275_acc[2]);
          ir2[22] = (v275_acc[3]);
          tensorforge::VectorT<float, 4> v280_acc{};
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v93_data, v280_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v94_data, v285_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v95_data, v286_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v96_data, v287_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v101_data, v288_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v102_data, v293_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v103_data, v294_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v104_data, v295_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v109_data, v296_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v110_data, v301_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v111_data, v302_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v112_data, v303_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v117_data, v304_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v118_data, v309_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v119_data, v310_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v120_data, v311_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v125_data, v312_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v126_data, v317_acc, 3, 4, 0);
          ir2[17] = (v318_acc[0]);
          ir2[19] = (v318_acc[1]);
          ir2[21] = (v318_acc[2]);
          ir2[23] = (v318_acc[3]);
          float v323_data = r1[12];
          float v324_data = r1[13];
          float v325_data = r1[14];
          float v326_data = r1[15];
          float v327_tp{};
          float v328_tp{};
          float v329_tp{};
          float v330_tp{};
          tensorforge::transpose4x4b32(v327_tp, v328_tp, v329_tp, v330_tp, v323_data, v324_data, v325_data, v326_data);
          tensorforge::VectorT<float, 4> v331_acc{};
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v50_data, v331_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v51_data, v336_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v52_data, v337_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v53_data, v338_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v58_data, v339_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v59_data, v344_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v60_data, v345_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v61_data, v346_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v66_data, v347_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v67_data, v352_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v68_data, v353_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v69_data, v354_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v74_data, v355_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v75_data, v360_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v76_data, v361_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v77_data, v362_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v82_data, v363_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v83_data, v368_acc, 3, 4, 0);
          ir2[24] = (v369_acc[0]);
          ir2[26] = (v369_acc[1]);
          ir2[28] = (v369_acc[2]);
          ir2[30] = (v369_acc[3]);
          tensorforge::VectorT<float, 4> v374_acc{};
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v93_data, v374_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v94_data, v379_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v95_data, v380_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v96_data, v381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v101_data, v382_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v102_data, v387_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v103_data, v388_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v104_data, v389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v109_data, v390_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v110_data, v395_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v111_data, v396_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v112_data, v397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v117_data, v398_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v118_data, v403_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v119_data, v404_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v120_data, v405_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v125_data, v406_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v126_data, v411_acc, 3, 4, 0);
          ir2[25] = (v412_acc[0]);
          ir2[27] = (v412_acc[1]);
          ir2[29] = (v412_acc[2]);
          ir2[31] = (v412_acc[3]);
          float v417_data = r1[16];
          float v418_data = r1[17];
          float v421_tp{};
          float v422_tp{};
          float v423_tp{};
          float v424_tp{};
          tensorforge::transpose4x4b32(v421_tp, v422_tp, v423_tp, v424_tp, v417_data, v418_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v425_acc{};
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v50_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v51_data, v430_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v52_data, v431_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v53_data, v432_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v58_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v59_data, v438_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v60_data, v439_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v61_data, v440_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v66_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v67_data, v446_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v68_data, v447_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v69_data, v448_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v74_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v75_data, v454_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v76_data, v455_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v77_data, v456_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v82_data, v457_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v83_data, v462_acc, 3, 4, 0);
          ir2[32] = (v463_acc[0]);
          ir2[34] = (v463_acc[1]);
          tensorforge::VectorT<float, 4> v466_acc{};
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v93_data, v466_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v94_data, v471_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v95_data, v472_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v96_data, v473_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v101_data, v474_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v102_data, v479_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v103_data, v480_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v104_data, v481_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v109_data, v482_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v110_data, v487_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v111_data, v488_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v112_data, v489_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v117_data, v490_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v118_data, v495_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v119_data, v496_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v120_data, v497_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v125_data, v498_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v126_data, v503_acc, 3, 4, 0);
          ir2[33] = (v504_acc[0]);
          ir2[35] = (v504_acc[1]);
          // glb_m0 = store{r>g}(r2);
          int32_t v509_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v510_i0 = 0; v510_i0 < 1; ++v510_i0) {
            int32_t v521_lead = v509_lead + (v510_i0 * 32);
            #pragma unroll
            for (int32_t v511_i1 = 0; v511_i1 < 18; ++v511_i1) {
              int32_t v512_a = v511_i1 * 2;
              int32_t v513_a = v510_i0 + v512_a;
              float v516_data = r2[(v510_i0 + v512_a)];
              int32_t v523_a = v521_lead + (v511_i1 * 56);
              glb_m0[v523_a] = v516_data;
            }
          }
          if (v509_lead < 24) {
            int32_t v535_lead = v509_lead + 32_i32;
            #pragma unroll
            for (int32_t v525_i1 = 0; v525_i1 < 18; ++v525_i1) {
              int32_t v526_a = v525_i1 * 2;
              int32_t v527_a = 1 + v526_a;
              float v530_data = r2[(1 + v526_a)];
              int32_t v537_a = v535_lead + (v525_i1 * 56);
              glb_m0[v537_a] = v530_data;
            }
          }
          ;
        }
      }
    }
  }
}

