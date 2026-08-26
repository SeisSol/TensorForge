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
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
            float v48 = glb_m2[48 + threadIdx.x * 1];
            r1[3] = v48;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[4] = v64;
            float v80 = glb_m2[80 + threadIdx.x * 1];
            r1[5] = v80;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[6] = v96;
            float v112 = glb_m2[112 + threadIdx.x * 1];
            r1[7] = v112;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[8] = v128;
            float v144 = glb_m2[144 + threadIdx.x * 1];
            r1[9] = v144;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[10] = v160;
            float v176 = glb_m2[176 + threadIdx.x * 1];
            r1[11] = v176;
            float v192 = glb_m2[192 + threadIdx.x * 1];
            r1[12] = v192;
            float v208 = glb_m2[208 + threadIdx.x * 1];
            r1[13] = v208;
            float v224 = glb_m2[224 + threadIdx.x * 1];
            r1[14] = v224;
            float v240 = glb_m2[240 + threadIdx.x * 1];
            r1[15] = v240;
            float v256 = glb_m2[256 + threadIdx.x * 1];
            r1[16] = v256;
            float v272 = glb_m2[272 + threadIdx.x * 1];
            r1[17] = v272;
            float v288 = glb_m2[288 + threadIdx.x * 1];
            r1[18] = v288;
            float v304 = glb_m2[304 + threadIdx.x * 1];
            r1[19] = v304;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          auto& ir2 = r2;
          float v24_data = r1[0];
          float v25_data = r1[2];
          float v26_data = r1[4];
          float v27_data = r1[6];
          float v28_tp{};
          float v29_tp{};
          float v30_tp{};
          float v31_tp{};
          tensorforge::transpose4x4b32(v28_tp, v29_tp, v30_tp, v31_tp, v24_data, v25_data, v26_data, v27_data);
          float v32_data = r1[1];
          float v33_data = r1[3];
          float v34_data = r1[5];
          float v35_data = r1[7];
          float v36_tp{};
          float v37_tp{};
          float v38_tp{};
          float v39_tp{};
          tensorforge::transpose4x4b32(v36_tp, v37_tp, v38_tp, v39_tp, v32_data, v33_data, v34_data, v35_data);
          tensorforge::VectorT<float, 4> v40_acc{};
          float v41_data = r0[0];
          float v42_data = r0[1];
          float v43_data = r0[2];
          float v44_data = r0[3];
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v41_data, v40_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v42_data, v45_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v43_data, v46_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v44_data, v47_acc, 2, 0, 0);
          float v49_data = r0[4];
          float v50_data = r0[5];
          float v51_data = r0[6];
          float v52_data = r0[7];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v49_data, v48_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v50_data, v53_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v51_data, v54_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v52_data, v55_acc, 2, 1, 0);
          float v57_data = r0[8];
          float v58_data = r0[9];
          float v59_data = r0[10];
          float v60_data = r0[11];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v56_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v58_data, v61_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v59_data, v62_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v63_acc, 2, 2, 0);
          float v65_data = r0[12];
          float v66_data = r0[13];
          float v67_data = r0[14];
          float v68_data = r0[15];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v65_data, v64_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v66_data, v69_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v67_data, v70_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v68_data, v71_acc, 2, 3, 0);
          float v73_data = r0[16];
          float v74_data = r0[17];
          float v75_data = r0[18];
          float v76_data = r0[19];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v73_data, v72_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v74_data, v77_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v75_data, v78_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v76_data, v79_acc, 2, 0, 0);
          ir2[0] = (v80_acc[0]);
          ir2[1] = (v80_acc[1]);
          ir2[2] = (v80_acc[2]);
          ir2[3] = (v80_acc[3]);
          float v85_data = r1[8];
          float v86_data = r1[10];
          float v87_data = r1[12];
          float v88_data = r1[14];
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          float v92_tp{};
          tensorforge::transpose4x4b32(v89_tp, v90_tp, v91_tp, v92_tp, v85_data, v86_data, v87_data, v88_data);
          float v93_data = r1[9];
          float v94_data = r1[11];
          float v95_data = r1[13];
          float v96_data = r1[15];
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          tensorforge::transpose4x4b32(v97_tp, v98_tp, v99_tp, v100_tp, v93_data, v94_data, v95_data, v96_data);
          tensorforge::VectorT<float, 4> v101_acc{};
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v41_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v42_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v43_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v44_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v116_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v124_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v125_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v130_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v131_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v132_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v139_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v140_acc, 2, 0, 0);
          ir2[4] = (v141_acc[0]);
          ir2[5] = (v141_acc[1]);
          ir2[6] = (v141_acc[2]);
          ir2[7] = (v141_acc[3]);
          float v146_data = r1[16];
          float v147_data = r1[18];
          float v148_data = r1[20];
          float v149_data = r1[22];
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          float v153_tp{};
          tensorforge::transpose4x4b32(v150_tp, v151_tp, v152_tp, v153_tp, v146_data, v147_data, v148_data, v149_data);
          float v154_data = r1[17];
          float v155_data = r1[19];
          float v156_data = r1[21];
          float v157_data = r1[23];
          float v158_tp{};
          float v159_tp{};
          float v160_tp{};
          float v161_tp{};
          tensorforge::transpose4x4b32(v158_tp, v159_tp, v160_tp, v161_tp, v154_data, v155_data, v156_data, v157_data);
          tensorforge::VectorT<float, 4> v162_acc{};
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v41_data, v162_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v42_data, v167_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v43_data, v168_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v44_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v49_data, v170_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v50_data, v175_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v51_data, v176_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v52_data, v177_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v57_data, v178_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v58_data, v183_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v59_data, v184_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v60_data, v185_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v65_data, v186_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v66_data, v191_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v67_data, v192_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v68_data, v193_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v158_tp, v73_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v74_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v75_data, v200_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v76_data, v201_acc, 2, 0, 0);
          ir2[8] = (v202_acc[0]);
          ir2[9] = (v202_acc[1]);
          ir2[10] = (v202_acc[2]);
          ir2[11] = (v202_acc[3]);
          float v207_data = r1[24];
          float v208_data = r1[26];
          float v209_data = r1[28];
          float v210_data = r1[30];
          float v211_tp{};
          float v212_tp{};
          float v213_tp{};
          float v214_tp{};
          tensorforge::transpose4x4b32(v211_tp, v212_tp, v213_tp, v214_tp, v207_data, v208_data, v209_data, v210_data);
          float v215_data = r1[25];
          float v216_data = r1[27];
          float v217_data = r1[29];
          float v218_data = r1[31];
          float v219_tp{};
          float v220_tp{};
          float v221_tp{};
          float v222_tp{};
          tensorforge::transpose4x4b32(v219_tp, v220_tp, v221_tp, v222_tp, v215_data, v216_data, v217_data, v218_data);
          tensorforge::VectorT<float, 4> v223_acc{};
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v41_data, v223_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v42_data, v228_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v43_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v44_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v49_data, v231_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v50_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v51_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v52_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v57_data, v239_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v58_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v59_data, v245_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v60_data, v246_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v65_data, v247_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v66_data, v252_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v67_data, v253_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v68_data, v254_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v73_data, v255_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v74_data, v260_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v75_data, v261_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v76_data, v262_acc, 2, 0, 0);
          ir2[12] = (v263_acc[0]);
          ir2[13] = (v263_acc[1]);
          ir2[14] = (v263_acc[2]);
          ir2[15] = (v263_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v270_lead = threadIdx.x % 16;
          if (v270_lead < 12) {
            #pragma unroll
            for (int32_t v272_i1 = 0; v272_i1 < 16; ++v272_i1) {
              int32_t v273_a = 0 + v272_i1;
              float v275_data = r2[v272_i1];
              int32_t v282_a = v270_lead + (v272_i1 * 12);
              glb_m0[v282_a] = v275_data;
            }
          }
          ;
        }
      }
    }
  }
}

