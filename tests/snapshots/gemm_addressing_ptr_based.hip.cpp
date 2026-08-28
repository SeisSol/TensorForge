// === base name ===
kernel_769f1b7f89

// === header ===
void launcher_kernel_769f1b7f89(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_769f1b7f89(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_769f1b7f89, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_769f1b7f89), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_769f1b7f89, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_769f1b7f89(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} pointer_based
    // m1 16×16(16×16) {0..16}×{0..16} pointer_based
    // m2 16×16(16×16) {0..16}×{0..16} pointer_based
    // m0 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[-1, 1]
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
          auto glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 16;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 16;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + v9_i1)] = v24_data;
            }
          }
          float r1[16]{};
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
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v44_data = r1[0];
          float v45_data = r1[1];
          float v46_data = r1[2];
          float v47_data = r1[3];
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          tensorforge::transpose4x4b32(v48_tp, v49_tp, v50_tp, v51_tp, v44_data, v45_data, v46_data, v47_data);
          tensorforge::VectorT<float, 4> v52_acc{};
          float v53_data = r0[0];
          float v54_data = r0[1];
          float v55_data = r0[2];
          float v56_data = r0[3];
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v52_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 2, 0, 0);
          float v61_data = r0[4];
          float v62_data = r0[5];
          float v63_data = r0[6];
          float v64_data = r0[7];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v60_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 1, 0);
          float v69_data = r0[8];
          float v70_data = r0[9];
          float v71_data = r0[10];
          float v72_data = r0[11];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v68_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 2, 0);
          float v77_data = r0[12];
          float v78_data = r0[13];
          float v79_data = r0[14];
          float v80_data = r0[15];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v76_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v83_acc, 2, 3, 0);
          r2[0] = (v84_acc[0]);
          r2[1] = (v84_acc[1]);
          r2[2] = (v84_acc[2]);
          r2[3] = (v84_acc[3]);
          float v89_data = r1[4];
          float v90_data = r1[5];
          float v91_data = r1[6];
          float v92_data = r1[7];
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          tensorforge::transpose4x4b32(v93_tp, v94_tp, v95_tp, v96_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v97_acc{};
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v55_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v56_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v110_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v118_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v121_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v126_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v127_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v128_acc, 2, 3, 0);
          r2[4] = (v129_acc[0]);
          r2[5] = (v129_acc[1]);
          r2[6] = (v129_acc[2]);
          r2[7] = (v129_acc[3]);
          float v134_data = r1[8];
          float v135_data = r1[9];
          float v136_data = r1[10];
          float v137_data = r1[11];
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          tensorforge::transpose4x4b32(v138_tp, v139_tp, v140_tp, v141_tp, v134_data, v135_data, v136_data, v137_data);
          tensorforge::VectorT<float, 4> v142_acc{};
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v53_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v54_data, v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v55_data, v148_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v56_data, v149_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v61_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v62_data, v155_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v63_data, v156_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v64_data, v157_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v158_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v163_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v164_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v165_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v166_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v171_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v172_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v173_acc, 2, 3, 0);
          r2[8] = (v174_acc[0]);
          r2[9] = (v174_acc[1]);
          r2[10] = (v174_acc[2]);
          r2[11] = (v174_acc[3]);
          float v179_data = r1[12];
          float v180_data = r1[13];
          float v181_data = r1[14];
          float v182_data = r1[15];
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          tensorforge::transpose4x4b32(v183_tp, v184_tp, v185_tp, v186_tp, v179_data, v180_data, v181_data, v182_data);
          tensorforge::VectorT<float, 4> v187_acc{};
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v53_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v54_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v55_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v56_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v61_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v62_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v63_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v64_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v69_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v70_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v71_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v72_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v77_data, v211_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v78_data, v216_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v79_data, v217_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v80_data, v218_acc, 2, 3, 0);
          r2[12] = (v219_acc[0]);
          r2[13] = (v219_acc[1]);
          r2[14] = (v219_acc[2]);
          r2[15] = (v219_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v227_i0 = 0; v227_i0 < 1; ++v227_i0) {
            int32_t v236_lead = v7_lead + (v227_i0 * 16);
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 16; ++v228_i1) {
              int32_t v229_a = v227_i0 + v228_i1;
              float v231_data = r2[(v227_i0 + v228_i1)];
              glb_m0[(v236_lead + (v228_i1 * 16))] = v231_data;
            }
          }
        }
      }
    }
  }
}

