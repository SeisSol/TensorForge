// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_30948bd44e, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_30948bd44e), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_30948bd44e, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 16;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v18_a = v12_i1 * 16;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + v12_i1)] = v27_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          float v31_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v31_lin;
          float v32_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v32_lin;
          float v33_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v33_lin;
          float v34_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v34_lin;
          float v35_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v35_lin;
          float v36_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v36_lin;
          float v37_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v37_lin;
          float v38_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v38_lin;
          float v39_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v39_lin;
          float v40_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v40_lin;
          float v41_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v41_lin;
          float v42_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v42_lin;
          float v43_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v43_lin;
          float v44_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v44_lin;
          float v45_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v45_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v47_data = r1[0];
          float v48_data = r1[1];
          float v49_data = r1[2];
          float v50_data = r1[3];
          float v51_tp{};
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          tensorforge::transpose4x4b32(v51_tp, v52_tp, v53_tp, v54_tp, v47_data, v48_data, v49_data, v50_data);
          tensorforge::VectorT<float, 4> v55_acc{};
          float v56_data = r0[0];
          float v57_data = r0[1];
          float v58_data = r0[2];
          float v59_data = r0[3];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 2, 0, 0);
          float v64_data = r0[4];
          float v65_data = r0[5];
          float v66_data = r0[6];
          float v67_data = r0[7];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 2, 1, 0);
          float v72_data = r0[8];
          float v73_data = r0[9];
          float v74_data = r0[10];
          float v75_data = r0[11];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v71_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 2, 2, 0);
          float v80_data = r0[12];
          float v81_data = r0[13];
          float v82_data = r0[14];
          float v83_data = r0[15];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v79_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v81_data, v84_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_data, v85_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v83_data, v86_acc, 2, 3, 0);
          r2[0] = (v87_acc[0]);
          r2[1] = (v87_acc[1]);
          r2[2] = (v87_acc[2]);
          r2[3] = (v87_acc[3]);
          float v92_data = r1[4];
          float v93_data = r1[5];
          float v94_data = r1[6];
          float v95_data = r1[7];
          float v96_tp{};
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          tensorforge::transpose4x4b32(v96_tp, v97_tp, v98_tp, v99_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v100_acc{};
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v56_data, v100_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v57_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v58_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v59_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v108_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v116_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v121_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v124_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v129_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v130_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v131_acc, 2, 3, 0);
          r2[4] = (v132_acc[0]);
          r2[5] = (v132_acc[1]);
          r2[6] = (v132_acc[2]);
          r2[7] = (v132_acc[3]);
          float v137_data = r1[8];
          float v138_data = r1[9];
          float v139_data = r1[10];
          float v140_data = r1[11];
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          tensorforge::transpose4x4b32(v141_tp, v142_tp, v143_tp, v144_tp, v137_data, v138_data, v139_data, v140_data);
          tensorforge::VectorT<float, 4> v145_acc{};
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v56_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v57_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v58_data, v151_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v59_data, v152_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v64_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v65_data, v158_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v66_data, v159_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v67_data, v160_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v161_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v73_data, v166_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v74_data, v167_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v75_data, v168_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v169_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v81_data, v174_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v82_data, v175_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v83_data, v176_acc, 2, 3, 0);
          r2[8] = (v177_acc[0]);
          r2[9] = (v177_acc[1]);
          r2[10] = (v177_acc[2]);
          r2[11] = (v177_acc[3]);
          float v182_data = r1[12];
          float v183_data = r1[13];
          float v184_data = r1[14];
          float v185_data = r1[15];
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          tensorforge::transpose4x4b32(v186_tp, v187_tp, v188_tp, v189_tp, v182_data, v183_data, v184_data, v185_data);
          tensorforge::VectorT<float, 4> v190_acc{};
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v56_data, v190_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v57_data, v195_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v58_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v59_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v64_data, v198_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v65_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v66_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v67_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v72_data, v206_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v73_data, v211_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v74_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v75_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v80_data, v214_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v81_data, v219_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v82_data, v220_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v83_data, v221_acc, 2, 3, 0);
          r2[12] = (v222_acc[0]);
          r2[13] = (v222_acc[1]);
          r2[14] = (v222_acc[2]);
          r2[15] = (v222_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v230_i0 = 0; v230_i0 < 1; ++v230_i0) {
            int32_t v239_lead = v10_lead + (v230_i0 * 16);
            #pragma unroll
            for (int32_t v231_i1 = 0; v231_i1 < 16; ++v231_i1) {
              int32_t v232_a = v230_i0 + v231_i1;
              float v234_data = r2[(v230_i0 + v231_i1)];
              glb_m0[(v239_lead + (v231_i1 * 16))] = v234_data;
            }
          }
        }
      }
    }
  }
}

