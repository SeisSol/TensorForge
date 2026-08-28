// === base name ===
kernel_f816e7b0ea

// === header ===
void launcher_kernel_f816e7b0ea(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f816e7b0ea(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f816e7b0ea, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_f816e7b0ea), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_f816e7b0ea, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f816e7b0ea(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m3 16×16(16×16) {0..16}×{0..16} strided
    // m4 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    // fence
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v8_lead = threadIdx.x % 32;
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
              int32_t v16_a = v10_i1 * 16;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v8_lead + v16_a)]);
              int32_t v26_a = 0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v28_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          float v29_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v29_lin;
          float v30_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v30_lin;
          float v31_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v31_lin;
          float v32_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v32_lin;
          float v33_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v33_lin;
          float v34_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v34_lin;
          float v35_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v35_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v37_data = r1[0];
          float v38_data = r1[1];
          float v39_data = r1[2];
          float v40_data = r1[3];
          float v41_tp{};
          float v42_tp{};
          float v43_tp{};
          float v44_tp{};
          tensorforge::transpose4x4b32(v41_tp, v42_tp, v43_tp, v44_tp, v37_data, v38_data, v39_data, v40_data);
          tensorforge::VectorT<float, 4> v45_acc{};
          float v46_data = r0[0];
          float v47_data = r0[1];
          float v48_data = r0[2];
          float v49_data = r0[3];
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v46_data, v45_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v50_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v51_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 3, 0, 0);
          float v54_data = r0[4];
          float v55_data = r0[5];
          float v56_data = r0[6];
          float v57_data = r0[7];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v54_data, v53_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v58_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v59_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 3, 1, 0);
          float v62_data = r0[8];
          float v63_data = r0[9];
          float v64_data = r0[10];
          float v65_data = r0[11];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v62_data, v61_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v66_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v67_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 3, 2, 0);
          float v70_data = r0[12];
          float v71_data = r0[13];
          float v72_data = r0[14];
          float v73_data = r0[15];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v70_data, v69_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v74_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v75_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 3, 3, 0);
          r2[0] = (v77_acc[0]);
          r2[1] = (v77_acc[1]);
          r2[2] = (v77_acc[2]);
          r2[3] = (v77_acc[3]);
          float v82_data = r1[4];
          float v83_data = r1[5];
          float v84_data = r1[6];
          float v85_data = r1[7];
          float v86_tp{};
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          tensorforge::transpose4x4b32(v86_tp, v87_tp, v88_tp, v89_tp, v82_data, v83_data, v84_data, v85_data);
          tensorforge::VectorT<float, 4> v90_acc{};
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v46_data, v90_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v47_data, v95_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v48_data, v96_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v98_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v103_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v104_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v106_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v111_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v112_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v113_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v114_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v119_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v120_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v121_acc, 3, 3, 0);
          r2[4] = (v122_acc[0]);
          r2[5] = (v122_acc[1]);
          r2[6] = (v122_acc[2]);
          r2[7] = (v122_acc[3]);
          float v127_data = r1[8];
          float v128_data = r1[9];
          float v129_data = r1[10];
          float v130_data = r1[11];
          float v131_tp{};
          float v132_tp{};
          float v133_tp{};
          float v134_tp{};
          tensorforge::transpose4x4b32(v131_tp, v132_tp, v133_tp, v134_tp, v127_data, v128_data, v129_data, v130_data);
          tensorforge::VectorT<float, 4> v135_acc{};
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v46_data, v135_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v47_data, v140_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v48_data, v141_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v49_data, v142_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v54_data, v143_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v55_data, v148_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v56_data, v149_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v57_data, v150_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v151_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v156_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v157_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v158_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v159_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v164_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v165_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v166_acc, 3, 3, 0);
          r2[8] = (v167_acc[0]);
          r2[9] = (v167_acc[1]);
          r2[10] = (v167_acc[2]);
          r2[11] = (v167_acc[3]);
          float v172_data = r1[12];
          float v173_data = r1[13];
          float v174_data = r1[14];
          float v175_data = r1[15];
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v172_data, v173_data, v174_data, v175_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v46_data, v180_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v47_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v48_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v49_data, v187_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v54_data, v188_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v55_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v56_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v57_data, v195_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v62_data, v196_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v63_data, v201_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v64_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v65_data, v203_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v70_data, v204_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v71_data, v209_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v72_data, v210_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v73_data, v211_acc, 3, 3, 0);
          r2[12] = (v212_acc[0]);
          r2[13] = (v212_acc[1]);
          r2[14] = (v212_acc[2]);
          r2[15] = (v212_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v221_i1 = 0; v221_i1 < 16; ++v221_i1) {
              int32_t v222_a = 0 + v221_i1;
              float v224_data = r2[v221_i1];
              glb_m0[(v8_lead + (v221_i1 * 16))] = v224_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x) + numElements0) % (gridDim.x * blockDim.y);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      __syncthreads();
      for (size_t batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x) + numElements0) % (gridDim.x * blockDim.y); batchId0 < numElements1; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements1 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags1 != nullptr) {
          allowed = static_cast<bool>(flags1[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v240_lead = threadIdx.x % 32;
          if (v240_lead < 16) {
            #pragma unroll
            for (int32_t v242_i1 = 0; v242_i1 < 16; ++v242_i1) {
              int32_t v248_a = v242_i1 * 16;
              int32_t v249_a = v240_lead + v248_a;
              float v257_data = __builtin_nontemporal_load(&glb_m0[(v240_lead + v248_a)]);
              int32_t v258_a = 0 + v242_i1;
              r0[v258_a] = v257_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          float v260_lin = glb_m4[0 + threadIdx.x * 1];
          r1[0] = v260_lin;
          float v261_lin = glb_m4[32 + threadIdx.x * 1];
          r1[1] = v261_lin;
          float v262_lin = glb_m4[64 + threadIdx.x * 1];
          r1[2] = v262_lin;
          float v263_lin = glb_m4[96 + threadIdx.x * 1];
          r1[3] = v263_lin;
          float v264_lin = glb_m4[128 + threadIdx.x * 1];
          r1[4] = v264_lin;
          float v265_lin = glb_m4[160 + threadIdx.x * 1];
          r1[5] = v265_lin;
          float v266_lin = glb_m4[192 + threadIdx.x * 1];
          r1[6] = v266_lin;
          float v267_lin = glb_m4[224 + threadIdx.x * 1];
          r1[7] = v267_lin;
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v269_data = r1[0];
          float v270_data = r1[1];
          float v271_data = r1[2];
          float v272_data = r1[3];
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          float v276_tp{};
          tensorforge::transpose4x4b32(v273_tp, v274_tp, v275_tp, v276_tp, v269_data, v270_data, v271_data, v272_data);
          tensorforge::VectorT<float, 4> v277_acc{};
          float v278_data = r0[0];
          float v279_data = r0[1];
          float v280_data = r0[2];
          float v281_data = r0[3];
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v278_data, v277_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v279_data, v282_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v280_data, v283_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v281_data, v284_acc, 3, 0, 0);
          float v286_data = r0[4];
          float v287_data = r0[5];
          float v288_data = r0[6];
          float v289_data = r0[7];
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v286_data, v285_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v287_data, v290_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v288_data, v291_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v289_data, v292_acc, 3, 1, 0);
          float v294_data = r0[8];
          float v295_data = r0[9];
          float v296_data = r0[10];
          float v297_data = r0[11];
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v294_data, v293_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v295_data, v298_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v296_data, v299_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v297_data, v300_acc, 3, 2, 0);
          float v302_data = r0[12];
          float v303_data = r0[13];
          float v304_data = r0[14];
          float v305_data = r0[15];
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v302_data, v301_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v303_data, v306_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v304_data, v307_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v305_data, v308_acc, 3, 3, 0);
          r2[0] = (v309_acc[0]);
          r2[1] = (v309_acc[1]);
          r2[2] = (v309_acc[2]);
          r2[3] = (v309_acc[3]);
          float v314_data = r1[4];
          float v315_data = r1[5];
          float v316_data = r1[6];
          float v317_data = r1[7];
          float v318_tp{};
          float v319_tp{};
          float v320_tp{};
          float v321_tp{};
          tensorforge::transpose4x4b32(v318_tp, v319_tp, v320_tp, v321_tp, v314_data, v315_data, v316_data, v317_data);
          tensorforge::VectorT<float, 4> v322_acc{};
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v278_data, v322_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v279_data, v327_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v280_data, v328_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v281_data, v329_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v286_data, v330_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v287_data, v335_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v288_data, v336_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v289_data, v337_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v294_data, v338_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v295_data, v343_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v296_data, v344_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v297_data, v345_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v302_data, v346_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v303_data, v351_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v304_data, v352_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v305_data, v353_acc, 3, 3, 0);
          r2[4] = (v354_acc[0]);
          r2[5] = (v354_acc[1]);
          r2[6] = (v354_acc[2]);
          r2[7] = (v354_acc[3]);
          float v359_data = r1[8];
          float v360_data = r1[9];
          float v361_data = r1[10];
          float v362_data = r1[11];
          float v363_tp{};
          float v364_tp{};
          float v365_tp{};
          float v366_tp{};
          tensorforge::transpose4x4b32(v363_tp, v364_tp, v365_tp, v366_tp, v359_data, v360_data, v361_data, v362_data);
          tensorforge::VectorT<float, 4> v367_acc{};
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v278_data, v367_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v279_data, v372_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v280_data, v373_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v281_data, v374_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v286_data, v375_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v287_data, v380_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v288_data, v381_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v289_data, v382_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v294_data, v383_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v295_data, v388_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v296_data, v389_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v297_data, v390_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v302_data, v391_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v303_data, v396_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v304_data, v397_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v305_data, v398_acc, 3, 3, 0);
          r2[8] = (v399_acc[0]);
          r2[9] = (v399_acc[1]);
          r2[10] = (v399_acc[2]);
          r2[11] = (v399_acc[3]);
          float v404_data = r1[12];
          float v405_data = r1[13];
          float v406_data = r1[14];
          float v407_data = r1[15];
          float v408_tp{};
          float v409_tp{};
          float v410_tp{};
          float v411_tp{};
          tensorforge::transpose4x4b32(v408_tp, v409_tp, v410_tp, v411_tp, v404_data, v405_data, v406_data, v407_data);
          tensorforge::VectorT<float, 4> v412_acc{};
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v278_data, v412_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v279_data, v417_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v280_data, v418_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v281_data, v419_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v286_data, v420_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v287_data, v425_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v288_data, v426_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v289_data, v427_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v294_data, v428_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v295_data, v433_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v296_data, v434_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v297_data, v435_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v302_data, v436_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v303_data, v441_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v304_data, v442_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v305_data, v443_acc, 3, 3, 0);
          r2[12] = (v444_acc[0]);
          r2[13] = (v444_acc[1]);
          r2[14] = (v444_acc[2]);
          r2[15] = (v444_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v240_lead < 16) {
            #pragma unroll
            for (int32_t v453_i1 = 0; v453_i1 < 16; ++v453_i1) {
              int32_t v454_a = 0 + v453_i1;
              float v456_data = r2[v453_i1];
              glb_m3[(v240_lead + (v453_i1 * 16))] = v456_data;
            }
          }
        }
      }
    }
  }
}

