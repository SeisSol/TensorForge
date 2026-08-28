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
          int32_t v9_lead = threadIdx.x % 32;
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
              int32_t v17_a = v11_i1 * 16;
              int32_t v18_a = v9_lead + v17_a;
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v9_lead + v17_a)]);
              r0[v11_i1] = v26_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v29_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          float v30_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v30_lin;
          float v31_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v31_lin;
          float v32_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v32_lin;
          float v33_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v33_lin;
          float v34_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v34_lin;
          float v35_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v35_lin;
          float v36_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v36_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v38_data = r1[0];
          float v39_data = r1[1];
          float v40_data = r1[2];
          float v41_data = r1[3];
          float v42_tp{};
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          tensorforge::transpose4x4b32(v42_tp, v43_tp, v44_tp, v45_tp, v38_data, v39_data, v40_data, v41_data);
          tensorforge::VectorT<float, 4> v46_acc{};
          float v47_data = r0[0];
          float v48_data = r0[1];
          float v49_data = r0[2];
          float v50_data = r0[3];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v46_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v51_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 3, 0, 0);
          float v55_data = r0[4];
          float v56_data = r0[5];
          float v57_data = r0[6];
          float v58_data = r0[7];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v54_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v59_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 3, 1, 0);
          float v63_data = r0[8];
          float v64_data = r0[9];
          float v65_data = r0[10];
          float v66_data = r0[11];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v62_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v67_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 3, 2, 0);
          float v71_data = r0[12];
          float v72_data = r0[13];
          float v73_data = r0[14];
          float v74_data = r0[15];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v70_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v75_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 3, 3, 0);
          r2[0] = (v78_acc[0]);
          r2[1] = (v78_acc[1]);
          r2[2] = (v78_acc[2]);
          r2[3] = (v78_acc[3]);
          float v83_data = r1[4];
          float v84_data = r1[5];
          float v85_data = r1[6];
          float v86_data = r1[7];
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          tensorforge::transpose4x4b32(v87_tp, v88_tp, v89_tp, v90_tp, v83_data, v84_data, v85_data, v86_data);
          tensorforge::VectorT<float, 4> v91_acc{};
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v47_data, v91_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v48_data, v96_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v99_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v104_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v107_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v112_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v113_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v114_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v115_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v120_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v121_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v122_acc, 3, 3, 0);
          r2[4] = (v123_acc[0]);
          r2[5] = (v123_acc[1]);
          r2[6] = (v123_acc[2]);
          r2[7] = (v123_acc[3]);
          float v128_data = r1[8];
          float v129_data = r1[9];
          float v130_data = r1[10];
          float v131_data = r1[11];
          float v132_tp{};
          float v133_tp{};
          float v134_tp{};
          float v135_tp{};
          tensorforge::transpose4x4b32(v132_tp, v133_tp, v134_tp, v135_tp, v128_data, v129_data, v130_data, v131_data);
          tensorforge::VectorT<float, 4> v136_acc{};
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v47_data, v136_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v48_data, v141_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v49_data, v142_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v50_data, v143_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v55_data, v144_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v56_data, v149_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v57_data, v150_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v58_data, v151_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v152_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v157_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v158_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v159_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v160_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v165_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v166_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v167_acc, 3, 3, 0);
          r2[8] = (v168_acc[0]);
          r2[9] = (v168_acc[1]);
          r2[10] = (v168_acc[2]);
          r2[11] = (v168_acc[3]);
          float v173_data = r1[12];
          float v174_data = r1[13];
          float v175_data = r1[14];
          float v176_data = r1[15];
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          tensorforge::transpose4x4b32(v177_tp, v178_tp, v179_tp, v180_tp, v173_data, v174_data, v175_data, v176_data);
          tensorforge::VectorT<float, 4> v181_acc{};
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v47_data, v181_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v48_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v49_data, v187_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v50_data, v188_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v55_data, v189_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v56_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v57_data, v195_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v58_data, v196_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v63_data, v197_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v64_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v65_data, v203_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v66_data, v204_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v71_data, v205_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v72_data, v210_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v73_data, v211_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v74_data, v212_acc, 3, 3, 0);
          r2[12] = (v213_acc[0]);
          r2[13] = (v213_acc[1]);
          r2[14] = (v213_acc[2]);
          r2[15] = (v213_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v222_i1 = 0; v222_i1 < 16; ++v222_i1) {
              int32_t v223_a = 0 + v222_i1;
              float v225_data = r2[v222_i1];
              glb_m0[(v9_lead + (v222_i1 * 16))] = v225_data;
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
          int32_t v242_lead = threadIdx.x % 32;
          if (v242_lead < 16) {
            #pragma unroll
            for (int32_t v244_i1 = 0; v244_i1 < 16; ++v244_i1) {
              int32_t v250_a = v244_i1 * 16;
              int32_t v251_a = v242_lead + v250_a;
              float v259_data = __builtin_nontemporal_load(&glb_m0[(v242_lead + v250_a)]);
              r0[v244_i1] = v259_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          float v262_lin = glb_m4[0 + threadIdx.x * 1];
          r1[0] = v262_lin;
          float v263_lin = glb_m4[32 + threadIdx.x * 1];
          r1[1] = v263_lin;
          float v264_lin = glb_m4[64 + threadIdx.x * 1];
          r1[2] = v264_lin;
          float v265_lin = glb_m4[96 + threadIdx.x * 1];
          r1[3] = v265_lin;
          float v266_lin = glb_m4[128 + threadIdx.x * 1];
          r1[4] = v266_lin;
          float v267_lin = glb_m4[160 + threadIdx.x * 1];
          r1[5] = v267_lin;
          float v268_lin = glb_m4[192 + threadIdx.x * 1];
          r1[6] = v268_lin;
          float v269_lin = glb_m4[224 + threadIdx.x * 1];
          r1[7] = v269_lin;
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v271_data = r1[0];
          float v272_data = r1[1];
          float v273_data = r1[2];
          float v274_data = r1[3];
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          float v278_tp{};
          tensorforge::transpose4x4b32(v275_tp, v276_tp, v277_tp, v278_tp, v271_data, v272_data, v273_data, v274_data);
          tensorforge::VectorT<float, 4> v279_acc{};
          float v280_data = r0[0];
          float v281_data = r0[1];
          float v282_data = r0[2];
          float v283_data = r0[3];
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v280_data, v279_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v281_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v282_data, v285_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v283_data, v286_acc, 3, 0, 0);
          float v288_data = r0[4];
          float v289_data = r0[5];
          float v290_data = r0[6];
          float v291_data = r0[7];
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v288_data, v287_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v289_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v290_data, v293_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v291_data, v294_acc, 3, 1, 0);
          float v296_data = r0[8];
          float v297_data = r0[9];
          float v298_data = r0[10];
          float v299_data = r0[11];
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v296_data, v295_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v297_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v298_data, v301_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v299_data, v302_acc, 3, 2, 0);
          float v304_data = r0[12];
          float v305_data = r0[13];
          float v306_data = r0[14];
          float v307_data = r0[15];
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v304_data, v303_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v305_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v306_data, v309_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v307_data, v310_acc, 3, 3, 0);
          r2[0] = (v311_acc[0]);
          r2[1] = (v311_acc[1]);
          r2[2] = (v311_acc[2]);
          r2[3] = (v311_acc[3]);
          float v316_data = r1[4];
          float v317_data = r1[5];
          float v318_data = r1[6];
          float v319_data = r1[7];
          float v320_tp{};
          float v321_tp{};
          float v322_tp{};
          float v323_tp{};
          tensorforge::transpose4x4b32(v320_tp, v321_tp, v322_tp, v323_tp, v316_data, v317_data, v318_data, v319_data);
          tensorforge::VectorT<float, 4> v324_acc{};
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v280_data, v324_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v281_data, v329_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v282_data, v330_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v283_data, v331_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v288_data, v332_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v289_data, v337_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v290_data, v338_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v291_data, v339_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v296_data, v340_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v297_data, v345_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v298_data, v346_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v299_data, v347_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v304_data, v348_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v305_data, v353_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v306_data, v354_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v307_data, v355_acc, 3, 3, 0);
          r2[4] = (v356_acc[0]);
          r2[5] = (v356_acc[1]);
          r2[6] = (v356_acc[2]);
          r2[7] = (v356_acc[3]);
          float v361_data = r1[8];
          float v362_data = r1[9];
          float v363_data = r1[10];
          float v364_data = r1[11];
          float v365_tp{};
          float v366_tp{};
          float v367_tp{};
          float v368_tp{};
          tensorforge::transpose4x4b32(v365_tp, v366_tp, v367_tp, v368_tp, v361_data, v362_data, v363_data, v364_data);
          tensorforge::VectorT<float, 4> v369_acc{};
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v280_data, v369_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v281_data, v374_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v282_data, v375_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v283_data, v376_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v288_data, v377_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v289_data, v382_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v290_data, v383_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v291_data, v384_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v296_data, v385_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v297_data, v390_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v298_data, v391_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v299_data, v392_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v304_data, v393_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v305_data, v398_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v306_data, v399_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v307_data, v400_acc, 3, 3, 0);
          r2[8] = (v401_acc[0]);
          r2[9] = (v401_acc[1]);
          r2[10] = (v401_acc[2]);
          r2[11] = (v401_acc[3]);
          float v406_data = r1[12];
          float v407_data = r1[13];
          float v408_data = r1[14];
          float v409_data = r1[15];
          float v410_tp{};
          float v411_tp{};
          float v412_tp{};
          float v413_tp{};
          tensorforge::transpose4x4b32(v410_tp, v411_tp, v412_tp, v413_tp, v406_data, v407_data, v408_data, v409_data);
          tensorforge::VectorT<float, 4> v414_acc{};
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v280_data, v414_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v281_data, v419_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v282_data, v420_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v283_data, v421_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v288_data, v422_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v289_data, v427_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v290_data, v428_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v291_data, v429_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v296_data, v430_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v297_data, v435_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v298_data, v436_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v299_data, v437_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v304_data, v438_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v305_data, v443_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v306_data, v444_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v307_data, v445_acc, 3, 3, 0);
          r2[12] = (v446_acc[0]);
          r2[13] = (v446_acc[1]);
          r2[14] = (v446_acc[2]);
          r2[15] = (v446_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v242_lead < 16) {
            #pragma unroll
            for (int32_t v455_i1 = 0; v455_i1 < 16; ++v455_i1) {
              int32_t v456_a = 0 + v455_i1;
              float v458_data = r2[v455_i1];
              glb_m3[(v242_lead + (v455_i1 * 16))] = v458_data;
            }
          }
        }
      }
    }
  }
}

