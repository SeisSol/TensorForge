// === base name ===
kernel_a5ad08b73b

// === header ===
void launcher_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a5ad08b73b, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a5ad08b73b), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_a5ad08b73b, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m3 16×16(16×16) {0..16}×{0..16} strided
    // m4 16×16(16×16) {0..16}×{0..16} strided
    // m5 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    // fence
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m5 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 32;
          if (v13_lead < 16) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v13_lead + (v15_i1 * 16))]);
              r0[v15_i1] = v23_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v26_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v26_lin;
          float v27_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v27_lin;
          float v28_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v28_lin;
          float v29_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v29_lin;
          float v30_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v30_lin;
          float v31_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v31_lin;
          float v32_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v32_lin;
          float v33_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v33_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v35_data = r1[0];
          float v36_data = r1[1];
          float v37_data = r1[2];
          float v38_data = r1[3];
          float v39_tp{};
          float v40_tp{};
          float v41_tp{};
          float v42_tp{};
          tensorforge::transpose4x4b32(v39_tp, v40_tp, v41_tp, v42_tp, v35_data, v36_data, v37_data, v38_data);
          tensorforge::VectorT<float, 4> v43_acc{};
          float v44_data = r0[0];
          float v45_data = r0[1];
          float v46_data = r0[2];
          float v47_data = r0[3];
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v44_data, v43_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v45_data, v48_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v46_data, v49_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v50_acc, 3, 0, 0);
          float v52_data = r0[4];
          float v53_data = r0[5];
          float v54_data = r0[6];
          float v55_data = r0[7];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v52_data, v51_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v53_data, v56_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v54_data, v57_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v58_acc, 3, 1, 0);
          float v60_data = r0[8];
          float v61_data = r0[9];
          float v62_data = r0[10];
          float v63_data = r0[11];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v60_data, v59_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v61_data, v64_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v62_data, v65_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v66_acc, 3, 2, 0);
          float v68_data = r0[12];
          float v69_data = r0[13];
          float v70_data = r0[14];
          float v71_data = r0[15];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v68_data, v67_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v69_data, v72_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v70_data, v73_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v74_acc, 3, 3, 0);
          r2[0] = (v75_acc[0]);
          r2[1] = (v75_acc[1]);
          r2[2] = (v75_acc[2]);
          r2[3] = (v75_acc[3]);
          float v80_data = r1[4];
          float v81_data = r1[5];
          float v82_data = r1[6];
          float v83_data = r1[7];
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          float v87_tp{};
          tensorforge::transpose4x4b32(v84_tp, v85_tp, v86_tp, v87_tp, v80_data, v81_data, v82_data, v83_data);
          tensorforge::VectorT<float, 4> v88_acc{};
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v44_data, v88_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v45_data, v93_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v46_data, v94_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v47_data, v95_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v96_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v101_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v102_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v103_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v104_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v109_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v110_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v111_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v112_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v117_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v118_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v119_acc, 3, 3, 0);
          r2[4] = (v120_acc[0]);
          r2[5] = (v120_acc[1]);
          r2[6] = (v120_acc[2]);
          r2[7] = (v120_acc[3]);
          float v125_data = r1[8];
          float v126_data = r1[9];
          float v127_data = r1[10];
          float v128_data = r1[11];
          float v129_tp{};
          float v130_tp{};
          float v131_tp{};
          float v132_tp{};
          tensorforge::transpose4x4b32(v129_tp, v130_tp, v131_tp, v132_tp, v125_data, v126_data, v127_data, v128_data);
          tensorforge::VectorT<float, 4> v133_acc{};
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v44_data, v133_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v45_data, v138_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v46_data, v139_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v47_data, v140_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v52_data, v141_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v53_data, v146_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v54_data, v147_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v55_data, v148_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v149_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v61_data, v154_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v155_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v156_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v157_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v69_data, v162_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v163_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v164_acc, 3, 3, 0);
          r2[8] = (v165_acc[0]);
          r2[9] = (v165_acc[1]);
          r2[10] = (v165_acc[2]);
          r2[11] = (v165_acc[3]);
          float v170_data = r1[12];
          float v171_data = r1[13];
          float v172_data = r1[14];
          float v173_data = r1[15];
          float v174_tp{};
          float v175_tp{};
          float v176_tp{};
          float v177_tp{};
          tensorforge::transpose4x4b32(v174_tp, v175_tp, v176_tp, v177_tp, v170_data, v171_data, v172_data, v173_data);
          tensorforge::VectorT<float, 4> v178_acc{};
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v44_data, v178_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v45_data, v183_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v46_data, v184_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v47_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v52_data, v186_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v53_data, v191_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v54_data, v192_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v55_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v60_data, v194_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v61_data, v199_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v62_data, v200_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v63_data, v201_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v68_data, v202_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v69_data, v207_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v70_data, v208_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v71_data, v209_acc, 3, 3, 0);
          r2[12] = (v210_acc[0]);
          r2[13] = (v210_acc[1]);
          r2[14] = (v210_acc[2]);
          r2[15] = (v210_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 16) {
            #pragma unroll
            for (int32_t v219_i1 = 0; v219_i1 < 16; ++v219_i1) {
              float v221_data = r2[v219_i1];
              glb_m0[(v13_lead + (v219_i1 * 16))] = v221_data;
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
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m4);
          int32_t v242_lead = threadIdx.x % 32;
          if (v242_lead < 16) {
            #pragma unroll
            for (int32_t v244_i1 = 0; v244_i1 < 16; ++v244_i1) {
              float v252_data = __builtin_nontemporal_load(&glb_m4[(v242_lead + (v244_i1 * 16))]);
              r0[v244_i1] = v252_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m5);
          float v255_lin = glb_m5[0 + threadIdx.x * 1];
          r1[0] = v255_lin;
          float v256_lin = glb_m5[32 + threadIdx.x * 1];
          r1[1] = v256_lin;
          float v257_lin = glb_m5[64 + threadIdx.x * 1];
          r1[2] = v257_lin;
          float v258_lin = glb_m5[96 + threadIdx.x * 1];
          r1[3] = v258_lin;
          float v259_lin = glb_m5[128 + threadIdx.x * 1];
          r1[4] = v259_lin;
          float v260_lin = glb_m5[160 + threadIdx.x * 1];
          r1[5] = v260_lin;
          float v261_lin = glb_m5[192 + threadIdx.x * 1];
          r1[6] = v261_lin;
          float v262_lin = glb_m5[224 + threadIdx.x * 1];
          r1[7] = v262_lin;
          // wait(r0 = load{g>r}(glb_m4););
          // wait(r1 = load{g>r}(glb_m5););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v264_data = r1[0];
          float v265_data = r1[1];
          float v266_data = r1[2];
          float v267_data = r1[3];
          float v268_tp{};
          float v269_tp{};
          float v270_tp{};
          float v271_tp{};
          tensorforge::transpose4x4b32(v268_tp, v269_tp, v270_tp, v271_tp, v264_data, v265_data, v266_data, v267_data);
          tensorforge::VectorT<float, 4> v272_acc{};
          float v273_data = r0[0];
          float v274_data = r0[1];
          float v275_data = r0[2];
          float v276_data = r0[3];
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v273_data, v272_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v274_data, v277_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v275_data, v278_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v276_data, v279_acc, 3, 0, 0);
          float v281_data = r0[4];
          float v282_data = r0[5];
          float v283_data = r0[6];
          float v284_data = r0[7];
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v281_data, v280_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v282_data, v285_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v283_data, v286_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v284_data, v287_acc, 3, 1, 0);
          float v289_data = r0[8];
          float v290_data = r0[9];
          float v291_data = r0[10];
          float v292_data = r0[11];
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v289_data, v288_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v290_data, v293_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v291_data, v294_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v292_data, v295_acc, 3, 2, 0);
          float v297_data = r0[12];
          float v298_data = r0[13];
          float v299_data = r0[14];
          float v300_data = r0[15];
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v297_data, v296_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v298_data, v301_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v299_data, v302_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v300_data, v303_acc, 3, 3, 0);
          r2[0] = (v304_acc[0]);
          r2[1] = (v304_acc[1]);
          r2[2] = (v304_acc[2]);
          r2[3] = (v304_acc[3]);
          float v309_data = r1[4];
          float v310_data = r1[5];
          float v311_data = r1[6];
          float v312_data = r1[7];
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          float v316_tp{};
          tensorforge::transpose4x4b32(v313_tp, v314_tp, v315_tp, v316_tp, v309_data, v310_data, v311_data, v312_data);
          tensorforge::VectorT<float, 4> v317_acc{};
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v273_data, v317_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v274_data, v322_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v275_data, v323_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v276_data, v324_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v281_data, v325_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v282_data, v330_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v283_data, v331_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v284_data, v332_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v289_data, v333_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v290_data, v338_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v291_data, v339_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v292_data, v340_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v297_data, v341_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v298_data, v346_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v299_data, v347_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v300_data, v348_acc, 3, 3, 0);
          r2[4] = (v349_acc[0]);
          r2[5] = (v349_acc[1]);
          r2[6] = (v349_acc[2]);
          r2[7] = (v349_acc[3]);
          float v354_data = r1[8];
          float v355_data = r1[9];
          float v356_data = r1[10];
          float v357_data = r1[11];
          float v358_tp{};
          float v359_tp{};
          float v360_tp{};
          float v361_tp{};
          tensorforge::transpose4x4b32(v358_tp, v359_tp, v360_tp, v361_tp, v354_data, v355_data, v356_data, v357_data);
          tensorforge::VectorT<float, 4> v362_acc{};
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v273_data, v362_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v274_data, v367_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v275_data, v368_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v276_data, v369_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v281_data, v370_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v282_data, v375_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v283_data, v376_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v284_data, v377_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v289_data, v378_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v290_data, v383_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v291_data, v384_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v292_data, v385_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v297_data, v386_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v298_data, v391_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v299_data, v392_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v300_data, v393_acc, 3, 3, 0);
          r2[8] = (v394_acc[0]);
          r2[9] = (v394_acc[1]);
          r2[10] = (v394_acc[2]);
          r2[11] = (v394_acc[3]);
          float v399_data = r1[12];
          float v400_data = r1[13];
          float v401_data = r1[14];
          float v402_data = r1[15];
          float v403_tp{};
          float v404_tp{};
          float v405_tp{};
          float v406_tp{};
          tensorforge::transpose4x4b32(v403_tp, v404_tp, v405_tp, v406_tp, v399_data, v400_data, v401_data, v402_data);
          tensorforge::VectorT<float, 4> v407_acc{};
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v273_data, v407_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v274_data, v412_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v275_data, v413_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v276_data, v414_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v281_data, v415_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v282_data, v420_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v283_data, v421_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v284_data, v422_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v289_data, v423_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v290_data, v428_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v291_data, v429_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v292_data, v430_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v297_data, v431_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v298_data, v436_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v299_data, v437_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v300_data, v438_acc, 3, 3, 0);
          r2[12] = (v439_acc[0]);
          r2[13] = (v439_acc[1]);
          r2[14] = (v439_acc[2]);
          r2[15] = (v439_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v242_lead < 16) {
            #pragma unroll
            for (int32_t v448_i1 = 0; v448_i1 < 16; ++v448_i1) {
              float v450_data = r2[v448_i1];
              glb_m3[(v242_lead + (v448_i1 * 16))] = v450_data;
            }
          }
        }
      }
    }
  }
}

