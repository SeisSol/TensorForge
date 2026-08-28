// === base name ===
kernel_9367114bd9

// === header ===
void launcher_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9367114bd9, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (gridsize, 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9367114bd9), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_9367114bd9, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
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
    // barrier
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v12_lead + (v14_i1 * 16))]);
              r0[v14_i1] = v22_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v25_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v25_lin;
          float v26_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v26_lin;
          float v27_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v27_lin;
          float v28_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v28_lin;
          float v29_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v29_lin;
          float v30_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v30_lin;
          float v31_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v31_lin;
          float v32_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v34_data = r1[0];
          float v35_data = r1[1];
          float v36_data = r1[2];
          float v37_data = r1[3];
          float v38_tp{};
          float v39_tp{};
          float v40_tp{};
          float v41_tp{};
          tensorforge::transpose4x4b32(v38_tp, v39_tp, v40_tp, v41_tp, v34_data, v35_data, v36_data, v37_data);
          tensorforge::VectorT<float, 4> v42_acc{};
          float v43_data = r0[0];
          float v44_data = r0[1];
          float v45_data = r0[2];
          float v46_data = r0[3];
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v43_data, v42_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v44_data, v47_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v45_data, v48_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v46_data, v49_acc, 3, 0, 0);
          float v51_data = r0[4];
          float v52_data = r0[5];
          float v53_data = r0[6];
          float v54_data = r0[7];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v51_data, v50_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v52_data, v55_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v53_data, v56_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v54_data, v57_acc, 3, 1, 0);
          float v59_data = r0[8];
          float v60_data = r0[9];
          float v61_data = r0[10];
          float v62_data = r0[11];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v59_data, v58_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v60_data, v63_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v61_data, v64_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v62_data, v65_acc, 3, 2, 0);
          float v67_data = r0[12];
          float v68_data = r0[13];
          float v69_data = r0[14];
          float v70_data = r0[15];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v66_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v68_data, v71_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v69_data, v72_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v70_data, v73_acc, 3, 3, 0);
          r2[0] = (v74_acc[0]);
          r2[1] = (v74_acc[1]);
          r2[2] = (v74_acc[2]);
          r2[3] = (v74_acc[3]);
          float v79_data = r1[4];
          float v80_data = r1[5];
          float v81_data = r1[6];
          float v82_data = r1[7];
          float v83_tp{};
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          tensorforge::transpose4x4b32(v83_tp, v84_tp, v85_tp, v86_tp, v79_data, v80_data, v81_data, v82_data);
          tensorforge::VectorT<float, 4> v87_acc{};
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v43_data, v87_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v44_data, v92_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v45_data, v93_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v46_data, v94_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v95_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v100_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v101_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v102_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v103_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v108_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v109_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v110_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v111_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v116_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v117_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v118_acc, 3, 3, 0);
          r2[4] = (v119_acc[0]);
          r2[5] = (v119_acc[1]);
          r2[6] = (v119_acc[2]);
          r2[7] = (v119_acc[3]);
          float v124_data = r1[8];
          float v125_data = r1[9];
          float v126_data = r1[10];
          float v127_data = r1[11];
          float v128_tp{};
          float v129_tp{};
          float v130_tp{};
          float v131_tp{};
          tensorforge::transpose4x4b32(v128_tp, v129_tp, v130_tp, v131_tp, v124_data, v125_data, v126_data, v127_data);
          tensorforge::VectorT<float, 4> v132_acc{};
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v43_data, v132_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v44_data, v137_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v45_data, v138_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v46_data, v139_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v51_data, v140_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v52_data, v145_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v53_data, v146_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v54_data, v147_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v148_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v153_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v61_data, v154_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v155_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v156_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v161_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v69_data, v162_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v163_acc, 3, 3, 0);
          r2[8] = (v164_acc[0]);
          r2[9] = (v164_acc[1]);
          r2[10] = (v164_acc[2]);
          r2[11] = (v164_acc[3]);
          float v169_data = r1[12];
          float v170_data = r1[13];
          float v171_data = r1[14];
          float v172_data = r1[15];
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          float v176_tp{};
          tensorforge::transpose4x4b32(v173_tp, v174_tp, v175_tp, v176_tp, v169_data, v170_data, v171_data, v172_data);
          tensorforge::VectorT<float, 4> v177_acc{};
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v43_data, v177_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v44_data, v182_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v45_data, v183_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v46_data, v184_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v51_data, v185_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v52_data, v190_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v53_data, v191_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v54_data, v192_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v59_data, v193_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v60_data, v198_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v61_data, v199_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v62_data, v200_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v67_data, v201_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v68_data, v206_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v69_data, v207_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v70_data, v208_acc, 3, 3, 0);
          r2[12] = (v209_acc[0]);
          r2[13] = (v209_acc[1]);
          r2[14] = (v209_acc[2]);
          r2[15] = (v209_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v218_i1 = 0; v218_i1 < 16; ++v218_i1) {
              float v220_data = r2[v218_i1];
              glb_m0[(v12_lead + (v218_i1 * 16))] = v220_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      cooperative_groups::this_grid().sync();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements1; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements1 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[batchId0]);
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
              float v250_data = __builtin_nontemporal_load(&glb_m0[(v240_lead + (v242_i1 * 16))]);
              r0[v242_i1] = v250_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          float v253_lin = glb_m4[0 + threadIdx.x * 1];
          r1[0] = v253_lin;
          float v254_lin = glb_m4[32 + threadIdx.x * 1];
          r1[1] = v254_lin;
          float v255_lin = glb_m4[64 + threadIdx.x * 1];
          r1[2] = v255_lin;
          float v256_lin = glb_m4[96 + threadIdx.x * 1];
          r1[3] = v256_lin;
          float v257_lin = glb_m4[128 + threadIdx.x * 1];
          r1[4] = v257_lin;
          float v258_lin = glb_m4[160 + threadIdx.x * 1];
          r1[5] = v258_lin;
          float v259_lin = glb_m4[192 + threadIdx.x * 1];
          r1[6] = v259_lin;
          float v260_lin = glb_m4[224 + threadIdx.x * 1];
          r1[7] = v260_lin;
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v262_data = r1[0];
          float v263_data = r1[1];
          float v264_data = r1[2];
          float v265_data = r1[3];
          float v266_tp{};
          float v267_tp{};
          float v268_tp{};
          float v269_tp{};
          tensorforge::transpose4x4b32(v266_tp, v267_tp, v268_tp, v269_tp, v262_data, v263_data, v264_data, v265_data);
          tensorforge::VectorT<float, 4> v270_acc{};
          float v271_data = r0[0];
          float v272_data = r0[1];
          float v273_data = r0[2];
          float v274_data = r0[3];
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v271_data, v270_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v272_data, v275_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v273_data, v276_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v274_data, v277_acc, 3, 0, 0);
          float v279_data = r0[4];
          float v280_data = r0[5];
          float v281_data = r0[6];
          float v282_data = r0[7];
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v279_data, v278_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v280_data, v283_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v281_data, v284_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v282_data, v285_acc, 3, 1, 0);
          float v287_data = r0[8];
          float v288_data = r0[9];
          float v289_data = r0[10];
          float v290_data = r0[11];
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v287_data, v286_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v288_data, v291_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v289_data, v292_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v290_data, v293_acc, 3, 2, 0);
          float v295_data = r0[12];
          float v296_data = r0[13];
          float v297_data = r0[14];
          float v298_data = r0[15];
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v295_data, v294_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v296_data, v299_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v297_data, v300_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v298_data, v301_acc, 3, 3, 0);
          r2[0] = (v302_acc[0]);
          r2[1] = (v302_acc[1]);
          r2[2] = (v302_acc[2]);
          r2[3] = (v302_acc[3]);
          float v307_data = r1[4];
          float v308_data = r1[5];
          float v309_data = r1[6];
          float v310_data = r1[7];
          float v311_tp{};
          float v312_tp{};
          float v313_tp{};
          float v314_tp{};
          tensorforge::transpose4x4b32(v311_tp, v312_tp, v313_tp, v314_tp, v307_data, v308_data, v309_data, v310_data);
          tensorforge::VectorT<float, 4> v315_acc{};
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v271_data, v315_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v272_data, v320_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v273_data, v321_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v274_data, v322_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v279_data, v323_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v280_data, v328_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v281_data, v329_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v282_data, v330_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v287_data, v331_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v288_data, v336_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v289_data, v337_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v290_data, v338_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v295_data, v339_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v296_data, v344_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v297_data, v345_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v298_data, v346_acc, 3, 3, 0);
          r2[4] = (v347_acc[0]);
          r2[5] = (v347_acc[1]);
          r2[6] = (v347_acc[2]);
          r2[7] = (v347_acc[3]);
          float v352_data = r1[8];
          float v353_data = r1[9];
          float v354_data = r1[10];
          float v355_data = r1[11];
          float v356_tp{};
          float v357_tp{};
          float v358_tp{};
          float v359_tp{};
          tensorforge::transpose4x4b32(v356_tp, v357_tp, v358_tp, v359_tp, v352_data, v353_data, v354_data, v355_data);
          tensorforge::VectorT<float, 4> v360_acc{};
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v271_data, v360_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v272_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v273_data, v366_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v274_data, v367_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v279_data, v368_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v280_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v281_data, v374_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v282_data, v375_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v287_data, v376_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v288_data, v381_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v289_data, v382_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v290_data, v383_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v295_data, v384_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v296_data, v389_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v297_data, v390_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v298_data, v391_acc, 3, 3, 0);
          r2[8] = (v392_acc[0]);
          r2[9] = (v392_acc[1]);
          r2[10] = (v392_acc[2]);
          r2[11] = (v392_acc[3]);
          float v397_data = r1[12];
          float v398_data = r1[13];
          float v399_data = r1[14];
          float v400_data = r1[15];
          float v401_tp{};
          float v402_tp{};
          float v403_tp{};
          float v404_tp{};
          tensorforge::transpose4x4b32(v401_tp, v402_tp, v403_tp, v404_tp, v397_data, v398_data, v399_data, v400_data);
          tensorforge::VectorT<float, 4> v405_acc{};
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v271_data, v405_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v272_data, v410_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v273_data, v411_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v274_data, v412_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v279_data, v413_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v280_data, v418_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v281_data, v419_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v282_data, v420_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v287_data, v421_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v288_data, v426_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v289_data, v427_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v290_data, v428_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v295_data, v429_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v296_data, v434_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v297_data, v435_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v298_data, v436_acc, 3, 3, 0);
          r2[12] = (v437_acc[0]);
          r2[13] = (v437_acc[1]);
          r2[14] = (v437_acc[2]);
          r2[15] = (v437_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v240_lead < 16) {
            #pragma unroll
            for (int32_t v446_i1 = 0; v446_i1 < 16; ++v446_i1) {
              float v448_data = r2[v446_i1];
              glb_m3[(v240_lead + (v446_i1 * 16))] = v448_data;
            }
          }
        }
      }
    }
  }
}

