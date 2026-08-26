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
          int32_t v3_lead = threadIdx.x % 32;
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 16;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v28_lin;
          float v29_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v29_lin;
          float v30_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v30_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v32_data = r1[0];
          float v33_data = r1[1];
          float v34_data = r1[2];
          float v35_data = r1[3];
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
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v41_data, v40_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v42_data, v45_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v43_data, v46_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v44_data, v47_acc, 3, 0, 0);
          float v49_data = r0[4];
          float v50_data = r0[5];
          float v51_data = r0[6];
          float v52_data = r0[7];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v49_data, v48_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v50_data, v53_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v51_data, v54_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v52_data, v55_acc, 3, 1, 0);
          float v57_data = r0[8];
          float v58_data = r0[9];
          float v59_data = r0[10];
          float v60_data = r0[11];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v57_data, v56_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v58_data, v61_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v59_data, v62_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v60_data, v63_acc, 3, 2, 0);
          float v65_data = r0[12];
          float v66_data = r0[13];
          float v67_data = r0[14];
          float v68_data = r0[15];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v65_data, v64_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v66_data, v69_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v70_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v68_data, v71_acc, 3, 3, 0);
          ir2[0] = (v72_acc[0]);
          ir2[1] = (v72_acc[1]);
          ir2[2] = (v72_acc[2]);
          ir2[3] = (v72_acc[3]);
          float v77_data = r1[4];
          float v78_data = r1[5];
          float v79_data = r1[6];
          float v80_data = r1[7];
          float v81_tp{};
          float v82_tp{};
          float v83_tp{};
          float v84_tp{};
          tensorforge::transpose4x4b32(v81_tp, v82_tp, v83_tp, v84_tp, v77_data, v78_data, v79_data, v80_data);
          tensorforge::VectorT<float, 4> v85_acc{};
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v41_data, v85_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v42_data, v90_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v43_data, v91_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v44_data, v92_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v49_data, v93_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v50_data, v98_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v99_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v100_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v57_data, v101_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v106_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v107_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v108_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v65_data, v109_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v66_data, v114_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v115_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v116_acc, 3, 3, 0);
          ir2[4] = (v117_acc[0]);
          ir2[5] = (v117_acc[1]);
          ir2[6] = (v117_acc[2]);
          ir2[7] = (v117_acc[3]);
          float v122_data = r1[8];
          float v123_data = r1[9];
          float v124_data = r1[10];
          float v125_data = r1[11];
          float v126_tp{};
          float v127_tp{};
          float v128_tp{};
          float v129_tp{};
          tensorforge::transpose4x4b32(v126_tp, v127_tp, v128_tp, v129_tp, v122_data, v123_data, v124_data, v125_data);
          tensorforge::VectorT<float, 4> v130_acc{};
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v41_data, v130_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v42_data, v135_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v43_data, v136_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v44_data, v137_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v49_data, v138_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v50_data, v143_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v51_data, v144_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v52_data, v145_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v57_data, v146_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v151_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v152_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v153_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v65_data, v154_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v159_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v160_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v161_acc, 3, 3, 0);
          ir2[8] = (v162_acc[0]);
          ir2[9] = (v162_acc[1]);
          ir2[10] = (v162_acc[2]);
          ir2[11] = (v162_acc[3]);
          float v167_data = r1[12];
          float v168_data = r1[13];
          float v169_data = r1[14];
          float v170_data = r1[15];
          float v171_tp{};
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          tensorforge::transpose4x4b32(v171_tp, v172_tp, v173_tp, v174_tp, v167_data, v168_data, v169_data, v170_data);
          tensorforge::VectorT<float, 4> v175_acc{};
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v41_data, v175_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v42_data, v180_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v43_data, v181_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v44_data, v182_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v49_data, v183_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v50_data, v188_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v51_data, v189_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v52_data, v190_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v57_data, v191_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v58_data, v196_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v59_data, v197_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v60_data, v198_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v65_data, v199_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v66_data, v204_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v67_data, v205_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v68_data, v206_acc, 3, 3, 0);
          ir2[12] = (v207_acc[0]);
          ir2[13] = (v207_acc[1]);
          ir2[14] = (v207_acc[2]);
          ir2[15] = (v207_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v216_i1 = 0; v216_i1 < 16; ++v216_i1) {
              int32_t v217_a = 0 + v216_i1;
              float v219_data = r2[v216_i1];
              int32_t v226_a = v3_lead + (v216_i1 * 16);
              glb_m0[v226_a] = v219_data;
            }
          }
          ;
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
          int32_t v230_lead = threadIdx.x % 32;
          if (v230_lead < 16) {
            #pragma unroll
            for (int32_t v232_i1 = 0; v232_i1 < 16; ++v232_i1) {
              int32_t v238_a = v232_i1 * 16;
              int32_t v239_a = v230_lead + v238_a;
              float v247_data = __builtin_nontemporal_load(&glb_m0[(v230_lead + v238_a)]);
              int32_t v248_a = 0 + v232_i1;
              r0[v248_a] = v247_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          float v250_lin = glb_m4[0 + threadIdx.x * 1];
          r1[0] = v250_lin;
          float v251_lin = glb_m4[32 + threadIdx.x * 1];
          r1[1] = v251_lin;
          float v252_lin = glb_m4[64 + threadIdx.x * 1];
          r1[2] = v252_lin;
          float v253_lin = glb_m4[96 + threadIdx.x * 1];
          r1[3] = v253_lin;
          float v254_lin = glb_m4[128 + threadIdx.x * 1];
          r1[4] = v254_lin;
          float v255_lin = glb_m4[160 + threadIdx.x * 1];
          r1[5] = v255_lin;
          float v256_lin = glb_m4[192 + threadIdx.x * 1];
          r1[6] = v256_lin;
          float v257_lin = glb_m4[224 + threadIdx.x * 1];
          r1[7] = v257_lin;
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v259_data = r1[0];
          float v260_data = r1[1];
          float v261_data = r1[2];
          float v262_data = r1[3];
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          tensorforge::transpose4x4b32(v263_tp, v264_tp, v265_tp, v266_tp, v259_data, v260_data, v261_data, v262_data);
          tensorforge::VectorT<float, 4> v267_acc{};
          float v268_data = r0[0];
          float v269_data = r0[1];
          float v270_data = r0[2];
          float v271_data = r0[3];
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v268_data, v267_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v269_data, v272_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v270_data, v273_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v271_data, v274_acc, 3, 0, 0);
          float v276_data = r0[4];
          float v277_data = r0[5];
          float v278_data = r0[6];
          float v279_data = r0[7];
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v276_data, v275_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v277_data, v280_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v278_data, v281_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v279_data, v282_acc, 3, 1, 0);
          float v284_data = r0[8];
          float v285_data = r0[9];
          float v286_data = r0[10];
          float v287_data = r0[11];
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v284_data, v283_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v285_data, v288_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v286_data, v289_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v287_data, v290_acc, 3, 2, 0);
          float v292_data = r0[12];
          float v293_data = r0[13];
          float v294_data = r0[14];
          float v295_data = r0[15];
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v292_data, v291_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v293_data, v296_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v294_data, v297_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v295_data, v298_acc, 3, 3, 0);
          ir2[0] = (v299_acc[0]);
          ir2[1] = (v299_acc[1]);
          ir2[2] = (v299_acc[2]);
          ir2[3] = (v299_acc[3]);
          float v304_data = r1[4];
          float v305_data = r1[5];
          float v306_data = r1[6];
          float v307_data = r1[7];
          float v308_tp{};
          float v309_tp{};
          float v310_tp{};
          float v311_tp{};
          tensorforge::transpose4x4b32(v308_tp, v309_tp, v310_tp, v311_tp, v304_data, v305_data, v306_data, v307_data);
          tensorforge::VectorT<float, 4> v312_acc{};
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v268_data, v312_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v269_data, v317_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v270_data, v318_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v271_data, v319_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v276_data, v320_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v277_data, v325_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v278_data, v326_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v279_data, v327_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v284_data, v328_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v285_data, v333_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v286_data, v334_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v287_data, v335_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v292_data, v336_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v293_data, v341_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v294_data, v342_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v295_data, v343_acc, 3, 3, 0);
          ir2[4] = (v344_acc[0]);
          ir2[5] = (v344_acc[1]);
          ir2[6] = (v344_acc[2]);
          ir2[7] = (v344_acc[3]);
          float v349_data = r1[8];
          float v350_data = r1[9];
          float v351_data = r1[10];
          float v352_data = r1[11];
          float v353_tp{};
          float v354_tp{};
          float v355_tp{};
          float v356_tp{};
          tensorforge::transpose4x4b32(v353_tp, v354_tp, v355_tp, v356_tp, v349_data, v350_data, v351_data, v352_data);
          tensorforge::VectorT<float, 4> v357_acc{};
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v268_data, v357_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v269_data, v362_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v270_data, v363_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v271_data, v364_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v276_data, v365_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v277_data, v370_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v278_data, v371_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v279_data, v372_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v284_data, v373_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v285_data, v378_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v286_data, v379_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v287_data, v380_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v292_data, v381_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v293_data, v386_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v294_data, v387_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v295_data, v388_acc, 3, 3, 0);
          ir2[8] = (v389_acc[0]);
          ir2[9] = (v389_acc[1]);
          ir2[10] = (v389_acc[2]);
          ir2[11] = (v389_acc[3]);
          float v394_data = r1[12];
          float v395_data = r1[13];
          float v396_data = r1[14];
          float v397_data = r1[15];
          float v398_tp{};
          float v399_tp{};
          float v400_tp{};
          float v401_tp{};
          tensorforge::transpose4x4b32(v398_tp, v399_tp, v400_tp, v401_tp, v394_data, v395_data, v396_data, v397_data);
          tensorforge::VectorT<float, 4> v402_acc{};
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v268_data, v402_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v269_data, v407_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v270_data, v408_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v271_data, v409_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v276_data, v410_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v277_data, v415_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v278_data, v416_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v279_data, v417_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v284_data, v418_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v285_data, v423_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v286_data, v424_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v287_data, v425_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v292_data, v426_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v293_data, v431_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v294_data, v432_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v295_data, v433_acc, 3, 3, 0);
          ir2[12] = (v434_acc[0]);
          ir2[13] = (v434_acc[1]);
          ir2[14] = (v434_acc[2]);
          ir2[15] = (v434_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v230_lead < 16) {
            #pragma unroll
            for (int32_t v443_i1 = 0; v443_i1 < 16; ++v443_i1) {
              int32_t v444_a = 0 + v443_i1;
              float v446_data = r2[v443_i1];
              int32_t v453_a = v230_lead + (v443_i1 * 16);
              glb_m3[v453_a] = v446_data;
            }
          }
          ;
        }
      }
    }
  }
}

