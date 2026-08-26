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
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v24_data = r1[0];
          float v25_data = r1[1];
          float v26_data = r1[2];
          float v27_data = r1[3];
          float v28_tp{};
          float v29_tp{};
          float v30_tp{};
          float v31_tp{};
          tensorforge::transpose4x4b32(v28_tp, v29_tp, v30_tp, v31_tp, v24_data, v25_data, v26_data, v27_data);
          tensorforge::VectorT<float, 4> v32_acc{};
          float v33_data = r0[0];
          float v34_data = r0[1];
          float v35_data = r0[2];
          float v36_data = r0[3];
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v33_data, v32_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v34_data, v37_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v39_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v35_data, v38_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v36_data, v39_acc, 3, 0, 0);
          float v41_data = r0[4];
          float v42_data = r0[5];
          float v43_data = r0[6];
          float v44_data = r0[7];
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v41_data, v40_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v42_data, v45_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v43_data, v46_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v44_data, v47_acc, 3, 1, 0);
          float v49_data = r0[8];
          float v50_data = r0[9];
          float v51_data = r0[10];
          float v52_data = r0[11];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v49_data, v48_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v50_data, v53_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v51_data, v54_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v52_data, v55_acc, 3, 2, 0);
          float v57_data = r0[12];
          float v58_data = r0[13];
          float v59_data = r0[14];
          float v60_data = r0[15];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v56_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v58_data, v61_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v59_data, v62_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v63_acc, 3, 3, 0);
          ir2[0] = (v64_acc[0]);
          ir2[1] = (v64_acc[1]);
          ir2[2] = (v64_acc[2]);
          ir2[3] = (v64_acc[3]);
          float v69_data = r1[4];
          float v70_data = r1[5];
          float v71_data = r1[6];
          float v72_data = r1[7];
          float v73_tp{};
          float v74_tp{};
          float v75_tp{};
          float v76_tp{};
          tensorforge::transpose4x4b32(v73_tp, v74_tp, v75_tp, v76_tp, v69_data, v70_data, v71_data, v72_data);
          tensorforge::VectorT<float, 4> v77_acc{};
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v33_data, v77_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v34_data, v82_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v35_data, v83_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v36_data, v84_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v41_data, v85_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v42_data, v90_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v43_data, v91_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v44_data, v92_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v93_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v50_data, v98_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v51_data, v99_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v52_data, v100_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v101_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v58_data, v106_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v59_data, v107_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v60_data, v108_acc, 3, 3, 0);
          ir2[4] = (v109_acc[0]);
          ir2[5] = (v109_acc[1]);
          ir2[6] = (v109_acc[2]);
          ir2[7] = (v109_acc[3]);
          float v114_data = r1[8];
          float v115_data = r1[9];
          float v116_data = r1[10];
          float v117_data = r1[11];
          float v118_tp{};
          float v119_tp{};
          float v120_tp{};
          float v121_tp{};
          tensorforge::transpose4x4b32(v118_tp, v119_tp, v120_tp, v121_tp, v114_data, v115_data, v116_data, v117_data);
          tensorforge::VectorT<float, 4> v122_acc{};
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v33_data, v122_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v34_data, v127_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v35_data, v128_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v36_data, v129_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v41_data, v130_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v42_data, v135_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v43_data, v136_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v44_data, v137_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v49_data, v138_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v50_data, v143_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v51_data, v144_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v52_data, v145_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v57_data, v146_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v58_data, v151_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v59_data, v152_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v60_data, v153_acc, 3, 3, 0);
          ir2[8] = (v154_acc[0]);
          ir2[9] = (v154_acc[1]);
          ir2[10] = (v154_acc[2]);
          ir2[11] = (v154_acc[3]);
          float v159_data = r1[12];
          float v160_data = r1[13];
          float v161_data = r1[14];
          float v162_data = r1[15];
          float v163_tp{};
          float v164_tp{};
          float v165_tp{};
          float v166_tp{};
          tensorforge::transpose4x4b32(v163_tp, v164_tp, v165_tp, v166_tp, v159_data, v160_data, v161_data, v162_data);
          tensorforge::VectorT<float, 4> v167_acc{};
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v33_data, v167_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v34_data, v172_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v35_data, v173_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v36_data, v174_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v41_data, v175_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v42_data, v180_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v43_data, v181_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v44_data, v182_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v49_data, v183_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v50_data, v188_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v51_data, v189_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v52_data, v190_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v57_data, v191_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v58_data, v196_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v59_data, v197_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v60_data, v198_acc, 3, 3, 0);
          ir2[12] = (v199_acc[0]);
          ir2[13] = (v199_acc[1]);
          ir2[14] = (v199_acc[2]);
          ir2[15] = (v199_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v208_i1 = 0; v208_i1 < 16; ++v208_i1) {
              int32_t v209_a = 0 + v208_i1;
              float v211_data = r2[v208_i1];
              int32_t v218_a = v3_lead + (v208_i1 * 16);
              glb_m0[v218_a] = v211_data;
            }
          }
          ;
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
          int32_t v222_lead = threadIdx.x % 32;
          if (v222_lead < 16) {
            #pragma unroll
            for (int32_t v224_i1 = 0; v224_i1 < 16; ++v224_i1) {
              int32_t v230_a = v224_i1 * 16;
              int32_t v231_a = v222_lead + v230_a;
              float v239_data = __builtin_nontemporal_load(&glb_m0[(v222_lead + v230_a)]);
              int32_t v240_a = 0 + v224_i1;
              r0[v240_a] = v239_data;
            }
          }
          float r1[16]{};
          {
            // r1 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r1[2] = v64;
            float v96 = glb_m4[96 + threadIdx.x * 1];
            r1[3] = v96;
            float v128 = glb_m4[128 + threadIdx.x * 1];
            r1[4] = v128;
            float v160 = glb_m4[160 + threadIdx.x * 1];
            r1[5] = v160;
            float v192 = glb_m4[192 + threadIdx.x * 1];
            r1[6] = v192;
            float v224 = glb_m4[224 + threadIdx.x * 1];
            r1[7] = v224;
          }
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v243_data = r1[0];
          float v244_data = r1[1];
          float v245_data = r1[2];
          float v246_data = r1[3];
          float v247_tp{};
          float v248_tp{};
          float v249_tp{};
          float v250_tp{};
          tensorforge::transpose4x4b32(v247_tp, v248_tp, v249_tp, v250_tp, v243_data, v244_data, v245_data, v246_data);
          tensorforge::VectorT<float, 4> v251_acc{};
          float v252_data = r0[0];
          float v253_data = r0[1];
          float v254_data = r0[2];
          float v255_data = r0[3];
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v252_data, v251_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v253_data, v256_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v254_data, v257_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v255_data, v258_acc, 3, 0, 0);
          float v260_data = r0[4];
          float v261_data = r0[5];
          float v262_data = r0[6];
          float v263_data = r0[7];
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v260_data, v259_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v261_data, v264_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v262_data, v265_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v263_data, v266_acc, 3, 1, 0);
          float v268_data = r0[8];
          float v269_data = r0[9];
          float v270_data = r0[10];
          float v271_data = r0[11];
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v268_data, v267_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v269_data, v272_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v270_data, v273_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v271_data, v274_acc, 3, 2, 0);
          float v276_data = r0[12];
          float v277_data = r0[13];
          float v278_data = r0[14];
          float v279_data = r0[15];
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v276_data, v275_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v277_data, v280_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v278_data, v281_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v279_data, v282_acc, 3, 3, 0);
          ir2[0] = (v283_acc[0]);
          ir2[1] = (v283_acc[1]);
          ir2[2] = (v283_acc[2]);
          ir2[3] = (v283_acc[3]);
          float v288_data = r1[4];
          float v289_data = r1[5];
          float v290_data = r1[6];
          float v291_data = r1[7];
          float v292_tp{};
          float v293_tp{};
          float v294_tp{};
          float v295_tp{};
          tensorforge::transpose4x4b32(v292_tp, v293_tp, v294_tp, v295_tp, v288_data, v289_data, v290_data, v291_data);
          tensorforge::VectorT<float, 4> v296_acc{};
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v252_data, v296_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v253_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v254_data, v302_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v255_data, v303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v260_data, v304_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v261_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v262_data, v310_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v263_data, v311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v268_data, v312_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v269_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v270_data, v318_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v271_data, v319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v276_data, v320_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v277_data, v325_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v278_data, v326_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v279_data, v327_acc, 3, 3, 0);
          ir2[4] = (v328_acc[0]);
          ir2[5] = (v328_acc[1]);
          ir2[6] = (v328_acc[2]);
          ir2[7] = (v328_acc[3]);
          float v333_data = r1[8];
          float v334_data = r1[9];
          float v335_data = r1[10];
          float v336_data = r1[11];
          float v337_tp{};
          float v338_tp{};
          float v339_tp{};
          float v340_tp{};
          tensorforge::transpose4x4b32(v337_tp, v338_tp, v339_tp, v340_tp, v333_data, v334_data, v335_data, v336_data);
          tensorforge::VectorT<float, 4> v341_acc{};
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v252_data, v341_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v253_data, v346_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v254_data, v347_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v255_data, v348_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v260_data, v349_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v261_data, v354_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v262_data, v355_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v263_data, v356_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v268_data, v357_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v269_data, v362_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v270_data, v363_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v271_data, v364_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v276_data, v365_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v277_data, v370_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v278_data, v371_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v279_data, v372_acc, 3, 3, 0);
          ir2[8] = (v373_acc[0]);
          ir2[9] = (v373_acc[1]);
          ir2[10] = (v373_acc[2]);
          ir2[11] = (v373_acc[3]);
          float v378_data = r1[12];
          float v379_data = r1[13];
          float v380_data = r1[14];
          float v381_data = r1[15];
          float v382_tp{};
          float v383_tp{};
          float v384_tp{};
          float v385_tp{};
          tensorforge::transpose4x4b32(v382_tp, v383_tp, v384_tp, v385_tp, v378_data, v379_data, v380_data, v381_data);
          tensorforge::VectorT<float, 4> v386_acc{};
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v252_data, v386_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v253_data, v391_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v254_data, v392_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v255_data, v393_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v260_data, v394_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v261_data, v399_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v262_data, v400_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v263_data, v401_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v268_data, v402_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v269_data, v407_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v270_data, v408_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v271_data, v409_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v276_data, v410_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v277_data, v415_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v278_data, v416_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v279_data, v417_acc, 3, 3, 0);
          ir2[12] = (v418_acc[0]);
          ir2[13] = (v418_acc[1]);
          ir2[14] = (v418_acc[2]);
          ir2[15] = (v418_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v222_lead < 16) {
            #pragma unroll
            for (int32_t v427_i1 = 0; v427_i1 < 16; ++v427_i1) {
              int32_t v428_a = 0 + v427_i1;
              float v430_data = r2[v427_i1];
              int32_t v437_a = v222_lead + (v427_i1 * 16);
              glb_m3[v437_a] = v430_data;
            }
          }
          ;
        }
      }
    }
  }
}

