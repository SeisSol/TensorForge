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
          int32_t v2_lead = threadIdx.x % 32;
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 16);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
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
          float v14_data = r1[0];
          float v15_data = r1[1];
          float v16_data = r1[2];
          float v17_data = r1[3];
          float v18_tp{};
          float v19_tp{};
          float v20_tp{};
          float v21_tp{};
          tensorforge::transpose4x4b32(v18_tp, v19_tp, v20_tp, v21_tp, v14_data, v15_data, v16_data, v17_data);
          tensorforge::VectorT<float, 4> v22_acc{};
          float v23_data = r0[0];
          float v24_data = r0[1];
          float v25_data = r0[2];
          float v26_data = r0[3];
          tensorforge::VectorT<float, 4> v27_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v23_data, v22_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v28_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v24_data, v27_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v29_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v25_data, v28_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v30_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v26_data, v29_acc, 3, 0, 0);
          float v31_data = r0[4];
          float v32_data = r0[5];
          float v33_data = r0[6];
          float v34_data = r0[7];
          tensorforge::VectorT<float, 4> v35_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v31_data, v30_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v36_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v32_data, v35_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v33_data, v36_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v34_data, v37_acc, 3, 1, 0);
          float v39_data = r0[8];
          float v40_data = r0[9];
          float v41_data = r0[10];
          float v42_data = r0[11];
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v39_data, v38_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v40_data, v43_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v41_data, v44_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v42_data, v45_acc, 3, 2, 0);
          float v47_data = r0[12];
          float v48_data = r0[13];
          float v49_data = r0[14];
          float v50_data = r0[15];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v47_data, v46_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v48_data, v51_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v49_data, v52_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v50_data, v53_acc, 3, 3, 0);
          ir2[0] = (v54_acc[0]);
          ir2[1] = (v54_acc[1]);
          ir2[2] = (v54_acc[2]);
          ir2[3] = (v54_acc[3]);
          float v59_data = r1[4];
          float v60_data = r1[5];
          float v61_data = r1[6];
          float v62_data = r1[7];
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v23_data, v67_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v24_data, v72_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v25_data, v73_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v26_data, v74_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v31_data, v75_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v32_data, v80_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v33_data, v81_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v34_data, v82_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v39_data, v83_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v40_data, v88_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v41_data, v89_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v42_data, v90_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v47_data, v91_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v48_data, v96_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v49_data, v97_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v50_data, v98_acc, 3, 3, 0);
          ir2[4] = (v99_acc[0]);
          ir2[5] = (v99_acc[1]);
          ir2[6] = (v99_acc[2]);
          ir2[7] = (v99_acc[3]);
          float v104_data = r1[8];
          float v105_data = r1[9];
          float v106_data = r1[10];
          float v107_data = r1[11];
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          float v111_tp{};
          tensorforge::transpose4x4b32(v108_tp, v109_tp, v110_tp, v111_tp, v104_data, v105_data, v106_data, v107_data);
          tensorforge::VectorT<float, 4> v112_acc{};
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v23_data, v112_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v24_data, v117_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v25_data, v118_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v26_data, v119_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v31_data, v120_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v32_data, v125_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v33_data, v126_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v34_data, v127_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v39_data, v128_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v40_data, v133_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v41_data, v134_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v42_data, v135_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v47_data, v136_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v48_data, v141_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v49_data, v142_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v50_data, v143_acc, 3, 3, 0);
          ir2[8] = (v144_acc[0]);
          ir2[9] = (v144_acc[1]);
          ir2[10] = (v144_acc[2]);
          ir2[11] = (v144_acc[3]);
          float v149_data = r1[12];
          float v150_data = r1[13];
          float v151_data = r1[14];
          float v152_data = r1[15];
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          float v156_tp{};
          tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v149_data, v150_data, v151_data, v152_data);
          tensorforge::VectorT<float, 4> v157_acc{};
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v23_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v24_data, v162_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v25_data, v163_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v26_data, v164_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v31_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v32_data, v170_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v33_data, v171_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v34_data, v172_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v39_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v40_data, v178_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v41_data, v179_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v42_data, v180_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v47_data, v181_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v48_data, v186_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v49_data, v187_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v50_data, v188_acc, 3, 3, 0);
          ir2[12] = (v189_acc[0]);
          ir2[13] = (v189_acc[1]);
          ir2[14] = (v189_acc[2]);
          ir2[15] = (v189_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v196_lead = threadIdx.x % 32;
          if (v196_lead < 16) {
            #pragma unroll
            for (int32_t v198_i1 = 0; v198_i1 < 16; ++v198_i1) {
              int32_t v199_a = 0 + v198_i1;
              float v200_data = r2[v199_a];
              int32_t v207_a = v196_lead + (v198_i1 * 16);
              glb_m0[v207_a] = v200_data;
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
          int32_t v210_lead = threadIdx.x % 32;
          if (v210_lead < 16) {
            #pragma unroll
            for (int32_t v212_i1 = 0; v212_i1 < 16; ++v212_i1) {
              int32_t v219_a = v210_lead + (v212_i1 * 16);
              float v220_data;
              {
                v220_data = __builtin_nontemporal_load(&glb_m0[v219_a]);
              }
              int32_t v221_a = 0 + v212_i1;
              r0[v221_a] = v220_data;
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
          float v222_data = r1[0];
          float v223_data = r1[1];
          float v224_data = r1[2];
          float v225_data = r1[3];
          float v226_tp{};
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          tensorforge::transpose4x4b32(v226_tp, v227_tp, v228_tp, v229_tp, v222_data, v223_data, v224_data, v225_data);
          tensorforge::VectorT<float, 4> v230_acc{};
          float v231_data = r0[0];
          float v232_data = r0[1];
          float v233_data = r0[2];
          float v234_data = r0[3];
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v231_data, v230_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v232_data, v235_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v233_data, v236_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v234_data, v237_acc, 3, 0, 0);
          float v239_data = r0[4];
          float v240_data = r0[5];
          float v241_data = r0[6];
          float v242_data = r0[7];
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v239_data, v238_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v240_data, v243_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v241_data, v244_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v242_data, v245_acc, 3, 1, 0);
          float v247_data = r0[8];
          float v248_data = r0[9];
          float v249_data = r0[10];
          float v250_data = r0[11];
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v247_data, v246_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v248_data, v251_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v249_data, v252_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v250_data, v253_acc, 3, 2, 0);
          float v255_data = r0[12];
          float v256_data = r0[13];
          float v257_data = r0[14];
          float v258_data = r0[15];
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v255_data, v254_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v256_data, v259_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v257_data, v260_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v258_data, v261_acc, 3, 3, 0);
          ir2[0] = (v262_acc[0]);
          ir2[1] = (v262_acc[1]);
          ir2[2] = (v262_acc[2]);
          ir2[3] = (v262_acc[3]);
          float v267_data = r1[4];
          float v268_data = r1[5];
          float v269_data = r1[6];
          float v270_data = r1[7];
          float v271_tp{};
          float v272_tp{};
          float v273_tp{};
          float v274_tp{};
          tensorforge::transpose4x4b32(v271_tp, v272_tp, v273_tp, v274_tp, v267_data, v268_data, v269_data, v270_data);
          tensorforge::VectorT<float, 4> v275_acc{};
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v231_data, v275_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v232_data, v280_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v233_data, v281_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v234_data, v282_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v239_data, v283_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v240_data, v288_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v241_data, v289_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v242_data, v290_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v247_data, v291_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v248_data, v296_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v249_data, v297_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v250_data, v298_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v255_data, v299_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v256_data, v304_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v257_data, v305_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v258_data, v306_acc, 3, 3, 0);
          ir2[4] = (v307_acc[0]);
          ir2[5] = (v307_acc[1]);
          ir2[6] = (v307_acc[2]);
          ir2[7] = (v307_acc[3]);
          float v312_data = r1[8];
          float v313_data = r1[9];
          float v314_data = r1[10];
          float v315_data = r1[11];
          float v316_tp{};
          float v317_tp{};
          float v318_tp{};
          float v319_tp{};
          tensorforge::transpose4x4b32(v316_tp, v317_tp, v318_tp, v319_tp, v312_data, v313_data, v314_data, v315_data);
          tensorforge::VectorT<float, 4> v320_acc{};
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v231_data, v320_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v232_data, v325_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v233_data, v326_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v234_data, v327_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v239_data, v328_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v240_data, v333_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v241_data, v334_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v242_data, v335_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v247_data, v336_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v248_data, v341_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v249_data, v342_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v250_data, v343_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v255_data, v344_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v256_data, v349_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v257_data, v350_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v258_data, v351_acc, 3, 3, 0);
          ir2[8] = (v352_acc[0]);
          ir2[9] = (v352_acc[1]);
          ir2[10] = (v352_acc[2]);
          ir2[11] = (v352_acc[3]);
          float v357_data = r1[12];
          float v358_data = r1[13];
          float v359_data = r1[14];
          float v360_data = r1[15];
          float v361_tp{};
          float v362_tp{};
          float v363_tp{};
          float v364_tp{};
          tensorforge::transpose4x4b32(v361_tp, v362_tp, v363_tp, v364_tp, v357_data, v358_data, v359_data, v360_data);
          tensorforge::VectorT<float, 4> v365_acc{};
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v231_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v232_data, v370_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v233_data, v371_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v234_data, v372_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v239_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v240_data, v378_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v241_data, v379_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v242_data, v380_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v247_data, v381_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v248_data, v386_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v249_data, v387_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v250_data, v388_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v255_data, v389_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v256_data, v394_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v257_data, v395_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v258_data, v396_acc, 3, 3, 0);
          ir2[12] = (v397_acc[0]);
          ir2[13] = (v397_acc[1]);
          ir2[14] = (v397_acc[2]);
          ir2[15] = (v397_acc[3]);
          // glb_m3 = store{r>g}(r2);
          int32_t v404_lead = threadIdx.x % 32;
          if (v404_lead < 16) {
            #pragma unroll
            for (int32_t v406_i1 = 0; v406_i1 < 16; ++v406_i1) {
              int32_t v407_a = 0 + v406_i1;
              float v408_data = r2[v407_a];
              int32_t v415_a = v404_lead + (v406_i1 * 16);
              glb_m3[v415_a] = v408_data;
            }
          }
          ;
        }
      }
    }
  }
}

