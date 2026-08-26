// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 16;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 16;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float r1[16]{};
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
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir2 = r2;
          float v21_data = r1[0];
          float v22_data = r1[1];
          float v23_data = r1[2];
          float v24_data = r1[3];
          float v25_tp{};
          float v26_tp{};
          float v27_tp{};
          float v28_tp{};
          tensorforge::transpose4x4b32(v25_tp, v26_tp, v27_tp, v28_tp, v21_data, v22_data, v23_data, v24_data);
          tensorforge::VectorT<float, 4> v29_acc{};
          float v30_data = r0[0];
          float v31_data = r0[1];
          float v32_data = r0[2];
          float v33_data = r0[3];
          tensorforge::VectorT<float, 4> v34_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v30_data, v29_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v35_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v31_data, v34_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v36_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v32_data, v35_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v33_data, v36_acc, 2, 0, 0);
          float v38_data = r0[4];
          float v39_data = r0[5];
          float v40_data = r0[6];
          float v41_data = r0[7];
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v38_data, v37_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v39_data, v42_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v40_data, v43_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v41_data, v44_acc, 2, 1, 0);
          float v46_data = r0[8];
          float v47_data = r0[9];
          float v48_data = r0[10];
          float v49_data = r0[11];
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v46_data, v45_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v47_data, v50_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v48_data, v51_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v49_data, v52_acc, 2, 2, 0);
          float v54_data = r0[12];
          float v55_data = r0[13];
          float v56_data = r0[14];
          float v57_data = r0[15];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v54_data, v53_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v55_data, v58_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v56_data, v59_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v60_acc, 2, 3, 0);
          ir2[0] = (v61_acc[0]);
          ir2[1] = (v61_acc[1]);
          ir2[2] = (v61_acc[2]);
          ir2[3] = (v61_acc[3]);
          float v66_data = r1[4];
          float v67_data = r1[5];
          float v68_data = r1[6];
          float v69_data = r1[7];
          float v70_tp{};
          float v71_tp{};
          float v72_tp{};
          float v73_tp{};
          tensorforge::transpose4x4b32(v70_tp, v71_tp, v72_tp, v73_tp, v66_data, v67_data, v68_data, v69_data);
          tensorforge::VectorT<float, 4> v74_acc{};
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v30_data, v74_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v31_data, v79_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v32_data, v80_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v33_data, v81_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v38_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v39_data, v87_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v40_data, v88_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v41_data, v89_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v46_data, v90_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v47_data, v95_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v48_data, v96_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v97_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v54_data, v98_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v55_data, v103_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v56_data, v104_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v105_acc, 2, 3, 0);
          ir2[4] = (v106_acc[0]);
          ir2[5] = (v106_acc[1]);
          ir2[6] = (v106_acc[2]);
          ir2[7] = (v106_acc[3]);
          float v111_data = r1[8];
          float v112_data = r1[9];
          float v113_data = r1[10];
          float v114_data = r1[11];
          float v115_tp{};
          float v116_tp{};
          float v117_tp{};
          float v118_tp{};
          tensorforge::transpose4x4b32(v115_tp, v116_tp, v117_tp, v118_tp, v111_data, v112_data, v113_data, v114_data);
          tensorforge::VectorT<float, 4> v119_acc{};
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v30_data, v119_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v31_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v32_data, v125_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v33_data, v126_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v38_data, v127_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v39_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v40_data, v133_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v41_data, v134_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v46_data, v135_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v47_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v48_data, v141_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v49_data, v142_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v54_data, v143_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v55_data, v148_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v56_data, v149_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v57_data, v150_acc, 2, 3, 0);
          ir2[8] = (v151_acc[0]);
          ir2[9] = (v151_acc[1]);
          ir2[10] = (v151_acc[2]);
          ir2[11] = (v151_acc[3]);
          float v156_data = r1[12];
          float v157_data = r1[13];
          float v158_data = r1[14];
          float v159_data = r1[15];
          float v160_tp{};
          float v161_tp{};
          float v162_tp{};
          float v163_tp{};
          tensorforge::transpose4x4b32(v160_tp, v161_tp, v162_tp, v163_tp, v156_data, v157_data, v158_data, v159_data);
          tensorforge::VectorT<float, 4> v164_acc{};
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v30_data, v164_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v31_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v32_data, v170_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v33_data, v171_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v38_data, v172_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v39_data, v177_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v40_data, v178_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v41_data, v179_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v46_data, v180_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v47_data, v185_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v48_data, v186_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v49_data, v187_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v54_data, v188_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v55_data, v193_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v56_data, v194_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v57_data, v195_acc, 2, 3, 0);
          ir2[12] = (v196_acc[0]);
          ir2[13] = (v196_acc[1]);
          ir2[14] = (v196_acc[2]);
          ir2[15] = (v196_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v203_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v204_i0 = 0; v204_i0 < 1; ++v204_i0) {
            int32_t v213_lead = v203_lead + (v204_i0 * 16);
            #pragma unroll
            for (int32_t v205_i1 = 0; v205_i1 < 16; ++v205_i1) {
              int32_t v206_a = v204_i0 + v205_i1;
              float v208_data = r2[(v204_i0 + v205_i1)];
              int32_t v215_a = v213_lead + (v205_i1 * 16);
              glb_m0[v215_a] = v208_data;
            }
          }
          ;
        }
      }
    }
  }
}

