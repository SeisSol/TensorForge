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
          int32_t v2_lead = threadIdx.x % 32;
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 16;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          tensorforge::VectorT<float, 4> v34_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v30_data, v29_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v35_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v31_data, v34_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v36_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v32_data, v35_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v33_data, v36_acc, 3, 0, 0);
          float v38_data = r0[4];
          float v39_data = r0[5];
          float v40_data = r0[6];
          float v41_data = r0[7];
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v38_data, v37_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v39_data, v42_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v40_data, v43_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v41_data, v44_acc, 3, 1, 0);
          float v46_data = r0[8];
          float v47_data = r0[9];
          float v48_data = r0[10];
          float v49_data = r0[11];
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v46_data, v45_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v47_data, v50_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v48_data, v51_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v49_data, v52_acc, 3, 2, 0);
          float v54_data = r0[12];
          float v55_data = r0[13];
          float v56_data = r0[14];
          float v57_data = r0[15];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v54_data, v53_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v55_data, v58_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v56_data, v59_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v60_acc, 3, 3, 0);
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
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v30_data, v74_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v31_data, v79_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v32_data, v80_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v33_data, v81_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v38_data, v82_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v39_data, v87_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v40_data, v88_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v41_data, v89_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v46_data, v90_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v47_data, v95_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v48_data, v96_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v97_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v54_data, v98_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v55_data, v103_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v56_data, v104_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v105_acc, 3, 3, 0);
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
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v30_data, v119_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v31_data, v124_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v32_data, v125_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v33_data, v126_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v38_data, v127_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v39_data, v132_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v40_data, v133_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v41_data, v134_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v46_data, v135_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v47_data, v140_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v48_data, v141_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v49_data, v142_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v54_data, v143_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v55_data, v148_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v56_data, v149_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v57_data, v150_acc, 3, 3, 0);
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
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v30_data, v164_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v31_data, v169_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v32_data, v170_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v33_data, v171_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v38_data, v172_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v39_data, v177_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v40_data, v178_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v41_data, v179_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v46_data, v180_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v47_data, v185_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v48_data, v186_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v49_data, v187_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v54_data, v188_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v55_data, v193_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v56_data, v194_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v57_data, v195_acc, 3, 3, 0);
          ir2[12] = (v196_acc[0]);
          ir2[13] = (v196_acc[1]);
          ir2[14] = (v196_acc[2]);
          ir2[15] = (v196_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v205_i1 = 0; v205_i1 < 16; ++v205_i1) {
              int32_t v206_a = 0 + v205_i1;
              float v208_data = r2[v205_i1];
              int32_t v215_a = v2_lead + (v205_i1 * 16);
              glb_m0[v215_a] = v208_data;
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
          int32_t v218_lead = threadIdx.x % 32;
          if (v218_lead < 16) {
            #pragma unroll
            for (int32_t v220_i1 = 0; v220_i1 < 16; ++v220_i1) {
              int32_t v226_a = v220_i1 * 16;
              int32_t v227_a = v218_lead + v226_a;
              float v235_data = __builtin_nontemporal_load(&glb_m0[(v218_lead + v226_a)]);
              int32_t v236_a = 0 + v220_i1;
              r0[v236_a] = v235_data;
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
          float v237_data = r1[0];
          float v238_data = r1[1];
          float v239_data = r1[2];
          float v240_data = r1[3];
          float v241_tp{};
          float v242_tp{};
          float v243_tp{};
          float v244_tp{};
          tensorforge::transpose4x4b32(v241_tp, v242_tp, v243_tp, v244_tp, v237_data, v238_data, v239_data, v240_data);
          tensorforge::VectorT<float, 4> v245_acc{};
          float v246_data = r0[0];
          float v247_data = r0[1];
          float v248_data = r0[2];
          float v249_data = r0[3];
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v246_data, v245_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v247_data, v250_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v248_data, v251_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v249_data, v252_acc, 3, 0, 0);
          float v254_data = r0[4];
          float v255_data = r0[5];
          float v256_data = r0[6];
          float v257_data = r0[7];
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v254_data, v253_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v255_data, v258_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v256_data, v259_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v257_data, v260_acc, 3, 1, 0);
          float v262_data = r0[8];
          float v263_data = r0[9];
          float v264_data = r0[10];
          float v265_data = r0[11];
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v262_data, v261_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v263_data, v266_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v264_data, v267_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v265_data, v268_acc, 3, 2, 0);
          float v270_data = r0[12];
          float v271_data = r0[13];
          float v272_data = r0[14];
          float v273_data = r0[15];
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v270_data, v269_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v271_data, v274_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v272_data, v275_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v273_data, v276_acc, 3, 3, 0);
          ir2[0] = (v277_acc[0]);
          ir2[1] = (v277_acc[1]);
          ir2[2] = (v277_acc[2]);
          ir2[3] = (v277_acc[3]);
          float v282_data = r1[4];
          float v283_data = r1[5];
          float v284_data = r1[6];
          float v285_data = r1[7];
          float v286_tp{};
          float v287_tp{};
          float v288_tp{};
          float v289_tp{};
          tensorforge::transpose4x4b32(v286_tp, v287_tp, v288_tp, v289_tp, v282_data, v283_data, v284_data, v285_data);
          tensorforge::VectorT<float, 4> v290_acc{};
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v246_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v247_data, v295_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v248_data, v296_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v249_data, v297_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v254_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v255_data, v303_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v256_data, v304_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v257_data, v305_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v262_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v263_data, v311_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v264_data, v312_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v265_data, v313_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v270_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v271_data, v319_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v272_data, v320_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v273_data, v321_acc, 3, 3, 0);
          ir2[4] = (v322_acc[0]);
          ir2[5] = (v322_acc[1]);
          ir2[6] = (v322_acc[2]);
          ir2[7] = (v322_acc[3]);
          float v327_data = r1[8];
          float v328_data = r1[9];
          float v329_data = r1[10];
          float v330_data = r1[11];
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          float v334_tp{};
          tensorforge::transpose4x4b32(v331_tp, v332_tp, v333_tp, v334_tp, v327_data, v328_data, v329_data, v330_data);
          tensorforge::VectorT<float, 4> v335_acc{};
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v246_data, v335_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v247_data, v340_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v248_data, v341_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v249_data, v342_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v254_data, v343_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v255_data, v348_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v256_data, v349_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v257_data, v350_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v262_data, v351_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v263_data, v356_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v264_data, v357_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v265_data, v358_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v270_data, v359_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v271_data, v364_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v272_data, v365_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v273_data, v366_acc, 3, 3, 0);
          ir2[8] = (v367_acc[0]);
          ir2[9] = (v367_acc[1]);
          ir2[10] = (v367_acc[2]);
          ir2[11] = (v367_acc[3]);
          float v372_data = r1[12];
          float v373_data = r1[13];
          float v374_data = r1[14];
          float v375_data = r1[15];
          float v376_tp{};
          float v377_tp{};
          float v378_tp{};
          float v379_tp{};
          tensorforge::transpose4x4b32(v376_tp, v377_tp, v378_tp, v379_tp, v372_data, v373_data, v374_data, v375_data);
          tensorforge::VectorT<float, 4> v380_acc{};
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v246_data, v380_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v247_data, v385_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v248_data, v386_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v249_data, v387_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v254_data, v388_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v255_data, v393_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v256_data, v394_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v257_data, v395_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v262_data, v396_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v263_data, v401_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v264_data, v402_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v265_data, v403_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v270_data, v404_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v271_data, v409_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v272_data, v410_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v273_data, v411_acc, 3, 3, 0);
          ir2[12] = (v412_acc[0]);
          ir2[13] = (v412_acc[1]);
          ir2[14] = (v412_acc[2]);
          ir2[15] = (v412_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v218_lead < 16) {
            #pragma unroll
            for (int32_t v421_i1 = 0; v421_i1 < 16; ++v421_i1) {
              int32_t v422_a = 0 + v421_i1;
              float v424_data = r2[v421_i1];
              int32_t v431_a = v218_lead + (v421_i1 * 16);
              glb_m3[v431_a] = v424_data;
            }
          }
          ;
        }
      }
    }
  }
}

