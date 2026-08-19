// === base name ===
kernel_d08f36e369

// === header ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d08f36e369, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d08f36e369), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_d08f36e369, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 56×18(56×18) {0..56}×{0..18} strided
    // m1 56×18(56×18) {0..56}×{0..18} strided
    // m2 18×18(18×18) {0..18}×{0..18} strided
    // m0 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, 1] = m1 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, -1]×m2 18×18(18×18) {0..18}×{0..18} strided({0..18}×{0..18})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 1008 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1008 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 18; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 56);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 * 2);
              r0[v14_a] = v12_data;
            }
          }
          if (v2_lead < 24) {
            int32_t v21_lead = v2_lead + 32_i32;
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 18; ++v16_i1) {
              int32_t v23_a = v21_lead + (v16_i1 * 56);
              float v24_data;
              {
                v24_data = __builtin_nontemporal_load(&glb_m1[v23_a]);
              }
              int32_t v26_a = 1 + (v16_i1 * 2);
              r0[v26_a] = v24_data;
            }
          }
          float r1[18]{};
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
            float v256 = glb_m2[256 + threadIdx.x * 1];
            r1[8] = v256;
            float v288 = glb_m2[288 + threadIdx.x * 1];
            r1[9] = v288;
            float v320 = glb_m2[320 + threadIdx.x * 1];
            r1[10] = v320;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          auto& ir2 = r2;
          float v27_data = r1[0];
          float v28_data = r1[1];
          float v29_data = r1[2];
          float v30_data = r1[3];
          float v31_tp{};
          float v32_tp{};
          float v33_tp{};
          float v34_tp{};
          tensorforge::transpose4x4b32(v31_tp, v32_tp, v33_tp, v34_tp, v27_data, v28_data, v29_data, v30_data);
          tensorforge::VectorT<float, 4> v35_acc{};
          float v36_data = r0[0];
          float v37_data = r0[2];
          float v38_data = r0[4];
          float v39_data = r0[6];
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v36_data, v35_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v41_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v37_data, v40_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v38_data, v41_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v39_data, v42_acc, 3, 0, 0);
          float v44_data = r0[8];
          float v45_data = r0[10];
          float v46_data = r0[12];
          float v47_data = r0[14];
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v44_data, v43_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v45_data, v48_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v46_data, v49_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v47_data, v50_acc, 3, 1, 0);
          float v52_data = r0[16];
          float v53_data = r0[18];
          float v54_data = r0[20];
          float v55_data = r0[22];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v52_data, v51_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v53_data, v56_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v54_data, v57_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v55_data, v58_acc, 3, 2, 0);
          float v60_data = r0[24];
          float v61_data = r0[26];
          float v62_data = r0[28];
          float v63_data = r0[30];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v59_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v61_data, v64_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v62_data, v65_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v63_data, v66_acc, 3, 3, 0);
          float v68_data = r0[32];
          float v69_data = r0[34];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v68_data, v67_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v69_data, v72_acc, 3, 4, 0);
          ir2[0] = (v73_acc[0]);
          ir2[2] = (v73_acc[1]);
          ir2[4] = (v73_acc[2]);
          ir2[6] = (v73_acc[3]);
          tensorforge::VectorT<float, 4> v78_acc{};
          float v79_data = r0[1];
          float v80_data = r0[3];
          float v81_data = r0[5];
          float v82_data = r0[7];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v79_data, v78_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v80_data, v83_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v81_data, v84_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v82_data, v85_acc, 3, 0, 0);
          float v87_data = r0[9];
          float v88_data = r0[11];
          float v89_data = r0[13];
          float v90_data = r0[15];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v87_data, v86_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v88_data, v91_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v89_data, v92_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v90_data, v93_acc, 3, 1, 0);
          float v95_data = r0[17];
          float v96_data = r0[19];
          float v97_data = r0[21];
          float v98_data = r0[23];
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v95_data, v94_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v96_data, v99_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v97_data, v100_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v98_data, v101_acc, 3, 2, 0);
          float v103_data = r0[25];
          float v104_data = r0[27];
          float v105_data = r0[29];
          float v106_data = r0[31];
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v103_data, v102_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v104_data, v107_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v105_data, v108_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v106_data, v109_acc, 3, 3, 0);
          float v111_data = r0[33];
          float v112_data = r0[35];
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v111_data, v110_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v112_data, v115_acc, 3, 4, 0);
          ir2[1] = (v116_acc[0]);
          ir2[3] = (v116_acc[1]);
          ir2[5] = (v116_acc[2]);
          ir2[7] = (v116_acc[3]);
          float v121_data = r1[4];
          float v122_data = r1[5];
          float v123_data = r1[6];
          float v124_data = r1[7];
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          float v128_tp{};
          tensorforge::transpose4x4b32(v125_tp, v126_tp, v127_tp, v128_tp, v121_data, v122_data, v123_data, v124_data);
          tensorforge::VectorT<float, 4> v129_acc{};
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v36_data, v129_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v37_data, v134_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v38_data, v135_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v39_data, v136_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v44_data, v137_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v45_data, v142_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v46_data, v143_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v47_data, v144_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v52_data, v145_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v53_data, v150_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v54_data, v151_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v55_data, v152_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v60_data, v153_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v61_data, v158_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v62_data, v159_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v63_data, v160_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v68_data, v161_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v69_data, v166_acc, 3, 4, 0);
          ir2[8] = (v167_acc[0]);
          ir2[10] = (v167_acc[1]);
          ir2[12] = (v167_acc[2]);
          ir2[14] = (v167_acc[3]);
          tensorforge::VectorT<float, 4> v172_acc{};
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v79_data, v172_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v80_data, v177_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v81_data, v178_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v82_data, v179_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v87_data, v180_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v88_data, v185_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v89_data, v186_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v90_data, v187_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v95_data, v188_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v96_data, v193_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v97_data, v194_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v98_data, v195_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v103_data, v196_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v104_data, v201_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v105_data, v202_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v106_data, v203_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v111_data, v204_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v112_data, v209_acc, 3, 4, 0);
          ir2[9] = (v210_acc[0]);
          ir2[11] = (v210_acc[1]);
          ir2[13] = (v210_acc[2]);
          ir2[15] = (v210_acc[3]);
          float v215_data = r1[8];
          float v216_data = r1[9];
          float v217_data = r1[10];
          float v218_data = r1[11];
          float v219_tp{};
          float v220_tp{};
          float v221_tp{};
          float v222_tp{};
          tensorforge::transpose4x4b32(v219_tp, v220_tp, v221_tp, v222_tp, v215_data, v216_data, v217_data, v218_data);
          tensorforge::VectorT<float, 4> v223_acc{};
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v36_data, v223_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v37_data, v228_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v38_data, v229_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v39_data, v230_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v44_data, v231_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v45_data, v236_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v46_data, v237_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v47_data, v238_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v52_data, v239_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v53_data, v244_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v54_data, v245_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v55_data, v246_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v60_data, v247_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v61_data, v252_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v62_data, v253_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v63_data, v254_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v68_data, v255_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v69_data, v260_acc, 3, 4, 0);
          ir2[16] = (v261_acc[0]);
          ir2[18] = (v261_acc[1]);
          ir2[20] = (v261_acc[2]);
          ir2[22] = (v261_acc[3]);
          tensorforge::VectorT<float, 4> v266_acc{};
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v79_data, v266_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v80_data, v271_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v81_data, v272_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v82_data, v273_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v87_data, v274_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v88_data, v279_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v89_data, v280_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v90_data, v281_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v95_data, v282_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v96_data, v287_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v97_data, v288_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v98_data, v289_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v103_data, v290_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v104_data, v295_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v105_data, v296_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v106_data, v297_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v111_data, v298_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v112_data, v303_acc, 3, 4, 0);
          ir2[17] = (v304_acc[0]);
          ir2[19] = (v304_acc[1]);
          ir2[21] = (v304_acc[2]);
          ir2[23] = (v304_acc[3]);
          float v309_data = r1[12];
          float v310_data = r1[13];
          float v311_data = r1[14];
          float v312_data = r1[15];
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          float v316_tp{};
          tensorforge::transpose4x4b32(v313_tp, v314_tp, v315_tp, v316_tp, v309_data, v310_data, v311_data, v312_data);
          tensorforge::VectorT<float, 4> v317_acc{};
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v36_data, v317_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v37_data, v322_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v38_data, v323_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v39_data, v324_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v44_data, v325_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v45_data, v330_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v46_data, v331_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v47_data, v332_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v52_data, v333_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v53_data, v338_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v54_data, v339_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v55_data, v340_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v60_data, v341_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v61_data, v346_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v62_data, v347_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v63_data, v348_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v68_data, v349_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v69_data, v354_acc, 3, 4, 0);
          ir2[24] = (v355_acc[0]);
          ir2[26] = (v355_acc[1]);
          ir2[28] = (v355_acc[2]);
          ir2[30] = (v355_acc[3]);
          tensorforge::VectorT<float, 4> v360_acc{};
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v79_data, v360_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v80_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v81_data, v366_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v82_data, v367_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v87_data, v368_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v88_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v89_data, v374_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v90_data, v375_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v95_data, v376_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v96_data, v381_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v97_data, v382_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v98_data, v383_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v103_data, v384_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v104_data, v389_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v105_data, v390_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v106_data, v391_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v111_data, v392_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v112_data, v397_acc, 3, 4, 0);
          ir2[25] = (v398_acc[0]);
          ir2[27] = (v398_acc[1]);
          ir2[29] = (v398_acc[2]);
          ir2[31] = (v398_acc[3]);
          float v403_data = r1[16];
          float v404_data = r1[17];
          float v407_tp{};
          float v408_tp{};
          float v409_tp{};
          float v410_tp{};
          tensorforge::transpose4x4b32(v407_tp, v408_tp, v409_tp, v410_tp, v403_data, v404_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v411_acc{};
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v36_data, v411_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v37_data, v416_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v38_data, v417_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v39_data, v418_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v44_data, v419_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v45_data, v424_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v46_data, v425_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v47_data, v426_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v52_data, v427_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v53_data, v432_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v54_data, v433_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v55_data, v434_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v60_data, v435_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v61_data, v440_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v62_data, v441_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v63_data, v442_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v68_data, v443_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v69_data, v448_acc, 3, 4, 0);
          ir2[32] = (v449_acc[0]);
          ir2[34] = (v449_acc[1]);
          tensorforge::VectorT<float, 4> v452_acc{};
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v79_data, v452_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v80_data, v457_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v81_data, v458_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v82_data, v459_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v87_data, v460_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v88_data, v465_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v89_data, v466_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v90_data, v467_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v95_data, v468_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v96_data, v473_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v97_data, v474_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v98_data, v475_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v103_data, v476_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v104_data, v481_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v105_data, v482_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v106_data, v483_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v111_data, v484_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v112_data, v489_acc, 3, 4, 0);
          ir2[33] = (v490_acc[0]);
          ir2[35] = (v490_acc[1]);
          // glb_m0 = store{r>g}(r2);
          int32_t v495_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v496_i0 = 0; v496_i0 < 1; ++v496_i0) {
            int32_t v505_lead = v495_lead + (v496_i0 * 32);
            #pragma unroll
            for (int32_t v497_i1 = 0; v497_i1 < 18; ++v497_i1) {
              int32_t v499_a = v496_i0 + (v497_i1 * 2);
              float v500_data = r2[v499_a];
              int32_t v507_a = v505_lead + (v497_i1 * 56);
              glb_m0[v507_a] = v500_data;
            }
          }
          if (v495_lead < 24) {
            int32_t v517_lead = v495_lead + 32_i32;
            #pragma unroll
            for (int32_t v509_i1 = 0; v509_i1 < 18; ++v509_i1) {
              int32_t v511_a = 1 + (v509_i1 * 2);
              float v512_data = r2[v511_a];
              int32_t v519_a = v517_lead + (v509_i1 * 56);
              glb_m0[v519_a] = v512_data;
            }
          }
          ;
        }
      }
    }
  }
}

