// === base name ===
kernel_0f149e12ca19e3eb

// === header ===
void launcher_kernel_0f149e12ca19e3eb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0f149e12ca19e3eb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0f149e12ca19e3eb, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0f149e12ca19e3eb, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0f149e12ca19e3eb), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_0f149e12ca19e3eb, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0f149e12ca19e3eb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×16(32×16) {0..32}×{0..16} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×16(12×16) {0..12}×{0..16} strided
    // m3 32×12(32×12) {0..32}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 32×12(32×12) {0..32}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..16})[0, 1] = m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m3 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m5 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 512 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 192 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 384 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v0_batchId0 * 384 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v0_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 32);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v20_i1 * 32))]);
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 16; ++v35_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m2[(v18_lead + (v35_i1 * 12))]);
              r1[v35_i1] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v49_i0 = 0; v49_i0 < 1; ++v49_i0) {
            int32_t v55_lead = v18_lead + (v49_i0 * 32);
            #pragma unroll
            for (int32_t v50_i1 = 0; v50_i1 < 12; ++v50_i1) {
              float v58_data = __builtin_nontemporal_load(&glb_m3[(v55_lead + (v50_i1 * 32))]);
              r3[(v49_i0 + v50_i1)] = v58_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float v61_data = r1[0];
          float v62_data = r1[1];
          float v63_data = r1[2];
          float v64_data = r1[3];
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, v61_data, v62_data, v63_data, v64_data);
          tensorforge::VectorT<float, 4> v69_acc{};
          float v70_data = r0[0];
          float v71_data = r0[1];
          float v72_data = r0[2];
          float v73_data = r0[3];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v69_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v75_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 3, 0, 0);
          float v78_data = r0[4];
          float v79_data = r0[5];
          float v80_data = r0[6];
          float v81_data = r0[7];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v77_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v83_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 3, 1, 0);
          float v86_data = r0[8];
          float v87_data = r0[9];
          float v88_data = r0[10];
          float v89_data = r0[11];
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v85_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v90_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v91_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v89_data, v92_acc, 3, 2, 0);
          r2[0] = (v93_acc[0]);
          r2[1] = (v93_acc[1]);
          r2[2] = (v93_acc[2]);
          r2[3] = (v93_acc[3]);
          float v98_data = r1[4];
          float v99_data = r1[5];
          float v100_data = r1[6];
          float v101_data = r1[7];
          float v102_tp{};
          float v103_tp{};
          float v104_tp{};
          float v105_tp{};
          tensorforge::transpose4x4b32(v102_tp, v103_tp, v104_tp, v105_tp, v98_data, v99_data, v100_data, v101_data);
          tensorforge::VectorT<float, 4> v106_acc{};
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v70_data, v106_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v71_data, v111_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v72_data, v112_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v73_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v78_data, v114_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v79_data, v119_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v80_data, v120_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v81_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v86_data, v122_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v87_data, v127_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v88_data, v128_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v89_data, v129_acc, 3, 2, 0);
          r2[4] = (v130_acc[0]);
          r2[5] = (v130_acc[1]);
          r2[6] = (v130_acc[2]);
          r2[7] = (v130_acc[3]);
          float v135_data = r1[8];
          float v136_data = r1[9];
          float v137_data = r1[10];
          float v138_data = r1[11];
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          float v142_tp{};
          tensorforge::transpose4x4b32(v139_tp, v140_tp, v141_tp, v142_tp, v135_data, v136_data, v137_data, v138_data);
          tensorforge::VectorT<float, 4> v143_acc{};
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v143_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v73_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v151_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v81_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v86_data, v159_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v87_data, v164_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v88_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v89_data, v166_acc, 3, 2, 0);
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
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v70_data, v180_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v71_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v72_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v73_data, v187_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v78_data, v188_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v79_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v80_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v81_data, v195_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v86_data, v196_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v87_data, v201_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v88_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v89_data, v203_acc, 3, 2, 0);
          r2[12] = (v204_acc[0]);
          r2[13] = (v204_acc[1]);
          r2[14] = (v204_acc[2]);
          r2[15] = (v204_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v212_i0 = 0; v212_i0 < 1; ++v212_i0) {
            int32_t v220_lead = v18_lead + (v212_i0 * 32);
            #pragma unroll
            for (int32_t v213_i1 = 0; v213_i1 < 16; ++v213_i1) {
              float v215_data = r2[(v212_i0 + v213_i1)];
              glb_m0[(v220_lead + (v213_i1 * 32))] = v215_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 8; ++v228_i1) {
              float v236_data = __builtin_nontemporal_load(&glb_m4[(v18_lead + (v228_i1 * 12))]);
              r4[v228_i1] = v236_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v242_i0 = 0; v242_i0 < 1; ++v242_i0) {
            int32_t v248_lead = v18_lead + (v242_i0 * 32);
            #pragma unroll
            for (int32_t v243_i1 = 0; v243_i1 < 12; ++v243_i1) {
              float v251_data = __builtin_nontemporal_load(&glb_m5[(v248_lead + (v243_i1 * 32))]);
              r6[(v242_i0 + v243_i1)] = v251_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v254_data = r4[0];
          float v255_data = r4[1];
          float v256_data = r4[2];
          float v257_data = r4[3];
          float v258_tp{};
          float v259_tp{};
          float v260_tp{};
          float v261_tp{};
          tensorforge::transpose4x4b32(v258_tp, v259_tp, v260_tp, v261_tp, v254_data, v255_data, v256_data, v257_data);
          tensorforge::VectorT<float, 4> v262_acc{};
          float v263_data = r3[0];
          float v264_data = r3[1];
          float v265_data = r3[2];
          float v266_data = r3[3];
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v263_data, v262_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v264_data, v267_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v265_data, v268_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v266_data, v269_acc, 3, 0, 0);
          float v271_data = r3[4];
          float v272_data = r3[5];
          float v273_data = r3[6];
          float v274_data = r3[7];
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v271_data, v270_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v272_data, v275_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v273_data, v276_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v274_data, v277_acc, 3, 1, 0);
          float v279_data = r3[8];
          float v280_data = r3[9];
          float v281_data = r3[10];
          float v282_data = r3[11];
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v279_data, v278_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v280_data, v283_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v281_data, v284_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v282_data, v285_acc, 3, 2, 0);
          r5[0] = (v286_acc[0]);
          r5[1] = (v286_acc[1]);
          r5[2] = (v286_acc[2]);
          r5[3] = (v286_acc[3]);
          float v291_data = r4[4];
          float v292_data = r4[5];
          float v293_data = r4[6];
          float v294_data = r4[7];
          float v295_tp{};
          float v296_tp{};
          float v297_tp{};
          float v298_tp{};
          tensorforge::transpose4x4b32(v295_tp, v296_tp, v297_tp, v298_tp, v291_data, v292_data, v293_data, v294_data);
          tensorforge::VectorT<float, 4> v299_acc{};
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v263_data, v299_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v264_data, v304_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v265_data, v305_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v266_data, v306_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v271_data, v307_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v272_data, v312_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v273_data, v313_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v274_data, v314_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v279_data, v315_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v280_data, v320_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v281_data, v321_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v282_data, v322_acc, 3, 2, 0);
          r5[4] = (v323_acc[0]);
          r5[5] = (v323_acc[1]);
          r5[6] = (v323_acc[2]);
          r5[7] = (v323_acc[3]);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v331_i0 = 0; v331_i0 < 1; ++v331_i0) {
            int32_t v339_lead = v18_lead + (v331_i0 * 32);
            #pragma unroll
            for (int32_t v332_i1 = 0; v332_i1 < 8; ++v332_i1) {
              float v334_data = r5[(v331_i0 + v332_i1)];
              int32_t v341_a = v339_lead + (v332_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v341_a], v334_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v347_i1 = 0; v347_i1 < 8; ++v347_i1) {
              float v355_data = __builtin_nontemporal_load(&glb_m6[(v18_lead + (v347_i1 * 12))]);
              r7[v347_i1] = v355_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v358_data = r7[0];
          float v359_data = r7[1];
          float v360_data = r7[2];
          float v361_data = r7[3];
          float v362_tp{};
          float v363_tp{};
          float v364_tp{};
          float v365_tp{};
          tensorforge::transpose4x4b32(v362_tp, v363_tp, v364_tp, v365_tp, v358_data, v359_data, v360_data, v361_data);
          tensorforge::VectorT<float, 4> v366_acc{};
          float v367_data = r6[0];
          float v368_data = r6[1];
          float v369_data = r6[2];
          float v370_data = r6[3];
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v367_data, v366_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v368_data, v371_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v369_data, v372_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v370_data, v373_acc, 3, 0, 0);
          float v375_data = r6[4];
          float v376_data = r6[5];
          float v377_data = r6[6];
          float v378_data = r6[7];
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v375_data, v374_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v376_data, v379_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v377_data, v380_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v378_data, v381_acc, 3, 1, 0);
          float v383_data = r6[8];
          float v384_data = r6[9];
          float v385_data = r6[10];
          float v386_data = r6[11];
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v383_data, v382_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v384_data, v387_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v385_data, v388_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v386_data, v389_acc, 3, 2, 0);
          r8[0] = (v390_acc[0]);
          r8[1] = (v390_acc[1]);
          r8[2] = (v390_acc[2]);
          r8[3] = (v390_acc[3]);
          float v395_data = r7[4];
          float v396_data = r7[5];
          float v397_data = r7[6];
          float v398_data = r7[7];
          float v399_tp{};
          float v400_tp{};
          float v401_tp{};
          float v402_tp{};
          tensorforge::transpose4x4b32(v399_tp, v400_tp, v401_tp, v402_tp, v395_data, v396_data, v397_data, v398_data);
          tensorforge::VectorT<float, 4> v403_acc{};
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v367_data, v403_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v368_data, v408_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v369_data, v409_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v370_data, v410_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v375_data, v411_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v376_data, v416_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v377_data, v417_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v378_data, v418_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v383_data, v419_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v384_data, v424_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v385_data, v425_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v386_data, v426_acc, 3, 2, 0);
          r8[4] = (v427_acc[0]);
          r8[5] = (v427_acc[1]);
          r8[6] = (v427_acc[2]);
          r8[7] = (v427_acc[3]);
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v435_i0 = 0; v435_i0 < 1; ++v435_i0) {
            int32_t v443_lead = v18_lead + (v435_i0 * 32);
            #pragma unroll
            for (int32_t v436_i1 = 0; v436_i1 < 8; ++v436_i1) {
              float v438_data = r8[(v435_i0 + v436_i1)];
              int32_t v446_a = v443_lead + ((v436_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v446_a], v438_data);
            }
          }
        }
      }
    }
  }
}

