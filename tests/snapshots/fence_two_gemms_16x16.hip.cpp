// === base name ===
kernel_973a1e2ea21d9a54

// === header ===
void launcher_kernel_973a1e2ea21d9a54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_973a1e2ea21d9a54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_973a1e2ea21d9a54, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_973a1e2ea21d9a54, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_973a1e2ea21d9a54), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_973a1e2ea21d9a54, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_973a1e2ea21d9a54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v0_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 32;
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v19_i1 * 16))]);
              r0[v19_i1] = v27_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 16; ++v34_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v34_i1 * 16))]);
              r1[v34_i1] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v45_data = r1[0];
          float v46_data = r1[1];
          float v47_data = r1[2];
          float v48_data = r1[3];
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          float v52_tp{};
          tensorforge::transpose4x4b32(v49_tp, v50_tp, v51_tp, v52_tp, v45_data, v46_data, v47_data, v48_data);
          tensorforge::VectorT<float, 4> v53_acc{};
          float v54_data = r0[0];
          float v55_data = r0[1];
          float v56_data = r0[2];
          float v57_data = r0[3];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v53_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 3, 0, 0);
          float v62_data = r0[4];
          float v63_data = r0[5];
          float v64_data = r0[6];
          float v65_data = r0[7];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v61_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 3, 1, 0);
          float v70_data = r0[8];
          float v71_data = r0[9];
          float v72_data = r0[10];
          float v73_data = r0[11];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v69_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 3, 2, 0);
          float v78_data = r0[12];
          float v79_data = r0[13];
          float v80_data = r0[14];
          float v81_data = r0[15];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v77_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v83_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v81_data, v84_acc, 3, 3, 0);
          r2[0] = (v85_acc[0]);
          r2[1] = (v85_acc[1]);
          r2[2] = (v85_acc[2]);
          r2[3] = (v85_acc[3]);
          float v90_data = r1[4];
          float v91_data = r1[5];
          float v92_data = r1[6];
          float v93_data = r1[7];
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          float v97_tp{};
          tensorforge::transpose4x4b32(v94_tp, v95_tp, v96_tp, v97_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v98_acc{};
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v55_data, v103_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v56_data, v104_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v57_data, v105_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v111_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v112_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v113_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v114_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v119_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v120_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v121_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v122_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v127_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v128_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v129_acc, 3, 3, 0);
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
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v54_data, v143_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v55_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v56_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v57_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v62_data, v151_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v63_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v64_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v65_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v159_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v164_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v73_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v167_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v172_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v173_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v81_data, v174_acc, 3, 3, 0);
          r2[8] = (v175_acc[0]);
          r2[9] = (v175_acc[1]);
          r2[10] = (v175_acc[2]);
          r2[11] = (v175_acc[3]);
          float v180_data = r1[12];
          float v181_data = r1[13];
          float v182_data = r1[14];
          float v183_data = r1[15];
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          tensorforge::transpose4x4b32(v184_tp, v185_tp, v186_tp, v187_tp, v180_data, v181_data, v182_data, v183_data);
          tensorforge::VectorT<float, 4> v188_acc{};
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v54_data, v188_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v55_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v56_data, v194_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v57_data, v195_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v62_data, v196_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v63_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v64_data, v202_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v65_data, v203_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v70_data, v204_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v71_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v72_data, v210_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v73_data, v211_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v78_data, v212_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v79_data, v217_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v80_data, v218_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v81_data, v219_acc, 3, 3, 0);
          r2[12] = (v220_acc[0]);
          r2[13] = (v220_acc[1]);
          r2[14] = (v220_acc[2]);
          r2[15] = (v220_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v229_i1 = 0; v229_i1 < 16; ++v229_i1) {
              float v231_data = r2[v229_i1];
              glb_m0[(v17_lead + (v229_i1 * 16))] = v231_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      __syncthreads();
      for (size_t v239_batchId0 = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y); v239_batchId0 < numElements1; v239_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v240_ahead1 = v239_batchId0 + (gridDim.x * blockDim.y);
        size_t v242_batchId1 = (v240_ahead1 < numElements1) ? v240_ahead1 : v239_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v239_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v239_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v239_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v239_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v239_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v239_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v239_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m4);
          int32_t v256_lead = threadIdx.x % 32;
          if (v256_lead < 16) {
            #pragma unroll
            for (int32_t v258_i1 = 0; v258_i1 < 16; ++v258_i1) {
              float v266_data = __builtin_nontemporal_load(&glb_m4[(v256_lead + (v258_i1 * 16))]);
              r0[v258_i1] = v266_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m5);
          if (v256_lead < 16) {
            #pragma unroll
            for (int32_t v273_i1 = 0; v273_i1 < 16; ++v273_i1) {
              float v281_data = __builtin_nontemporal_load(&glb_m5[(v256_lead + (v273_i1 * 16))]);
              r1[v273_i1] = v281_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m4););
          // wait(r1 = load{g>r}(glb_m5););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v284_data = r1[0];
          float v285_data = r1[1];
          float v286_data = r1[2];
          float v287_data = r1[3];
          float v288_tp{};
          float v289_tp{};
          float v290_tp{};
          float v291_tp{};
          tensorforge::transpose4x4b32(v288_tp, v289_tp, v290_tp, v291_tp, v284_data, v285_data, v286_data, v287_data);
          tensorforge::VectorT<float, 4> v292_acc{};
          float v293_data = r0[0];
          float v294_data = r0[1];
          float v295_data = r0[2];
          float v296_data = r0[3];
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v293_data, v292_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v294_data, v297_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v295_data, v298_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v296_data, v299_acc, 3, 0, 0);
          float v301_data = r0[4];
          float v302_data = r0[5];
          float v303_data = r0[6];
          float v304_data = r0[7];
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v301_data, v300_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v302_data, v305_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v303_data, v306_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v304_data, v307_acc, 3, 1, 0);
          float v309_data = r0[8];
          float v310_data = r0[9];
          float v311_data = r0[10];
          float v312_data = r0[11];
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v309_data, v308_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v310_data, v313_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v311_data, v314_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v312_data, v315_acc, 3, 2, 0);
          float v317_data = r0[12];
          float v318_data = r0[13];
          float v319_data = r0[14];
          float v320_data = r0[15];
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v317_data, v316_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v318_data, v321_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v319_data, v322_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v320_data, v323_acc, 3, 3, 0);
          r2[0] = (v324_acc[0]);
          r2[1] = (v324_acc[1]);
          r2[2] = (v324_acc[2]);
          r2[3] = (v324_acc[3]);
          float v329_data = r1[4];
          float v330_data = r1[5];
          float v331_data = r1[6];
          float v332_data = r1[7];
          float v333_tp{};
          float v334_tp{};
          float v335_tp{};
          float v336_tp{};
          tensorforge::transpose4x4b32(v333_tp, v334_tp, v335_tp, v336_tp, v329_data, v330_data, v331_data, v332_data);
          tensorforge::VectorT<float, 4> v337_acc{};
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v293_data, v337_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v294_data, v342_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v295_data, v343_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v296_data, v344_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v301_data, v345_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v302_data, v350_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v303_data, v351_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v304_data, v352_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v309_data, v353_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v310_data, v358_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v311_data, v359_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v312_data, v360_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v317_data, v361_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v318_data, v366_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v319_data, v367_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v320_data, v368_acc, 3, 3, 0);
          r2[4] = (v369_acc[0]);
          r2[5] = (v369_acc[1]);
          r2[6] = (v369_acc[2]);
          r2[7] = (v369_acc[3]);
          float v374_data = r1[8];
          float v375_data = r1[9];
          float v376_data = r1[10];
          float v377_data = r1[11];
          float v378_tp{};
          float v379_tp{};
          float v380_tp{};
          float v381_tp{};
          tensorforge::transpose4x4b32(v378_tp, v379_tp, v380_tp, v381_tp, v374_data, v375_data, v376_data, v377_data);
          tensorforge::VectorT<float, 4> v382_acc{};
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v293_data, v382_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v294_data, v387_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v295_data, v388_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v296_data, v389_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v301_data, v390_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v302_data, v395_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v303_data, v396_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v304_data, v397_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v309_data, v398_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v310_data, v403_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v311_data, v404_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v312_data, v405_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v317_data, v406_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v318_data, v411_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v319_data, v412_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v320_data, v413_acc, 3, 3, 0);
          r2[8] = (v414_acc[0]);
          r2[9] = (v414_acc[1]);
          r2[10] = (v414_acc[2]);
          r2[11] = (v414_acc[3]);
          float v419_data = r1[12];
          float v420_data = r1[13];
          float v421_data = r1[14];
          float v422_data = r1[15];
          float v423_tp{};
          float v424_tp{};
          float v425_tp{};
          float v426_tp{};
          tensorforge::transpose4x4b32(v423_tp, v424_tp, v425_tp, v426_tp, v419_data, v420_data, v421_data, v422_data);
          tensorforge::VectorT<float, 4> v427_acc{};
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v293_data, v427_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v294_data, v432_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v425_tp, v295_data, v433_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v426_tp, v296_data, v434_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v301_data, v435_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v302_data, v440_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v425_tp, v303_data, v441_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v426_tp, v304_data, v442_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v309_data, v443_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v310_data, v448_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v425_tp, v311_data, v449_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v426_tp, v312_data, v450_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v317_data, v451_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v318_data, v456_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v425_tp, v319_data, v457_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v426_tp, v320_data, v458_acc, 3, 3, 0);
          r2[12] = (v459_acc[0]);
          r2[13] = (v459_acc[1]);
          r2[14] = (v459_acc[2]);
          r2[15] = (v459_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v256_lead < 16) {
            #pragma unroll
            for (int32_t v468_i1 = 0; v468_i1 < 16; ++v468_i1) {
              float v470_data = r2[v468_i1];
              glb_m3[(v256_lead + (v468_i1 * 16))] = v470_data;
            }
          }
        }
      }
    }
  }
}

