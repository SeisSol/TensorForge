// === base name ===
kernel_82c30404161113f3

// === header ===
void launcher_kernel_82c30404161113f3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_82c30404161113f3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_82c30404161113f3, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_82c30404161113f3, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_82c30404161113f3), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_82c30404161113f3, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_82c30404161113f3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t v3_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v3_batchId0 < numElements0; v3_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v4_ahead1 = v3_batchId0 + (gridDim.x * blockDim.y);
        size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v3_batchId0 * 96 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v3_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v3_batchId0 * 96 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v3_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v3_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v3_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v3_batchId0 * 96 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v3_batchId0 * 144 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v3_batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v25_i1 * 12))]);
              r0[v25_i1] = v33_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v40_i1 * 12))]);
              r1[v40_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v55_i1 = 0; v55_i1 < 12; ++v55_i1) {
              float v63_data = __builtin_nontemporal_load(&glb_m3[(v23_lead + (v55_i1 * 12))]);
              r3[v55_i1] = v63_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v66_data = r1[0];
          float v67_data = r1[1];
          float v68_data = r1[2];
          float v69_data = r1[3];
          float v70_tp{};
          float v71_tp{};
          float v72_tp{};
          float v73_tp{};
          tensorforge::transpose4x4b32(v70_tp, v71_tp, v72_tp, v73_tp, v66_data, v67_data, v68_data, v69_data);
          tensorforge::VectorT<float, 4> v74_acc{};
          float v75_data = r0[0];
          float v76_data = r0[1];
          float v77_data = r0[2];
          float v78_data = r0[3];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v75_data, v74_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v76_data, v79_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v77_data, v80_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v78_data, v81_acc, 2, 0, 0);
          float v83_data = r0[4];
          float v84_data = r0[5];
          float v85_data = r0[6];
          float v86_data = r0[7];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v83_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v84_data, v87_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v85_data, v88_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v86_data, v89_acc, 2, 1, 0);
          float v91_data = r0[8];
          float v92_data = r0[9];
          float v93_data = r0[10];
          float v94_data = r0[11];
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v91_data, v90_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v92_data, v95_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v93_data, v96_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v94_data, v97_acc, 2, 2, 0);
          r2[0] = (v98_acc[0]);
          r2[1] = (v98_acc[1]);
          r2[2] = (v98_acc[2]);
          r2[3] = (v98_acc[3]);
          float v103_data = r1[4];
          float v104_data = r1[5];
          float v105_data = r1[6];
          float v106_data = r1[7];
          float v107_tp{};
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          tensorforge::transpose4x4b32(v107_tp, v108_tp, v109_tp, v110_tp, v103_data, v104_data, v105_data, v106_data);
          tensorforge::VectorT<float, 4> v111_acc{};
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v75_data, v111_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v76_data, v116_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v77_data, v117_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v78_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v83_data, v119_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v84_data, v124_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v85_data, v125_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v86_data, v126_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v91_data, v127_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v92_data, v132_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v93_data, v133_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v94_data, v134_acc, 2, 2, 0);
          r2[4] = (v135_acc[0]);
          r2[5] = (v135_acc[1]);
          r2[6] = (v135_acc[2]);
          r2[7] = (v135_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v145_i1 = 0; v145_i1 < 8; ++v145_i1) {
              float v153_data = __builtin_nontemporal_load(&glb_m4[(v23_lead + (v145_i1 * 12))]);
              r4[v145_i1] = v153_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 12; ++v160_i1) {
              float v168_data = __builtin_nontemporal_load(&glb_m5[(v23_lead + (v160_i1 * 12))]);
              r6[v160_i1] = v168_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v172_data = r4[0];
          float v173_data = r4[1];
          float v174_data = r4[2];
          float v175_data = r4[3];
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v172_data, v173_data, v174_data, v175_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          float v181_data = r3[0];
          float v182_data = r3[1];
          float v183_data = r3[2];
          float v184_data = r3[3];
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v181_data, v180_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v182_data, v185_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v183_data, v186_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v184_data, v187_acc, 2, 0, 0);
          float v189_data = r3[4];
          float v190_data = r3[5];
          float v191_data = r3[6];
          float v192_data = r3[7];
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v189_data, v188_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v190_data, v193_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v191_data, v194_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v192_data, v195_acc, 2, 1, 0);
          float v197_data = r3[8];
          float v198_data = r3[9];
          float v199_data = r3[10];
          float v200_data = r3[11];
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v197_data, v196_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v198_data, v201_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v199_data, v202_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v200_data, v203_acc, 2, 2, 0);
          ir5[0] = (v204_acc[0]);
          ir5[1] = (v204_acc[1]);
          ir5[2] = (v204_acc[2]);
          ir5[3] = (v204_acc[3]);
          float v209_data = r4[4];
          float v210_data = r4[5];
          float v211_data = r4[6];
          float v212_data = r4[7];
          float v213_tp{};
          float v214_tp{};
          float v215_tp{};
          float v216_tp{};
          tensorforge::transpose4x4b32(v213_tp, v214_tp, v215_tp, v216_tp, v209_data, v210_data, v211_data, v212_data);
          tensorforge::VectorT<float, 4> v217_acc{};
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v181_data, v217_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v182_data, v222_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v183_data, v223_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v184_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v189_data, v225_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v190_data, v230_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v191_data, v231_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v192_data, v232_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v197_data, v233_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v198_data, v238_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v199_data, v239_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v200_data, v240_acc, 2, 2, 0);
          ir5[4] = (v241_acc[0]);
          ir5[5] = (v241_acc[1]);
          ir5[6] = (v241_acc[2]);
          ir5[7] = (v241_acc[3]);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v250_n1 = 0; v250_n1 < 8; ++v250_n1) {
              float v252_data = ir5[v250_n1];
              float v254_data = r2[v250_n1];
              r5[v250_n1] = (v254_data + v252_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v262_i1 = 0; v262_i1 < 8; ++v262_i1) {
              float v270_data = __builtin_nontemporal_load(&glb_m6[(v23_lead + (v262_i1 * 12))]);
              r7[v262_i1] = v270_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v277_i1 = 0; v277_i1 < 12; ++v277_i1) {
              float v285_data = __builtin_nontemporal_load(&glb_m7[(v23_lead + (v277_i1 * 12))]);
              r9[v277_i1] = v285_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v289_data = r7[0];
          float v290_data = r7[1];
          float v291_data = r7[2];
          float v292_data = r7[3];
          float v293_tp{};
          float v294_tp{};
          float v295_tp{};
          float v296_tp{};
          tensorforge::transpose4x4b32(v293_tp, v294_tp, v295_tp, v296_tp, v289_data, v290_data, v291_data, v292_data);
          tensorforge::VectorT<float, 4> v297_acc{};
          float v298_data = r6[0];
          float v299_data = r6[1];
          float v300_data = r6[2];
          float v301_data = r6[3];
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v298_data, v297_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v299_data, v302_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v300_data, v303_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v301_data, v304_acc, 2, 0, 0);
          float v306_data = r6[4];
          float v307_data = r6[5];
          float v308_data = r6[6];
          float v309_data = r6[7];
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v306_data, v305_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v307_data, v310_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v308_data, v311_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v309_data, v312_acc, 2, 1, 0);
          float v314_data = r6[8];
          float v315_data = r6[9];
          float v316_data = r6[10];
          float v317_data = r6[11];
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v314_data, v313_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v315_data, v318_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v316_data, v319_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v317_data, v320_acc, 2, 2, 0);
          ir8[0] = (v321_acc[0]);
          ir8[1] = (v321_acc[1]);
          ir8[2] = (v321_acc[2]);
          ir8[3] = (v321_acc[3]);
          float v326_data = r7[4];
          float v327_data = r7[5];
          float v328_data = r7[6];
          float v329_data = r7[7];
          float v330_tp{};
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          tensorforge::transpose4x4b32(v330_tp, v331_tp, v332_tp, v333_tp, v326_data, v327_data, v328_data, v329_data);
          tensorforge::VectorT<float, 4> v334_acc{};
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v298_data, v334_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v299_data, v339_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v300_data, v340_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v301_data, v341_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v306_data, v342_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v307_data, v347_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v308_data, v348_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v309_data, v349_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v314_data, v350_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v315_data, v355_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v316_data, v356_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v317_data, v357_acc, 2, 2, 0);
          ir8[4] = (v358_acc[0]);
          ir8[5] = (v358_acc[1]);
          ir8[6] = (v358_acc[2]);
          ir8[7] = (v358_acc[3]);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v367_n1 = 0; v367_n1 < 8; ++v367_n1) {
              float v369_data = ir8[v367_n1];
              float v371_data = r5[v367_n1];
              r8[v367_n1] = (v371_data + v369_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v379_i1 = 0; v379_i1 < 8; ++v379_i1) {
              float v387_data = __builtin_nontemporal_load(&glb_m8[(v23_lead + (v379_i1 * 12))]);
              r10[v379_i1] = v387_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v391_data = r10[0];
          float v392_data = r10[1];
          float v393_data = r10[2];
          float v394_data = r10[3];
          float v395_tp{};
          float v396_tp{};
          float v397_tp{};
          float v398_tp{};
          tensorforge::transpose4x4b32(v395_tp, v396_tp, v397_tp, v398_tp, v391_data, v392_data, v393_data, v394_data);
          tensorforge::VectorT<float, 4> v399_acc{};
          float v400_data = r9[0];
          float v401_data = r9[1];
          float v402_data = r9[2];
          float v403_data = r9[3];
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v400_data, v399_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v401_data, v404_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v402_data, v405_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v403_data, v406_acc, 2, 0, 0);
          float v408_data = r9[4];
          float v409_data = r9[5];
          float v410_data = r9[6];
          float v411_data = r9[7];
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v408_data, v407_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v409_data, v412_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v410_data, v413_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v411_data, v414_acc, 2, 1, 0);
          float v416_data = r9[8];
          float v417_data = r9[9];
          float v418_data = r9[10];
          float v419_data = r9[11];
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v416_data, v415_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v417_data, v420_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v418_data, v421_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v419_data, v422_acc, 2, 2, 0);
          ir11[0] = (v423_acc[0]);
          ir11[1] = (v423_acc[1]);
          ir11[2] = (v423_acc[2]);
          ir11[3] = (v423_acc[3]);
          float v428_data = r10[4];
          float v429_data = r10[5];
          float v430_data = r10[6];
          float v431_data = r10[7];
          float v432_tp{};
          float v433_tp{};
          float v434_tp{};
          float v435_tp{};
          tensorforge::transpose4x4b32(v432_tp, v433_tp, v434_tp, v435_tp, v428_data, v429_data, v430_data, v431_data);
          tensorforge::VectorT<float, 4> v436_acc{};
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v400_data, v436_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v401_data, v441_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v402_data, v442_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v403_data, v443_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v408_data, v444_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v409_data, v449_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v410_data, v450_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v411_data, v451_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v416_data, v452_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v417_data, v457_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v418_data, v458_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v419_data, v459_acc, 2, 2, 0);
          ir11[4] = (v460_acc[0]);
          ir11[5] = (v460_acc[1]);
          ir11[6] = (v460_acc[2]);
          ir11[7] = (v460_acc[3]);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v469_n1 = 0; v469_n1 < 8; ++v469_n1) {
              float v471_data = ir11[v469_n1];
              float v473_data = r8[v469_n1];
              r11[v469_n1] = (v473_data + v471_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v480_i1 = 0; v480_i1 < 8; ++v480_i1) {
              float v482_data = r11[v480_i1];
              glb_m0[(v23_lead + (v480_i1 * 12))] = v482_data;
            }
          }
        }
      }
    }
  }
}

