// === base name ===
kernel_2a485fa6cbd677a3

// === header ===
void launcher_kernel_2a485fa6cbd677a3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2a485fa6cbd677a3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2a485fa6cbd677a3, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_2a485fa6cbd677a3, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (3328 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_2a485fa6cbd677a3), hipFuncAttributeMaxDynamicSharedMemorySize, 3328 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_2a485fa6cbd677a3, grid, block, 3328 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_2a485fa6cbd677a3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×32(6×12) {0..6}×{0..12} strided
    // m1 32×32(12×12) {0..12}×{0..12} strided
    // m2 32×32(6×12) {0..6}×{0..12} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // m4 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v4_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v4_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 16;
          if (v20_lead < 6) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v20_lead + (v22_i1 * 6))]);
              r0[v22_i1] = v30_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v37_i1 * 12))]);
              r1[v37_i1] = v45_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v20_lead < 6) {
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 12; ++v52_i1) {
              float v60_data = __builtin_nontemporal_load(&glb_m2[(v20_lead + (v52_i1 * 6))]);
              r3[v52_i1] = v60_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v63_data = r1[0];
          float v64_data = r1[1];
          float v65_data = r1[2];
          float v66_data = r1[3];
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          float v70_tp{};
          tensorforge::transpose4x4b32(v67_tp, v68_tp, v69_tp, v70_tp, v63_data, v64_data, v65_data, v66_data);
          tensorforge::VectorT<float, 4> v71_acc{};
          float v72_data = r0[0];
          float v73_data = r0[1];
          float v74_data = r0[2];
          float v75_data = r0[3];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v71_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v75_data, v78_acc, 2, 0, 0);
          float v80_data = r0[4];
          float v81_data = r0[5];
          float v82_data = r0[6];
          float v83_data = r0[7];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v79_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v83_data, v86_acc, 2, 1, 0);
          float v88_data = r0[8];
          float v89_data = r0[9];
          float v90_data = r0[10];
          float v91_data = r0[11];
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v87_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v89_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v90_data, v93_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v91_data, v94_acc, 2, 2, 0);
          r2[0] = (v95_acc[0]);
          r2[1] = (v95_acc[1]);
          r2[2] = (v95_acc[2]);
          r2[3] = (v95_acc[3]);
          float v100_data = r1[4];
          float v101_data = r1[5];
          float v102_data = r1[6];
          float v103_data = r1[7];
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          float v107_tp{};
          tensorforge::transpose4x4b32(v104_tp, v105_tp, v106_tp, v107_tp, v100_data, v101_data, v102_data, v103_data);
          tensorforge::VectorT<float, 4> v108_acc{};
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v72_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v73_data, v113_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v74_data, v114_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v75_data, v115_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v80_data, v116_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v81_data, v121_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v82_data, v122_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v83_data, v123_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v88_data, v124_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v89_data, v129_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v90_data, v130_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v91_data, v131_acc, 2, 2, 0);
          r2[4] = (v132_acc[0]);
          r2[5] = (v132_acc[1]);
          r2[6] = (v132_acc[2]);
          r2[7] = (v132_acc[3]);
          float v137_data = r1[8];
          float v138_data = r1[9];
          float v139_data = r1[10];
          float v140_data = r1[11];
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          tensorforge::transpose4x4b32(v141_tp, v142_tp, v143_tp, v144_tp, v137_data, v138_data, v139_data, v140_data);
          tensorforge::VectorT<float, 4> v145_acc{};
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v73_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v74_data, v151_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v75_data, v152_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v81_data, v158_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v82_data, v159_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v83_data, v160_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v88_data, v161_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v89_data, v166_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v90_data, v167_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v91_data, v168_acc, 2, 2, 0);
          r2[8] = (v169_acc[0]);
          r2[9] = (v169_acc[1]);
          r2[10] = (v169_acc[2]);
          r2[11] = (v169_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v20_lead < 6) {
            #pragma unroll
            for (int32_t v178_i1 = 0; v178_i1 < 12; ++v178_i1) {
              float v180_data = r2[v178_i1];
              int32_t v187_a = v20_lead + (v178_i1 * 12);
              s0[(v187_a ^ ((v187_a >> 4) & 15))] = v180_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v196_i1 = 0; v196_i1 < 12; ++v196_i1) {
              float v204_data = __builtin_nontemporal_load(&glb_m4[(v20_lead + (v196_i1 * 12))]);
              r5[v196_i1] = v204_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v211_tp{};
          float v212_tp{};
          float v213_tp{};
          float v214_tp{};
          tensorforge::transpose4x4b32(v211_tp, v212_tp, v213_tp, v214_tp, v63_data, v64_data, v65_data, v66_data);
          tensorforge::VectorT<float, 4> v215_acc{};
          float v216_data = r3[0];
          float v217_data = r3[1];
          float v218_data = r3[2];
          float v219_data = r3[3];
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v216_data, v215_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v217_data, v220_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v218_data, v221_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v219_data, v222_acc, 2, 0, 0);
          float v224_data = r3[4];
          float v225_data = r3[5];
          float v226_data = r3[6];
          float v227_data = r3[7];
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v224_data, v223_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v225_data, v228_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v226_data, v229_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v227_data, v230_acc, 2, 1, 0);
          float v232_data = r3[8];
          float v233_data = r3[9];
          float v234_data = r3[10];
          float v235_data = r3[11];
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v232_data, v231_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v233_data, v236_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v234_data, v237_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v235_data, v238_acc, 2, 2, 0);
          r4[0] = (v239_acc[0]);
          r4[1] = (v239_acc[1]);
          r4[2] = (v239_acc[2]);
          r4[3] = (v239_acc[3]);
          float v248_tp{};
          float v249_tp{};
          float v250_tp{};
          float v251_tp{};
          tensorforge::transpose4x4b32(v248_tp, v249_tp, v250_tp, v251_tp, v100_data, v101_data, v102_data, v103_data);
          tensorforge::VectorT<float, 4> v252_acc{};
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v216_data, v252_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v217_data, v257_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v218_data, v258_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v219_data, v259_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v224_data, v260_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v225_data, v265_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v226_data, v266_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v227_data, v267_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v232_data, v268_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v233_data, v273_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v234_data, v274_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v235_data, v275_acc, 2, 2, 0);
          r4[4] = (v276_acc[0]);
          r4[5] = (v276_acc[1]);
          r4[6] = (v276_acc[2]);
          r4[7] = (v276_acc[3]);
          float v285_tp{};
          float v286_tp{};
          float v287_tp{};
          float v288_tp{};
          tensorforge::transpose4x4b32(v285_tp, v286_tp, v287_tp, v288_tp, v137_data, v138_data, v139_data, v140_data);
          tensorforge::VectorT<float, 4> v289_acc{};
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v216_data, v289_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v217_data, v294_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v218_data, v295_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v219_data, v296_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v224_data, v297_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v225_data, v302_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v226_data, v303_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v227_data, v304_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v232_data, v305_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v233_data, v310_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v234_data, v311_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v235_data, v312_acc, 2, 2, 0);
          r4[8] = (v313_acc[0]);
          r4[9] = (v313_acc[1]);
          r4[10] = (v313_acc[2]);
          r4[11] = (v313_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v20_lead < 6) {
            int32_t v330_off = v20_lead + 6;
            #pragma unroll
            for (int32_t v322_i1 = 0; v322_i1 < 12; ++v322_i1) {
              float v324_data = r4[v322_i1];
              int32_t v332_a = v330_off + (v322_i1 * 12);
              s0[(v332_a ^ ((v332_a >> 4) & 15))] = v324_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v346_data = s0[(v20_lead ^ ((v20_lead >> 4) & 15))];
          int32_t v352_a = v20_lead + 12;
          float v356_data = s0[(v352_a ^ ((v352_a >> 4) & 15))];
          int32_t v362_a = v20_lead + 24;
          float v366_data = s0[(v362_a ^ ((v362_a >> 4) & 15))];
          int32_t v372_a = v20_lead + 36;
          float v376_data = s0[(v372_a ^ ((v372_a >> 4) & 15))];
          float v377_tp{};
          float v378_tp{};
          float v379_tp{};
          float v380_tp{};
          tensorforge::transpose4x4b32(v377_tp, v378_tp, v379_tp, v380_tp, v346_data, v356_data, v366_data, v376_data);
          tensorforge::VectorT<float, 4> v381_acc{};
          float v382_data = r5[0];
          float v383_data = r5[1];
          float v384_data = r5[2];
          float v385_data = r5[3];
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v382_data, v381_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v383_data, v386_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v384_data, v387_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v385_data, v388_acc, 2, 0, 0);
          float v390_data = r5[4];
          float v391_data = r5[5];
          float v392_data = r5[6];
          float v393_data = r5[7];
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v390_data, v389_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v391_data, v394_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v392_data, v395_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v393_data, v396_acc, 2, 1, 0);
          float v398_data = r5[8];
          float v399_data = r5[9];
          float v400_data = r5[10];
          float v401_data = r5[11];
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v398_data, v397_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v399_data, v402_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v400_data, v403_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v401_data, v404_acc, 2, 2, 0);
          r6[0] = (v405_acc[0]);
          r6[1] = (v405_acc[1]);
          r6[2] = (v405_acc[2]);
          r6[3] = (v405_acc[3]);
          int32_t v415_a = v20_lead + 48;
          float v419_data = s0[(v415_a ^ ((v415_a >> 4) & 15))];
          int32_t v425_a = v20_lead + 60;
          float v429_data = s0[(v425_a ^ ((v425_a >> 4) & 15))];
          int32_t v435_a = v20_lead + 72;
          float v439_data = s0[(v435_a ^ ((v435_a >> 4) & 15))];
          int32_t v445_a = v20_lead + 84;
          float v449_data = s0[(v445_a ^ ((v445_a >> 4) & 15))];
          float v450_tp{};
          float v451_tp{};
          float v452_tp{};
          float v453_tp{};
          tensorforge::transpose4x4b32(v450_tp, v451_tp, v452_tp, v453_tp, v419_data, v429_data, v439_data, v449_data);
          tensorforge::VectorT<float, 4> v454_acc{};
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v382_data, v454_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v383_data, v459_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v384_data, v460_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v385_data, v461_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v390_data, v462_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v391_data, v467_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v392_data, v468_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v393_data, v469_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v398_data, v470_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v399_data, v475_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v400_data, v476_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v401_data, v477_acc, 2, 2, 0);
          r6[4] = (v478_acc[0]);
          r6[5] = (v478_acc[1]);
          r6[6] = (v478_acc[2]);
          r6[7] = (v478_acc[3]);
          int32_t v488_a = v20_lead + 96;
          float v492_data = s0[(v488_a ^ ((v488_a >> 4) & 15))];
          int32_t v498_a = v20_lead + 108;
          float v502_data = s0[(v498_a ^ ((v498_a >> 4) & 15))];
          int32_t v508_a = v20_lead + 120;
          float v512_data = s0[(v508_a ^ ((v508_a >> 4) & 15))];
          int32_t v518_a = v20_lead + 132;
          float v522_data = s0[(v518_a ^ ((v518_a >> 4) & 15))];
          float v523_tp{};
          float v524_tp{};
          float v525_tp{};
          float v526_tp{};
          tensorforge::transpose4x4b32(v523_tp, v524_tp, v525_tp, v526_tp, v492_data, v502_data, v512_data, v522_data);
          tensorforge::VectorT<float, 4> v527_acc{};
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v382_data, v527_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v383_data, v532_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v384_data, v533_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v385_data, v534_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v390_data, v535_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v391_data, v540_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v392_data, v541_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v393_data, v542_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v398_data, v543_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v399_data, v548_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v400_data, v549_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v401_data, v550_acc, 2, 2, 0);
          r6[8] = (v551_acc[0]);
          r6[9] = (v551_acc[1]);
          r6[10] = (v551_acc[2]);
          r6[11] = (v551_acc[3]);
          // glb_m3 = store{r>g}(r6);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v560_i1 = 0; v560_i1 < 12; ++v560_i1) {
              float v562_data = r6[v560_i1];
              glb_m3[(v20_lead + (v560_i1 * 12))] = v562_data;
            }
          }
        }
      }
    }
  }
}

