// === base name ===
kernel_11f9adc1d83b5122

// === header ===
void launcher_kernel_11f9adc1d83b5122(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_11f9adc1d83b5122(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_11f9adc1d83b5122, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_11f9adc1d83b5122, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_11f9adc1d83b5122), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_11f9adc1d83b5122, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_11f9adc1d83b5122(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v4_batchId0 * 64 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v4_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 64;
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v20_lead + (v22_i1 * 8))]);
              r0[v22_i1] = v30_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v37_i1 * 8))]);
              r1[v37_i1] = v45_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[8]{};
          // r3 = load{g>r}(glb_m2);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 8; ++v52_i1) {
              float v60_data = __builtin_nontemporal_load(&glb_m2[(v20_lead + (v52_i1 * 8))]);
              r3[v52_i1] = v60_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
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
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v71_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v75_data, v78_acc, 4, 0, 0);
          float v80_data = r0[4];
          float v81_data = r0[5];
          float v82_data = r0[6];
          float v83_data = r0[7];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v79_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v83_data, v86_acc, 4, 1, 0);
          r2[0] = (v87_acc[0]);
          r2[1] = (v87_acc[1]);
          r2[2] = (v87_acc[2]);
          r2[3] = (v87_acc[3]);
          float v92_data = r1[4];
          float v93_data = r1[5];
          float v94_data = r1[6];
          float v95_data = r1[7];
          float v96_tp{};
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          tensorforge::transpose4x4b32(v96_tp, v97_tp, v98_tp, v99_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v100_acc{};
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v100_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v105_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v106_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v107_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v108_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v113_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v114_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v115_acc, 4, 1, 0);
          r2[4] = (v116_acc[0]);
          r2[5] = (v116_acc[1]);
          r2[6] = (v116_acc[2]);
          r2[7] = (v116_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m3);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v126_i1 = 0; v126_i1 < 8; ++v126_i1) {
              float v134_data = __builtin_nontemporal_load(&glb_m3[(v20_lead + (v126_i1 * 8))]);
              r4[v126_i1] = v134_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          // wait(r4 = load{g>r}(glb_m3););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir5[8]{};
          float v138_data = r4[0];
          float v139_data = r4[1];
          float v140_data = r4[2];
          float v141_data = r4[3];
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          float v145_tp{};
          tensorforge::transpose4x4b32(v142_tp, v143_tp, v144_tp, v145_tp, v138_data, v139_data, v140_data, v141_data);
          tensorforge::VectorT<float, 4> v146_acc{};
          float v147_data = r3[0];
          float v148_data = r3[1];
          float v149_data = r3[2];
          float v150_data = r3[3];
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v147_data, v146_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v148_data, v151_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v149_data, v152_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v150_data, v153_acc, 4, 0, 0);
          float v155_data = r3[4];
          float v156_data = r3[5];
          float v157_data = r3[6];
          float v158_data = r3[7];
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v155_data, v154_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v156_data, v159_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v157_data, v160_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v158_data, v161_acc, 4, 1, 0);
          ir5[0] = (v162_acc[0]);
          ir5[1] = (v162_acc[1]);
          ir5[2] = (v162_acc[2]);
          ir5[3] = (v162_acc[3]);
          float v167_data = r4[4];
          float v168_data = r4[5];
          float v169_data = r4[6];
          float v170_data = r4[7];
          float v171_tp{};
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          tensorforge::transpose4x4b32(v171_tp, v172_tp, v173_tp, v174_tp, v167_data, v168_data, v169_data, v170_data);
          tensorforge::VectorT<float, 4> v175_acc{};
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v147_data, v175_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v148_data, v180_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v149_data, v181_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v150_data, v182_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v155_data, v183_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v156_data, v188_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v157_data, v189_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v158_data, v190_acc, 4, 1, 0);
          ir5[4] = (v191_acc[0]);
          ir5[5] = (v191_acc[1]);
          ir5[6] = (v191_acc[2]);
          ir5[7] = (v191_acc[3]);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v200_n1 = 0; v200_n1 < 8; ++v200_n1) {
              float v202_data = ir5[v200_n1];
              float v204_data = r2[v200_n1];
              r5[v200_n1] = (v204_data + v202_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v211_i1 = 0; v211_i1 < 8; ++v211_i1) {
              float v213_data = r5[v211_i1];
              int32_t v220_a = v20_lead + (v211_i1 * 8);
              s0[(v220_a ^ ((v220_a >> 5) & 31))] = v213_data;
            }
          }
          // glb_m4 = abs(s0)
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v228_k1 = 0; v228_k1 < 8; ++v228_k1) {
              int32_t v234_a = v228_k1 * 8;
              int32_t v235_a = v20_lead + v234_a;
              float v239_data = s0[(v235_a ^ ((v235_a >> 5) & 31))];
              glb_m4[(v20_lead + v234_a)] = (fabsf(v239_data));
            }
          }
        }
      }
    }
  }
}

