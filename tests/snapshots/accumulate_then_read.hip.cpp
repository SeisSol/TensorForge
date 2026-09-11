// === base name ===
kernel_5f94085f10475beb

// === header ===
void launcher_kernel_5f94085f10475beb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5f94085f10475beb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5f94085f10475beb, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5f94085f10475beb, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5f94085f10475beb), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_5f94085f10475beb, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  m9,  m9_extraOffset,  m10,  m10_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5f94085f10475beb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 13×13(13×13) {0..13}×{0..13} strided
    // m4 16×32(16×32) {0..16}×{0..32} strided
    // m5 13×13(13×13) {0..13}×{0..13} strided
    // m6 16×32(16×32) {0..16}×{0..32} strided
    // m7 13×13(13×13) {0..13}×{0..13} strided
    // m8 16×32(16×32) {0..16}×{0..32} strided
    // m9 32×13(32×13) {0..32}×{0..13} strided
    // m10 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // t0 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m3 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m4 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t0 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // t1 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m5 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m6 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t1 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // t2 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m7 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m8 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t2 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // m9 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m10 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 416 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 416 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 169 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 512 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v0_batchId0 * 169 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v0_batchId0 * 512 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v0_batchId0 * 169 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v0_batchId0 * 512 + 0 + m8_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m9 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m9[v0_batchId0 * 416 + 0 + m9_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m10 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m10[v0_batchId0 * 169 + 0 + m10_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v29_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 13; ++v24_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v24_i1 * 32))]);
              r0[(v23_i0 + v24_i1)] = v32_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          if (v22_lead < 13) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 13; ++v39_i1) {
              float v47_data = __builtin_nontemporal_load(&glb_m2[(v22_lead + (v39_i1 * 13))]);
              r1[v39_i1] = v47_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[13]{};
          // r3 = load{g>r}(glb_m3);
          if (v22_lead < 13) {
            #pragma unroll
            for (int32_t v54_i1 = 0; v54_i1 < 13; ++v54_i1) {
              float v62_data = __builtin_nontemporal_load(&glb_m3[(v22_lead + (v54_i1 * 13))]);
              r3[v54_i1] = v62_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[13]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v65_data = r1[0];
          float v66_data = r1[1];
          float v67_data = r1[2];
          float v68_data = r1[3];
          float v69_tp{};
          float v70_tp{};
          float v71_tp{};
          float v72_tp{};
          tensorforge::transpose4x4b32(v69_tp, v70_tp, v71_tp, v72_tp, v65_data, v66_data, v67_data, v68_data);
          tensorforge::VectorT<float, 4> v73_acc{};
          float v74_data = r0[0];
          float v75_data = r0[1];
          float v76_data = r0[2];
          float v77_data = r0[3];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v73_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v75_data, v78_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v76_data, v79_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v77_data, v80_acc, 3, 0, 0);
          float v82_data = r0[4];
          float v83_data = r0[5];
          float v84_data = r0[6];
          float v85_data = r0[7];
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v81_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v83_data, v86_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v84_data, v87_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v85_data, v88_acc, 3, 1, 0);
          float v90_data = r0[8];
          float v91_data = r0[9];
          float v92_data = r0[10];
          float v93_data = r0[11];
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v90_data, v89_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v91_data, v94_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v92_data, v95_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v93_data, v96_acc, 3, 2, 0);
          float v98_data = r0[12];
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v98_data, v97_acc, 3, 3, 0);
          r2[0] = (v102_acc[0]);
          r2[1] = (v102_acc[1]);
          r2[2] = (v102_acc[2]);
          r2[3] = (v102_acc[3]);
          float v107_data = r1[4];
          float v108_data = r1[5];
          float v109_data = r1[6];
          float v110_data = r1[7];
          float v111_tp{};
          float v112_tp{};
          float v113_tp{};
          float v114_tp{};
          tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
          tensorforge::VectorT<float, 4> v115_acc{};
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v74_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v75_data, v120_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v76_data, v121_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v77_data, v122_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v82_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v83_data, v128_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v84_data, v129_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v85_data, v130_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v90_data, v131_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v91_data, v136_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v92_data, v137_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v93_data, v138_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v98_data, v139_acc, 3, 3, 0);
          r2[4] = (v144_acc[0]);
          r2[5] = (v144_acc[1]);
          r2[6] = (v144_acc[2]);
          r2[7] = (v144_acc[3]);
          float v149_data = r1[8];
          float v150_data = r1[9];
          float v151_data = r1[10];
          float v152_data = r1[11];
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          float v156_tp{};
          tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v149_data, v150_data, v151_data, v152_data);
          tensorforge::VectorT<float, 4> v157_acc{};
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v74_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v75_data, v162_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v76_data, v163_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v77_data, v164_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v82_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v83_data, v170_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v84_data, v171_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v85_data, v172_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v90_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v91_data, v178_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v92_data, v179_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v93_data, v180_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v98_data, v181_acc, 3, 3, 0);
          r2[8] = (v186_acc[0]);
          r2[9] = (v186_acc[1]);
          r2[10] = (v186_acc[2]);
          r2[11] = (v186_acc[3]);
          float v204_acc{};
          float v205_data = r1[12];
          float v206_bc = tensorforge::broadcast<32, 16, 0>(v205_data);
          tensorforge::fmacdpp16<0>(v204_acc, v206_bc, v74_data);
          tensorforge::fmacdpp16<1>(v204_acc, v206_bc, v75_data);
          tensorforge::fmacdpp16<2>(v204_acc, v206_bc, v76_data);
          tensorforge::fmacdpp16<3>(v204_acc, v206_bc, v77_data);
          tensorforge::fmacdpp16<4>(v204_acc, v206_bc, v82_data);
          tensorforge::fmacdpp16<5>(v204_acc, v206_bc, v83_data);
          tensorforge::fmacdpp16<6>(v204_acc, v206_bc, v84_data);
          tensorforge::fmacdpp16<7>(v204_acc, v206_bc, v85_data);
          tensorforge::fmacdpp16<8>(v204_acc, v206_bc, v90_data);
          tensorforge::fmacdpp16<9>(v204_acc, v206_bc, v91_data);
          tensorforge::fmacdpp16<10>(v204_acc, v206_bc, v92_data);
          tensorforge::fmacdpp16<11>(v204_acc, v206_bc, v93_data);
          tensorforge::fmacdpp16<12>(v204_acc, v206_bc, v98_data);
          r2[12] = v204_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v210_i0 = 0; v210_i0 < 1; ++v210_i0) {
            int32_t v218_lead = v22_lead + (v210_i0 * 32);
            #pragma unroll
            for (int32_t v211_i1 = 0; v211_i1 < 13; ++v211_i1) {
              float v213_data = r2[(v210_i0 + v211_i1)];
              glb_m0[(v218_lead + (v211_i1 * 32))] = v213_data;
            }
          }
          float r5[32]{};
          // r5 = load{g>r}(glb_m4);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v226_i1 = 0; v226_i1 < 32; ++v226_i1) {
              float v234_data = __builtin_nontemporal_load(&glb_m4[(v22_lead + (v226_i1 * 16))]);
              r5[v226_i1] = v234_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[13]{};
          // r4 = +(r0 * r3) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v237_data = r3[0];
          float v238_data = r3[1];
          float v239_data = r3[2];
          float v240_data = r3[3];
          float v241_tp{};
          float v242_tp{};
          float v243_tp{};
          float v244_tp{};
          tensorforge::transpose4x4b32(v241_tp, v242_tp, v243_tp, v244_tp, v237_data, v238_data, v239_data, v240_data);
          tensorforge::VectorT<float, 4> v245_acc{};
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v74_data, v245_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v75_data, v250_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v76_data, v251_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v77_data, v252_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v82_data, v253_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v83_data, v258_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v84_data, v259_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v85_data, v260_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v90_data, v261_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v91_data, v266_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v92_data, v267_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v93_data, v268_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v98_data, v269_acc, 3, 3, 0);
          r4[0] = (v274_acc[0]);
          r4[1] = (v274_acc[1]);
          r4[2] = (v274_acc[2]);
          r4[3] = (v274_acc[3]);
          float v279_data = r3[4];
          float v280_data = r3[5];
          float v281_data = r3[6];
          float v282_data = r3[7];
          float v283_tp{};
          float v284_tp{};
          float v285_tp{};
          float v286_tp{};
          tensorforge::transpose4x4b32(v283_tp, v284_tp, v285_tp, v286_tp, v279_data, v280_data, v281_data, v282_data);
          tensorforge::VectorT<float, 4> v287_acc{};
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v74_data, v287_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v75_data, v292_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v76_data, v293_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v77_data, v294_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v82_data, v295_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v83_data, v300_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v84_data, v301_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v85_data, v302_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v90_data, v303_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v91_data, v308_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v92_data, v309_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v93_data, v310_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v98_data, v311_acc, 3, 3, 0);
          r4[4] = (v316_acc[0]);
          r4[5] = (v316_acc[1]);
          r4[6] = (v316_acc[2]);
          r4[7] = (v316_acc[3]);
          float v321_data = r3[8];
          float v322_data = r3[9];
          float v323_data = r3[10];
          float v324_data = r3[11];
          float v325_tp{};
          float v326_tp{};
          float v327_tp{};
          float v328_tp{};
          tensorforge::transpose4x4b32(v325_tp, v326_tp, v327_tp, v328_tp, v321_data, v322_data, v323_data, v324_data);
          tensorforge::VectorT<float, 4> v329_acc{};
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v74_data, v329_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v75_data, v334_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v76_data, v335_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v77_data, v336_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v82_data, v337_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v83_data, v342_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v84_data, v343_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v85_data, v344_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v90_data, v345_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v91_data, v350_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v92_data, v351_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v93_data, v352_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v98_data, v353_acc, 3, 3, 0);
          r4[8] = (v358_acc[0]);
          r4[9] = (v358_acc[1]);
          r4[10] = (v358_acc[2]);
          r4[11] = (v358_acc[3]);
          float v376_acc{};
          float v377_data = r3[12];
          float v378_bc = tensorforge::broadcast<32, 16, 0>(v377_data);
          tensorforge::fmacdpp16<0>(v376_acc, v378_bc, v74_data);
          tensorforge::fmacdpp16<1>(v376_acc, v378_bc, v75_data);
          tensorforge::fmacdpp16<2>(v376_acc, v378_bc, v76_data);
          tensorforge::fmacdpp16<3>(v376_acc, v378_bc, v77_data);
          tensorforge::fmacdpp16<4>(v376_acc, v378_bc, v82_data);
          tensorforge::fmacdpp16<5>(v376_acc, v378_bc, v83_data);
          tensorforge::fmacdpp16<6>(v376_acc, v378_bc, v84_data);
          tensorforge::fmacdpp16<7>(v376_acc, v378_bc, v85_data);
          tensorforge::fmacdpp16<8>(v376_acc, v378_bc, v90_data);
          tensorforge::fmacdpp16<9>(v376_acc, v378_bc, v91_data);
          tensorforge::fmacdpp16<10>(v376_acc, v378_bc, v92_data);
          tensorforge::fmacdpp16<11>(v376_acc, v378_bc, v93_data);
          tensorforge::fmacdpp16<12>(v376_acc, v378_bc, v98_data);
          r4[12] = v376_acc;
          float r7[13]{};
          // r7 = load{g>r}(glb_m5);
          if (v22_lead < 13) {
            #pragma unroll
            for (int32_t v384_i1 = 0; v384_i1 < 13; ++v384_i1) {
              float v392_data = __builtin_nontemporal_load(&glb_m5[(v22_lead + (v384_i1 * 13))]);
              r7[v384_i1] = v392_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[13]{};
          // r6 = +(r5 * r4) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v395_data = r4[0];
          float v396_data = r4[1];
          float v397_data = r4[2];
          float v398_data = r4[3];
          float v399_tp{};
          float v400_tp{};
          float v401_tp{};
          float v402_tp{};
          tensorforge::transpose4x4b32(v399_tp, v400_tp, v401_tp, v402_tp, v395_data, v396_data, v397_data, v398_data);
          tensorforge::VectorT<float, 4> v403_acc{};
          float v404_data = r5[0];
          float v405_data = r5[1];
          float v406_data = r5[2];
          float v407_data = r5[3];
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v404_data, v403_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v405_data, v408_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v406_data, v409_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v407_data, v410_acc, 3, 0, 0);
          float v412_data = r5[4];
          float v413_data = r5[5];
          float v414_data = r5[6];
          float v415_data = r5[7];
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v412_data, v411_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v413_data, v416_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v414_data, v417_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v415_data, v418_acc, 3, 1, 0);
          float v420_data = r5[8];
          float v421_data = r5[9];
          float v422_data = r5[10];
          float v423_data = r5[11];
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v420_data, v419_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v421_data, v424_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v422_data, v425_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v423_data, v426_acc, 3, 2, 0);
          float v428_data = r5[12];
          float v429_data = r5[13];
          float v430_data = r5[14];
          float v431_data = r5[15];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v428_data, v427_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v429_data, v432_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v430_data, v433_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v431_data, v434_acc, 3, 3, 0);
          float v436_data = r5[16];
          float v437_data = r5[17];
          float v438_data = r5[18];
          float v439_data = r5[19];
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v436_data, v435_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v437_data, v440_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v438_data, v441_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v439_data, v442_acc, 3, 4, 0);
          float v444_data = r5[20];
          float v445_data = r5[21];
          float v446_data = r5[22];
          float v447_data = r5[23];
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v444_data, v443_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v445_data, v448_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v446_data, v449_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v447_data, v450_acc, 3, 5, 0);
          float v452_data = r5[24];
          float v453_data = r5[25];
          float v454_data = r5[26];
          float v455_data = r5[27];
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v452_data, v451_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v453_data, v456_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v454_data, v457_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v455_data, v458_acc, 3, 6, 0);
          float v460_data = r5[28];
          float v461_data = r5[29];
          float v462_data = r5[30];
          float v463_data = r5[31];
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v460_data, v459_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v461_data, v464_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v462_data, v465_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v463_data, v466_acc, 3, 7, 0);
          r6[0] = (v467_acc[0]);
          r6[1] = (v467_acc[1]);
          r6[2] = (v467_acc[2]);
          r6[3] = (v467_acc[3]);
          float v472_data = r4[4];
          float v473_data = r4[5];
          float v474_data = r4[6];
          float v475_data = r4[7];
          float v476_tp{};
          float v477_tp{};
          float v478_tp{};
          float v479_tp{};
          tensorforge::transpose4x4b32(v476_tp, v477_tp, v478_tp, v479_tp, v472_data, v473_data, v474_data, v475_data);
          tensorforge::VectorT<float, 4> v480_acc{};
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v404_data, v480_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v405_data, v485_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v406_data, v486_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v407_data, v487_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v412_data, v488_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v413_data, v493_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v414_data, v494_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v415_data, v495_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v420_data, v496_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v421_data, v501_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v422_data, v502_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v423_data, v503_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v428_data, v504_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v429_data, v509_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v430_data, v510_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v431_data, v511_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v436_data, v512_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v437_data, v517_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v438_data, v518_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v439_data, v519_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v444_data, v520_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v445_data, v525_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v446_data, v526_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v447_data, v527_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v452_data, v528_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v453_data, v533_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v454_data, v534_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v455_data, v535_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v460_data, v536_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v461_data, v541_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v462_data, v542_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v463_data, v543_acc, 3, 7, 0);
          r6[4] = (v544_acc[0]);
          r6[5] = (v544_acc[1]);
          r6[6] = (v544_acc[2]);
          r6[7] = (v544_acc[3]);
          float v549_data = r4[8];
          float v550_data = r4[9];
          float v551_data = r4[10];
          float v552_data = r4[11];
          float v553_tp{};
          float v554_tp{};
          float v555_tp{};
          float v556_tp{};
          tensorforge::transpose4x4b32(v553_tp, v554_tp, v555_tp, v556_tp, v549_data, v550_data, v551_data, v552_data);
          tensorforge::VectorT<float, 4> v557_acc{};
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v404_data, v557_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v405_data, v562_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v406_data, v563_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v407_data, v564_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v412_data, v565_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v413_data, v570_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v414_data, v571_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v415_data, v572_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v420_data, v573_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v421_data, v578_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v422_data, v579_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v423_data, v580_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v428_data, v581_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v429_data, v586_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v430_data, v587_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v589_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v431_data, v588_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v436_data, v589_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v437_data, v594_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v438_data, v595_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v439_data, v596_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v444_data, v597_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v445_data, v602_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v446_data, v603_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v447_data, v604_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v452_data, v605_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v453_data, v610_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v454_data, v611_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v455_data, v612_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v460_data, v613_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v461_data, v618_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v462_data, v619_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v463_data, v620_acc, 3, 7, 0);
          r6[8] = (v621_acc[0]);
          r6[9] = (v621_acc[1]);
          r6[10] = (v621_acc[2]);
          r6[11] = (v621_acc[3]);
          float v658_acc{};
          float v659_data = r4[12];
          float v660_bc = tensorforge::broadcast<32, 16, 0>(v659_data);
          tensorforge::fmacdpp16<0>(v658_acc, v660_bc, v404_data);
          tensorforge::fmacdpp16<1>(v658_acc, v660_bc, v405_data);
          tensorforge::fmacdpp16<2>(v658_acc, v660_bc, v406_data);
          tensorforge::fmacdpp16<3>(v658_acc, v660_bc, v407_data);
          tensorforge::fmacdpp16<4>(v658_acc, v660_bc, v412_data);
          tensorforge::fmacdpp16<5>(v658_acc, v660_bc, v413_data);
          tensorforge::fmacdpp16<6>(v658_acc, v660_bc, v414_data);
          tensorforge::fmacdpp16<7>(v658_acc, v660_bc, v415_data);
          tensorforge::fmacdpp16<8>(v658_acc, v660_bc, v420_data);
          tensorforge::fmacdpp16<9>(v658_acc, v660_bc, v421_data);
          tensorforge::fmacdpp16<10>(v658_acc, v660_bc, v422_data);
          tensorforge::fmacdpp16<11>(v658_acc, v660_bc, v423_data);
          tensorforge::fmacdpp16<12>(v658_acc, v660_bc, v428_data);
          tensorforge::fmacdpp16<13>(v658_acc, v660_bc, v429_data);
          tensorforge::fmacdpp16<14>(v658_acc, v660_bc, v430_data);
          tensorforge::fmacdpp16<15>(v658_acc, v660_bc, v431_data);
          float v661_bc = tensorforge::broadcast<32, 16, 1>(v659_data);
          tensorforge::fmacdpp16<0>(v658_acc, v661_bc, v436_data);
          tensorforge::fmacdpp16<1>(v658_acc, v661_bc, v437_data);
          tensorforge::fmacdpp16<2>(v658_acc, v661_bc, v438_data);
          tensorforge::fmacdpp16<3>(v658_acc, v661_bc, v439_data);
          tensorforge::fmacdpp16<4>(v658_acc, v661_bc, v444_data);
          tensorforge::fmacdpp16<5>(v658_acc, v661_bc, v445_data);
          tensorforge::fmacdpp16<6>(v658_acc, v661_bc, v446_data);
          tensorforge::fmacdpp16<7>(v658_acc, v661_bc, v447_data);
          tensorforge::fmacdpp16<8>(v658_acc, v661_bc, v452_data);
          tensorforge::fmacdpp16<9>(v658_acc, v661_bc, v453_data);
          tensorforge::fmacdpp16<10>(v658_acc, v661_bc, v454_data);
          tensorforge::fmacdpp16<11>(v658_acc, v661_bc, v455_data);
          tensorforge::fmacdpp16<12>(v658_acc, v661_bc, v460_data);
          tensorforge::fmacdpp16<13>(v658_acc, v661_bc, v461_data);
          tensorforge::fmacdpp16<14>(v658_acc, v661_bc, v462_data);
          tensorforge::fmacdpp16<15>(v658_acc, v661_bc, v463_data);
          r6[12] = v658_acc;
          // glb_m0 = store{r>g}(r6);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v666_i1 = 0; v666_i1 < 13; ++v666_i1) {
              float v668_data = r6[v666_i1];
              int32_t v675_a = v22_lead + (v666_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v675_a], v668_data);
            }
          }
          float r9[32]{};
          // r9 = load{g>r}(glb_m6);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v681_i1 = 0; v681_i1 < 32; ++v681_i1) {
              float v689_data = __builtin_nontemporal_load(&glb_m6[(v22_lead + (v681_i1 * 16))]);
              r9[v681_i1] = v689_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r8[13]{};
          // r8 = +(r0 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v692_data = r7[0];
          float v693_data = r7[1];
          float v694_data = r7[2];
          float v695_data = r7[3];
          float v696_tp{};
          float v697_tp{};
          float v698_tp{};
          float v699_tp{};
          tensorforge::transpose4x4b32(v696_tp, v697_tp, v698_tp, v699_tp, v692_data, v693_data, v694_data, v695_data);
          tensorforge::VectorT<float, 4> v700_acc{};
          tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v696_tp, v74_data, v700_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v697_tp, v75_data, v705_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v698_tp, v76_data, v706_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v699_tp, v77_data, v707_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v713_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v696_tp, v82_data, v708_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v697_tp, v83_data, v713_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v715_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v698_tp, v84_data, v714_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v699_tp, v85_data, v715_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v721_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v696_tp, v90_data, v716_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v697_tp, v91_data, v721_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v698_tp, v92_data, v722_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v699_tp, v93_data, v723_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v729_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v696_tp, v98_data, v724_acc, 3, 3, 0);
          r8[0] = (v729_acc[0]);
          r8[1] = (v729_acc[1]);
          r8[2] = (v729_acc[2]);
          r8[3] = (v729_acc[3]);
          float v734_data = r7[4];
          float v735_data = r7[5];
          float v736_data = r7[6];
          float v737_data = r7[7];
          float v738_tp{};
          float v739_tp{};
          float v740_tp{};
          float v741_tp{};
          tensorforge::transpose4x4b32(v738_tp, v739_tp, v740_tp, v741_tp, v734_data, v735_data, v736_data, v737_data);
          tensorforge::VectorT<float, 4> v742_acc{};
          tensorforge::VectorT<float, 4> v747_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v74_data, v742_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v75_data, v747_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v749_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v76_data, v748_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v750_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v77_data, v749_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v755_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v82_data, v750_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v83_data, v755_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v757_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v84_data, v756_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v85_data, v757_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v763_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v90_data, v758_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v764_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v91_data, v763_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v765_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v92_data, v764_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v93_data, v765_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v98_data, v766_acc, 3, 3, 0);
          r8[4] = (v771_acc[0]);
          r8[5] = (v771_acc[1]);
          r8[6] = (v771_acc[2]);
          r8[7] = (v771_acc[3]);
          float v776_data = r7[8];
          float v777_data = r7[9];
          float v778_data = r7[10];
          float v779_data = r7[11];
          float v780_tp{};
          float v781_tp{};
          float v782_tp{};
          float v783_tp{};
          tensorforge::transpose4x4b32(v780_tp, v781_tp, v782_tp, v783_tp, v776_data, v777_data, v778_data, v779_data);
          tensorforge::VectorT<float, 4> v784_acc{};
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v780_tp, v74_data, v784_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v781_tp, v75_data, v789_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v782_tp, v76_data, v790_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v783_tp, v77_data, v791_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v797_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v780_tp, v82_data, v792_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v781_tp, v83_data, v797_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v799_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v782_tp, v84_data, v798_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v783_tp, v85_data, v799_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v805_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v780_tp, v90_data, v800_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v781_tp, v91_data, v805_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v807_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v782_tp, v92_data, v806_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v808_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v783_tp, v93_data, v807_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v780_tp, v98_data, v808_acc, 3, 3, 0);
          r8[8] = (v813_acc[0]);
          r8[9] = (v813_acc[1]);
          r8[10] = (v813_acc[2]);
          r8[11] = (v813_acc[3]);
          float v831_acc{};
          float v832_data = r7[12];
          float v833_bc = tensorforge::broadcast<32, 16, 0>(v832_data);
          tensorforge::fmacdpp16<0>(v831_acc, v833_bc, v74_data);
          tensorforge::fmacdpp16<1>(v831_acc, v833_bc, v75_data);
          tensorforge::fmacdpp16<2>(v831_acc, v833_bc, v76_data);
          tensorforge::fmacdpp16<3>(v831_acc, v833_bc, v77_data);
          tensorforge::fmacdpp16<4>(v831_acc, v833_bc, v82_data);
          tensorforge::fmacdpp16<5>(v831_acc, v833_bc, v83_data);
          tensorforge::fmacdpp16<6>(v831_acc, v833_bc, v84_data);
          tensorforge::fmacdpp16<7>(v831_acc, v833_bc, v85_data);
          tensorforge::fmacdpp16<8>(v831_acc, v833_bc, v90_data);
          tensorforge::fmacdpp16<9>(v831_acc, v833_bc, v91_data);
          tensorforge::fmacdpp16<10>(v831_acc, v833_bc, v92_data);
          tensorforge::fmacdpp16<11>(v831_acc, v833_bc, v93_data);
          tensorforge::fmacdpp16<12>(v831_acc, v833_bc, v98_data);
          r8[12] = v831_acc;
          float r11[13]{};
          // r11 = load{g>r}(glb_m7);
          if (v22_lead < 13) {
            #pragma unroll
            for (int32_t v839_i1 = 0; v839_i1 < 13; ++v839_i1) {
              float v847_data = __builtin_nontemporal_load(&glb_m7[(v22_lead + (v839_i1 * 13))]);
              r11[v839_i1] = v847_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[13]{};
          // r10 = +(r9 * r8) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v850_data = r8[0];
          float v851_data = r8[1];
          float v852_data = r8[2];
          float v853_data = r8[3];
          float v854_tp{};
          float v855_tp{};
          float v856_tp{};
          float v857_tp{};
          tensorforge::transpose4x4b32(v854_tp, v855_tp, v856_tp, v857_tp, v850_data, v851_data, v852_data, v853_data);
          tensorforge::VectorT<float, 4> v858_acc{};
          float v859_data = r9[0];
          float v860_data = r9[1];
          float v861_data = r9[2];
          float v862_data = r9[3];
          tensorforge::VectorT<float, 4> v863_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v859_data, v858_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v864_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v860_data, v863_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v865_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v861_data, v864_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v866_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v862_data, v865_acc, 3, 0, 0);
          float v867_data = r9[4];
          float v868_data = r9[5];
          float v869_data = r9[6];
          float v870_data = r9[7];
          tensorforge::VectorT<float, 4> v871_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v867_data, v866_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v872_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v868_data, v871_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v873_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v869_data, v872_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v874_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v870_data, v873_acc, 3, 1, 0);
          float v875_data = r9[8];
          float v876_data = r9[9];
          float v877_data = r9[10];
          float v878_data = r9[11];
          tensorforge::VectorT<float, 4> v879_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v875_data, v874_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v876_data, v879_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v877_data, v880_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v878_data, v881_acc, 3, 2, 0);
          float v883_data = r9[12];
          float v884_data = r9[13];
          float v885_data = r9[14];
          float v886_data = r9[15];
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v883_data, v882_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v884_data, v887_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v885_data, v888_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v890_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v886_data, v889_acc, 3, 3, 0);
          float v891_data = r9[16];
          float v892_data = r9[17];
          float v893_data = r9[18];
          float v894_data = r9[19];
          tensorforge::VectorT<float, 4> v895_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v891_data, v890_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v892_data, v895_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v897_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v893_data, v896_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v898_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v894_data, v897_acc, 3, 4, 0);
          float v899_data = r9[20];
          float v900_data = r9[21];
          float v901_data = r9[22];
          float v902_data = r9[23];
          tensorforge::VectorT<float, 4> v903_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v899_data, v898_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v900_data, v903_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v901_data, v904_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v902_data, v905_acc, 3, 5, 0);
          float v907_data = r9[24];
          float v908_data = r9[25];
          float v909_data = r9[26];
          float v910_data = r9[27];
          tensorforge::VectorT<float, 4> v911_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v907_data, v906_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v912_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v908_data, v911_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v913_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v909_data, v912_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v914_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v910_data, v913_acc, 3, 6, 0);
          float v915_data = r9[28];
          float v916_data = r9[29];
          float v917_data = r9[30];
          float v918_data = r9[31];
          tensorforge::VectorT<float, 4> v919_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v915_data, v914_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v920_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v916_data, v919_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v921_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v917_data, v920_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v922_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v918_data, v921_acc, 3, 7, 0);
          r10[0] = (v922_acc[0]);
          r10[1] = (v922_acc[1]);
          r10[2] = (v922_acc[2]);
          r10[3] = (v922_acc[3]);
          float v927_data = r8[4];
          float v928_data = r8[5];
          float v929_data = r8[6];
          float v930_data = r8[7];
          float v931_tp{};
          float v932_tp{};
          float v933_tp{};
          float v934_tp{};
          tensorforge::transpose4x4b32(v931_tp, v932_tp, v933_tp, v934_tp, v927_data, v928_data, v929_data, v930_data);
          tensorforge::VectorT<float, 4> v935_acc{};
          tensorforge::VectorT<float, 4> v940_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v859_data, v935_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v941_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v860_data, v940_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v942_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v861_data, v941_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v862_data, v942_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v948_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v867_data, v943_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v949_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v868_data, v948_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v950_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v869_data, v949_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v951_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v870_data, v950_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v956_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v875_data, v951_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v957_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v876_data, v956_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v958_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v877_data, v957_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v959_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v878_data, v958_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v964_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v883_data, v959_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v884_data, v964_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v966_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v885_data, v965_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v886_data, v966_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v972_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v891_data, v967_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v973_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v892_data, v972_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v974_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v893_data, v973_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v975_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v894_data, v974_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v980_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v899_data, v975_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v981_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v900_data, v980_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v982_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v901_data, v981_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v983_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v902_data, v982_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v988_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v907_data, v983_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v908_data, v988_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v990_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v909_data, v989_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v910_data, v990_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v996_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v931_tp, v915_data, v991_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v997_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v932_tp, v916_data, v996_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v998_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v933_tp, v917_data, v997_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v999_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v934_tp, v918_data, v998_acc, 3, 7, 0);
          r10[4] = (v999_acc[0]);
          r10[5] = (v999_acc[1]);
          r10[6] = (v999_acc[2]);
          r10[7] = (v999_acc[3]);
          float v1004_data = r8[8];
          float v1005_data = r8[9];
          float v1006_data = r8[10];
          float v1007_data = r8[11];
          float v1008_tp{};
          float v1009_tp{};
          float v1010_tp{};
          float v1011_tp{};
          tensorforge::transpose4x4b32(v1008_tp, v1009_tp, v1010_tp, v1011_tp, v1004_data, v1005_data, v1006_data, v1007_data);
          tensorforge::VectorT<float, 4> v1012_acc{};
          tensorforge::VectorT<float, 4> v1017_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v859_data, v1012_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1018_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v860_data, v1017_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1019_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v861_data, v1018_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1020_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v862_data, v1019_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1025_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v867_data, v1020_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1026_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v868_data, v1025_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1027_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v869_data, v1026_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1028_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v870_data, v1027_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1033_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v875_data, v1028_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1034_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v876_data, v1033_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1035_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v877_data, v1034_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1036_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v878_data, v1035_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1041_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v883_data, v1036_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1042_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v884_data, v1041_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1043_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v885_data, v1042_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1044_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v886_data, v1043_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1049_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v891_data, v1044_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1050_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v892_data, v1049_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1051_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v893_data, v1050_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1052_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v894_data, v1051_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v899_data, v1052_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1058_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v900_data, v1057_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1059_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v901_data, v1058_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1060_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v902_data, v1059_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1065_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v907_data, v1060_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1066_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v908_data, v1065_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1067_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v909_data, v1066_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1068_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v910_data, v1067_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1073_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1008_tp, v915_data, v1068_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1074_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1009_tp, v916_data, v1073_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1075_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1010_tp, v917_data, v1074_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1076_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1011_tp, v918_data, v1075_acc, 3, 7, 0);
          r10[8] = (v1076_acc[0]);
          r10[9] = (v1076_acc[1]);
          r10[10] = (v1076_acc[2]);
          r10[11] = (v1076_acc[3]);
          float v1113_acc{};
          float v1114_data = r8[12];
          float v1115_bc = tensorforge::broadcast<32, 16, 0>(v1114_data);
          tensorforge::fmacdpp16<0>(v1113_acc, v1115_bc, v859_data);
          tensorforge::fmacdpp16<1>(v1113_acc, v1115_bc, v860_data);
          tensorforge::fmacdpp16<2>(v1113_acc, v1115_bc, v861_data);
          tensorforge::fmacdpp16<3>(v1113_acc, v1115_bc, v862_data);
          tensorforge::fmacdpp16<4>(v1113_acc, v1115_bc, v867_data);
          tensorforge::fmacdpp16<5>(v1113_acc, v1115_bc, v868_data);
          tensorforge::fmacdpp16<6>(v1113_acc, v1115_bc, v869_data);
          tensorforge::fmacdpp16<7>(v1113_acc, v1115_bc, v870_data);
          tensorforge::fmacdpp16<8>(v1113_acc, v1115_bc, v875_data);
          tensorforge::fmacdpp16<9>(v1113_acc, v1115_bc, v876_data);
          tensorforge::fmacdpp16<10>(v1113_acc, v1115_bc, v877_data);
          tensorforge::fmacdpp16<11>(v1113_acc, v1115_bc, v878_data);
          tensorforge::fmacdpp16<12>(v1113_acc, v1115_bc, v883_data);
          tensorforge::fmacdpp16<13>(v1113_acc, v1115_bc, v884_data);
          tensorforge::fmacdpp16<14>(v1113_acc, v1115_bc, v885_data);
          tensorforge::fmacdpp16<15>(v1113_acc, v1115_bc, v886_data);
          float v1116_bc = tensorforge::broadcast<32, 16, 1>(v1114_data);
          tensorforge::fmacdpp16<0>(v1113_acc, v1116_bc, v891_data);
          tensorforge::fmacdpp16<1>(v1113_acc, v1116_bc, v892_data);
          tensorforge::fmacdpp16<2>(v1113_acc, v1116_bc, v893_data);
          tensorforge::fmacdpp16<3>(v1113_acc, v1116_bc, v894_data);
          tensorforge::fmacdpp16<4>(v1113_acc, v1116_bc, v899_data);
          tensorforge::fmacdpp16<5>(v1113_acc, v1116_bc, v900_data);
          tensorforge::fmacdpp16<6>(v1113_acc, v1116_bc, v901_data);
          tensorforge::fmacdpp16<7>(v1113_acc, v1116_bc, v902_data);
          tensorforge::fmacdpp16<8>(v1113_acc, v1116_bc, v907_data);
          tensorforge::fmacdpp16<9>(v1113_acc, v1116_bc, v908_data);
          tensorforge::fmacdpp16<10>(v1113_acc, v1116_bc, v909_data);
          tensorforge::fmacdpp16<11>(v1113_acc, v1116_bc, v910_data);
          tensorforge::fmacdpp16<12>(v1113_acc, v1116_bc, v915_data);
          tensorforge::fmacdpp16<13>(v1113_acc, v1116_bc, v916_data);
          tensorforge::fmacdpp16<14>(v1113_acc, v1116_bc, v917_data);
          tensorforge::fmacdpp16<15>(v1113_acc, v1116_bc, v918_data);
          r10[12] = v1113_acc;
          // glb_m0 = store{r>g}(r10);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v1121_i1 = 0; v1121_i1 < 13; ++v1121_i1) {
              float v1123_data = r10[v1121_i1];
              int32_t v1130_a = v22_lead + (v1121_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1130_a], v1123_data);
            }
          }
          float r13[32]{};
          // r13 = load{g>r}(glb_m8);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v1136_i1 = 0; v1136_i1 < 32; ++v1136_i1) {
              float v1144_data = __builtin_nontemporal_load(&glb_m8[(v22_lead + (v1136_i1 * 16))]);
              r13[v1136_i1] = v1144_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m7););
          float r12[13]{};
          // r12 = +(r0 * r11) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1147_data = r11[0];
          float v1148_data = r11[1];
          float v1149_data = r11[2];
          float v1150_data = r11[3];
          float v1151_tp{};
          float v1152_tp{};
          float v1153_tp{};
          float v1154_tp{};
          tensorforge::transpose4x4b32(v1151_tp, v1152_tp, v1153_tp, v1154_tp, v1147_data, v1148_data, v1149_data, v1150_data);
          tensorforge::VectorT<float, 4> v1155_acc{};
          tensorforge::VectorT<float, 4> v1160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1151_tp, v74_data, v1155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1152_tp, v75_data, v1160_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1153_tp, v76_data, v1161_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1154_tp, v77_data, v1162_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1151_tp, v82_data, v1163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1152_tp, v83_data, v1168_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1153_tp, v84_data, v1169_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1154_tp, v85_data, v1170_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1151_tp, v90_data, v1171_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1152_tp, v91_data, v1176_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1153_tp, v92_data, v1177_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1154_tp, v93_data, v1178_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1151_tp, v98_data, v1179_acc, 3, 3, 0);
          r12[0] = (v1184_acc[0]);
          r12[1] = (v1184_acc[1]);
          r12[2] = (v1184_acc[2]);
          r12[3] = (v1184_acc[3]);
          float v1189_data = r11[4];
          float v1190_data = r11[5];
          float v1191_data = r11[6];
          float v1192_data = r11[7];
          float v1193_tp{};
          float v1194_tp{};
          float v1195_tp{};
          float v1196_tp{};
          tensorforge::transpose4x4b32(v1193_tp, v1194_tp, v1195_tp, v1196_tp, v1189_data, v1190_data, v1191_data, v1192_data);
          tensorforge::VectorT<float, 4> v1197_acc{};
          tensorforge::VectorT<float, 4> v1202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1193_tp, v74_data, v1197_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1194_tp, v75_data, v1202_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1195_tp, v76_data, v1203_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1196_tp, v77_data, v1204_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1193_tp, v82_data, v1205_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1194_tp, v83_data, v1210_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1195_tp, v84_data, v1211_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1196_tp, v85_data, v1212_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1193_tp, v90_data, v1213_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1194_tp, v91_data, v1218_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1195_tp, v92_data, v1219_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1196_tp, v93_data, v1220_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1193_tp, v98_data, v1221_acc, 3, 3, 0);
          r12[4] = (v1226_acc[0]);
          r12[5] = (v1226_acc[1]);
          r12[6] = (v1226_acc[2]);
          r12[7] = (v1226_acc[3]);
          float v1231_data = r11[8];
          float v1232_data = r11[9];
          float v1233_data = r11[10];
          float v1234_data = r11[11];
          float v1235_tp{};
          float v1236_tp{};
          float v1237_tp{};
          float v1238_tp{};
          tensorforge::transpose4x4b32(v1235_tp, v1236_tp, v1237_tp, v1238_tp, v1231_data, v1232_data, v1233_data, v1234_data);
          tensorforge::VectorT<float, 4> v1239_acc{};
          tensorforge::VectorT<float, 4> v1244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1235_tp, v74_data, v1239_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1236_tp, v75_data, v1244_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1237_tp, v76_data, v1245_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1238_tp, v77_data, v1246_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1235_tp, v82_data, v1247_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1236_tp, v83_data, v1252_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1237_tp, v84_data, v1253_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1238_tp, v85_data, v1254_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1235_tp, v90_data, v1255_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1236_tp, v91_data, v1260_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1237_tp, v92_data, v1261_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1238_tp, v93_data, v1262_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1235_tp, v98_data, v1263_acc, 3, 3, 0);
          r12[8] = (v1268_acc[0]);
          r12[9] = (v1268_acc[1]);
          r12[10] = (v1268_acc[2]);
          r12[11] = (v1268_acc[3]);
          float v1286_acc{};
          float v1287_data = r11[12];
          float v1288_bc = tensorforge::broadcast<32, 16, 0>(v1287_data);
          tensorforge::fmacdpp16<0>(v1286_acc, v1288_bc, v74_data);
          tensorforge::fmacdpp16<1>(v1286_acc, v1288_bc, v75_data);
          tensorforge::fmacdpp16<2>(v1286_acc, v1288_bc, v76_data);
          tensorforge::fmacdpp16<3>(v1286_acc, v1288_bc, v77_data);
          tensorforge::fmacdpp16<4>(v1286_acc, v1288_bc, v82_data);
          tensorforge::fmacdpp16<5>(v1286_acc, v1288_bc, v83_data);
          tensorforge::fmacdpp16<6>(v1286_acc, v1288_bc, v84_data);
          tensorforge::fmacdpp16<7>(v1286_acc, v1288_bc, v85_data);
          tensorforge::fmacdpp16<8>(v1286_acc, v1288_bc, v90_data);
          tensorforge::fmacdpp16<9>(v1286_acc, v1288_bc, v91_data);
          tensorforge::fmacdpp16<10>(v1286_acc, v1288_bc, v92_data);
          tensorforge::fmacdpp16<11>(v1286_acc, v1288_bc, v93_data);
          tensorforge::fmacdpp16<12>(v1286_acc, v1288_bc, v98_data);
          r12[12] = v1286_acc;
          // wait(r13 = load{g>r}(glb_m8););
          float r14[13]{};
          // r14 = +(r13 * r12) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v1290_data = r12[0];
          float v1291_data = r12[1];
          float v1292_data = r12[2];
          float v1293_data = r12[3];
          float v1294_tp{};
          float v1295_tp{};
          float v1296_tp{};
          float v1297_tp{};
          tensorforge::transpose4x4b32(v1294_tp, v1295_tp, v1296_tp, v1297_tp, v1290_data, v1291_data, v1292_data, v1293_data);
          tensorforge::VectorT<float, 4> v1298_acc{};
          float v1299_data = r13[0];
          float v1300_data = r13[1];
          float v1301_data = r13[2];
          float v1302_data = r13[3];
          tensorforge::VectorT<float, 4> v1303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1299_data, v1298_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1300_data, v1303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1301_data, v1304_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1302_data, v1305_acc, 3, 0, 0);
          float v1307_data = r13[4];
          float v1308_data = r13[5];
          float v1309_data = r13[6];
          float v1310_data = r13[7];
          tensorforge::VectorT<float, 4> v1311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1307_data, v1306_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1308_data, v1311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1309_data, v1312_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1310_data, v1313_acc, 3, 1, 0);
          float v1315_data = r13[8];
          float v1316_data = r13[9];
          float v1317_data = r13[10];
          float v1318_data = r13[11];
          tensorforge::VectorT<float, 4> v1319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1315_data, v1314_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1316_data, v1319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1317_data, v1320_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1318_data, v1321_acc, 3, 2, 0);
          float v1323_data = r13[12];
          float v1324_data = r13[13];
          float v1325_data = r13[14];
          float v1326_data = r13[15];
          tensorforge::VectorT<float, 4> v1327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1323_data, v1322_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1324_data, v1327_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1325_data, v1328_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1326_data, v1329_acc, 3, 3, 0);
          float v1331_data = r13[16];
          float v1332_data = r13[17];
          float v1333_data = r13[18];
          float v1334_data = r13[19];
          tensorforge::VectorT<float, 4> v1335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1331_data, v1330_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1332_data, v1335_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1333_data, v1336_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1334_data, v1337_acc, 3, 4, 0);
          float v1339_data = r13[20];
          float v1340_data = r13[21];
          float v1341_data = r13[22];
          float v1342_data = r13[23];
          tensorforge::VectorT<float, 4> v1343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1339_data, v1338_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1340_data, v1343_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1341_data, v1344_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1342_data, v1345_acc, 3, 5, 0);
          float v1347_data = r13[24];
          float v1348_data = r13[25];
          float v1349_data = r13[26];
          float v1350_data = r13[27];
          tensorforge::VectorT<float, 4> v1351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1347_data, v1346_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1348_data, v1351_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1349_data, v1352_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1350_data, v1353_acc, 3, 6, 0);
          float v1355_data = r13[28];
          float v1356_data = r13[29];
          float v1357_data = r13[30];
          float v1358_data = r13[31];
          tensorforge::VectorT<float, 4> v1359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1294_tp, v1355_data, v1354_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1295_tp, v1356_data, v1359_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1296_tp, v1357_data, v1360_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1297_tp, v1358_data, v1361_acc, 3, 7, 0);
          r14[0] = (v1362_acc[0]);
          r14[1] = (v1362_acc[1]);
          r14[2] = (v1362_acc[2]);
          r14[3] = (v1362_acc[3]);
          float v1367_data = r12[4];
          float v1368_data = r12[5];
          float v1369_data = r12[6];
          float v1370_data = r12[7];
          float v1371_tp{};
          float v1372_tp{};
          float v1373_tp{};
          float v1374_tp{};
          tensorforge::transpose4x4b32(v1371_tp, v1372_tp, v1373_tp, v1374_tp, v1367_data, v1368_data, v1369_data, v1370_data);
          tensorforge::VectorT<float, 4> v1375_acc{};
          tensorforge::VectorT<float, 4> v1380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1299_data, v1375_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1300_data, v1380_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1301_data, v1381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1302_data, v1382_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1307_data, v1383_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1308_data, v1388_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1309_data, v1389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1310_data, v1390_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1315_data, v1391_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1316_data, v1396_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1317_data, v1397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1318_data, v1398_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1323_data, v1399_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1324_data, v1404_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1325_data, v1405_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1326_data, v1406_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1331_data, v1407_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1332_data, v1412_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1333_data, v1413_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1334_data, v1414_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1339_data, v1415_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1340_data, v1420_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1341_data, v1421_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1342_data, v1422_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1347_data, v1423_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1348_data, v1428_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1349_data, v1429_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1350_data, v1430_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1371_tp, v1355_data, v1431_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1372_tp, v1356_data, v1436_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1373_tp, v1357_data, v1437_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1374_tp, v1358_data, v1438_acc, 3, 7, 0);
          r14[4] = (v1439_acc[0]);
          r14[5] = (v1439_acc[1]);
          r14[6] = (v1439_acc[2]);
          r14[7] = (v1439_acc[3]);
          float v1444_data = r12[8];
          float v1445_data = r12[9];
          float v1446_data = r12[10];
          float v1447_data = r12[11];
          float v1448_tp{};
          float v1449_tp{};
          float v1450_tp{};
          float v1451_tp{};
          tensorforge::transpose4x4b32(v1448_tp, v1449_tp, v1450_tp, v1451_tp, v1444_data, v1445_data, v1446_data, v1447_data);
          tensorforge::VectorT<float, 4> v1452_acc{};
          tensorforge::VectorT<float, 4> v1457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1299_data, v1452_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1300_data, v1457_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1301_data, v1458_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1302_data, v1459_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1307_data, v1460_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1308_data, v1465_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1309_data, v1466_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1310_data, v1467_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1315_data, v1468_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1316_data, v1473_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1317_data, v1474_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1318_data, v1475_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1323_data, v1476_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1324_data, v1481_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1325_data, v1482_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1326_data, v1483_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1331_data, v1484_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1332_data, v1489_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1333_data, v1490_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1334_data, v1491_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1339_data, v1492_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1340_data, v1497_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1341_data, v1498_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1342_data, v1499_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1347_data, v1500_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1348_data, v1505_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1349_data, v1506_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1350_data, v1507_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1448_tp, v1355_data, v1508_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1449_tp, v1356_data, v1513_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1450_tp, v1357_data, v1514_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1451_tp, v1358_data, v1515_acc, 3, 7, 0);
          r14[8] = (v1516_acc[0]);
          r14[9] = (v1516_acc[1]);
          r14[10] = (v1516_acc[2]);
          r14[11] = (v1516_acc[3]);
          float v1553_acc{};
          float v1554_data = r12[12];
          float v1555_bc = tensorforge::broadcast<32, 16, 0>(v1554_data);
          tensorforge::fmacdpp16<0>(v1553_acc, v1555_bc, v1299_data);
          tensorforge::fmacdpp16<1>(v1553_acc, v1555_bc, v1300_data);
          tensorforge::fmacdpp16<2>(v1553_acc, v1555_bc, v1301_data);
          tensorforge::fmacdpp16<3>(v1553_acc, v1555_bc, v1302_data);
          tensorforge::fmacdpp16<4>(v1553_acc, v1555_bc, v1307_data);
          tensorforge::fmacdpp16<5>(v1553_acc, v1555_bc, v1308_data);
          tensorforge::fmacdpp16<6>(v1553_acc, v1555_bc, v1309_data);
          tensorforge::fmacdpp16<7>(v1553_acc, v1555_bc, v1310_data);
          tensorforge::fmacdpp16<8>(v1553_acc, v1555_bc, v1315_data);
          tensorforge::fmacdpp16<9>(v1553_acc, v1555_bc, v1316_data);
          tensorforge::fmacdpp16<10>(v1553_acc, v1555_bc, v1317_data);
          tensorforge::fmacdpp16<11>(v1553_acc, v1555_bc, v1318_data);
          tensorforge::fmacdpp16<12>(v1553_acc, v1555_bc, v1323_data);
          tensorforge::fmacdpp16<13>(v1553_acc, v1555_bc, v1324_data);
          tensorforge::fmacdpp16<14>(v1553_acc, v1555_bc, v1325_data);
          tensorforge::fmacdpp16<15>(v1553_acc, v1555_bc, v1326_data);
          float v1556_bc = tensorforge::broadcast<32, 16, 1>(v1554_data);
          tensorforge::fmacdpp16<0>(v1553_acc, v1556_bc, v1331_data);
          tensorforge::fmacdpp16<1>(v1553_acc, v1556_bc, v1332_data);
          tensorforge::fmacdpp16<2>(v1553_acc, v1556_bc, v1333_data);
          tensorforge::fmacdpp16<3>(v1553_acc, v1556_bc, v1334_data);
          tensorforge::fmacdpp16<4>(v1553_acc, v1556_bc, v1339_data);
          tensorforge::fmacdpp16<5>(v1553_acc, v1556_bc, v1340_data);
          tensorforge::fmacdpp16<6>(v1553_acc, v1556_bc, v1341_data);
          tensorforge::fmacdpp16<7>(v1553_acc, v1556_bc, v1342_data);
          tensorforge::fmacdpp16<8>(v1553_acc, v1556_bc, v1347_data);
          tensorforge::fmacdpp16<9>(v1553_acc, v1556_bc, v1348_data);
          tensorforge::fmacdpp16<10>(v1553_acc, v1556_bc, v1349_data);
          tensorforge::fmacdpp16<11>(v1553_acc, v1556_bc, v1350_data);
          tensorforge::fmacdpp16<12>(v1553_acc, v1556_bc, v1355_data);
          tensorforge::fmacdpp16<13>(v1553_acc, v1556_bc, v1356_data);
          tensorforge::fmacdpp16<14>(v1553_acc, v1556_bc, v1357_data);
          tensorforge::fmacdpp16<15>(v1553_acc, v1556_bc, v1358_data);
          r14[12] = v1553_acc;
          // glb_m0 = store{r>g}(r14);
          if (v22_lead < 16) {
            #pragma unroll
            for (int32_t v1561_i1 = 0; v1561_i1 < 13; ++v1561_i1) {
              float v1563_data = r14[v1561_i1];
              int32_t v1570_a = v22_lead + (v1561_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1570_a], v1563_data);
            }
          }
          float r15[13]{};
          // r15 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1575_i0 = 0; v1575_i0 < 1; ++v1575_i0) {
            int32_t v1581_lead = v22_lead + (v1575_i0 * 32);
            #pragma unroll
            for (int32_t v1576_i1 = 0; v1576_i1 < 13; ++v1576_i1) {
              float v1584_data = glb_m0[(v1581_lead + (v1576_i1 * 32))];
              r15[(v1575_i0 + v1576_i1)] = v1584_data;
            }
          }
          float r16[13]{};
          // r16 = load{g>r}(glb_m10);
          if (v22_lead < 13) {
            #pragma unroll
            for (int32_t v1591_i1 = 0; v1591_i1 < 13; ++v1591_i1) {
              float v1599_data = __builtin_nontemporal_load(&glb_m10[(v22_lead + (v1591_i1 * 13))]);
              r16[v1591_i1] = v1599_data;
            }
          }
          // wait(r15 = load{g>r}(glb_m0););
          // wait(r16 = load{g>r}(glb_m10););
          float r17[13]{};
          // r17 = +(r15 * r16) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1602_data = r16[0];
          float v1603_data = r16[1];
          float v1604_data = r16[2];
          float v1605_data = r16[3];
          float v1606_tp{};
          float v1607_tp{};
          float v1608_tp{};
          float v1609_tp{};
          tensorforge::transpose4x4b32(v1606_tp, v1607_tp, v1608_tp, v1609_tp, v1602_data, v1603_data, v1604_data, v1605_data);
          tensorforge::VectorT<float, 4> v1610_acc{};
          float v1611_data = r15[0];
          float v1612_data = r15[1];
          float v1613_data = r15[2];
          float v1614_data = r15[3];
          tensorforge::VectorT<float, 4> v1615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1606_tp, v1611_data, v1610_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1607_tp, v1612_data, v1615_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1608_tp, v1613_data, v1616_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1609_tp, v1614_data, v1617_acc, 3, 0, 0);
          float v1619_data = r15[4];
          float v1620_data = r15[5];
          float v1621_data = r15[6];
          float v1622_data = r15[7];
          tensorforge::VectorT<float, 4> v1623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1606_tp, v1619_data, v1618_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1607_tp, v1620_data, v1623_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1608_tp, v1621_data, v1624_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1609_tp, v1622_data, v1625_acc, 3, 1, 0);
          float v1627_data = r15[8];
          float v1628_data = r15[9];
          float v1629_data = r15[10];
          float v1630_data = r15[11];
          tensorforge::VectorT<float, 4> v1631_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1606_tp, v1627_data, v1626_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1607_tp, v1628_data, v1631_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1608_tp, v1629_data, v1632_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1609_tp, v1630_data, v1633_acc, 3, 2, 0);
          float v1635_data = r15[12];
          tensorforge::VectorT<float, 4> v1639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1606_tp, v1635_data, v1634_acc, 3, 3, 0);
          r17[0] = (v1639_acc[0]);
          r17[1] = (v1639_acc[1]);
          r17[2] = (v1639_acc[2]);
          r17[3] = (v1639_acc[3]);
          float v1644_data = r16[4];
          float v1645_data = r16[5];
          float v1646_data = r16[6];
          float v1647_data = r16[7];
          float v1648_tp{};
          float v1649_tp{};
          float v1650_tp{};
          float v1651_tp{};
          tensorforge::transpose4x4b32(v1648_tp, v1649_tp, v1650_tp, v1651_tp, v1644_data, v1645_data, v1646_data, v1647_data);
          tensorforge::VectorT<float, 4> v1652_acc{};
          tensorforge::VectorT<float, 4> v1657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1648_tp, v1611_data, v1652_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1649_tp, v1612_data, v1657_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1650_tp, v1613_data, v1658_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1651_tp, v1614_data, v1659_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1648_tp, v1619_data, v1660_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1649_tp, v1620_data, v1665_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1650_tp, v1621_data, v1666_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1651_tp, v1622_data, v1667_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1673_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1648_tp, v1627_data, v1668_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1649_tp, v1628_data, v1673_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1650_tp, v1629_data, v1674_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1651_tp, v1630_data, v1675_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1648_tp, v1635_data, v1676_acc, 3, 3, 0);
          r17[4] = (v1681_acc[0]);
          r17[5] = (v1681_acc[1]);
          r17[6] = (v1681_acc[2]);
          r17[7] = (v1681_acc[3]);
          float v1686_data = r16[8];
          float v1687_data = r16[9];
          float v1688_data = r16[10];
          float v1689_data = r16[11];
          float v1690_tp{};
          float v1691_tp{};
          float v1692_tp{};
          float v1693_tp{};
          tensorforge::transpose4x4b32(v1690_tp, v1691_tp, v1692_tp, v1693_tp, v1686_data, v1687_data, v1688_data, v1689_data);
          tensorforge::VectorT<float, 4> v1694_acc{};
          tensorforge::VectorT<float, 4> v1699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1690_tp, v1611_data, v1694_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1691_tp, v1612_data, v1699_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1692_tp, v1613_data, v1700_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1693_tp, v1614_data, v1701_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1690_tp, v1619_data, v1702_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1691_tp, v1620_data, v1707_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1692_tp, v1621_data, v1708_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1693_tp, v1622_data, v1709_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1715_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1690_tp, v1627_data, v1710_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1691_tp, v1628_data, v1715_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1717_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1692_tp, v1629_data, v1716_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1693_tp, v1630_data, v1717_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1690_tp, v1635_data, v1718_acc, 3, 3, 0);
          r17[8] = (v1723_acc[0]);
          r17[9] = (v1723_acc[1]);
          r17[10] = (v1723_acc[2]);
          r17[11] = (v1723_acc[3]);
          float v1741_acc{};
          float v1742_data = r16[12];
          float v1743_bc = tensorforge::broadcast<32, 16, 0>(v1742_data);
          tensorforge::fmacdpp16<0>(v1741_acc, v1743_bc, v1611_data);
          tensorforge::fmacdpp16<1>(v1741_acc, v1743_bc, v1612_data);
          tensorforge::fmacdpp16<2>(v1741_acc, v1743_bc, v1613_data);
          tensorforge::fmacdpp16<3>(v1741_acc, v1743_bc, v1614_data);
          tensorforge::fmacdpp16<4>(v1741_acc, v1743_bc, v1619_data);
          tensorforge::fmacdpp16<5>(v1741_acc, v1743_bc, v1620_data);
          tensorforge::fmacdpp16<6>(v1741_acc, v1743_bc, v1621_data);
          tensorforge::fmacdpp16<7>(v1741_acc, v1743_bc, v1622_data);
          tensorforge::fmacdpp16<8>(v1741_acc, v1743_bc, v1627_data);
          tensorforge::fmacdpp16<9>(v1741_acc, v1743_bc, v1628_data);
          tensorforge::fmacdpp16<10>(v1741_acc, v1743_bc, v1629_data);
          tensorforge::fmacdpp16<11>(v1741_acc, v1743_bc, v1630_data);
          tensorforge::fmacdpp16<12>(v1741_acc, v1743_bc, v1635_data);
          r17[12] = v1741_acc;
          // glb_m9 = store{r>g}(r17);
          #pragma unroll
          for (int32_t v1747_i0 = 0; v1747_i0 < 1; ++v1747_i0) {
            int32_t v1755_lead = v22_lead + (v1747_i0 * 32);
            #pragma unroll
            for (int32_t v1748_i1 = 0; v1748_i1 < 13; ++v1748_i1) {
              float v1750_data = r17[(v1747_i0 + v1748_i1)];
              glb_m9[(v1755_lead + (v1748_i1 * 32))] = v1750_data;
            }
          }
        }
      }
    }
  }
}

