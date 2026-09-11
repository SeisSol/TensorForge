// === base name ===
kernel_17a31762e17bc03e

// === header ===
void launcher_kernel_17a31762e17bc03e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_17a31762e17bc03e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_17a31762e17bc03e, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_17a31762e17bc03e, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_17a31762e17bc03e), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_17a31762e17bc03e, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_17a31762e17bc03e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 416 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 10; v18_i1 < 13; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v18_i1 * 32))]);
              r0[(v17_i0 + (v18_i1 - 10))] = v26_data;
            }
          }
          float r1[1]{};
          // r1 = load{g>r}(glb_m2);
          if ((v16_lead >= 10) && (v16_lead < 13)) {
            #pragma unroll
            for (int32_t v36_i1 = 8; v36_i1 < 9; ++v36_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v36_i1 * 13))]);
              r1[(v36_i1 - 8)] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v48_data = r0[0];
          float v49_data = r0[1];
          float v50_data = r0[2];
          float v51_acc{};
          float v52_data = r1[0];
          float v53_bc = tensorforge::broadcast<32, 16, 0>(v52_data);
          tensorforge::fmacdpp16<10>(v51_acc, v53_bc, v48_data);
          tensorforge::fmacdpp16<11>(v51_acc, v53_bc, v49_data);
          tensorforge::fmacdpp16<12>(v51_acc, v53_bc, v50_data);
          r2[0] = v51_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v57_i0 = 0; v57_i0 < 1; ++v57_i0) {
            int32_t v65_lead = v16_lead + (v57_i0 * 32);
            #pragma unroll
            for (int32_t v58_i1 = 0; v58_i1 < 1; ++v58_i1) {
              float v60_data = r2[(v57_i0 + v58_i1)];
              glb_m0[(v65_lead + ((v58_i1 + 8) * 32))] = v60_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
            int32_t v79_lead = v16_lead + (v73_i0 * 32);
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 13; ++v74_i1) {
              float v82_data = glb_m0[(v79_lead + (v74_i1 * 32))];
              r3[(v73_i0 + v74_i1)] = v82_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          if (v16_lead < 13) {
            #pragma unroll
            for (int32_t v89_i1 = 0; v89_i1 < 13; ++v89_i1) {
              float v97_data = __builtin_nontemporal_load(&glb_m4[(v16_lead + (v89_i1 * 13))]);
              r4[v89_i1] = v97_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v100_data = r4[0];
          float v101_data = r4[1];
          float v102_data = r4[2];
          float v103_data = r4[3];
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          float v107_tp{};
          tensorforge::transpose4x4b32(v104_tp, v105_tp, v106_tp, v107_tp, v100_data, v101_data, v102_data, v103_data);
          tensorforge::VectorT<float, 4> v108_acc{};
          float v109_data = r3[0];
          float v110_data = r3[1];
          float v111_data = r3[2];
          float v112_data = r3[3];
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v109_data, v108_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v110_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v111_data, v114_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v112_data, v115_acc, 3, 0, 0);
          float v117_data = r3[4];
          float v118_data = r3[5];
          float v119_data = r3[6];
          float v120_data = r3[7];
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v117_data, v116_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v118_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v119_data, v122_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v120_data, v123_acc, 3, 1, 0);
          float v125_data = r3[8];
          float v126_data = r3[9];
          float v127_data = r3[10];
          float v128_data = r3[11];
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v125_data, v124_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v126_data, v129_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v127_data, v130_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v128_data, v131_acc, 3, 2, 0);
          float v133_data = r3[12];
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v133_data, v132_acc, 3, 3, 0);
          r5[0] = (v137_acc[0]);
          r5[1] = (v137_acc[1]);
          r5[2] = (v137_acc[2]);
          r5[3] = (v137_acc[3]);
          float v142_data = r4[4];
          float v143_data = r4[5];
          float v144_data = r4[6];
          float v145_data = r4[7];
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          float v149_tp{};
          tensorforge::transpose4x4b32(v146_tp, v147_tp, v148_tp, v149_tp, v142_data, v143_data, v144_data, v145_data);
          tensorforge::VectorT<float, 4> v150_acc{};
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v109_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v110_data, v155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v111_data, v156_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v112_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v117_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v118_data, v163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v119_data, v164_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v120_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v125_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v126_data, v171_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v127_data, v172_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v128_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v133_data, v174_acc, 3, 3, 0);
          r5[4] = (v179_acc[0]);
          r5[5] = (v179_acc[1]);
          r5[6] = (v179_acc[2]);
          r5[7] = (v179_acc[3]);
          float v184_data = r4[8];
          float v185_data = r4[9];
          float v186_data = r4[10];
          float v187_data = r4[11];
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          tensorforge::transpose4x4b32(v188_tp, v189_tp, v190_tp, v191_tp, v184_data, v185_data, v186_data, v187_data);
          tensorforge::VectorT<float, 4> v192_acc{};
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v109_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v110_data, v197_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v111_data, v198_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v112_data, v199_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v117_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v118_data, v205_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v119_data, v206_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v120_data, v207_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v125_data, v208_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v126_data, v213_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v127_data, v214_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v128_data, v215_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v133_data, v216_acc, 3, 3, 0);
          r5[8] = (v221_acc[0]);
          r5[9] = (v221_acc[1]);
          r5[10] = (v221_acc[2]);
          r5[11] = (v221_acc[3]);
          float v239_acc{};
          float v240_data = r4[12];
          float v241_bc = tensorforge::broadcast<32, 16, 0>(v240_data);
          tensorforge::fmacdpp16<0>(v239_acc, v241_bc, v109_data);
          tensorforge::fmacdpp16<1>(v239_acc, v241_bc, v110_data);
          tensorforge::fmacdpp16<2>(v239_acc, v241_bc, v111_data);
          tensorforge::fmacdpp16<3>(v239_acc, v241_bc, v112_data);
          tensorforge::fmacdpp16<4>(v239_acc, v241_bc, v117_data);
          tensorforge::fmacdpp16<5>(v239_acc, v241_bc, v118_data);
          tensorforge::fmacdpp16<6>(v239_acc, v241_bc, v119_data);
          tensorforge::fmacdpp16<7>(v239_acc, v241_bc, v120_data);
          tensorforge::fmacdpp16<8>(v239_acc, v241_bc, v125_data);
          tensorforge::fmacdpp16<9>(v239_acc, v241_bc, v126_data);
          tensorforge::fmacdpp16<10>(v239_acc, v241_bc, v127_data);
          tensorforge::fmacdpp16<11>(v239_acc, v241_bc, v128_data);
          tensorforge::fmacdpp16<12>(v239_acc, v241_bc, v133_data);
          r5[12] = v239_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v245_i0 = 0; v245_i0 < 1; ++v245_i0) {
            int32_t v253_lead = v16_lead + (v245_i0 * 32);
            #pragma unroll
            for (int32_t v246_i1 = 0; v246_i1 < 13; ++v246_i1) {
              float v248_data = r5[(v245_i0 + v246_i1)];
              glb_m3[(v253_lead + (v246_i1 * 32))] = v248_data;
            }
          }
        }
      }
    }
  }
}

