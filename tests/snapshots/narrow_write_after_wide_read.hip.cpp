// === base name ===
kernel_c8ed1ac0b6b57c54

// === header ===
void launcher_kernel_c8ed1ac0b6b57c54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c8ed1ac0b6b57c54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c8ed1ac0b6b57c54, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c8ed1ac0b6b57c54, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c8ed1ac0b6b57c54), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_c8ed1ac0b6b57c54, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c8ed1ac0b6b57c54(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×13(12×13) {0..12}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1]
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] += m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×13(12×13) {0..12}×{0..13} strided({0..12}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1]
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 156 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 416 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 13; ++v18_i1) {
              float v26_data = glb_m0[(v23_lead + (v18_i1 * 32))];
              r0[(v17_i0 + v18_i1)] = v26_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
            int32_t v38_lead = v16_lead + (v32_i0 * 32);
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v38_lead + (v33_i1 * 32))]);
              r2[(v32_i0 + v33_i1)] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v47_data = r0[0];
          float v48_data = r1[0];
          r1[0] = (v48_data + v47_data);
          float v50_data = r0[1];
          float v51_data = r1[1];
          r1[1] = (v51_data + v50_data);
          float v53_data = r0[2];
          float v54_data = r1[2];
          r1[2] = (v54_data + v53_data);
          float v56_data = r0[3];
          float v57_data = r1[3];
          r1[3] = (v57_data + v56_data);
          float v59_data = r0[4];
          float v60_data = r1[4];
          r1[4] = (v60_data + v59_data);
          float v62_data = r0[5];
          float v63_data = r1[5];
          r1[5] = (v63_data + v62_data);
          float v65_data = r0[6];
          float v66_data = r1[6];
          r1[6] = (v66_data + v65_data);
          float v68_data = r0[7];
          float v69_data = r1[7];
          r1[7] = (v69_data + v68_data);
          float v71_data = r0[8];
          float v72_data = r1[8];
          r1[8] = (v72_data + v71_data);
          float v74_data = r0[9];
          float v75_data = r1[9];
          r1[9] = (v75_data + v74_data);
          float v77_data = r0[10];
          float v78_data = r1[10];
          r1[10] = (v78_data + v77_data);
          float v80_data = r0[11];
          float v81_data = r1[11];
          r1[11] = (v81_data + v80_data);
          float v83_data = r0[12];
          float v84_data = r1[12];
          r1[12] = (v84_data + v83_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v91_i1 = 0; v91_i1 < 13; ++v91_i1) {
              float v99_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v91_i1 * 12))]);
              r3[v91_i1] = v99_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v103_data = r3[0];
          float v104_data = r3[1];
          float v105_data = r3[2];
          float v106_data = r3[3];
          float v107_tp{};
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          tensorforge::transpose4x4b32(v107_tp, v108_tp, v109_tp, v110_tp, v103_data, v104_data, v105_data, v106_data);
          tensorforge::VectorT<float, 4> v111_acc{};
          float v112_data = r2[0];
          float v113_data = r2[1];
          float v114_data = r2[2];
          float v115_data = r2[3];
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v112_data, v111_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v113_data, v116_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v114_data, v117_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v115_data, v118_acc, 3, 0, 0);
          float v120_data = r2[4];
          float v121_data = r2[5];
          float v122_data = r2[6];
          float v123_data = r2[7];
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v120_data, v119_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v121_data, v124_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v122_data, v125_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v123_data, v126_acc, 3, 1, 0);
          float v128_data = r2[8];
          float v129_data = r2[9];
          float v130_data = r2[10];
          float v131_data = r2[11];
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v128_data, v127_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v129_data, v132_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v130_data, v133_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v131_data, v134_acc, 3, 2, 0);
          ir4[0] = (v135_acc[0]);
          ir4[1] = (v135_acc[1]);
          ir4[2] = (v135_acc[2]);
          ir4[3] = (v135_acc[3]);
          float v140_data = r3[4];
          float v141_data = r3[5];
          float v142_data = r3[6];
          float v143_data = r3[7];
          float v144_tp{};
          float v145_tp{};
          float v146_tp{};
          float v147_tp{};
          tensorforge::transpose4x4b32(v144_tp, v145_tp, v146_tp, v147_tp, v140_data, v141_data, v142_data, v143_data);
          tensorforge::VectorT<float, 4> v148_acc{};
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v112_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v113_data, v153_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v114_data, v154_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v115_data, v155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v120_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v121_data, v161_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v122_data, v162_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v123_data, v163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v128_data, v164_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v129_data, v169_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v130_data, v170_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v131_data, v171_acc, 3, 2, 0);
          ir4[4] = (v172_acc[0]);
          ir4[5] = (v172_acc[1]);
          ir4[6] = (v172_acc[2]);
          ir4[7] = (v172_acc[3]);
          float v177_data = r3[8];
          float v178_data = r3[9];
          float v179_data = r3[10];
          float v180_data = r3[11];
          float v181_tp{};
          float v182_tp{};
          float v183_tp{};
          float v184_tp{};
          tensorforge::transpose4x4b32(v181_tp, v182_tp, v183_tp, v184_tp, v177_data, v178_data, v179_data, v180_data);
          tensorforge::VectorT<float, 4> v185_acc{};
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v112_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v113_data, v190_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v114_data, v191_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v115_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v120_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v121_data, v198_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v122_data, v199_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v123_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v128_data, v201_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v129_data, v206_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v130_data, v207_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v131_data, v208_acc, 3, 2, 0);
          ir4[8] = (v209_acc[0]);
          ir4[9] = (v209_acc[1]);
          ir4[10] = (v209_acc[2]);
          ir4[11] = (v209_acc[3]);
          float v226_acc{};
          float v227_data = r3[12];
          float v228_bc = tensorforge::broadcast<32, 16, 0>(v227_data);
          tensorforge::fmacdpp16<0>(v226_acc, v228_bc, v112_data);
          tensorforge::fmacdpp16<1>(v226_acc, v228_bc, v113_data);
          tensorforge::fmacdpp16<2>(v226_acc, v228_bc, v114_data);
          tensorforge::fmacdpp16<3>(v226_acc, v228_bc, v115_data);
          tensorforge::fmacdpp16<4>(v226_acc, v228_bc, v120_data);
          tensorforge::fmacdpp16<5>(v226_acc, v228_bc, v121_data);
          tensorforge::fmacdpp16<6>(v226_acc, v228_bc, v122_data);
          tensorforge::fmacdpp16<7>(v226_acc, v228_bc, v123_data);
          tensorforge::fmacdpp16<8>(v226_acc, v228_bc, v128_data);
          tensorforge::fmacdpp16<9>(v226_acc, v228_bc, v129_data);
          tensorforge::fmacdpp16<10>(v226_acc, v228_bc, v130_data);
          tensorforge::fmacdpp16<11>(v226_acc, v228_bc, v131_data);
          ir4[12] = v226_acc;
          #pragma unroll
          for (int32_t v232_n0 = 0; v232_n0 < 1; ++v232_n0) {
            #pragma unroll
            for (int32_t v233_n1 = 0; v233_n1 < 13; ++v233_n1) {
              int32_t v234_a = v232_n0 + v233_n1;
              float v235_data = ir4[v234_a];
              float v237_data = r1[v234_a];
              r4[v234_a] = (v237_data + v235_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v244_data = r4[4];
          float v245_data = r5[0];
          r5[0] = (v245_data + v244_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v250_i0 = 0; v250_i0 < 1; ++v250_i0) {
            int32_t v258_lead = v16_lead + (v250_i0 * 32);
            #pragma unroll
            for (int32_t v251_i1 = 0; v251_i1 < 1; ++v251_i1) {
              float v253_data = r5[(v250_i0 + v251_i1)];
              glb_m0[(v258_lead + ((v251_i1 + 4) * 32))] = v253_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v266_i0 = 0; v266_i0 < 1; ++v266_i0) {
            int32_t v272_lead = v16_lead + (v266_i0 * 32);
            #pragma unroll
            for (int32_t v267_i1 = 0; v267_i1 < 13; ++v267_i1) {
              float v275_data = glb_m0[(v272_lead + (v267_i1 * 32))];
              r6[(v266_i0 + v267_i1)] = v275_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          if (v16_lead < 13) {
            #pragma unroll
            for (int32_t v282_i1 = 0; v282_i1 < 13; ++v282_i1) {
              float v290_data = __builtin_nontemporal_load(&glb_m4[(v16_lead + (v282_i1 * 13))]);
              r7[v282_i1] = v290_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v293_data = r7[0];
          float v294_data = r7[1];
          float v295_data = r7[2];
          float v296_data = r7[3];
          float v297_tp{};
          float v298_tp{};
          float v299_tp{};
          float v300_tp{};
          tensorforge::transpose4x4b32(v297_tp, v298_tp, v299_tp, v300_tp, v293_data, v294_data, v295_data, v296_data);
          tensorforge::VectorT<float, 4> v301_acc{};
          float v302_data = r6[0];
          float v303_data = r6[1];
          float v304_data = r6[2];
          float v305_data = r6[3];
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v302_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v303_data, v306_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v299_tp, v304_data, v307_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v305_data, v308_acc, 3, 0, 0);
          float v310_data = r6[4];
          float v311_data = r6[5];
          float v312_data = r6[6];
          float v313_data = r6[7];
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v310_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v311_data, v314_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v299_tp, v312_data, v315_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v313_data, v316_acc, 3, 1, 0);
          float v318_data = r6[8];
          float v319_data = r6[9];
          float v320_data = r6[10];
          float v321_data = r6[11];
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v318_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v298_tp, v319_data, v322_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v299_tp, v320_data, v323_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v300_tp, v321_data, v324_acc, 3, 2, 0);
          float v326_data = r6[12];
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v297_tp, v326_data, v325_acc, 3, 3, 0);
          r8[0] = (v330_acc[0]);
          r8[1] = (v330_acc[1]);
          r8[2] = (v330_acc[2]);
          r8[3] = (v330_acc[3]);
          float v335_data = r7[4];
          float v336_data = r7[5];
          float v337_data = r7[6];
          float v338_data = r7[7];
          float v339_tp{};
          float v340_tp{};
          float v341_tp{};
          float v342_tp{};
          tensorforge::transpose4x4b32(v339_tp, v340_tp, v341_tp, v342_tp, v335_data, v336_data, v337_data, v338_data);
          tensorforge::VectorT<float, 4> v343_acc{};
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v302_data, v343_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v303_data, v348_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v304_data, v349_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v305_data, v350_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v310_data, v351_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v311_data, v356_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v312_data, v357_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v313_data, v358_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v318_data, v359_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v319_data, v364_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v320_data, v365_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v321_data, v366_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v326_data, v367_acc, 3, 3, 0);
          r8[4] = (v372_acc[0]);
          r8[5] = (v372_acc[1]);
          r8[6] = (v372_acc[2]);
          r8[7] = (v372_acc[3]);
          float v377_data = r7[8];
          float v378_data = r7[9];
          float v379_data = r7[10];
          float v380_data = r7[11];
          float v381_tp{};
          float v382_tp{};
          float v383_tp{};
          float v384_tp{};
          tensorforge::transpose4x4b32(v381_tp, v382_tp, v383_tp, v384_tp, v377_data, v378_data, v379_data, v380_data);
          tensorforge::VectorT<float, 4> v385_acc{};
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v302_data, v385_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v303_data, v390_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v304_data, v391_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v305_data, v392_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v310_data, v393_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v311_data, v398_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v312_data, v399_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v313_data, v400_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v318_data, v401_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v319_data, v406_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v320_data, v407_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v321_data, v408_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v326_data, v409_acc, 3, 3, 0);
          r8[8] = (v414_acc[0]);
          r8[9] = (v414_acc[1]);
          r8[10] = (v414_acc[2]);
          r8[11] = (v414_acc[3]);
          float v432_acc{};
          float v433_data = r7[12];
          float v434_bc = tensorforge::broadcast<32, 16, 0>(v433_data);
          tensorforge::fmacdpp16<0>(v432_acc, v434_bc, v302_data);
          tensorforge::fmacdpp16<1>(v432_acc, v434_bc, v303_data);
          tensorforge::fmacdpp16<2>(v432_acc, v434_bc, v304_data);
          tensorforge::fmacdpp16<3>(v432_acc, v434_bc, v305_data);
          tensorforge::fmacdpp16<4>(v432_acc, v434_bc, v310_data);
          tensorforge::fmacdpp16<5>(v432_acc, v434_bc, v311_data);
          tensorforge::fmacdpp16<6>(v432_acc, v434_bc, v312_data);
          tensorforge::fmacdpp16<7>(v432_acc, v434_bc, v313_data);
          tensorforge::fmacdpp16<8>(v432_acc, v434_bc, v318_data);
          tensorforge::fmacdpp16<9>(v432_acc, v434_bc, v319_data);
          tensorforge::fmacdpp16<10>(v432_acc, v434_bc, v320_data);
          tensorforge::fmacdpp16<11>(v432_acc, v434_bc, v321_data);
          tensorforge::fmacdpp16<12>(v432_acc, v434_bc, v326_data);
          r8[12] = v432_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v438_i0 = 0; v438_i0 < 1; ++v438_i0) {
            int32_t v446_lead = v16_lead + (v438_i0 * 32);
            #pragma unroll
            for (int32_t v439_i1 = 0; v439_i1 < 13; ++v439_i1) {
              float v441_data = r8[(v438_i0 + v439_i1)];
              glb_m3[(v446_lead + (v439_i1 * 32))] = v441_data;
            }
          }
        }
      }
    }
  }
}

