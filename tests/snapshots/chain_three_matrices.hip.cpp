// === base name ===
kernel_3056ac16c90eab0c

// === header ===
void launcher_kernel_3056ac16c90eab0c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3056ac16c90eab0c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3056ac16c90eab0c, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_3056ac16c90eab0c, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3056ac16c90eab0c), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3056ac16c90eab0c, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3056ac16c90eab0c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 56×9(56×9) {0..56}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 56×9(56×9) {0..56}×{0..9} strided
    // m3 56×56(56×56) {0..56}×{0..56} strided
    // t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[0, 1] = m0 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, -1]×m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]
    // m2 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, 1] = m3 56×56(56×56) {0..56}×{0..56} strided({0..56}×{0..56})[0, -1]×t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[-1, 1]
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 504 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 81 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 504 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 3136 + 0 + m3_extraOffset];
          float r0[18]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            int32_t v22_lead = v15_lead + (v16_i0 * 32);
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 9; ++v17_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v22_lead + (v17_i1 * 56))]);
              r0[(v16_i0 + (v17_i1 * 2))] = v25_data;
            }
          }
          if (v15_lead < 24) {
            int32_t v34_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m0[(v34_lead + (v29_i1 * 56))]);
              r0[(1 + (v29_i1 * 2))] = v37_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m1);
          if (v15_lead < 9) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 9; ++v45_i1) {
              float v53_data = __builtin_nontemporal_load(&glb_m1[(v15_lead + (v45_i1 * 9))]);
              r1[v45_i1] = v53_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[112]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v59_i0 = 0; v59_i0 < 1; ++v59_i0) {
            int32_t v65_lead = v15_lead + (v59_i0 * 32);
            #pragma unroll
            for (int32_t v60_i1 = 0; v60_i1 < 56; ++v60_i1) {
              float v68_data = __builtin_nontemporal_load(&glb_m3[(v65_lead + (v60_i1 * 56))]);
              r3[(v59_i0 + (v60_i1 * 2))] = v68_data;
            }
          }
          if (v15_lead < 24) {
            int32_t v77_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v72_i1 = 0; v72_i1 < 56; ++v72_i1) {
              float v80_data = __builtin_nontemporal_load(&glb_m3[(v77_lead + (v72_i1 * 56))]);
              r3[(1 + (v72_i1 * 2))] = v80_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[18]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 9)] [(0, 9)]
          float v84_data = r1[0];
          float v85_data = r1[1];
          float v86_data = r1[2];
          float v87_data = r1[3];
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, v86_data, v87_data);
          tensorforge::VectorT<float, 4> v92_acc{};
          float v93_data = r0[0];
          float v94_data = r0[2];
          float v95_data = r0[4];
          float v96_data = r0[6];
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v93_data, v92_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v94_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v95_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v96_data, v99_acc, 3, 0, 0);
          float v101_data = r0[8];
          float v102_data = r0[10];
          float v103_data = r0[12];
          float v104_data = r0[14];
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v101_data, v100_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v102_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v103_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v104_data, v107_acc, 3, 1, 0);
          float v109_data = r0[16];
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v109_data, v108_acc, 3, 2, 0);
          r2[0] = (v113_acc[0]);
          r2[2] = (v113_acc[1]);
          r2[4] = (v113_acc[2]);
          r2[6] = (v113_acc[3]);
          tensorforge::VectorT<float, 4> v118_acc{};
          float v119_data = r0[1];
          float v120_data = r0[3];
          float v121_data = r0[5];
          float v122_data = r0[7];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v119_data, v118_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v120_data, v123_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v121_data, v124_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v122_data, v125_acc, 3, 0, 0);
          float v127_data = r0[9];
          float v128_data = r0[11];
          float v129_data = r0[13];
          float v130_data = r0[15];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v127_data, v126_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v128_data, v131_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v129_data, v132_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v130_data, v133_acc, 3, 1, 0);
          float v135_data = r0[17];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v135_data, v134_acc, 3, 2, 0);
          r2[1] = (v139_acc[0]);
          r2[3] = (v139_acc[1]);
          r2[5] = (v139_acc[2]);
          r2[7] = (v139_acc[3]);
          float v144_data = r1[4];
          float v145_data = r1[5];
          float v146_data = r1[6];
          float v147_data = r1[7];
          float v148_tp{};
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          tensorforge::transpose4x4b32(v148_tp, v149_tp, v150_tp, v151_tp, v144_data, v145_data, v146_data, v147_data);
          tensorforge::VectorT<float, 4> v152_acc{};
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v93_data, v152_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v94_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v95_data, v158_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v96_data, v159_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v101_data, v160_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v102_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v103_data, v166_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v104_data, v167_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v109_data, v168_acc, 3, 2, 0);
          r2[8] = (v173_acc[0]);
          r2[10] = (v173_acc[1]);
          r2[12] = (v173_acc[2]);
          r2[14] = (v173_acc[3]);
          tensorforge::VectorT<float, 4> v178_acc{};
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v119_data, v178_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v120_data, v183_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v121_data, v184_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v122_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v127_data, v186_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v128_data, v191_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v129_data, v192_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v130_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v135_data, v194_acc, 3, 2, 0);
          r2[9] = (v199_acc[0]);
          r2[11] = (v199_acc[1]);
          r2[13] = (v199_acc[2]);
          r2[15] = (v199_acc[3]);
          float v222_acc{};
          float v223_acc{};
          float v224_data = r1[8];
          float v225_bc = tensorforge::broadcast<32, 16, 0>(v224_data);
          tensorforge::fmacdpp16<0>(v222_acc, v225_bc, v93_data);
          tensorforge::fmacdpp16<0>(v223_acc, v225_bc, v119_data);
          tensorforge::fmacdpp16<1>(v222_acc, v225_bc, v94_data);
          tensorforge::fmacdpp16<1>(v223_acc, v225_bc, v120_data);
          tensorforge::fmacdpp16<2>(v222_acc, v225_bc, v95_data);
          tensorforge::fmacdpp16<2>(v223_acc, v225_bc, v121_data);
          tensorforge::fmacdpp16<3>(v222_acc, v225_bc, v96_data);
          tensorforge::fmacdpp16<3>(v223_acc, v225_bc, v122_data);
          tensorforge::fmacdpp16<4>(v222_acc, v225_bc, v101_data);
          tensorforge::fmacdpp16<4>(v223_acc, v225_bc, v127_data);
          tensorforge::fmacdpp16<5>(v222_acc, v225_bc, v102_data);
          tensorforge::fmacdpp16<5>(v223_acc, v225_bc, v128_data);
          tensorforge::fmacdpp16<6>(v222_acc, v225_bc, v103_data);
          tensorforge::fmacdpp16<6>(v223_acc, v225_bc, v129_data);
          tensorforge::fmacdpp16<7>(v222_acc, v225_bc, v104_data);
          tensorforge::fmacdpp16<7>(v223_acc, v225_bc, v130_data);
          tensorforge::fmacdpp16<8>(v222_acc, v225_bc, v109_data);
          tensorforge::fmacdpp16<8>(v223_acc, v225_bc, v135_data);
          r2[16] = v222_acc;
          r2[17] = v223_acc;
          // wait(r3 = load{g>r}(glb_m3););
          float r4[18]{};
          // r4 = +(r3 * r2) + None
          // [(0, 56), (0, 9)] [(0, 56)]
          float v227_data = r2[0];
          float v228_data = r2[2];
          float v229_data = r2[4];
          float v230_data = r2[6];
          float v231_tp{};
          float v232_tp{};
          float v233_tp{};
          float v234_tp{};
          tensorforge::transpose4x4b32(v231_tp, v232_tp, v233_tp, v234_tp, v227_data, v228_data, v229_data, v230_data);
          float v235_data = r2[1];
          float v236_data = r2[3];
          float v237_data = r2[5];
          float v238_data = r2[7];
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          float v242_tp{};
          tensorforge::transpose4x4b32(v239_tp, v240_tp, v241_tp, v242_tp, v235_data, v236_data, v237_data, v238_data);
          tensorforge::VectorT<float, 4> v243_acc{};
          float v244_data = r3[0];
          float v245_data = r3[2];
          float v246_data = r3[4];
          float v247_data = r3[6];
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v244_data, v243_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v245_data, v248_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v246_data, v249_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v247_data, v250_acc, 3, 0, 0);
          float v252_data = r3[8];
          float v253_data = r3[10];
          float v254_data = r3[12];
          float v255_data = r3[14];
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v252_data, v251_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v253_data, v256_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v254_data, v257_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v255_data, v258_acc, 3, 1, 0);
          float v260_data = r3[16];
          float v261_data = r3[18];
          float v262_data = r3[20];
          float v263_data = r3[22];
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v260_data, v259_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v261_data, v264_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v262_data, v265_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v263_data, v266_acc, 3, 2, 0);
          float v268_data = r3[24];
          float v269_data = r3[26];
          float v270_data = r3[28];
          float v271_data = r3[30];
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v268_data, v267_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v269_data, v272_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v270_data, v273_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v271_data, v274_acc, 3, 3, 0);
          float v276_data = r3[32];
          float v277_data = r3[34];
          float v278_data = r3[36];
          float v279_data = r3[38];
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v276_data, v275_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v277_data, v280_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v278_data, v281_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v279_data, v282_acc, 3, 4, 0);
          float v284_data = r3[40];
          float v285_data = r3[42];
          float v286_data = r3[44];
          float v287_data = r3[46];
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v284_data, v283_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v285_data, v288_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v286_data, v289_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v287_data, v290_acc, 3, 5, 0);
          float v292_data = r3[48];
          float v293_data = r3[50];
          float v294_data = r3[52];
          float v295_data = r3[54];
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v292_data, v291_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v293_data, v296_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v294_data, v297_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v295_data, v298_acc, 3, 6, 0);
          float v300_data = r3[56];
          float v301_data = r3[58];
          float v302_data = r3[60];
          float v303_data = r3[62];
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v300_data, v299_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v301_data, v304_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v302_data, v305_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v303_data, v306_acc, 3, 7, 0);
          float v308_data = r3[64];
          float v309_data = r3[66];
          float v310_data = r3[68];
          float v311_data = r3[70];
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v308_data, v307_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v309_data, v312_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v310_data, v313_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v311_data, v314_acc, 3, 0, 0);
          float v316_data = r3[72];
          float v317_data = r3[74];
          float v318_data = r3[76];
          float v319_data = r3[78];
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v316_data, v315_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v317_data, v320_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v318_data, v321_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v319_data, v322_acc, 3, 1, 0);
          float v324_data = r3[80];
          float v325_data = r3[82];
          float v326_data = r3[84];
          float v327_data = r3[86];
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v324_data, v323_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v325_data, v328_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v326_data, v329_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v327_data, v330_acc, 3, 2, 0);
          float v332_data = r3[88];
          float v333_data = r3[90];
          float v334_data = r3[92];
          float v335_data = r3[94];
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v332_data, v331_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v333_data, v336_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v334_data, v337_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v335_data, v338_acc, 3, 3, 0);
          float v340_data = r3[96];
          float v341_data = r3[98];
          float v342_data = r3[100];
          float v343_data = r3[102];
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v340_data, v339_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v341_data, v344_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v342_data, v345_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v343_data, v346_acc, 3, 4, 0);
          float v348_data = r3[104];
          float v349_data = r3[106];
          float v350_data = r3[108];
          float v351_data = r3[110];
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v348_data, v347_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v349_data, v352_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v350_data, v353_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v351_data, v354_acc, 3, 5, 0);
          r4[0] = (v355_acc[0]);
          r4[2] = (v355_acc[1]);
          r4[4] = (v355_acc[2]);
          r4[6] = (v355_acc[3]);
          tensorforge::VectorT<float, 4> v360_acc{};
          float v361_data = r3[1];
          float v362_data = r3[3];
          float v363_data = r3[5];
          float v364_data = r3[7];
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v361_data, v360_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v362_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v363_data, v366_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v364_data, v367_acc, 3, 0, 0);
          float v369_data = r3[9];
          float v370_data = r3[11];
          float v371_data = r3[13];
          float v372_data = r3[15];
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v369_data, v368_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v370_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v371_data, v374_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v372_data, v375_acc, 3, 1, 0);
          float v377_data = r3[17];
          float v378_data = r3[19];
          float v379_data = r3[21];
          float v380_data = r3[23];
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v377_data, v376_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v378_data, v381_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v379_data, v382_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v380_data, v383_acc, 3, 2, 0);
          float v385_data = r3[25];
          float v386_data = r3[27];
          float v387_data = r3[29];
          float v388_data = r3[31];
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v385_data, v384_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v386_data, v389_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v387_data, v390_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v388_data, v391_acc, 3, 3, 0);
          float v393_data = r3[33];
          float v394_data = r3[35];
          float v395_data = r3[37];
          float v396_data = r3[39];
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v393_data, v392_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v394_data, v397_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v395_data, v398_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v396_data, v399_acc, 3, 4, 0);
          float v401_data = r3[41];
          float v402_data = r3[43];
          float v403_data = r3[45];
          float v404_data = r3[47];
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v401_data, v400_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v402_data, v405_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v403_data, v406_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v404_data, v407_acc, 3, 5, 0);
          float v409_data = r3[49];
          float v410_data = r3[51];
          float v411_data = r3[53];
          float v412_data = r3[55];
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v409_data, v408_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v410_data, v413_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v411_data, v414_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v412_data, v415_acc, 3, 6, 0);
          float v417_data = r3[57];
          float v418_data = r3[59];
          float v419_data = r3[61];
          float v420_data = r3[63];
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v417_data, v416_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v418_data, v421_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v419_data, v422_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v420_data, v423_acc, 3, 7, 0);
          float v425_data = r3[65];
          float v426_data = r3[67];
          float v427_data = r3[69];
          float v428_data = r3[71];
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v425_data, v424_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v426_data, v429_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v427_data, v430_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v428_data, v431_acc, 3, 0, 0);
          float v433_data = r3[73];
          float v434_data = r3[75];
          float v435_data = r3[77];
          float v436_data = r3[79];
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v433_data, v432_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v434_data, v437_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v435_data, v438_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v436_data, v439_acc, 3, 1, 0);
          float v441_data = r3[81];
          float v442_data = r3[83];
          float v443_data = r3[85];
          float v444_data = r3[87];
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v441_data, v440_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v442_data, v445_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v443_data, v446_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v444_data, v447_acc, 3, 2, 0);
          float v449_data = r3[89];
          float v450_data = r3[91];
          float v451_data = r3[93];
          float v452_data = r3[95];
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v449_data, v448_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v450_data, v453_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v451_data, v454_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v452_data, v455_acc, 3, 3, 0);
          float v457_data = r3[97];
          float v458_data = r3[99];
          float v459_data = r3[101];
          float v460_data = r3[103];
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v457_data, v456_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v458_data, v461_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v459_data, v462_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v460_data, v463_acc, 3, 4, 0);
          float v465_data = r3[105];
          float v466_data = r3[107];
          float v467_data = r3[109];
          float v468_data = r3[111];
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v465_data, v464_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v466_data, v469_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v467_data, v470_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v468_data, v471_acc, 3, 5, 0);
          r4[1] = (v472_acc[0]);
          r4[3] = (v472_acc[1]);
          r4[5] = (v472_acc[2]);
          r4[7] = (v472_acc[3]);
          float v477_data = r2[8];
          float v478_data = r2[10];
          float v479_data = r2[12];
          float v480_data = r2[14];
          float v481_tp{};
          float v482_tp{};
          float v483_tp{};
          float v484_tp{};
          tensorforge::transpose4x4b32(v481_tp, v482_tp, v483_tp, v484_tp, v477_data, v478_data, v479_data, v480_data);
          float v485_data = r2[9];
          float v486_data = r2[11];
          float v487_data = r2[13];
          float v488_data = r2[15];
          float v489_tp{};
          float v490_tp{};
          float v491_tp{};
          float v492_tp{};
          tensorforge::transpose4x4b32(v489_tp, v490_tp, v491_tp, v492_tp, v485_data, v486_data, v487_data, v488_data);
          tensorforge::VectorT<float, 4> v493_acc{};
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v244_data, v493_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v245_data, v498_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v246_data, v499_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v247_data, v500_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v252_data, v501_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v253_data, v506_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v254_data, v507_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v255_data, v508_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v260_data, v509_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v261_data, v514_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v262_data, v515_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v263_data, v516_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v268_data, v517_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v269_data, v522_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v270_data, v523_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v271_data, v524_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v276_data, v525_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v277_data, v530_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v278_data, v531_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v279_data, v532_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v284_data, v533_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v285_data, v538_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v286_data, v539_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v287_data, v540_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v292_data, v541_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v293_data, v546_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v294_data, v547_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v295_data, v548_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v300_data, v549_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v301_data, v554_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v302_data, v555_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v303_data, v556_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v308_data, v557_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v309_data, v562_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v310_data, v563_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v311_data, v564_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v316_data, v565_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v317_data, v570_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v318_data, v571_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v319_data, v572_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v324_data, v573_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v325_data, v578_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v326_data, v579_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v327_data, v580_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v332_data, v581_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v333_data, v586_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v334_data, v587_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v589_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v335_data, v588_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v340_data, v589_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v341_data, v594_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v342_data, v595_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v343_data, v596_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v348_data, v597_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v349_data, v602_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v350_data, v603_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v351_data, v604_acc, 3, 5, 0);
          r4[8] = (v605_acc[0]);
          r4[10] = (v605_acc[1]);
          r4[12] = (v605_acc[2]);
          r4[14] = (v605_acc[3]);
          tensorforge::VectorT<float, 4> v610_acc{};
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v361_data, v610_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v362_data, v615_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v363_data, v616_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v364_data, v617_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v369_data, v618_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v370_data, v623_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v371_data, v624_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v372_data, v625_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v631_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v377_data, v626_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v378_data, v631_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v379_data, v632_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v380_data, v633_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v385_data, v634_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v386_data, v639_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v641_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v387_data, v640_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v388_data, v641_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v393_data, v642_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v394_data, v647_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v395_data, v648_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v396_data, v649_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v401_data, v650_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v402_data, v655_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v403_data, v656_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v404_data, v657_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v409_data, v658_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v410_data, v663_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v411_data, v664_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v412_data, v665_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v671_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v417_data, v666_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v418_data, v671_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v673_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v419_data, v672_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v420_data, v673_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v425_data, v674_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v426_data, v679_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v427_data, v680_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v428_data, v681_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v433_data, v682_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v434_data, v687_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v435_data, v688_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v436_data, v689_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v441_data, v690_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v442_data, v695_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v443_data, v696_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v444_data, v697_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v449_data, v698_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v450_data, v703_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v451_data, v704_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v452_data, v705_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v457_data, v706_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v458_data, v711_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v713_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v459_data, v712_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v460_data, v713_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v465_data, v714_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v720_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v466_data, v719_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v721_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v467_data, v720_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v468_data, v721_acc, 3, 5, 0);
          r4[9] = (v722_acc[0]);
          r4[11] = (v722_acc[1]);
          r4[13] = (v722_acc[2]);
          r4[15] = (v722_acc[3]);
          float v839_acc{};
          float v840_acc{};
          float v841_data = r2[16];
          float v842_data = r2[17];
          float v843_bc = tensorforge::broadcast<32, 16, 0>(v841_data);
          tensorforge::fmacdpp16<0>(v839_acc, v843_bc, v244_data);
          tensorforge::fmacdpp16<0>(v840_acc, v843_bc, v361_data);
          tensorforge::fmacdpp16<1>(v839_acc, v843_bc, v245_data);
          tensorforge::fmacdpp16<1>(v840_acc, v843_bc, v362_data);
          tensorforge::fmacdpp16<2>(v839_acc, v843_bc, v246_data);
          tensorforge::fmacdpp16<2>(v840_acc, v843_bc, v363_data);
          tensorforge::fmacdpp16<3>(v839_acc, v843_bc, v247_data);
          tensorforge::fmacdpp16<3>(v840_acc, v843_bc, v364_data);
          tensorforge::fmacdpp16<4>(v839_acc, v843_bc, v252_data);
          tensorforge::fmacdpp16<4>(v840_acc, v843_bc, v369_data);
          tensorforge::fmacdpp16<5>(v839_acc, v843_bc, v253_data);
          tensorforge::fmacdpp16<5>(v840_acc, v843_bc, v370_data);
          tensorforge::fmacdpp16<6>(v839_acc, v843_bc, v254_data);
          tensorforge::fmacdpp16<6>(v840_acc, v843_bc, v371_data);
          tensorforge::fmacdpp16<7>(v839_acc, v843_bc, v255_data);
          tensorforge::fmacdpp16<7>(v840_acc, v843_bc, v372_data);
          tensorforge::fmacdpp16<8>(v839_acc, v843_bc, v260_data);
          tensorforge::fmacdpp16<8>(v840_acc, v843_bc, v377_data);
          tensorforge::fmacdpp16<9>(v839_acc, v843_bc, v261_data);
          tensorforge::fmacdpp16<9>(v840_acc, v843_bc, v378_data);
          tensorforge::fmacdpp16<10>(v839_acc, v843_bc, v262_data);
          tensorforge::fmacdpp16<10>(v840_acc, v843_bc, v379_data);
          tensorforge::fmacdpp16<11>(v839_acc, v843_bc, v263_data);
          tensorforge::fmacdpp16<11>(v840_acc, v843_bc, v380_data);
          tensorforge::fmacdpp16<12>(v839_acc, v843_bc, v268_data);
          tensorforge::fmacdpp16<12>(v840_acc, v843_bc, v385_data);
          tensorforge::fmacdpp16<13>(v839_acc, v843_bc, v269_data);
          tensorforge::fmacdpp16<13>(v840_acc, v843_bc, v386_data);
          tensorforge::fmacdpp16<14>(v839_acc, v843_bc, v270_data);
          tensorforge::fmacdpp16<14>(v840_acc, v843_bc, v387_data);
          tensorforge::fmacdpp16<15>(v839_acc, v843_bc, v271_data);
          tensorforge::fmacdpp16<15>(v840_acc, v843_bc, v388_data);
          float v844_bc = tensorforge::broadcast<32, 16, 1>(v841_data);
          tensorforge::fmacdpp16<0>(v839_acc, v844_bc, v276_data);
          tensorforge::fmacdpp16<0>(v840_acc, v844_bc, v393_data);
          tensorforge::fmacdpp16<1>(v839_acc, v844_bc, v277_data);
          tensorforge::fmacdpp16<1>(v840_acc, v844_bc, v394_data);
          tensorforge::fmacdpp16<2>(v839_acc, v844_bc, v278_data);
          tensorforge::fmacdpp16<2>(v840_acc, v844_bc, v395_data);
          tensorforge::fmacdpp16<3>(v839_acc, v844_bc, v279_data);
          tensorforge::fmacdpp16<3>(v840_acc, v844_bc, v396_data);
          tensorforge::fmacdpp16<4>(v839_acc, v844_bc, v284_data);
          tensorforge::fmacdpp16<4>(v840_acc, v844_bc, v401_data);
          tensorforge::fmacdpp16<5>(v839_acc, v844_bc, v285_data);
          tensorforge::fmacdpp16<5>(v840_acc, v844_bc, v402_data);
          tensorforge::fmacdpp16<6>(v839_acc, v844_bc, v286_data);
          tensorforge::fmacdpp16<6>(v840_acc, v844_bc, v403_data);
          tensorforge::fmacdpp16<7>(v839_acc, v844_bc, v287_data);
          tensorforge::fmacdpp16<7>(v840_acc, v844_bc, v404_data);
          tensorforge::fmacdpp16<8>(v839_acc, v844_bc, v292_data);
          tensorforge::fmacdpp16<8>(v840_acc, v844_bc, v409_data);
          tensorforge::fmacdpp16<9>(v839_acc, v844_bc, v293_data);
          tensorforge::fmacdpp16<9>(v840_acc, v844_bc, v410_data);
          tensorforge::fmacdpp16<10>(v839_acc, v844_bc, v294_data);
          tensorforge::fmacdpp16<10>(v840_acc, v844_bc, v411_data);
          tensorforge::fmacdpp16<11>(v839_acc, v844_bc, v295_data);
          tensorforge::fmacdpp16<11>(v840_acc, v844_bc, v412_data);
          tensorforge::fmacdpp16<12>(v839_acc, v844_bc, v300_data);
          tensorforge::fmacdpp16<12>(v840_acc, v844_bc, v417_data);
          tensorforge::fmacdpp16<13>(v839_acc, v844_bc, v301_data);
          tensorforge::fmacdpp16<13>(v840_acc, v844_bc, v418_data);
          tensorforge::fmacdpp16<14>(v839_acc, v844_bc, v302_data);
          tensorforge::fmacdpp16<14>(v840_acc, v844_bc, v419_data);
          tensorforge::fmacdpp16<15>(v839_acc, v844_bc, v303_data);
          tensorforge::fmacdpp16<15>(v840_acc, v844_bc, v420_data);
          float v845_bc = tensorforge::broadcast<32, 16, 0>(v842_data);
          tensorforge::fmacdpp16<0>(v839_acc, v845_bc, v308_data);
          tensorforge::fmacdpp16<0>(v840_acc, v845_bc, v425_data);
          tensorforge::fmacdpp16<1>(v839_acc, v845_bc, v309_data);
          tensorforge::fmacdpp16<1>(v840_acc, v845_bc, v426_data);
          tensorforge::fmacdpp16<2>(v839_acc, v845_bc, v310_data);
          tensorforge::fmacdpp16<2>(v840_acc, v845_bc, v427_data);
          tensorforge::fmacdpp16<3>(v839_acc, v845_bc, v311_data);
          tensorforge::fmacdpp16<3>(v840_acc, v845_bc, v428_data);
          tensorforge::fmacdpp16<4>(v839_acc, v845_bc, v316_data);
          tensorforge::fmacdpp16<4>(v840_acc, v845_bc, v433_data);
          tensorforge::fmacdpp16<5>(v839_acc, v845_bc, v317_data);
          tensorforge::fmacdpp16<5>(v840_acc, v845_bc, v434_data);
          tensorforge::fmacdpp16<6>(v839_acc, v845_bc, v318_data);
          tensorforge::fmacdpp16<6>(v840_acc, v845_bc, v435_data);
          tensorforge::fmacdpp16<7>(v839_acc, v845_bc, v319_data);
          tensorforge::fmacdpp16<7>(v840_acc, v845_bc, v436_data);
          tensorforge::fmacdpp16<8>(v839_acc, v845_bc, v324_data);
          tensorforge::fmacdpp16<8>(v840_acc, v845_bc, v441_data);
          tensorforge::fmacdpp16<9>(v839_acc, v845_bc, v325_data);
          tensorforge::fmacdpp16<9>(v840_acc, v845_bc, v442_data);
          tensorforge::fmacdpp16<10>(v839_acc, v845_bc, v326_data);
          tensorforge::fmacdpp16<10>(v840_acc, v845_bc, v443_data);
          tensorforge::fmacdpp16<11>(v839_acc, v845_bc, v327_data);
          tensorforge::fmacdpp16<11>(v840_acc, v845_bc, v444_data);
          tensorforge::fmacdpp16<12>(v839_acc, v845_bc, v332_data);
          tensorforge::fmacdpp16<12>(v840_acc, v845_bc, v449_data);
          tensorforge::fmacdpp16<13>(v839_acc, v845_bc, v333_data);
          tensorforge::fmacdpp16<13>(v840_acc, v845_bc, v450_data);
          tensorforge::fmacdpp16<14>(v839_acc, v845_bc, v334_data);
          tensorforge::fmacdpp16<14>(v840_acc, v845_bc, v451_data);
          tensorforge::fmacdpp16<15>(v839_acc, v845_bc, v335_data);
          tensorforge::fmacdpp16<15>(v840_acc, v845_bc, v452_data);
          float v846_bc = tensorforge::broadcast<32, 16, 1>(v842_data);
          tensorforge::fmacdpp16<0>(v839_acc, v846_bc, v340_data);
          tensorforge::fmacdpp16<0>(v840_acc, v846_bc, v457_data);
          tensorforge::fmacdpp16<1>(v839_acc, v846_bc, v341_data);
          tensorforge::fmacdpp16<1>(v840_acc, v846_bc, v458_data);
          tensorforge::fmacdpp16<2>(v839_acc, v846_bc, v342_data);
          tensorforge::fmacdpp16<2>(v840_acc, v846_bc, v459_data);
          tensorforge::fmacdpp16<3>(v839_acc, v846_bc, v343_data);
          tensorforge::fmacdpp16<3>(v840_acc, v846_bc, v460_data);
          tensorforge::fmacdpp16<4>(v839_acc, v846_bc, v348_data);
          tensorforge::fmacdpp16<4>(v840_acc, v846_bc, v465_data);
          tensorforge::fmacdpp16<5>(v839_acc, v846_bc, v349_data);
          tensorforge::fmacdpp16<5>(v840_acc, v846_bc, v466_data);
          tensorforge::fmacdpp16<6>(v839_acc, v846_bc, v350_data);
          tensorforge::fmacdpp16<6>(v840_acc, v846_bc, v467_data);
          tensorforge::fmacdpp16<7>(v839_acc, v846_bc, v351_data);
          tensorforge::fmacdpp16<7>(v840_acc, v846_bc, v468_data);
          r4[16] = v839_acc;
          r4[17] = v840_acc;
          // glb_m2 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v850_i0 = 0; v850_i0 < 1; ++v850_i0) {
            int32_t v859_lead = v15_lead + (v850_i0 * 32);
            #pragma unroll
            for (int32_t v851_i1 = 0; v851_i1 < 9; ++v851_i1) {
              float v854_data = r4[(v850_i0 + (v851_i1 * 2))];
              glb_m2[(v859_lead + (v851_i1 * 56))] = v854_data;
            }
          }
          if (v15_lead < 24) {
            int32_t v871_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v863_i1 = 0; v863_i1 < 9; ++v863_i1) {
              float v866_data = r4[(1 + (v863_i1 * 2))];
              glb_m2[(v871_lead + (v863_i1 * 56))] = v866_data;
            }
          }
        }
      }
    }
  }
}

