// === base name ===
kernel_e5f66540cde6a5d4

// === header ===
void launcher_kernel_e5f66540cde6a5d4(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e5f66540cde6a5d4(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e5f66540cde6a5d4, block.x * block.y * block.z, 2560 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (2560 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e5f66540cde6a5d4, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (2560 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e5f66540cde6a5d4), hipFuncAttributeMaxDynamicSharedMemorySize, 2560 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_e5f66540cde6a5d4, grid, block, 2560 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e5f66540cde6a5d4(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[320 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v4_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0][0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v4_batchId0][0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v4_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
            int32_t v27_lead = v20_lead + (v21_i0 * 32);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 9; ++v22_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v22_i1 * 32))]);
              r0[(v21_i0 + v22_i1)] = v30_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v20_lead < 16) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 9; ++v37_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v37_i1 * 16))]);
              r2[v37_i1] = v45_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v51_data = r0[0];
          float v52_data = r1[0];
          r1[0] = (v52_data + v51_data);
          float v54_data = r0[1];
          float v55_data = r1[1];
          r1[1] = (v55_data + v54_data);
          float v57_data = r0[2];
          float v58_data = r1[2];
          r1[2] = (v58_data + v57_data);
          float v60_data = r0[3];
          float v61_data = r1[3];
          r1[3] = (v61_data + v60_data);
          float v63_data = r0[4];
          float v64_data = r1[4];
          r1[4] = (v64_data + v63_data);
          float v66_data = r0[5];
          float v67_data = r1[5];
          r1[5] = (v67_data + v66_data);
          float v69_data = r0[6];
          float v70_data = r1[6];
          r1[6] = (v70_data + v69_data);
          float v72_data = r0[7];
          float v73_data = r1[7];
          r1[7] = (v73_data + v72_data);
          float v75_data = r0[8];
          float v76_data = r1[8];
          r1[8] = (v76_data + v75_data);
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v81_i0 = 0; v81_i0 < 1; ++v81_i0) {
            int32_t v89_lead = v20_lead + (v81_i0 * 32);
            #pragma unroll
            for (int32_t v82_i1 = 0; v82_i1 < 9; ++v82_i1) {
              float v84_data = r1[(v81_i0 + v82_i1)];
              int32_t v91_a = v89_lead + (v82_i1 * 32);
              s0[(v91_a ^ ((v91_a >> 5) & 31))] = v84_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v20_lead < 16) {
            #pragma unroll
            for (int32_t v100_i1 = 0; v100_i1 < 9; ++v100_i1) {
              float v108_data = __builtin_nontemporal_load(&glb_m2[(v20_lead + (v100_i1 * 16))]);
              r4[v100_i1] = v108_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v20_lead < 16) {
            float v115_data = r2[0];
            float v116_data = r3[0];
            r3[0] = (v116_data + v115_data);
            float v118_data = r2[1];
            float v119_data = r3[1];
            r3[1] = (v119_data + v118_data);
            float v121_data = r2[2];
            float v122_data = r3[2];
            r3[2] = (v122_data + v121_data);
            float v124_data = r2[3];
            float v125_data = r3[3];
            r3[3] = (v125_data + v124_data);
            float v127_data = r2[4];
            float v128_data = r3[4];
            r3[4] = (v128_data + v127_data);
            float v130_data = r2[5];
            float v131_data = r3[5];
            r3[5] = (v131_data + v130_data);
            float v133_data = r2[6];
            float v134_data = r3[6];
            r3[6] = (v134_data + v133_data);
            float v136_data = r2[7];
            float v137_data = r3[7];
            r3[7] = (v137_data + v136_data);
            float v139_data = r2[8];
            float v140_data = r3[8];
            r3[8] = (v140_data + v139_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v20_lead < 16) {
            #pragma unroll
            for (int32_t v146_i1 = 0; v146_i1 < 9; ++v146_i1) {
              float v148_data = r3[v146_i1];
              int32_t v155_a = v20_lead + (v146_i1 * 32);
              s0[(v155_a ^ ((v155_a >> 5) & 31))] = v148_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v20_lead < 9) {
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 9; ++v164_i1) {
              float v172_data = __builtin_nontemporal_load(&glb_m4[(v20_lead + (v164_i1 * 9))]);
              r6[v164_i1] = v172_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v20_lead < 16) {
            float v179_data = r4[0];
            float v180_data = r5[0];
            r5[0] = (v180_data + v179_data);
            float v182_data = r4[1];
            float v183_data = r5[1];
            r5[1] = (v183_data + v182_data);
            float v185_data = r4[2];
            float v186_data = r5[2];
            r5[2] = (v186_data + v185_data);
            float v188_data = r4[3];
            float v189_data = r5[3];
            r5[3] = (v189_data + v188_data);
            float v191_data = r4[4];
            float v192_data = r5[4];
            r5[4] = (v192_data + v191_data);
            float v194_data = r4[5];
            float v195_data = r5[5];
            r5[5] = (v195_data + v194_data);
            float v197_data = r4[6];
            float v198_data = r5[6];
            r5[6] = (v198_data + v197_data);
            float v200_data = r4[7];
            float v201_data = r5[7];
            r5[7] = (v201_data + v200_data);
            float v203_data = r4[8];
            float v204_data = r5[8];
            r5[8] = (v204_data + v203_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v20_lead < 16) {
            #pragma unroll
            for (int32_t v210_i1 = 0; v210_i1 < 9; ++v210_i1) {
              float v212_data = r5[v210_i1];
              int32_t v219_a = v20_lead + (v210_i1 * 32);
              s0[(v219_a ^ ((v219_a >> 5) & 31))] = v212_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v224_data = r6[0];
          float v225_data = r6[1];
          float v226_data = r6[2];
          float v227_data = r6[3];
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          float v231_tp{};
          tensorforge::transpose4x4b32(v228_tp, v229_tp, v230_tp, v231_tp, v224_data, v225_data, v226_data, v227_data);
          tensorforge::VectorT<float, 4> v232_acc{};
          float v242_data = s0[(v20_lead ^ ((v20_lead >> 5) & 31))];
          int32_t v248_a = v20_lead + 32;
          float v252_data = s0[(v248_a ^ ((v248_a >> 5) & 31))];
          int32_t v258_a = v20_lead + 64;
          float v262_data = s0[(v258_a ^ ((v258_a >> 5) & 31))];
          int32_t v268_a = v20_lead + 96;
          float v272_data = s0[(v268_a ^ ((v268_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v242_data, v232_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v252_data, v273_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v262_data, v274_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v272_data, v275_acc, 3, 0, 0);
          int32_t v282_a = v20_lead + 128;
          float v286_data = s0[(v282_a ^ ((v282_a >> 5) & 31))];
          int32_t v292_a = v20_lead + 160;
          float v296_data = s0[(v292_a ^ ((v292_a >> 5) & 31))];
          int32_t v302_a = v20_lead + 192;
          float v306_data = s0[(v302_a ^ ((v302_a >> 5) & 31))];
          int32_t v312_a = v20_lead + 224;
          float v316_data = s0[(v312_a ^ ((v312_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v286_data, v276_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v296_data, v317_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v306_data, v318_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v316_data, v319_acc, 3, 1, 0);
          int32_t v326_a = v20_lead + 256;
          float v330_data = s0[(v326_a ^ ((v326_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v330_data, v320_acc, 3, 2, 0);
          r7[0] = (v334_acc[0]);
          r7[1] = (v334_acc[1]);
          r7[2] = (v334_acc[2]);
          r7[3] = (v334_acc[3]);
          float v339_data = r6[4];
          float v340_data = r6[5];
          float v341_data = r6[6];
          float v342_data = r6[7];
          float v343_tp{};
          float v344_tp{};
          float v345_tp{};
          float v346_tp{};
          tensorforge::transpose4x4b32(v343_tp, v344_tp, v345_tp, v346_tp, v339_data, v340_data, v341_data, v342_data);
          tensorforge::VectorT<float, 4> v347_acc{};
          float v357_data = s0[(v20_lead ^ ((v20_lead >> 5) & 31))];
          int32_t v363_a = v20_lead + 32;
          float v367_data = s0[(v363_a ^ ((v363_a >> 5) & 31))];
          int32_t v373_a = v20_lead + 64;
          float v377_data = s0[(v373_a ^ ((v373_a >> 5) & 31))];
          int32_t v383_a = v20_lead + 96;
          float v387_data = s0[(v383_a ^ ((v383_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v357_data, v347_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v367_data, v388_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v377_data, v389_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v387_data, v390_acc, 3, 0, 0);
          int32_t v397_a = v20_lead + 128;
          float v401_data = s0[(v397_a ^ ((v397_a >> 5) & 31))];
          int32_t v407_a = v20_lead + 160;
          float v411_data = s0[(v407_a ^ ((v407_a >> 5) & 31))];
          int32_t v417_a = v20_lead + 192;
          float v421_data = s0[(v417_a ^ ((v417_a >> 5) & 31))];
          int32_t v427_a = v20_lead + 224;
          float v431_data = s0[(v427_a ^ ((v427_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v401_data, v391_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v411_data, v432_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v421_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v431_data, v434_acc, 3, 1, 0);
          int32_t v441_a = v20_lead + 256;
          float v445_data = s0[(v441_a ^ ((v441_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v343_tp, v445_data, v435_acc, 3, 2, 0);
          r7[4] = (v449_acc[0]);
          r7[5] = (v449_acc[1]);
          r7[6] = (v449_acc[2]);
          r7[7] = (v449_acc[3]);
          float v463_data = s0[(v20_lead ^ ((v20_lead >> 5) & 31))];
          int32_t v469_a = v20_lead + 32;
          float v473_data = s0[(v469_a ^ ((v469_a >> 5) & 31))];
          int32_t v479_a = v20_lead + 64;
          float v483_data = s0[(v479_a ^ ((v479_a >> 5) & 31))];
          int32_t v489_a = v20_lead + 96;
          float v493_data = s0[(v489_a ^ ((v489_a >> 5) & 31))];
          int32_t v499_a = v20_lead + 128;
          float v503_data = s0[(v499_a ^ ((v499_a >> 5) & 31))];
          int32_t v509_a = v20_lead + 160;
          float v513_data = s0[(v509_a ^ ((v509_a >> 5) & 31))];
          int32_t v519_a = v20_lead + 192;
          float v523_data = s0[(v519_a ^ ((v519_a >> 5) & 31))];
          int32_t v529_a = v20_lead + 224;
          float v533_data = s0[(v529_a ^ ((v529_a >> 5) & 31))];
          int32_t v539_a = v20_lead + 256;
          float v543_data = s0[(v539_a ^ ((v539_a >> 5) & 31))];
          float v544_acc{};
          float v545_data = r6[8];
          float v546_bc = tensorforge::broadcast<32, 16, 0>(v545_data);
          tensorforge::fmacdpp16<0>(v544_acc, v546_bc, v463_data);
          tensorforge::fmacdpp16<1>(v544_acc, v546_bc, v473_data);
          tensorforge::fmacdpp16<2>(v544_acc, v546_bc, v483_data);
          tensorforge::fmacdpp16<3>(v544_acc, v546_bc, v493_data);
          tensorforge::fmacdpp16<4>(v544_acc, v546_bc, v503_data);
          tensorforge::fmacdpp16<5>(v544_acc, v546_bc, v513_data);
          tensorforge::fmacdpp16<6>(v544_acc, v546_bc, v523_data);
          tensorforge::fmacdpp16<7>(v544_acc, v546_bc, v533_data);
          tensorforge::fmacdpp16<8>(v544_acc, v546_bc, v543_data);
          r7[8] = v544_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v550_i0 = 0; v550_i0 < 1; ++v550_i0) {
            int32_t v558_lead = v20_lead + (v550_i0 * 32);
            #pragma unroll
            for (int32_t v551_i1 = 0; v551_i1 < 9; ++v551_i1) {
              float v553_data = r7[(v550_i0 + v551_i1)];
              glb_m3[(v558_lead + (v551_i1 * 32))] = v553_data;
            }
          }
        }
      }
    }
  }
}

