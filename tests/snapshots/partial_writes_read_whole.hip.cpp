// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 2560 * sizeof(float)));
        CHECK_ERR;
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7ab185b978), hipFuncAttributeMaxDynamicSharedMemorySize, 2560 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_7ab185b978, grid, block, 2560 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[320 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      __syncthreads();
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          auto glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[batchId0][0 + m3_extraOffset];
          auto glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v23_lead + (v18_i1 * 32))]);
              r0[(v17_i0 + v18_i1)] = v26_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v33_i1 * 16))]);
              r2[v33_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
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
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
            int32_t v85_lead = v16_lead + (v77_i0 * 32);
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 9; ++v78_i1) {
              float v80_data = r1[(v77_i0 + v78_i1)];
              int32_t v87_a = v85_lead + (v78_i1 * 32);
              s0[(v87_a ^ ((v87_a >> 5) & 31))] = v80_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v96_i1 = 0; v96_i1 < 9; ++v96_i1) {
              float v104_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v96_i1 * 16))]);
              r4[v96_i1] = v104_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v16_lead < 16) {
            float v111_data = r2[0];
            float v112_data = r3[0];
            r3[0] = (v112_data + v111_data);
            float v114_data = r2[1];
            float v115_data = r3[1];
            r3[1] = (v115_data + v114_data);
            float v117_data = r2[2];
            float v118_data = r3[2];
            r3[2] = (v118_data + v117_data);
            float v120_data = r2[3];
            float v121_data = r3[3];
            r3[3] = (v121_data + v120_data);
            float v123_data = r2[4];
            float v124_data = r3[4];
            r3[4] = (v124_data + v123_data);
            float v126_data = r2[5];
            float v127_data = r3[5];
            r3[5] = (v127_data + v126_data);
            float v129_data = r2[6];
            float v130_data = r3[6];
            r3[6] = (v130_data + v129_data);
            float v132_data = r2[7];
            float v133_data = r3[7];
            r3[7] = (v133_data + v132_data);
            float v135_data = r2[8];
            float v136_data = r3[8];
            r3[8] = (v136_data + v135_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v142_i1 = 0; v142_i1 < 9; ++v142_i1) {
              float v144_data = r3[v142_i1];
              int32_t v151_a = v16_lead + (v142_i1 * 32);
              s0[(v151_a ^ ((v151_a >> 5) & 31))] = v144_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v16_lead < 9) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 9; ++v160_i1) {
              float v168_data = __builtin_nontemporal_load(&glb_m4[(v16_lead + (v160_i1 * 9))]);
              r6[v160_i1] = v168_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v16_lead < 16) {
            float v175_data = r4[0];
            float v176_data = r5[0];
            r5[0] = (v176_data + v175_data);
            float v178_data = r4[1];
            float v179_data = r5[1];
            r5[1] = (v179_data + v178_data);
            float v181_data = r4[2];
            float v182_data = r5[2];
            r5[2] = (v182_data + v181_data);
            float v184_data = r4[3];
            float v185_data = r5[3];
            r5[3] = (v185_data + v184_data);
            float v187_data = r4[4];
            float v188_data = r5[4];
            r5[4] = (v188_data + v187_data);
            float v190_data = r4[5];
            float v191_data = r5[5];
            r5[5] = (v191_data + v190_data);
            float v193_data = r4[6];
            float v194_data = r5[6];
            r5[6] = (v194_data + v193_data);
            float v196_data = r4[7];
            float v197_data = r5[7];
            r5[7] = (v197_data + v196_data);
            float v199_data = r4[8];
            float v200_data = r5[8];
            r5[8] = (v200_data + v199_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v206_i1 = 0; v206_i1 < 9; ++v206_i1) {
              float v208_data = r5[v206_i1];
              int32_t v215_a = v16_lead + (v206_i1 * 32);
              s0[(v215_a ^ ((v215_a >> 5) & 31))] = v208_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v220_data = r6[0];
          float v221_data = r6[1];
          float v222_data = r6[2];
          float v223_data = r6[3];
          float v224_tp{};
          float v225_tp{};
          float v226_tp{};
          float v227_tp{};
          tensorforge::transpose4x4b32(v224_tp, v225_tp, v226_tp, v227_tp, v220_data, v221_data, v222_data, v223_data);
          tensorforge::VectorT<float, 4> v228_acc{};
          float v238_data = s0[(v16_lead ^ ((v16_lead >> 5) & 31))];
          int32_t v244_a = v16_lead + 32;
          float v248_data = s0[(v244_a ^ ((v244_a >> 5) & 31))];
          int32_t v254_a = v16_lead + 64;
          float v258_data = s0[(v254_a ^ ((v254_a >> 5) & 31))];
          int32_t v264_a = v16_lead + 96;
          float v268_data = s0[(v264_a ^ ((v264_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v238_data, v228_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v248_data, v269_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v258_data, v270_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v268_data, v271_acc, 3, 0, 0);
          int32_t v278_a = v16_lead + 128;
          float v282_data = s0[(v278_a ^ ((v278_a >> 5) & 31))];
          int32_t v288_a = v16_lead + 160;
          float v292_data = s0[(v288_a ^ ((v288_a >> 5) & 31))];
          int32_t v298_a = v16_lead + 192;
          float v302_data = s0[(v298_a ^ ((v298_a >> 5) & 31))];
          int32_t v308_a = v16_lead + 224;
          float v312_data = s0[(v308_a ^ ((v308_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v282_data, v272_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v292_data, v313_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v302_data, v314_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v312_data, v315_acc, 3, 1, 0);
          int32_t v322_a = v16_lead + 256;
          float v326_data = s0[(v322_a ^ ((v322_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v326_data, v316_acc, 3, 2, 0);
          r7[0] = (v330_acc[0]);
          r7[1] = (v330_acc[1]);
          r7[2] = (v330_acc[2]);
          r7[3] = (v330_acc[3]);
          float v335_data = r6[4];
          float v336_data = r6[5];
          float v337_data = r6[6];
          float v338_data = r6[7];
          float v339_tp{};
          float v340_tp{};
          float v341_tp{};
          float v342_tp{};
          tensorforge::transpose4x4b32(v339_tp, v340_tp, v341_tp, v342_tp, v335_data, v336_data, v337_data, v338_data);
          tensorforge::VectorT<float, 4> v343_acc{};
          float v353_data = s0[(v16_lead ^ ((v16_lead >> 5) & 31))];
          int32_t v359_a = v16_lead + 32;
          float v363_data = s0[(v359_a ^ ((v359_a >> 5) & 31))];
          int32_t v369_a = v16_lead + 64;
          float v373_data = s0[(v369_a ^ ((v369_a >> 5) & 31))];
          int32_t v379_a = v16_lead + 96;
          float v383_data = s0[(v379_a ^ ((v379_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v353_data, v343_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v363_data, v384_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v373_data, v385_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v383_data, v386_acc, 3, 0, 0);
          int32_t v393_a = v16_lead + 128;
          float v397_data = s0[(v393_a ^ ((v393_a >> 5) & 31))];
          int32_t v403_a = v16_lead + 160;
          float v407_data = s0[(v403_a ^ ((v403_a >> 5) & 31))];
          int32_t v413_a = v16_lead + 192;
          float v417_data = s0[(v413_a ^ ((v413_a >> 5) & 31))];
          int32_t v423_a = v16_lead + 224;
          float v427_data = s0[(v423_a ^ ((v423_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v397_data, v387_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v407_data, v428_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v417_data, v429_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v342_tp, v427_data, v430_acc, 3, 1, 0);
          int32_t v437_a = v16_lead + 256;
          float v441_data = s0[(v437_a ^ ((v437_a >> 5) & 31))];
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v441_data, v431_acc, 3, 2, 0);
          r7[4] = (v445_acc[0]);
          r7[5] = (v445_acc[1]);
          r7[6] = (v445_acc[2]);
          r7[7] = (v445_acc[3]);
          float v459_data = s0[(v16_lead ^ ((v16_lead >> 5) & 31))];
          int32_t v465_a = v16_lead + 32;
          float v469_data = s0[(v465_a ^ ((v465_a >> 5) & 31))];
          int32_t v475_a = v16_lead + 64;
          float v479_data = s0[(v475_a ^ ((v475_a >> 5) & 31))];
          int32_t v485_a = v16_lead + 96;
          float v489_data = s0[(v485_a ^ ((v485_a >> 5) & 31))];
          int32_t v495_a = v16_lead + 128;
          float v499_data = s0[(v495_a ^ ((v495_a >> 5) & 31))];
          int32_t v505_a = v16_lead + 160;
          float v509_data = s0[(v505_a ^ ((v505_a >> 5) & 31))];
          int32_t v515_a = v16_lead + 192;
          float v519_data = s0[(v515_a ^ ((v515_a >> 5) & 31))];
          int32_t v525_a = v16_lead + 224;
          float v529_data = s0[(v525_a ^ ((v525_a >> 5) & 31))];
          int32_t v535_a = v16_lead + 256;
          float v539_data = s0[(v535_a ^ ((v535_a >> 5) & 31))];
          float v540_acc{};
          float v541_data = r6[8];
          float v542_bc = tensorforge::broadcast<32, 16, 0>(v541_data);
          tensorforge::fmacdpp16<0>(v540_acc, v542_bc, v459_data);
          tensorforge::fmacdpp16<1>(v540_acc, v542_bc, v469_data);
          tensorforge::fmacdpp16<2>(v540_acc, v542_bc, v479_data);
          tensorforge::fmacdpp16<3>(v540_acc, v542_bc, v489_data);
          tensorforge::fmacdpp16<4>(v540_acc, v542_bc, v499_data);
          tensorforge::fmacdpp16<5>(v540_acc, v542_bc, v509_data);
          tensorforge::fmacdpp16<6>(v540_acc, v542_bc, v519_data);
          tensorforge::fmacdpp16<7>(v540_acc, v542_bc, v529_data);
          tensorforge::fmacdpp16<8>(v540_acc, v542_bc, v539_data);
          r7[8] = v540_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v546_i0 = 0; v546_i0 < 1; ++v546_i0) {
            int32_t v554_lead = v16_lead + (v546_i0 * 32);
            #pragma unroll
            for (int32_t v547_i1 = 0; v547_i1 < 9; ++v547_i1) {
              float v549_data = r7[(v546_i0 + v547_i1)];
              glb_m3[(v554_lead + (v547_i1 * 32))] = v549_data;
            }
          }
        }
      }
    }
  }
}

