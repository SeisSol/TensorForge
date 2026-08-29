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
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 9; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m0[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + v14_i1)] = v22_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v12_lead + (v29_i1 * 16))]);
              r2[v29_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v74_i0 = 0; v74_i0 < 1; ++v74_i0) {
            int32_t v82_lead = v12_lead + (v74_i0 * 32);
            #pragma unroll
            for (int32_t v75_i1 = 0; v75_i1 < 9; ++v75_i1) {
              float v77_data = r1[(v74_i0 + v75_i1)];
              int32_t v84_a = v82_lead + (v75_i1 * 32);
              s0[(v84_a ^ ((v84_a >> 5) & 31))] = v77_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 9; ++v93_i1) {
              float v101_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v93_i1 * 16))]);
              r4[v93_i1] = v101_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v12_lead < 16) {
            float v108_data = r2[0];
            float v109_data = r3[0];
            r3[0] = (v109_data + v108_data);
            float v111_data = r2[1];
            float v112_data = r3[1];
            r3[1] = (v112_data + v111_data);
            float v114_data = r2[2];
            float v115_data = r3[2];
            r3[2] = (v115_data + v114_data);
            float v117_data = r2[3];
            float v118_data = r3[3];
            r3[3] = (v118_data + v117_data);
            float v120_data = r2[4];
            float v121_data = r3[4];
            r3[4] = (v121_data + v120_data);
            float v123_data = r2[5];
            float v124_data = r3[5];
            r3[5] = (v124_data + v123_data);
            float v126_data = r2[6];
            float v127_data = r3[6];
            r3[6] = (v127_data + v126_data);
            float v129_data = r2[7];
            float v130_data = r3[7];
            r3[7] = (v130_data + v129_data);
            float v132_data = r2[8];
            float v133_data = r3[8];
            r3[8] = (v133_data + v132_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v139_i1 = 0; v139_i1 < 9; ++v139_i1) {
              float v141_data = r3[v139_i1];
              int32_t v148_a = v12_lead + (v139_i1 * 32);
              s0[(v148_a ^ ((v148_a >> 5) & 31))] = v141_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v153_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v153_lin;
          float v154_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v154_lin;
          float v155_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v155_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v12_lead < 16) {
            float v161_data = r4[0];
            float v162_data = r5[0];
            r5[0] = (v162_data + v161_data);
            float v164_data = r4[1];
            float v165_data = r5[1];
            r5[1] = (v165_data + v164_data);
            float v167_data = r4[2];
            float v168_data = r5[2];
            r5[2] = (v168_data + v167_data);
            float v170_data = r4[3];
            float v171_data = r5[3];
            r5[3] = (v171_data + v170_data);
            float v173_data = r4[4];
            float v174_data = r5[4];
            r5[4] = (v174_data + v173_data);
            float v176_data = r4[5];
            float v177_data = r5[5];
            r5[5] = (v177_data + v176_data);
            float v179_data = r4[6];
            float v180_data = r5[6];
            r5[6] = (v180_data + v179_data);
            float v182_data = r4[7];
            float v183_data = r5[7];
            r5[7] = (v183_data + v182_data);
            float v185_data = r4[8];
            float v186_data = r5[8];
            r5[8] = (v186_data + v185_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v192_i1 = 0; v192_i1 < 9; ++v192_i1) {
              float v194_data = r5[v192_i1];
              int32_t v201_a = v12_lead + (v192_i1 * 32);
              s0[(v201_a ^ ((v201_a >> 5) & 31))] = v194_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v215_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          int32_t v221_a = v12_lead + 32;
          float v225_data = s0[(v221_a ^ ((v221_a >> 5) & 31))];
          int32_t v231_a = v12_lead + 64;
          float v235_data = s0[(v231_a ^ ((v231_a >> 5) & 31))];
          int32_t v241_a = v12_lead + 96;
          float v245_data = s0[(v241_a ^ ((v241_a >> 5) & 31))];
          int32_t v251_a = v12_lead + 128;
          float v255_data = s0[(v251_a ^ ((v251_a >> 5) & 31))];
          int32_t v261_a = v12_lead + 160;
          float v265_data = s0[(v261_a ^ ((v261_a >> 5) & 31))];
          int32_t v271_a = v12_lead + 192;
          float v275_data = s0[(v271_a ^ ((v271_a >> 5) & 31))];
          int32_t v281_a = v12_lead + 224;
          float v285_data = s0[(v281_a ^ ((v281_a >> 5) & 31))];
          int32_t v291_a = v12_lead + 256;
          float v295_data = s0[(v291_a ^ ((v291_a >> 5) & 31))];
          float v296_acc{};
          float v297_acc{};
          float v298_acc{};
          float v299_acc{};
          float v300_acc{};
          float v301_acc{};
          float v302_acc{};
          float v303_acc{};
          float v304_acc{};
          float v305_lin = r6[0];
          float v306_bc = tensorforge::broadcast<32, 16, 0>(v305_lin);
          tensorforge::fmacdpp16<0>(v296_acc, v306_bc, v215_data);
          tensorforge::fmacdpp16<1>(v296_acc, v306_bc, v225_data);
          tensorforge::fmacdpp16<2>(v296_acc, v306_bc, v235_data);
          tensorforge::fmacdpp16<3>(v296_acc, v306_bc, v245_data);
          tensorforge::fmacdpp16<4>(v296_acc, v306_bc, v255_data);
          tensorforge::fmacdpp16<5>(v296_acc, v306_bc, v265_data);
          tensorforge::fmacdpp16<6>(v296_acc, v306_bc, v275_data);
          tensorforge::fmacdpp16<7>(v296_acc, v306_bc, v285_data);
          tensorforge::fmacdpp16<8>(v296_acc, v306_bc, v295_data);
          tensorforge::fmacdpp16<9>(v297_acc, v306_bc, v215_data);
          tensorforge::fmacdpp16<10>(v297_acc, v306_bc, v225_data);
          tensorforge::fmacdpp16<11>(v297_acc, v306_bc, v235_data);
          tensorforge::fmacdpp16<12>(v297_acc, v306_bc, v245_data);
          tensorforge::fmacdpp16<13>(v297_acc, v306_bc, v255_data);
          tensorforge::fmacdpp16<14>(v297_acc, v306_bc, v265_data);
          tensorforge::fmacdpp16<15>(v297_acc, v306_bc, v275_data);
          float v307_bc = tensorforge::broadcast<32, 16, 1>(v305_lin);
          tensorforge::fmacdpp16<0>(v297_acc, v307_bc, v285_data);
          tensorforge::fmacdpp16<1>(v297_acc, v307_bc, v295_data);
          tensorforge::fmacdpp16<2>(v298_acc, v307_bc, v215_data);
          tensorforge::fmacdpp16<3>(v298_acc, v307_bc, v225_data);
          tensorforge::fmacdpp16<4>(v298_acc, v307_bc, v235_data);
          tensorforge::fmacdpp16<5>(v298_acc, v307_bc, v245_data);
          tensorforge::fmacdpp16<6>(v298_acc, v307_bc, v255_data);
          tensorforge::fmacdpp16<7>(v298_acc, v307_bc, v265_data);
          tensorforge::fmacdpp16<8>(v298_acc, v307_bc, v275_data);
          tensorforge::fmacdpp16<9>(v298_acc, v307_bc, v285_data);
          tensorforge::fmacdpp16<10>(v298_acc, v307_bc, v295_data);
          tensorforge::fmacdpp16<11>(v299_acc, v307_bc, v215_data);
          tensorforge::fmacdpp16<12>(v299_acc, v307_bc, v225_data);
          tensorforge::fmacdpp16<13>(v299_acc, v307_bc, v235_data);
          tensorforge::fmacdpp16<14>(v299_acc, v307_bc, v245_data);
          tensorforge::fmacdpp16<15>(v299_acc, v307_bc, v255_data);
          float v308_lin = r6[1];
          float v309_bc = tensorforge::broadcast<32, 16, 0>(v308_lin);
          tensorforge::fmacdpp16<0>(v299_acc, v309_bc, v265_data);
          tensorforge::fmacdpp16<1>(v299_acc, v309_bc, v275_data);
          tensorforge::fmacdpp16<2>(v299_acc, v309_bc, v285_data);
          tensorforge::fmacdpp16<3>(v299_acc, v309_bc, v295_data);
          tensorforge::fmacdpp16<4>(v300_acc, v309_bc, v215_data);
          tensorforge::fmacdpp16<5>(v300_acc, v309_bc, v225_data);
          tensorforge::fmacdpp16<6>(v300_acc, v309_bc, v235_data);
          tensorforge::fmacdpp16<7>(v300_acc, v309_bc, v245_data);
          tensorforge::fmacdpp16<8>(v300_acc, v309_bc, v255_data);
          tensorforge::fmacdpp16<9>(v300_acc, v309_bc, v265_data);
          tensorforge::fmacdpp16<10>(v300_acc, v309_bc, v275_data);
          tensorforge::fmacdpp16<11>(v300_acc, v309_bc, v285_data);
          tensorforge::fmacdpp16<12>(v300_acc, v309_bc, v295_data);
          tensorforge::fmacdpp16<13>(v301_acc, v309_bc, v215_data);
          tensorforge::fmacdpp16<14>(v301_acc, v309_bc, v225_data);
          tensorforge::fmacdpp16<15>(v301_acc, v309_bc, v235_data);
          float v310_bc = tensorforge::broadcast<32, 16, 1>(v308_lin);
          tensorforge::fmacdpp16<0>(v301_acc, v310_bc, v245_data);
          tensorforge::fmacdpp16<1>(v301_acc, v310_bc, v255_data);
          tensorforge::fmacdpp16<2>(v301_acc, v310_bc, v265_data);
          tensorforge::fmacdpp16<3>(v301_acc, v310_bc, v275_data);
          tensorforge::fmacdpp16<4>(v301_acc, v310_bc, v285_data);
          tensorforge::fmacdpp16<5>(v301_acc, v310_bc, v295_data);
          tensorforge::fmacdpp16<6>(v302_acc, v310_bc, v215_data);
          tensorforge::fmacdpp16<7>(v302_acc, v310_bc, v225_data);
          tensorforge::fmacdpp16<8>(v302_acc, v310_bc, v235_data);
          tensorforge::fmacdpp16<9>(v302_acc, v310_bc, v245_data);
          tensorforge::fmacdpp16<10>(v302_acc, v310_bc, v255_data);
          tensorforge::fmacdpp16<11>(v302_acc, v310_bc, v265_data);
          tensorforge::fmacdpp16<12>(v302_acc, v310_bc, v275_data);
          tensorforge::fmacdpp16<13>(v302_acc, v310_bc, v285_data);
          tensorforge::fmacdpp16<14>(v302_acc, v310_bc, v295_data);
          tensorforge::fmacdpp16<15>(v303_acc, v310_bc, v215_data);
          float v311_lin = r6[2];
          float v312_bc = tensorforge::broadcast<32, 16, 0>(v311_lin);
          tensorforge::fmacdpp16<0>(v303_acc, v312_bc, v225_data);
          tensorforge::fmacdpp16<1>(v303_acc, v312_bc, v235_data);
          tensorforge::fmacdpp16<2>(v303_acc, v312_bc, v245_data);
          tensorforge::fmacdpp16<3>(v303_acc, v312_bc, v255_data);
          tensorforge::fmacdpp16<4>(v303_acc, v312_bc, v265_data);
          tensorforge::fmacdpp16<5>(v303_acc, v312_bc, v275_data);
          tensorforge::fmacdpp16<6>(v303_acc, v312_bc, v285_data);
          tensorforge::fmacdpp16<7>(v303_acc, v312_bc, v295_data);
          tensorforge::fmacdpp16<8>(v304_acc, v312_bc, v215_data);
          tensorforge::fmacdpp16<9>(v304_acc, v312_bc, v225_data);
          tensorforge::fmacdpp16<10>(v304_acc, v312_bc, v235_data);
          tensorforge::fmacdpp16<11>(v304_acc, v312_bc, v245_data);
          tensorforge::fmacdpp16<12>(v304_acc, v312_bc, v255_data);
          tensorforge::fmacdpp16<13>(v304_acc, v312_bc, v265_data);
          tensorforge::fmacdpp16<14>(v304_acc, v312_bc, v275_data);
          tensorforge::fmacdpp16<15>(v304_acc, v312_bc, v285_data);
          tensorforge::fmacdpp16<0>(v304_acc, (tensorforge::broadcast<32, 16, 1>(v311_lin)), v295_data);
          r7[0] = v296_acc;
          r7[1] = v297_acc;
          r7[2] = v298_acc;
          r7[3] = v299_acc;
          r7[4] = v300_acc;
          r7[5] = v301_acc;
          r7[6] = v302_acc;
          r7[7] = v303_acc;
          r7[8] = v304_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v317_i0 = 0; v317_i0 < 1; ++v317_i0) {
            int32_t v325_lead = v12_lead + (v317_i0 * 32);
            #pragma unroll
            for (int32_t v318_i1 = 0; v318_i1 < 9; ++v318_i1) {
              float v320_data = r7[(v317_i0 + v318_i1)];
              glb_m3[(v325_lead + (v318_i1 * 32))] = v320_data;
            }
          }
        }
      }
    }
  }
}

