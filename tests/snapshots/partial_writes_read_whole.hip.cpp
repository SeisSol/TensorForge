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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          auto glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[batchId0][0 + m3_extraOffset];
          auto glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v9_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v15_lead = v10_i0 * 32;
            int32_t v16_lead = v9_lead + v15_lead;
            int32_t v23_lead = v9_lead + v15_lead;
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 9; ++v11_i1) {
              int32_t v17_a = v11_i1 * 32;
              int32_t v18_a = v16_lead + v17_a;
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v23_lead + v17_a)]);
              r0[(v10_i0 + v11_i1)] = v26_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
              int32_t v39_a = v33_i1 * 16;
              int32_t v40_a = v9_lead + v39_a;
              float v48_data = __builtin_nontemporal_load(&glb_m1[(v9_lead + v39_a)]);
              r2[v33_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v54_data = r0[0];
          float v55_data = r1[0];
          r1[0] = (v55_data + v54_data);
          float v57_data = r0[1];
          float v58_data = r1[1];
          r1[1] = (v58_data + v57_data);
          float v60_data = r0[2];
          float v61_data = r1[2];
          r1[2] = (v61_data + v60_data);
          float v63_data = r0[3];
          float v64_data = r1[3];
          r1[3] = (v64_data + v63_data);
          float v66_data = r0[4];
          float v67_data = r1[4];
          r1[4] = (v67_data + v66_data);
          float v69_data = r0[5];
          float v70_data = r1[5];
          r1[5] = (v70_data + v69_data);
          float v72_data = r0[6];
          float v73_data = r1[6];
          r1[6] = (v73_data + v72_data);
          float v75_data = r0[7];
          float v76_data = r1[7];
          r1[7] = (v76_data + v75_data);
          float v78_data = r0[8];
          float v79_data = r1[8];
          r1[8] = (v79_data + v78_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v85_i0 = 0; v85_i0 < 1; ++v85_i0) {
            int32_t v94_lead = v9_lead + (v85_i0 * 32);
            #pragma unroll
            for (int32_t v86_i1 = 0; v86_i1 < 9; ++v86_i1) {
              int32_t v87_a = v85_i0 + v86_i1;
              float v89_data = r1[(v85_i0 + v86_i1)];
              s0[(v94_lead + (v86_i1 * 32))] = v89_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v102_i1 = 0; v102_i1 < 9; ++v102_i1) {
              int32_t v108_a = v102_i1 * 16;
              int32_t v109_a = v9_lead + v108_a;
              float v117_data = __builtin_nontemporal_load(&glb_m2[(v9_lead + v108_a)]);
              r4[v102_i1] = v117_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v9_lead < 16) {
            float v124_data = r2[0];
            float v125_data = r3[0];
            r3[0] = (v125_data + v124_data);
            float v127_data = r2[1];
            float v128_data = r3[1];
            r3[1] = (v128_data + v127_data);
            float v130_data = r2[2];
            float v131_data = r3[2];
            r3[2] = (v131_data + v130_data);
            float v133_data = r2[3];
            float v134_data = r3[3];
            r3[3] = (v134_data + v133_data);
            float v136_data = r2[4];
            float v137_data = r3[4];
            r3[4] = (v137_data + v136_data);
            float v139_data = r2[5];
            float v140_data = r3[5];
            r3[5] = (v140_data + v139_data);
            float v142_data = r2[6];
            float v143_data = r3[6];
            r3[6] = (v143_data + v142_data);
            float v145_data = r2[7];
            float v146_data = r3[7];
            r3[7] = (v146_data + v145_data);
            float v148_data = r2[8];
            float v149_data = r3[8];
            r3[8] = (v149_data + v148_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v155_i1 = 0; v155_i1 < 9; ++v155_i1) {
              int32_t v156_a = 0 + v155_i1;
              float v158_data = r3[v155_i1];
              s0[(v9_lead + (v155_i1 * 32))] = v158_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v167_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v167_lin;
          float v168_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v168_lin;
          float v169_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v169_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v9_lead < 16) {
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
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v206_i1 = 0; v206_i1 < 9; ++v206_i1) {
              int32_t v207_a = 0 + v206_i1;
              float v209_data = r5[v206_i1];
              s0[(v9_lead + (v206_i1 * 32))] = v209_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          int32_t v223_a = v9_lead + 0;
          float v230_data = s0[v9_lead];
          int32_t v236_a = v9_lead + 32;
          float v243_data = s0[(v9_lead + 32)];
          int32_t v249_a = v9_lead + 64;
          float v256_data = s0[(v9_lead + 64)];
          int32_t v262_a = v9_lead + 96;
          float v269_data = s0[(v9_lead + 96)];
          int32_t v275_a = v9_lead + 128;
          float v282_data = s0[(v9_lead + 128)];
          int32_t v288_a = v9_lead + 160;
          float v295_data = s0[(v9_lead + 160)];
          int32_t v301_a = v9_lead + 192;
          float v308_data = s0[(v9_lead + 192)];
          int32_t v314_a = v9_lead + 224;
          float v321_data = s0[(v9_lead + 224)];
          int32_t v327_a = v9_lead + 256;
          float v334_data = s0[(v9_lead + 256)];
          float v335_acc{};
          float v336_acc{};
          float v337_acc{};
          float v338_acc{};
          float v339_acc{};
          float v340_acc{};
          float v341_acc{};
          float v342_acc{};
          float v343_acc{};
          float v344_lin = r6[0];
          float v345_bc = tensorforge::broadcast<32, 16, 0>(v344_lin);
          tensorforge::fmacdpp16<0>(v335_acc, v345_bc, v230_data);
          tensorforge::fmacdpp16<1>(v335_acc, v345_bc, v243_data);
          tensorforge::fmacdpp16<2>(v335_acc, v345_bc, v256_data);
          tensorforge::fmacdpp16<3>(v335_acc, v345_bc, v269_data);
          tensorforge::fmacdpp16<4>(v335_acc, v345_bc, v282_data);
          tensorforge::fmacdpp16<5>(v335_acc, v345_bc, v295_data);
          tensorforge::fmacdpp16<6>(v335_acc, v345_bc, v308_data);
          tensorforge::fmacdpp16<7>(v335_acc, v345_bc, v321_data);
          tensorforge::fmacdpp16<8>(v335_acc, v345_bc, v334_data);
          tensorforge::fmacdpp16<9>(v336_acc, v345_bc, v230_data);
          tensorforge::fmacdpp16<10>(v336_acc, v345_bc, v243_data);
          tensorforge::fmacdpp16<11>(v336_acc, v345_bc, v256_data);
          tensorforge::fmacdpp16<12>(v336_acc, v345_bc, v269_data);
          tensorforge::fmacdpp16<13>(v336_acc, v345_bc, v282_data);
          tensorforge::fmacdpp16<14>(v336_acc, v345_bc, v295_data);
          tensorforge::fmacdpp16<15>(v336_acc, v345_bc, v308_data);
          float v346_bc = tensorforge::broadcast<32, 16, 1>(v344_lin);
          tensorforge::fmacdpp16<0>(v336_acc, v346_bc, v321_data);
          tensorforge::fmacdpp16<1>(v336_acc, v346_bc, v334_data);
          tensorforge::fmacdpp16<2>(v337_acc, v346_bc, v230_data);
          tensorforge::fmacdpp16<3>(v337_acc, v346_bc, v243_data);
          tensorforge::fmacdpp16<4>(v337_acc, v346_bc, v256_data);
          tensorforge::fmacdpp16<5>(v337_acc, v346_bc, v269_data);
          tensorforge::fmacdpp16<6>(v337_acc, v346_bc, v282_data);
          tensorforge::fmacdpp16<7>(v337_acc, v346_bc, v295_data);
          tensorforge::fmacdpp16<8>(v337_acc, v346_bc, v308_data);
          tensorforge::fmacdpp16<9>(v337_acc, v346_bc, v321_data);
          tensorforge::fmacdpp16<10>(v337_acc, v346_bc, v334_data);
          tensorforge::fmacdpp16<11>(v338_acc, v346_bc, v230_data);
          tensorforge::fmacdpp16<12>(v338_acc, v346_bc, v243_data);
          tensorforge::fmacdpp16<13>(v338_acc, v346_bc, v256_data);
          tensorforge::fmacdpp16<14>(v338_acc, v346_bc, v269_data);
          tensorforge::fmacdpp16<15>(v338_acc, v346_bc, v282_data);
          float v347_lin = r6[1];
          float v348_bc = tensorforge::broadcast<32, 16, 0>(v347_lin);
          tensorforge::fmacdpp16<0>(v338_acc, v348_bc, v295_data);
          tensorforge::fmacdpp16<1>(v338_acc, v348_bc, v308_data);
          tensorforge::fmacdpp16<2>(v338_acc, v348_bc, v321_data);
          tensorforge::fmacdpp16<3>(v338_acc, v348_bc, v334_data);
          tensorforge::fmacdpp16<4>(v339_acc, v348_bc, v230_data);
          tensorforge::fmacdpp16<5>(v339_acc, v348_bc, v243_data);
          tensorforge::fmacdpp16<6>(v339_acc, v348_bc, v256_data);
          tensorforge::fmacdpp16<7>(v339_acc, v348_bc, v269_data);
          tensorforge::fmacdpp16<8>(v339_acc, v348_bc, v282_data);
          tensorforge::fmacdpp16<9>(v339_acc, v348_bc, v295_data);
          tensorforge::fmacdpp16<10>(v339_acc, v348_bc, v308_data);
          tensorforge::fmacdpp16<11>(v339_acc, v348_bc, v321_data);
          tensorforge::fmacdpp16<12>(v339_acc, v348_bc, v334_data);
          tensorforge::fmacdpp16<13>(v340_acc, v348_bc, v230_data);
          tensorforge::fmacdpp16<14>(v340_acc, v348_bc, v243_data);
          tensorforge::fmacdpp16<15>(v340_acc, v348_bc, v256_data);
          float v349_bc = tensorforge::broadcast<32, 16, 1>(v347_lin);
          tensorforge::fmacdpp16<0>(v340_acc, v349_bc, v269_data);
          tensorforge::fmacdpp16<1>(v340_acc, v349_bc, v282_data);
          tensorforge::fmacdpp16<2>(v340_acc, v349_bc, v295_data);
          tensorforge::fmacdpp16<3>(v340_acc, v349_bc, v308_data);
          tensorforge::fmacdpp16<4>(v340_acc, v349_bc, v321_data);
          tensorforge::fmacdpp16<5>(v340_acc, v349_bc, v334_data);
          tensorforge::fmacdpp16<6>(v341_acc, v349_bc, v230_data);
          tensorforge::fmacdpp16<7>(v341_acc, v349_bc, v243_data);
          tensorforge::fmacdpp16<8>(v341_acc, v349_bc, v256_data);
          tensorforge::fmacdpp16<9>(v341_acc, v349_bc, v269_data);
          tensorforge::fmacdpp16<10>(v341_acc, v349_bc, v282_data);
          tensorforge::fmacdpp16<11>(v341_acc, v349_bc, v295_data);
          tensorforge::fmacdpp16<12>(v341_acc, v349_bc, v308_data);
          tensorforge::fmacdpp16<13>(v341_acc, v349_bc, v321_data);
          tensorforge::fmacdpp16<14>(v341_acc, v349_bc, v334_data);
          tensorforge::fmacdpp16<15>(v342_acc, v349_bc, v230_data);
          float v350_lin = r6[2];
          float v351_bc = tensorforge::broadcast<32, 16, 0>(v350_lin);
          tensorforge::fmacdpp16<0>(v342_acc, v351_bc, v243_data);
          tensorforge::fmacdpp16<1>(v342_acc, v351_bc, v256_data);
          tensorforge::fmacdpp16<2>(v342_acc, v351_bc, v269_data);
          tensorforge::fmacdpp16<3>(v342_acc, v351_bc, v282_data);
          tensorforge::fmacdpp16<4>(v342_acc, v351_bc, v295_data);
          tensorforge::fmacdpp16<5>(v342_acc, v351_bc, v308_data);
          tensorforge::fmacdpp16<6>(v342_acc, v351_bc, v321_data);
          tensorforge::fmacdpp16<7>(v342_acc, v351_bc, v334_data);
          tensorforge::fmacdpp16<8>(v343_acc, v351_bc, v230_data);
          tensorforge::fmacdpp16<9>(v343_acc, v351_bc, v243_data);
          tensorforge::fmacdpp16<10>(v343_acc, v351_bc, v256_data);
          tensorforge::fmacdpp16<11>(v343_acc, v351_bc, v269_data);
          tensorforge::fmacdpp16<12>(v343_acc, v351_bc, v282_data);
          tensorforge::fmacdpp16<13>(v343_acc, v351_bc, v295_data);
          tensorforge::fmacdpp16<14>(v343_acc, v351_bc, v308_data);
          tensorforge::fmacdpp16<15>(v343_acc, v351_bc, v321_data);
          tensorforge::fmacdpp16<0>(v343_acc, (tensorforge::broadcast<32, 16, 1>(v350_lin)), v334_data);
          r7[0] = v335_acc;
          r7[1] = v336_acc;
          r7[2] = v337_acc;
          r7[3] = v338_acc;
          r7[4] = v339_acc;
          r7[5] = v340_acc;
          r7[6] = v341_acc;
          r7[7] = v342_acc;
          r7[8] = v343_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v356_i0 = 0; v356_i0 < 1; ++v356_i0) {
            int32_t v365_lead = v9_lead + (v356_i0 * 32);
            #pragma unroll
            for (int32_t v357_i1 = 0; v357_i1 < 9; ++v357_i1) {
              int32_t v358_a = v356_i0 + v357_i1;
              float v360_data = r7[(v356_i0 + v357_i1)];
              glb_m3[(v365_lead + (v357_i1 * 32))] = v360_data;
            }
          }
        }
      }
    }
  }
}

