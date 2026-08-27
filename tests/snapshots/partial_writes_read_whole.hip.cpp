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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m0[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              int32_t v33_a = v27_i1 * 16;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v48_data = r0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = ir1[1];
          ir1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = ir1[8];
          ir1[8] = (v73_data + v72_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
            int32_t v88_lead = v3_lead + (v79_i0 * 32);
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 9; ++v80_i1) {
              int32_t v81_a = v79_i0 + v80_i1;
              float v83_data = r1[(v79_i0 + v80_i1)];
              int32_t v90_a = v88_lead + (v80_i1 * 32);
              s0[v90_a] = v83_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v96_i1 = 0; v96_i1 < 9; ++v96_i1) {
              int32_t v102_a = v96_i1 * 16;
              int32_t v103_a = v3_lead + v102_a;
              float v111_data = __builtin_nontemporal_load(&glb_m2[(v3_lead + v102_a)]);
              int32_t v112_a = 0 + v96_i1;
              r4[v112_a] = v111_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if (v3_lead < 16) {
            float v118_data = r2[0];
            float v119_data = ir3[0];
            ir3[0] = (v119_data + v118_data);
            float v121_data = r2[1];
            float v122_data = ir3[1];
            ir3[1] = (v122_data + v121_data);
            float v124_data = r2[2];
            float v125_data = ir3[2];
            ir3[2] = (v125_data + v124_data);
            float v127_data = r2[3];
            float v128_data = ir3[3];
            ir3[3] = (v128_data + v127_data);
            float v130_data = r2[4];
            float v131_data = ir3[4];
            ir3[4] = (v131_data + v130_data);
            float v133_data = r2[5];
            float v134_data = ir3[5];
            ir3[5] = (v134_data + v133_data);
            float v136_data = r2[6];
            float v137_data = ir3[6];
            ir3[6] = (v137_data + v136_data);
            float v139_data = r2[7];
            float v140_data = ir3[7];
            ir3[7] = (v140_data + v139_data);
            float v142_data = r2[8];
            float v143_data = ir3[8];
            ir3[8] = (v143_data + v142_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v149_i1 = 0; v149_i1 < 9; ++v149_i1) {
              int32_t v150_a = 0 + v149_i1;
              float v152_data = r3[v149_i1];
              int32_t v159_a = v3_lead + (v149_i1 * 32);
              s0[v159_a] = v152_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v161_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v161_lin;
          float v162_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v162_lin;
          float v163_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v163_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          auto& ir5 = r5;
          if (v3_lead < 16) {
            float v169_data = r4[0];
            float v170_data = ir5[0];
            ir5[0] = (v170_data + v169_data);
            float v172_data = r4[1];
            float v173_data = ir5[1];
            ir5[1] = (v173_data + v172_data);
            float v175_data = r4[2];
            float v176_data = ir5[2];
            ir5[2] = (v176_data + v175_data);
            float v178_data = r4[3];
            float v179_data = ir5[3];
            ir5[3] = (v179_data + v178_data);
            float v181_data = r4[4];
            float v182_data = ir5[4];
            ir5[4] = (v182_data + v181_data);
            float v184_data = r4[5];
            float v185_data = ir5[5];
            ir5[5] = (v185_data + v184_data);
            float v187_data = r4[6];
            float v188_data = ir5[6];
            ir5[6] = (v188_data + v187_data);
            float v190_data = r4[7];
            float v191_data = ir5[7];
            ir5[7] = (v191_data + v190_data);
            float v193_data = r4[8];
            float v194_data = ir5[8];
            ir5[8] = (v194_data + v193_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v200_i1 = 0; v200_i1 < 9; ++v200_i1) {
              int32_t v201_a = 0 + v200_i1;
              float v203_data = r5[v200_i1];
              int32_t v210_a = v3_lead + (v200_i1 * 32);
              s0[v210_a] = v203_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v217_a = v3_lead + 0;
          float v224_data = s0[v3_lead];
          int32_t v230_a = v3_lead + 32;
          float v237_data = s0[(v3_lead + 32)];
          int32_t v243_a = v3_lead + 64;
          float v250_data = s0[(v3_lead + 64)];
          int32_t v256_a = v3_lead + 96;
          float v263_data = s0[(v3_lead + 96)];
          int32_t v269_a = v3_lead + 128;
          float v276_data = s0[(v3_lead + 128)];
          int32_t v282_a = v3_lead + 160;
          float v289_data = s0[(v3_lead + 160)];
          int32_t v295_a = v3_lead + 192;
          float v302_data = s0[(v3_lead + 192)];
          int32_t v308_a = v3_lead + 224;
          float v315_data = s0[(v3_lead + 224)];
          int32_t v321_a = v3_lead + 256;
          float v328_data = s0[(v3_lead + 256)];
          float v329_acc{};
          float v330_acc{};
          float v331_acc{};
          float v332_acc{};
          float v333_acc{};
          float v334_acc{};
          float v335_acc{};
          float v336_acc{};
          float v337_acc{};
          float v338_lin = r6[0];
          float v339_bc = tensorforge::broadcast<32, 16, 0>(v338_lin);
          tensorforge::fmacdpp16<0>(v329_acc, v339_bc, v224_data);
          tensorforge::fmacdpp16<1>(v329_acc, v339_bc, v237_data);
          tensorforge::fmacdpp16<2>(v329_acc, v339_bc, v250_data);
          tensorforge::fmacdpp16<3>(v329_acc, v339_bc, v263_data);
          tensorforge::fmacdpp16<4>(v329_acc, v339_bc, v276_data);
          tensorforge::fmacdpp16<5>(v329_acc, v339_bc, v289_data);
          tensorforge::fmacdpp16<6>(v329_acc, v339_bc, v302_data);
          tensorforge::fmacdpp16<7>(v329_acc, v339_bc, v315_data);
          tensorforge::fmacdpp16<8>(v329_acc, v339_bc, v328_data);
          tensorforge::fmacdpp16<9>(v330_acc, v339_bc, v224_data);
          tensorforge::fmacdpp16<10>(v330_acc, v339_bc, v237_data);
          tensorforge::fmacdpp16<11>(v330_acc, v339_bc, v250_data);
          tensorforge::fmacdpp16<12>(v330_acc, v339_bc, v263_data);
          tensorforge::fmacdpp16<13>(v330_acc, v339_bc, v276_data);
          tensorforge::fmacdpp16<14>(v330_acc, v339_bc, v289_data);
          tensorforge::fmacdpp16<15>(v330_acc, v339_bc, v302_data);
          float v340_bc = tensorforge::broadcast<32, 16, 1>(v338_lin);
          tensorforge::fmacdpp16<0>(v330_acc, v340_bc, v315_data);
          tensorforge::fmacdpp16<1>(v330_acc, v340_bc, v328_data);
          tensorforge::fmacdpp16<2>(v331_acc, v340_bc, v224_data);
          tensorforge::fmacdpp16<3>(v331_acc, v340_bc, v237_data);
          tensorforge::fmacdpp16<4>(v331_acc, v340_bc, v250_data);
          tensorforge::fmacdpp16<5>(v331_acc, v340_bc, v263_data);
          tensorforge::fmacdpp16<6>(v331_acc, v340_bc, v276_data);
          tensorforge::fmacdpp16<7>(v331_acc, v340_bc, v289_data);
          tensorforge::fmacdpp16<8>(v331_acc, v340_bc, v302_data);
          tensorforge::fmacdpp16<9>(v331_acc, v340_bc, v315_data);
          tensorforge::fmacdpp16<10>(v331_acc, v340_bc, v328_data);
          tensorforge::fmacdpp16<11>(v332_acc, v340_bc, v224_data);
          tensorforge::fmacdpp16<12>(v332_acc, v340_bc, v237_data);
          tensorforge::fmacdpp16<13>(v332_acc, v340_bc, v250_data);
          tensorforge::fmacdpp16<14>(v332_acc, v340_bc, v263_data);
          tensorforge::fmacdpp16<15>(v332_acc, v340_bc, v276_data);
          float v341_lin = r6[1];
          float v342_bc = tensorforge::broadcast<32, 16, 0>(v341_lin);
          tensorforge::fmacdpp16<0>(v332_acc, v342_bc, v289_data);
          tensorforge::fmacdpp16<1>(v332_acc, v342_bc, v302_data);
          tensorforge::fmacdpp16<2>(v332_acc, v342_bc, v315_data);
          tensorforge::fmacdpp16<3>(v332_acc, v342_bc, v328_data);
          tensorforge::fmacdpp16<4>(v333_acc, v342_bc, v224_data);
          tensorforge::fmacdpp16<5>(v333_acc, v342_bc, v237_data);
          tensorforge::fmacdpp16<6>(v333_acc, v342_bc, v250_data);
          tensorforge::fmacdpp16<7>(v333_acc, v342_bc, v263_data);
          tensorforge::fmacdpp16<8>(v333_acc, v342_bc, v276_data);
          tensorforge::fmacdpp16<9>(v333_acc, v342_bc, v289_data);
          tensorforge::fmacdpp16<10>(v333_acc, v342_bc, v302_data);
          tensorforge::fmacdpp16<11>(v333_acc, v342_bc, v315_data);
          tensorforge::fmacdpp16<12>(v333_acc, v342_bc, v328_data);
          tensorforge::fmacdpp16<13>(v334_acc, v342_bc, v224_data);
          tensorforge::fmacdpp16<14>(v334_acc, v342_bc, v237_data);
          tensorforge::fmacdpp16<15>(v334_acc, v342_bc, v250_data);
          float v343_bc = tensorforge::broadcast<32, 16, 1>(v341_lin);
          tensorforge::fmacdpp16<0>(v334_acc, v343_bc, v263_data);
          tensorforge::fmacdpp16<1>(v334_acc, v343_bc, v276_data);
          tensorforge::fmacdpp16<2>(v334_acc, v343_bc, v289_data);
          tensorforge::fmacdpp16<3>(v334_acc, v343_bc, v302_data);
          tensorforge::fmacdpp16<4>(v334_acc, v343_bc, v315_data);
          tensorforge::fmacdpp16<5>(v334_acc, v343_bc, v328_data);
          tensorforge::fmacdpp16<6>(v335_acc, v343_bc, v224_data);
          tensorforge::fmacdpp16<7>(v335_acc, v343_bc, v237_data);
          tensorforge::fmacdpp16<8>(v335_acc, v343_bc, v250_data);
          tensorforge::fmacdpp16<9>(v335_acc, v343_bc, v263_data);
          tensorforge::fmacdpp16<10>(v335_acc, v343_bc, v276_data);
          tensorforge::fmacdpp16<11>(v335_acc, v343_bc, v289_data);
          tensorforge::fmacdpp16<12>(v335_acc, v343_bc, v302_data);
          tensorforge::fmacdpp16<13>(v335_acc, v343_bc, v315_data);
          tensorforge::fmacdpp16<14>(v335_acc, v343_bc, v328_data);
          tensorforge::fmacdpp16<15>(v336_acc, v343_bc, v224_data);
          float v344_lin = r6[2];
          float v345_bc = tensorforge::broadcast<32, 16, 0>(v344_lin);
          tensorforge::fmacdpp16<0>(v336_acc, v345_bc, v237_data);
          tensorforge::fmacdpp16<1>(v336_acc, v345_bc, v250_data);
          tensorforge::fmacdpp16<2>(v336_acc, v345_bc, v263_data);
          tensorforge::fmacdpp16<3>(v336_acc, v345_bc, v276_data);
          tensorforge::fmacdpp16<4>(v336_acc, v345_bc, v289_data);
          tensorforge::fmacdpp16<5>(v336_acc, v345_bc, v302_data);
          tensorforge::fmacdpp16<6>(v336_acc, v345_bc, v315_data);
          tensorforge::fmacdpp16<7>(v336_acc, v345_bc, v328_data);
          tensorforge::fmacdpp16<8>(v337_acc, v345_bc, v224_data);
          tensorforge::fmacdpp16<9>(v337_acc, v345_bc, v237_data);
          tensorforge::fmacdpp16<10>(v337_acc, v345_bc, v250_data);
          tensorforge::fmacdpp16<11>(v337_acc, v345_bc, v263_data);
          tensorforge::fmacdpp16<12>(v337_acc, v345_bc, v276_data);
          tensorforge::fmacdpp16<13>(v337_acc, v345_bc, v289_data);
          tensorforge::fmacdpp16<14>(v337_acc, v345_bc, v302_data);
          tensorforge::fmacdpp16<15>(v337_acc, v345_bc, v315_data);
          tensorforge::fmacdpp16<0>(v337_acc, (tensorforge::broadcast<32, 16, 1>(v344_lin)), v328_data);
          ir7[0] = v329_acc;
          ir7[1] = v330_acc;
          ir7[2] = v331_acc;
          ir7[3] = v332_acc;
          ir7[4] = v333_acc;
          ir7[5] = v334_acc;
          ir7[6] = v335_acc;
          ir7[7] = v336_acc;
          ir7[8] = v337_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v350_i0 = 0; v350_i0 < 1; ++v350_i0) {
            int32_t v359_lead = v3_lead + (v350_i0 * 32);
            #pragma unroll
            for (int32_t v351_i1 = 0; v351_i1 < 9; ++v351_i1) {
              int32_t v352_a = v350_i0 + v351_i1;
              float v354_data = r7[(v350_i0 + v351_i1)];
              int32_t v361_a = v359_lead + (v351_i1 * 32);
              glb_m3[v361_a] = v354_data;
            }
          }
          ;
        }
      }
    }
  }
}

