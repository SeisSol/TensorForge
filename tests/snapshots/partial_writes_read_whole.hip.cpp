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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m0[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 16);
              float v26_data;
              {
                v26_data = __builtin_nontemporal_load(&glb_m1[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r2[v27_a] = v26_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v31_data = r0[0];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + v31_data);
          float v34_data = r0[1];
          float v35_data = ir1[1];
          ir1[1] = (v35_data + v34_data);
          float v37_data = r0[2];
          float v38_data = ir1[2];
          ir1[2] = (v38_data + v37_data);
          float v40_data = r0[3];
          float v41_data = ir1[3];
          ir1[3] = (v41_data + v40_data);
          float v43_data = r0[4];
          float v44_data = ir1[4];
          ir1[4] = (v44_data + v43_data);
          float v46_data = r0[5];
          float v47_data = ir1[5];
          ir1[5] = (v47_data + v46_data);
          float v49_data = r0[6];
          float v50_data = ir1[6];
          ir1[6] = (v50_data + v49_data);
          float v52_data = r0[7];
          float v53_data = ir1[7];
          ir1[7] = (v53_data + v52_data);
          float v55_data = r0[8];
          float v56_data = ir1[8];
          ir1[8] = (v56_data + v55_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          int32_t v60_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
            int32_t v70_lead = v60_lead + (v61_i0 * 32);
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
              int32_t v63_a = v61_i0 + v62_i1;
              float v65_data = r1[(v61_i0 + v62_i1)];
              int32_t v72_a = v70_lead + (v62_i1 * 32);
              s0[v72_a] = v65_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          int32_t v75_lead = threadIdx.x % 32;
          if (v75_lead < 16) {
            #pragma unroll
            for (int32_t v77_i1 = 0; v77_i1 < 9; ++v77_i1) {
              int32_t v84_a = v75_lead + (v77_i1 * 16);
              float v85_data;
              {
                v85_data = __builtin_nontemporal_load(&glb_m2[v84_a]);
              }
              int32_t v86_a = 0 + v77_i1;
              r4[v86_a] = v85_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if ((threadIdx.x % 32) < 16) {
            float v91_data = r2[0];
            float v92_data = ir3[0];
            ir3[0] = (v92_data + v91_data);
            float v94_data = r2[1];
            float v95_data = ir3[1];
            ir3[1] = (v95_data + v94_data);
            float v97_data = r2[2];
            float v98_data = ir3[2];
            ir3[2] = (v98_data + v97_data);
            float v100_data = r2[3];
            float v101_data = ir3[3];
            ir3[3] = (v101_data + v100_data);
            float v103_data = r2[4];
            float v104_data = ir3[4];
            ir3[4] = (v104_data + v103_data);
            float v106_data = r2[5];
            float v107_data = ir3[5];
            ir3[5] = (v107_data + v106_data);
            float v109_data = r2[6];
            float v110_data = ir3[6];
            ir3[6] = (v110_data + v109_data);
            float v112_data = r2[7];
            float v113_data = ir3[7];
            ir3[7] = (v113_data + v112_data);
            float v115_data = r2[8];
            float v116_data = ir3[8];
            ir3[8] = (v116_data + v115_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          int32_t v120_lead = threadIdx.x % 32;
          if (v120_lead < 16) {
            #pragma unroll
            for (int32_t v122_i1 = 0; v122_i1 < 9; ++v122_i1) {
              int32_t v123_a = 0 + v122_i1;
              float v125_data = r3[v122_i1];
              int32_t v132_a = v120_lead + (v122_i1 * 32);
              s0[v132_a] = v125_data;
            }
          }
          float r6[9]{};
          {
            // r6 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r6[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r6[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r6[2] = v64;
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          auto& ir5 = r5;
          if ((threadIdx.x % 32) < 16) {
            float v137_data = r4[0];
            float v138_data = ir5[0];
            ir5[0] = (v138_data + v137_data);
            float v140_data = r4[1];
            float v141_data = ir5[1];
            ir5[1] = (v141_data + v140_data);
            float v143_data = r4[2];
            float v144_data = ir5[2];
            ir5[2] = (v144_data + v143_data);
            float v146_data = r4[3];
            float v147_data = ir5[3];
            ir5[3] = (v147_data + v146_data);
            float v149_data = r4[4];
            float v150_data = ir5[4];
            ir5[4] = (v150_data + v149_data);
            float v152_data = r4[5];
            float v153_data = ir5[5];
            ir5[5] = (v153_data + v152_data);
            float v155_data = r4[6];
            float v156_data = ir5[6];
            ir5[6] = (v156_data + v155_data);
            float v158_data = r4[7];
            float v159_data = ir5[7];
            ir5[7] = (v159_data + v158_data);
            float v161_data = r4[8];
            float v162_data = ir5[8];
            ir5[8] = (v162_data + v161_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          int32_t v166_lead = threadIdx.x % 32;
          if (v166_lead < 16) {
            #pragma unroll
            for (int32_t v168_i1 = 0; v168_i1 < 9; ++v168_i1) {
              int32_t v169_a = 0 + v168_i1;
              float v171_data = r5[v168_i1];
              int32_t v178_a = v166_lead + (v168_i1 * 32);
              s0[v178_a] = v171_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v181_lane = threadIdx.x % 32;
          int32_t v184_a = v181_lane + 0;
          float v191_data = s0[v181_lane];
          int32_t v197_a = v181_lane + 32;
          float v204_data = s0[(v181_lane + 32)];
          int32_t v210_a = v181_lane + 64;
          float v217_data = s0[(v181_lane + 64)];
          int32_t v223_a = v181_lane + 96;
          float v230_data = s0[(v181_lane + 96)];
          int32_t v236_a = v181_lane + 128;
          float v243_data = s0[(v181_lane + 128)];
          int32_t v249_a = v181_lane + 160;
          float v256_data = s0[(v181_lane + 160)];
          int32_t v262_a = v181_lane + 192;
          float v269_data = s0[(v181_lane + 192)];
          int32_t v275_a = v181_lane + 224;
          float v282_data = s0[(v181_lane + 224)];
          int32_t v288_a = v181_lane + 256;
          float v295_data = s0[(v181_lane + 256)];
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
          tensorforge::fmacdpp16<0>(v296_acc, v306_bc, v191_data);
          tensorforge::fmacdpp16<1>(v296_acc, v306_bc, v204_data);
          tensorforge::fmacdpp16<2>(v296_acc, v306_bc, v217_data);
          tensorforge::fmacdpp16<3>(v296_acc, v306_bc, v230_data);
          tensorforge::fmacdpp16<4>(v296_acc, v306_bc, v243_data);
          tensorforge::fmacdpp16<5>(v296_acc, v306_bc, v256_data);
          tensorforge::fmacdpp16<6>(v296_acc, v306_bc, v269_data);
          tensorforge::fmacdpp16<7>(v296_acc, v306_bc, v282_data);
          tensorforge::fmacdpp16<8>(v296_acc, v306_bc, v295_data);
          tensorforge::fmacdpp16<9>(v297_acc, v306_bc, v191_data);
          tensorforge::fmacdpp16<10>(v297_acc, v306_bc, v204_data);
          tensorforge::fmacdpp16<11>(v297_acc, v306_bc, v217_data);
          tensorforge::fmacdpp16<12>(v297_acc, v306_bc, v230_data);
          tensorforge::fmacdpp16<13>(v297_acc, v306_bc, v243_data);
          tensorforge::fmacdpp16<14>(v297_acc, v306_bc, v256_data);
          tensorforge::fmacdpp16<15>(v297_acc, v306_bc, v269_data);
          float v307_bc = tensorforge::broadcast<32, 16, 1>(v305_lin);
          tensorforge::fmacdpp16<0>(v297_acc, v307_bc, v282_data);
          tensorforge::fmacdpp16<1>(v297_acc, v307_bc, v295_data);
          tensorforge::fmacdpp16<2>(v298_acc, v307_bc, v191_data);
          tensorforge::fmacdpp16<3>(v298_acc, v307_bc, v204_data);
          tensorforge::fmacdpp16<4>(v298_acc, v307_bc, v217_data);
          tensorforge::fmacdpp16<5>(v298_acc, v307_bc, v230_data);
          tensorforge::fmacdpp16<6>(v298_acc, v307_bc, v243_data);
          tensorforge::fmacdpp16<7>(v298_acc, v307_bc, v256_data);
          tensorforge::fmacdpp16<8>(v298_acc, v307_bc, v269_data);
          tensorforge::fmacdpp16<9>(v298_acc, v307_bc, v282_data);
          tensorforge::fmacdpp16<10>(v298_acc, v307_bc, v295_data);
          tensorforge::fmacdpp16<11>(v299_acc, v307_bc, v191_data);
          tensorforge::fmacdpp16<12>(v299_acc, v307_bc, v204_data);
          tensorforge::fmacdpp16<13>(v299_acc, v307_bc, v217_data);
          tensorforge::fmacdpp16<14>(v299_acc, v307_bc, v230_data);
          tensorforge::fmacdpp16<15>(v299_acc, v307_bc, v243_data);
          float v308_lin = r6[1];
          float v309_bc = tensorforge::broadcast<32, 16, 0>(v308_lin);
          tensorforge::fmacdpp16<0>(v299_acc, v309_bc, v256_data);
          tensorforge::fmacdpp16<1>(v299_acc, v309_bc, v269_data);
          tensorforge::fmacdpp16<2>(v299_acc, v309_bc, v282_data);
          tensorforge::fmacdpp16<3>(v299_acc, v309_bc, v295_data);
          tensorforge::fmacdpp16<4>(v300_acc, v309_bc, v191_data);
          tensorforge::fmacdpp16<5>(v300_acc, v309_bc, v204_data);
          tensorforge::fmacdpp16<6>(v300_acc, v309_bc, v217_data);
          tensorforge::fmacdpp16<7>(v300_acc, v309_bc, v230_data);
          tensorforge::fmacdpp16<8>(v300_acc, v309_bc, v243_data);
          tensorforge::fmacdpp16<9>(v300_acc, v309_bc, v256_data);
          tensorforge::fmacdpp16<10>(v300_acc, v309_bc, v269_data);
          tensorforge::fmacdpp16<11>(v300_acc, v309_bc, v282_data);
          tensorforge::fmacdpp16<12>(v300_acc, v309_bc, v295_data);
          tensorforge::fmacdpp16<13>(v301_acc, v309_bc, v191_data);
          tensorforge::fmacdpp16<14>(v301_acc, v309_bc, v204_data);
          tensorforge::fmacdpp16<15>(v301_acc, v309_bc, v217_data);
          float v310_bc = tensorforge::broadcast<32, 16, 1>(v308_lin);
          tensorforge::fmacdpp16<0>(v301_acc, v310_bc, v230_data);
          tensorforge::fmacdpp16<1>(v301_acc, v310_bc, v243_data);
          tensorforge::fmacdpp16<2>(v301_acc, v310_bc, v256_data);
          tensorforge::fmacdpp16<3>(v301_acc, v310_bc, v269_data);
          tensorforge::fmacdpp16<4>(v301_acc, v310_bc, v282_data);
          tensorforge::fmacdpp16<5>(v301_acc, v310_bc, v295_data);
          tensorforge::fmacdpp16<6>(v302_acc, v310_bc, v191_data);
          tensorforge::fmacdpp16<7>(v302_acc, v310_bc, v204_data);
          tensorforge::fmacdpp16<8>(v302_acc, v310_bc, v217_data);
          tensorforge::fmacdpp16<9>(v302_acc, v310_bc, v230_data);
          tensorforge::fmacdpp16<10>(v302_acc, v310_bc, v243_data);
          tensorforge::fmacdpp16<11>(v302_acc, v310_bc, v256_data);
          tensorforge::fmacdpp16<12>(v302_acc, v310_bc, v269_data);
          tensorforge::fmacdpp16<13>(v302_acc, v310_bc, v282_data);
          tensorforge::fmacdpp16<14>(v302_acc, v310_bc, v295_data);
          tensorforge::fmacdpp16<15>(v303_acc, v310_bc, v191_data);
          float v311_lin = r6[2];
          float v312_bc = tensorforge::broadcast<32, 16, 0>(v311_lin);
          tensorforge::fmacdpp16<0>(v303_acc, v312_bc, v204_data);
          tensorforge::fmacdpp16<1>(v303_acc, v312_bc, v217_data);
          tensorforge::fmacdpp16<2>(v303_acc, v312_bc, v230_data);
          tensorforge::fmacdpp16<3>(v303_acc, v312_bc, v243_data);
          tensorforge::fmacdpp16<4>(v303_acc, v312_bc, v256_data);
          tensorforge::fmacdpp16<5>(v303_acc, v312_bc, v269_data);
          tensorforge::fmacdpp16<6>(v303_acc, v312_bc, v282_data);
          tensorforge::fmacdpp16<7>(v303_acc, v312_bc, v295_data);
          tensorforge::fmacdpp16<8>(v304_acc, v312_bc, v191_data);
          tensorforge::fmacdpp16<9>(v304_acc, v312_bc, v204_data);
          tensorforge::fmacdpp16<10>(v304_acc, v312_bc, v217_data);
          tensorforge::fmacdpp16<11>(v304_acc, v312_bc, v230_data);
          tensorforge::fmacdpp16<12>(v304_acc, v312_bc, v243_data);
          tensorforge::fmacdpp16<13>(v304_acc, v312_bc, v256_data);
          tensorforge::fmacdpp16<14>(v304_acc, v312_bc, v269_data);
          tensorforge::fmacdpp16<15>(v304_acc, v312_bc, v282_data);
          tensorforge::fmacdpp16<0>(v304_acc, (tensorforge::broadcast<32, 16, 1>(v311_lin)), v295_data);
          ir7[0] = v296_acc;
          ir7[1] = v297_acc;
          ir7[2] = v298_acc;
          ir7[3] = v299_acc;
          ir7[4] = v300_acc;
          ir7[5] = v301_acc;
          ir7[6] = v302_acc;
          ir7[7] = v303_acc;
          ir7[8] = v304_acc;
          // glb_m3 = store{r>g}(r7);
          int32_t v316_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v317_i0 = 0; v317_i0 < 1; ++v317_i0) {
            int32_t v326_lead = v316_lead + (v317_i0 * 32);
            #pragma unroll
            for (int32_t v318_i1 = 0; v318_i1 < 9; ++v318_i1) {
              int32_t v319_a = v317_i0 + v318_i1;
              float v321_data = r7[(v317_i0 + v318_i1)];
              int32_t v328_a = v326_lead + (v318_i1 * 32);
              glb_m3[v328_a] = v321_data;
            }
          }
          ;
        }
      }
    }
  }
}

