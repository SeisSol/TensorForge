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
            int32_t v69_lead = v60_lead + (v61_i0 * 32);
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
              int32_t v63_a = v61_i0 + v62_i1;
              float v64_data = r1[v63_a];
              int32_t v71_a = v69_lead + (v62_i1 * 32);
              s0[v71_a] = v64_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          int32_t v74_lead = threadIdx.x % 32;
          if (v74_lead < 16) {
            #pragma unroll
            for (int32_t v76_i1 = 0; v76_i1 < 9; ++v76_i1) {
              int32_t v83_a = v74_lead + (v76_i1 * 16);
              float v84_data;
              {
                v84_data = __builtin_nontemporal_load(&glb_m2[v83_a]);
              }
              int32_t v85_a = 0 + v76_i1;
              r4[v85_a] = v84_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if ((threadIdx.x % 32) < 16) {
            float v90_data = r2[0];
            float v91_data = ir3[0];
            ir3[0] = (v91_data + v90_data);
            float v93_data = r2[1];
            float v94_data = ir3[1];
            ir3[1] = (v94_data + v93_data);
            float v96_data = r2[2];
            float v97_data = ir3[2];
            ir3[2] = (v97_data + v96_data);
            float v99_data = r2[3];
            float v100_data = ir3[3];
            ir3[3] = (v100_data + v99_data);
            float v102_data = r2[4];
            float v103_data = ir3[4];
            ir3[4] = (v103_data + v102_data);
            float v105_data = r2[5];
            float v106_data = ir3[5];
            ir3[5] = (v106_data + v105_data);
            float v108_data = r2[6];
            float v109_data = ir3[6];
            ir3[6] = (v109_data + v108_data);
            float v111_data = r2[7];
            float v112_data = ir3[7];
            ir3[7] = (v112_data + v111_data);
            float v114_data = r2[8];
            float v115_data = ir3[8];
            ir3[8] = (v115_data + v114_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          int32_t v119_lead = threadIdx.x % 32;
          if (v119_lead < 16) {
            #pragma unroll
            for (int32_t v121_i1 = 0; v121_i1 < 9; ++v121_i1) {
              int32_t v122_a = 0 + v121_i1;
              float v123_data = r3[v122_a];
              int32_t v130_a = v119_lead + (v121_i1 * 32);
              s0[v130_a] = v123_data;
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
            float v135_data = r4[0];
            float v136_data = ir5[0];
            ir5[0] = (v136_data + v135_data);
            float v138_data = r4[1];
            float v139_data = ir5[1];
            ir5[1] = (v139_data + v138_data);
            float v141_data = r4[2];
            float v142_data = ir5[2];
            ir5[2] = (v142_data + v141_data);
            float v144_data = r4[3];
            float v145_data = ir5[3];
            ir5[3] = (v145_data + v144_data);
            float v147_data = r4[4];
            float v148_data = ir5[4];
            ir5[4] = (v148_data + v147_data);
            float v150_data = r4[5];
            float v151_data = ir5[5];
            ir5[5] = (v151_data + v150_data);
            float v153_data = r4[6];
            float v154_data = ir5[6];
            ir5[6] = (v154_data + v153_data);
            float v156_data = r4[7];
            float v157_data = ir5[7];
            ir5[7] = (v157_data + v156_data);
            float v159_data = r4[8];
            float v160_data = ir5[8];
            ir5[8] = (v160_data + v159_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          int32_t v164_lead = threadIdx.x % 32;
          if (v164_lead < 16) {
            #pragma unroll
            for (int32_t v166_i1 = 0; v166_i1 < 9; ++v166_i1) {
              int32_t v167_a = 0 + v166_i1;
              float v168_data = r5[v167_a];
              int32_t v175_a = v164_lead + (v166_i1 * 32);
              s0[v175_a] = v168_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v178_lane = threadIdx.x % 32;
          int32_t v181_a = v178_lane + 0;
          float v182_data = s0[v181_a];
          int32_t v188_a = v178_lane + 32;
          float v189_data = s0[v188_a];
          int32_t v195_a = v178_lane + 64;
          float v196_data = s0[v195_a];
          int32_t v202_a = v178_lane + 96;
          float v203_data = s0[v202_a];
          int32_t v209_a = v178_lane + 128;
          float v210_data = s0[v209_a];
          int32_t v216_a = v178_lane + 160;
          float v217_data = s0[v216_a];
          int32_t v223_a = v178_lane + 192;
          float v224_data = s0[v223_a];
          int32_t v230_a = v178_lane + 224;
          float v231_data = s0[v230_a];
          int32_t v237_a = v178_lane + 256;
          float v238_data = s0[v237_a];
          float v239_acc{};
          float v240_acc{};
          float v241_acc{};
          float v242_acc{};
          float v243_acc{};
          float v244_acc{};
          float v245_acc{};
          float v246_acc{};
          float v247_acc{};
          float v248_lin = r6[0];
          float v249_bc = tensorforge::broadcast<32, 16, 0>(v248_lin);
          tensorforge::fmacdpp16<0>(v239_acc, v249_bc, v182_data);
          tensorforge::fmacdpp16<1>(v239_acc, v249_bc, v189_data);
          tensorforge::fmacdpp16<2>(v239_acc, v249_bc, v196_data);
          tensorforge::fmacdpp16<3>(v239_acc, v249_bc, v203_data);
          tensorforge::fmacdpp16<4>(v239_acc, v249_bc, v210_data);
          tensorforge::fmacdpp16<5>(v239_acc, v249_bc, v217_data);
          tensorforge::fmacdpp16<6>(v239_acc, v249_bc, v224_data);
          tensorforge::fmacdpp16<7>(v239_acc, v249_bc, v231_data);
          tensorforge::fmacdpp16<8>(v239_acc, v249_bc, v238_data);
          tensorforge::fmacdpp16<9>(v240_acc, v249_bc, v182_data);
          tensorforge::fmacdpp16<10>(v240_acc, v249_bc, v189_data);
          tensorforge::fmacdpp16<11>(v240_acc, v249_bc, v196_data);
          tensorforge::fmacdpp16<12>(v240_acc, v249_bc, v203_data);
          tensorforge::fmacdpp16<13>(v240_acc, v249_bc, v210_data);
          tensorforge::fmacdpp16<14>(v240_acc, v249_bc, v217_data);
          tensorforge::fmacdpp16<15>(v240_acc, v249_bc, v224_data);
          float v250_bc = tensorforge::broadcast<32, 16, 1>(v248_lin);
          tensorforge::fmacdpp16<0>(v240_acc, v250_bc, v231_data);
          tensorforge::fmacdpp16<1>(v240_acc, v250_bc, v238_data);
          tensorforge::fmacdpp16<2>(v241_acc, v250_bc, v182_data);
          tensorforge::fmacdpp16<3>(v241_acc, v250_bc, v189_data);
          tensorforge::fmacdpp16<4>(v241_acc, v250_bc, v196_data);
          tensorforge::fmacdpp16<5>(v241_acc, v250_bc, v203_data);
          tensorforge::fmacdpp16<6>(v241_acc, v250_bc, v210_data);
          tensorforge::fmacdpp16<7>(v241_acc, v250_bc, v217_data);
          tensorforge::fmacdpp16<8>(v241_acc, v250_bc, v224_data);
          tensorforge::fmacdpp16<9>(v241_acc, v250_bc, v231_data);
          tensorforge::fmacdpp16<10>(v241_acc, v250_bc, v238_data);
          tensorforge::fmacdpp16<11>(v242_acc, v250_bc, v182_data);
          tensorforge::fmacdpp16<12>(v242_acc, v250_bc, v189_data);
          tensorforge::fmacdpp16<13>(v242_acc, v250_bc, v196_data);
          tensorforge::fmacdpp16<14>(v242_acc, v250_bc, v203_data);
          tensorforge::fmacdpp16<15>(v242_acc, v250_bc, v210_data);
          float v251_lin = r6[1];
          float v252_bc = tensorforge::broadcast<32, 16, 0>(v251_lin);
          tensorforge::fmacdpp16<0>(v242_acc, v252_bc, v217_data);
          tensorforge::fmacdpp16<1>(v242_acc, v252_bc, v224_data);
          tensorforge::fmacdpp16<2>(v242_acc, v252_bc, v231_data);
          tensorforge::fmacdpp16<3>(v242_acc, v252_bc, v238_data);
          tensorforge::fmacdpp16<4>(v243_acc, v252_bc, v182_data);
          tensorforge::fmacdpp16<5>(v243_acc, v252_bc, v189_data);
          tensorforge::fmacdpp16<6>(v243_acc, v252_bc, v196_data);
          tensorforge::fmacdpp16<7>(v243_acc, v252_bc, v203_data);
          tensorforge::fmacdpp16<8>(v243_acc, v252_bc, v210_data);
          tensorforge::fmacdpp16<9>(v243_acc, v252_bc, v217_data);
          tensorforge::fmacdpp16<10>(v243_acc, v252_bc, v224_data);
          tensorforge::fmacdpp16<11>(v243_acc, v252_bc, v231_data);
          tensorforge::fmacdpp16<12>(v243_acc, v252_bc, v238_data);
          tensorforge::fmacdpp16<13>(v244_acc, v252_bc, v182_data);
          tensorforge::fmacdpp16<14>(v244_acc, v252_bc, v189_data);
          tensorforge::fmacdpp16<15>(v244_acc, v252_bc, v196_data);
          float v253_bc = tensorforge::broadcast<32, 16, 1>(v251_lin);
          tensorforge::fmacdpp16<0>(v244_acc, v253_bc, v203_data);
          tensorforge::fmacdpp16<1>(v244_acc, v253_bc, v210_data);
          tensorforge::fmacdpp16<2>(v244_acc, v253_bc, v217_data);
          tensorforge::fmacdpp16<3>(v244_acc, v253_bc, v224_data);
          tensorforge::fmacdpp16<4>(v244_acc, v253_bc, v231_data);
          tensorforge::fmacdpp16<5>(v244_acc, v253_bc, v238_data);
          tensorforge::fmacdpp16<6>(v245_acc, v253_bc, v182_data);
          tensorforge::fmacdpp16<7>(v245_acc, v253_bc, v189_data);
          tensorforge::fmacdpp16<8>(v245_acc, v253_bc, v196_data);
          tensorforge::fmacdpp16<9>(v245_acc, v253_bc, v203_data);
          tensorforge::fmacdpp16<10>(v245_acc, v253_bc, v210_data);
          tensorforge::fmacdpp16<11>(v245_acc, v253_bc, v217_data);
          tensorforge::fmacdpp16<12>(v245_acc, v253_bc, v224_data);
          tensorforge::fmacdpp16<13>(v245_acc, v253_bc, v231_data);
          tensorforge::fmacdpp16<14>(v245_acc, v253_bc, v238_data);
          tensorforge::fmacdpp16<15>(v246_acc, v253_bc, v182_data);
          float v254_lin = r6[2];
          float v255_bc = tensorforge::broadcast<32, 16, 0>(v254_lin);
          tensorforge::fmacdpp16<0>(v246_acc, v255_bc, v189_data);
          tensorforge::fmacdpp16<1>(v246_acc, v255_bc, v196_data);
          tensorforge::fmacdpp16<2>(v246_acc, v255_bc, v203_data);
          tensorforge::fmacdpp16<3>(v246_acc, v255_bc, v210_data);
          tensorforge::fmacdpp16<4>(v246_acc, v255_bc, v217_data);
          tensorforge::fmacdpp16<5>(v246_acc, v255_bc, v224_data);
          tensorforge::fmacdpp16<6>(v246_acc, v255_bc, v231_data);
          tensorforge::fmacdpp16<7>(v246_acc, v255_bc, v238_data);
          tensorforge::fmacdpp16<8>(v247_acc, v255_bc, v182_data);
          tensorforge::fmacdpp16<9>(v247_acc, v255_bc, v189_data);
          tensorforge::fmacdpp16<10>(v247_acc, v255_bc, v196_data);
          tensorforge::fmacdpp16<11>(v247_acc, v255_bc, v203_data);
          tensorforge::fmacdpp16<12>(v247_acc, v255_bc, v210_data);
          tensorforge::fmacdpp16<13>(v247_acc, v255_bc, v217_data);
          tensorforge::fmacdpp16<14>(v247_acc, v255_bc, v224_data);
          tensorforge::fmacdpp16<15>(v247_acc, v255_bc, v231_data);
          tensorforge::fmacdpp16<0>(v247_acc, (tensorforge::broadcast<32, 16, 1>(v254_lin)), v238_data);
          ir7[0] = v239_acc;
          ir7[1] = v240_acc;
          ir7[2] = v241_acc;
          ir7[3] = v242_acc;
          ir7[4] = v243_acc;
          ir7[5] = v244_acc;
          ir7[6] = v245_acc;
          ir7[7] = v246_acc;
          ir7[8] = v247_acc;
          // glb_m3 = store{r>g}(r7);
          int32_t v259_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v260_i0 = 0; v260_i0 < 1; ++v260_i0) {
            int32_t v268_lead = v259_lead + (v260_i0 * 32);
            #pragma unroll
            for (int32_t v261_i1 = 0; v261_i1 < 9; ++v261_i1) {
              int32_t v262_a = v260_i0 + v261_i1;
              float v263_data = r7[v262_a];
              int32_t v270_a = v268_lead + (v261_i1 * 32);
              glb_m3[v270_a] = v263_data;
            }
          }
          ;
        }
      }
    }
  }
}

