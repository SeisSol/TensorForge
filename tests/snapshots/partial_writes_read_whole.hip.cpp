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
              s0[(v82_lead + (v75_i1 * 32))] = v77_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v90_i1 = 0; v90_i1 < 9; ++v90_i1) {
              float v98_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v90_i1 * 16))]);
              r4[v90_i1] = v98_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v12_lead < 16) {
            float v105_data = r2[0];
            float v106_data = r3[0];
            r3[0] = (v106_data + v105_data);
            float v108_data = r2[1];
            float v109_data = r3[1];
            r3[1] = (v109_data + v108_data);
            float v111_data = r2[2];
            float v112_data = r3[2];
            r3[2] = (v112_data + v111_data);
            float v114_data = r2[3];
            float v115_data = r3[3];
            r3[3] = (v115_data + v114_data);
            float v117_data = r2[4];
            float v118_data = r3[4];
            r3[4] = (v118_data + v117_data);
            float v120_data = r2[5];
            float v121_data = r3[5];
            r3[5] = (v121_data + v120_data);
            float v123_data = r2[6];
            float v124_data = r3[6];
            r3[6] = (v124_data + v123_data);
            float v126_data = r2[7];
            float v127_data = r3[7];
            r3[7] = (v127_data + v126_data);
            float v129_data = r2[8];
            float v130_data = r3[8];
            r3[8] = (v130_data + v129_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v136_i1 = 0; v136_i1 < 9; ++v136_i1) {
              float v138_data = r3[v136_i1];
              s0[(v12_lead + (v136_i1 * 32))] = v138_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v147_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v147_lin;
          float v148_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v148_lin;
          float v149_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v149_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v12_lead < 16) {
            float v155_data = r4[0];
            float v156_data = r5[0];
            r5[0] = (v156_data + v155_data);
            float v158_data = r4[1];
            float v159_data = r5[1];
            r5[1] = (v159_data + v158_data);
            float v161_data = r4[2];
            float v162_data = r5[2];
            r5[2] = (v162_data + v161_data);
            float v164_data = r4[3];
            float v165_data = r5[3];
            r5[3] = (v165_data + v164_data);
            float v167_data = r4[4];
            float v168_data = r5[4];
            r5[4] = (v168_data + v167_data);
            float v170_data = r4[5];
            float v171_data = r5[5];
            r5[5] = (v171_data + v170_data);
            float v173_data = r4[6];
            float v174_data = r5[6];
            r5[6] = (v174_data + v173_data);
            float v176_data = r4[7];
            float v177_data = r5[7];
            r5[7] = (v177_data + v176_data);
            float v179_data = r4[8];
            float v180_data = r5[8];
            r5[8] = (v180_data + v179_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v186_i1 = 0; v186_i1 < 9; ++v186_i1) {
              float v188_data = r5[v186_i1];
              s0[(v12_lead + (v186_i1 * 32))] = v188_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v203_data = s0[v12_lead];
          float v210_data = s0[(v12_lead + 32)];
          float v217_data = s0[(v12_lead + 64)];
          float v224_data = s0[(v12_lead + 96)];
          float v231_data = s0[(v12_lead + 128)];
          float v238_data = s0[(v12_lead + 160)];
          float v245_data = s0[(v12_lead + 192)];
          float v252_data = s0[(v12_lead + 224)];
          float v259_data = s0[(v12_lead + 256)];
          float v260_acc{};
          float v261_acc{};
          float v262_acc{};
          float v263_acc{};
          float v264_acc{};
          float v265_acc{};
          float v266_acc{};
          float v267_acc{};
          float v268_acc{};
          float v269_lin = r6[0];
          float v270_bc = tensorforge::broadcast<32, 16, 0>(v269_lin);
          tensorforge::fmacdpp16<0>(v260_acc, v270_bc, v203_data);
          tensorforge::fmacdpp16<1>(v260_acc, v270_bc, v210_data);
          tensorforge::fmacdpp16<2>(v260_acc, v270_bc, v217_data);
          tensorforge::fmacdpp16<3>(v260_acc, v270_bc, v224_data);
          tensorforge::fmacdpp16<4>(v260_acc, v270_bc, v231_data);
          tensorforge::fmacdpp16<5>(v260_acc, v270_bc, v238_data);
          tensorforge::fmacdpp16<6>(v260_acc, v270_bc, v245_data);
          tensorforge::fmacdpp16<7>(v260_acc, v270_bc, v252_data);
          tensorforge::fmacdpp16<8>(v260_acc, v270_bc, v259_data);
          tensorforge::fmacdpp16<9>(v261_acc, v270_bc, v203_data);
          tensorforge::fmacdpp16<10>(v261_acc, v270_bc, v210_data);
          tensorforge::fmacdpp16<11>(v261_acc, v270_bc, v217_data);
          tensorforge::fmacdpp16<12>(v261_acc, v270_bc, v224_data);
          tensorforge::fmacdpp16<13>(v261_acc, v270_bc, v231_data);
          tensorforge::fmacdpp16<14>(v261_acc, v270_bc, v238_data);
          tensorforge::fmacdpp16<15>(v261_acc, v270_bc, v245_data);
          float v271_bc = tensorforge::broadcast<32, 16, 1>(v269_lin);
          tensorforge::fmacdpp16<0>(v261_acc, v271_bc, v252_data);
          tensorforge::fmacdpp16<1>(v261_acc, v271_bc, v259_data);
          tensorforge::fmacdpp16<2>(v262_acc, v271_bc, v203_data);
          tensorforge::fmacdpp16<3>(v262_acc, v271_bc, v210_data);
          tensorforge::fmacdpp16<4>(v262_acc, v271_bc, v217_data);
          tensorforge::fmacdpp16<5>(v262_acc, v271_bc, v224_data);
          tensorforge::fmacdpp16<6>(v262_acc, v271_bc, v231_data);
          tensorforge::fmacdpp16<7>(v262_acc, v271_bc, v238_data);
          tensorforge::fmacdpp16<8>(v262_acc, v271_bc, v245_data);
          tensorforge::fmacdpp16<9>(v262_acc, v271_bc, v252_data);
          tensorforge::fmacdpp16<10>(v262_acc, v271_bc, v259_data);
          tensorforge::fmacdpp16<11>(v263_acc, v271_bc, v203_data);
          tensorforge::fmacdpp16<12>(v263_acc, v271_bc, v210_data);
          tensorforge::fmacdpp16<13>(v263_acc, v271_bc, v217_data);
          tensorforge::fmacdpp16<14>(v263_acc, v271_bc, v224_data);
          tensorforge::fmacdpp16<15>(v263_acc, v271_bc, v231_data);
          float v272_lin = r6[1];
          float v273_bc = tensorforge::broadcast<32, 16, 0>(v272_lin);
          tensorforge::fmacdpp16<0>(v263_acc, v273_bc, v238_data);
          tensorforge::fmacdpp16<1>(v263_acc, v273_bc, v245_data);
          tensorforge::fmacdpp16<2>(v263_acc, v273_bc, v252_data);
          tensorforge::fmacdpp16<3>(v263_acc, v273_bc, v259_data);
          tensorforge::fmacdpp16<4>(v264_acc, v273_bc, v203_data);
          tensorforge::fmacdpp16<5>(v264_acc, v273_bc, v210_data);
          tensorforge::fmacdpp16<6>(v264_acc, v273_bc, v217_data);
          tensorforge::fmacdpp16<7>(v264_acc, v273_bc, v224_data);
          tensorforge::fmacdpp16<8>(v264_acc, v273_bc, v231_data);
          tensorforge::fmacdpp16<9>(v264_acc, v273_bc, v238_data);
          tensorforge::fmacdpp16<10>(v264_acc, v273_bc, v245_data);
          tensorforge::fmacdpp16<11>(v264_acc, v273_bc, v252_data);
          tensorforge::fmacdpp16<12>(v264_acc, v273_bc, v259_data);
          tensorforge::fmacdpp16<13>(v265_acc, v273_bc, v203_data);
          tensorforge::fmacdpp16<14>(v265_acc, v273_bc, v210_data);
          tensorforge::fmacdpp16<15>(v265_acc, v273_bc, v217_data);
          float v274_bc = tensorforge::broadcast<32, 16, 1>(v272_lin);
          tensorforge::fmacdpp16<0>(v265_acc, v274_bc, v224_data);
          tensorforge::fmacdpp16<1>(v265_acc, v274_bc, v231_data);
          tensorforge::fmacdpp16<2>(v265_acc, v274_bc, v238_data);
          tensorforge::fmacdpp16<3>(v265_acc, v274_bc, v245_data);
          tensorforge::fmacdpp16<4>(v265_acc, v274_bc, v252_data);
          tensorforge::fmacdpp16<5>(v265_acc, v274_bc, v259_data);
          tensorforge::fmacdpp16<6>(v266_acc, v274_bc, v203_data);
          tensorforge::fmacdpp16<7>(v266_acc, v274_bc, v210_data);
          tensorforge::fmacdpp16<8>(v266_acc, v274_bc, v217_data);
          tensorforge::fmacdpp16<9>(v266_acc, v274_bc, v224_data);
          tensorforge::fmacdpp16<10>(v266_acc, v274_bc, v231_data);
          tensorforge::fmacdpp16<11>(v266_acc, v274_bc, v238_data);
          tensorforge::fmacdpp16<12>(v266_acc, v274_bc, v245_data);
          tensorforge::fmacdpp16<13>(v266_acc, v274_bc, v252_data);
          tensorforge::fmacdpp16<14>(v266_acc, v274_bc, v259_data);
          tensorforge::fmacdpp16<15>(v267_acc, v274_bc, v203_data);
          float v275_lin = r6[2];
          float v276_bc = tensorforge::broadcast<32, 16, 0>(v275_lin);
          tensorforge::fmacdpp16<0>(v267_acc, v276_bc, v210_data);
          tensorforge::fmacdpp16<1>(v267_acc, v276_bc, v217_data);
          tensorforge::fmacdpp16<2>(v267_acc, v276_bc, v224_data);
          tensorforge::fmacdpp16<3>(v267_acc, v276_bc, v231_data);
          tensorforge::fmacdpp16<4>(v267_acc, v276_bc, v238_data);
          tensorforge::fmacdpp16<5>(v267_acc, v276_bc, v245_data);
          tensorforge::fmacdpp16<6>(v267_acc, v276_bc, v252_data);
          tensorforge::fmacdpp16<7>(v267_acc, v276_bc, v259_data);
          tensorforge::fmacdpp16<8>(v268_acc, v276_bc, v203_data);
          tensorforge::fmacdpp16<9>(v268_acc, v276_bc, v210_data);
          tensorforge::fmacdpp16<10>(v268_acc, v276_bc, v217_data);
          tensorforge::fmacdpp16<11>(v268_acc, v276_bc, v224_data);
          tensorforge::fmacdpp16<12>(v268_acc, v276_bc, v231_data);
          tensorforge::fmacdpp16<13>(v268_acc, v276_bc, v238_data);
          tensorforge::fmacdpp16<14>(v268_acc, v276_bc, v245_data);
          tensorforge::fmacdpp16<15>(v268_acc, v276_bc, v252_data);
          tensorforge::fmacdpp16<0>(v268_acc, (tensorforge::broadcast<32, 16, 1>(v275_lin)), v259_data);
          r7[0] = v260_acc;
          r7[1] = v261_acc;
          r7[2] = v262_acc;
          r7[3] = v263_acc;
          r7[4] = v264_acc;
          r7[5] = v265_acc;
          r7[6] = v266_acc;
          r7[7] = v267_acc;
          r7[8] = v268_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v281_i0 = 0; v281_i0 < 1; ++v281_i0) {
            int32_t v289_lead = v12_lead + (v281_i0 * 32);
            #pragma unroll
            for (int32_t v282_i1 = 0; v282_i1 < 9; ++v282_i1) {
              float v284_data = r7[(v281_i0 + v282_i1)];
              glb_m3[(v289_lead + (v282_i1 * 32))] = v284_data;
            }
          }
        }
      }
    }
  }
}

