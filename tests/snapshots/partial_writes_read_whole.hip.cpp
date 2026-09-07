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
              s0[(v85_lead + (v78_i1 * 32))] = v80_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 9; ++v93_i1) {
              float v101_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v93_i1 * 16))]);
              r4[v93_i1] = v101_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          if (v16_lead < 16) {
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
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v139_i1 = 0; v139_i1 < 9; ++v139_i1) {
              float v141_data = r3[v139_i1];
              s0[(v16_lead + (v139_i1 * 32))] = v141_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v150_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v150_lin;
          float v151_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v151_lin;
          float v152_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v152_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          if (v16_lead < 16) {
            float v158_data = r4[0];
            float v159_data = r5[0];
            r5[0] = (v159_data + v158_data);
            float v161_data = r4[1];
            float v162_data = r5[1];
            r5[1] = (v162_data + v161_data);
            float v164_data = r4[2];
            float v165_data = r5[2];
            r5[2] = (v165_data + v164_data);
            float v167_data = r4[3];
            float v168_data = r5[3];
            r5[3] = (v168_data + v167_data);
            float v170_data = r4[4];
            float v171_data = r5[4];
            r5[4] = (v171_data + v170_data);
            float v173_data = r4[5];
            float v174_data = r5[5];
            r5[5] = (v174_data + v173_data);
            float v176_data = r4[6];
            float v177_data = r5[6];
            r5[6] = (v177_data + v176_data);
            float v179_data = r4[7];
            float v180_data = r5[7];
            r5[7] = (v180_data + v179_data);
            float v182_data = r4[8];
            float v183_data = r5[8];
            r5[8] = (v183_data + v182_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v189_i1 = 0; v189_i1 < 9; ++v189_i1) {
              float v191_data = r5[v189_i1];
              s0[(v16_lead + (v189_i1 * 32))] = v191_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v206_data = s0[v16_lead];
          float v213_data = s0[(v16_lead + 32)];
          float v220_data = s0[(v16_lead + 64)];
          float v227_data = s0[(v16_lead + 96)];
          float v234_data = s0[(v16_lead + 128)];
          float v241_data = s0[(v16_lead + 160)];
          float v248_data = s0[(v16_lead + 192)];
          float v255_data = s0[(v16_lead + 224)];
          float v262_data = s0[(v16_lead + 256)];
          float v263_acc{};
          float v264_acc{};
          float v265_acc{};
          float v266_acc{};
          float v267_acc{};
          float v268_acc{};
          float v269_acc{};
          float v270_acc{};
          float v271_acc{};
          float v272_lin = r6[0];
          float v273_bc = tensorforge::broadcast<32, 16, 0>(v272_lin);
          tensorforge::fmacdpp16<0>(v263_acc, v273_bc, v206_data);
          tensorforge::fmacdpp16<1>(v263_acc, v273_bc, v213_data);
          tensorforge::fmacdpp16<2>(v263_acc, v273_bc, v220_data);
          tensorforge::fmacdpp16<3>(v263_acc, v273_bc, v227_data);
          tensorforge::fmacdpp16<4>(v263_acc, v273_bc, v234_data);
          tensorforge::fmacdpp16<5>(v263_acc, v273_bc, v241_data);
          tensorforge::fmacdpp16<6>(v263_acc, v273_bc, v248_data);
          tensorforge::fmacdpp16<7>(v263_acc, v273_bc, v255_data);
          tensorforge::fmacdpp16<8>(v263_acc, v273_bc, v262_data);
          tensorforge::fmacdpp16<9>(v264_acc, v273_bc, v206_data);
          tensorforge::fmacdpp16<10>(v264_acc, v273_bc, v213_data);
          tensorforge::fmacdpp16<11>(v264_acc, v273_bc, v220_data);
          tensorforge::fmacdpp16<12>(v264_acc, v273_bc, v227_data);
          tensorforge::fmacdpp16<13>(v264_acc, v273_bc, v234_data);
          tensorforge::fmacdpp16<14>(v264_acc, v273_bc, v241_data);
          tensorforge::fmacdpp16<15>(v264_acc, v273_bc, v248_data);
          float v274_bc = tensorforge::broadcast<32, 16, 1>(v272_lin);
          tensorforge::fmacdpp16<0>(v264_acc, v274_bc, v255_data);
          tensorforge::fmacdpp16<1>(v264_acc, v274_bc, v262_data);
          tensorforge::fmacdpp16<2>(v265_acc, v274_bc, v206_data);
          tensorforge::fmacdpp16<3>(v265_acc, v274_bc, v213_data);
          tensorforge::fmacdpp16<4>(v265_acc, v274_bc, v220_data);
          tensorforge::fmacdpp16<5>(v265_acc, v274_bc, v227_data);
          tensorforge::fmacdpp16<6>(v265_acc, v274_bc, v234_data);
          tensorforge::fmacdpp16<7>(v265_acc, v274_bc, v241_data);
          tensorforge::fmacdpp16<8>(v265_acc, v274_bc, v248_data);
          tensorforge::fmacdpp16<9>(v265_acc, v274_bc, v255_data);
          tensorforge::fmacdpp16<10>(v265_acc, v274_bc, v262_data);
          tensorforge::fmacdpp16<11>(v266_acc, v274_bc, v206_data);
          tensorforge::fmacdpp16<12>(v266_acc, v274_bc, v213_data);
          tensorforge::fmacdpp16<13>(v266_acc, v274_bc, v220_data);
          tensorforge::fmacdpp16<14>(v266_acc, v274_bc, v227_data);
          tensorforge::fmacdpp16<15>(v266_acc, v274_bc, v234_data);
          float v275_lin = r6[1];
          float v276_bc = tensorforge::broadcast<32, 16, 0>(v275_lin);
          tensorforge::fmacdpp16<0>(v266_acc, v276_bc, v241_data);
          tensorforge::fmacdpp16<1>(v266_acc, v276_bc, v248_data);
          tensorforge::fmacdpp16<2>(v266_acc, v276_bc, v255_data);
          tensorforge::fmacdpp16<3>(v266_acc, v276_bc, v262_data);
          tensorforge::fmacdpp16<4>(v267_acc, v276_bc, v206_data);
          tensorforge::fmacdpp16<5>(v267_acc, v276_bc, v213_data);
          tensorforge::fmacdpp16<6>(v267_acc, v276_bc, v220_data);
          tensorforge::fmacdpp16<7>(v267_acc, v276_bc, v227_data);
          tensorforge::fmacdpp16<8>(v267_acc, v276_bc, v234_data);
          tensorforge::fmacdpp16<9>(v267_acc, v276_bc, v241_data);
          tensorforge::fmacdpp16<10>(v267_acc, v276_bc, v248_data);
          tensorforge::fmacdpp16<11>(v267_acc, v276_bc, v255_data);
          tensorforge::fmacdpp16<12>(v267_acc, v276_bc, v262_data);
          tensorforge::fmacdpp16<13>(v268_acc, v276_bc, v206_data);
          tensorforge::fmacdpp16<14>(v268_acc, v276_bc, v213_data);
          tensorforge::fmacdpp16<15>(v268_acc, v276_bc, v220_data);
          float v277_bc = tensorforge::broadcast<32, 16, 1>(v275_lin);
          tensorforge::fmacdpp16<0>(v268_acc, v277_bc, v227_data);
          tensorforge::fmacdpp16<1>(v268_acc, v277_bc, v234_data);
          tensorforge::fmacdpp16<2>(v268_acc, v277_bc, v241_data);
          tensorforge::fmacdpp16<3>(v268_acc, v277_bc, v248_data);
          tensorforge::fmacdpp16<4>(v268_acc, v277_bc, v255_data);
          tensorforge::fmacdpp16<5>(v268_acc, v277_bc, v262_data);
          tensorforge::fmacdpp16<6>(v269_acc, v277_bc, v206_data);
          tensorforge::fmacdpp16<7>(v269_acc, v277_bc, v213_data);
          tensorforge::fmacdpp16<8>(v269_acc, v277_bc, v220_data);
          tensorforge::fmacdpp16<9>(v269_acc, v277_bc, v227_data);
          tensorforge::fmacdpp16<10>(v269_acc, v277_bc, v234_data);
          tensorforge::fmacdpp16<11>(v269_acc, v277_bc, v241_data);
          tensorforge::fmacdpp16<12>(v269_acc, v277_bc, v248_data);
          tensorforge::fmacdpp16<13>(v269_acc, v277_bc, v255_data);
          tensorforge::fmacdpp16<14>(v269_acc, v277_bc, v262_data);
          tensorforge::fmacdpp16<15>(v270_acc, v277_bc, v206_data);
          float v278_lin = r6[2];
          float v279_bc = tensorforge::broadcast<32, 16, 0>(v278_lin);
          tensorforge::fmacdpp16<0>(v270_acc, v279_bc, v213_data);
          tensorforge::fmacdpp16<1>(v270_acc, v279_bc, v220_data);
          tensorforge::fmacdpp16<2>(v270_acc, v279_bc, v227_data);
          tensorforge::fmacdpp16<3>(v270_acc, v279_bc, v234_data);
          tensorforge::fmacdpp16<4>(v270_acc, v279_bc, v241_data);
          tensorforge::fmacdpp16<5>(v270_acc, v279_bc, v248_data);
          tensorforge::fmacdpp16<6>(v270_acc, v279_bc, v255_data);
          tensorforge::fmacdpp16<7>(v270_acc, v279_bc, v262_data);
          tensorforge::fmacdpp16<8>(v271_acc, v279_bc, v206_data);
          tensorforge::fmacdpp16<9>(v271_acc, v279_bc, v213_data);
          tensorforge::fmacdpp16<10>(v271_acc, v279_bc, v220_data);
          tensorforge::fmacdpp16<11>(v271_acc, v279_bc, v227_data);
          tensorforge::fmacdpp16<12>(v271_acc, v279_bc, v234_data);
          tensorforge::fmacdpp16<13>(v271_acc, v279_bc, v241_data);
          tensorforge::fmacdpp16<14>(v271_acc, v279_bc, v248_data);
          tensorforge::fmacdpp16<15>(v271_acc, v279_bc, v255_data);
          tensorforge::fmacdpp16<0>(v271_acc, (tensorforge::broadcast<32, 16, 1>(v278_lin)), v262_data);
          r7[0] = v263_acc;
          r7[1] = v264_acc;
          r7[2] = v265_acc;
          r7[3] = v266_acc;
          r7[4] = v267_acc;
          r7[5] = v268_acc;
          r7[6] = v269_acc;
          r7[7] = v270_acc;
          r7[8] = v271_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v284_i0 = 0; v284_i0 < 1; ++v284_i0) {
            int32_t v292_lead = v16_lead + (v284_i0 * 32);
            #pragma unroll
            for (int32_t v285_i1 = 0; v285_i1 < 9; ++v285_i1) {
              float v287_data = r7[(v284_i0 + v285_i1)];
              glb_m3[(v292_lead + (v285_i1 * 32))] = v287_data;
            }
          }
        }
      }
    }
  }
}

