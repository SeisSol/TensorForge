// === base name ===
kernel_e8722818e98812c1

// === header ===
void launcher_kernel_e8722818e98812c1(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e8722818e98812c1(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e8722818e98812c1, block.x * block.y * block.z, 384 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_e8722818e98812c1, cudaFuncAttributeMaxDynamicSharedMemorySize, 384 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_e8722818e98812c1<<<grid,block,384 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_e8722818e98812c1(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×4(8×4) {0..8}×{0..4} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 32;
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
              float v31_data = __ldcg(&glb_m0[(v21_lead + (v23_i1 * 8))]);
              r0[v23_i1] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(1);
          float r1[4]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          if (v21_lead < 8) {
            float v40_data = r0[0];
            float v41_data = s0[0];
            float v43_data = r1[0];
            r1[0] = (v43_data + (v40_data * v41_data));
            float v46_data = s0[8];
            float v48_data = r1[1];
            r1[1] = (v48_data + (v40_data * v46_data));
            float v51_data = s0[16];
            float v53_data = r1[2];
            r1[2] = (v53_data + (v40_data * v51_data));
            float v56_data = s0[24];
            float v58_data = r1[3];
            r1[3] = (v58_data + (v40_data * v56_data));
          }
          if (v21_lead < 8) {
            float v64_data = r0[1];
            float v65_data = s0[1];
            float v67_data = r1[0];
            r1[0] = (v67_data + (v64_data * v65_data));
            float v70_data = s0[9];
            float v72_data = r1[1];
            r1[1] = (v72_data + (v64_data * v70_data));
            float v75_data = s0[17];
            float v77_data = r1[2];
            r1[2] = (v77_data + (v64_data * v75_data));
            float v80_data = s0[25];
            float v82_data = r1[3];
            r1[3] = (v82_data + (v64_data * v80_data));
          }
          if (v21_lead < 8) {
            float v88_data = r0[2];
            float v89_data = s0[2];
            float v91_data = r1[0];
            r1[0] = (v91_data + (v88_data * v89_data));
            float v94_data = s0[10];
            float v96_data = r1[1];
            r1[1] = (v96_data + (v88_data * v94_data));
            float v99_data = s0[18];
            float v101_data = r1[2];
            r1[2] = (v101_data + (v88_data * v99_data));
            float v104_data = s0[26];
            float v106_data = r1[3];
            r1[3] = (v106_data + (v88_data * v104_data));
          }
          if (v21_lead < 8) {
            float v112_data = r0[3];
            float v113_data = s0[3];
            float v115_data = r1[0];
            r1[0] = (v115_data + (v112_data * v113_data));
            float v118_data = s0[11];
            float v120_data = r1[1];
            r1[1] = (v120_data + (v112_data * v118_data));
            float v123_data = s0[19];
            float v125_data = r1[2];
            r1[2] = (v125_data + (v112_data * v123_data));
            float v128_data = s0[27];
            float v130_data = r1[3];
            r1[3] = (v130_data + (v112_data * v128_data));
          }
          if (v21_lead < 8) {
            float v136_data = r0[4];
            float v137_data = s0[4];
            float v139_data = r1[0];
            r1[0] = (v139_data + (v136_data * v137_data));
            float v142_data = s0[12];
            float v144_data = r1[1];
            r1[1] = (v144_data + (v136_data * v142_data));
            float v147_data = s0[20];
            float v149_data = r1[2];
            r1[2] = (v149_data + (v136_data * v147_data));
            float v152_data = s0[28];
            float v154_data = r1[3];
            r1[3] = (v154_data + (v136_data * v152_data));
          }
          if (v21_lead < 8) {
            float v160_data = r0[5];
            float v161_data = s0[5];
            float v163_data = r1[0];
            r1[0] = (v163_data + (v160_data * v161_data));
            float v166_data = s0[13];
            float v168_data = r1[1];
            r1[1] = (v168_data + (v160_data * v166_data));
            float v171_data = s0[21];
            float v173_data = r1[2];
            r1[2] = (v173_data + (v160_data * v171_data));
            float v176_data = s0[29];
            float v178_data = r1[3];
            r1[3] = (v178_data + (v160_data * v176_data));
          }
          if (v21_lead < 8) {
            float v184_data = r0[6];
            float v185_data = s0[6];
            float v187_data = r1[0];
            r1[0] = (v187_data + (v184_data * v185_data));
            float v190_data = s0[14];
            float v192_data = r1[1];
            r1[1] = (v192_data + (v184_data * v190_data));
            float v195_data = s0[22];
            float v197_data = r1[2];
            r1[2] = (v197_data + (v184_data * v195_data));
            float v200_data = s0[30];
            float v202_data = r1[3];
            r1[3] = (v202_data + (v184_data * v200_data));
          }
          if (v21_lead < 8) {
            float v208_data = r0[7];
            float v209_data = s0[7];
            float v211_data = r1[0];
            r1[0] = (v211_data + (v208_data * v209_data));
            float v214_data = s0[15];
            float v216_data = r1[1];
            r1[1] = (v216_data + (v208_data * v214_data));
            float v219_data = s0[23];
            float v221_data = r1[2];
            r1[2] = (v221_data + (v208_data * v219_data));
            float v224_data = s0[31];
            float v226_data = r1[3];
            r1[3] = (v226_data + (v208_data * v224_data));
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v232_i1 = 0; v232_i1 < 4; ++v232_i1) {
              float v234_data = r1[v232_i1];
              int32_t v241_a = v21_lead + (v232_i1 * 8);
              s1[(v241_a ^ ((v241_a >> 5) & 31))] = v234_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          // r2 = +(r0 * s2) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          if (v21_lead < 8) {
            float v251_data = r0[0];
            float v252_data = s2[0];
            float v254_data = ir2[0];
            ir2[0] = (v254_data + (v251_data * v252_data));
            float v257_data = s2[8];
            float v259_data = ir2[1];
            ir2[1] = (v259_data + (v251_data * v257_data));
            float v262_data = s2[16];
            float v264_data = ir2[2];
            ir2[2] = (v264_data + (v251_data * v262_data));
            float v267_data = s2[24];
            float v269_data = ir2[3];
            ir2[3] = (v269_data + (v251_data * v267_data));
          }
          if (v21_lead < 8) {
            float v275_data = r0[1];
            float v276_data = s2[1];
            float v278_data = ir2[0];
            ir2[0] = (v278_data + (v275_data * v276_data));
            float v281_data = s2[9];
            float v283_data = ir2[1];
            ir2[1] = (v283_data + (v275_data * v281_data));
            float v286_data = s2[17];
            float v288_data = ir2[2];
            ir2[2] = (v288_data + (v275_data * v286_data));
            float v291_data = s2[25];
            float v293_data = ir2[3];
            ir2[3] = (v293_data + (v275_data * v291_data));
          }
          if (v21_lead < 8) {
            float v299_data = r0[2];
            float v300_data = s2[2];
            float v302_data = ir2[0];
            ir2[0] = (v302_data + (v299_data * v300_data));
            float v305_data = s2[10];
            float v307_data = ir2[1];
            ir2[1] = (v307_data + (v299_data * v305_data));
            float v310_data = s2[18];
            float v312_data = ir2[2];
            ir2[2] = (v312_data + (v299_data * v310_data));
            float v315_data = s2[26];
            float v317_data = ir2[3];
            ir2[3] = (v317_data + (v299_data * v315_data));
          }
          if (v21_lead < 8) {
            float v323_data = r0[3];
            float v324_data = s2[3];
            float v326_data = ir2[0];
            ir2[0] = (v326_data + (v323_data * v324_data));
            float v329_data = s2[11];
            float v331_data = ir2[1];
            ir2[1] = (v331_data + (v323_data * v329_data));
            float v334_data = s2[19];
            float v336_data = ir2[2];
            ir2[2] = (v336_data + (v323_data * v334_data));
            float v339_data = s2[27];
            float v341_data = ir2[3];
            ir2[3] = (v341_data + (v323_data * v339_data));
          }
          if (v21_lead < 8) {
            float v347_data = r0[4];
            float v348_data = s2[4];
            float v350_data = ir2[0];
            ir2[0] = (v350_data + (v347_data * v348_data));
            float v353_data = s2[12];
            float v355_data = ir2[1];
            ir2[1] = (v355_data + (v347_data * v353_data));
            float v358_data = s2[20];
            float v360_data = ir2[2];
            ir2[2] = (v360_data + (v347_data * v358_data));
            float v363_data = s2[28];
            float v365_data = ir2[3];
            ir2[3] = (v365_data + (v347_data * v363_data));
          }
          if (v21_lead < 8) {
            float v371_data = r0[5];
            float v372_data = s2[5];
            float v374_data = ir2[0];
            ir2[0] = (v374_data + (v371_data * v372_data));
            float v377_data = s2[13];
            float v379_data = ir2[1];
            ir2[1] = (v379_data + (v371_data * v377_data));
            float v382_data = s2[21];
            float v384_data = ir2[2];
            ir2[2] = (v384_data + (v371_data * v382_data));
            float v387_data = s2[29];
            float v389_data = ir2[3];
            ir2[3] = (v389_data + (v371_data * v387_data));
          }
          if (v21_lead < 8) {
            float v395_data = r0[6];
            float v396_data = s2[6];
            float v398_data = ir2[0];
            ir2[0] = (v398_data + (v395_data * v396_data));
            float v401_data = s2[14];
            float v403_data = ir2[1];
            ir2[1] = (v403_data + (v395_data * v401_data));
            float v406_data = s2[22];
            float v408_data = ir2[2];
            ir2[2] = (v408_data + (v395_data * v406_data));
            float v411_data = s2[30];
            float v413_data = ir2[3];
            ir2[3] = (v413_data + (v395_data * v411_data));
          }
          if (v21_lead < 8) {
            float v419_data = r0[7];
            float v420_data = s2[7];
            float v422_data = ir2[0];
            ir2[0] = (v422_data + (v419_data * v420_data));
            float v425_data = s2[15];
            float v427_data = ir2[1];
            ir2[1] = (v427_data + (v419_data * v425_data));
            float v430_data = s2[23];
            float v432_data = ir2[2];
            ir2[2] = (v432_data + (v419_data * v430_data));
            float v435_data = s2[31];
            float v437_data = ir2[3];
            ir2[3] = (v437_data + (v419_data * v435_data));
          }
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v443_n1 = 0; v443_n1 < 4; ++v443_n1) {
              float v445_data = ir2[v443_n1];
              r2[v443_n1] = v445_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r2);
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v451_i1 = 0; v451_i1 < 4; ++v451_i1) {
              float v453_data = r2[v451_i1];
              int32_t v461_a = v21_lead + ((v451_i1 + 4) * 8);
              s1[(v461_a ^ ((v461_a >> 5) & 31))] = v453_data;
            }
          }
          __syncwarp();
          // glb_m3 = abs(s1)
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v469_k1 = 0; v469_k1 < 8; ++v469_k1) {
              int32_t v475_a = v469_k1 * 8;
              int32_t v476_a = v21_lead + v475_a;
              float v480_data = s1[(v476_a ^ ((v476_a >> 5) & 31))];
              glb_m3[(v21_lead + v475_a)] = (fabsf(v480_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

