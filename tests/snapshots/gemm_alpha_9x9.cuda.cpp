// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08a27dccde, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08a27dccde, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08a27dccde<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 9) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v10_a = v4_i1 * 9;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __ldcg(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            if (threadIdx.x < 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[9]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 9), (0, 9)] [(0, 9)]
            float ir1[9]{};
            int32_t v23_lead = threadIdx.x % 16;
            if (v23_lead < 9) {
              float v25_data = r0[0];
              float v26_data = s0[0];
              float v28_data = ir1[0];
              ir1[0] = (v28_data + (v25_data * v26_data));
              float v31_data = s0[9];
              float v33_data = ir1[1];
              ir1[1] = (v33_data + (v25_data * v31_data));
              float v36_data = s0[18];
              float v38_data = ir1[2];
              ir1[2] = (v38_data + (v25_data * v36_data));
              float v41_data = s0[27];
              float v43_data = ir1[3];
              ir1[3] = (v43_data + (v25_data * v41_data));
              float v46_data = s0[36];
              float v48_data = ir1[4];
              ir1[4] = (v48_data + (v25_data * v46_data));
              float v51_data = s0[45];
              float v53_data = ir1[5];
              ir1[5] = (v53_data + (v25_data * v51_data));
              float v56_data = s0[54];
              float v58_data = ir1[6];
              ir1[6] = (v58_data + (v25_data * v56_data));
              float v61_data = s0[63];
              float v63_data = ir1[7];
              ir1[7] = (v63_data + (v25_data * v61_data));
              float v66_data = s0[72];
              float v68_data = ir1[8];
              ir1[8] = (v68_data + (v25_data * v66_data));
            }
            if (v23_lead < 9) {
              float v74_data = r0[1];
              float v75_data = s0[1];
              float v77_data = ir1[0];
              ir1[0] = (v77_data + (v74_data * v75_data));
              float v80_data = s0[10];
              float v82_data = ir1[1];
              ir1[1] = (v82_data + (v74_data * v80_data));
              float v85_data = s0[19];
              float v87_data = ir1[2];
              ir1[2] = (v87_data + (v74_data * v85_data));
              float v90_data = s0[28];
              float v92_data = ir1[3];
              ir1[3] = (v92_data + (v74_data * v90_data));
              float v95_data = s0[37];
              float v97_data = ir1[4];
              ir1[4] = (v97_data + (v74_data * v95_data));
              float v100_data = s0[46];
              float v102_data = ir1[5];
              ir1[5] = (v102_data + (v74_data * v100_data));
              float v105_data = s0[55];
              float v107_data = ir1[6];
              ir1[6] = (v107_data + (v74_data * v105_data));
              float v110_data = s0[64];
              float v112_data = ir1[7];
              ir1[7] = (v112_data + (v74_data * v110_data));
              float v115_data = s0[73];
              float v117_data = ir1[8];
              ir1[8] = (v117_data + (v74_data * v115_data));
            }
            if (v23_lead < 9) {
              float v123_data = r0[2];
              float v124_data = s0[2];
              float v126_data = ir1[0];
              ir1[0] = (v126_data + (v123_data * v124_data));
              float v129_data = s0[11];
              float v131_data = ir1[1];
              ir1[1] = (v131_data + (v123_data * v129_data));
              float v134_data = s0[20];
              float v136_data = ir1[2];
              ir1[2] = (v136_data + (v123_data * v134_data));
              float v139_data = s0[29];
              float v141_data = ir1[3];
              ir1[3] = (v141_data + (v123_data * v139_data));
              float v144_data = s0[38];
              float v146_data = ir1[4];
              ir1[4] = (v146_data + (v123_data * v144_data));
              float v149_data = s0[47];
              float v151_data = ir1[5];
              ir1[5] = (v151_data + (v123_data * v149_data));
              float v154_data = s0[56];
              float v156_data = ir1[6];
              ir1[6] = (v156_data + (v123_data * v154_data));
              float v159_data = s0[65];
              float v161_data = ir1[7];
              ir1[7] = (v161_data + (v123_data * v159_data));
              float v164_data = s0[74];
              float v166_data = ir1[8];
              ir1[8] = (v166_data + (v123_data * v164_data));
            }
            if (v23_lead < 9) {
              float v172_data = r0[3];
              float v173_data = s0[3];
              float v175_data = ir1[0];
              ir1[0] = (v175_data + (v172_data * v173_data));
              float v178_data = s0[12];
              float v180_data = ir1[1];
              ir1[1] = (v180_data + (v172_data * v178_data));
              float v183_data = s0[21];
              float v185_data = ir1[2];
              ir1[2] = (v185_data + (v172_data * v183_data));
              float v188_data = s0[30];
              float v190_data = ir1[3];
              ir1[3] = (v190_data + (v172_data * v188_data));
              float v193_data = s0[39];
              float v195_data = ir1[4];
              ir1[4] = (v195_data + (v172_data * v193_data));
              float v198_data = s0[48];
              float v200_data = ir1[5];
              ir1[5] = (v200_data + (v172_data * v198_data));
              float v203_data = s0[57];
              float v205_data = ir1[6];
              ir1[6] = (v205_data + (v172_data * v203_data));
              float v208_data = s0[66];
              float v210_data = ir1[7];
              ir1[7] = (v210_data + (v172_data * v208_data));
              float v213_data = s0[75];
              float v215_data = ir1[8];
              ir1[8] = (v215_data + (v172_data * v213_data));
            }
            if (v23_lead < 9) {
              float v221_data = r0[4];
              float v222_data = s0[4];
              float v224_data = ir1[0];
              ir1[0] = (v224_data + (v221_data * v222_data));
              float v227_data = s0[13];
              float v229_data = ir1[1];
              ir1[1] = (v229_data + (v221_data * v227_data));
              float v232_data = s0[22];
              float v234_data = ir1[2];
              ir1[2] = (v234_data + (v221_data * v232_data));
              float v237_data = s0[31];
              float v239_data = ir1[3];
              ir1[3] = (v239_data + (v221_data * v237_data));
              float v242_data = s0[40];
              float v244_data = ir1[4];
              ir1[4] = (v244_data + (v221_data * v242_data));
              float v247_data = s0[49];
              float v249_data = ir1[5];
              ir1[5] = (v249_data + (v221_data * v247_data));
              float v252_data = s0[58];
              float v254_data = ir1[6];
              ir1[6] = (v254_data + (v221_data * v252_data));
              float v257_data = s0[67];
              float v259_data = ir1[7];
              ir1[7] = (v259_data + (v221_data * v257_data));
              float v262_data = s0[76];
              float v264_data = ir1[8];
              ir1[8] = (v264_data + (v221_data * v262_data));
            }
            if (v23_lead < 9) {
              float v270_data = r0[5];
              float v271_data = s0[5];
              float v273_data = ir1[0];
              ir1[0] = (v273_data + (v270_data * v271_data));
              float v276_data = s0[14];
              float v278_data = ir1[1];
              ir1[1] = (v278_data + (v270_data * v276_data));
              float v281_data = s0[23];
              float v283_data = ir1[2];
              ir1[2] = (v283_data + (v270_data * v281_data));
              float v286_data = s0[32];
              float v288_data = ir1[3];
              ir1[3] = (v288_data + (v270_data * v286_data));
              float v291_data = s0[41];
              float v293_data = ir1[4];
              ir1[4] = (v293_data + (v270_data * v291_data));
              float v296_data = s0[50];
              float v298_data = ir1[5];
              ir1[5] = (v298_data + (v270_data * v296_data));
              float v301_data = s0[59];
              float v303_data = ir1[6];
              ir1[6] = (v303_data + (v270_data * v301_data));
              float v306_data = s0[68];
              float v308_data = ir1[7];
              ir1[7] = (v308_data + (v270_data * v306_data));
              float v311_data = s0[77];
              float v313_data = ir1[8];
              ir1[8] = (v313_data + (v270_data * v311_data));
            }
            if (v23_lead < 9) {
              float v319_data = r0[6];
              float v320_data = s0[6];
              float v322_data = ir1[0];
              ir1[0] = (v322_data + (v319_data * v320_data));
              float v325_data = s0[15];
              float v327_data = ir1[1];
              ir1[1] = (v327_data + (v319_data * v325_data));
              float v330_data = s0[24];
              float v332_data = ir1[2];
              ir1[2] = (v332_data + (v319_data * v330_data));
              float v335_data = s0[33];
              float v337_data = ir1[3];
              ir1[3] = (v337_data + (v319_data * v335_data));
              float v340_data = s0[42];
              float v342_data = ir1[4];
              ir1[4] = (v342_data + (v319_data * v340_data));
              float v345_data = s0[51];
              float v347_data = ir1[5];
              ir1[5] = (v347_data + (v319_data * v345_data));
              float v350_data = s0[60];
              float v352_data = ir1[6];
              ir1[6] = (v352_data + (v319_data * v350_data));
              float v355_data = s0[69];
              float v357_data = ir1[7];
              ir1[7] = (v357_data + (v319_data * v355_data));
              float v360_data = s0[78];
              float v362_data = ir1[8];
              ir1[8] = (v362_data + (v319_data * v360_data));
            }
            if (v23_lead < 9) {
              float v368_data = r0[7];
              float v369_data = s0[7];
              float v371_data = ir1[0];
              ir1[0] = (v371_data + (v368_data * v369_data));
              float v374_data = s0[16];
              float v376_data = ir1[1];
              ir1[1] = (v376_data + (v368_data * v374_data));
              float v379_data = s0[25];
              float v381_data = ir1[2];
              ir1[2] = (v381_data + (v368_data * v379_data));
              float v384_data = s0[34];
              float v386_data = ir1[3];
              ir1[3] = (v386_data + (v368_data * v384_data));
              float v389_data = s0[43];
              float v391_data = ir1[4];
              ir1[4] = (v391_data + (v368_data * v389_data));
              float v394_data = s0[52];
              float v396_data = ir1[5];
              ir1[5] = (v396_data + (v368_data * v394_data));
              float v399_data = s0[61];
              float v401_data = ir1[6];
              ir1[6] = (v401_data + (v368_data * v399_data));
              float v404_data = s0[70];
              float v406_data = ir1[7];
              ir1[7] = (v406_data + (v368_data * v404_data));
              float v409_data = s0[79];
              float v411_data = ir1[8];
              ir1[8] = (v411_data + (v368_data * v409_data));
            }
            if (v23_lead < 9) {
              float v417_data = r0[8];
              float v418_data = s0[8];
              float v420_data = ir1[0];
              ir1[0] = (v420_data + (v417_data * v418_data));
              float v423_data = s0[17];
              float v425_data = ir1[1];
              ir1[1] = (v425_data + (v417_data * v423_data));
              float v428_data = s0[26];
              float v430_data = ir1[2];
              ir1[2] = (v430_data + (v417_data * v428_data));
              float v433_data = s0[35];
              float v435_data = ir1[3];
              ir1[3] = (v435_data + (v417_data * v433_data));
              float v438_data = s0[44];
              float v440_data = ir1[4];
              ir1[4] = (v440_data + (v417_data * v438_data));
              float v443_data = s0[53];
              float v445_data = ir1[5];
              ir1[5] = (v445_data + (v417_data * v443_data));
              float v448_data = s0[62];
              float v450_data = ir1[6];
              ir1[6] = (v450_data + (v417_data * v448_data));
              float v453_data = s0[71];
              float v455_data = ir1[7];
              ir1[7] = (v455_data + (v417_data * v453_data));
              float v458_data = s0[80];
              float v460_data = ir1[8];
              ir1[8] = (v460_data + (v417_data * v458_data));
            }
            if (v23_lead < 9) {
              #pragma unroll
              for (int32_t v467_n1 = 0; v467_n1 < 9; ++v467_n1) {
                int32_t v468_a = 0 + v467_n1;
                float v470_data = ir1[v467_n1];
                int32_t v472_a = 0 + v467_n1;
                r1[v467_n1] = (v470_data * 13.0f);
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v476_lead = threadIdx.x % 16;
          if (v476_lead < 9) {
            #pragma unroll
            for (int32_t v478_i1 = 0; v478_i1 < 9; ++v478_i1) {
              int32_t v479_a = 0 + v478_i1;
              float v481_data = r1[v478_i1];
              int32_t v488_a = v476_lead + (v478_i1 * 9);
              glb_m0[v488_a] = v481_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

