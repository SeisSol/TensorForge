// === base name ===
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
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 9; ++v8_i1) {
              int32_t v14_a = v8_i1 * 9;
              int32_t v15_a = v6_lead + v14_a;
              float v23_data = __ldcg(&glb_m1[(v6_lead + v14_a)]);
              int32_t v24_a = 0 + v8_i1;
              r0[v24_a] = v23_data;
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
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          if (v6_lead < 9) {
            float v32_data = r0[0];
            float v33_data = s0[0];
            float v35_data = ir1[0];
            ir1[0] = (v35_data + (v32_data * v33_data));
            float v38_data = s0[9];
            float v40_data = ir1[1];
            ir1[1] = (v40_data + (v32_data * v38_data));
            float v43_data = s0[18];
            float v45_data = ir1[2];
            ir1[2] = (v45_data + (v32_data * v43_data));
            float v48_data = s0[27];
            float v50_data = ir1[3];
            ir1[3] = (v50_data + (v32_data * v48_data));
            float v53_data = s0[36];
            float v55_data = ir1[4];
            ir1[4] = (v55_data + (v32_data * v53_data));
            float v58_data = s0[45];
            float v60_data = ir1[5];
            ir1[5] = (v60_data + (v32_data * v58_data));
            float v63_data = s0[54];
            float v65_data = ir1[6];
            ir1[6] = (v65_data + (v32_data * v63_data));
            float v68_data = s0[63];
            float v70_data = ir1[7];
            ir1[7] = (v70_data + (v32_data * v68_data));
            float v73_data = s0[72];
            float v75_data = ir1[8];
            ir1[8] = (v75_data + (v32_data * v73_data));
          }
          if (v6_lead < 9) {
            float v81_data = r0[1];
            float v82_data = s0[1];
            float v84_data = ir1[0];
            ir1[0] = (v84_data + (v81_data * v82_data));
            float v87_data = s0[10];
            float v89_data = ir1[1];
            ir1[1] = (v89_data + (v81_data * v87_data));
            float v92_data = s0[19];
            float v94_data = ir1[2];
            ir1[2] = (v94_data + (v81_data * v92_data));
            float v97_data = s0[28];
            float v99_data = ir1[3];
            ir1[3] = (v99_data + (v81_data * v97_data));
            float v102_data = s0[37];
            float v104_data = ir1[4];
            ir1[4] = (v104_data + (v81_data * v102_data));
            float v107_data = s0[46];
            float v109_data = ir1[5];
            ir1[5] = (v109_data + (v81_data * v107_data));
            float v112_data = s0[55];
            float v114_data = ir1[6];
            ir1[6] = (v114_data + (v81_data * v112_data));
            float v117_data = s0[64];
            float v119_data = ir1[7];
            ir1[7] = (v119_data + (v81_data * v117_data));
            float v122_data = s0[73];
            float v124_data = ir1[8];
            ir1[8] = (v124_data + (v81_data * v122_data));
          }
          if (v6_lead < 9) {
            float v130_data = r0[2];
            float v131_data = s0[2];
            float v133_data = ir1[0];
            ir1[0] = (v133_data + (v130_data * v131_data));
            float v136_data = s0[11];
            float v138_data = ir1[1];
            ir1[1] = (v138_data + (v130_data * v136_data));
            float v141_data = s0[20];
            float v143_data = ir1[2];
            ir1[2] = (v143_data + (v130_data * v141_data));
            float v146_data = s0[29];
            float v148_data = ir1[3];
            ir1[3] = (v148_data + (v130_data * v146_data));
            float v151_data = s0[38];
            float v153_data = ir1[4];
            ir1[4] = (v153_data + (v130_data * v151_data));
            float v156_data = s0[47];
            float v158_data = ir1[5];
            ir1[5] = (v158_data + (v130_data * v156_data));
            float v161_data = s0[56];
            float v163_data = ir1[6];
            ir1[6] = (v163_data + (v130_data * v161_data));
            float v166_data = s0[65];
            float v168_data = ir1[7];
            ir1[7] = (v168_data + (v130_data * v166_data));
            float v171_data = s0[74];
            float v173_data = ir1[8];
            ir1[8] = (v173_data + (v130_data * v171_data));
          }
          if (v6_lead < 9) {
            float v179_data = r0[3];
            float v180_data = s0[3];
            float v182_data = ir1[0];
            ir1[0] = (v182_data + (v179_data * v180_data));
            float v185_data = s0[12];
            float v187_data = ir1[1];
            ir1[1] = (v187_data + (v179_data * v185_data));
            float v190_data = s0[21];
            float v192_data = ir1[2];
            ir1[2] = (v192_data + (v179_data * v190_data));
            float v195_data = s0[30];
            float v197_data = ir1[3];
            ir1[3] = (v197_data + (v179_data * v195_data));
            float v200_data = s0[39];
            float v202_data = ir1[4];
            ir1[4] = (v202_data + (v179_data * v200_data));
            float v205_data = s0[48];
            float v207_data = ir1[5];
            ir1[5] = (v207_data + (v179_data * v205_data));
            float v210_data = s0[57];
            float v212_data = ir1[6];
            ir1[6] = (v212_data + (v179_data * v210_data));
            float v215_data = s0[66];
            float v217_data = ir1[7];
            ir1[7] = (v217_data + (v179_data * v215_data));
            float v220_data = s0[75];
            float v222_data = ir1[8];
            ir1[8] = (v222_data + (v179_data * v220_data));
          }
          if (v6_lead < 9) {
            float v228_data = r0[4];
            float v229_data = s0[4];
            float v231_data = ir1[0];
            ir1[0] = (v231_data + (v228_data * v229_data));
            float v234_data = s0[13];
            float v236_data = ir1[1];
            ir1[1] = (v236_data + (v228_data * v234_data));
            float v239_data = s0[22];
            float v241_data = ir1[2];
            ir1[2] = (v241_data + (v228_data * v239_data));
            float v244_data = s0[31];
            float v246_data = ir1[3];
            ir1[3] = (v246_data + (v228_data * v244_data));
            float v249_data = s0[40];
            float v251_data = ir1[4];
            ir1[4] = (v251_data + (v228_data * v249_data));
            float v254_data = s0[49];
            float v256_data = ir1[5];
            ir1[5] = (v256_data + (v228_data * v254_data));
            float v259_data = s0[58];
            float v261_data = ir1[6];
            ir1[6] = (v261_data + (v228_data * v259_data));
            float v264_data = s0[67];
            float v266_data = ir1[7];
            ir1[7] = (v266_data + (v228_data * v264_data));
            float v269_data = s0[76];
            float v271_data = ir1[8];
            ir1[8] = (v271_data + (v228_data * v269_data));
          }
          if (v6_lead < 9) {
            float v277_data = r0[5];
            float v278_data = s0[5];
            float v280_data = ir1[0];
            ir1[0] = (v280_data + (v277_data * v278_data));
            float v283_data = s0[14];
            float v285_data = ir1[1];
            ir1[1] = (v285_data + (v277_data * v283_data));
            float v288_data = s0[23];
            float v290_data = ir1[2];
            ir1[2] = (v290_data + (v277_data * v288_data));
            float v293_data = s0[32];
            float v295_data = ir1[3];
            ir1[3] = (v295_data + (v277_data * v293_data));
            float v298_data = s0[41];
            float v300_data = ir1[4];
            ir1[4] = (v300_data + (v277_data * v298_data));
            float v303_data = s0[50];
            float v305_data = ir1[5];
            ir1[5] = (v305_data + (v277_data * v303_data));
            float v308_data = s0[59];
            float v310_data = ir1[6];
            ir1[6] = (v310_data + (v277_data * v308_data));
            float v313_data = s0[68];
            float v315_data = ir1[7];
            ir1[7] = (v315_data + (v277_data * v313_data));
            float v318_data = s0[77];
            float v320_data = ir1[8];
            ir1[8] = (v320_data + (v277_data * v318_data));
          }
          if (v6_lead < 9) {
            float v326_data = r0[6];
            float v327_data = s0[6];
            float v329_data = ir1[0];
            ir1[0] = (v329_data + (v326_data * v327_data));
            float v332_data = s0[15];
            float v334_data = ir1[1];
            ir1[1] = (v334_data + (v326_data * v332_data));
            float v337_data = s0[24];
            float v339_data = ir1[2];
            ir1[2] = (v339_data + (v326_data * v337_data));
            float v342_data = s0[33];
            float v344_data = ir1[3];
            ir1[3] = (v344_data + (v326_data * v342_data));
            float v347_data = s0[42];
            float v349_data = ir1[4];
            ir1[4] = (v349_data + (v326_data * v347_data));
            float v352_data = s0[51];
            float v354_data = ir1[5];
            ir1[5] = (v354_data + (v326_data * v352_data));
            float v357_data = s0[60];
            float v359_data = ir1[6];
            ir1[6] = (v359_data + (v326_data * v357_data));
            float v362_data = s0[69];
            float v364_data = ir1[7];
            ir1[7] = (v364_data + (v326_data * v362_data));
            float v367_data = s0[78];
            float v369_data = ir1[8];
            ir1[8] = (v369_data + (v326_data * v367_data));
          }
          if (v6_lead < 9) {
            float v375_data = r0[7];
            float v376_data = s0[7];
            float v378_data = ir1[0];
            ir1[0] = (v378_data + (v375_data * v376_data));
            float v381_data = s0[16];
            float v383_data = ir1[1];
            ir1[1] = (v383_data + (v375_data * v381_data));
            float v386_data = s0[25];
            float v388_data = ir1[2];
            ir1[2] = (v388_data + (v375_data * v386_data));
            float v391_data = s0[34];
            float v393_data = ir1[3];
            ir1[3] = (v393_data + (v375_data * v391_data));
            float v396_data = s0[43];
            float v398_data = ir1[4];
            ir1[4] = (v398_data + (v375_data * v396_data));
            float v401_data = s0[52];
            float v403_data = ir1[5];
            ir1[5] = (v403_data + (v375_data * v401_data));
            float v406_data = s0[61];
            float v408_data = ir1[6];
            ir1[6] = (v408_data + (v375_data * v406_data));
            float v411_data = s0[70];
            float v413_data = ir1[7];
            ir1[7] = (v413_data + (v375_data * v411_data));
            float v416_data = s0[79];
            float v418_data = ir1[8];
            ir1[8] = (v418_data + (v375_data * v416_data));
          }
          if (v6_lead < 9) {
            float v424_data = r0[8];
            float v425_data = s0[8];
            float v427_data = ir1[0];
            ir1[0] = (v427_data + (v424_data * v425_data));
            float v430_data = s0[17];
            float v432_data = ir1[1];
            ir1[1] = (v432_data + (v424_data * v430_data));
            float v435_data = s0[26];
            float v437_data = ir1[2];
            ir1[2] = (v437_data + (v424_data * v435_data));
            float v440_data = s0[35];
            float v442_data = ir1[3];
            ir1[3] = (v442_data + (v424_data * v440_data));
            float v445_data = s0[44];
            float v447_data = ir1[4];
            ir1[4] = (v447_data + (v424_data * v445_data));
            float v450_data = s0[53];
            float v452_data = ir1[5];
            ir1[5] = (v452_data + (v424_data * v450_data));
            float v455_data = s0[62];
            float v457_data = ir1[6];
            ir1[6] = (v457_data + (v424_data * v455_data));
            float v460_data = s0[71];
            float v462_data = ir1[7];
            ir1[7] = (v462_data + (v424_data * v460_data));
            float v465_data = s0[80];
            float v467_data = ir1[8];
            ir1[8] = (v467_data + (v424_data * v465_data));
          }
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v474_n1 = 0; v474_n1 < 9; ++v474_n1) {
              int32_t v475_a = 0 + v474_n1;
              float v477_data = ir1[v474_n1];
              int32_t v479_a = 0 + v474_n1;
              r1[v474_n1] = (v477_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v485_i1 = 0; v485_i1 < 9; ++v485_i1) {
              int32_t v486_a = 0 + v485_i1;
              float v488_data = r1[v485_i1];
              int32_t v495_a = v6_lead + (v485_i1 * 9);
              glb_m0[v495_a] = v488_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

