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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 9;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
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
          if (v3_lead < 9) {
            float v29_data = r0[0];
            float v30_data = s0[0];
            float v32_data = ir1[0];
            ir1[0] = (v32_data + (v29_data * v30_data));
            float v35_data = s0[9];
            float v37_data = ir1[1];
            ir1[1] = (v37_data + (v29_data * v35_data));
            float v40_data = s0[18];
            float v42_data = ir1[2];
            ir1[2] = (v42_data + (v29_data * v40_data));
            float v45_data = s0[27];
            float v47_data = ir1[3];
            ir1[3] = (v47_data + (v29_data * v45_data));
            float v50_data = s0[36];
            float v52_data = ir1[4];
            ir1[4] = (v52_data + (v29_data * v50_data));
            float v55_data = s0[45];
            float v57_data = ir1[5];
            ir1[5] = (v57_data + (v29_data * v55_data));
            float v60_data = s0[54];
            float v62_data = ir1[6];
            ir1[6] = (v62_data + (v29_data * v60_data));
            float v65_data = s0[63];
            float v67_data = ir1[7];
            ir1[7] = (v67_data + (v29_data * v65_data));
            float v70_data = s0[72];
            float v72_data = ir1[8];
            ir1[8] = (v72_data + (v29_data * v70_data));
          }
          if (v3_lead < 9) {
            float v78_data = r0[1];
            float v79_data = s0[1];
            float v81_data = ir1[0];
            ir1[0] = (v81_data + (v78_data * v79_data));
            float v84_data = s0[10];
            float v86_data = ir1[1];
            ir1[1] = (v86_data + (v78_data * v84_data));
            float v89_data = s0[19];
            float v91_data = ir1[2];
            ir1[2] = (v91_data + (v78_data * v89_data));
            float v94_data = s0[28];
            float v96_data = ir1[3];
            ir1[3] = (v96_data + (v78_data * v94_data));
            float v99_data = s0[37];
            float v101_data = ir1[4];
            ir1[4] = (v101_data + (v78_data * v99_data));
            float v104_data = s0[46];
            float v106_data = ir1[5];
            ir1[5] = (v106_data + (v78_data * v104_data));
            float v109_data = s0[55];
            float v111_data = ir1[6];
            ir1[6] = (v111_data + (v78_data * v109_data));
            float v114_data = s0[64];
            float v116_data = ir1[7];
            ir1[7] = (v116_data + (v78_data * v114_data));
            float v119_data = s0[73];
            float v121_data = ir1[8];
            ir1[8] = (v121_data + (v78_data * v119_data));
          }
          if (v3_lead < 9) {
            float v127_data = r0[2];
            float v128_data = s0[2];
            float v130_data = ir1[0];
            ir1[0] = (v130_data + (v127_data * v128_data));
            float v133_data = s0[11];
            float v135_data = ir1[1];
            ir1[1] = (v135_data + (v127_data * v133_data));
            float v138_data = s0[20];
            float v140_data = ir1[2];
            ir1[2] = (v140_data + (v127_data * v138_data));
            float v143_data = s0[29];
            float v145_data = ir1[3];
            ir1[3] = (v145_data + (v127_data * v143_data));
            float v148_data = s0[38];
            float v150_data = ir1[4];
            ir1[4] = (v150_data + (v127_data * v148_data));
            float v153_data = s0[47];
            float v155_data = ir1[5];
            ir1[5] = (v155_data + (v127_data * v153_data));
            float v158_data = s0[56];
            float v160_data = ir1[6];
            ir1[6] = (v160_data + (v127_data * v158_data));
            float v163_data = s0[65];
            float v165_data = ir1[7];
            ir1[7] = (v165_data + (v127_data * v163_data));
            float v168_data = s0[74];
            float v170_data = ir1[8];
            ir1[8] = (v170_data + (v127_data * v168_data));
          }
          if (v3_lead < 9) {
            float v176_data = r0[3];
            float v177_data = s0[3];
            float v179_data = ir1[0];
            ir1[0] = (v179_data + (v176_data * v177_data));
            float v182_data = s0[12];
            float v184_data = ir1[1];
            ir1[1] = (v184_data + (v176_data * v182_data));
            float v187_data = s0[21];
            float v189_data = ir1[2];
            ir1[2] = (v189_data + (v176_data * v187_data));
            float v192_data = s0[30];
            float v194_data = ir1[3];
            ir1[3] = (v194_data + (v176_data * v192_data));
            float v197_data = s0[39];
            float v199_data = ir1[4];
            ir1[4] = (v199_data + (v176_data * v197_data));
            float v202_data = s0[48];
            float v204_data = ir1[5];
            ir1[5] = (v204_data + (v176_data * v202_data));
            float v207_data = s0[57];
            float v209_data = ir1[6];
            ir1[6] = (v209_data + (v176_data * v207_data));
            float v212_data = s0[66];
            float v214_data = ir1[7];
            ir1[7] = (v214_data + (v176_data * v212_data));
            float v217_data = s0[75];
            float v219_data = ir1[8];
            ir1[8] = (v219_data + (v176_data * v217_data));
          }
          if (v3_lead < 9) {
            float v225_data = r0[4];
            float v226_data = s0[4];
            float v228_data = ir1[0];
            ir1[0] = (v228_data + (v225_data * v226_data));
            float v231_data = s0[13];
            float v233_data = ir1[1];
            ir1[1] = (v233_data + (v225_data * v231_data));
            float v236_data = s0[22];
            float v238_data = ir1[2];
            ir1[2] = (v238_data + (v225_data * v236_data));
            float v241_data = s0[31];
            float v243_data = ir1[3];
            ir1[3] = (v243_data + (v225_data * v241_data));
            float v246_data = s0[40];
            float v248_data = ir1[4];
            ir1[4] = (v248_data + (v225_data * v246_data));
            float v251_data = s0[49];
            float v253_data = ir1[5];
            ir1[5] = (v253_data + (v225_data * v251_data));
            float v256_data = s0[58];
            float v258_data = ir1[6];
            ir1[6] = (v258_data + (v225_data * v256_data));
            float v261_data = s0[67];
            float v263_data = ir1[7];
            ir1[7] = (v263_data + (v225_data * v261_data));
            float v266_data = s0[76];
            float v268_data = ir1[8];
            ir1[8] = (v268_data + (v225_data * v266_data));
          }
          if (v3_lead < 9) {
            float v274_data = r0[5];
            float v275_data = s0[5];
            float v277_data = ir1[0];
            ir1[0] = (v277_data + (v274_data * v275_data));
            float v280_data = s0[14];
            float v282_data = ir1[1];
            ir1[1] = (v282_data + (v274_data * v280_data));
            float v285_data = s0[23];
            float v287_data = ir1[2];
            ir1[2] = (v287_data + (v274_data * v285_data));
            float v290_data = s0[32];
            float v292_data = ir1[3];
            ir1[3] = (v292_data + (v274_data * v290_data));
            float v295_data = s0[41];
            float v297_data = ir1[4];
            ir1[4] = (v297_data + (v274_data * v295_data));
            float v300_data = s0[50];
            float v302_data = ir1[5];
            ir1[5] = (v302_data + (v274_data * v300_data));
            float v305_data = s0[59];
            float v307_data = ir1[6];
            ir1[6] = (v307_data + (v274_data * v305_data));
            float v310_data = s0[68];
            float v312_data = ir1[7];
            ir1[7] = (v312_data + (v274_data * v310_data));
            float v315_data = s0[77];
            float v317_data = ir1[8];
            ir1[8] = (v317_data + (v274_data * v315_data));
          }
          if (v3_lead < 9) {
            float v323_data = r0[6];
            float v324_data = s0[6];
            float v326_data = ir1[0];
            ir1[0] = (v326_data + (v323_data * v324_data));
            float v329_data = s0[15];
            float v331_data = ir1[1];
            ir1[1] = (v331_data + (v323_data * v329_data));
            float v334_data = s0[24];
            float v336_data = ir1[2];
            ir1[2] = (v336_data + (v323_data * v334_data));
            float v339_data = s0[33];
            float v341_data = ir1[3];
            ir1[3] = (v341_data + (v323_data * v339_data));
            float v344_data = s0[42];
            float v346_data = ir1[4];
            ir1[4] = (v346_data + (v323_data * v344_data));
            float v349_data = s0[51];
            float v351_data = ir1[5];
            ir1[5] = (v351_data + (v323_data * v349_data));
            float v354_data = s0[60];
            float v356_data = ir1[6];
            ir1[6] = (v356_data + (v323_data * v354_data));
            float v359_data = s0[69];
            float v361_data = ir1[7];
            ir1[7] = (v361_data + (v323_data * v359_data));
            float v364_data = s0[78];
            float v366_data = ir1[8];
            ir1[8] = (v366_data + (v323_data * v364_data));
          }
          if (v3_lead < 9) {
            float v372_data = r0[7];
            float v373_data = s0[7];
            float v375_data = ir1[0];
            ir1[0] = (v375_data + (v372_data * v373_data));
            float v378_data = s0[16];
            float v380_data = ir1[1];
            ir1[1] = (v380_data + (v372_data * v378_data));
            float v383_data = s0[25];
            float v385_data = ir1[2];
            ir1[2] = (v385_data + (v372_data * v383_data));
            float v388_data = s0[34];
            float v390_data = ir1[3];
            ir1[3] = (v390_data + (v372_data * v388_data));
            float v393_data = s0[43];
            float v395_data = ir1[4];
            ir1[4] = (v395_data + (v372_data * v393_data));
            float v398_data = s0[52];
            float v400_data = ir1[5];
            ir1[5] = (v400_data + (v372_data * v398_data));
            float v403_data = s0[61];
            float v405_data = ir1[6];
            ir1[6] = (v405_data + (v372_data * v403_data));
            float v408_data = s0[70];
            float v410_data = ir1[7];
            ir1[7] = (v410_data + (v372_data * v408_data));
            float v413_data = s0[79];
            float v415_data = ir1[8];
            ir1[8] = (v415_data + (v372_data * v413_data));
          }
          if (v3_lead < 9) {
            float v421_data = r0[8];
            float v422_data = s0[8];
            float v424_data = ir1[0];
            ir1[0] = (v424_data + (v421_data * v422_data));
            float v427_data = s0[17];
            float v429_data = ir1[1];
            ir1[1] = (v429_data + (v421_data * v427_data));
            float v432_data = s0[26];
            float v434_data = ir1[2];
            ir1[2] = (v434_data + (v421_data * v432_data));
            float v437_data = s0[35];
            float v439_data = ir1[3];
            ir1[3] = (v439_data + (v421_data * v437_data));
            float v442_data = s0[44];
            float v444_data = ir1[4];
            ir1[4] = (v444_data + (v421_data * v442_data));
            float v447_data = s0[53];
            float v449_data = ir1[5];
            ir1[5] = (v449_data + (v421_data * v447_data));
            float v452_data = s0[62];
            float v454_data = ir1[6];
            ir1[6] = (v454_data + (v421_data * v452_data));
            float v457_data = s0[71];
            float v459_data = ir1[7];
            ir1[7] = (v459_data + (v421_data * v457_data));
            float v462_data = s0[80];
            float v464_data = ir1[8];
            ir1[8] = (v464_data + (v421_data * v462_data));
          }
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v471_n1 = 0; v471_n1 < 9; ++v471_n1) {
              int32_t v472_a = 0 + v471_n1;
              float v474_data = ir1[v471_n1];
              int32_t v476_a = 0 + v471_n1;
              r1[v471_n1] = (v474_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v482_i1 = 0; v482_i1 < 9; ++v482_i1) {
              int32_t v483_a = 0 + v482_i1;
              float v485_data = r1[v482_i1];
              int32_t v492_a = v3_lead + (v482_i1 * 9);
              glb_m0[v492_a] = v485_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

