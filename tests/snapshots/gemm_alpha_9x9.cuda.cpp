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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 9), (0, 9)] [(0, 9)]
            float ir1[9]{};
            if (v3_lead < 9) {
              float v27_data = r0[0];
              float v28_data = s0[0];
              float v30_data = ir1[0];
              ir1[0] = (v30_data + (v27_data * v28_data));
              float v33_data = s0[9];
              float v35_data = ir1[1];
              ir1[1] = (v35_data + (v27_data * v33_data));
              float v38_data = s0[18];
              float v40_data = ir1[2];
              ir1[2] = (v40_data + (v27_data * v38_data));
              float v43_data = s0[27];
              float v45_data = ir1[3];
              ir1[3] = (v45_data + (v27_data * v43_data));
              float v48_data = s0[36];
              float v50_data = ir1[4];
              ir1[4] = (v50_data + (v27_data * v48_data));
              float v53_data = s0[45];
              float v55_data = ir1[5];
              ir1[5] = (v55_data + (v27_data * v53_data));
              float v58_data = s0[54];
              float v60_data = ir1[6];
              ir1[6] = (v60_data + (v27_data * v58_data));
              float v63_data = s0[63];
              float v65_data = ir1[7];
              ir1[7] = (v65_data + (v27_data * v63_data));
              float v68_data = s0[72];
              float v70_data = ir1[8];
              ir1[8] = (v70_data + (v27_data * v68_data));
            }
            if (v3_lead < 9) {
              float v76_data = r0[1];
              float v77_data = s0[1];
              float v79_data = ir1[0];
              ir1[0] = (v79_data + (v76_data * v77_data));
              float v82_data = s0[10];
              float v84_data = ir1[1];
              ir1[1] = (v84_data + (v76_data * v82_data));
              float v87_data = s0[19];
              float v89_data = ir1[2];
              ir1[2] = (v89_data + (v76_data * v87_data));
              float v92_data = s0[28];
              float v94_data = ir1[3];
              ir1[3] = (v94_data + (v76_data * v92_data));
              float v97_data = s0[37];
              float v99_data = ir1[4];
              ir1[4] = (v99_data + (v76_data * v97_data));
              float v102_data = s0[46];
              float v104_data = ir1[5];
              ir1[5] = (v104_data + (v76_data * v102_data));
              float v107_data = s0[55];
              float v109_data = ir1[6];
              ir1[6] = (v109_data + (v76_data * v107_data));
              float v112_data = s0[64];
              float v114_data = ir1[7];
              ir1[7] = (v114_data + (v76_data * v112_data));
              float v117_data = s0[73];
              float v119_data = ir1[8];
              ir1[8] = (v119_data + (v76_data * v117_data));
            }
            if (v3_lead < 9) {
              float v125_data = r0[2];
              float v126_data = s0[2];
              float v128_data = ir1[0];
              ir1[0] = (v128_data + (v125_data * v126_data));
              float v131_data = s0[11];
              float v133_data = ir1[1];
              ir1[1] = (v133_data + (v125_data * v131_data));
              float v136_data = s0[20];
              float v138_data = ir1[2];
              ir1[2] = (v138_data + (v125_data * v136_data));
              float v141_data = s0[29];
              float v143_data = ir1[3];
              ir1[3] = (v143_data + (v125_data * v141_data));
              float v146_data = s0[38];
              float v148_data = ir1[4];
              ir1[4] = (v148_data + (v125_data * v146_data));
              float v151_data = s0[47];
              float v153_data = ir1[5];
              ir1[5] = (v153_data + (v125_data * v151_data));
              float v156_data = s0[56];
              float v158_data = ir1[6];
              ir1[6] = (v158_data + (v125_data * v156_data));
              float v161_data = s0[65];
              float v163_data = ir1[7];
              ir1[7] = (v163_data + (v125_data * v161_data));
              float v166_data = s0[74];
              float v168_data = ir1[8];
              ir1[8] = (v168_data + (v125_data * v166_data));
            }
            if (v3_lead < 9) {
              float v174_data = r0[3];
              float v175_data = s0[3];
              float v177_data = ir1[0];
              ir1[0] = (v177_data + (v174_data * v175_data));
              float v180_data = s0[12];
              float v182_data = ir1[1];
              ir1[1] = (v182_data + (v174_data * v180_data));
              float v185_data = s0[21];
              float v187_data = ir1[2];
              ir1[2] = (v187_data + (v174_data * v185_data));
              float v190_data = s0[30];
              float v192_data = ir1[3];
              ir1[3] = (v192_data + (v174_data * v190_data));
              float v195_data = s0[39];
              float v197_data = ir1[4];
              ir1[4] = (v197_data + (v174_data * v195_data));
              float v200_data = s0[48];
              float v202_data = ir1[5];
              ir1[5] = (v202_data + (v174_data * v200_data));
              float v205_data = s0[57];
              float v207_data = ir1[6];
              ir1[6] = (v207_data + (v174_data * v205_data));
              float v210_data = s0[66];
              float v212_data = ir1[7];
              ir1[7] = (v212_data + (v174_data * v210_data));
              float v215_data = s0[75];
              float v217_data = ir1[8];
              ir1[8] = (v217_data + (v174_data * v215_data));
            }
            if (v3_lead < 9) {
              float v223_data = r0[4];
              float v224_data = s0[4];
              float v226_data = ir1[0];
              ir1[0] = (v226_data + (v223_data * v224_data));
              float v229_data = s0[13];
              float v231_data = ir1[1];
              ir1[1] = (v231_data + (v223_data * v229_data));
              float v234_data = s0[22];
              float v236_data = ir1[2];
              ir1[2] = (v236_data + (v223_data * v234_data));
              float v239_data = s0[31];
              float v241_data = ir1[3];
              ir1[3] = (v241_data + (v223_data * v239_data));
              float v244_data = s0[40];
              float v246_data = ir1[4];
              ir1[4] = (v246_data + (v223_data * v244_data));
              float v249_data = s0[49];
              float v251_data = ir1[5];
              ir1[5] = (v251_data + (v223_data * v249_data));
              float v254_data = s0[58];
              float v256_data = ir1[6];
              ir1[6] = (v256_data + (v223_data * v254_data));
              float v259_data = s0[67];
              float v261_data = ir1[7];
              ir1[7] = (v261_data + (v223_data * v259_data));
              float v264_data = s0[76];
              float v266_data = ir1[8];
              ir1[8] = (v266_data + (v223_data * v264_data));
            }
            if (v3_lead < 9) {
              float v272_data = r0[5];
              float v273_data = s0[5];
              float v275_data = ir1[0];
              ir1[0] = (v275_data + (v272_data * v273_data));
              float v278_data = s0[14];
              float v280_data = ir1[1];
              ir1[1] = (v280_data + (v272_data * v278_data));
              float v283_data = s0[23];
              float v285_data = ir1[2];
              ir1[2] = (v285_data + (v272_data * v283_data));
              float v288_data = s0[32];
              float v290_data = ir1[3];
              ir1[3] = (v290_data + (v272_data * v288_data));
              float v293_data = s0[41];
              float v295_data = ir1[4];
              ir1[4] = (v295_data + (v272_data * v293_data));
              float v298_data = s0[50];
              float v300_data = ir1[5];
              ir1[5] = (v300_data + (v272_data * v298_data));
              float v303_data = s0[59];
              float v305_data = ir1[6];
              ir1[6] = (v305_data + (v272_data * v303_data));
              float v308_data = s0[68];
              float v310_data = ir1[7];
              ir1[7] = (v310_data + (v272_data * v308_data));
              float v313_data = s0[77];
              float v315_data = ir1[8];
              ir1[8] = (v315_data + (v272_data * v313_data));
            }
            if (v3_lead < 9) {
              float v321_data = r0[6];
              float v322_data = s0[6];
              float v324_data = ir1[0];
              ir1[0] = (v324_data + (v321_data * v322_data));
              float v327_data = s0[15];
              float v329_data = ir1[1];
              ir1[1] = (v329_data + (v321_data * v327_data));
              float v332_data = s0[24];
              float v334_data = ir1[2];
              ir1[2] = (v334_data + (v321_data * v332_data));
              float v337_data = s0[33];
              float v339_data = ir1[3];
              ir1[3] = (v339_data + (v321_data * v337_data));
              float v342_data = s0[42];
              float v344_data = ir1[4];
              ir1[4] = (v344_data + (v321_data * v342_data));
              float v347_data = s0[51];
              float v349_data = ir1[5];
              ir1[5] = (v349_data + (v321_data * v347_data));
              float v352_data = s0[60];
              float v354_data = ir1[6];
              ir1[6] = (v354_data + (v321_data * v352_data));
              float v357_data = s0[69];
              float v359_data = ir1[7];
              ir1[7] = (v359_data + (v321_data * v357_data));
              float v362_data = s0[78];
              float v364_data = ir1[8];
              ir1[8] = (v364_data + (v321_data * v362_data));
            }
            if (v3_lead < 9) {
              float v370_data = r0[7];
              float v371_data = s0[7];
              float v373_data = ir1[0];
              ir1[0] = (v373_data + (v370_data * v371_data));
              float v376_data = s0[16];
              float v378_data = ir1[1];
              ir1[1] = (v378_data + (v370_data * v376_data));
              float v381_data = s0[25];
              float v383_data = ir1[2];
              ir1[2] = (v383_data + (v370_data * v381_data));
              float v386_data = s0[34];
              float v388_data = ir1[3];
              ir1[3] = (v388_data + (v370_data * v386_data));
              float v391_data = s0[43];
              float v393_data = ir1[4];
              ir1[4] = (v393_data + (v370_data * v391_data));
              float v396_data = s0[52];
              float v398_data = ir1[5];
              ir1[5] = (v398_data + (v370_data * v396_data));
              float v401_data = s0[61];
              float v403_data = ir1[6];
              ir1[6] = (v403_data + (v370_data * v401_data));
              float v406_data = s0[70];
              float v408_data = ir1[7];
              ir1[7] = (v408_data + (v370_data * v406_data));
              float v411_data = s0[79];
              float v413_data = ir1[8];
              ir1[8] = (v413_data + (v370_data * v411_data));
            }
            if (v3_lead < 9) {
              float v419_data = r0[8];
              float v420_data = s0[8];
              float v422_data = ir1[0];
              ir1[0] = (v422_data + (v419_data * v420_data));
              float v425_data = s0[17];
              float v427_data = ir1[1];
              ir1[1] = (v427_data + (v419_data * v425_data));
              float v430_data = s0[26];
              float v432_data = ir1[2];
              ir1[2] = (v432_data + (v419_data * v430_data));
              float v435_data = s0[35];
              float v437_data = ir1[3];
              ir1[3] = (v437_data + (v419_data * v435_data));
              float v440_data = s0[44];
              float v442_data = ir1[4];
              ir1[4] = (v442_data + (v419_data * v440_data));
              float v445_data = s0[53];
              float v447_data = ir1[5];
              ir1[5] = (v447_data + (v419_data * v445_data));
              float v450_data = s0[62];
              float v452_data = ir1[6];
              ir1[6] = (v452_data + (v419_data * v450_data));
              float v455_data = s0[71];
              float v457_data = ir1[7];
              ir1[7] = (v457_data + (v419_data * v455_data));
              float v460_data = s0[80];
              float v462_data = ir1[8];
              ir1[8] = (v462_data + (v419_data * v460_data));
            }
            if (v3_lead < 9) {
              #pragma unroll
              for (int32_t v469_n1 = 0; v469_n1 < 9; ++v469_n1) {
                int32_t v470_a = 0 + v469_n1;
                float v472_data = ir1[v469_n1];
                int32_t v474_a = 0 + v469_n1;
                r1[v469_n1] = (v472_data * 13.0f);
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v480_i1 = 0; v480_i1 < 9; ++v480_i1) {
              int32_t v481_a = 0 + v480_i1;
              float v483_data = r1[v480_i1];
              int32_t v490_a = v3_lead + (v480_i1 * 9);
              glb_m0[v490_a] = v483_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

