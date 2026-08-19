// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a03a3cd0d, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_8a03a3cd0d, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8a03a3cd0d<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 12);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 12);
              float v26_data;
              {
                v26_data = __ldcg(&glb_m3[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r2[v27_a] = v26_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir1[8]{};
            int32_t v30_lead = threadIdx.x % 16;
            if (v30_lead < 12) {
              float v32_data = r0[0];
              float v33_data = s0[0];
              float v35_data = ir1[0];
              ir1[0] = (v35_data + (v32_data * v33_data));
              float v38_data = s0[12];
              float v40_data = ir1[1];
              ir1[1] = (v40_data + (v32_data * v38_data));
              float v43_data = s0[24];
              float v45_data = ir1[2];
              ir1[2] = (v45_data + (v32_data * v43_data));
              float v48_data = s0[36];
              float v50_data = ir1[3];
              ir1[3] = (v50_data + (v32_data * v48_data));
              float v53_data = s0[48];
              float v55_data = ir1[4];
              ir1[4] = (v55_data + (v32_data * v53_data));
              float v58_data = s0[60];
              float v60_data = ir1[5];
              ir1[5] = (v60_data + (v32_data * v58_data));
              float v63_data = s0[72];
              float v65_data = ir1[6];
              ir1[6] = (v65_data + (v32_data * v63_data));
              float v68_data = s0[84];
              float v70_data = ir1[7];
              ir1[7] = (v70_data + (v32_data * v68_data));
            }
            if (v30_lead < 12) {
              float v76_data = r0[1];
              float v77_data = s0[1];
              float v79_data = ir1[0];
              ir1[0] = (v79_data + (v76_data * v77_data));
              float v82_data = s0[13];
              float v84_data = ir1[1];
              ir1[1] = (v84_data + (v76_data * v82_data));
              float v87_data = s0[25];
              float v89_data = ir1[2];
              ir1[2] = (v89_data + (v76_data * v87_data));
              float v92_data = s0[37];
              float v94_data = ir1[3];
              ir1[3] = (v94_data + (v76_data * v92_data));
              float v97_data = s0[49];
              float v99_data = ir1[4];
              ir1[4] = (v99_data + (v76_data * v97_data));
              float v102_data = s0[61];
              float v104_data = ir1[5];
              ir1[5] = (v104_data + (v76_data * v102_data));
              float v107_data = s0[73];
              float v109_data = ir1[6];
              ir1[6] = (v109_data + (v76_data * v107_data));
              float v112_data = s0[85];
              float v114_data = ir1[7];
              ir1[7] = (v114_data + (v76_data * v112_data));
            }
            if (v30_lead < 12) {
              float v120_data = r0[2];
              float v121_data = s0[2];
              float v123_data = ir1[0];
              ir1[0] = (v123_data + (v120_data * v121_data));
              float v126_data = s0[14];
              float v128_data = ir1[1];
              ir1[1] = (v128_data + (v120_data * v126_data));
              float v131_data = s0[26];
              float v133_data = ir1[2];
              ir1[2] = (v133_data + (v120_data * v131_data));
              float v136_data = s0[38];
              float v138_data = ir1[3];
              ir1[3] = (v138_data + (v120_data * v136_data));
              float v141_data = s0[50];
              float v143_data = ir1[4];
              ir1[4] = (v143_data + (v120_data * v141_data));
              float v146_data = s0[62];
              float v148_data = ir1[5];
              ir1[5] = (v148_data + (v120_data * v146_data));
              float v151_data = s0[74];
              float v153_data = ir1[6];
              ir1[6] = (v153_data + (v120_data * v151_data));
              float v156_data = s0[86];
              float v158_data = ir1[7];
              ir1[7] = (v158_data + (v120_data * v156_data));
            }
            if (v30_lead < 12) {
              float v164_data = r0[3];
              float v165_data = s0[3];
              float v167_data = ir1[0];
              ir1[0] = (v167_data + (v164_data * v165_data));
              float v170_data = s0[15];
              float v172_data = ir1[1];
              ir1[1] = (v172_data + (v164_data * v170_data));
              float v175_data = s0[27];
              float v177_data = ir1[2];
              ir1[2] = (v177_data + (v164_data * v175_data));
              float v180_data = s0[39];
              float v182_data = ir1[3];
              ir1[3] = (v182_data + (v164_data * v180_data));
              float v185_data = s0[51];
              float v187_data = ir1[4];
              ir1[4] = (v187_data + (v164_data * v185_data));
              float v190_data = s0[63];
              float v192_data = ir1[5];
              ir1[5] = (v192_data + (v164_data * v190_data));
              float v195_data = s0[75];
              float v197_data = ir1[6];
              ir1[6] = (v197_data + (v164_data * v195_data));
              float v200_data = s0[87];
              float v202_data = ir1[7];
              ir1[7] = (v202_data + (v164_data * v200_data));
            }
            if (v30_lead < 12) {
              float v208_data = r0[4];
              float v209_data = s0[4];
              float v211_data = ir1[0];
              ir1[0] = (v211_data + (v208_data * v209_data));
              float v214_data = s0[16];
              float v216_data = ir1[1];
              ir1[1] = (v216_data + (v208_data * v214_data));
              float v219_data = s0[28];
              float v221_data = ir1[2];
              ir1[2] = (v221_data + (v208_data * v219_data));
              float v224_data = s0[40];
              float v226_data = ir1[3];
              ir1[3] = (v226_data + (v208_data * v224_data));
              float v229_data = s0[52];
              float v231_data = ir1[4];
              ir1[4] = (v231_data + (v208_data * v229_data));
              float v234_data = s0[64];
              float v236_data = ir1[5];
              ir1[5] = (v236_data + (v208_data * v234_data));
              float v239_data = s0[76];
              float v241_data = ir1[6];
              ir1[6] = (v241_data + (v208_data * v239_data));
              float v244_data = s0[88];
              float v246_data = ir1[7];
              ir1[7] = (v246_data + (v208_data * v244_data));
            }
            if (v30_lead < 12) {
              float v252_data = r0[5];
              float v253_data = s0[5];
              float v255_data = ir1[0];
              ir1[0] = (v255_data + (v252_data * v253_data));
              float v258_data = s0[17];
              float v260_data = ir1[1];
              ir1[1] = (v260_data + (v252_data * v258_data));
              float v263_data = s0[29];
              float v265_data = ir1[2];
              ir1[2] = (v265_data + (v252_data * v263_data));
              float v268_data = s0[41];
              float v270_data = ir1[3];
              ir1[3] = (v270_data + (v252_data * v268_data));
              float v273_data = s0[53];
              float v275_data = ir1[4];
              ir1[4] = (v275_data + (v252_data * v273_data));
              float v278_data = s0[65];
              float v280_data = ir1[5];
              ir1[5] = (v280_data + (v252_data * v278_data));
              float v283_data = s0[77];
              float v285_data = ir1[6];
              ir1[6] = (v285_data + (v252_data * v283_data));
              float v288_data = s0[89];
              float v290_data = ir1[7];
              ir1[7] = (v290_data + (v252_data * v288_data));
            }
            if (v30_lead < 12) {
              float v296_data = r0[6];
              float v297_data = s0[6];
              float v299_data = ir1[0];
              ir1[0] = (v299_data + (v296_data * v297_data));
              float v302_data = s0[18];
              float v304_data = ir1[1];
              ir1[1] = (v304_data + (v296_data * v302_data));
              float v307_data = s0[30];
              float v309_data = ir1[2];
              ir1[2] = (v309_data + (v296_data * v307_data));
              float v312_data = s0[42];
              float v314_data = ir1[3];
              ir1[3] = (v314_data + (v296_data * v312_data));
              float v317_data = s0[54];
              float v319_data = ir1[4];
              ir1[4] = (v319_data + (v296_data * v317_data));
              float v322_data = s0[66];
              float v324_data = ir1[5];
              ir1[5] = (v324_data + (v296_data * v322_data));
              float v327_data = s0[78];
              float v329_data = ir1[6];
              ir1[6] = (v329_data + (v296_data * v327_data));
              float v332_data = s0[90];
              float v334_data = ir1[7];
              ir1[7] = (v334_data + (v296_data * v332_data));
            }
            if (v30_lead < 12) {
              float v340_data = r0[7];
              float v341_data = s0[7];
              float v343_data = ir1[0];
              ir1[0] = (v343_data + (v340_data * v341_data));
              float v346_data = s0[19];
              float v348_data = ir1[1];
              ir1[1] = (v348_data + (v340_data * v346_data));
              float v351_data = s0[31];
              float v353_data = ir1[2];
              ir1[2] = (v353_data + (v340_data * v351_data));
              float v356_data = s0[43];
              float v358_data = ir1[3];
              ir1[3] = (v358_data + (v340_data * v356_data));
              float v361_data = s0[55];
              float v363_data = ir1[4];
              ir1[4] = (v363_data + (v340_data * v361_data));
              float v366_data = s0[67];
              float v368_data = ir1[5];
              ir1[5] = (v368_data + (v340_data * v366_data));
              float v371_data = s0[79];
              float v373_data = ir1[6];
              ir1[6] = (v373_data + (v340_data * v371_data));
              float v376_data = s0[91];
              float v378_data = ir1[7];
              ir1[7] = (v378_data + (v340_data * v376_data));
            }
            if (v30_lead < 12) {
              float v384_data = r0[8];
              float v385_data = s0[8];
              float v387_data = ir1[0];
              ir1[0] = (v387_data + (v384_data * v385_data));
              float v390_data = s0[20];
              float v392_data = ir1[1];
              ir1[1] = (v392_data + (v384_data * v390_data));
              float v395_data = s0[32];
              float v397_data = ir1[2];
              ir1[2] = (v397_data + (v384_data * v395_data));
              float v400_data = s0[44];
              float v402_data = ir1[3];
              ir1[3] = (v402_data + (v384_data * v400_data));
              float v405_data = s0[56];
              float v407_data = ir1[4];
              ir1[4] = (v407_data + (v384_data * v405_data));
              float v410_data = s0[68];
              float v412_data = ir1[5];
              ir1[5] = (v412_data + (v384_data * v410_data));
              float v415_data = s0[80];
              float v417_data = ir1[6];
              ir1[6] = (v417_data + (v384_data * v415_data));
              float v420_data = s0[92];
              float v422_data = ir1[7];
              ir1[7] = (v422_data + (v384_data * v420_data));
            }
            if (v30_lead < 12) {
              float v428_data = r0[9];
              float v429_data = s0[9];
              float v431_data = ir1[0];
              ir1[0] = (v431_data + (v428_data * v429_data));
              float v434_data = s0[21];
              float v436_data = ir1[1];
              ir1[1] = (v436_data + (v428_data * v434_data));
              float v439_data = s0[33];
              float v441_data = ir1[2];
              ir1[2] = (v441_data + (v428_data * v439_data));
              float v444_data = s0[45];
              float v446_data = ir1[3];
              ir1[3] = (v446_data + (v428_data * v444_data));
              float v449_data = s0[57];
              float v451_data = ir1[4];
              ir1[4] = (v451_data + (v428_data * v449_data));
              float v454_data = s0[69];
              float v456_data = ir1[5];
              ir1[5] = (v456_data + (v428_data * v454_data));
              float v459_data = s0[81];
              float v461_data = ir1[6];
              ir1[6] = (v461_data + (v428_data * v459_data));
              float v464_data = s0[93];
              float v466_data = ir1[7];
              ir1[7] = (v466_data + (v428_data * v464_data));
            }
            if (v30_lead < 12) {
              float v472_data = r0[10];
              float v473_data = s0[10];
              float v475_data = ir1[0];
              ir1[0] = (v475_data + (v472_data * v473_data));
              float v478_data = s0[22];
              float v480_data = ir1[1];
              ir1[1] = (v480_data + (v472_data * v478_data));
              float v483_data = s0[34];
              float v485_data = ir1[2];
              ir1[2] = (v485_data + (v472_data * v483_data));
              float v488_data = s0[46];
              float v490_data = ir1[3];
              ir1[3] = (v490_data + (v472_data * v488_data));
              float v493_data = s0[58];
              float v495_data = ir1[4];
              ir1[4] = (v495_data + (v472_data * v493_data));
              float v498_data = s0[70];
              float v500_data = ir1[5];
              ir1[5] = (v500_data + (v472_data * v498_data));
              float v503_data = s0[82];
              float v505_data = ir1[6];
              ir1[6] = (v505_data + (v472_data * v503_data));
              float v508_data = s0[94];
              float v510_data = ir1[7];
              ir1[7] = (v510_data + (v472_data * v508_data));
            }
            if (v30_lead < 12) {
              float v516_data = r0[11];
              float v517_data = s0[11];
              float v519_data = ir1[0];
              ir1[0] = (v519_data + (v516_data * v517_data));
              float v522_data = s0[23];
              float v524_data = ir1[1];
              ir1[1] = (v524_data + (v516_data * v522_data));
              float v527_data = s0[35];
              float v529_data = ir1[2];
              ir1[2] = (v529_data + (v516_data * v527_data));
              float v532_data = s0[47];
              float v534_data = ir1[3];
              ir1[3] = (v534_data + (v516_data * v532_data));
              float v537_data = s0[59];
              float v539_data = ir1[4];
              ir1[4] = (v539_data + (v516_data * v537_data));
              float v542_data = s0[71];
              float v544_data = ir1[5];
              ir1[5] = (v544_data + (v516_data * v542_data));
              float v547_data = s0[83];
              float v549_data = ir1[6];
              ir1[6] = (v549_data + (v516_data * v547_data));
              float v552_data = s0[95];
              float v554_data = ir1[7];
              ir1[7] = (v554_data + (v516_data * v552_data));
            }
            if (v30_lead < 12) {
              #pragma unroll
              for (int32_t v560_n1 = 0; v560_n1 < 8; ++v560_n1) {
                int32_t v561_a = 0 + v560_n1;
                float v563_data = ir1[v560_n1];
                int32_t v564_a = 0 + v560_n1;
                r1[v564_a] = v563_data;
              }
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          int32_t v567_lead = threadIdx.x % 16;
          if (v567_lead < 12) {
            #pragma unroll
            for (int32_t v569_i1 = 0; v569_i1 < 12; ++v569_i1) {
              int32_t v576_a = v567_lead + (v569_i1 * 12);
              float v577_data;
              {
                v577_data = __ldcg(&glb_m5[v576_a]);
              }
              int32_t v578_a = 0 + v569_i1;
              r4[v578_a] = v577_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r3[8]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir3[8]{};
            int32_t v581_lead = threadIdx.x % 16;
            if (v581_lead < 12) {
              float v583_data = r2[0];
              float v584_data = s1[0];
              float v586_data = ir3[0];
              ir3[0] = (v586_data + (v583_data * v584_data));
              float v589_data = s1[12];
              float v591_data = ir3[1];
              ir3[1] = (v591_data + (v583_data * v589_data));
              float v594_data = s1[24];
              float v596_data = ir3[2];
              ir3[2] = (v596_data + (v583_data * v594_data));
              float v599_data = s1[36];
              float v601_data = ir3[3];
              ir3[3] = (v601_data + (v583_data * v599_data));
              float v604_data = s1[48];
              float v606_data = ir3[4];
              ir3[4] = (v606_data + (v583_data * v604_data));
              float v609_data = s1[60];
              float v611_data = ir3[5];
              ir3[5] = (v611_data + (v583_data * v609_data));
              float v614_data = s1[72];
              float v616_data = ir3[6];
              ir3[6] = (v616_data + (v583_data * v614_data));
              float v619_data = s1[84];
              float v621_data = ir3[7];
              ir3[7] = (v621_data + (v583_data * v619_data));
            }
            if (v581_lead < 12) {
              float v627_data = r2[1];
              float v628_data = s1[1];
              float v630_data = ir3[0];
              ir3[0] = (v630_data + (v627_data * v628_data));
              float v633_data = s1[13];
              float v635_data = ir3[1];
              ir3[1] = (v635_data + (v627_data * v633_data));
              float v638_data = s1[25];
              float v640_data = ir3[2];
              ir3[2] = (v640_data + (v627_data * v638_data));
              float v643_data = s1[37];
              float v645_data = ir3[3];
              ir3[3] = (v645_data + (v627_data * v643_data));
              float v648_data = s1[49];
              float v650_data = ir3[4];
              ir3[4] = (v650_data + (v627_data * v648_data));
              float v653_data = s1[61];
              float v655_data = ir3[5];
              ir3[5] = (v655_data + (v627_data * v653_data));
              float v658_data = s1[73];
              float v660_data = ir3[6];
              ir3[6] = (v660_data + (v627_data * v658_data));
              float v663_data = s1[85];
              float v665_data = ir3[7];
              ir3[7] = (v665_data + (v627_data * v663_data));
            }
            if (v581_lead < 12) {
              float v671_data = r2[2];
              float v672_data = s1[2];
              float v674_data = ir3[0];
              ir3[0] = (v674_data + (v671_data * v672_data));
              float v677_data = s1[14];
              float v679_data = ir3[1];
              ir3[1] = (v679_data + (v671_data * v677_data));
              float v682_data = s1[26];
              float v684_data = ir3[2];
              ir3[2] = (v684_data + (v671_data * v682_data));
              float v687_data = s1[38];
              float v689_data = ir3[3];
              ir3[3] = (v689_data + (v671_data * v687_data));
              float v692_data = s1[50];
              float v694_data = ir3[4];
              ir3[4] = (v694_data + (v671_data * v692_data));
              float v697_data = s1[62];
              float v699_data = ir3[5];
              ir3[5] = (v699_data + (v671_data * v697_data));
              float v702_data = s1[74];
              float v704_data = ir3[6];
              ir3[6] = (v704_data + (v671_data * v702_data));
              float v707_data = s1[86];
              float v709_data = ir3[7];
              ir3[7] = (v709_data + (v671_data * v707_data));
            }
            if (v581_lead < 12) {
              float v715_data = r2[3];
              float v716_data = s1[3];
              float v718_data = ir3[0];
              ir3[0] = (v718_data + (v715_data * v716_data));
              float v721_data = s1[15];
              float v723_data = ir3[1];
              ir3[1] = (v723_data + (v715_data * v721_data));
              float v726_data = s1[27];
              float v728_data = ir3[2];
              ir3[2] = (v728_data + (v715_data * v726_data));
              float v731_data = s1[39];
              float v733_data = ir3[3];
              ir3[3] = (v733_data + (v715_data * v731_data));
              float v736_data = s1[51];
              float v738_data = ir3[4];
              ir3[4] = (v738_data + (v715_data * v736_data));
              float v741_data = s1[63];
              float v743_data = ir3[5];
              ir3[5] = (v743_data + (v715_data * v741_data));
              float v746_data = s1[75];
              float v748_data = ir3[6];
              ir3[6] = (v748_data + (v715_data * v746_data));
              float v751_data = s1[87];
              float v753_data = ir3[7];
              ir3[7] = (v753_data + (v715_data * v751_data));
            }
            if (v581_lead < 12) {
              float v759_data = r2[4];
              float v760_data = s1[4];
              float v762_data = ir3[0];
              ir3[0] = (v762_data + (v759_data * v760_data));
              float v765_data = s1[16];
              float v767_data = ir3[1];
              ir3[1] = (v767_data + (v759_data * v765_data));
              float v770_data = s1[28];
              float v772_data = ir3[2];
              ir3[2] = (v772_data + (v759_data * v770_data));
              float v775_data = s1[40];
              float v777_data = ir3[3];
              ir3[3] = (v777_data + (v759_data * v775_data));
              float v780_data = s1[52];
              float v782_data = ir3[4];
              ir3[4] = (v782_data + (v759_data * v780_data));
              float v785_data = s1[64];
              float v787_data = ir3[5];
              ir3[5] = (v787_data + (v759_data * v785_data));
              float v790_data = s1[76];
              float v792_data = ir3[6];
              ir3[6] = (v792_data + (v759_data * v790_data));
              float v795_data = s1[88];
              float v797_data = ir3[7];
              ir3[7] = (v797_data + (v759_data * v795_data));
            }
            if (v581_lead < 12) {
              float v803_data = r2[5];
              float v804_data = s1[5];
              float v806_data = ir3[0];
              ir3[0] = (v806_data + (v803_data * v804_data));
              float v809_data = s1[17];
              float v811_data = ir3[1];
              ir3[1] = (v811_data + (v803_data * v809_data));
              float v814_data = s1[29];
              float v816_data = ir3[2];
              ir3[2] = (v816_data + (v803_data * v814_data));
              float v819_data = s1[41];
              float v821_data = ir3[3];
              ir3[3] = (v821_data + (v803_data * v819_data));
              float v824_data = s1[53];
              float v826_data = ir3[4];
              ir3[4] = (v826_data + (v803_data * v824_data));
              float v829_data = s1[65];
              float v831_data = ir3[5];
              ir3[5] = (v831_data + (v803_data * v829_data));
              float v834_data = s1[77];
              float v836_data = ir3[6];
              ir3[6] = (v836_data + (v803_data * v834_data));
              float v839_data = s1[89];
              float v841_data = ir3[7];
              ir3[7] = (v841_data + (v803_data * v839_data));
            }
            if (v581_lead < 12) {
              float v847_data = r2[6];
              float v848_data = s1[6];
              float v850_data = ir3[0];
              ir3[0] = (v850_data + (v847_data * v848_data));
              float v853_data = s1[18];
              float v855_data = ir3[1];
              ir3[1] = (v855_data + (v847_data * v853_data));
              float v858_data = s1[30];
              float v860_data = ir3[2];
              ir3[2] = (v860_data + (v847_data * v858_data));
              float v863_data = s1[42];
              float v865_data = ir3[3];
              ir3[3] = (v865_data + (v847_data * v863_data));
              float v868_data = s1[54];
              float v870_data = ir3[4];
              ir3[4] = (v870_data + (v847_data * v868_data));
              float v873_data = s1[66];
              float v875_data = ir3[5];
              ir3[5] = (v875_data + (v847_data * v873_data));
              float v878_data = s1[78];
              float v880_data = ir3[6];
              ir3[6] = (v880_data + (v847_data * v878_data));
              float v883_data = s1[90];
              float v885_data = ir3[7];
              ir3[7] = (v885_data + (v847_data * v883_data));
            }
            if (v581_lead < 12) {
              float v891_data = r2[7];
              float v892_data = s1[7];
              float v894_data = ir3[0];
              ir3[0] = (v894_data + (v891_data * v892_data));
              float v897_data = s1[19];
              float v899_data = ir3[1];
              ir3[1] = (v899_data + (v891_data * v897_data));
              float v902_data = s1[31];
              float v904_data = ir3[2];
              ir3[2] = (v904_data + (v891_data * v902_data));
              float v907_data = s1[43];
              float v909_data = ir3[3];
              ir3[3] = (v909_data + (v891_data * v907_data));
              float v912_data = s1[55];
              float v914_data = ir3[4];
              ir3[4] = (v914_data + (v891_data * v912_data));
              float v917_data = s1[67];
              float v919_data = ir3[5];
              ir3[5] = (v919_data + (v891_data * v917_data));
              float v922_data = s1[79];
              float v924_data = ir3[6];
              ir3[6] = (v924_data + (v891_data * v922_data));
              float v927_data = s1[91];
              float v929_data = ir3[7];
              ir3[7] = (v929_data + (v891_data * v927_data));
            }
            if (v581_lead < 12) {
              float v935_data = r2[8];
              float v936_data = s1[8];
              float v938_data = ir3[0];
              ir3[0] = (v938_data + (v935_data * v936_data));
              float v941_data = s1[20];
              float v943_data = ir3[1];
              ir3[1] = (v943_data + (v935_data * v941_data));
              float v946_data = s1[32];
              float v948_data = ir3[2];
              ir3[2] = (v948_data + (v935_data * v946_data));
              float v951_data = s1[44];
              float v953_data = ir3[3];
              ir3[3] = (v953_data + (v935_data * v951_data));
              float v956_data = s1[56];
              float v958_data = ir3[4];
              ir3[4] = (v958_data + (v935_data * v956_data));
              float v961_data = s1[68];
              float v963_data = ir3[5];
              ir3[5] = (v963_data + (v935_data * v961_data));
              float v966_data = s1[80];
              float v968_data = ir3[6];
              ir3[6] = (v968_data + (v935_data * v966_data));
              float v971_data = s1[92];
              float v973_data = ir3[7];
              ir3[7] = (v973_data + (v935_data * v971_data));
            }
            if (v581_lead < 12) {
              float v979_data = r2[9];
              float v980_data = s1[9];
              float v982_data = ir3[0];
              ir3[0] = (v982_data + (v979_data * v980_data));
              float v985_data = s1[21];
              float v987_data = ir3[1];
              ir3[1] = (v987_data + (v979_data * v985_data));
              float v990_data = s1[33];
              float v992_data = ir3[2];
              ir3[2] = (v992_data + (v979_data * v990_data));
              float v995_data = s1[45];
              float v997_data = ir3[3];
              ir3[3] = (v997_data + (v979_data * v995_data));
              float v1000_data = s1[57];
              float v1002_data = ir3[4];
              ir3[4] = (v1002_data + (v979_data * v1000_data));
              float v1005_data = s1[69];
              float v1007_data = ir3[5];
              ir3[5] = (v1007_data + (v979_data * v1005_data));
              float v1010_data = s1[81];
              float v1012_data = ir3[6];
              ir3[6] = (v1012_data + (v979_data * v1010_data));
              float v1015_data = s1[93];
              float v1017_data = ir3[7];
              ir3[7] = (v1017_data + (v979_data * v1015_data));
            }
            if (v581_lead < 12) {
              float v1023_data = r2[10];
              float v1024_data = s1[10];
              float v1026_data = ir3[0];
              ir3[0] = (v1026_data + (v1023_data * v1024_data));
              float v1029_data = s1[22];
              float v1031_data = ir3[1];
              ir3[1] = (v1031_data + (v1023_data * v1029_data));
              float v1034_data = s1[34];
              float v1036_data = ir3[2];
              ir3[2] = (v1036_data + (v1023_data * v1034_data));
              float v1039_data = s1[46];
              float v1041_data = ir3[3];
              ir3[3] = (v1041_data + (v1023_data * v1039_data));
              float v1044_data = s1[58];
              float v1046_data = ir3[4];
              ir3[4] = (v1046_data + (v1023_data * v1044_data));
              float v1049_data = s1[70];
              float v1051_data = ir3[5];
              ir3[5] = (v1051_data + (v1023_data * v1049_data));
              float v1054_data = s1[82];
              float v1056_data = ir3[6];
              ir3[6] = (v1056_data + (v1023_data * v1054_data));
              float v1059_data = s1[94];
              float v1061_data = ir3[7];
              ir3[7] = (v1061_data + (v1023_data * v1059_data));
            }
            if (v581_lead < 12) {
              float v1067_data = r2[11];
              float v1068_data = s1[11];
              float v1070_data = ir3[0];
              ir3[0] = (v1070_data + (v1067_data * v1068_data));
              float v1073_data = s1[23];
              float v1075_data = ir3[1];
              ir3[1] = (v1075_data + (v1067_data * v1073_data));
              float v1078_data = s1[35];
              float v1080_data = ir3[2];
              ir3[2] = (v1080_data + (v1067_data * v1078_data));
              float v1083_data = s1[47];
              float v1085_data = ir3[3];
              ir3[3] = (v1085_data + (v1067_data * v1083_data));
              float v1088_data = s1[59];
              float v1090_data = ir3[4];
              ir3[4] = (v1090_data + (v1067_data * v1088_data));
              float v1093_data = s1[71];
              float v1095_data = ir3[5];
              ir3[5] = (v1095_data + (v1067_data * v1093_data));
              float v1098_data = s1[83];
              float v1100_data = ir3[6];
              ir3[6] = (v1100_data + (v1067_data * v1098_data));
              float v1103_data = s1[95];
              float v1105_data = ir3[7];
              ir3[7] = (v1105_data + (v1067_data * v1103_data));
            }
            if (v581_lead < 12) {
              #pragma unroll
              for (int32_t v1111_n1 = 0; v1111_n1 < 8; ++v1111_n1) {
                int32_t v1112_a = 0 + v1111_n1;
                float v1114_data = ir3[v1111_n1];
                int32_t v1115_a = 0 + v1111_n1;
                float v1117_data = r1[v1111_n1];
                int32_t v1119_a = 0 + v1111_n1;
                r3[v1119_a] = (v1117_data + v1114_data);
              }
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          {
            // s2 = load{g>s}(glb_m6[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          int32_t v1122_lead = threadIdx.x % 16;
          if (v1122_lead < 12) {
            #pragma unroll
            for (int32_t v1124_i1 = 0; v1124_i1 < 12; ++v1124_i1) {
              int32_t v1131_a = v1122_lead + (v1124_i1 * 12);
              float v1132_data;
              {
                v1132_data = __ldcg(&glb_m7[v1131_a]);
              }
              int32_t v1133_a = 0 + v1124_i1;
              r6[v1133_a] = v1132_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r5[8]{};
          __syncwarp();
          {
            // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            int32_t v1136_lead = threadIdx.x % 16;
            if (v1136_lead < 12) {
              float v1138_data = r4[0];
              float v1139_data = s2[0];
              float v1141_data = ir5[0];
              ir5[0] = (v1141_data + (v1138_data * v1139_data));
              float v1144_data = s2[12];
              float v1146_data = ir5[1];
              ir5[1] = (v1146_data + (v1138_data * v1144_data));
              float v1149_data = s2[24];
              float v1151_data = ir5[2];
              ir5[2] = (v1151_data + (v1138_data * v1149_data));
              float v1154_data = s2[36];
              float v1156_data = ir5[3];
              ir5[3] = (v1156_data + (v1138_data * v1154_data));
              float v1159_data = s2[48];
              float v1161_data = ir5[4];
              ir5[4] = (v1161_data + (v1138_data * v1159_data));
              float v1164_data = s2[60];
              float v1166_data = ir5[5];
              ir5[5] = (v1166_data + (v1138_data * v1164_data));
              float v1169_data = s2[72];
              float v1171_data = ir5[6];
              ir5[6] = (v1171_data + (v1138_data * v1169_data));
              float v1174_data = s2[84];
              float v1176_data = ir5[7];
              ir5[7] = (v1176_data + (v1138_data * v1174_data));
            }
            if (v1136_lead < 12) {
              float v1182_data = r4[1];
              float v1183_data = s2[1];
              float v1185_data = ir5[0];
              ir5[0] = (v1185_data + (v1182_data * v1183_data));
              float v1188_data = s2[13];
              float v1190_data = ir5[1];
              ir5[1] = (v1190_data + (v1182_data * v1188_data));
              float v1193_data = s2[25];
              float v1195_data = ir5[2];
              ir5[2] = (v1195_data + (v1182_data * v1193_data));
              float v1198_data = s2[37];
              float v1200_data = ir5[3];
              ir5[3] = (v1200_data + (v1182_data * v1198_data));
              float v1203_data = s2[49];
              float v1205_data = ir5[4];
              ir5[4] = (v1205_data + (v1182_data * v1203_data));
              float v1208_data = s2[61];
              float v1210_data = ir5[5];
              ir5[5] = (v1210_data + (v1182_data * v1208_data));
              float v1213_data = s2[73];
              float v1215_data = ir5[6];
              ir5[6] = (v1215_data + (v1182_data * v1213_data));
              float v1218_data = s2[85];
              float v1220_data = ir5[7];
              ir5[7] = (v1220_data + (v1182_data * v1218_data));
            }
            if (v1136_lead < 12) {
              float v1226_data = r4[2];
              float v1227_data = s2[2];
              float v1229_data = ir5[0];
              ir5[0] = (v1229_data + (v1226_data * v1227_data));
              float v1232_data = s2[14];
              float v1234_data = ir5[1];
              ir5[1] = (v1234_data + (v1226_data * v1232_data));
              float v1237_data = s2[26];
              float v1239_data = ir5[2];
              ir5[2] = (v1239_data + (v1226_data * v1237_data));
              float v1242_data = s2[38];
              float v1244_data = ir5[3];
              ir5[3] = (v1244_data + (v1226_data * v1242_data));
              float v1247_data = s2[50];
              float v1249_data = ir5[4];
              ir5[4] = (v1249_data + (v1226_data * v1247_data));
              float v1252_data = s2[62];
              float v1254_data = ir5[5];
              ir5[5] = (v1254_data + (v1226_data * v1252_data));
              float v1257_data = s2[74];
              float v1259_data = ir5[6];
              ir5[6] = (v1259_data + (v1226_data * v1257_data));
              float v1262_data = s2[86];
              float v1264_data = ir5[7];
              ir5[7] = (v1264_data + (v1226_data * v1262_data));
            }
            if (v1136_lead < 12) {
              float v1270_data = r4[3];
              float v1271_data = s2[3];
              float v1273_data = ir5[0];
              ir5[0] = (v1273_data + (v1270_data * v1271_data));
              float v1276_data = s2[15];
              float v1278_data = ir5[1];
              ir5[1] = (v1278_data + (v1270_data * v1276_data));
              float v1281_data = s2[27];
              float v1283_data = ir5[2];
              ir5[2] = (v1283_data + (v1270_data * v1281_data));
              float v1286_data = s2[39];
              float v1288_data = ir5[3];
              ir5[3] = (v1288_data + (v1270_data * v1286_data));
              float v1291_data = s2[51];
              float v1293_data = ir5[4];
              ir5[4] = (v1293_data + (v1270_data * v1291_data));
              float v1296_data = s2[63];
              float v1298_data = ir5[5];
              ir5[5] = (v1298_data + (v1270_data * v1296_data));
              float v1301_data = s2[75];
              float v1303_data = ir5[6];
              ir5[6] = (v1303_data + (v1270_data * v1301_data));
              float v1306_data = s2[87];
              float v1308_data = ir5[7];
              ir5[7] = (v1308_data + (v1270_data * v1306_data));
            }
            if (v1136_lead < 12) {
              float v1314_data = r4[4];
              float v1315_data = s2[4];
              float v1317_data = ir5[0];
              ir5[0] = (v1317_data + (v1314_data * v1315_data));
              float v1320_data = s2[16];
              float v1322_data = ir5[1];
              ir5[1] = (v1322_data + (v1314_data * v1320_data));
              float v1325_data = s2[28];
              float v1327_data = ir5[2];
              ir5[2] = (v1327_data + (v1314_data * v1325_data));
              float v1330_data = s2[40];
              float v1332_data = ir5[3];
              ir5[3] = (v1332_data + (v1314_data * v1330_data));
              float v1335_data = s2[52];
              float v1337_data = ir5[4];
              ir5[4] = (v1337_data + (v1314_data * v1335_data));
              float v1340_data = s2[64];
              float v1342_data = ir5[5];
              ir5[5] = (v1342_data + (v1314_data * v1340_data));
              float v1345_data = s2[76];
              float v1347_data = ir5[6];
              ir5[6] = (v1347_data + (v1314_data * v1345_data));
              float v1350_data = s2[88];
              float v1352_data = ir5[7];
              ir5[7] = (v1352_data + (v1314_data * v1350_data));
            }
            if (v1136_lead < 12) {
              float v1358_data = r4[5];
              float v1359_data = s2[5];
              float v1361_data = ir5[0];
              ir5[0] = (v1361_data + (v1358_data * v1359_data));
              float v1364_data = s2[17];
              float v1366_data = ir5[1];
              ir5[1] = (v1366_data + (v1358_data * v1364_data));
              float v1369_data = s2[29];
              float v1371_data = ir5[2];
              ir5[2] = (v1371_data + (v1358_data * v1369_data));
              float v1374_data = s2[41];
              float v1376_data = ir5[3];
              ir5[3] = (v1376_data + (v1358_data * v1374_data));
              float v1379_data = s2[53];
              float v1381_data = ir5[4];
              ir5[4] = (v1381_data + (v1358_data * v1379_data));
              float v1384_data = s2[65];
              float v1386_data = ir5[5];
              ir5[5] = (v1386_data + (v1358_data * v1384_data));
              float v1389_data = s2[77];
              float v1391_data = ir5[6];
              ir5[6] = (v1391_data + (v1358_data * v1389_data));
              float v1394_data = s2[89];
              float v1396_data = ir5[7];
              ir5[7] = (v1396_data + (v1358_data * v1394_data));
            }
            if (v1136_lead < 12) {
              float v1402_data = r4[6];
              float v1403_data = s2[6];
              float v1405_data = ir5[0];
              ir5[0] = (v1405_data + (v1402_data * v1403_data));
              float v1408_data = s2[18];
              float v1410_data = ir5[1];
              ir5[1] = (v1410_data + (v1402_data * v1408_data));
              float v1413_data = s2[30];
              float v1415_data = ir5[2];
              ir5[2] = (v1415_data + (v1402_data * v1413_data));
              float v1418_data = s2[42];
              float v1420_data = ir5[3];
              ir5[3] = (v1420_data + (v1402_data * v1418_data));
              float v1423_data = s2[54];
              float v1425_data = ir5[4];
              ir5[4] = (v1425_data + (v1402_data * v1423_data));
              float v1428_data = s2[66];
              float v1430_data = ir5[5];
              ir5[5] = (v1430_data + (v1402_data * v1428_data));
              float v1433_data = s2[78];
              float v1435_data = ir5[6];
              ir5[6] = (v1435_data + (v1402_data * v1433_data));
              float v1438_data = s2[90];
              float v1440_data = ir5[7];
              ir5[7] = (v1440_data + (v1402_data * v1438_data));
            }
            if (v1136_lead < 12) {
              float v1446_data = r4[7];
              float v1447_data = s2[7];
              float v1449_data = ir5[0];
              ir5[0] = (v1449_data + (v1446_data * v1447_data));
              float v1452_data = s2[19];
              float v1454_data = ir5[1];
              ir5[1] = (v1454_data + (v1446_data * v1452_data));
              float v1457_data = s2[31];
              float v1459_data = ir5[2];
              ir5[2] = (v1459_data + (v1446_data * v1457_data));
              float v1462_data = s2[43];
              float v1464_data = ir5[3];
              ir5[3] = (v1464_data + (v1446_data * v1462_data));
              float v1467_data = s2[55];
              float v1469_data = ir5[4];
              ir5[4] = (v1469_data + (v1446_data * v1467_data));
              float v1472_data = s2[67];
              float v1474_data = ir5[5];
              ir5[5] = (v1474_data + (v1446_data * v1472_data));
              float v1477_data = s2[79];
              float v1479_data = ir5[6];
              ir5[6] = (v1479_data + (v1446_data * v1477_data));
              float v1482_data = s2[91];
              float v1484_data = ir5[7];
              ir5[7] = (v1484_data + (v1446_data * v1482_data));
            }
            if (v1136_lead < 12) {
              float v1490_data = r4[8];
              float v1491_data = s2[8];
              float v1493_data = ir5[0];
              ir5[0] = (v1493_data + (v1490_data * v1491_data));
              float v1496_data = s2[20];
              float v1498_data = ir5[1];
              ir5[1] = (v1498_data + (v1490_data * v1496_data));
              float v1501_data = s2[32];
              float v1503_data = ir5[2];
              ir5[2] = (v1503_data + (v1490_data * v1501_data));
              float v1506_data = s2[44];
              float v1508_data = ir5[3];
              ir5[3] = (v1508_data + (v1490_data * v1506_data));
              float v1511_data = s2[56];
              float v1513_data = ir5[4];
              ir5[4] = (v1513_data + (v1490_data * v1511_data));
              float v1516_data = s2[68];
              float v1518_data = ir5[5];
              ir5[5] = (v1518_data + (v1490_data * v1516_data));
              float v1521_data = s2[80];
              float v1523_data = ir5[6];
              ir5[6] = (v1523_data + (v1490_data * v1521_data));
              float v1526_data = s2[92];
              float v1528_data = ir5[7];
              ir5[7] = (v1528_data + (v1490_data * v1526_data));
            }
            if (v1136_lead < 12) {
              float v1534_data = r4[9];
              float v1535_data = s2[9];
              float v1537_data = ir5[0];
              ir5[0] = (v1537_data + (v1534_data * v1535_data));
              float v1540_data = s2[21];
              float v1542_data = ir5[1];
              ir5[1] = (v1542_data + (v1534_data * v1540_data));
              float v1545_data = s2[33];
              float v1547_data = ir5[2];
              ir5[2] = (v1547_data + (v1534_data * v1545_data));
              float v1550_data = s2[45];
              float v1552_data = ir5[3];
              ir5[3] = (v1552_data + (v1534_data * v1550_data));
              float v1555_data = s2[57];
              float v1557_data = ir5[4];
              ir5[4] = (v1557_data + (v1534_data * v1555_data));
              float v1560_data = s2[69];
              float v1562_data = ir5[5];
              ir5[5] = (v1562_data + (v1534_data * v1560_data));
              float v1565_data = s2[81];
              float v1567_data = ir5[6];
              ir5[6] = (v1567_data + (v1534_data * v1565_data));
              float v1570_data = s2[93];
              float v1572_data = ir5[7];
              ir5[7] = (v1572_data + (v1534_data * v1570_data));
            }
            if (v1136_lead < 12) {
              float v1578_data = r4[10];
              float v1579_data = s2[10];
              float v1581_data = ir5[0];
              ir5[0] = (v1581_data + (v1578_data * v1579_data));
              float v1584_data = s2[22];
              float v1586_data = ir5[1];
              ir5[1] = (v1586_data + (v1578_data * v1584_data));
              float v1589_data = s2[34];
              float v1591_data = ir5[2];
              ir5[2] = (v1591_data + (v1578_data * v1589_data));
              float v1594_data = s2[46];
              float v1596_data = ir5[3];
              ir5[3] = (v1596_data + (v1578_data * v1594_data));
              float v1599_data = s2[58];
              float v1601_data = ir5[4];
              ir5[4] = (v1601_data + (v1578_data * v1599_data));
              float v1604_data = s2[70];
              float v1606_data = ir5[5];
              ir5[5] = (v1606_data + (v1578_data * v1604_data));
              float v1609_data = s2[82];
              float v1611_data = ir5[6];
              ir5[6] = (v1611_data + (v1578_data * v1609_data));
              float v1614_data = s2[94];
              float v1616_data = ir5[7];
              ir5[7] = (v1616_data + (v1578_data * v1614_data));
            }
            if (v1136_lead < 12) {
              float v1622_data = r4[11];
              float v1623_data = s2[11];
              float v1625_data = ir5[0];
              ir5[0] = (v1625_data + (v1622_data * v1623_data));
              float v1628_data = s2[23];
              float v1630_data = ir5[1];
              ir5[1] = (v1630_data + (v1622_data * v1628_data));
              float v1633_data = s2[35];
              float v1635_data = ir5[2];
              ir5[2] = (v1635_data + (v1622_data * v1633_data));
              float v1638_data = s2[47];
              float v1640_data = ir5[3];
              ir5[3] = (v1640_data + (v1622_data * v1638_data));
              float v1643_data = s2[59];
              float v1645_data = ir5[4];
              ir5[4] = (v1645_data + (v1622_data * v1643_data));
              float v1648_data = s2[71];
              float v1650_data = ir5[5];
              ir5[5] = (v1650_data + (v1622_data * v1648_data));
              float v1653_data = s2[83];
              float v1655_data = ir5[6];
              ir5[6] = (v1655_data + (v1622_data * v1653_data));
              float v1658_data = s2[95];
              float v1660_data = ir5[7];
              ir5[7] = (v1660_data + (v1622_data * v1658_data));
            }
            if (v1136_lead < 12) {
              #pragma unroll
              for (int32_t v1666_n1 = 0; v1666_n1 < 8; ++v1666_n1) {
                int32_t v1667_a = 0 + v1666_n1;
                float v1669_data = ir5[v1666_n1];
                int32_t v1670_a = 0 + v1666_n1;
                float v1672_data = r3[v1666_n1];
                int32_t v1674_a = 0 + v1666_n1;
                r5[v1674_a] = (v1672_data + v1669_data);
              }
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          {
            // s3 = load{g>s}(glb_m8[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r7[8]{};
          __syncwarp();
          {
            // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir7[8]{};
            int32_t v1677_lead = threadIdx.x % 16;
            if (v1677_lead < 12) {
              float v1679_data = r6[0];
              float v1680_data = s3[0];
              float v1682_data = ir7[0];
              ir7[0] = (v1682_data + (v1679_data * v1680_data));
              float v1685_data = s3[12];
              float v1687_data = ir7[1];
              ir7[1] = (v1687_data + (v1679_data * v1685_data));
              float v1690_data = s3[24];
              float v1692_data = ir7[2];
              ir7[2] = (v1692_data + (v1679_data * v1690_data));
              float v1695_data = s3[36];
              float v1697_data = ir7[3];
              ir7[3] = (v1697_data + (v1679_data * v1695_data));
              float v1700_data = s3[48];
              float v1702_data = ir7[4];
              ir7[4] = (v1702_data + (v1679_data * v1700_data));
              float v1705_data = s3[60];
              float v1707_data = ir7[5];
              ir7[5] = (v1707_data + (v1679_data * v1705_data));
              float v1710_data = s3[72];
              float v1712_data = ir7[6];
              ir7[6] = (v1712_data + (v1679_data * v1710_data));
              float v1715_data = s3[84];
              float v1717_data = ir7[7];
              ir7[7] = (v1717_data + (v1679_data * v1715_data));
            }
            if (v1677_lead < 12) {
              float v1723_data = r6[1];
              float v1724_data = s3[1];
              float v1726_data = ir7[0];
              ir7[0] = (v1726_data + (v1723_data * v1724_data));
              float v1729_data = s3[13];
              float v1731_data = ir7[1];
              ir7[1] = (v1731_data + (v1723_data * v1729_data));
              float v1734_data = s3[25];
              float v1736_data = ir7[2];
              ir7[2] = (v1736_data + (v1723_data * v1734_data));
              float v1739_data = s3[37];
              float v1741_data = ir7[3];
              ir7[3] = (v1741_data + (v1723_data * v1739_data));
              float v1744_data = s3[49];
              float v1746_data = ir7[4];
              ir7[4] = (v1746_data + (v1723_data * v1744_data));
              float v1749_data = s3[61];
              float v1751_data = ir7[5];
              ir7[5] = (v1751_data + (v1723_data * v1749_data));
              float v1754_data = s3[73];
              float v1756_data = ir7[6];
              ir7[6] = (v1756_data + (v1723_data * v1754_data));
              float v1759_data = s3[85];
              float v1761_data = ir7[7];
              ir7[7] = (v1761_data + (v1723_data * v1759_data));
            }
            if (v1677_lead < 12) {
              float v1767_data = r6[2];
              float v1768_data = s3[2];
              float v1770_data = ir7[0];
              ir7[0] = (v1770_data + (v1767_data * v1768_data));
              float v1773_data = s3[14];
              float v1775_data = ir7[1];
              ir7[1] = (v1775_data + (v1767_data * v1773_data));
              float v1778_data = s3[26];
              float v1780_data = ir7[2];
              ir7[2] = (v1780_data + (v1767_data * v1778_data));
              float v1783_data = s3[38];
              float v1785_data = ir7[3];
              ir7[3] = (v1785_data + (v1767_data * v1783_data));
              float v1788_data = s3[50];
              float v1790_data = ir7[4];
              ir7[4] = (v1790_data + (v1767_data * v1788_data));
              float v1793_data = s3[62];
              float v1795_data = ir7[5];
              ir7[5] = (v1795_data + (v1767_data * v1793_data));
              float v1798_data = s3[74];
              float v1800_data = ir7[6];
              ir7[6] = (v1800_data + (v1767_data * v1798_data));
              float v1803_data = s3[86];
              float v1805_data = ir7[7];
              ir7[7] = (v1805_data + (v1767_data * v1803_data));
            }
            if (v1677_lead < 12) {
              float v1811_data = r6[3];
              float v1812_data = s3[3];
              float v1814_data = ir7[0];
              ir7[0] = (v1814_data + (v1811_data * v1812_data));
              float v1817_data = s3[15];
              float v1819_data = ir7[1];
              ir7[1] = (v1819_data + (v1811_data * v1817_data));
              float v1822_data = s3[27];
              float v1824_data = ir7[2];
              ir7[2] = (v1824_data + (v1811_data * v1822_data));
              float v1827_data = s3[39];
              float v1829_data = ir7[3];
              ir7[3] = (v1829_data + (v1811_data * v1827_data));
              float v1832_data = s3[51];
              float v1834_data = ir7[4];
              ir7[4] = (v1834_data + (v1811_data * v1832_data));
              float v1837_data = s3[63];
              float v1839_data = ir7[5];
              ir7[5] = (v1839_data + (v1811_data * v1837_data));
              float v1842_data = s3[75];
              float v1844_data = ir7[6];
              ir7[6] = (v1844_data + (v1811_data * v1842_data));
              float v1847_data = s3[87];
              float v1849_data = ir7[7];
              ir7[7] = (v1849_data + (v1811_data * v1847_data));
            }
            if (v1677_lead < 12) {
              float v1855_data = r6[4];
              float v1856_data = s3[4];
              float v1858_data = ir7[0];
              ir7[0] = (v1858_data + (v1855_data * v1856_data));
              float v1861_data = s3[16];
              float v1863_data = ir7[1];
              ir7[1] = (v1863_data + (v1855_data * v1861_data));
              float v1866_data = s3[28];
              float v1868_data = ir7[2];
              ir7[2] = (v1868_data + (v1855_data * v1866_data));
              float v1871_data = s3[40];
              float v1873_data = ir7[3];
              ir7[3] = (v1873_data + (v1855_data * v1871_data));
              float v1876_data = s3[52];
              float v1878_data = ir7[4];
              ir7[4] = (v1878_data + (v1855_data * v1876_data));
              float v1881_data = s3[64];
              float v1883_data = ir7[5];
              ir7[5] = (v1883_data + (v1855_data * v1881_data));
              float v1886_data = s3[76];
              float v1888_data = ir7[6];
              ir7[6] = (v1888_data + (v1855_data * v1886_data));
              float v1891_data = s3[88];
              float v1893_data = ir7[7];
              ir7[7] = (v1893_data + (v1855_data * v1891_data));
            }
            if (v1677_lead < 12) {
              float v1899_data = r6[5];
              float v1900_data = s3[5];
              float v1902_data = ir7[0];
              ir7[0] = (v1902_data + (v1899_data * v1900_data));
              float v1905_data = s3[17];
              float v1907_data = ir7[1];
              ir7[1] = (v1907_data + (v1899_data * v1905_data));
              float v1910_data = s3[29];
              float v1912_data = ir7[2];
              ir7[2] = (v1912_data + (v1899_data * v1910_data));
              float v1915_data = s3[41];
              float v1917_data = ir7[3];
              ir7[3] = (v1917_data + (v1899_data * v1915_data));
              float v1920_data = s3[53];
              float v1922_data = ir7[4];
              ir7[4] = (v1922_data + (v1899_data * v1920_data));
              float v1925_data = s3[65];
              float v1927_data = ir7[5];
              ir7[5] = (v1927_data + (v1899_data * v1925_data));
              float v1930_data = s3[77];
              float v1932_data = ir7[6];
              ir7[6] = (v1932_data + (v1899_data * v1930_data));
              float v1935_data = s3[89];
              float v1937_data = ir7[7];
              ir7[7] = (v1937_data + (v1899_data * v1935_data));
            }
            if (v1677_lead < 12) {
              float v1943_data = r6[6];
              float v1944_data = s3[6];
              float v1946_data = ir7[0];
              ir7[0] = (v1946_data + (v1943_data * v1944_data));
              float v1949_data = s3[18];
              float v1951_data = ir7[1];
              ir7[1] = (v1951_data + (v1943_data * v1949_data));
              float v1954_data = s3[30];
              float v1956_data = ir7[2];
              ir7[2] = (v1956_data + (v1943_data * v1954_data));
              float v1959_data = s3[42];
              float v1961_data = ir7[3];
              ir7[3] = (v1961_data + (v1943_data * v1959_data));
              float v1964_data = s3[54];
              float v1966_data = ir7[4];
              ir7[4] = (v1966_data + (v1943_data * v1964_data));
              float v1969_data = s3[66];
              float v1971_data = ir7[5];
              ir7[5] = (v1971_data + (v1943_data * v1969_data));
              float v1974_data = s3[78];
              float v1976_data = ir7[6];
              ir7[6] = (v1976_data + (v1943_data * v1974_data));
              float v1979_data = s3[90];
              float v1981_data = ir7[7];
              ir7[7] = (v1981_data + (v1943_data * v1979_data));
            }
            if (v1677_lead < 12) {
              float v1987_data = r6[7];
              float v1988_data = s3[7];
              float v1990_data = ir7[0];
              ir7[0] = (v1990_data + (v1987_data * v1988_data));
              float v1993_data = s3[19];
              float v1995_data = ir7[1];
              ir7[1] = (v1995_data + (v1987_data * v1993_data));
              float v1998_data = s3[31];
              float v2000_data = ir7[2];
              ir7[2] = (v2000_data + (v1987_data * v1998_data));
              float v2003_data = s3[43];
              float v2005_data = ir7[3];
              ir7[3] = (v2005_data + (v1987_data * v2003_data));
              float v2008_data = s3[55];
              float v2010_data = ir7[4];
              ir7[4] = (v2010_data + (v1987_data * v2008_data));
              float v2013_data = s3[67];
              float v2015_data = ir7[5];
              ir7[5] = (v2015_data + (v1987_data * v2013_data));
              float v2018_data = s3[79];
              float v2020_data = ir7[6];
              ir7[6] = (v2020_data + (v1987_data * v2018_data));
              float v2023_data = s3[91];
              float v2025_data = ir7[7];
              ir7[7] = (v2025_data + (v1987_data * v2023_data));
            }
            if (v1677_lead < 12) {
              float v2031_data = r6[8];
              float v2032_data = s3[8];
              float v2034_data = ir7[0];
              ir7[0] = (v2034_data + (v2031_data * v2032_data));
              float v2037_data = s3[20];
              float v2039_data = ir7[1];
              ir7[1] = (v2039_data + (v2031_data * v2037_data));
              float v2042_data = s3[32];
              float v2044_data = ir7[2];
              ir7[2] = (v2044_data + (v2031_data * v2042_data));
              float v2047_data = s3[44];
              float v2049_data = ir7[3];
              ir7[3] = (v2049_data + (v2031_data * v2047_data));
              float v2052_data = s3[56];
              float v2054_data = ir7[4];
              ir7[4] = (v2054_data + (v2031_data * v2052_data));
              float v2057_data = s3[68];
              float v2059_data = ir7[5];
              ir7[5] = (v2059_data + (v2031_data * v2057_data));
              float v2062_data = s3[80];
              float v2064_data = ir7[6];
              ir7[6] = (v2064_data + (v2031_data * v2062_data));
              float v2067_data = s3[92];
              float v2069_data = ir7[7];
              ir7[7] = (v2069_data + (v2031_data * v2067_data));
            }
            if (v1677_lead < 12) {
              float v2075_data = r6[9];
              float v2076_data = s3[9];
              float v2078_data = ir7[0];
              ir7[0] = (v2078_data + (v2075_data * v2076_data));
              float v2081_data = s3[21];
              float v2083_data = ir7[1];
              ir7[1] = (v2083_data + (v2075_data * v2081_data));
              float v2086_data = s3[33];
              float v2088_data = ir7[2];
              ir7[2] = (v2088_data + (v2075_data * v2086_data));
              float v2091_data = s3[45];
              float v2093_data = ir7[3];
              ir7[3] = (v2093_data + (v2075_data * v2091_data));
              float v2096_data = s3[57];
              float v2098_data = ir7[4];
              ir7[4] = (v2098_data + (v2075_data * v2096_data));
              float v2101_data = s3[69];
              float v2103_data = ir7[5];
              ir7[5] = (v2103_data + (v2075_data * v2101_data));
              float v2106_data = s3[81];
              float v2108_data = ir7[6];
              ir7[6] = (v2108_data + (v2075_data * v2106_data));
              float v2111_data = s3[93];
              float v2113_data = ir7[7];
              ir7[7] = (v2113_data + (v2075_data * v2111_data));
            }
            if (v1677_lead < 12) {
              float v2119_data = r6[10];
              float v2120_data = s3[10];
              float v2122_data = ir7[0];
              ir7[0] = (v2122_data + (v2119_data * v2120_data));
              float v2125_data = s3[22];
              float v2127_data = ir7[1];
              ir7[1] = (v2127_data + (v2119_data * v2125_data));
              float v2130_data = s3[34];
              float v2132_data = ir7[2];
              ir7[2] = (v2132_data + (v2119_data * v2130_data));
              float v2135_data = s3[46];
              float v2137_data = ir7[3];
              ir7[3] = (v2137_data + (v2119_data * v2135_data));
              float v2140_data = s3[58];
              float v2142_data = ir7[4];
              ir7[4] = (v2142_data + (v2119_data * v2140_data));
              float v2145_data = s3[70];
              float v2147_data = ir7[5];
              ir7[5] = (v2147_data + (v2119_data * v2145_data));
              float v2150_data = s3[82];
              float v2152_data = ir7[6];
              ir7[6] = (v2152_data + (v2119_data * v2150_data));
              float v2155_data = s3[94];
              float v2157_data = ir7[7];
              ir7[7] = (v2157_data + (v2119_data * v2155_data));
            }
            if (v1677_lead < 12) {
              float v2163_data = r6[11];
              float v2164_data = s3[11];
              float v2166_data = ir7[0];
              ir7[0] = (v2166_data + (v2163_data * v2164_data));
              float v2169_data = s3[23];
              float v2171_data = ir7[1];
              ir7[1] = (v2171_data + (v2163_data * v2169_data));
              float v2174_data = s3[35];
              float v2176_data = ir7[2];
              ir7[2] = (v2176_data + (v2163_data * v2174_data));
              float v2179_data = s3[47];
              float v2181_data = ir7[3];
              ir7[3] = (v2181_data + (v2163_data * v2179_data));
              float v2184_data = s3[59];
              float v2186_data = ir7[4];
              ir7[4] = (v2186_data + (v2163_data * v2184_data));
              float v2189_data = s3[71];
              float v2191_data = ir7[5];
              ir7[5] = (v2191_data + (v2163_data * v2189_data));
              float v2194_data = s3[83];
              float v2196_data = ir7[6];
              ir7[6] = (v2196_data + (v2163_data * v2194_data));
              float v2199_data = s3[95];
              float v2201_data = ir7[7];
              ir7[7] = (v2201_data + (v2163_data * v2199_data));
            }
            if (v1677_lead < 12) {
              #pragma unroll
              for (int32_t v2207_n1 = 0; v2207_n1 < 8; ++v2207_n1) {
                int32_t v2208_a = 0 + v2207_n1;
                float v2210_data = ir7[v2207_n1];
                int32_t v2211_a = 0 + v2207_n1;
                float v2213_data = r5[v2207_n1];
                int32_t v2215_a = 0 + v2207_n1;
                r7[v2215_a] = (v2213_data + v2210_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r7);
          int32_t v2218_lead = threadIdx.x % 16;
          if (v2218_lead < 12) {
            #pragma unroll
            for (int32_t v2220_i1 = 0; v2220_i1 < 8; ++v2220_i1) {
              int32_t v2221_a = 0 + v2220_i1;
              float v2223_data = r7[v2220_i1];
              int32_t v2230_a = v2218_lead + (v2220_i1 * 12);
              glb_m0[v2230_a] = v2223_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

