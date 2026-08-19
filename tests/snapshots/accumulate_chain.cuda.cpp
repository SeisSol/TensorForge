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
                float v562_data = ir1[v561_a];
                int32_t v563_a = 0 + v560_n1;
                r1[v563_a] = v562_data;
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
          int32_t v566_lead = threadIdx.x % 16;
          if (v566_lead < 12) {
            #pragma unroll
            for (int32_t v568_i1 = 0; v568_i1 < 12; ++v568_i1) {
              int32_t v575_a = v566_lead + (v568_i1 * 12);
              float v576_data;
              {
                v576_data = __ldcg(&glb_m5[v575_a]);
              }
              int32_t v577_a = 0 + v568_i1;
              r4[v577_a] = v576_data;
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
            int32_t v580_lead = threadIdx.x % 16;
            if (v580_lead < 12) {
              float v582_data = r2[0];
              float v583_data = s1[0];
              float v585_data = ir3[0];
              ir3[0] = (v585_data + (v582_data * v583_data));
              float v588_data = s1[12];
              float v590_data = ir3[1];
              ir3[1] = (v590_data + (v582_data * v588_data));
              float v593_data = s1[24];
              float v595_data = ir3[2];
              ir3[2] = (v595_data + (v582_data * v593_data));
              float v598_data = s1[36];
              float v600_data = ir3[3];
              ir3[3] = (v600_data + (v582_data * v598_data));
              float v603_data = s1[48];
              float v605_data = ir3[4];
              ir3[4] = (v605_data + (v582_data * v603_data));
              float v608_data = s1[60];
              float v610_data = ir3[5];
              ir3[5] = (v610_data + (v582_data * v608_data));
              float v613_data = s1[72];
              float v615_data = ir3[6];
              ir3[6] = (v615_data + (v582_data * v613_data));
              float v618_data = s1[84];
              float v620_data = ir3[7];
              ir3[7] = (v620_data + (v582_data * v618_data));
            }
            if (v580_lead < 12) {
              float v626_data = r2[1];
              float v627_data = s1[1];
              float v629_data = ir3[0];
              ir3[0] = (v629_data + (v626_data * v627_data));
              float v632_data = s1[13];
              float v634_data = ir3[1];
              ir3[1] = (v634_data + (v626_data * v632_data));
              float v637_data = s1[25];
              float v639_data = ir3[2];
              ir3[2] = (v639_data + (v626_data * v637_data));
              float v642_data = s1[37];
              float v644_data = ir3[3];
              ir3[3] = (v644_data + (v626_data * v642_data));
              float v647_data = s1[49];
              float v649_data = ir3[4];
              ir3[4] = (v649_data + (v626_data * v647_data));
              float v652_data = s1[61];
              float v654_data = ir3[5];
              ir3[5] = (v654_data + (v626_data * v652_data));
              float v657_data = s1[73];
              float v659_data = ir3[6];
              ir3[6] = (v659_data + (v626_data * v657_data));
              float v662_data = s1[85];
              float v664_data = ir3[7];
              ir3[7] = (v664_data + (v626_data * v662_data));
            }
            if (v580_lead < 12) {
              float v670_data = r2[2];
              float v671_data = s1[2];
              float v673_data = ir3[0];
              ir3[0] = (v673_data + (v670_data * v671_data));
              float v676_data = s1[14];
              float v678_data = ir3[1];
              ir3[1] = (v678_data + (v670_data * v676_data));
              float v681_data = s1[26];
              float v683_data = ir3[2];
              ir3[2] = (v683_data + (v670_data * v681_data));
              float v686_data = s1[38];
              float v688_data = ir3[3];
              ir3[3] = (v688_data + (v670_data * v686_data));
              float v691_data = s1[50];
              float v693_data = ir3[4];
              ir3[4] = (v693_data + (v670_data * v691_data));
              float v696_data = s1[62];
              float v698_data = ir3[5];
              ir3[5] = (v698_data + (v670_data * v696_data));
              float v701_data = s1[74];
              float v703_data = ir3[6];
              ir3[6] = (v703_data + (v670_data * v701_data));
              float v706_data = s1[86];
              float v708_data = ir3[7];
              ir3[7] = (v708_data + (v670_data * v706_data));
            }
            if (v580_lead < 12) {
              float v714_data = r2[3];
              float v715_data = s1[3];
              float v717_data = ir3[0];
              ir3[0] = (v717_data + (v714_data * v715_data));
              float v720_data = s1[15];
              float v722_data = ir3[1];
              ir3[1] = (v722_data + (v714_data * v720_data));
              float v725_data = s1[27];
              float v727_data = ir3[2];
              ir3[2] = (v727_data + (v714_data * v725_data));
              float v730_data = s1[39];
              float v732_data = ir3[3];
              ir3[3] = (v732_data + (v714_data * v730_data));
              float v735_data = s1[51];
              float v737_data = ir3[4];
              ir3[4] = (v737_data + (v714_data * v735_data));
              float v740_data = s1[63];
              float v742_data = ir3[5];
              ir3[5] = (v742_data + (v714_data * v740_data));
              float v745_data = s1[75];
              float v747_data = ir3[6];
              ir3[6] = (v747_data + (v714_data * v745_data));
              float v750_data = s1[87];
              float v752_data = ir3[7];
              ir3[7] = (v752_data + (v714_data * v750_data));
            }
            if (v580_lead < 12) {
              float v758_data = r2[4];
              float v759_data = s1[4];
              float v761_data = ir3[0];
              ir3[0] = (v761_data + (v758_data * v759_data));
              float v764_data = s1[16];
              float v766_data = ir3[1];
              ir3[1] = (v766_data + (v758_data * v764_data));
              float v769_data = s1[28];
              float v771_data = ir3[2];
              ir3[2] = (v771_data + (v758_data * v769_data));
              float v774_data = s1[40];
              float v776_data = ir3[3];
              ir3[3] = (v776_data + (v758_data * v774_data));
              float v779_data = s1[52];
              float v781_data = ir3[4];
              ir3[4] = (v781_data + (v758_data * v779_data));
              float v784_data = s1[64];
              float v786_data = ir3[5];
              ir3[5] = (v786_data + (v758_data * v784_data));
              float v789_data = s1[76];
              float v791_data = ir3[6];
              ir3[6] = (v791_data + (v758_data * v789_data));
              float v794_data = s1[88];
              float v796_data = ir3[7];
              ir3[7] = (v796_data + (v758_data * v794_data));
            }
            if (v580_lead < 12) {
              float v802_data = r2[5];
              float v803_data = s1[5];
              float v805_data = ir3[0];
              ir3[0] = (v805_data + (v802_data * v803_data));
              float v808_data = s1[17];
              float v810_data = ir3[1];
              ir3[1] = (v810_data + (v802_data * v808_data));
              float v813_data = s1[29];
              float v815_data = ir3[2];
              ir3[2] = (v815_data + (v802_data * v813_data));
              float v818_data = s1[41];
              float v820_data = ir3[3];
              ir3[3] = (v820_data + (v802_data * v818_data));
              float v823_data = s1[53];
              float v825_data = ir3[4];
              ir3[4] = (v825_data + (v802_data * v823_data));
              float v828_data = s1[65];
              float v830_data = ir3[5];
              ir3[5] = (v830_data + (v802_data * v828_data));
              float v833_data = s1[77];
              float v835_data = ir3[6];
              ir3[6] = (v835_data + (v802_data * v833_data));
              float v838_data = s1[89];
              float v840_data = ir3[7];
              ir3[7] = (v840_data + (v802_data * v838_data));
            }
            if (v580_lead < 12) {
              float v846_data = r2[6];
              float v847_data = s1[6];
              float v849_data = ir3[0];
              ir3[0] = (v849_data + (v846_data * v847_data));
              float v852_data = s1[18];
              float v854_data = ir3[1];
              ir3[1] = (v854_data + (v846_data * v852_data));
              float v857_data = s1[30];
              float v859_data = ir3[2];
              ir3[2] = (v859_data + (v846_data * v857_data));
              float v862_data = s1[42];
              float v864_data = ir3[3];
              ir3[3] = (v864_data + (v846_data * v862_data));
              float v867_data = s1[54];
              float v869_data = ir3[4];
              ir3[4] = (v869_data + (v846_data * v867_data));
              float v872_data = s1[66];
              float v874_data = ir3[5];
              ir3[5] = (v874_data + (v846_data * v872_data));
              float v877_data = s1[78];
              float v879_data = ir3[6];
              ir3[6] = (v879_data + (v846_data * v877_data));
              float v882_data = s1[90];
              float v884_data = ir3[7];
              ir3[7] = (v884_data + (v846_data * v882_data));
            }
            if (v580_lead < 12) {
              float v890_data = r2[7];
              float v891_data = s1[7];
              float v893_data = ir3[0];
              ir3[0] = (v893_data + (v890_data * v891_data));
              float v896_data = s1[19];
              float v898_data = ir3[1];
              ir3[1] = (v898_data + (v890_data * v896_data));
              float v901_data = s1[31];
              float v903_data = ir3[2];
              ir3[2] = (v903_data + (v890_data * v901_data));
              float v906_data = s1[43];
              float v908_data = ir3[3];
              ir3[3] = (v908_data + (v890_data * v906_data));
              float v911_data = s1[55];
              float v913_data = ir3[4];
              ir3[4] = (v913_data + (v890_data * v911_data));
              float v916_data = s1[67];
              float v918_data = ir3[5];
              ir3[5] = (v918_data + (v890_data * v916_data));
              float v921_data = s1[79];
              float v923_data = ir3[6];
              ir3[6] = (v923_data + (v890_data * v921_data));
              float v926_data = s1[91];
              float v928_data = ir3[7];
              ir3[7] = (v928_data + (v890_data * v926_data));
            }
            if (v580_lead < 12) {
              float v934_data = r2[8];
              float v935_data = s1[8];
              float v937_data = ir3[0];
              ir3[0] = (v937_data + (v934_data * v935_data));
              float v940_data = s1[20];
              float v942_data = ir3[1];
              ir3[1] = (v942_data + (v934_data * v940_data));
              float v945_data = s1[32];
              float v947_data = ir3[2];
              ir3[2] = (v947_data + (v934_data * v945_data));
              float v950_data = s1[44];
              float v952_data = ir3[3];
              ir3[3] = (v952_data + (v934_data * v950_data));
              float v955_data = s1[56];
              float v957_data = ir3[4];
              ir3[4] = (v957_data + (v934_data * v955_data));
              float v960_data = s1[68];
              float v962_data = ir3[5];
              ir3[5] = (v962_data + (v934_data * v960_data));
              float v965_data = s1[80];
              float v967_data = ir3[6];
              ir3[6] = (v967_data + (v934_data * v965_data));
              float v970_data = s1[92];
              float v972_data = ir3[7];
              ir3[7] = (v972_data + (v934_data * v970_data));
            }
            if (v580_lead < 12) {
              float v978_data = r2[9];
              float v979_data = s1[9];
              float v981_data = ir3[0];
              ir3[0] = (v981_data + (v978_data * v979_data));
              float v984_data = s1[21];
              float v986_data = ir3[1];
              ir3[1] = (v986_data + (v978_data * v984_data));
              float v989_data = s1[33];
              float v991_data = ir3[2];
              ir3[2] = (v991_data + (v978_data * v989_data));
              float v994_data = s1[45];
              float v996_data = ir3[3];
              ir3[3] = (v996_data + (v978_data * v994_data));
              float v999_data = s1[57];
              float v1001_data = ir3[4];
              ir3[4] = (v1001_data + (v978_data * v999_data));
              float v1004_data = s1[69];
              float v1006_data = ir3[5];
              ir3[5] = (v1006_data + (v978_data * v1004_data));
              float v1009_data = s1[81];
              float v1011_data = ir3[6];
              ir3[6] = (v1011_data + (v978_data * v1009_data));
              float v1014_data = s1[93];
              float v1016_data = ir3[7];
              ir3[7] = (v1016_data + (v978_data * v1014_data));
            }
            if (v580_lead < 12) {
              float v1022_data = r2[10];
              float v1023_data = s1[10];
              float v1025_data = ir3[0];
              ir3[0] = (v1025_data + (v1022_data * v1023_data));
              float v1028_data = s1[22];
              float v1030_data = ir3[1];
              ir3[1] = (v1030_data + (v1022_data * v1028_data));
              float v1033_data = s1[34];
              float v1035_data = ir3[2];
              ir3[2] = (v1035_data + (v1022_data * v1033_data));
              float v1038_data = s1[46];
              float v1040_data = ir3[3];
              ir3[3] = (v1040_data + (v1022_data * v1038_data));
              float v1043_data = s1[58];
              float v1045_data = ir3[4];
              ir3[4] = (v1045_data + (v1022_data * v1043_data));
              float v1048_data = s1[70];
              float v1050_data = ir3[5];
              ir3[5] = (v1050_data + (v1022_data * v1048_data));
              float v1053_data = s1[82];
              float v1055_data = ir3[6];
              ir3[6] = (v1055_data + (v1022_data * v1053_data));
              float v1058_data = s1[94];
              float v1060_data = ir3[7];
              ir3[7] = (v1060_data + (v1022_data * v1058_data));
            }
            if (v580_lead < 12) {
              float v1066_data = r2[11];
              float v1067_data = s1[11];
              float v1069_data = ir3[0];
              ir3[0] = (v1069_data + (v1066_data * v1067_data));
              float v1072_data = s1[23];
              float v1074_data = ir3[1];
              ir3[1] = (v1074_data + (v1066_data * v1072_data));
              float v1077_data = s1[35];
              float v1079_data = ir3[2];
              ir3[2] = (v1079_data + (v1066_data * v1077_data));
              float v1082_data = s1[47];
              float v1084_data = ir3[3];
              ir3[3] = (v1084_data + (v1066_data * v1082_data));
              float v1087_data = s1[59];
              float v1089_data = ir3[4];
              ir3[4] = (v1089_data + (v1066_data * v1087_data));
              float v1092_data = s1[71];
              float v1094_data = ir3[5];
              ir3[5] = (v1094_data + (v1066_data * v1092_data));
              float v1097_data = s1[83];
              float v1099_data = ir3[6];
              ir3[6] = (v1099_data + (v1066_data * v1097_data));
              float v1102_data = s1[95];
              float v1104_data = ir3[7];
              ir3[7] = (v1104_data + (v1066_data * v1102_data));
            }
            if (v580_lead < 12) {
              #pragma unroll
              for (int32_t v1110_n1 = 0; v1110_n1 < 8; ++v1110_n1) {
                int32_t v1111_a = 0 + v1110_n1;
                float v1112_data = ir3[v1111_a];
                int32_t v1113_a = 0 + v1110_n1;
                float v1114_data = r1[v1113_a];
                int32_t v1116_a = 0 + v1110_n1;
                r3[v1116_a] = (v1114_data + v1112_data);
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
          int32_t v1119_lead = threadIdx.x % 16;
          if (v1119_lead < 12) {
            #pragma unroll
            for (int32_t v1121_i1 = 0; v1121_i1 < 12; ++v1121_i1) {
              int32_t v1128_a = v1119_lead + (v1121_i1 * 12);
              float v1129_data;
              {
                v1129_data = __ldcg(&glb_m7[v1128_a]);
              }
              int32_t v1130_a = 0 + v1121_i1;
              r6[v1130_a] = v1129_data;
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
            int32_t v1133_lead = threadIdx.x % 16;
            if (v1133_lead < 12) {
              float v1135_data = r4[0];
              float v1136_data = s2[0];
              float v1138_data = ir5[0];
              ir5[0] = (v1138_data + (v1135_data * v1136_data));
              float v1141_data = s2[12];
              float v1143_data = ir5[1];
              ir5[1] = (v1143_data + (v1135_data * v1141_data));
              float v1146_data = s2[24];
              float v1148_data = ir5[2];
              ir5[2] = (v1148_data + (v1135_data * v1146_data));
              float v1151_data = s2[36];
              float v1153_data = ir5[3];
              ir5[3] = (v1153_data + (v1135_data * v1151_data));
              float v1156_data = s2[48];
              float v1158_data = ir5[4];
              ir5[4] = (v1158_data + (v1135_data * v1156_data));
              float v1161_data = s2[60];
              float v1163_data = ir5[5];
              ir5[5] = (v1163_data + (v1135_data * v1161_data));
              float v1166_data = s2[72];
              float v1168_data = ir5[6];
              ir5[6] = (v1168_data + (v1135_data * v1166_data));
              float v1171_data = s2[84];
              float v1173_data = ir5[7];
              ir5[7] = (v1173_data + (v1135_data * v1171_data));
            }
            if (v1133_lead < 12) {
              float v1179_data = r4[1];
              float v1180_data = s2[1];
              float v1182_data = ir5[0];
              ir5[0] = (v1182_data + (v1179_data * v1180_data));
              float v1185_data = s2[13];
              float v1187_data = ir5[1];
              ir5[1] = (v1187_data + (v1179_data * v1185_data));
              float v1190_data = s2[25];
              float v1192_data = ir5[2];
              ir5[2] = (v1192_data + (v1179_data * v1190_data));
              float v1195_data = s2[37];
              float v1197_data = ir5[3];
              ir5[3] = (v1197_data + (v1179_data * v1195_data));
              float v1200_data = s2[49];
              float v1202_data = ir5[4];
              ir5[4] = (v1202_data + (v1179_data * v1200_data));
              float v1205_data = s2[61];
              float v1207_data = ir5[5];
              ir5[5] = (v1207_data + (v1179_data * v1205_data));
              float v1210_data = s2[73];
              float v1212_data = ir5[6];
              ir5[6] = (v1212_data + (v1179_data * v1210_data));
              float v1215_data = s2[85];
              float v1217_data = ir5[7];
              ir5[7] = (v1217_data + (v1179_data * v1215_data));
            }
            if (v1133_lead < 12) {
              float v1223_data = r4[2];
              float v1224_data = s2[2];
              float v1226_data = ir5[0];
              ir5[0] = (v1226_data + (v1223_data * v1224_data));
              float v1229_data = s2[14];
              float v1231_data = ir5[1];
              ir5[1] = (v1231_data + (v1223_data * v1229_data));
              float v1234_data = s2[26];
              float v1236_data = ir5[2];
              ir5[2] = (v1236_data + (v1223_data * v1234_data));
              float v1239_data = s2[38];
              float v1241_data = ir5[3];
              ir5[3] = (v1241_data + (v1223_data * v1239_data));
              float v1244_data = s2[50];
              float v1246_data = ir5[4];
              ir5[4] = (v1246_data + (v1223_data * v1244_data));
              float v1249_data = s2[62];
              float v1251_data = ir5[5];
              ir5[5] = (v1251_data + (v1223_data * v1249_data));
              float v1254_data = s2[74];
              float v1256_data = ir5[6];
              ir5[6] = (v1256_data + (v1223_data * v1254_data));
              float v1259_data = s2[86];
              float v1261_data = ir5[7];
              ir5[7] = (v1261_data + (v1223_data * v1259_data));
            }
            if (v1133_lead < 12) {
              float v1267_data = r4[3];
              float v1268_data = s2[3];
              float v1270_data = ir5[0];
              ir5[0] = (v1270_data + (v1267_data * v1268_data));
              float v1273_data = s2[15];
              float v1275_data = ir5[1];
              ir5[1] = (v1275_data + (v1267_data * v1273_data));
              float v1278_data = s2[27];
              float v1280_data = ir5[2];
              ir5[2] = (v1280_data + (v1267_data * v1278_data));
              float v1283_data = s2[39];
              float v1285_data = ir5[3];
              ir5[3] = (v1285_data + (v1267_data * v1283_data));
              float v1288_data = s2[51];
              float v1290_data = ir5[4];
              ir5[4] = (v1290_data + (v1267_data * v1288_data));
              float v1293_data = s2[63];
              float v1295_data = ir5[5];
              ir5[5] = (v1295_data + (v1267_data * v1293_data));
              float v1298_data = s2[75];
              float v1300_data = ir5[6];
              ir5[6] = (v1300_data + (v1267_data * v1298_data));
              float v1303_data = s2[87];
              float v1305_data = ir5[7];
              ir5[7] = (v1305_data + (v1267_data * v1303_data));
            }
            if (v1133_lead < 12) {
              float v1311_data = r4[4];
              float v1312_data = s2[4];
              float v1314_data = ir5[0];
              ir5[0] = (v1314_data + (v1311_data * v1312_data));
              float v1317_data = s2[16];
              float v1319_data = ir5[1];
              ir5[1] = (v1319_data + (v1311_data * v1317_data));
              float v1322_data = s2[28];
              float v1324_data = ir5[2];
              ir5[2] = (v1324_data + (v1311_data * v1322_data));
              float v1327_data = s2[40];
              float v1329_data = ir5[3];
              ir5[3] = (v1329_data + (v1311_data * v1327_data));
              float v1332_data = s2[52];
              float v1334_data = ir5[4];
              ir5[4] = (v1334_data + (v1311_data * v1332_data));
              float v1337_data = s2[64];
              float v1339_data = ir5[5];
              ir5[5] = (v1339_data + (v1311_data * v1337_data));
              float v1342_data = s2[76];
              float v1344_data = ir5[6];
              ir5[6] = (v1344_data + (v1311_data * v1342_data));
              float v1347_data = s2[88];
              float v1349_data = ir5[7];
              ir5[7] = (v1349_data + (v1311_data * v1347_data));
            }
            if (v1133_lead < 12) {
              float v1355_data = r4[5];
              float v1356_data = s2[5];
              float v1358_data = ir5[0];
              ir5[0] = (v1358_data + (v1355_data * v1356_data));
              float v1361_data = s2[17];
              float v1363_data = ir5[1];
              ir5[1] = (v1363_data + (v1355_data * v1361_data));
              float v1366_data = s2[29];
              float v1368_data = ir5[2];
              ir5[2] = (v1368_data + (v1355_data * v1366_data));
              float v1371_data = s2[41];
              float v1373_data = ir5[3];
              ir5[3] = (v1373_data + (v1355_data * v1371_data));
              float v1376_data = s2[53];
              float v1378_data = ir5[4];
              ir5[4] = (v1378_data + (v1355_data * v1376_data));
              float v1381_data = s2[65];
              float v1383_data = ir5[5];
              ir5[5] = (v1383_data + (v1355_data * v1381_data));
              float v1386_data = s2[77];
              float v1388_data = ir5[6];
              ir5[6] = (v1388_data + (v1355_data * v1386_data));
              float v1391_data = s2[89];
              float v1393_data = ir5[7];
              ir5[7] = (v1393_data + (v1355_data * v1391_data));
            }
            if (v1133_lead < 12) {
              float v1399_data = r4[6];
              float v1400_data = s2[6];
              float v1402_data = ir5[0];
              ir5[0] = (v1402_data + (v1399_data * v1400_data));
              float v1405_data = s2[18];
              float v1407_data = ir5[1];
              ir5[1] = (v1407_data + (v1399_data * v1405_data));
              float v1410_data = s2[30];
              float v1412_data = ir5[2];
              ir5[2] = (v1412_data + (v1399_data * v1410_data));
              float v1415_data = s2[42];
              float v1417_data = ir5[3];
              ir5[3] = (v1417_data + (v1399_data * v1415_data));
              float v1420_data = s2[54];
              float v1422_data = ir5[4];
              ir5[4] = (v1422_data + (v1399_data * v1420_data));
              float v1425_data = s2[66];
              float v1427_data = ir5[5];
              ir5[5] = (v1427_data + (v1399_data * v1425_data));
              float v1430_data = s2[78];
              float v1432_data = ir5[6];
              ir5[6] = (v1432_data + (v1399_data * v1430_data));
              float v1435_data = s2[90];
              float v1437_data = ir5[7];
              ir5[7] = (v1437_data + (v1399_data * v1435_data));
            }
            if (v1133_lead < 12) {
              float v1443_data = r4[7];
              float v1444_data = s2[7];
              float v1446_data = ir5[0];
              ir5[0] = (v1446_data + (v1443_data * v1444_data));
              float v1449_data = s2[19];
              float v1451_data = ir5[1];
              ir5[1] = (v1451_data + (v1443_data * v1449_data));
              float v1454_data = s2[31];
              float v1456_data = ir5[2];
              ir5[2] = (v1456_data + (v1443_data * v1454_data));
              float v1459_data = s2[43];
              float v1461_data = ir5[3];
              ir5[3] = (v1461_data + (v1443_data * v1459_data));
              float v1464_data = s2[55];
              float v1466_data = ir5[4];
              ir5[4] = (v1466_data + (v1443_data * v1464_data));
              float v1469_data = s2[67];
              float v1471_data = ir5[5];
              ir5[5] = (v1471_data + (v1443_data * v1469_data));
              float v1474_data = s2[79];
              float v1476_data = ir5[6];
              ir5[6] = (v1476_data + (v1443_data * v1474_data));
              float v1479_data = s2[91];
              float v1481_data = ir5[7];
              ir5[7] = (v1481_data + (v1443_data * v1479_data));
            }
            if (v1133_lead < 12) {
              float v1487_data = r4[8];
              float v1488_data = s2[8];
              float v1490_data = ir5[0];
              ir5[0] = (v1490_data + (v1487_data * v1488_data));
              float v1493_data = s2[20];
              float v1495_data = ir5[1];
              ir5[1] = (v1495_data + (v1487_data * v1493_data));
              float v1498_data = s2[32];
              float v1500_data = ir5[2];
              ir5[2] = (v1500_data + (v1487_data * v1498_data));
              float v1503_data = s2[44];
              float v1505_data = ir5[3];
              ir5[3] = (v1505_data + (v1487_data * v1503_data));
              float v1508_data = s2[56];
              float v1510_data = ir5[4];
              ir5[4] = (v1510_data + (v1487_data * v1508_data));
              float v1513_data = s2[68];
              float v1515_data = ir5[5];
              ir5[5] = (v1515_data + (v1487_data * v1513_data));
              float v1518_data = s2[80];
              float v1520_data = ir5[6];
              ir5[6] = (v1520_data + (v1487_data * v1518_data));
              float v1523_data = s2[92];
              float v1525_data = ir5[7];
              ir5[7] = (v1525_data + (v1487_data * v1523_data));
            }
            if (v1133_lead < 12) {
              float v1531_data = r4[9];
              float v1532_data = s2[9];
              float v1534_data = ir5[0];
              ir5[0] = (v1534_data + (v1531_data * v1532_data));
              float v1537_data = s2[21];
              float v1539_data = ir5[1];
              ir5[1] = (v1539_data + (v1531_data * v1537_data));
              float v1542_data = s2[33];
              float v1544_data = ir5[2];
              ir5[2] = (v1544_data + (v1531_data * v1542_data));
              float v1547_data = s2[45];
              float v1549_data = ir5[3];
              ir5[3] = (v1549_data + (v1531_data * v1547_data));
              float v1552_data = s2[57];
              float v1554_data = ir5[4];
              ir5[4] = (v1554_data + (v1531_data * v1552_data));
              float v1557_data = s2[69];
              float v1559_data = ir5[5];
              ir5[5] = (v1559_data + (v1531_data * v1557_data));
              float v1562_data = s2[81];
              float v1564_data = ir5[6];
              ir5[6] = (v1564_data + (v1531_data * v1562_data));
              float v1567_data = s2[93];
              float v1569_data = ir5[7];
              ir5[7] = (v1569_data + (v1531_data * v1567_data));
            }
            if (v1133_lead < 12) {
              float v1575_data = r4[10];
              float v1576_data = s2[10];
              float v1578_data = ir5[0];
              ir5[0] = (v1578_data + (v1575_data * v1576_data));
              float v1581_data = s2[22];
              float v1583_data = ir5[1];
              ir5[1] = (v1583_data + (v1575_data * v1581_data));
              float v1586_data = s2[34];
              float v1588_data = ir5[2];
              ir5[2] = (v1588_data + (v1575_data * v1586_data));
              float v1591_data = s2[46];
              float v1593_data = ir5[3];
              ir5[3] = (v1593_data + (v1575_data * v1591_data));
              float v1596_data = s2[58];
              float v1598_data = ir5[4];
              ir5[4] = (v1598_data + (v1575_data * v1596_data));
              float v1601_data = s2[70];
              float v1603_data = ir5[5];
              ir5[5] = (v1603_data + (v1575_data * v1601_data));
              float v1606_data = s2[82];
              float v1608_data = ir5[6];
              ir5[6] = (v1608_data + (v1575_data * v1606_data));
              float v1611_data = s2[94];
              float v1613_data = ir5[7];
              ir5[7] = (v1613_data + (v1575_data * v1611_data));
            }
            if (v1133_lead < 12) {
              float v1619_data = r4[11];
              float v1620_data = s2[11];
              float v1622_data = ir5[0];
              ir5[0] = (v1622_data + (v1619_data * v1620_data));
              float v1625_data = s2[23];
              float v1627_data = ir5[1];
              ir5[1] = (v1627_data + (v1619_data * v1625_data));
              float v1630_data = s2[35];
              float v1632_data = ir5[2];
              ir5[2] = (v1632_data + (v1619_data * v1630_data));
              float v1635_data = s2[47];
              float v1637_data = ir5[3];
              ir5[3] = (v1637_data + (v1619_data * v1635_data));
              float v1640_data = s2[59];
              float v1642_data = ir5[4];
              ir5[4] = (v1642_data + (v1619_data * v1640_data));
              float v1645_data = s2[71];
              float v1647_data = ir5[5];
              ir5[5] = (v1647_data + (v1619_data * v1645_data));
              float v1650_data = s2[83];
              float v1652_data = ir5[6];
              ir5[6] = (v1652_data + (v1619_data * v1650_data));
              float v1655_data = s2[95];
              float v1657_data = ir5[7];
              ir5[7] = (v1657_data + (v1619_data * v1655_data));
            }
            if (v1133_lead < 12) {
              #pragma unroll
              for (int32_t v1663_n1 = 0; v1663_n1 < 8; ++v1663_n1) {
                int32_t v1664_a = 0 + v1663_n1;
                float v1665_data = ir5[v1664_a];
                int32_t v1666_a = 0 + v1663_n1;
                float v1667_data = r3[v1666_a];
                int32_t v1669_a = 0 + v1663_n1;
                r5[v1669_a] = (v1667_data + v1665_data);
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
            int32_t v1672_lead = threadIdx.x % 16;
            if (v1672_lead < 12) {
              float v1674_data = r6[0];
              float v1675_data = s3[0];
              float v1677_data = ir7[0];
              ir7[0] = (v1677_data + (v1674_data * v1675_data));
              float v1680_data = s3[12];
              float v1682_data = ir7[1];
              ir7[1] = (v1682_data + (v1674_data * v1680_data));
              float v1685_data = s3[24];
              float v1687_data = ir7[2];
              ir7[2] = (v1687_data + (v1674_data * v1685_data));
              float v1690_data = s3[36];
              float v1692_data = ir7[3];
              ir7[3] = (v1692_data + (v1674_data * v1690_data));
              float v1695_data = s3[48];
              float v1697_data = ir7[4];
              ir7[4] = (v1697_data + (v1674_data * v1695_data));
              float v1700_data = s3[60];
              float v1702_data = ir7[5];
              ir7[5] = (v1702_data + (v1674_data * v1700_data));
              float v1705_data = s3[72];
              float v1707_data = ir7[6];
              ir7[6] = (v1707_data + (v1674_data * v1705_data));
              float v1710_data = s3[84];
              float v1712_data = ir7[7];
              ir7[7] = (v1712_data + (v1674_data * v1710_data));
            }
            if (v1672_lead < 12) {
              float v1718_data = r6[1];
              float v1719_data = s3[1];
              float v1721_data = ir7[0];
              ir7[0] = (v1721_data + (v1718_data * v1719_data));
              float v1724_data = s3[13];
              float v1726_data = ir7[1];
              ir7[1] = (v1726_data + (v1718_data * v1724_data));
              float v1729_data = s3[25];
              float v1731_data = ir7[2];
              ir7[2] = (v1731_data + (v1718_data * v1729_data));
              float v1734_data = s3[37];
              float v1736_data = ir7[3];
              ir7[3] = (v1736_data + (v1718_data * v1734_data));
              float v1739_data = s3[49];
              float v1741_data = ir7[4];
              ir7[4] = (v1741_data + (v1718_data * v1739_data));
              float v1744_data = s3[61];
              float v1746_data = ir7[5];
              ir7[5] = (v1746_data + (v1718_data * v1744_data));
              float v1749_data = s3[73];
              float v1751_data = ir7[6];
              ir7[6] = (v1751_data + (v1718_data * v1749_data));
              float v1754_data = s3[85];
              float v1756_data = ir7[7];
              ir7[7] = (v1756_data + (v1718_data * v1754_data));
            }
            if (v1672_lead < 12) {
              float v1762_data = r6[2];
              float v1763_data = s3[2];
              float v1765_data = ir7[0];
              ir7[0] = (v1765_data + (v1762_data * v1763_data));
              float v1768_data = s3[14];
              float v1770_data = ir7[1];
              ir7[1] = (v1770_data + (v1762_data * v1768_data));
              float v1773_data = s3[26];
              float v1775_data = ir7[2];
              ir7[2] = (v1775_data + (v1762_data * v1773_data));
              float v1778_data = s3[38];
              float v1780_data = ir7[3];
              ir7[3] = (v1780_data + (v1762_data * v1778_data));
              float v1783_data = s3[50];
              float v1785_data = ir7[4];
              ir7[4] = (v1785_data + (v1762_data * v1783_data));
              float v1788_data = s3[62];
              float v1790_data = ir7[5];
              ir7[5] = (v1790_data + (v1762_data * v1788_data));
              float v1793_data = s3[74];
              float v1795_data = ir7[6];
              ir7[6] = (v1795_data + (v1762_data * v1793_data));
              float v1798_data = s3[86];
              float v1800_data = ir7[7];
              ir7[7] = (v1800_data + (v1762_data * v1798_data));
            }
            if (v1672_lead < 12) {
              float v1806_data = r6[3];
              float v1807_data = s3[3];
              float v1809_data = ir7[0];
              ir7[0] = (v1809_data + (v1806_data * v1807_data));
              float v1812_data = s3[15];
              float v1814_data = ir7[1];
              ir7[1] = (v1814_data + (v1806_data * v1812_data));
              float v1817_data = s3[27];
              float v1819_data = ir7[2];
              ir7[2] = (v1819_data + (v1806_data * v1817_data));
              float v1822_data = s3[39];
              float v1824_data = ir7[3];
              ir7[3] = (v1824_data + (v1806_data * v1822_data));
              float v1827_data = s3[51];
              float v1829_data = ir7[4];
              ir7[4] = (v1829_data + (v1806_data * v1827_data));
              float v1832_data = s3[63];
              float v1834_data = ir7[5];
              ir7[5] = (v1834_data + (v1806_data * v1832_data));
              float v1837_data = s3[75];
              float v1839_data = ir7[6];
              ir7[6] = (v1839_data + (v1806_data * v1837_data));
              float v1842_data = s3[87];
              float v1844_data = ir7[7];
              ir7[7] = (v1844_data + (v1806_data * v1842_data));
            }
            if (v1672_lead < 12) {
              float v1850_data = r6[4];
              float v1851_data = s3[4];
              float v1853_data = ir7[0];
              ir7[0] = (v1853_data + (v1850_data * v1851_data));
              float v1856_data = s3[16];
              float v1858_data = ir7[1];
              ir7[1] = (v1858_data + (v1850_data * v1856_data));
              float v1861_data = s3[28];
              float v1863_data = ir7[2];
              ir7[2] = (v1863_data + (v1850_data * v1861_data));
              float v1866_data = s3[40];
              float v1868_data = ir7[3];
              ir7[3] = (v1868_data + (v1850_data * v1866_data));
              float v1871_data = s3[52];
              float v1873_data = ir7[4];
              ir7[4] = (v1873_data + (v1850_data * v1871_data));
              float v1876_data = s3[64];
              float v1878_data = ir7[5];
              ir7[5] = (v1878_data + (v1850_data * v1876_data));
              float v1881_data = s3[76];
              float v1883_data = ir7[6];
              ir7[6] = (v1883_data + (v1850_data * v1881_data));
              float v1886_data = s3[88];
              float v1888_data = ir7[7];
              ir7[7] = (v1888_data + (v1850_data * v1886_data));
            }
            if (v1672_lead < 12) {
              float v1894_data = r6[5];
              float v1895_data = s3[5];
              float v1897_data = ir7[0];
              ir7[0] = (v1897_data + (v1894_data * v1895_data));
              float v1900_data = s3[17];
              float v1902_data = ir7[1];
              ir7[1] = (v1902_data + (v1894_data * v1900_data));
              float v1905_data = s3[29];
              float v1907_data = ir7[2];
              ir7[2] = (v1907_data + (v1894_data * v1905_data));
              float v1910_data = s3[41];
              float v1912_data = ir7[3];
              ir7[3] = (v1912_data + (v1894_data * v1910_data));
              float v1915_data = s3[53];
              float v1917_data = ir7[4];
              ir7[4] = (v1917_data + (v1894_data * v1915_data));
              float v1920_data = s3[65];
              float v1922_data = ir7[5];
              ir7[5] = (v1922_data + (v1894_data * v1920_data));
              float v1925_data = s3[77];
              float v1927_data = ir7[6];
              ir7[6] = (v1927_data + (v1894_data * v1925_data));
              float v1930_data = s3[89];
              float v1932_data = ir7[7];
              ir7[7] = (v1932_data + (v1894_data * v1930_data));
            }
            if (v1672_lead < 12) {
              float v1938_data = r6[6];
              float v1939_data = s3[6];
              float v1941_data = ir7[0];
              ir7[0] = (v1941_data + (v1938_data * v1939_data));
              float v1944_data = s3[18];
              float v1946_data = ir7[1];
              ir7[1] = (v1946_data + (v1938_data * v1944_data));
              float v1949_data = s3[30];
              float v1951_data = ir7[2];
              ir7[2] = (v1951_data + (v1938_data * v1949_data));
              float v1954_data = s3[42];
              float v1956_data = ir7[3];
              ir7[3] = (v1956_data + (v1938_data * v1954_data));
              float v1959_data = s3[54];
              float v1961_data = ir7[4];
              ir7[4] = (v1961_data + (v1938_data * v1959_data));
              float v1964_data = s3[66];
              float v1966_data = ir7[5];
              ir7[5] = (v1966_data + (v1938_data * v1964_data));
              float v1969_data = s3[78];
              float v1971_data = ir7[6];
              ir7[6] = (v1971_data + (v1938_data * v1969_data));
              float v1974_data = s3[90];
              float v1976_data = ir7[7];
              ir7[7] = (v1976_data + (v1938_data * v1974_data));
            }
            if (v1672_lead < 12) {
              float v1982_data = r6[7];
              float v1983_data = s3[7];
              float v1985_data = ir7[0];
              ir7[0] = (v1985_data + (v1982_data * v1983_data));
              float v1988_data = s3[19];
              float v1990_data = ir7[1];
              ir7[1] = (v1990_data + (v1982_data * v1988_data));
              float v1993_data = s3[31];
              float v1995_data = ir7[2];
              ir7[2] = (v1995_data + (v1982_data * v1993_data));
              float v1998_data = s3[43];
              float v2000_data = ir7[3];
              ir7[3] = (v2000_data + (v1982_data * v1998_data));
              float v2003_data = s3[55];
              float v2005_data = ir7[4];
              ir7[4] = (v2005_data + (v1982_data * v2003_data));
              float v2008_data = s3[67];
              float v2010_data = ir7[5];
              ir7[5] = (v2010_data + (v1982_data * v2008_data));
              float v2013_data = s3[79];
              float v2015_data = ir7[6];
              ir7[6] = (v2015_data + (v1982_data * v2013_data));
              float v2018_data = s3[91];
              float v2020_data = ir7[7];
              ir7[7] = (v2020_data + (v1982_data * v2018_data));
            }
            if (v1672_lead < 12) {
              float v2026_data = r6[8];
              float v2027_data = s3[8];
              float v2029_data = ir7[0];
              ir7[0] = (v2029_data + (v2026_data * v2027_data));
              float v2032_data = s3[20];
              float v2034_data = ir7[1];
              ir7[1] = (v2034_data + (v2026_data * v2032_data));
              float v2037_data = s3[32];
              float v2039_data = ir7[2];
              ir7[2] = (v2039_data + (v2026_data * v2037_data));
              float v2042_data = s3[44];
              float v2044_data = ir7[3];
              ir7[3] = (v2044_data + (v2026_data * v2042_data));
              float v2047_data = s3[56];
              float v2049_data = ir7[4];
              ir7[4] = (v2049_data + (v2026_data * v2047_data));
              float v2052_data = s3[68];
              float v2054_data = ir7[5];
              ir7[5] = (v2054_data + (v2026_data * v2052_data));
              float v2057_data = s3[80];
              float v2059_data = ir7[6];
              ir7[6] = (v2059_data + (v2026_data * v2057_data));
              float v2062_data = s3[92];
              float v2064_data = ir7[7];
              ir7[7] = (v2064_data + (v2026_data * v2062_data));
            }
            if (v1672_lead < 12) {
              float v2070_data = r6[9];
              float v2071_data = s3[9];
              float v2073_data = ir7[0];
              ir7[0] = (v2073_data + (v2070_data * v2071_data));
              float v2076_data = s3[21];
              float v2078_data = ir7[1];
              ir7[1] = (v2078_data + (v2070_data * v2076_data));
              float v2081_data = s3[33];
              float v2083_data = ir7[2];
              ir7[2] = (v2083_data + (v2070_data * v2081_data));
              float v2086_data = s3[45];
              float v2088_data = ir7[3];
              ir7[3] = (v2088_data + (v2070_data * v2086_data));
              float v2091_data = s3[57];
              float v2093_data = ir7[4];
              ir7[4] = (v2093_data + (v2070_data * v2091_data));
              float v2096_data = s3[69];
              float v2098_data = ir7[5];
              ir7[5] = (v2098_data + (v2070_data * v2096_data));
              float v2101_data = s3[81];
              float v2103_data = ir7[6];
              ir7[6] = (v2103_data + (v2070_data * v2101_data));
              float v2106_data = s3[93];
              float v2108_data = ir7[7];
              ir7[7] = (v2108_data + (v2070_data * v2106_data));
            }
            if (v1672_lead < 12) {
              float v2114_data = r6[10];
              float v2115_data = s3[10];
              float v2117_data = ir7[0];
              ir7[0] = (v2117_data + (v2114_data * v2115_data));
              float v2120_data = s3[22];
              float v2122_data = ir7[1];
              ir7[1] = (v2122_data + (v2114_data * v2120_data));
              float v2125_data = s3[34];
              float v2127_data = ir7[2];
              ir7[2] = (v2127_data + (v2114_data * v2125_data));
              float v2130_data = s3[46];
              float v2132_data = ir7[3];
              ir7[3] = (v2132_data + (v2114_data * v2130_data));
              float v2135_data = s3[58];
              float v2137_data = ir7[4];
              ir7[4] = (v2137_data + (v2114_data * v2135_data));
              float v2140_data = s3[70];
              float v2142_data = ir7[5];
              ir7[5] = (v2142_data + (v2114_data * v2140_data));
              float v2145_data = s3[82];
              float v2147_data = ir7[6];
              ir7[6] = (v2147_data + (v2114_data * v2145_data));
              float v2150_data = s3[94];
              float v2152_data = ir7[7];
              ir7[7] = (v2152_data + (v2114_data * v2150_data));
            }
            if (v1672_lead < 12) {
              float v2158_data = r6[11];
              float v2159_data = s3[11];
              float v2161_data = ir7[0];
              ir7[0] = (v2161_data + (v2158_data * v2159_data));
              float v2164_data = s3[23];
              float v2166_data = ir7[1];
              ir7[1] = (v2166_data + (v2158_data * v2164_data));
              float v2169_data = s3[35];
              float v2171_data = ir7[2];
              ir7[2] = (v2171_data + (v2158_data * v2169_data));
              float v2174_data = s3[47];
              float v2176_data = ir7[3];
              ir7[3] = (v2176_data + (v2158_data * v2174_data));
              float v2179_data = s3[59];
              float v2181_data = ir7[4];
              ir7[4] = (v2181_data + (v2158_data * v2179_data));
              float v2184_data = s3[71];
              float v2186_data = ir7[5];
              ir7[5] = (v2186_data + (v2158_data * v2184_data));
              float v2189_data = s3[83];
              float v2191_data = ir7[6];
              ir7[6] = (v2191_data + (v2158_data * v2189_data));
              float v2194_data = s3[95];
              float v2196_data = ir7[7];
              ir7[7] = (v2196_data + (v2158_data * v2194_data));
            }
            if (v1672_lead < 12) {
              #pragma unroll
              for (int32_t v2202_n1 = 0; v2202_n1 < 8; ++v2202_n1) {
                int32_t v2203_a = 0 + v2202_n1;
                float v2204_data = ir7[v2203_a];
                int32_t v2205_a = 0 + v2202_n1;
                float v2206_data = r5[v2205_a];
                int32_t v2208_a = 0 + v2202_n1;
                r7[v2208_a] = (v2206_data + v2204_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r7);
          int32_t v2211_lead = threadIdx.x % 16;
          if (v2211_lead < 12) {
            #pragma unroll
            for (int32_t v2213_i1 = 0; v2213_i1 < 8; ++v2213_i1) {
              int32_t v2214_a = 0 + v2213_i1;
              float v2215_data = r7[v2214_a];
              int32_t v2222_a = v2211_lead + (v2213_i1 * 12);
              glb_m0[v2222_a] = v2215_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

