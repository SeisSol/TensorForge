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
                r1[v560_n1] = v563_data;
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
          int32_t v568_lead = threadIdx.x % 16;
          if (v568_lead < 12) {
            #pragma unroll
            for (int32_t v570_i1 = 0; v570_i1 < 12; ++v570_i1) {
              int32_t v577_a = v568_lead + (v570_i1 * 12);
              float v578_data;
              {
                v578_data = __ldcg(&glb_m5[v577_a]);
              }
              int32_t v579_a = 0 + v570_i1;
              r4[v579_a] = v578_data;
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
            int32_t v582_lead = threadIdx.x % 16;
            if (v582_lead < 12) {
              float v584_data = r2[0];
              float v585_data = s1[0];
              float v587_data = ir3[0];
              ir3[0] = (v587_data + (v584_data * v585_data));
              float v590_data = s1[12];
              float v592_data = ir3[1];
              ir3[1] = (v592_data + (v584_data * v590_data));
              float v595_data = s1[24];
              float v597_data = ir3[2];
              ir3[2] = (v597_data + (v584_data * v595_data));
              float v600_data = s1[36];
              float v602_data = ir3[3];
              ir3[3] = (v602_data + (v584_data * v600_data));
              float v605_data = s1[48];
              float v607_data = ir3[4];
              ir3[4] = (v607_data + (v584_data * v605_data));
              float v610_data = s1[60];
              float v612_data = ir3[5];
              ir3[5] = (v612_data + (v584_data * v610_data));
              float v615_data = s1[72];
              float v617_data = ir3[6];
              ir3[6] = (v617_data + (v584_data * v615_data));
              float v620_data = s1[84];
              float v622_data = ir3[7];
              ir3[7] = (v622_data + (v584_data * v620_data));
            }
            if (v582_lead < 12) {
              float v628_data = r2[1];
              float v629_data = s1[1];
              float v631_data = ir3[0];
              ir3[0] = (v631_data + (v628_data * v629_data));
              float v634_data = s1[13];
              float v636_data = ir3[1];
              ir3[1] = (v636_data + (v628_data * v634_data));
              float v639_data = s1[25];
              float v641_data = ir3[2];
              ir3[2] = (v641_data + (v628_data * v639_data));
              float v644_data = s1[37];
              float v646_data = ir3[3];
              ir3[3] = (v646_data + (v628_data * v644_data));
              float v649_data = s1[49];
              float v651_data = ir3[4];
              ir3[4] = (v651_data + (v628_data * v649_data));
              float v654_data = s1[61];
              float v656_data = ir3[5];
              ir3[5] = (v656_data + (v628_data * v654_data));
              float v659_data = s1[73];
              float v661_data = ir3[6];
              ir3[6] = (v661_data + (v628_data * v659_data));
              float v664_data = s1[85];
              float v666_data = ir3[7];
              ir3[7] = (v666_data + (v628_data * v664_data));
            }
            if (v582_lead < 12) {
              float v672_data = r2[2];
              float v673_data = s1[2];
              float v675_data = ir3[0];
              ir3[0] = (v675_data + (v672_data * v673_data));
              float v678_data = s1[14];
              float v680_data = ir3[1];
              ir3[1] = (v680_data + (v672_data * v678_data));
              float v683_data = s1[26];
              float v685_data = ir3[2];
              ir3[2] = (v685_data + (v672_data * v683_data));
              float v688_data = s1[38];
              float v690_data = ir3[3];
              ir3[3] = (v690_data + (v672_data * v688_data));
              float v693_data = s1[50];
              float v695_data = ir3[4];
              ir3[4] = (v695_data + (v672_data * v693_data));
              float v698_data = s1[62];
              float v700_data = ir3[5];
              ir3[5] = (v700_data + (v672_data * v698_data));
              float v703_data = s1[74];
              float v705_data = ir3[6];
              ir3[6] = (v705_data + (v672_data * v703_data));
              float v708_data = s1[86];
              float v710_data = ir3[7];
              ir3[7] = (v710_data + (v672_data * v708_data));
            }
            if (v582_lead < 12) {
              float v716_data = r2[3];
              float v717_data = s1[3];
              float v719_data = ir3[0];
              ir3[0] = (v719_data + (v716_data * v717_data));
              float v722_data = s1[15];
              float v724_data = ir3[1];
              ir3[1] = (v724_data + (v716_data * v722_data));
              float v727_data = s1[27];
              float v729_data = ir3[2];
              ir3[2] = (v729_data + (v716_data * v727_data));
              float v732_data = s1[39];
              float v734_data = ir3[3];
              ir3[3] = (v734_data + (v716_data * v732_data));
              float v737_data = s1[51];
              float v739_data = ir3[4];
              ir3[4] = (v739_data + (v716_data * v737_data));
              float v742_data = s1[63];
              float v744_data = ir3[5];
              ir3[5] = (v744_data + (v716_data * v742_data));
              float v747_data = s1[75];
              float v749_data = ir3[6];
              ir3[6] = (v749_data + (v716_data * v747_data));
              float v752_data = s1[87];
              float v754_data = ir3[7];
              ir3[7] = (v754_data + (v716_data * v752_data));
            }
            if (v582_lead < 12) {
              float v760_data = r2[4];
              float v761_data = s1[4];
              float v763_data = ir3[0];
              ir3[0] = (v763_data + (v760_data * v761_data));
              float v766_data = s1[16];
              float v768_data = ir3[1];
              ir3[1] = (v768_data + (v760_data * v766_data));
              float v771_data = s1[28];
              float v773_data = ir3[2];
              ir3[2] = (v773_data + (v760_data * v771_data));
              float v776_data = s1[40];
              float v778_data = ir3[3];
              ir3[3] = (v778_data + (v760_data * v776_data));
              float v781_data = s1[52];
              float v783_data = ir3[4];
              ir3[4] = (v783_data + (v760_data * v781_data));
              float v786_data = s1[64];
              float v788_data = ir3[5];
              ir3[5] = (v788_data + (v760_data * v786_data));
              float v791_data = s1[76];
              float v793_data = ir3[6];
              ir3[6] = (v793_data + (v760_data * v791_data));
              float v796_data = s1[88];
              float v798_data = ir3[7];
              ir3[7] = (v798_data + (v760_data * v796_data));
            }
            if (v582_lead < 12) {
              float v804_data = r2[5];
              float v805_data = s1[5];
              float v807_data = ir3[0];
              ir3[0] = (v807_data + (v804_data * v805_data));
              float v810_data = s1[17];
              float v812_data = ir3[1];
              ir3[1] = (v812_data + (v804_data * v810_data));
              float v815_data = s1[29];
              float v817_data = ir3[2];
              ir3[2] = (v817_data + (v804_data * v815_data));
              float v820_data = s1[41];
              float v822_data = ir3[3];
              ir3[3] = (v822_data + (v804_data * v820_data));
              float v825_data = s1[53];
              float v827_data = ir3[4];
              ir3[4] = (v827_data + (v804_data * v825_data));
              float v830_data = s1[65];
              float v832_data = ir3[5];
              ir3[5] = (v832_data + (v804_data * v830_data));
              float v835_data = s1[77];
              float v837_data = ir3[6];
              ir3[6] = (v837_data + (v804_data * v835_data));
              float v840_data = s1[89];
              float v842_data = ir3[7];
              ir3[7] = (v842_data + (v804_data * v840_data));
            }
            if (v582_lead < 12) {
              float v848_data = r2[6];
              float v849_data = s1[6];
              float v851_data = ir3[0];
              ir3[0] = (v851_data + (v848_data * v849_data));
              float v854_data = s1[18];
              float v856_data = ir3[1];
              ir3[1] = (v856_data + (v848_data * v854_data));
              float v859_data = s1[30];
              float v861_data = ir3[2];
              ir3[2] = (v861_data + (v848_data * v859_data));
              float v864_data = s1[42];
              float v866_data = ir3[3];
              ir3[3] = (v866_data + (v848_data * v864_data));
              float v869_data = s1[54];
              float v871_data = ir3[4];
              ir3[4] = (v871_data + (v848_data * v869_data));
              float v874_data = s1[66];
              float v876_data = ir3[5];
              ir3[5] = (v876_data + (v848_data * v874_data));
              float v879_data = s1[78];
              float v881_data = ir3[6];
              ir3[6] = (v881_data + (v848_data * v879_data));
              float v884_data = s1[90];
              float v886_data = ir3[7];
              ir3[7] = (v886_data + (v848_data * v884_data));
            }
            if (v582_lead < 12) {
              float v892_data = r2[7];
              float v893_data = s1[7];
              float v895_data = ir3[0];
              ir3[0] = (v895_data + (v892_data * v893_data));
              float v898_data = s1[19];
              float v900_data = ir3[1];
              ir3[1] = (v900_data + (v892_data * v898_data));
              float v903_data = s1[31];
              float v905_data = ir3[2];
              ir3[2] = (v905_data + (v892_data * v903_data));
              float v908_data = s1[43];
              float v910_data = ir3[3];
              ir3[3] = (v910_data + (v892_data * v908_data));
              float v913_data = s1[55];
              float v915_data = ir3[4];
              ir3[4] = (v915_data + (v892_data * v913_data));
              float v918_data = s1[67];
              float v920_data = ir3[5];
              ir3[5] = (v920_data + (v892_data * v918_data));
              float v923_data = s1[79];
              float v925_data = ir3[6];
              ir3[6] = (v925_data + (v892_data * v923_data));
              float v928_data = s1[91];
              float v930_data = ir3[7];
              ir3[7] = (v930_data + (v892_data * v928_data));
            }
            if (v582_lead < 12) {
              float v936_data = r2[8];
              float v937_data = s1[8];
              float v939_data = ir3[0];
              ir3[0] = (v939_data + (v936_data * v937_data));
              float v942_data = s1[20];
              float v944_data = ir3[1];
              ir3[1] = (v944_data + (v936_data * v942_data));
              float v947_data = s1[32];
              float v949_data = ir3[2];
              ir3[2] = (v949_data + (v936_data * v947_data));
              float v952_data = s1[44];
              float v954_data = ir3[3];
              ir3[3] = (v954_data + (v936_data * v952_data));
              float v957_data = s1[56];
              float v959_data = ir3[4];
              ir3[4] = (v959_data + (v936_data * v957_data));
              float v962_data = s1[68];
              float v964_data = ir3[5];
              ir3[5] = (v964_data + (v936_data * v962_data));
              float v967_data = s1[80];
              float v969_data = ir3[6];
              ir3[6] = (v969_data + (v936_data * v967_data));
              float v972_data = s1[92];
              float v974_data = ir3[7];
              ir3[7] = (v974_data + (v936_data * v972_data));
            }
            if (v582_lead < 12) {
              float v980_data = r2[9];
              float v981_data = s1[9];
              float v983_data = ir3[0];
              ir3[0] = (v983_data + (v980_data * v981_data));
              float v986_data = s1[21];
              float v988_data = ir3[1];
              ir3[1] = (v988_data + (v980_data * v986_data));
              float v991_data = s1[33];
              float v993_data = ir3[2];
              ir3[2] = (v993_data + (v980_data * v991_data));
              float v996_data = s1[45];
              float v998_data = ir3[3];
              ir3[3] = (v998_data + (v980_data * v996_data));
              float v1001_data = s1[57];
              float v1003_data = ir3[4];
              ir3[4] = (v1003_data + (v980_data * v1001_data));
              float v1006_data = s1[69];
              float v1008_data = ir3[5];
              ir3[5] = (v1008_data + (v980_data * v1006_data));
              float v1011_data = s1[81];
              float v1013_data = ir3[6];
              ir3[6] = (v1013_data + (v980_data * v1011_data));
              float v1016_data = s1[93];
              float v1018_data = ir3[7];
              ir3[7] = (v1018_data + (v980_data * v1016_data));
            }
            if (v582_lead < 12) {
              float v1024_data = r2[10];
              float v1025_data = s1[10];
              float v1027_data = ir3[0];
              ir3[0] = (v1027_data + (v1024_data * v1025_data));
              float v1030_data = s1[22];
              float v1032_data = ir3[1];
              ir3[1] = (v1032_data + (v1024_data * v1030_data));
              float v1035_data = s1[34];
              float v1037_data = ir3[2];
              ir3[2] = (v1037_data + (v1024_data * v1035_data));
              float v1040_data = s1[46];
              float v1042_data = ir3[3];
              ir3[3] = (v1042_data + (v1024_data * v1040_data));
              float v1045_data = s1[58];
              float v1047_data = ir3[4];
              ir3[4] = (v1047_data + (v1024_data * v1045_data));
              float v1050_data = s1[70];
              float v1052_data = ir3[5];
              ir3[5] = (v1052_data + (v1024_data * v1050_data));
              float v1055_data = s1[82];
              float v1057_data = ir3[6];
              ir3[6] = (v1057_data + (v1024_data * v1055_data));
              float v1060_data = s1[94];
              float v1062_data = ir3[7];
              ir3[7] = (v1062_data + (v1024_data * v1060_data));
            }
            if (v582_lead < 12) {
              float v1068_data = r2[11];
              float v1069_data = s1[11];
              float v1071_data = ir3[0];
              ir3[0] = (v1071_data + (v1068_data * v1069_data));
              float v1074_data = s1[23];
              float v1076_data = ir3[1];
              ir3[1] = (v1076_data + (v1068_data * v1074_data));
              float v1079_data = s1[35];
              float v1081_data = ir3[2];
              ir3[2] = (v1081_data + (v1068_data * v1079_data));
              float v1084_data = s1[47];
              float v1086_data = ir3[3];
              ir3[3] = (v1086_data + (v1068_data * v1084_data));
              float v1089_data = s1[59];
              float v1091_data = ir3[4];
              ir3[4] = (v1091_data + (v1068_data * v1089_data));
              float v1094_data = s1[71];
              float v1096_data = ir3[5];
              ir3[5] = (v1096_data + (v1068_data * v1094_data));
              float v1099_data = s1[83];
              float v1101_data = ir3[6];
              ir3[6] = (v1101_data + (v1068_data * v1099_data));
              float v1104_data = s1[95];
              float v1106_data = ir3[7];
              ir3[7] = (v1106_data + (v1068_data * v1104_data));
            }
            if (v582_lead < 12) {
              #pragma unroll
              for (int32_t v1112_n1 = 0; v1112_n1 < 8; ++v1112_n1) {
                int32_t v1113_a = 0 + v1112_n1;
                float v1115_data = ir3[v1112_n1];
                int32_t v1116_a = 0 + v1112_n1;
                float v1118_data = r1[v1112_n1];
                int32_t v1120_a = 0 + v1112_n1;
                r3[v1112_n1] = (v1118_data + v1115_data);
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
          int32_t v1124_lead = threadIdx.x % 16;
          if (v1124_lead < 12) {
            #pragma unroll
            for (int32_t v1126_i1 = 0; v1126_i1 < 12; ++v1126_i1) {
              int32_t v1133_a = v1124_lead + (v1126_i1 * 12);
              float v1134_data;
              {
                v1134_data = __ldcg(&glb_m7[v1133_a]);
              }
              int32_t v1135_a = 0 + v1126_i1;
              r6[v1135_a] = v1134_data;
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
            int32_t v1138_lead = threadIdx.x % 16;
            if (v1138_lead < 12) {
              float v1140_data = r4[0];
              float v1141_data = s2[0];
              float v1143_data = ir5[0];
              ir5[0] = (v1143_data + (v1140_data * v1141_data));
              float v1146_data = s2[12];
              float v1148_data = ir5[1];
              ir5[1] = (v1148_data + (v1140_data * v1146_data));
              float v1151_data = s2[24];
              float v1153_data = ir5[2];
              ir5[2] = (v1153_data + (v1140_data * v1151_data));
              float v1156_data = s2[36];
              float v1158_data = ir5[3];
              ir5[3] = (v1158_data + (v1140_data * v1156_data));
              float v1161_data = s2[48];
              float v1163_data = ir5[4];
              ir5[4] = (v1163_data + (v1140_data * v1161_data));
              float v1166_data = s2[60];
              float v1168_data = ir5[5];
              ir5[5] = (v1168_data + (v1140_data * v1166_data));
              float v1171_data = s2[72];
              float v1173_data = ir5[6];
              ir5[6] = (v1173_data + (v1140_data * v1171_data));
              float v1176_data = s2[84];
              float v1178_data = ir5[7];
              ir5[7] = (v1178_data + (v1140_data * v1176_data));
            }
            if (v1138_lead < 12) {
              float v1184_data = r4[1];
              float v1185_data = s2[1];
              float v1187_data = ir5[0];
              ir5[0] = (v1187_data + (v1184_data * v1185_data));
              float v1190_data = s2[13];
              float v1192_data = ir5[1];
              ir5[1] = (v1192_data + (v1184_data * v1190_data));
              float v1195_data = s2[25];
              float v1197_data = ir5[2];
              ir5[2] = (v1197_data + (v1184_data * v1195_data));
              float v1200_data = s2[37];
              float v1202_data = ir5[3];
              ir5[3] = (v1202_data + (v1184_data * v1200_data));
              float v1205_data = s2[49];
              float v1207_data = ir5[4];
              ir5[4] = (v1207_data + (v1184_data * v1205_data));
              float v1210_data = s2[61];
              float v1212_data = ir5[5];
              ir5[5] = (v1212_data + (v1184_data * v1210_data));
              float v1215_data = s2[73];
              float v1217_data = ir5[6];
              ir5[6] = (v1217_data + (v1184_data * v1215_data));
              float v1220_data = s2[85];
              float v1222_data = ir5[7];
              ir5[7] = (v1222_data + (v1184_data * v1220_data));
            }
            if (v1138_lead < 12) {
              float v1228_data = r4[2];
              float v1229_data = s2[2];
              float v1231_data = ir5[0];
              ir5[0] = (v1231_data + (v1228_data * v1229_data));
              float v1234_data = s2[14];
              float v1236_data = ir5[1];
              ir5[1] = (v1236_data + (v1228_data * v1234_data));
              float v1239_data = s2[26];
              float v1241_data = ir5[2];
              ir5[2] = (v1241_data + (v1228_data * v1239_data));
              float v1244_data = s2[38];
              float v1246_data = ir5[3];
              ir5[3] = (v1246_data + (v1228_data * v1244_data));
              float v1249_data = s2[50];
              float v1251_data = ir5[4];
              ir5[4] = (v1251_data + (v1228_data * v1249_data));
              float v1254_data = s2[62];
              float v1256_data = ir5[5];
              ir5[5] = (v1256_data + (v1228_data * v1254_data));
              float v1259_data = s2[74];
              float v1261_data = ir5[6];
              ir5[6] = (v1261_data + (v1228_data * v1259_data));
              float v1264_data = s2[86];
              float v1266_data = ir5[7];
              ir5[7] = (v1266_data + (v1228_data * v1264_data));
            }
            if (v1138_lead < 12) {
              float v1272_data = r4[3];
              float v1273_data = s2[3];
              float v1275_data = ir5[0];
              ir5[0] = (v1275_data + (v1272_data * v1273_data));
              float v1278_data = s2[15];
              float v1280_data = ir5[1];
              ir5[1] = (v1280_data + (v1272_data * v1278_data));
              float v1283_data = s2[27];
              float v1285_data = ir5[2];
              ir5[2] = (v1285_data + (v1272_data * v1283_data));
              float v1288_data = s2[39];
              float v1290_data = ir5[3];
              ir5[3] = (v1290_data + (v1272_data * v1288_data));
              float v1293_data = s2[51];
              float v1295_data = ir5[4];
              ir5[4] = (v1295_data + (v1272_data * v1293_data));
              float v1298_data = s2[63];
              float v1300_data = ir5[5];
              ir5[5] = (v1300_data + (v1272_data * v1298_data));
              float v1303_data = s2[75];
              float v1305_data = ir5[6];
              ir5[6] = (v1305_data + (v1272_data * v1303_data));
              float v1308_data = s2[87];
              float v1310_data = ir5[7];
              ir5[7] = (v1310_data + (v1272_data * v1308_data));
            }
            if (v1138_lead < 12) {
              float v1316_data = r4[4];
              float v1317_data = s2[4];
              float v1319_data = ir5[0];
              ir5[0] = (v1319_data + (v1316_data * v1317_data));
              float v1322_data = s2[16];
              float v1324_data = ir5[1];
              ir5[1] = (v1324_data + (v1316_data * v1322_data));
              float v1327_data = s2[28];
              float v1329_data = ir5[2];
              ir5[2] = (v1329_data + (v1316_data * v1327_data));
              float v1332_data = s2[40];
              float v1334_data = ir5[3];
              ir5[3] = (v1334_data + (v1316_data * v1332_data));
              float v1337_data = s2[52];
              float v1339_data = ir5[4];
              ir5[4] = (v1339_data + (v1316_data * v1337_data));
              float v1342_data = s2[64];
              float v1344_data = ir5[5];
              ir5[5] = (v1344_data + (v1316_data * v1342_data));
              float v1347_data = s2[76];
              float v1349_data = ir5[6];
              ir5[6] = (v1349_data + (v1316_data * v1347_data));
              float v1352_data = s2[88];
              float v1354_data = ir5[7];
              ir5[7] = (v1354_data + (v1316_data * v1352_data));
            }
            if (v1138_lead < 12) {
              float v1360_data = r4[5];
              float v1361_data = s2[5];
              float v1363_data = ir5[0];
              ir5[0] = (v1363_data + (v1360_data * v1361_data));
              float v1366_data = s2[17];
              float v1368_data = ir5[1];
              ir5[1] = (v1368_data + (v1360_data * v1366_data));
              float v1371_data = s2[29];
              float v1373_data = ir5[2];
              ir5[2] = (v1373_data + (v1360_data * v1371_data));
              float v1376_data = s2[41];
              float v1378_data = ir5[3];
              ir5[3] = (v1378_data + (v1360_data * v1376_data));
              float v1381_data = s2[53];
              float v1383_data = ir5[4];
              ir5[4] = (v1383_data + (v1360_data * v1381_data));
              float v1386_data = s2[65];
              float v1388_data = ir5[5];
              ir5[5] = (v1388_data + (v1360_data * v1386_data));
              float v1391_data = s2[77];
              float v1393_data = ir5[6];
              ir5[6] = (v1393_data + (v1360_data * v1391_data));
              float v1396_data = s2[89];
              float v1398_data = ir5[7];
              ir5[7] = (v1398_data + (v1360_data * v1396_data));
            }
            if (v1138_lead < 12) {
              float v1404_data = r4[6];
              float v1405_data = s2[6];
              float v1407_data = ir5[0];
              ir5[0] = (v1407_data + (v1404_data * v1405_data));
              float v1410_data = s2[18];
              float v1412_data = ir5[1];
              ir5[1] = (v1412_data + (v1404_data * v1410_data));
              float v1415_data = s2[30];
              float v1417_data = ir5[2];
              ir5[2] = (v1417_data + (v1404_data * v1415_data));
              float v1420_data = s2[42];
              float v1422_data = ir5[3];
              ir5[3] = (v1422_data + (v1404_data * v1420_data));
              float v1425_data = s2[54];
              float v1427_data = ir5[4];
              ir5[4] = (v1427_data + (v1404_data * v1425_data));
              float v1430_data = s2[66];
              float v1432_data = ir5[5];
              ir5[5] = (v1432_data + (v1404_data * v1430_data));
              float v1435_data = s2[78];
              float v1437_data = ir5[6];
              ir5[6] = (v1437_data + (v1404_data * v1435_data));
              float v1440_data = s2[90];
              float v1442_data = ir5[7];
              ir5[7] = (v1442_data + (v1404_data * v1440_data));
            }
            if (v1138_lead < 12) {
              float v1448_data = r4[7];
              float v1449_data = s2[7];
              float v1451_data = ir5[0];
              ir5[0] = (v1451_data + (v1448_data * v1449_data));
              float v1454_data = s2[19];
              float v1456_data = ir5[1];
              ir5[1] = (v1456_data + (v1448_data * v1454_data));
              float v1459_data = s2[31];
              float v1461_data = ir5[2];
              ir5[2] = (v1461_data + (v1448_data * v1459_data));
              float v1464_data = s2[43];
              float v1466_data = ir5[3];
              ir5[3] = (v1466_data + (v1448_data * v1464_data));
              float v1469_data = s2[55];
              float v1471_data = ir5[4];
              ir5[4] = (v1471_data + (v1448_data * v1469_data));
              float v1474_data = s2[67];
              float v1476_data = ir5[5];
              ir5[5] = (v1476_data + (v1448_data * v1474_data));
              float v1479_data = s2[79];
              float v1481_data = ir5[6];
              ir5[6] = (v1481_data + (v1448_data * v1479_data));
              float v1484_data = s2[91];
              float v1486_data = ir5[7];
              ir5[7] = (v1486_data + (v1448_data * v1484_data));
            }
            if (v1138_lead < 12) {
              float v1492_data = r4[8];
              float v1493_data = s2[8];
              float v1495_data = ir5[0];
              ir5[0] = (v1495_data + (v1492_data * v1493_data));
              float v1498_data = s2[20];
              float v1500_data = ir5[1];
              ir5[1] = (v1500_data + (v1492_data * v1498_data));
              float v1503_data = s2[32];
              float v1505_data = ir5[2];
              ir5[2] = (v1505_data + (v1492_data * v1503_data));
              float v1508_data = s2[44];
              float v1510_data = ir5[3];
              ir5[3] = (v1510_data + (v1492_data * v1508_data));
              float v1513_data = s2[56];
              float v1515_data = ir5[4];
              ir5[4] = (v1515_data + (v1492_data * v1513_data));
              float v1518_data = s2[68];
              float v1520_data = ir5[5];
              ir5[5] = (v1520_data + (v1492_data * v1518_data));
              float v1523_data = s2[80];
              float v1525_data = ir5[6];
              ir5[6] = (v1525_data + (v1492_data * v1523_data));
              float v1528_data = s2[92];
              float v1530_data = ir5[7];
              ir5[7] = (v1530_data + (v1492_data * v1528_data));
            }
            if (v1138_lead < 12) {
              float v1536_data = r4[9];
              float v1537_data = s2[9];
              float v1539_data = ir5[0];
              ir5[0] = (v1539_data + (v1536_data * v1537_data));
              float v1542_data = s2[21];
              float v1544_data = ir5[1];
              ir5[1] = (v1544_data + (v1536_data * v1542_data));
              float v1547_data = s2[33];
              float v1549_data = ir5[2];
              ir5[2] = (v1549_data + (v1536_data * v1547_data));
              float v1552_data = s2[45];
              float v1554_data = ir5[3];
              ir5[3] = (v1554_data + (v1536_data * v1552_data));
              float v1557_data = s2[57];
              float v1559_data = ir5[4];
              ir5[4] = (v1559_data + (v1536_data * v1557_data));
              float v1562_data = s2[69];
              float v1564_data = ir5[5];
              ir5[5] = (v1564_data + (v1536_data * v1562_data));
              float v1567_data = s2[81];
              float v1569_data = ir5[6];
              ir5[6] = (v1569_data + (v1536_data * v1567_data));
              float v1572_data = s2[93];
              float v1574_data = ir5[7];
              ir5[7] = (v1574_data + (v1536_data * v1572_data));
            }
            if (v1138_lead < 12) {
              float v1580_data = r4[10];
              float v1581_data = s2[10];
              float v1583_data = ir5[0];
              ir5[0] = (v1583_data + (v1580_data * v1581_data));
              float v1586_data = s2[22];
              float v1588_data = ir5[1];
              ir5[1] = (v1588_data + (v1580_data * v1586_data));
              float v1591_data = s2[34];
              float v1593_data = ir5[2];
              ir5[2] = (v1593_data + (v1580_data * v1591_data));
              float v1596_data = s2[46];
              float v1598_data = ir5[3];
              ir5[3] = (v1598_data + (v1580_data * v1596_data));
              float v1601_data = s2[58];
              float v1603_data = ir5[4];
              ir5[4] = (v1603_data + (v1580_data * v1601_data));
              float v1606_data = s2[70];
              float v1608_data = ir5[5];
              ir5[5] = (v1608_data + (v1580_data * v1606_data));
              float v1611_data = s2[82];
              float v1613_data = ir5[6];
              ir5[6] = (v1613_data + (v1580_data * v1611_data));
              float v1616_data = s2[94];
              float v1618_data = ir5[7];
              ir5[7] = (v1618_data + (v1580_data * v1616_data));
            }
            if (v1138_lead < 12) {
              float v1624_data = r4[11];
              float v1625_data = s2[11];
              float v1627_data = ir5[0];
              ir5[0] = (v1627_data + (v1624_data * v1625_data));
              float v1630_data = s2[23];
              float v1632_data = ir5[1];
              ir5[1] = (v1632_data + (v1624_data * v1630_data));
              float v1635_data = s2[35];
              float v1637_data = ir5[2];
              ir5[2] = (v1637_data + (v1624_data * v1635_data));
              float v1640_data = s2[47];
              float v1642_data = ir5[3];
              ir5[3] = (v1642_data + (v1624_data * v1640_data));
              float v1645_data = s2[59];
              float v1647_data = ir5[4];
              ir5[4] = (v1647_data + (v1624_data * v1645_data));
              float v1650_data = s2[71];
              float v1652_data = ir5[5];
              ir5[5] = (v1652_data + (v1624_data * v1650_data));
              float v1655_data = s2[83];
              float v1657_data = ir5[6];
              ir5[6] = (v1657_data + (v1624_data * v1655_data));
              float v1660_data = s2[95];
              float v1662_data = ir5[7];
              ir5[7] = (v1662_data + (v1624_data * v1660_data));
            }
            if (v1138_lead < 12) {
              #pragma unroll
              for (int32_t v1668_n1 = 0; v1668_n1 < 8; ++v1668_n1) {
                int32_t v1669_a = 0 + v1668_n1;
                float v1671_data = ir5[v1668_n1];
                int32_t v1672_a = 0 + v1668_n1;
                float v1674_data = r3[v1668_n1];
                int32_t v1676_a = 0 + v1668_n1;
                r5[v1668_n1] = (v1674_data + v1671_data);
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
            int32_t v1680_lead = threadIdx.x % 16;
            if (v1680_lead < 12) {
              float v1682_data = r6[0];
              float v1683_data = s3[0];
              float v1685_data = ir7[0];
              ir7[0] = (v1685_data + (v1682_data * v1683_data));
              float v1688_data = s3[12];
              float v1690_data = ir7[1];
              ir7[1] = (v1690_data + (v1682_data * v1688_data));
              float v1693_data = s3[24];
              float v1695_data = ir7[2];
              ir7[2] = (v1695_data + (v1682_data * v1693_data));
              float v1698_data = s3[36];
              float v1700_data = ir7[3];
              ir7[3] = (v1700_data + (v1682_data * v1698_data));
              float v1703_data = s3[48];
              float v1705_data = ir7[4];
              ir7[4] = (v1705_data + (v1682_data * v1703_data));
              float v1708_data = s3[60];
              float v1710_data = ir7[5];
              ir7[5] = (v1710_data + (v1682_data * v1708_data));
              float v1713_data = s3[72];
              float v1715_data = ir7[6];
              ir7[6] = (v1715_data + (v1682_data * v1713_data));
              float v1718_data = s3[84];
              float v1720_data = ir7[7];
              ir7[7] = (v1720_data + (v1682_data * v1718_data));
            }
            if (v1680_lead < 12) {
              float v1726_data = r6[1];
              float v1727_data = s3[1];
              float v1729_data = ir7[0];
              ir7[0] = (v1729_data + (v1726_data * v1727_data));
              float v1732_data = s3[13];
              float v1734_data = ir7[1];
              ir7[1] = (v1734_data + (v1726_data * v1732_data));
              float v1737_data = s3[25];
              float v1739_data = ir7[2];
              ir7[2] = (v1739_data + (v1726_data * v1737_data));
              float v1742_data = s3[37];
              float v1744_data = ir7[3];
              ir7[3] = (v1744_data + (v1726_data * v1742_data));
              float v1747_data = s3[49];
              float v1749_data = ir7[4];
              ir7[4] = (v1749_data + (v1726_data * v1747_data));
              float v1752_data = s3[61];
              float v1754_data = ir7[5];
              ir7[5] = (v1754_data + (v1726_data * v1752_data));
              float v1757_data = s3[73];
              float v1759_data = ir7[6];
              ir7[6] = (v1759_data + (v1726_data * v1757_data));
              float v1762_data = s3[85];
              float v1764_data = ir7[7];
              ir7[7] = (v1764_data + (v1726_data * v1762_data));
            }
            if (v1680_lead < 12) {
              float v1770_data = r6[2];
              float v1771_data = s3[2];
              float v1773_data = ir7[0];
              ir7[0] = (v1773_data + (v1770_data * v1771_data));
              float v1776_data = s3[14];
              float v1778_data = ir7[1];
              ir7[1] = (v1778_data + (v1770_data * v1776_data));
              float v1781_data = s3[26];
              float v1783_data = ir7[2];
              ir7[2] = (v1783_data + (v1770_data * v1781_data));
              float v1786_data = s3[38];
              float v1788_data = ir7[3];
              ir7[3] = (v1788_data + (v1770_data * v1786_data));
              float v1791_data = s3[50];
              float v1793_data = ir7[4];
              ir7[4] = (v1793_data + (v1770_data * v1791_data));
              float v1796_data = s3[62];
              float v1798_data = ir7[5];
              ir7[5] = (v1798_data + (v1770_data * v1796_data));
              float v1801_data = s3[74];
              float v1803_data = ir7[6];
              ir7[6] = (v1803_data + (v1770_data * v1801_data));
              float v1806_data = s3[86];
              float v1808_data = ir7[7];
              ir7[7] = (v1808_data + (v1770_data * v1806_data));
            }
            if (v1680_lead < 12) {
              float v1814_data = r6[3];
              float v1815_data = s3[3];
              float v1817_data = ir7[0];
              ir7[0] = (v1817_data + (v1814_data * v1815_data));
              float v1820_data = s3[15];
              float v1822_data = ir7[1];
              ir7[1] = (v1822_data + (v1814_data * v1820_data));
              float v1825_data = s3[27];
              float v1827_data = ir7[2];
              ir7[2] = (v1827_data + (v1814_data * v1825_data));
              float v1830_data = s3[39];
              float v1832_data = ir7[3];
              ir7[3] = (v1832_data + (v1814_data * v1830_data));
              float v1835_data = s3[51];
              float v1837_data = ir7[4];
              ir7[4] = (v1837_data + (v1814_data * v1835_data));
              float v1840_data = s3[63];
              float v1842_data = ir7[5];
              ir7[5] = (v1842_data + (v1814_data * v1840_data));
              float v1845_data = s3[75];
              float v1847_data = ir7[6];
              ir7[6] = (v1847_data + (v1814_data * v1845_data));
              float v1850_data = s3[87];
              float v1852_data = ir7[7];
              ir7[7] = (v1852_data + (v1814_data * v1850_data));
            }
            if (v1680_lead < 12) {
              float v1858_data = r6[4];
              float v1859_data = s3[4];
              float v1861_data = ir7[0];
              ir7[0] = (v1861_data + (v1858_data * v1859_data));
              float v1864_data = s3[16];
              float v1866_data = ir7[1];
              ir7[1] = (v1866_data + (v1858_data * v1864_data));
              float v1869_data = s3[28];
              float v1871_data = ir7[2];
              ir7[2] = (v1871_data + (v1858_data * v1869_data));
              float v1874_data = s3[40];
              float v1876_data = ir7[3];
              ir7[3] = (v1876_data + (v1858_data * v1874_data));
              float v1879_data = s3[52];
              float v1881_data = ir7[4];
              ir7[4] = (v1881_data + (v1858_data * v1879_data));
              float v1884_data = s3[64];
              float v1886_data = ir7[5];
              ir7[5] = (v1886_data + (v1858_data * v1884_data));
              float v1889_data = s3[76];
              float v1891_data = ir7[6];
              ir7[6] = (v1891_data + (v1858_data * v1889_data));
              float v1894_data = s3[88];
              float v1896_data = ir7[7];
              ir7[7] = (v1896_data + (v1858_data * v1894_data));
            }
            if (v1680_lead < 12) {
              float v1902_data = r6[5];
              float v1903_data = s3[5];
              float v1905_data = ir7[0];
              ir7[0] = (v1905_data + (v1902_data * v1903_data));
              float v1908_data = s3[17];
              float v1910_data = ir7[1];
              ir7[1] = (v1910_data + (v1902_data * v1908_data));
              float v1913_data = s3[29];
              float v1915_data = ir7[2];
              ir7[2] = (v1915_data + (v1902_data * v1913_data));
              float v1918_data = s3[41];
              float v1920_data = ir7[3];
              ir7[3] = (v1920_data + (v1902_data * v1918_data));
              float v1923_data = s3[53];
              float v1925_data = ir7[4];
              ir7[4] = (v1925_data + (v1902_data * v1923_data));
              float v1928_data = s3[65];
              float v1930_data = ir7[5];
              ir7[5] = (v1930_data + (v1902_data * v1928_data));
              float v1933_data = s3[77];
              float v1935_data = ir7[6];
              ir7[6] = (v1935_data + (v1902_data * v1933_data));
              float v1938_data = s3[89];
              float v1940_data = ir7[7];
              ir7[7] = (v1940_data + (v1902_data * v1938_data));
            }
            if (v1680_lead < 12) {
              float v1946_data = r6[6];
              float v1947_data = s3[6];
              float v1949_data = ir7[0];
              ir7[0] = (v1949_data + (v1946_data * v1947_data));
              float v1952_data = s3[18];
              float v1954_data = ir7[1];
              ir7[1] = (v1954_data + (v1946_data * v1952_data));
              float v1957_data = s3[30];
              float v1959_data = ir7[2];
              ir7[2] = (v1959_data + (v1946_data * v1957_data));
              float v1962_data = s3[42];
              float v1964_data = ir7[3];
              ir7[3] = (v1964_data + (v1946_data * v1962_data));
              float v1967_data = s3[54];
              float v1969_data = ir7[4];
              ir7[4] = (v1969_data + (v1946_data * v1967_data));
              float v1972_data = s3[66];
              float v1974_data = ir7[5];
              ir7[5] = (v1974_data + (v1946_data * v1972_data));
              float v1977_data = s3[78];
              float v1979_data = ir7[6];
              ir7[6] = (v1979_data + (v1946_data * v1977_data));
              float v1982_data = s3[90];
              float v1984_data = ir7[7];
              ir7[7] = (v1984_data + (v1946_data * v1982_data));
            }
            if (v1680_lead < 12) {
              float v1990_data = r6[7];
              float v1991_data = s3[7];
              float v1993_data = ir7[0];
              ir7[0] = (v1993_data + (v1990_data * v1991_data));
              float v1996_data = s3[19];
              float v1998_data = ir7[1];
              ir7[1] = (v1998_data + (v1990_data * v1996_data));
              float v2001_data = s3[31];
              float v2003_data = ir7[2];
              ir7[2] = (v2003_data + (v1990_data * v2001_data));
              float v2006_data = s3[43];
              float v2008_data = ir7[3];
              ir7[3] = (v2008_data + (v1990_data * v2006_data));
              float v2011_data = s3[55];
              float v2013_data = ir7[4];
              ir7[4] = (v2013_data + (v1990_data * v2011_data));
              float v2016_data = s3[67];
              float v2018_data = ir7[5];
              ir7[5] = (v2018_data + (v1990_data * v2016_data));
              float v2021_data = s3[79];
              float v2023_data = ir7[6];
              ir7[6] = (v2023_data + (v1990_data * v2021_data));
              float v2026_data = s3[91];
              float v2028_data = ir7[7];
              ir7[7] = (v2028_data + (v1990_data * v2026_data));
            }
            if (v1680_lead < 12) {
              float v2034_data = r6[8];
              float v2035_data = s3[8];
              float v2037_data = ir7[0];
              ir7[0] = (v2037_data + (v2034_data * v2035_data));
              float v2040_data = s3[20];
              float v2042_data = ir7[1];
              ir7[1] = (v2042_data + (v2034_data * v2040_data));
              float v2045_data = s3[32];
              float v2047_data = ir7[2];
              ir7[2] = (v2047_data + (v2034_data * v2045_data));
              float v2050_data = s3[44];
              float v2052_data = ir7[3];
              ir7[3] = (v2052_data + (v2034_data * v2050_data));
              float v2055_data = s3[56];
              float v2057_data = ir7[4];
              ir7[4] = (v2057_data + (v2034_data * v2055_data));
              float v2060_data = s3[68];
              float v2062_data = ir7[5];
              ir7[5] = (v2062_data + (v2034_data * v2060_data));
              float v2065_data = s3[80];
              float v2067_data = ir7[6];
              ir7[6] = (v2067_data + (v2034_data * v2065_data));
              float v2070_data = s3[92];
              float v2072_data = ir7[7];
              ir7[7] = (v2072_data + (v2034_data * v2070_data));
            }
            if (v1680_lead < 12) {
              float v2078_data = r6[9];
              float v2079_data = s3[9];
              float v2081_data = ir7[0];
              ir7[0] = (v2081_data + (v2078_data * v2079_data));
              float v2084_data = s3[21];
              float v2086_data = ir7[1];
              ir7[1] = (v2086_data + (v2078_data * v2084_data));
              float v2089_data = s3[33];
              float v2091_data = ir7[2];
              ir7[2] = (v2091_data + (v2078_data * v2089_data));
              float v2094_data = s3[45];
              float v2096_data = ir7[3];
              ir7[3] = (v2096_data + (v2078_data * v2094_data));
              float v2099_data = s3[57];
              float v2101_data = ir7[4];
              ir7[4] = (v2101_data + (v2078_data * v2099_data));
              float v2104_data = s3[69];
              float v2106_data = ir7[5];
              ir7[5] = (v2106_data + (v2078_data * v2104_data));
              float v2109_data = s3[81];
              float v2111_data = ir7[6];
              ir7[6] = (v2111_data + (v2078_data * v2109_data));
              float v2114_data = s3[93];
              float v2116_data = ir7[7];
              ir7[7] = (v2116_data + (v2078_data * v2114_data));
            }
            if (v1680_lead < 12) {
              float v2122_data = r6[10];
              float v2123_data = s3[10];
              float v2125_data = ir7[0];
              ir7[0] = (v2125_data + (v2122_data * v2123_data));
              float v2128_data = s3[22];
              float v2130_data = ir7[1];
              ir7[1] = (v2130_data + (v2122_data * v2128_data));
              float v2133_data = s3[34];
              float v2135_data = ir7[2];
              ir7[2] = (v2135_data + (v2122_data * v2133_data));
              float v2138_data = s3[46];
              float v2140_data = ir7[3];
              ir7[3] = (v2140_data + (v2122_data * v2138_data));
              float v2143_data = s3[58];
              float v2145_data = ir7[4];
              ir7[4] = (v2145_data + (v2122_data * v2143_data));
              float v2148_data = s3[70];
              float v2150_data = ir7[5];
              ir7[5] = (v2150_data + (v2122_data * v2148_data));
              float v2153_data = s3[82];
              float v2155_data = ir7[6];
              ir7[6] = (v2155_data + (v2122_data * v2153_data));
              float v2158_data = s3[94];
              float v2160_data = ir7[7];
              ir7[7] = (v2160_data + (v2122_data * v2158_data));
            }
            if (v1680_lead < 12) {
              float v2166_data = r6[11];
              float v2167_data = s3[11];
              float v2169_data = ir7[0];
              ir7[0] = (v2169_data + (v2166_data * v2167_data));
              float v2172_data = s3[23];
              float v2174_data = ir7[1];
              ir7[1] = (v2174_data + (v2166_data * v2172_data));
              float v2177_data = s3[35];
              float v2179_data = ir7[2];
              ir7[2] = (v2179_data + (v2166_data * v2177_data));
              float v2182_data = s3[47];
              float v2184_data = ir7[3];
              ir7[3] = (v2184_data + (v2166_data * v2182_data));
              float v2187_data = s3[59];
              float v2189_data = ir7[4];
              ir7[4] = (v2189_data + (v2166_data * v2187_data));
              float v2192_data = s3[71];
              float v2194_data = ir7[5];
              ir7[5] = (v2194_data + (v2166_data * v2192_data));
              float v2197_data = s3[83];
              float v2199_data = ir7[6];
              ir7[6] = (v2199_data + (v2166_data * v2197_data));
              float v2202_data = s3[95];
              float v2204_data = ir7[7];
              ir7[7] = (v2204_data + (v2166_data * v2202_data));
            }
            if (v1680_lead < 12) {
              #pragma unroll
              for (int32_t v2210_n1 = 0; v2210_n1 < 8; ++v2210_n1) {
                int32_t v2211_a = 0 + v2210_n1;
                float v2213_data = ir7[v2210_n1];
                int32_t v2214_a = 0 + v2210_n1;
                float v2216_data = r5[v2210_n1];
                int32_t v2218_a = 0 + v2210_n1;
                r7[v2210_n1] = (v2216_data + v2213_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r7);
          int32_t v2222_lead = threadIdx.x % 16;
          if (v2222_lead < 12) {
            #pragma unroll
            for (int32_t v2224_i1 = 0; v2224_i1 < 8; ++v2224_i1) {
              int32_t v2225_a = 0 + v2224_i1;
              float v2227_data = r7[v2224_i1];
              int32_t v2234_a = v2222_lead + (v2224_i1 * 12);
              glb_m0[v2234_a] = v2227_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

