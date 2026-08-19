// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5e7da3148f, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_5e7da3148f, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_5e7da3148f<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
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
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 12);
              float v26_data;
              {
                v26_data = glb_m0[v25_a];
              }
              int32_t v27_a = 0 + v18_i1;
              r1[v27_a] = v26_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          {
            // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 16)]
            float ir2[8]{};
            int32_t v30_lead = threadIdx.x % 16;
            if (v30_lead < 12) {
              float v32_data = r0[0];
              float v33_data = s0[0];
              float v35_data = ir2[0];
              ir2[0] = (v35_data + (v32_data * v33_data));
              float v38_data = s0[16];
              float v40_data = ir2[1];
              ir2[1] = (v40_data + (v32_data * v38_data));
              float v43_data = s0[32];
              float v45_data = ir2[2];
              ir2[2] = (v45_data + (v32_data * v43_data));
              float v48_data = s0[48];
              float v50_data = ir2[3];
              ir2[3] = (v50_data + (v32_data * v48_data));
              float v53_data = s0[64];
              float v55_data = ir2[4];
              ir2[4] = (v55_data + (v32_data * v53_data));
              float v58_data = s0[80];
              float v60_data = ir2[5];
              ir2[5] = (v60_data + (v32_data * v58_data));
              float v63_data = s0[96];
              float v65_data = ir2[6];
              ir2[6] = (v65_data + (v32_data * v63_data));
              float v68_data = s0[112];
              float v70_data = ir2[7];
              ir2[7] = (v70_data + (v32_data * v68_data));
            }
            if (v30_lead < 12) {
              float v76_data = r0[1];
              float v77_data = s0[1];
              float v79_data = ir2[0];
              ir2[0] = (v79_data + (v76_data * v77_data));
              float v82_data = s0[17];
              float v84_data = ir2[1];
              ir2[1] = (v84_data + (v76_data * v82_data));
              float v87_data = s0[33];
              float v89_data = ir2[2];
              ir2[2] = (v89_data + (v76_data * v87_data));
              float v92_data = s0[49];
              float v94_data = ir2[3];
              ir2[3] = (v94_data + (v76_data * v92_data));
              float v97_data = s0[65];
              float v99_data = ir2[4];
              ir2[4] = (v99_data + (v76_data * v97_data));
              float v102_data = s0[81];
              float v104_data = ir2[5];
              ir2[5] = (v104_data + (v76_data * v102_data));
              float v107_data = s0[97];
              float v109_data = ir2[6];
              ir2[6] = (v109_data + (v76_data * v107_data));
              float v112_data = s0[113];
              float v114_data = ir2[7];
              ir2[7] = (v114_data + (v76_data * v112_data));
            }
            if (v30_lead < 12) {
              float v120_data = r0[2];
              float v121_data = s0[2];
              float v123_data = ir2[0];
              ir2[0] = (v123_data + (v120_data * v121_data));
              float v126_data = s0[18];
              float v128_data = ir2[1];
              ir2[1] = (v128_data + (v120_data * v126_data));
              float v131_data = s0[34];
              float v133_data = ir2[2];
              ir2[2] = (v133_data + (v120_data * v131_data));
              float v136_data = s0[50];
              float v138_data = ir2[3];
              ir2[3] = (v138_data + (v120_data * v136_data));
              float v141_data = s0[66];
              float v143_data = ir2[4];
              ir2[4] = (v143_data + (v120_data * v141_data));
              float v146_data = s0[82];
              float v148_data = ir2[5];
              ir2[5] = (v148_data + (v120_data * v146_data));
              float v151_data = s0[98];
              float v153_data = ir2[6];
              ir2[6] = (v153_data + (v120_data * v151_data));
              float v156_data = s0[114];
              float v158_data = ir2[7];
              ir2[7] = (v158_data + (v120_data * v156_data));
            }
            if (v30_lead < 12) {
              float v164_data = r0[3];
              float v165_data = s0[3];
              float v167_data = ir2[0];
              ir2[0] = (v167_data + (v164_data * v165_data));
              float v170_data = s0[19];
              float v172_data = ir2[1];
              ir2[1] = (v172_data + (v164_data * v170_data));
              float v175_data = s0[35];
              float v177_data = ir2[2];
              ir2[2] = (v177_data + (v164_data * v175_data));
              float v180_data = s0[51];
              float v182_data = ir2[3];
              ir2[3] = (v182_data + (v164_data * v180_data));
              float v185_data = s0[67];
              float v187_data = ir2[4];
              ir2[4] = (v187_data + (v164_data * v185_data));
              float v190_data = s0[83];
              float v192_data = ir2[5];
              ir2[5] = (v192_data + (v164_data * v190_data));
              float v195_data = s0[99];
              float v197_data = ir2[6];
              ir2[6] = (v197_data + (v164_data * v195_data));
              float v200_data = s0[115];
              float v202_data = ir2[7];
              ir2[7] = (v202_data + (v164_data * v200_data));
            }
            if (v30_lead < 12) {
              float v208_data = r0[4];
              float v209_data = s0[4];
              float v211_data = ir2[0];
              ir2[0] = (v211_data + (v208_data * v209_data));
              float v214_data = s0[20];
              float v216_data = ir2[1];
              ir2[1] = (v216_data + (v208_data * v214_data));
              float v219_data = s0[36];
              float v221_data = ir2[2];
              ir2[2] = (v221_data + (v208_data * v219_data));
              float v224_data = s0[52];
              float v226_data = ir2[3];
              ir2[3] = (v226_data + (v208_data * v224_data));
              float v229_data = s0[68];
              float v231_data = ir2[4];
              ir2[4] = (v231_data + (v208_data * v229_data));
              float v234_data = s0[84];
              float v236_data = ir2[5];
              ir2[5] = (v236_data + (v208_data * v234_data));
              float v239_data = s0[100];
              float v241_data = ir2[6];
              ir2[6] = (v241_data + (v208_data * v239_data));
              float v244_data = s0[116];
              float v246_data = ir2[7];
              ir2[7] = (v246_data + (v208_data * v244_data));
            }
            if (v30_lead < 12) {
              float v252_data = r0[5];
              float v253_data = s0[5];
              float v255_data = ir2[0];
              ir2[0] = (v255_data + (v252_data * v253_data));
              float v258_data = s0[21];
              float v260_data = ir2[1];
              ir2[1] = (v260_data + (v252_data * v258_data));
              float v263_data = s0[37];
              float v265_data = ir2[2];
              ir2[2] = (v265_data + (v252_data * v263_data));
              float v268_data = s0[53];
              float v270_data = ir2[3];
              ir2[3] = (v270_data + (v252_data * v268_data));
              float v273_data = s0[69];
              float v275_data = ir2[4];
              ir2[4] = (v275_data + (v252_data * v273_data));
              float v278_data = s0[85];
              float v280_data = ir2[5];
              ir2[5] = (v280_data + (v252_data * v278_data));
              float v283_data = s0[101];
              float v285_data = ir2[6];
              ir2[6] = (v285_data + (v252_data * v283_data));
              float v288_data = s0[117];
              float v290_data = ir2[7];
              ir2[7] = (v290_data + (v252_data * v288_data));
            }
            if (v30_lead < 12) {
              float v296_data = r0[6];
              float v297_data = s0[6];
              float v299_data = ir2[0];
              ir2[0] = (v299_data + (v296_data * v297_data));
              float v302_data = s0[22];
              float v304_data = ir2[1];
              ir2[1] = (v304_data + (v296_data * v302_data));
              float v307_data = s0[38];
              float v309_data = ir2[2];
              ir2[2] = (v309_data + (v296_data * v307_data));
              float v312_data = s0[54];
              float v314_data = ir2[3];
              ir2[3] = (v314_data + (v296_data * v312_data));
              float v317_data = s0[70];
              float v319_data = ir2[4];
              ir2[4] = (v319_data + (v296_data * v317_data));
              float v322_data = s0[86];
              float v324_data = ir2[5];
              ir2[5] = (v324_data + (v296_data * v322_data));
              float v327_data = s0[102];
              float v329_data = ir2[6];
              ir2[6] = (v329_data + (v296_data * v327_data));
              float v332_data = s0[118];
              float v334_data = ir2[7];
              ir2[7] = (v334_data + (v296_data * v332_data));
            }
            if (v30_lead < 12) {
              float v340_data = r0[7];
              float v341_data = s0[7];
              float v343_data = ir2[0];
              ir2[0] = (v343_data + (v340_data * v341_data));
              float v346_data = s0[23];
              float v348_data = ir2[1];
              ir2[1] = (v348_data + (v340_data * v346_data));
              float v351_data = s0[39];
              float v353_data = ir2[2];
              ir2[2] = (v353_data + (v340_data * v351_data));
              float v356_data = s0[55];
              float v358_data = ir2[3];
              ir2[3] = (v358_data + (v340_data * v356_data));
              float v361_data = s0[71];
              float v363_data = ir2[4];
              ir2[4] = (v363_data + (v340_data * v361_data));
              float v366_data = s0[87];
              float v368_data = ir2[5];
              ir2[5] = (v368_data + (v340_data * v366_data));
              float v371_data = s0[103];
              float v373_data = ir2[6];
              ir2[6] = (v373_data + (v340_data * v371_data));
              float v376_data = s0[119];
              float v378_data = ir2[7];
              ir2[7] = (v378_data + (v340_data * v376_data));
            }
            if (v30_lead < 12) {
              float v384_data = r0[8];
              float v385_data = s0[8];
              float v387_data = ir2[0];
              ir2[0] = (v387_data + (v384_data * v385_data));
              float v390_data = s0[24];
              float v392_data = ir2[1];
              ir2[1] = (v392_data + (v384_data * v390_data));
              float v395_data = s0[40];
              float v397_data = ir2[2];
              ir2[2] = (v397_data + (v384_data * v395_data));
              float v400_data = s0[56];
              float v402_data = ir2[3];
              ir2[3] = (v402_data + (v384_data * v400_data));
              float v405_data = s0[72];
              float v407_data = ir2[4];
              ir2[4] = (v407_data + (v384_data * v405_data));
              float v410_data = s0[88];
              float v412_data = ir2[5];
              ir2[5] = (v412_data + (v384_data * v410_data));
              float v415_data = s0[104];
              float v417_data = ir2[6];
              ir2[6] = (v417_data + (v384_data * v415_data));
              float v420_data = s0[120];
              float v422_data = ir2[7];
              ir2[7] = (v422_data + (v384_data * v420_data));
            }
            if (v30_lead < 12) {
              float v428_data = r0[9];
              float v429_data = s0[9];
              float v431_data = ir2[0];
              ir2[0] = (v431_data + (v428_data * v429_data));
              float v434_data = s0[25];
              float v436_data = ir2[1];
              ir2[1] = (v436_data + (v428_data * v434_data));
              float v439_data = s0[41];
              float v441_data = ir2[2];
              ir2[2] = (v441_data + (v428_data * v439_data));
              float v444_data = s0[57];
              float v446_data = ir2[3];
              ir2[3] = (v446_data + (v428_data * v444_data));
              float v449_data = s0[73];
              float v451_data = ir2[4];
              ir2[4] = (v451_data + (v428_data * v449_data));
              float v454_data = s0[89];
              float v456_data = ir2[5];
              ir2[5] = (v456_data + (v428_data * v454_data));
              float v459_data = s0[105];
              float v461_data = ir2[6];
              ir2[6] = (v461_data + (v428_data * v459_data));
              float v464_data = s0[121];
              float v466_data = ir2[7];
              ir2[7] = (v466_data + (v428_data * v464_data));
            }
            if (v30_lead < 12) {
              float v472_data = r0[10];
              float v473_data = s0[10];
              float v475_data = ir2[0];
              ir2[0] = (v475_data + (v472_data * v473_data));
              float v478_data = s0[26];
              float v480_data = ir2[1];
              ir2[1] = (v480_data + (v472_data * v478_data));
              float v483_data = s0[42];
              float v485_data = ir2[2];
              ir2[2] = (v485_data + (v472_data * v483_data));
              float v488_data = s0[58];
              float v490_data = ir2[3];
              ir2[3] = (v490_data + (v472_data * v488_data));
              float v493_data = s0[74];
              float v495_data = ir2[4];
              ir2[4] = (v495_data + (v472_data * v493_data));
              float v498_data = s0[90];
              float v500_data = ir2[5];
              ir2[5] = (v500_data + (v472_data * v498_data));
              float v503_data = s0[106];
              float v505_data = ir2[6];
              ir2[6] = (v505_data + (v472_data * v503_data));
              float v508_data = s0[122];
              float v510_data = ir2[7];
              ir2[7] = (v510_data + (v472_data * v508_data));
            }
            if (v30_lead < 12) {
              float v516_data = r0[11];
              float v517_data = s0[11];
              float v519_data = ir2[0];
              ir2[0] = (v519_data + (v516_data * v517_data));
              float v522_data = s0[27];
              float v524_data = ir2[1];
              ir2[1] = (v524_data + (v516_data * v522_data));
              float v527_data = s0[43];
              float v529_data = ir2[2];
              ir2[2] = (v529_data + (v516_data * v527_data));
              float v532_data = s0[59];
              float v534_data = ir2[3];
              ir2[3] = (v534_data + (v516_data * v532_data));
              float v537_data = s0[75];
              float v539_data = ir2[4];
              ir2[4] = (v539_data + (v516_data * v537_data));
              float v542_data = s0[91];
              float v544_data = ir2[5];
              ir2[5] = (v544_data + (v516_data * v542_data));
              float v547_data = s0[107];
              float v549_data = ir2[6];
              ir2[6] = (v549_data + (v516_data * v547_data));
              float v552_data = s0[123];
              float v554_data = ir2[7];
              ir2[7] = (v554_data + (v516_data * v552_data));
            }
            if (v30_lead < 12) {
              float v560_data = r0[12];
              float v561_data = s0[12];
              float v563_data = ir2[0];
              ir2[0] = (v563_data + (v560_data * v561_data));
              float v566_data = s0[28];
              float v568_data = ir2[1];
              ir2[1] = (v568_data + (v560_data * v566_data));
              float v571_data = s0[44];
              float v573_data = ir2[2];
              ir2[2] = (v573_data + (v560_data * v571_data));
              float v576_data = s0[60];
              float v578_data = ir2[3];
              ir2[3] = (v578_data + (v560_data * v576_data));
              float v581_data = s0[76];
              float v583_data = ir2[4];
              ir2[4] = (v583_data + (v560_data * v581_data));
              float v586_data = s0[92];
              float v588_data = ir2[5];
              ir2[5] = (v588_data + (v560_data * v586_data));
              float v591_data = s0[108];
              float v593_data = ir2[6];
              ir2[6] = (v593_data + (v560_data * v591_data));
              float v596_data = s0[124];
              float v598_data = ir2[7];
              ir2[7] = (v598_data + (v560_data * v596_data));
            }
            if (v30_lead < 12) {
              float v604_data = r0[13];
              float v605_data = s0[13];
              float v607_data = ir2[0];
              ir2[0] = (v607_data + (v604_data * v605_data));
              float v610_data = s0[29];
              float v612_data = ir2[1];
              ir2[1] = (v612_data + (v604_data * v610_data));
              float v615_data = s0[45];
              float v617_data = ir2[2];
              ir2[2] = (v617_data + (v604_data * v615_data));
              float v620_data = s0[61];
              float v622_data = ir2[3];
              ir2[3] = (v622_data + (v604_data * v620_data));
              float v625_data = s0[77];
              float v627_data = ir2[4];
              ir2[4] = (v627_data + (v604_data * v625_data));
              float v630_data = s0[93];
              float v632_data = ir2[5];
              ir2[5] = (v632_data + (v604_data * v630_data));
              float v635_data = s0[109];
              float v637_data = ir2[6];
              ir2[6] = (v637_data + (v604_data * v635_data));
              float v640_data = s0[125];
              float v642_data = ir2[7];
              ir2[7] = (v642_data + (v604_data * v640_data));
            }
            if (v30_lead < 12) {
              float v648_data = r0[14];
              float v649_data = s0[14];
              float v651_data = ir2[0];
              ir2[0] = (v651_data + (v648_data * v649_data));
              float v654_data = s0[30];
              float v656_data = ir2[1];
              ir2[1] = (v656_data + (v648_data * v654_data));
              float v659_data = s0[46];
              float v661_data = ir2[2];
              ir2[2] = (v661_data + (v648_data * v659_data));
              float v664_data = s0[62];
              float v666_data = ir2[3];
              ir2[3] = (v666_data + (v648_data * v664_data));
              float v669_data = s0[78];
              float v671_data = ir2[4];
              ir2[4] = (v671_data + (v648_data * v669_data));
              float v674_data = s0[94];
              float v676_data = ir2[5];
              ir2[5] = (v676_data + (v648_data * v674_data));
              float v679_data = s0[110];
              float v681_data = ir2[6];
              ir2[6] = (v681_data + (v648_data * v679_data));
              float v684_data = s0[126];
              float v686_data = ir2[7];
              ir2[7] = (v686_data + (v648_data * v684_data));
            }
            if (v30_lead < 12) {
              float v692_data = r0[15];
              float v693_data = s0[15];
              float v695_data = ir2[0];
              ir2[0] = (v695_data + (v692_data * v693_data));
              float v698_data = s0[31];
              float v700_data = ir2[1];
              ir2[1] = (v700_data + (v692_data * v698_data));
              float v703_data = s0[47];
              float v705_data = ir2[2];
              ir2[2] = (v705_data + (v692_data * v703_data));
              float v708_data = s0[63];
              float v710_data = ir2[3];
              ir2[3] = (v710_data + (v692_data * v708_data));
              float v713_data = s0[79];
              float v715_data = ir2[4];
              ir2[4] = (v715_data + (v692_data * v713_data));
              float v718_data = s0[95];
              float v720_data = ir2[5];
              ir2[5] = (v720_data + (v692_data * v718_data));
              float v723_data = s0[111];
              float v725_data = ir2[6];
              ir2[6] = (v725_data + (v692_data * v723_data));
              float v728_data = s0[127];
              float v730_data = ir2[7];
              ir2[7] = (v730_data + (v692_data * v728_data));
            }
            if (v30_lead < 12) {
              #pragma unroll
              for (int32_t v736_n1 = 0; v736_n1 < 8; ++v736_n1) {
                int32_t v737_a = 0 + v736_n1;
                float v739_data = ir2[v736_n1];
                int32_t v740_a = 0 + v736_n1;
                float v742_data = r1[v736_n1];
                int32_t v744_a = 0 + v736_n1;
                r2[v736_n1] = (v742_data + v739_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r2);
          int32_t v748_lead = threadIdx.x % 16;
          if (v748_lead < 12) {
            #pragma unroll
            for (int32_t v750_i1 = 0; v750_i1 < 8; ++v750_i1) {
              int32_t v751_a = 0 + v750_i1;
              float v753_data = r2[v750_i1];
              int32_t v760_a = v748_lead + (v750_i1 * 12);
              glb_m0[v760_a] = v753_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

