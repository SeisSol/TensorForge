// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b59b6f027, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_4b59b6f027, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4b59b6f027<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(12×8) {4..16}×{0..8} strided
    // m1 16×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          alignas(16) float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            int32_t v15_a = (v6_lead + 4) - 4;
            int32_t v24_a = (v6_lead + 4) - 4;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v16_a = v8_i1 * 12;
              int32_t v17_a = v15_a + v16_a;
              float v27_data = __ldcg(&glb_m1[(v24_a + v16_a)]);
              int32_t v28_a = 0 + v8_i1;
              r0[v28_a] = v27_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(16, 28), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v6_lead < 12) {
            float v37_data = r0[0];
            float v38_data = s0[0];
            float v40_data = ir1[0];
            ir1[0] = (v40_data + (v37_data * v38_data));
            float v43_data = s0[16];
            float v45_data = ir1[1];
            ir1[1] = (v45_data + (v37_data * v43_data));
            float v48_data = s0[32];
            float v50_data = ir1[2];
            ir1[2] = (v50_data + (v37_data * v48_data));
            float v53_data = s0[48];
            float v55_data = ir1[3];
            ir1[3] = (v55_data + (v37_data * v53_data));
            float v58_data = s0[64];
            float v60_data = ir1[4];
            ir1[4] = (v60_data + (v37_data * v58_data));
            float v63_data = s0[80];
            float v65_data = ir1[5];
            ir1[5] = (v65_data + (v37_data * v63_data));
            float v68_data = s0[96];
            float v70_data = ir1[6];
            ir1[6] = (v70_data + (v37_data * v68_data));
            float v73_data = s0[112];
            float v75_data = ir1[7];
            ir1[7] = (v75_data + (v37_data * v73_data));
          }
          if (v6_lead < 12) {
            float v81_data = r0[1];
            float v82_data = s0[1];
            float v84_data = ir1[0];
            ir1[0] = (v84_data + (v81_data * v82_data));
            float v87_data = s0[17];
            float v89_data = ir1[1];
            ir1[1] = (v89_data + (v81_data * v87_data));
            float v92_data = s0[33];
            float v94_data = ir1[2];
            ir1[2] = (v94_data + (v81_data * v92_data));
            float v97_data = s0[49];
            float v99_data = ir1[3];
            ir1[3] = (v99_data + (v81_data * v97_data));
            float v102_data = s0[65];
            float v104_data = ir1[4];
            ir1[4] = (v104_data + (v81_data * v102_data));
            float v107_data = s0[81];
            float v109_data = ir1[5];
            ir1[5] = (v109_data + (v81_data * v107_data));
            float v112_data = s0[97];
            float v114_data = ir1[6];
            ir1[6] = (v114_data + (v81_data * v112_data));
            float v117_data = s0[113];
            float v119_data = ir1[7];
            ir1[7] = (v119_data + (v81_data * v117_data));
          }
          if (v6_lead < 12) {
            float v125_data = r0[2];
            float v126_data = s0[2];
            float v128_data = ir1[0];
            ir1[0] = (v128_data + (v125_data * v126_data));
            float v131_data = s0[18];
            float v133_data = ir1[1];
            ir1[1] = (v133_data + (v125_data * v131_data));
            float v136_data = s0[34];
            float v138_data = ir1[2];
            ir1[2] = (v138_data + (v125_data * v136_data));
            float v141_data = s0[50];
            float v143_data = ir1[3];
            ir1[3] = (v143_data + (v125_data * v141_data));
            float v146_data = s0[66];
            float v148_data = ir1[4];
            ir1[4] = (v148_data + (v125_data * v146_data));
            float v151_data = s0[82];
            float v153_data = ir1[5];
            ir1[5] = (v153_data + (v125_data * v151_data));
            float v156_data = s0[98];
            float v158_data = ir1[6];
            ir1[6] = (v158_data + (v125_data * v156_data));
            float v161_data = s0[114];
            float v163_data = ir1[7];
            ir1[7] = (v163_data + (v125_data * v161_data));
          }
          if (v6_lead < 12) {
            float v169_data = r0[3];
            float v170_data = s0[3];
            float v172_data = ir1[0];
            ir1[0] = (v172_data + (v169_data * v170_data));
            float v175_data = s0[19];
            float v177_data = ir1[1];
            ir1[1] = (v177_data + (v169_data * v175_data));
            float v180_data = s0[35];
            float v182_data = ir1[2];
            ir1[2] = (v182_data + (v169_data * v180_data));
            float v185_data = s0[51];
            float v187_data = ir1[3];
            ir1[3] = (v187_data + (v169_data * v185_data));
            float v190_data = s0[67];
            float v192_data = ir1[4];
            ir1[4] = (v192_data + (v169_data * v190_data));
            float v195_data = s0[83];
            float v197_data = ir1[5];
            ir1[5] = (v197_data + (v169_data * v195_data));
            float v200_data = s0[99];
            float v202_data = ir1[6];
            ir1[6] = (v202_data + (v169_data * v200_data));
            float v205_data = s0[115];
            float v207_data = ir1[7];
            ir1[7] = (v207_data + (v169_data * v205_data));
          }
          if (v6_lead < 12) {
            float v213_data = r0[4];
            float v214_data = s0[4];
            float v216_data = ir1[0];
            ir1[0] = (v216_data + (v213_data * v214_data));
            float v219_data = s0[20];
            float v221_data = ir1[1];
            ir1[1] = (v221_data + (v213_data * v219_data));
            float v224_data = s0[36];
            float v226_data = ir1[2];
            ir1[2] = (v226_data + (v213_data * v224_data));
            float v229_data = s0[52];
            float v231_data = ir1[3];
            ir1[3] = (v231_data + (v213_data * v229_data));
            float v234_data = s0[68];
            float v236_data = ir1[4];
            ir1[4] = (v236_data + (v213_data * v234_data));
            float v239_data = s0[84];
            float v241_data = ir1[5];
            ir1[5] = (v241_data + (v213_data * v239_data));
            float v244_data = s0[100];
            float v246_data = ir1[6];
            ir1[6] = (v246_data + (v213_data * v244_data));
            float v249_data = s0[116];
            float v251_data = ir1[7];
            ir1[7] = (v251_data + (v213_data * v249_data));
          }
          if (v6_lead < 12) {
            float v257_data = r0[5];
            float v258_data = s0[5];
            float v260_data = ir1[0];
            ir1[0] = (v260_data + (v257_data * v258_data));
            float v263_data = s0[21];
            float v265_data = ir1[1];
            ir1[1] = (v265_data + (v257_data * v263_data));
            float v268_data = s0[37];
            float v270_data = ir1[2];
            ir1[2] = (v270_data + (v257_data * v268_data));
            float v273_data = s0[53];
            float v275_data = ir1[3];
            ir1[3] = (v275_data + (v257_data * v273_data));
            float v278_data = s0[69];
            float v280_data = ir1[4];
            ir1[4] = (v280_data + (v257_data * v278_data));
            float v283_data = s0[85];
            float v285_data = ir1[5];
            ir1[5] = (v285_data + (v257_data * v283_data));
            float v288_data = s0[101];
            float v290_data = ir1[6];
            ir1[6] = (v290_data + (v257_data * v288_data));
            float v293_data = s0[117];
            float v295_data = ir1[7];
            ir1[7] = (v295_data + (v257_data * v293_data));
          }
          if (v6_lead < 12) {
            float v301_data = r0[6];
            float v302_data = s0[6];
            float v304_data = ir1[0];
            ir1[0] = (v304_data + (v301_data * v302_data));
            float v307_data = s0[22];
            float v309_data = ir1[1];
            ir1[1] = (v309_data + (v301_data * v307_data));
            float v312_data = s0[38];
            float v314_data = ir1[2];
            ir1[2] = (v314_data + (v301_data * v312_data));
            float v317_data = s0[54];
            float v319_data = ir1[3];
            ir1[3] = (v319_data + (v301_data * v317_data));
            float v322_data = s0[70];
            float v324_data = ir1[4];
            ir1[4] = (v324_data + (v301_data * v322_data));
            float v327_data = s0[86];
            float v329_data = ir1[5];
            ir1[5] = (v329_data + (v301_data * v327_data));
            float v332_data = s0[102];
            float v334_data = ir1[6];
            ir1[6] = (v334_data + (v301_data * v332_data));
            float v337_data = s0[118];
            float v339_data = ir1[7];
            ir1[7] = (v339_data + (v301_data * v337_data));
          }
          if (v6_lead < 12) {
            float v345_data = r0[7];
            float v346_data = s0[7];
            float v348_data = ir1[0];
            ir1[0] = (v348_data + (v345_data * v346_data));
            float v351_data = s0[23];
            float v353_data = ir1[1];
            ir1[1] = (v353_data + (v345_data * v351_data));
            float v356_data = s0[39];
            float v358_data = ir1[2];
            ir1[2] = (v358_data + (v345_data * v356_data));
            float v361_data = s0[55];
            float v363_data = ir1[3];
            ir1[3] = (v363_data + (v345_data * v361_data));
            float v366_data = s0[71];
            float v368_data = ir1[4];
            ir1[4] = (v368_data + (v345_data * v366_data));
            float v371_data = s0[87];
            float v373_data = ir1[5];
            ir1[5] = (v373_data + (v345_data * v371_data));
            float v376_data = s0[103];
            float v378_data = ir1[6];
            ir1[6] = (v378_data + (v345_data * v376_data));
            float v381_data = s0[119];
            float v383_data = ir1[7];
            ir1[7] = (v383_data + (v345_data * v381_data));
          }
          if (v6_lead < 12) {
            float v389_data = r0[8];
            float v390_data = s0[8];
            float v392_data = ir1[0];
            ir1[0] = (v392_data + (v389_data * v390_data));
            float v395_data = s0[24];
            float v397_data = ir1[1];
            ir1[1] = (v397_data + (v389_data * v395_data));
            float v400_data = s0[40];
            float v402_data = ir1[2];
            ir1[2] = (v402_data + (v389_data * v400_data));
            float v405_data = s0[56];
            float v407_data = ir1[3];
            ir1[3] = (v407_data + (v389_data * v405_data));
            float v410_data = s0[72];
            float v412_data = ir1[4];
            ir1[4] = (v412_data + (v389_data * v410_data));
            float v415_data = s0[88];
            float v417_data = ir1[5];
            ir1[5] = (v417_data + (v389_data * v415_data));
            float v420_data = s0[104];
            float v422_data = ir1[6];
            ir1[6] = (v422_data + (v389_data * v420_data));
            float v425_data = s0[120];
            float v427_data = ir1[7];
            ir1[7] = (v427_data + (v389_data * v425_data));
          }
          if (v6_lead < 12) {
            float v433_data = r0[9];
            float v434_data = s0[9];
            float v436_data = ir1[0];
            ir1[0] = (v436_data + (v433_data * v434_data));
            float v439_data = s0[25];
            float v441_data = ir1[1];
            ir1[1] = (v441_data + (v433_data * v439_data));
            float v444_data = s0[41];
            float v446_data = ir1[2];
            ir1[2] = (v446_data + (v433_data * v444_data));
            float v449_data = s0[57];
            float v451_data = ir1[3];
            ir1[3] = (v451_data + (v433_data * v449_data));
            float v454_data = s0[73];
            float v456_data = ir1[4];
            ir1[4] = (v456_data + (v433_data * v454_data));
            float v459_data = s0[89];
            float v461_data = ir1[5];
            ir1[5] = (v461_data + (v433_data * v459_data));
            float v464_data = s0[105];
            float v466_data = ir1[6];
            ir1[6] = (v466_data + (v433_data * v464_data));
            float v469_data = s0[121];
            float v471_data = ir1[7];
            ir1[7] = (v471_data + (v433_data * v469_data));
          }
          if (v6_lead < 12) {
            float v477_data = r0[10];
            float v478_data = s0[10];
            float v480_data = ir1[0];
            ir1[0] = (v480_data + (v477_data * v478_data));
            float v483_data = s0[26];
            float v485_data = ir1[1];
            ir1[1] = (v485_data + (v477_data * v483_data));
            float v488_data = s0[42];
            float v490_data = ir1[2];
            ir1[2] = (v490_data + (v477_data * v488_data));
            float v493_data = s0[58];
            float v495_data = ir1[3];
            ir1[3] = (v495_data + (v477_data * v493_data));
            float v498_data = s0[74];
            float v500_data = ir1[4];
            ir1[4] = (v500_data + (v477_data * v498_data));
            float v503_data = s0[90];
            float v505_data = ir1[5];
            ir1[5] = (v505_data + (v477_data * v503_data));
            float v508_data = s0[106];
            float v510_data = ir1[6];
            ir1[6] = (v510_data + (v477_data * v508_data));
            float v513_data = s0[122];
            float v515_data = ir1[7];
            ir1[7] = (v515_data + (v477_data * v513_data));
          }
          if (v6_lead < 12) {
            float v521_data = r0[11];
            float v522_data = s0[11];
            float v524_data = ir1[0];
            ir1[0] = (v524_data + (v521_data * v522_data));
            float v527_data = s0[27];
            float v529_data = ir1[1];
            ir1[1] = (v529_data + (v521_data * v527_data));
            float v532_data = s0[43];
            float v534_data = ir1[2];
            ir1[2] = (v534_data + (v521_data * v532_data));
            float v537_data = s0[59];
            float v539_data = ir1[3];
            ir1[3] = (v539_data + (v521_data * v537_data));
            float v542_data = s0[75];
            float v544_data = ir1[4];
            ir1[4] = (v544_data + (v521_data * v542_data));
            float v547_data = s0[91];
            float v549_data = ir1[5];
            ir1[5] = (v549_data + (v521_data * v547_data));
            float v552_data = s0[107];
            float v554_data = ir1[6];
            ir1[6] = (v554_data + (v521_data * v552_data));
            float v557_data = s0[123];
            float v559_data = ir1[7];
            ir1[7] = (v559_data + (v521_data * v557_data));
          }
          if (v6_lead < 12) {
            float v565_data = r0[12];
            float v566_data = s0[12];
            float v568_data = ir1[0];
            ir1[0] = (v568_data + (v565_data * v566_data));
            float v571_data = s0[28];
            float v573_data = ir1[1];
            ir1[1] = (v573_data + (v565_data * v571_data));
            float v576_data = s0[44];
            float v578_data = ir1[2];
            ir1[2] = (v578_data + (v565_data * v576_data));
            float v581_data = s0[60];
            float v583_data = ir1[3];
            ir1[3] = (v583_data + (v565_data * v581_data));
            float v586_data = s0[76];
            float v588_data = ir1[4];
            ir1[4] = (v588_data + (v565_data * v586_data));
            float v591_data = s0[92];
            float v593_data = ir1[5];
            ir1[5] = (v593_data + (v565_data * v591_data));
            float v596_data = s0[108];
            float v598_data = ir1[6];
            ir1[6] = (v598_data + (v565_data * v596_data));
            float v601_data = s0[124];
            float v603_data = ir1[7];
            ir1[7] = (v603_data + (v565_data * v601_data));
          }
          if (v6_lead < 12) {
            float v609_data = r0[13];
            float v610_data = s0[13];
            float v612_data = ir1[0];
            ir1[0] = (v612_data + (v609_data * v610_data));
            float v615_data = s0[29];
            float v617_data = ir1[1];
            ir1[1] = (v617_data + (v609_data * v615_data));
            float v620_data = s0[45];
            float v622_data = ir1[2];
            ir1[2] = (v622_data + (v609_data * v620_data));
            float v625_data = s0[61];
            float v627_data = ir1[3];
            ir1[3] = (v627_data + (v609_data * v625_data));
            float v630_data = s0[77];
            float v632_data = ir1[4];
            ir1[4] = (v632_data + (v609_data * v630_data));
            float v635_data = s0[93];
            float v637_data = ir1[5];
            ir1[5] = (v637_data + (v609_data * v635_data));
            float v640_data = s0[109];
            float v642_data = ir1[6];
            ir1[6] = (v642_data + (v609_data * v640_data));
            float v645_data = s0[125];
            float v647_data = ir1[7];
            ir1[7] = (v647_data + (v609_data * v645_data));
          }
          if (v6_lead < 12) {
            float v653_data = r0[14];
            float v654_data = s0[14];
            float v656_data = ir1[0];
            ir1[0] = (v656_data + (v653_data * v654_data));
            float v659_data = s0[30];
            float v661_data = ir1[1];
            ir1[1] = (v661_data + (v653_data * v659_data));
            float v664_data = s0[46];
            float v666_data = ir1[2];
            ir1[2] = (v666_data + (v653_data * v664_data));
            float v669_data = s0[62];
            float v671_data = ir1[3];
            ir1[3] = (v671_data + (v653_data * v669_data));
            float v674_data = s0[78];
            float v676_data = ir1[4];
            ir1[4] = (v676_data + (v653_data * v674_data));
            float v679_data = s0[94];
            float v681_data = ir1[5];
            ir1[5] = (v681_data + (v653_data * v679_data));
            float v684_data = s0[110];
            float v686_data = ir1[6];
            ir1[6] = (v686_data + (v653_data * v684_data));
            float v689_data = s0[126];
            float v691_data = ir1[7];
            ir1[7] = (v691_data + (v653_data * v689_data));
          }
          if (v6_lead < 12) {
            float v697_data = r0[15];
            float v698_data = s0[15];
            float v700_data = ir1[0];
            ir1[0] = (v700_data + (v697_data * v698_data));
            float v703_data = s0[31];
            float v705_data = ir1[1];
            ir1[1] = (v705_data + (v697_data * v703_data));
            float v708_data = s0[47];
            float v710_data = ir1[2];
            ir1[2] = (v710_data + (v697_data * v708_data));
            float v713_data = s0[63];
            float v715_data = ir1[3];
            ir1[3] = (v715_data + (v697_data * v713_data));
            float v718_data = s0[79];
            float v720_data = ir1[4];
            ir1[4] = (v720_data + (v697_data * v718_data));
            float v723_data = s0[95];
            float v725_data = ir1[5];
            ir1[5] = (v725_data + (v697_data * v723_data));
            float v728_data = s0[111];
            float v730_data = ir1[6];
            ir1[6] = (v730_data + (v697_data * v728_data));
            float v733_data = s0[127];
            float v735_data = ir1[7];
            ir1[7] = (v735_data + (v697_data * v733_data));
          }
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v741_n1 = 0; v741_n1 < 8; ++v741_n1) {
              int32_t v742_a = 0 + v741_n1;
              float v744_data = ir1[v741_n1];
              r1[v741_n1] = v744_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v6_lead < 12) {
            int32_t v760_a = ((v6_lead + 16_i32) + -12) - 4;
            #pragma unroll
            for (int32_t v750_i1 = 0; v750_i1 < 8; ++v750_i1) {
              int32_t v751_a = 0 + v750_i1;
              float v753_data = r1[v750_i1];
              int32_t v762_a = v760_a + (v750_i1 * 12);
              glb_m0[v762_a] = v753_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

