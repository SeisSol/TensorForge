// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3e24e7feaf, block.x * block.y * block.z, 2816 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_3e24e7feaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 2816 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3e24e7feaf<<<grid,block,2816 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(6×12) {0..6}×{0..12} strided
    // m1 32×32(12×12) {0..12}×{0..12} strided
    // m2 32×32(6×12) {0..6}×{0..12} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // m4 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v8_lead = threadIdx.x % 16;
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 12; ++v10_i1) {
              int32_t v16_a = v10_i1 * 6;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __ldcg(&glb_m0[(v8_lead + v16_a)]);
              int32_t v26_a = 0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m1[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 9; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              int32_t v39_a = v33_i1 * 6;
              int32_t v40_a = v8_lead + v39_a;
              float v48_data = __ldcg(&glb_m2[(v8_lead + v39_a)]);
              int32_t v49_a = 0 + v33_i1;
              r2[v49_a] = v48_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v8_lead < 6) {
            float v55_data = r0[0];
            float v56_data = s0[0];
            float v58_data = r1[0];
            r1[0] = (v58_data + (v55_data * v56_data));
            float v61_data = s0[12];
            float v63_data = r1[1];
            r1[1] = (v63_data + (v55_data * v61_data));
            float v66_data = s0[24];
            float v68_data = r1[2];
            r1[2] = (v68_data + (v55_data * v66_data));
            float v71_data = s0[36];
            float v73_data = r1[3];
            r1[3] = (v73_data + (v55_data * v71_data));
            float v76_data = s0[48];
            float v78_data = r1[4];
            r1[4] = (v78_data + (v55_data * v76_data));
            float v81_data = s0[60];
            float v83_data = r1[5];
            r1[5] = (v83_data + (v55_data * v81_data));
            float v86_data = s0[72];
            float v88_data = r1[6];
            r1[6] = (v88_data + (v55_data * v86_data));
            float v91_data = s0[84];
            float v93_data = r1[7];
            r1[7] = (v93_data + (v55_data * v91_data));
            float v96_data = s0[96];
            float v98_data = r1[8];
            r1[8] = (v98_data + (v55_data * v96_data));
            float v101_data = s0[108];
            float v103_data = r1[9];
            r1[9] = (v103_data + (v55_data * v101_data));
            float v106_data = s0[120];
            float v108_data = r1[10];
            r1[10] = (v108_data + (v55_data * v106_data));
            float v111_data = s0[132];
            float v113_data = r1[11];
            r1[11] = (v113_data + (v55_data * v111_data));
          }
          if (v8_lead < 6) {
            float v119_data = r0[1];
            float v120_data = s0[1];
            float v122_data = r1[0];
            r1[0] = (v122_data + (v119_data * v120_data));
            float v125_data = s0[13];
            float v127_data = r1[1];
            r1[1] = (v127_data + (v119_data * v125_data));
            float v130_data = s0[25];
            float v132_data = r1[2];
            r1[2] = (v132_data + (v119_data * v130_data));
            float v135_data = s0[37];
            float v137_data = r1[3];
            r1[3] = (v137_data + (v119_data * v135_data));
            float v140_data = s0[49];
            float v142_data = r1[4];
            r1[4] = (v142_data + (v119_data * v140_data));
            float v145_data = s0[61];
            float v147_data = r1[5];
            r1[5] = (v147_data + (v119_data * v145_data));
            float v150_data = s0[73];
            float v152_data = r1[6];
            r1[6] = (v152_data + (v119_data * v150_data));
            float v155_data = s0[85];
            float v157_data = r1[7];
            r1[7] = (v157_data + (v119_data * v155_data));
            float v160_data = s0[97];
            float v162_data = r1[8];
            r1[8] = (v162_data + (v119_data * v160_data));
            float v165_data = s0[109];
            float v167_data = r1[9];
            r1[9] = (v167_data + (v119_data * v165_data));
            float v170_data = s0[121];
            float v172_data = r1[10];
            r1[10] = (v172_data + (v119_data * v170_data));
            float v175_data = s0[133];
            float v177_data = r1[11];
            r1[11] = (v177_data + (v119_data * v175_data));
          }
          if (v8_lead < 6) {
            float v183_data = r0[2];
            float v184_data = s0[2];
            float v186_data = r1[0];
            r1[0] = (v186_data + (v183_data * v184_data));
            float v189_data = s0[14];
            float v191_data = r1[1];
            r1[1] = (v191_data + (v183_data * v189_data));
            float v194_data = s0[26];
            float v196_data = r1[2];
            r1[2] = (v196_data + (v183_data * v194_data));
            float v199_data = s0[38];
            float v201_data = r1[3];
            r1[3] = (v201_data + (v183_data * v199_data));
            float v204_data = s0[50];
            float v206_data = r1[4];
            r1[4] = (v206_data + (v183_data * v204_data));
            float v209_data = s0[62];
            float v211_data = r1[5];
            r1[5] = (v211_data + (v183_data * v209_data));
            float v214_data = s0[74];
            float v216_data = r1[6];
            r1[6] = (v216_data + (v183_data * v214_data));
            float v219_data = s0[86];
            float v221_data = r1[7];
            r1[7] = (v221_data + (v183_data * v219_data));
            float v224_data = s0[98];
            float v226_data = r1[8];
            r1[8] = (v226_data + (v183_data * v224_data));
            float v229_data = s0[110];
            float v231_data = r1[9];
            r1[9] = (v231_data + (v183_data * v229_data));
            float v234_data = s0[122];
            float v236_data = r1[10];
            r1[10] = (v236_data + (v183_data * v234_data));
            float v239_data = s0[134];
            float v241_data = r1[11];
            r1[11] = (v241_data + (v183_data * v239_data));
          }
          if (v8_lead < 6) {
            float v247_data = r0[3];
            float v248_data = s0[3];
            float v250_data = r1[0];
            r1[0] = (v250_data + (v247_data * v248_data));
            float v253_data = s0[15];
            float v255_data = r1[1];
            r1[1] = (v255_data + (v247_data * v253_data));
            float v258_data = s0[27];
            float v260_data = r1[2];
            r1[2] = (v260_data + (v247_data * v258_data));
            float v263_data = s0[39];
            float v265_data = r1[3];
            r1[3] = (v265_data + (v247_data * v263_data));
            float v268_data = s0[51];
            float v270_data = r1[4];
            r1[4] = (v270_data + (v247_data * v268_data));
            float v273_data = s0[63];
            float v275_data = r1[5];
            r1[5] = (v275_data + (v247_data * v273_data));
            float v278_data = s0[75];
            float v280_data = r1[6];
            r1[6] = (v280_data + (v247_data * v278_data));
            float v283_data = s0[87];
            float v285_data = r1[7];
            r1[7] = (v285_data + (v247_data * v283_data));
            float v288_data = s0[99];
            float v290_data = r1[8];
            r1[8] = (v290_data + (v247_data * v288_data));
            float v293_data = s0[111];
            float v295_data = r1[9];
            r1[9] = (v295_data + (v247_data * v293_data));
            float v298_data = s0[123];
            float v300_data = r1[10];
            r1[10] = (v300_data + (v247_data * v298_data));
            float v303_data = s0[135];
            float v305_data = r1[11];
            r1[11] = (v305_data + (v247_data * v303_data));
          }
          if (v8_lead < 6) {
            float v311_data = r0[4];
            float v312_data = s0[4];
            float v314_data = r1[0];
            r1[0] = (v314_data + (v311_data * v312_data));
            float v317_data = s0[16];
            float v319_data = r1[1];
            r1[1] = (v319_data + (v311_data * v317_data));
            float v322_data = s0[28];
            float v324_data = r1[2];
            r1[2] = (v324_data + (v311_data * v322_data));
            float v327_data = s0[40];
            float v329_data = r1[3];
            r1[3] = (v329_data + (v311_data * v327_data));
            float v332_data = s0[52];
            float v334_data = r1[4];
            r1[4] = (v334_data + (v311_data * v332_data));
            float v337_data = s0[64];
            float v339_data = r1[5];
            r1[5] = (v339_data + (v311_data * v337_data));
            float v342_data = s0[76];
            float v344_data = r1[6];
            r1[6] = (v344_data + (v311_data * v342_data));
            float v347_data = s0[88];
            float v349_data = r1[7];
            r1[7] = (v349_data + (v311_data * v347_data));
            float v352_data = s0[100];
            float v354_data = r1[8];
            r1[8] = (v354_data + (v311_data * v352_data));
            float v357_data = s0[112];
            float v359_data = r1[9];
            r1[9] = (v359_data + (v311_data * v357_data));
            float v362_data = s0[124];
            float v364_data = r1[10];
            r1[10] = (v364_data + (v311_data * v362_data));
            float v367_data = s0[136];
            float v369_data = r1[11];
            r1[11] = (v369_data + (v311_data * v367_data));
          }
          if (v8_lead < 6) {
            float v375_data = r0[5];
            float v376_data = s0[5];
            float v378_data = r1[0];
            r1[0] = (v378_data + (v375_data * v376_data));
            float v381_data = s0[17];
            float v383_data = r1[1];
            r1[1] = (v383_data + (v375_data * v381_data));
            float v386_data = s0[29];
            float v388_data = r1[2];
            r1[2] = (v388_data + (v375_data * v386_data));
            float v391_data = s0[41];
            float v393_data = r1[3];
            r1[3] = (v393_data + (v375_data * v391_data));
            float v396_data = s0[53];
            float v398_data = r1[4];
            r1[4] = (v398_data + (v375_data * v396_data));
            float v401_data = s0[65];
            float v403_data = r1[5];
            r1[5] = (v403_data + (v375_data * v401_data));
            float v406_data = s0[77];
            float v408_data = r1[6];
            r1[6] = (v408_data + (v375_data * v406_data));
            float v411_data = s0[89];
            float v413_data = r1[7];
            r1[7] = (v413_data + (v375_data * v411_data));
            float v416_data = s0[101];
            float v418_data = r1[8];
            r1[8] = (v418_data + (v375_data * v416_data));
            float v421_data = s0[113];
            float v423_data = r1[9];
            r1[9] = (v423_data + (v375_data * v421_data));
            float v426_data = s0[125];
            float v428_data = r1[10];
            r1[10] = (v428_data + (v375_data * v426_data));
            float v431_data = s0[137];
            float v433_data = r1[11];
            r1[11] = (v433_data + (v375_data * v431_data));
          }
          if (v8_lead < 6) {
            float v439_data = r0[6];
            float v440_data = s0[6];
            float v442_data = r1[0];
            r1[0] = (v442_data + (v439_data * v440_data));
            float v445_data = s0[18];
            float v447_data = r1[1];
            r1[1] = (v447_data + (v439_data * v445_data));
            float v450_data = s0[30];
            float v452_data = r1[2];
            r1[2] = (v452_data + (v439_data * v450_data));
            float v455_data = s0[42];
            float v457_data = r1[3];
            r1[3] = (v457_data + (v439_data * v455_data));
            float v460_data = s0[54];
            float v462_data = r1[4];
            r1[4] = (v462_data + (v439_data * v460_data));
            float v465_data = s0[66];
            float v467_data = r1[5];
            r1[5] = (v467_data + (v439_data * v465_data));
            float v470_data = s0[78];
            float v472_data = r1[6];
            r1[6] = (v472_data + (v439_data * v470_data));
            float v475_data = s0[90];
            float v477_data = r1[7];
            r1[7] = (v477_data + (v439_data * v475_data));
            float v480_data = s0[102];
            float v482_data = r1[8];
            r1[8] = (v482_data + (v439_data * v480_data));
            float v485_data = s0[114];
            float v487_data = r1[9];
            r1[9] = (v487_data + (v439_data * v485_data));
            float v490_data = s0[126];
            float v492_data = r1[10];
            r1[10] = (v492_data + (v439_data * v490_data));
            float v495_data = s0[138];
            float v497_data = r1[11];
            r1[11] = (v497_data + (v439_data * v495_data));
          }
          if (v8_lead < 6) {
            float v503_data = r0[7];
            float v504_data = s0[7];
            float v506_data = r1[0];
            r1[0] = (v506_data + (v503_data * v504_data));
            float v509_data = s0[19];
            float v511_data = r1[1];
            r1[1] = (v511_data + (v503_data * v509_data));
            float v514_data = s0[31];
            float v516_data = r1[2];
            r1[2] = (v516_data + (v503_data * v514_data));
            float v519_data = s0[43];
            float v521_data = r1[3];
            r1[3] = (v521_data + (v503_data * v519_data));
            float v524_data = s0[55];
            float v526_data = r1[4];
            r1[4] = (v526_data + (v503_data * v524_data));
            float v529_data = s0[67];
            float v531_data = r1[5];
            r1[5] = (v531_data + (v503_data * v529_data));
            float v534_data = s0[79];
            float v536_data = r1[6];
            r1[6] = (v536_data + (v503_data * v534_data));
            float v539_data = s0[91];
            float v541_data = r1[7];
            r1[7] = (v541_data + (v503_data * v539_data));
            float v544_data = s0[103];
            float v546_data = r1[8];
            r1[8] = (v546_data + (v503_data * v544_data));
            float v549_data = s0[115];
            float v551_data = r1[9];
            r1[9] = (v551_data + (v503_data * v549_data));
            float v554_data = s0[127];
            float v556_data = r1[10];
            r1[10] = (v556_data + (v503_data * v554_data));
            float v559_data = s0[139];
            float v561_data = r1[11];
            r1[11] = (v561_data + (v503_data * v559_data));
          }
          if (v8_lead < 6) {
            float v567_data = r0[8];
            float v568_data = s0[8];
            float v570_data = r1[0];
            r1[0] = (v570_data + (v567_data * v568_data));
            float v573_data = s0[20];
            float v575_data = r1[1];
            r1[1] = (v575_data + (v567_data * v573_data));
            float v578_data = s0[32];
            float v580_data = r1[2];
            r1[2] = (v580_data + (v567_data * v578_data));
            float v583_data = s0[44];
            float v585_data = r1[3];
            r1[3] = (v585_data + (v567_data * v583_data));
            float v588_data = s0[56];
            float v590_data = r1[4];
            r1[4] = (v590_data + (v567_data * v588_data));
            float v593_data = s0[68];
            float v595_data = r1[5];
            r1[5] = (v595_data + (v567_data * v593_data));
            float v598_data = s0[80];
            float v600_data = r1[6];
            r1[6] = (v600_data + (v567_data * v598_data));
            float v603_data = s0[92];
            float v605_data = r1[7];
            r1[7] = (v605_data + (v567_data * v603_data));
            float v608_data = s0[104];
            float v610_data = r1[8];
            r1[8] = (v610_data + (v567_data * v608_data));
            float v613_data = s0[116];
            float v615_data = r1[9];
            r1[9] = (v615_data + (v567_data * v613_data));
            float v618_data = s0[128];
            float v620_data = r1[10];
            r1[10] = (v620_data + (v567_data * v618_data));
            float v623_data = s0[140];
            float v625_data = r1[11];
            r1[11] = (v625_data + (v567_data * v623_data));
          }
          if (v8_lead < 6) {
            float v631_data = r0[9];
            float v632_data = s0[9];
            float v634_data = r1[0];
            r1[0] = (v634_data + (v631_data * v632_data));
            float v637_data = s0[21];
            float v639_data = r1[1];
            r1[1] = (v639_data + (v631_data * v637_data));
            float v642_data = s0[33];
            float v644_data = r1[2];
            r1[2] = (v644_data + (v631_data * v642_data));
            float v647_data = s0[45];
            float v649_data = r1[3];
            r1[3] = (v649_data + (v631_data * v647_data));
            float v652_data = s0[57];
            float v654_data = r1[4];
            r1[4] = (v654_data + (v631_data * v652_data));
            float v657_data = s0[69];
            float v659_data = r1[5];
            r1[5] = (v659_data + (v631_data * v657_data));
            float v662_data = s0[81];
            float v664_data = r1[6];
            r1[6] = (v664_data + (v631_data * v662_data));
            float v667_data = s0[93];
            float v669_data = r1[7];
            r1[7] = (v669_data + (v631_data * v667_data));
            float v672_data = s0[105];
            float v674_data = r1[8];
            r1[8] = (v674_data + (v631_data * v672_data));
            float v677_data = s0[117];
            float v679_data = r1[9];
            r1[9] = (v679_data + (v631_data * v677_data));
            float v682_data = s0[129];
            float v684_data = r1[10];
            r1[10] = (v684_data + (v631_data * v682_data));
            float v687_data = s0[141];
            float v689_data = r1[11];
            r1[11] = (v689_data + (v631_data * v687_data));
          }
          if (v8_lead < 6) {
            float v695_data = r0[10];
            float v696_data = s0[10];
            float v698_data = r1[0];
            r1[0] = (v698_data + (v695_data * v696_data));
            float v701_data = s0[22];
            float v703_data = r1[1];
            r1[1] = (v703_data + (v695_data * v701_data));
            float v706_data = s0[34];
            float v708_data = r1[2];
            r1[2] = (v708_data + (v695_data * v706_data));
            float v711_data = s0[46];
            float v713_data = r1[3];
            r1[3] = (v713_data + (v695_data * v711_data));
            float v716_data = s0[58];
            float v718_data = r1[4];
            r1[4] = (v718_data + (v695_data * v716_data));
            float v721_data = s0[70];
            float v723_data = r1[5];
            r1[5] = (v723_data + (v695_data * v721_data));
            float v726_data = s0[82];
            float v728_data = r1[6];
            r1[6] = (v728_data + (v695_data * v726_data));
            float v731_data = s0[94];
            float v733_data = r1[7];
            r1[7] = (v733_data + (v695_data * v731_data));
            float v736_data = s0[106];
            float v738_data = r1[8];
            r1[8] = (v738_data + (v695_data * v736_data));
            float v741_data = s0[118];
            float v743_data = r1[9];
            r1[9] = (v743_data + (v695_data * v741_data));
            float v746_data = s0[130];
            float v748_data = r1[10];
            r1[10] = (v748_data + (v695_data * v746_data));
            float v751_data = s0[142];
            float v753_data = r1[11];
            r1[11] = (v753_data + (v695_data * v751_data));
          }
          if (v8_lead < 6) {
            float v759_data = r0[11];
            float v760_data = s0[11];
            float v762_data = r1[0];
            r1[0] = (v762_data + (v759_data * v760_data));
            float v765_data = s0[23];
            float v767_data = r1[1];
            r1[1] = (v767_data + (v759_data * v765_data));
            float v770_data = s0[35];
            float v772_data = r1[2];
            r1[2] = (v772_data + (v759_data * v770_data));
            float v775_data = s0[47];
            float v777_data = r1[3];
            r1[3] = (v777_data + (v759_data * v775_data));
            float v780_data = s0[59];
            float v782_data = r1[4];
            r1[4] = (v782_data + (v759_data * v780_data));
            float v785_data = s0[71];
            float v787_data = r1[5];
            r1[5] = (v787_data + (v759_data * v785_data));
            float v790_data = s0[83];
            float v792_data = r1[6];
            r1[6] = (v792_data + (v759_data * v790_data));
            float v795_data = s0[95];
            float v797_data = r1[7];
            r1[7] = (v797_data + (v759_data * v795_data));
            float v800_data = s0[107];
            float v802_data = r1[8];
            r1[8] = (v802_data + (v759_data * v800_data));
            float v805_data = s0[119];
            float v807_data = r1[9];
            r1[9] = (v807_data + (v759_data * v805_data));
            float v810_data = s0[131];
            float v812_data = r1[10];
            r1[10] = (v812_data + (v759_data * v810_data));
            float v815_data = s0[143];
            float v817_data = r1[11];
            r1[11] = (v817_data + (v759_data * v815_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v824_i1 = 0; v824_i1 < 12; ++v824_i1) {
              int32_t v825_a = 0 + v824_i1;
              float v827_data = r1[v824_i1];
              int32_t v834_a = v8_lead + (v824_i1 * 12);
              s1[v834_a] = v827_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v840_i1 = 0; v840_i1 < 12; ++v840_i1) {
              int32_t v846_a = v840_i1 * 12;
              int32_t v847_a = v8_lead + v846_a;
              float v855_data = __ldcg(&glb_m4[(v8_lead + v846_a)]);
              int32_t v856_a = 0 + v840_i1;
              r4[v856_a] = v855_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v8_lead < 6) {
            float v863_data = r2[0];
            float v864_data = s0[0];
            float v866_data = ir3[0];
            ir3[0] = (v866_data + (v863_data * v864_data));
            float v869_data = s0[12];
            float v871_data = ir3[1];
            ir3[1] = (v871_data + (v863_data * v869_data));
            float v874_data = s0[24];
            float v876_data = ir3[2];
            ir3[2] = (v876_data + (v863_data * v874_data));
            float v879_data = s0[36];
            float v881_data = ir3[3];
            ir3[3] = (v881_data + (v863_data * v879_data));
            float v884_data = s0[48];
            float v886_data = ir3[4];
            ir3[4] = (v886_data + (v863_data * v884_data));
            float v889_data = s0[60];
            float v891_data = ir3[5];
            ir3[5] = (v891_data + (v863_data * v889_data));
            float v894_data = s0[72];
            float v896_data = ir3[6];
            ir3[6] = (v896_data + (v863_data * v894_data));
            float v899_data = s0[84];
            float v901_data = ir3[7];
            ir3[7] = (v901_data + (v863_data * v899_data));
            float v904_data = s0[96];
            float v906_data = ir3[8];
            ir3[8] = (v906_data + (v863_data * v904_data));
            float v909_data = s0[108];
            float v911_data = ir3[9];
            ir3[9] = (v911_data + (v863_data * v909_data));
            float v914_data = s0[120];
            float v916_data = ir3[10];
            ir3[10] = (v916_data + (v863_data * v914_data));
            float v919_data = s0[132];
            float v921_data = ir3[11];
            ir3[11] = (v921_data + (v863_data * v919_data));
          }
          if (v8_lead < 6) {
            float v927_data = r2[1];
            float v928_data = s0[1];
            float v930_data = ir3[0];
            ir3[0] = (v930_data + (v927_data * v928_data));
            float v933_data = s0[13];
            float v935_data = ir3[1];
            ir3[1] = (v935_data + (v927_data * v933_data));
            float v938_data = s0[25];
            float v940_data = ir3[2];
            ir3[2] = (v940_data + (v927_data * v938_data));
            float v943_data = s0[37];
            float v945_data = ir3[3];
            ir3[3] = (v945_data + (v927_data * v943_data));
            float v948_data = s0[49];
            float v950_data = ir3[4];
            ir3[4] = (v950_data + (v927_data * v948_data));
            float v953_data = s0[61];
            float v955_data = ir3[5];
            ir3[5] = (v955_data + (v927_data * v953_data));
            float v958_data = s0[73];
            float v960_data = ir3[6];
            ir3[6] = (v960_data + (v927_data * v958_data));
            float v963_data = s0[85];
            float v965_data = ir3[7];
            ir3[7] = (v965_data + (v927_data * v963_data));
            float v968_data = s0[97];
            float v970_data = ir3[8];
            ir3[8] = (v970_data + (v927_data * v968_data));
            float v973_data = s0[109];
            float v975_data = ir3[9];
            ir3[9] = (v975_data + (v927_data * v973_data));
            float v978_data = s0[121];
            float v980_data = ir3[10];
            ir3[10] = (v980_data + (v927_data * v978_data));
            float v983_data = s0[133];
            float v985_data = ir3[11];
            ir3[11] = (v985_data + (v927_data * v983_data));
          }
          if (v8_lead < 6) {
            float v991_data = r2[2];
            float v992_data = s0[2];
            float v994_data = ir3[0];
            ir3[0] = (v994_data + (v991_data * v992_data));
            float v997_data = s0[14];
            float v999_data = ir3[1];
            ir3[1] = (v999_data + (v991_data * v997_data));
            float v1002_data = s0[26];
            float v1004_data = ir3[2];
            ir3[2] = (v1004_data + (v991_data * v1002_data));
            float v1007_data = s0[38];
            float v1009_data = ir3[3];
            ir3[3] = (v1009_data + (v991_data * v1007_data));
            float v1012_data = s0[50];
            float v1014_data = ir3[4];
            ir3[4] = (v1014_data + (v991_data * v1012_data));
            float v1017_data = s0[62];
            float v1019_data = ir3[5];
            ir3[5] = (v1019_data + (v991_data * v1017_data));
            float v1022_data = s0[74];
            float v1024_data = ir3[6];
            ir3[6] = (v1024_data + (v991_data * v1022_data));
            float v1027_data = s0[86];
            float v1029_data = ir3[7];
            ir3[7] = (v1029_data + (v991_data * v1027_data));
            float v1032_data = s0[98];
            float v1034_data = ir3[8];
            ir3[8] = (v1034_data + (v991_data * v1032_data));
            float v1037_data = s0[110];
            float v1039_data = ir3[9];
            ir3[9] = (v1039_data + (v991_data * v1037_data));
            float v1042_data = s0[122];
            float v1044_data = ir3[10];
            ir3[10] = (v1044_data + (v991_data * v1042_data));
            float v1047_data = s0[134];
            float v1049_data = ir3[11];
            ir3[11] = (v1049_data + (v991_data * v1047_data));
          }
          if (v8_lead < 6) {
            float v1055_data = r2[3];
            float v1056_data = s0[3];
            float v1058_data = ir3[0];
            ir3[0] = (v1058_data + (v1055_data * v1056_data));
            float v1061_data = s0[15];
            float v1063_data = ir3[1];
            ir3[1] = (v1063_data + (v1055_data * v1061_data));
            float v1066_data = s0[27];
            float v1068_data = ir3[2];
            ir3[2] = (v1068_data + (v1055_data * v1066_data));
            float v1071_data = s0[39];
            float v1073_data = ir3[3];
            ir3[3] = (v1073_data + (v1055_data * v1071_data));
            float v1076_data = s0[51];
            float v1078_data = ir3[4];
            ir3[4] = (v1078_data + (v1055_data * v1076_data));
            float v1081_data = s0[63];
            float v1083_data = ir3[5];
            ir3[5] = (v1083_data + (v1055_data * v1081_data));
            float v1086_data = s0[75];
            float v1088_data = ir3[6];
            ir3[6] = (v1088_data + (v1055_data * v1086_data));
            float v1091_data = s0[87];
            float v1093_data = ir3[7];
            ir3[7] = (v1093_data + (v1055_data * v1091_data));
            float v1096_data = s0[99];
            float v1098_data = ir3[8];
            ir3[8] = (v1098_data + (v1055_data * v1096_data));
            float v1101_data = s0[111];
            float v1103_data = ir3[9];
            ir3[9] = (v1103_data + (v1055_data * v1101_data));
            float v1106_data = s0[123];
            float v1108_data = ir3[10];
            ir3[10] = (v1108_data + (v1055_data * v1106_data));
            float v1111_data = s0[135];
            float v1113_data = ir3[11];
            ir3[11] = (v1113_data + (v1055_data * v1111_data));
          }
          if (v8_lead < 6) {
            float v1119_data = r2[4];
            float v1120_data = s0[4];
            float v1122_data = ir3[0];
            ir3[0] = (v1122_data + (v1119_data * v1120_data));
            float v1125_data = s0[16];
            float v1127_data = ir3[1];
            ir3[1] = (v1127_data + (v1119_data * v1125_data));
            float v1130_data = s0[28];
            float v1132_data = ir3[2];
            ir3[2] = (v1132_data + (v1119_data * v1130_data));
            float v1135_data = s0[40];
            float v1137_data = ir3[3];
            ir3[3] = (v1137_data + (v1119_data * v1135_data));
            float v1140_data = s0[52];
            float v1142_data = ir3[4];
            ir3[4] = (v1142_data + (v1119_data * v1140_data));
            float v1145_data = s0[64];
            float v1147_data = ir3[5];
            ir3[5] = (v1147_data + (v1119_data * v1145_data));
            float v1150_data = s0[76];
            float v1152_data = ir3[6];
            ir3[6] = (v1152_data + (v1119_data * v1150_data));
            float v1155_data = s0[88];
            float v1157_data = ir3[7];
            ir3[7] = (v1157_data + (v1119_data * v1155_data));
            float v1160_data = s0[100];
            float v1162_data = ir3[8];
            ir3[8] = (v1162_data + (v1119_data * v1160_data));
            float v1165_data = s0[112];
            float v1167_data = ir3[9];
            ir3[9] = (v1167_data + (v1119_data * v1165_data));
            float v1170_data = s0[124];
            float v1172_data = ir3[10];
            ir3[10] = (v1172_data + (v1119_data * v1170_data));
            float v1175_data = s0[136];
            float v1177_data = ir3[11];
            ir3[11] = (v1177_data + (v1119_data * v1175_data));
          }
          if (v8_lead < 6) {
            float v1183_data = r2[5];
            float v1184_data = s0[5];
            float v1186_data = ir3[0];
            ir3[0] = (v1186_data + (v1183_data * v1184_data));
            float v1189_data = s0[17];
            float v1191_data = ir3[1];
            ir3[1] = (v1191_data + (v1183_data * v1189_data));
            float v1194_data = s0[29];
            float v1196_data = ir3[2];
            ir3[2] = (v1196_data + (v1183_data * v1194_data));
            float v1199_data = s0[41];
            float v1201_data = ir3[3];
            ir3[3] = (v1201_data + (v1183_data * v1199_data));
            float v1204_data = s0[53];
            float v1206_data = ir3[4];
            ir3[4] = (v1206_data + (v1183_data * v1204_data));
            float v1209_data = s0[65];
            float v1211_data = ir3[5];
            ir3[5] = (v1211_data + (v1183_data * v1209_data));
            float v1214_data = s0[77];
            float v1216_data = ir3[6];
            ir3[6] = (v1216_data + (v1183_data * v1214_data));
            float v1219_data = s0[89];
            float v1221_data = ir3[7];
            ir3[7] = (v1221_data + (v1183_data * v1219_data));
            float v1224_data = s0[101];
            float v1226_data = ir3[8];
            ir3[8] = (v1226_data + (v1183_data * v1224_data));
            float v1229_data = s0[113];
            float v1231_data = ir3[9];
            ir3[9] = (v1231_data + (v1183_data * v1229_data));
            float v1234_data = s0[125];
            float v1236_data = ir3[10];
            ir3[10] = (v1236_data + (v1183_data * v1234_data));
            float v1239_data = s0[137];
            float v1241_data = ir3[11];
            ir3[11] = (v1241_data + (v1183_data * v1239_data));
          }
          if (v8_lead < 6) {
            float v1247_data = r2[6];
            float v1248_data = s0[6];
            float v1250_data = ir3[0];
            ir3[0] = (v1250_data + (v1247_data * v1248_data));
            float v1253_data = s0[18];
            float v1255_data = ir3[1];
            ir3[1] = (v1255_data + (v1247_data * v1253_data));
            float v1258_data = s0[30];
            float v1260_data = ir3[2];
            ir3[2] = (v1260_data + (v1247_data * v1258_data));
            float v1263_data = s0[42];
            float v1265_data = ir3[3];
            ir3[3] = (v1265_data + (v1247_data * v1263_data));
            float v1268_data = s0[54];
            float v1270_data = ir3[4];
            ir3[4] = (v1270_data + (v1247_data * v1268_data));
            float v1273_data = s0[66];
            float v1275_data = ir3[5];
            ir3[5] = (v1275_data + (v1247_data * v1273_data));
            float v1278_data = s0[78];
            float v1280_data = ir3[6];
            ir3[6] = (v1280_data + (v1247_data * v1278_data));
            float v1283_data = s0[90];
            float v1285_data = ir3[7];
            ir3[7] = (v1285_data + (v1247_data * v1283_data));
            float v1288_data = s0[102];
            float v1290_data = ir3[8];
            ir3[8] = (v1290_data + (v1247_data * v1288_data));
            float v1293_data = s0[114];
            float v1295_data = ir3[9];
            ir3[9] = (v1295_data + (v1247_data * v1293_data));
            float v1298_data = s0[126];
            float v1300_data = ir3[10];
            ir3[10] = (v1300_data + (v1247_data * v1298_data));
            float v1303_data = s0[138];
            float v1305_data = ir3[11];
            ir3[11] = (v1305_data + (v1247_data * v1303_data));
          }
          if (v8_lead < 6) {
            float v1311_data = r2[7];
            float v1312_data = s0[7];
            float v1314_data = ir3[0];
            ir3[0] = (v1314_data + (v1311_data * v1312_data));
            float v1317_data = s0[19];
            float v1319_data = ir3[1];
            ir3[1] = (v1319_data + (v1311_data * v1317_data));
            float v1322_data = s0[31];
            float v1324_data = ir3[2];
            ir3[2] = (v1324_data + (v1311_data * v1322_data));
            float v1327_data = s0[43];
            float v1329_data = ir3[3];
            ir3[3] = (v1329_data + (v1311_data * v1327_data));
            float v1332_data = s0[55];
            float v1334_data = ir3[4];
            ir3[4] = (v1334_data + (v1311_data * v1332_data));
            float v1337_data = s0[67];
            float v1339_data = ir3[5];
            ir3[5] = (v1339_data + (v1311_data * v1337_data));
            float v1342_data = s0[79];
            float v1344_data = ir3[6];
            ir3[6] = (v1344_data + (v1311_data * v1342_data));
            float v1347_data = s0[91];
            float v1349_data = ir3[7];
            ir3[7] = (v1349_data + (v1311_data * v1347_data));
            float v1352_data = s0[103];
            float v1354_data = ir3[8];
            ir3[8] = (v1354_data + (v1311_data * v1352_data));
            float v1357_data = s0[115];
            float v1359_data = ir3[9];
            ir3[9] = (v1359_data + (v1311_data * v1357_data));
            float v1362_data = s0[127];
            float v1364_data = ir3[10];
            ir3[10] = (v1364_data + (v1311_data * v1362_data));
            float v1367_data = s0[139];
            float v1369_data = ir3[11];
            ir3[11] = (v1369_data + (v1311_data * v1367_data));
          }
          if (v8_lead < 6) {
            float v1375_data = r2[8];
            float v1376_data = s0[8];
            float v1378_data = ir3[0];
            ir3[0] = (v1378_data + (v1375_data * v1376_data));
            float v1381_data = s0[20];
            float v1383_data = ir3[1];
            ir3[1] = (v1383_data + (v1375_data * v1381_data));
            float v1386_data = s0[32];
            float v1388_data = ir3[2];
            ir3[2] = (v1388_data + (v1375_data * v1386_data));
            float v1391_data = s0[44];
            float v1393_data = ir3[3];
            ir3[3] = (v1393_data + (v1375_data * v1391_data));
            float v1396_data = s0[56];
            float v1398_data = ir3[4];
            ir3[4] = (v1398_data + (v1375_data * v1396_data));
            float v1401_data = s0[68];
            float v1403_data = ir3[5];
            ir3[5] = (v1403_data + (v1375_data * v1401_data));
            float v1406_data = s0[80];
            float v1408_data = ir3[6];
            ir3[6] = (v1408_data + (v1375_data * v1406_data));
            float v1411_data = s0[92];
            float v1413_data = ir3[7];
            ir3[7] = (v1413_data + (v1375_data * v1411_data));
            float v1416_data = s0[104];
            float v1418_data = ir3[8];
            ir3[8] = (v1418_data + (v1375_data * v1416_data));
            float v1421_data = s0[116];
            float v1423_data = ir3[9];
            ir3[9] = (v1423_data + (v1375_data * v1421_data));
            float v1426_data = s0[128];
            float v1428_data = ir3[10];
            ir3[10] = (v1428_data + (v1375_data * v1426_data));
            float v1431_data = s0[140];
            float v1433_data = ir3[11];
            ir3[11] = (v1433_data + (v1375_data * v1431_data));
          }
          if (v8_lead < 6) {
            float v1439_data = r2[9];
            float v1440_data = s0[9];
            float v1442_data = ir3[0];
            ir3[0] = (v1442_data + (v1439_data * v1440_data));
            float v1445_data = s0[21];
            float v1447_data = ir3[1];
            ir3[1] = (v1447_data + (v1439_data * v1445_data));
            float v1450_data = s0[33];
            float v1452_data = ir3[2];
            ir3[2] = (v1452_data + (v1439_data * v1450_data));
            float v1455_data = s0[45];
            float v1457_data = ir3[3];
            ir3[3] = (v1457_data + (v1439_data * v1455_data));
            float v1460_data = s0[57];
            float v1462_data = ir3[4];
            ir3[4] = (v1462_data + (v1439_data * v1460_data));
            float v1465_data = s0[69];
            float v1467_data = ir3[5];
            ir3[5] = (v1467_data + (v1439_data * v1465_data));
            float v1470_data = s0[81];
            float v1472_data = ir3[6];
            ir3[6] = (v1472_data + (v1439_data * v1470_data));
            float v1475_data = s0[93];
            float v1477_data = ir3[7];
            ir3[7] = (v1477_data + (v1439_data * v1475_data));
            float v1480_data = s0[105];
            float v1482_data = ir3[8];
            ir3[8] = (v1482_data + (v1439_data * v1480_data));
            float v1485_data = s0[117];
            float v1487_data = ir3[9];
            ir3[9] = (v1487_data + (v1439_data * v1485_data));
            float v1490_data = s0[129];
            float v1492_data = ir3[10];
            ir3[10] = (v1492_data + (v1439_data * v1490_data));
            float v1495_data = s0[141];
            float v1497_data = ir3[11];
            ir3[11] = (v1497_data + (v1439_data * v1495_data));
          }
          if (v8_lead < 6) {
            float v1503_data = r2[10];
            float v1504_data = s0[10];
            float v1506_data = ir3[0];
            ir3[0] = (v1506_data + (v1503_data * v1504_data));
            float v1509_data = s0[22];
            float v1511_data = ir3[1];
            ir3[1] = (v1511_data + (v1503_data * v1509_data));
            float v1514_data = s0[34];
            float v1516_data = ir3[2];
            ir3[2] = (v1516_data + (v1503_data * v1514_data));
            float v1519_data = s0[46];
            float v1521_data = ir3[3];
            ir3[3] = (v1521_data + (v1503_data * v1519_data));
            float v1524_data = s0[58];
            float v1526_data = ir3[4];
            ir3[4] = (v1526_data + (v1503_data * v1524_data));
            float v1529_data = s0[70];
            float v1531_data = ir3[5];
            ir3[5] = (v1531_data + (v1503_data * v1529_data));
            float v1534_data = s0[82];
            float v1536_data = ir3[6];
            ir3[6] = (v1536_data + (v1503_data * v1534_data));
            float v1539_data = s0[94];
            float v1541_data = ir3[7];
            ir3[7] = (v1541_data + (v1503_data * v1539_data));
            float v1544_data = s0[106];
            float v1546_data = ir3[8];
            ir3[8] = (v1546_data + (v1503_data * v1544_data));
            float v1549_data = s0[118];
            float v1551_data = ir3[9];
            ir3[9] = (v1551_data + (v1503_data * v1549_data));
            float v1554_data = s0[130];
            float v1556_data = ir3[10];
            ir3[10] = (v1556_data + (v1503_data * v1554_data));
            float v1559_data = s0[142];
            float v1561_data = ir3[11];
            ir3[11] = (v1561_data + (v1503_data * v1559_data));
          }
          if (v8_lead < 6) {
            float v1567_data = r2[11];
            float v1568_data = s0[11];
            float v1570_data = ir3[0];
            ir3[0] = (v1570_data + (v1567_data * v1568_data));
            float v1573_data = s0[23];
            float v1575_data = ir3[1];
            ir3[1] = (v1575_data + (v1567_data * v1573_data));
            float v1578_data = s0[35];
            float v1580_data = ir3[2];
            ir3[2] = (v1580_data + (v1567_data * v1578_data));
            float v1583_data = s0[47];
            float v1585_data = ir3[3];
            ir3[3] = (v1585_data + (v1567_data * v1583_data));
            float v1588_data = s0[59];
            float v1590_data = ir3[4];
            ir3[4] = (v1590_data + (v1567_data * v1588_data));
            float v1593_data = s0[71];
            float v1595_data = ir3[5];
            ir3[5] = (v1595_data + (v1567_data * v1593_data));
            float v1598_data = s0[83];
            float v1600_data = ir3[6];
            ir3[6] = (v1600_data + (v1567_data * v1598_data));
            float v1603_data = s0[95];
            float v1605_data = ir3[7];
            ir3[7] = (v1605_data + (v1567_data * v1603_data));
            float v1608_data = s0[107];
            float v1610_data = ir3[8];
            ir3[8] = (v1610_data + (v1567_data * v1608_data));
            float v1613_data = s0[119];
            float v1615_data = ir3[9];
            ir3[9] = (v1615_data + (v1567_data * v1613_data));
            float v1618_data = s0[131];
            float v1620_data = ir3[10];
            ir3[10] = (v1620_data + (v1567_data * v1618_data));
            float v1623_data = s0[143];
            float v1625_data = ir3[11];
            ir3[11] = (v1625_data + (v1567_data * v1623_data));
          }
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v1631_n1 = 0; v1631_n1 < 12; ++v1631_n1) {
              int32_t v1632_a = 0 + v1631_n1;
              float v1634_data = ir3[v1631_n1];
              int32_t v1635_a = 0 + v1631_n1;
              r3[v1631_n1] = v1634_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v8_lead < 6) {
            int32_t v1650_off = v8_lead + 6;
            #pragma unroll
            for (int32_t v1641_i1 = 0; v1641_i1 < 12; ++v1641_i1) {
              int32_t v1642_a = 0 + v1641_i1;
              float v1644_data = r3[v1641_i1];
              int32_t v1652_a = v1650_off + (v1641_i1 * 12);
              s1[v1652_a] = v1644_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v8_lead < 12) {
            float v1659_data = r4[0];
            float v1660_data = s1[0];
            float v1662_data = ir5[0];
            ir5[0] = (v1662_data + (v1659_data * v1660_data));
            float v1665_data = s1[12];
            float v1667_data = ir5[1];
            ir5[1] = (v1667_data + (v1659_data * v1665_data));
            float v1670_data = s1[24];
            float v1672_data = ir5[2];
            ir5[2] = (v1672_data + (v1659_data * v1670_data));
            float v1675_data = s1[36];
            float v1677_data = ir5[3];
            ir5[3] = (v1677_data + (v1659_data * v1675_data));
            float v1680_data = s1[48];
            float v1682_data = ir5[4];
            ir5[4] = (v1682_data + (v1659_data * v1680_data));
            float v1685_data = s1[60];
            float v1687_data = ir5[5];
            ir5[5] = (v1687_data + (v1659_data * v1685_data));
            float v1690_data = s1[72];
            float v1692_data = ir5[6];
            ir5[6] = (v1692_data + (v1659_data * v1690_data));
            float v1695_data = s1[84];
            float v1697_data = ir5[7];
            ir5[7] = (v1697_data + (v1659_data * v1695_data));
            float v1700_data = s1[96];
            float v1702_data = ir5[8];
            ir5[8] = (v1702_data + (v1659_data * v1700_data));
            float v1705_data = s1[108];
            float v1707_data = ir5[9];
            ir5[9] = (v1707_data + (v1659_data * v1705_data));
            float v1710_data = s1[120];
            float v1712_data = ir5[10];
            ir5[10] = (v1712_data + (v1659_data * v1710_data));
            float v1715_data = s1[132];
            float v1717_data = ir5[11];
            ir5[11] = (v1717_data + (v1659_data * v1715_data));
          }
          if (v8_lead < 12) {
            float v1723_data = r4[1];
            float v1724_data = s1[1];
            float v1726_data = ir5[0];
            ir5[0] = (v1726_data + (v1723_data * v1724_data));
            float v1729_data = s1[13];
            float v1731_data = ir5[1];
            ir5[1] = (v1731_data + (v1723_data * v1729_data));
            float v1734_data = s1[25];
            float v1736_data = ir5[2];
            ir5[2] = (v1736_data + (v1723_data * v1734_data));
            float v1739_data = s1[37];
            float v1741_data = ir5[3];
            ir5[3] = (v1741_data + (v1723_data * v1739_data));
            float v1744_data = s1[49];
            float v1746_data = ir5[4];
            ir5[4] = (v1746_data + (v1723_data * v1744_data));
            float v1749_data = s1[61];
            float v1751_data = ir5[5];
            ir5[5] = (v1751_data + (v1723_data * v1749_data));
            float v1754_data = s1[73];
            float v1756_data = ir5[6];
            ir5[6] = (v1756_data + (v1723_data * v1754_data));
            float v1759_data = s1[85];
            float v1761_data = ir5[7];
            ir5[7] = (v1761_data + (v1723_data * v1759_data));
            float v1764_data = s1[97];
            float v1766_data = ir5[8];
            ir5[8] = (v1766_data + (v1723_data * v1764_data));
            float v1769_data = s1[109];
            float v1771_data = ir5[9];
            ir5[9] = (v1771_data + (v1723_data * v1769_data));
            float v1774_data = s1[121];
            float v1776_data = ir5[10];
            ir5[10] = (v1776_data + (v1723_data * v1774_data));
            float v1779_data = s1[133];
            float v1781_data = ir5[11];
            ir5[11] = (v1781_data + (v1723_data * v1779_data));
          }
          if (v8_lead < 12) {
            float v1787_data = r4[2];
            float v1788_data = s1[2];
            float v1790_data = ir5[0];
            ir5[0] = (v1790_data + (v1787_data * v1788_data));
            float v1793_data = s1[14];
            float v1795_data = ir5[1];
            ir5[1] = (v1795_data + (v1787_data * v1793_data));
            float v1798_data = s1[26];
            float v1800_data = ir5[2];
            ir5[2] = (v1800_data + (v1787_data * v1798_data));
            float v1803_data = s1[38];
            float v1805_data = ir5[3];
            ir5[3] = (v1805_data + (v1787_data * v1803_data));
            float v1808_data = s1[50];
            float v1810_data = ir5[4];
            ir5[4] = (v1810_data + (v1787_data * v1808_data));
            float v1813_data = s1[62];
            float v1815_data = ir5[5];
            ir5[5] = (v1815_data + (v1787_data * v1813_data));
            float v1818_data = s1[74];
            float v1820_data = ir5[6];
            ir5[6] = (v1820_data + (v1787_data * v1818_data));
            float v1823_data = s1[86];
            float v1825_data = ir5[7];
            ir5[7] = (v1825_data + (v1787_data * v1823_data));
            float v1828_data = s1[98];
            float v1830_data = ir5[8];
            ir5[8] = (v1830_data + (v1787_data * v1828_data));
            float v1833_data = s1[110];
            float v1835_data = ir5[9];
            ir5[9] = (v1835_data + (v1787_data * v1833_data));
            float v1838_data = s1[122];
            float v1840_data = ir5[10];
            ir5[10] = (v1840_data + (v1787_data * v1838_data));
            float v1843_data = s1[134];
            float v1845_data = ir5[11];
            ir5[11] = (v1845_data + (v1787_data * v1843_data));
          }
          if (v8_lead < 12) {
            float v1851_data = r4[3];
            float v1852_data = s1[3];
            float v1854_data = ir5[0];
            ir5[0] = (v1854_data + (v1851_data * v1852_data));
            float v1857_data = s1[15];
            float v1859_data = ir5[1];
            ir5[1] = (v1859_data + (v1851_data * v1857_data));
            float v1862_data = s1[27];
            float v1864_data = ir5[2];
            ir5[2] = (v1864_data + (v1851_data * v1862_data));
            float v1867_data = s1[39];
            float v1869_data = ir5[3];
            ir5[3] = (v1869_data + (v1851_data * v1867_data));
            float v1872_data = s1[51];
            float v1874_data = ir5[4];
            ir5[4] = (v1874_data + (v1851_data * v1872_data));
            float v1877_data = s1[63];
            float v1879_data = ir5[5];
            ir5[5] = (v1879_data + (v1851_data * v1877_data));
            float v1882_data = s1[75];
            float v1884_data = ir5[6];
            ir5[6] = (v1884_data + (v1851_data * v1882_data));
            float v1887_data = s1[87];
            float v1889_data = ir5[7];
            ir5[7] = (v1889_data + (v1851_data * v1887_data));
            float v1892_data = s1[99];
            float v1894_data = ir5[8];
            ir5[8] = (v1894_data + (v1851_data * v1892_data));
            float v1897_data = s1[111];
            float v1899_data = ir5[9];
            ir5[9] = (v1899_data + (v1851_data * v1897_data));
            float v1902_data = s1[123];
            float v1904_data = ir5[10];
            ir5[10] = (v1904_data + (v1851_data * v1902_data));
            float v1907_data = s1[135];
            float v1909_data = ir5[11];
            ir5[11] = (v1909_data + (v1851_data * v1907_data));
          }
          if (v8_lead < 12) {
            float v1915_data = r4[4];
            float v1916_data = s1[4];
            float v1918_data = ir5[0];
            ir5[0] = (v1918_data + (v1915_data * v1916_data));
            float v1921_data = s1[16];
            float v1923_data = ir5[1];
            ir5[1] = (v1923_data + (v1915_data * v1921_data));
            float v1926_data = s1[28];
            float v1928_data = ir5[2];
            ir5[2] = (v1928_data + (v1915_data * v1926_data));
            float v1931_data = s1[40];
            float v1933_data = ir5[3];
            ir5[3] = (v1933_data + (v1915_data * v1931_data));
            float v1936_data = s1[52];
            float v1938_data = ir5[4];
            ir5[4] = (v1938_data + (v1915_data * v1936_data));
            float v1941_data = s1[64];
            float v1943_data = ir5[5];
            ir5[5] = (v1943_data + (v1915_data * v1941_data));
            float v1946_data = s1[76];
            float v1948_data = ir5[6];
            ir5[6] = (v1948_data + (v1915_data * v1946_data));
            float v1951_data = s1[88];
            float v1953_data = ir5[7];
            ir5[7] = (v1953_data + (v1915_data * v1951_data));
            float v1956_data = s1[100];
            float v1958_data = ir5[8];
            ir5[8] = (v1958_data + (v1915_data * v1956_data));
            float v1961_data = s1[112];
            float v1963_data = ir5[9];
            ir5[9] = (v1963_data + (v1915_data * v1961_data));
            float v1966_data = s1[124];
            float v1968_data = ir5[10];
            ir5[10] = (v1968_data + (v1915_data * v1966_data));
            float v1971_data = s1[136];
            float v1973_data = ir5[11];
            ir5[11] = (v1973_data + (v1915_data * v1971_data));
          }
          if (v8_lead < 12) {
            float v1979_data = r4[5];
            float v1980_data = s1[5];
            float v1982_data = ir5[0];
            ir5[0] = (v1982_data + (v1979_data * v1980_data));
            float v1985_data = s1[17];
            float v1987_data = ir5[1];
            ir5[1] = (v1987_data + (v1979_data * v1985_data));
            float v1990_data = s1[29];
            float v1992_data = ir5[2];
            ir5[2] = (v1992_data + (v1979_data * v1990_data));
            float v1995_data = s1[41];
            float v1997_data = ir5[3];
            ir5[3] = (v1997_data + (v1979_data * v1995_data));
            float v2000_data = s1[53];
            float v2002_data = ir5[4];
            ir5[4] = (v2002_data + (v1979_data * v2000_data));
            float v2005_data = s1[65];
            float v2007_data = ir5[5];
            ir5[5] = (v2007_data + (v1979_data * v2005_data));
            float v2010_data = s1[77];
            float v2012_data = ir5[6];
            ir5[6] = (v2012_data + (v1979_data * v2010_data));
            float v2015_data = s1[89];
            float v2017_data = ir5[7];
            ir5[7] = (v2017_data + (v1979_data * v2015_data));
            float v2020_data = s1[101];
            float v2022_data = ir5[8];
            ir5[8] = (v2022_data + (v1979_data * v2020_data));
            float v2025_data = s1[113];
            float v2027_data = ir5[9];
            ir5[9] = (v2027_data + (v1979_data * v2025_data));
            float v2030_data = s1[125];
            float v2032_data = ir5[10];
            ir5[10] = (v2032_data + (v1979_data * v2030_data));
            float v2035_data = s1[137];
            float v2037_data = ir5[11];
            ir5[11] = (v2037_data + (v1979_data * v2035_data));
          }
          if (v8_lead < 12) {
            float v2043_data = r4[6];
            float v2044_data = s1[6];
            float v2046_data = ir5[0];
            ir5[0] = (v2046_data + (v2043_data * v2044_data));
            float v2049_data = s1[18];
            float v2051_data = ir5[1];
            ir5[1] = (v2051_data + (v2043_data * v2049_data));
            float v2054_data = s1[30];
            float v2056_data = ir5[2];
            ir5[2] = (v2056_data + (v2043_data * v2054_data));
            float v2059_data = s1[42];
            float v2061_data = ir5[3];
            ir5[3] = (v2061_data + (v2043_data * v2059_data));
            float v2064_data = s1[54];
            float v2066_data = ir5[4];
            ir5[4] = (v2066_data + (v2043_data * v2064_data));
            float v2069_data = s1[66];
            float v2071_data = ir5[5];
            ir5[5] = (v2071_data + (v2043_data * v2069_data));
            float v2074_data = s1[78];
            float v2076_data = ir5[6];
            ir5[6] = (v2076_data + (v2043_data * v2074_data));
            float v2079_data = s1[90];
            float v2081_data = ir5[7];
            ir5[7] = (v2081_data + (v2043_data * v2079_data));
            float v2084_data = s1[102];
            float v2086_data = ir5[8];
            ir5[8] = (v2086_data + (v2043_data * v2084_data));
            float v2089_data = s1[114];
            float v2091_data = ir5[9];
            ir5[9] = (v2091_data + (v2043_data * v2089_data));
            float v2094_data = s1[126];
            float v2096_data = ir5[10];
            ir5[10] = (v2096_data + (v2043_data * v2094_data));
            float v2099_data = s1[138];
            float v2101_data = ir5[11];
            ir5[11] = (v2101_data + (v2043_data * v2099_data));
          }
          if (v8_lead < 12) {
            float v2107_data = r4[7];
            float v2108_data = s1[7];
            float v2110_data = ir5[0];
            ir5[0] = (v2110_data + (v2107_data * v2108_data));
            float v2113_data = s1[19];
            float v2115_data = ir5[1];
            ir5[1] = (v2115_data + (v2107_data * v2113_data));
            float v2118_data = s1[31];
            float v2120_data = ir5[2];
            ir5[2] = (v2120_data + (v2107_data * v2118_data));
            float v2123_data = s1[43];
            float v2125_data = ir5[3];
            ir5[3] = (v2125_data + (v2107_data * v2123_data));
            float v2128_data = s1[55];
            float v2130_data = ir5[4];
            ir5[4] = (v2130_data + (v2107_data * v2128_data));
            float v2133_data = s1[67];
            float v2135_data = ir5[5];
            ir5[5] = (v2135_data + (v2107_data * v2133_data));
            float v2138_data = s1[79];
            float v2140_data = ir5[6];
            ir5[6] = (v2140_data + (v2107_data * v2138_data));
            float v2143_data = s1[91];
            float v2145_data = ir5[7];
            ir5[7] = (v2145_data + (v2107_data * v2143_data));
            float v2148_data = s1[103];
            float v2150_data = ir5[8];
            ir5[8] = (v2150_data + (v2107_data * v2148_data));
            float v2153_data = s1[115];
            float v2155_data = ir5[9];
            ir5[9] = (v2155_data + (v2107_data * v2153_data));
            float v2158_data = s1[127];
            float v2160_data = ir5[10];
            ir5[10] = (v2160_data + (v2107_data * v2158_data));
            float v2163_data = s1[139];
            float v2165_data = ir5[11];
            ir5[11] = (v2165_data + (v2107_data * v2163_data));
          }
          if (v8_lead < 12) {
            float v2171_data = r4[8];
            float v2172_data = s1[8];
            float v2174_data = ir5[0];
            ir5[0] = (v2174_data + (v2171_data * v2172_data));
            float v2177_data = s1[20];
            float v2179_data = ir5[1];
            ir5[1] = (v2179_data + (v2171_data * v2177_data));
            float v2182_data = s1[32];
            float v2184_data = ir5[2];
            ir5[2] = (v2184_data + (v2171_data * v2182_data));
            float v2187_data = s1[44];
            float v2189_data = ir5[3];
            ir5[3] = (v2189_data + (v2171_data * v2187_data));
            float v2192_data = s1[56];
            float v2194_data = ir5[4];
            ir5[4] = (v2194_data + (v2171_data * v2192_data));
            float v2197_data = s1[68];
            float v2199_data = ir5[5];
            ir5[5] = (v2199_data + (v2171_data * v2197_data));
            float v2202_data = s1[80];
            float v2204_data = ir5[6];
            ir5[6] = (v2204_data + (v2171_data * v2202_data));
            float v2207_data = s1[92];
            float v2209_data = ir5[7];
            ir5[7] = (v2209_data + (v2171_data * v2207_data));
            float v2212_data = s1[104];
            float v2214_data = ir5[8];
            ir5[8] = (v2214_data + (v2171_data * v2212_data));
            float v2217_data = s1[116];
            float v2219_data = ir5[9];
            ir5[9] = (v2219_data + (v2171_data * v2217_data));
            float v2222_data = s1[128];
            float v2224_data = ir5[10];
            ir5[10] = (v2224_data + (v2171_data * v2222_data));
            float v2227_data = s1[140];
            float v2229_data = ir5[11];
            ir5[11] = (v2229_data + (v2171_data * v2227_data));
          }
          if (v8_lead < 12) {
            float v2235_data = r4[9];
            float v2236_data = s1[9];
            float v2238_data = ir5[0];
            ir5[0] = (v2238_data + (v2235_data * v2236_data));
            float v2241_data = s1[21];
            float v2243_data = ir5[1];
            ir5[1] = (v2243_data + (v2235_data * v2241_data));
            float v2246_data = s1[33];
            float v2248_data = ir5[2];
            ir5[2] = (v2248_data + (v2235_data * v2246_data));
            float v2251_data = s1[45];
            float v2253_data = ir5[3];
            ir5[3] = (v2253_data + (v2235_data * v2251_data));
            float v2256_data = s1[57];
            float v2258_data = ir5[4];
            ir5[4] = (v2258_data + (v2235_data * v2256_data));
            float v2261_data = s1[69];
            float v2263_data = ir5[5];
            ir5[5] = (v2263_data + (v2235_data * v2261_data));
            float v2266_data = s1[81];
            float v2268_data = ir5[6];
            ir5[6] = (v2268_data + (v2235_data * v2266_data));
            float v2271_data = s1[93];
            float v2273_data = ir5[7];
            ir5[7] = (v2273_data + (v2235_data * v2271_data));
            float v2276_data = s1[105];
            float v2278_data = ir5[8];
            ir5[8] = (v2278_data + (v2235_data * v2276_data));
            float v2281_data = s1[117];
            float v2283_data = ir5[9];
            ir5[9] = (v2283_data + (v2235_data * v2281_data));
            float v2286_data = s1[129];
            float v2288_data = ir5[10];
            ir5[10] = (v2288_data + (v2235_data * v2286_data));
            float v2291_data = s1[141];
            float v2293_data = ir5[11];
            ir5[11] = (v2293_data + (v2235_data * v2291_data));
          }
          if (v8_lead < 12) {
            float v2299_data = r4[10];
            float v2300_data = s1[10];
            float v2302_data = ir5[0];
            ir5[0] = (v2302_data + (v2299_data * v2300_data));
            float v2305_data = s1[22];
            float v2307_data = ir5[1];
            ir5[1] = (v2307_data + (v2299_data * v2305_data));
            float v2310_data = s1[34];
            float v2312_data = ir5[2];
            ir5[2] = (v2312_data + (v2299_data * v2310_data));
            float v2315_data = s1[46];
            float v2317_data = ir5[3];
            ir5[3] = (v2317_data + (v2299_data * v2315_data));
            float v2320_data = s1[58];
            float v2322_data = ir5[4];
            ir5[4] = (v2322_data + (v2299_data * v2320_data));
            float v2325_data = s1[70];
            float v2327_data = ir5[5];
            ir5[5] = (v2327_data + (v2299_data * v2325_data));
            float v2330_data = s1[82];
            float v2332_data = ir5[6];
            ir5[6] = (v2332_data + (v2299_data * v2330_data));
            float v2335_data = s1[94];
            float v2337_data = ir5[7];
            ir5[7] = (v2337_data + (v2299_data * v2335_data));
            float v2340_data = s1[106];
            float v2342_data = ir5[8];
            ir5[8] = (v2342_data + (v2299_data * v2340_data));
            float v2345_data = s1[118];
            float v2347_data = ir5[9];
            ir5[9] = (v2347_data + (v2299_data * v2345_data));
            float v2350_data = s1[130];
            float v2352_data = ir5[10];
            ir5[10] = (v2352_data + (v2299_data * v2350_data));
            float v2355_data = s1[142];
            float v2357_data = ir5[11];
            ir5[11] = (v2357_data + (v2299_data * v2355_data));
          }
          if (v8_lead < 12) {
            float v2363_data = r4[11];
            float v2364_data = s1[11];
            float v2366_data = ir5[0];
            ir5[0] = (v2366_data + (v2363_data * v2364_data));
            float v2369_data = s1[23];
            float v2371_data = ir5[1];
            ir5[1] = (v2371_data + (v2363_data * v2369_data));
            float v2374_data = s1[35];
            float v2376_data = ir5[2];
            ir5[2] = (v2376_data + (v2363_data * v2374_data));
            float v2379_data = s1[47];
            float v2381_data = ir5[3];
            ir5[3] = (v2381_data + (v2363_data * v2379_data));
            float v2384_data = s1[59];
            float v2386_data = ir5[4];
            ir5[4] = (v2386_data + (v2363_data * v2384_data));
            float v2389_data = s1[71];
            float v2391_data = ir5[5];
            ir5[5] = (v2391_data + (v2363_data * v2389_data));
            float v2394_data = s1[83];
            float v2396_data = ir5[6];
            ir5[6] = (v2396_data + (v2363_data * v2394_data));
            float v2399_data = s1[95];
            float v2401_data = ir5[7];
            ir5[7] = (v2401_data + (v2363_data * v2399_data));
            float v2404_data = s1[107];
            float v2406_data = ir5[8];
            ir5[8] = (v2406_data + (v2363_data * v2404_data));
            float v2409_data = s1[119];
            float v2411_data = ir5[9];
            ir5[9] = (v2411_data + (v2363_data * v2409_data));
            float v2414_data = s1[131];
            float v2416_data = ir5[10];
            ir5[10] = (v2416_data + (v2363_data * v2414_data));
            float v2419_data = s1[143];
            float v2421_data = ir5[11];
            ir5[11] = (v2421_data + (v2363_data * v2419_data));
          }
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v2427_n1 = 0; v2427_n1 < 12; ++v2427_n1) {
              int32_t v2428_a = 0 + v2427_n1;
              float v2430_data = ir5[v2427_n1];
              int32_t v2431_a = 0 + v2427_n1;
              r5[v2427_n1] = v2430_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v2437_i1 = 0; v2437_i1 < 12; ++v2437_i1) {
              int32_t v2438_a = 0 + v2437_i1;
              float v2440_data = r5[v2437_i1];
              int32_t v2447_a = v8_lead + (v2437_i1 * 12);
              glb_m3[v2447_a] = v2440_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

