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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 16;
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 12; ++v14_i1) {
              int32_t v20_a = v14_i1 * 6;
              int32_t v21_a = v12_lead + v20_a;
              float v29_data = __ldcg(&glb_m0[(v12_lead + v20_a)]);
              r0[v14_i1] = v29_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m1[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 9; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              int32_t v44_a = v38_i1 * 6;
              int32_t v45_a = v12_lead + v44_a;
              float v53_data = __ldcg(&glb_m2[(v12_lead + v44_a)]);
              r2[v38_i1] = v53_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v12_lead < 6) {
            float v60_data = r0[0];
            float v61_data = s0[0];
            float v63_data = r1[0];
            r1[0] = (v63_data + (v60_data * v61_data));
            float v66_data = s0[12];
            float v68_data = r1[1];
            r1[1] = (v68_data + (v60_data * v66_data));
            float v71_data = s0[24];
            float v73_data = r1[2];
            r1[2] = (v73_data + (v60_data * v71_data));
            float v76_data = s0[36];
            float v78_data = r1[3];
            r1[3] = (v78_data + (v60_data * v76_data));
            float v81_data = s0[48];
            float v83_data = r1[4];
            r1[4] = (v83_data + (v60_data * v81_data));
            float v86_data = s0[60];
            float v88_data = r1[5];
            r1[5] = (v88_data + (v60_data * v86_data));
            float v91_data = s0[72];
            float v93_data = r1[6];
            r1[6] = (v93_data + (v60_data * v91_data));
            float v96_data = s0[84];
            float v98_data = r1[7];
            r1[7] = (v98_data + (v60_data * v96_data));
            float v101_data = s0[96];
            float v103_data = r1[8];
            r1[8] = (v103_data + (v60_data * v101_data));
            float v106_data = s0[108];
            float v108_data = r1[9];
            r1[9] = (v108_data + (v60_data * v106_data));
            float v111_data = s0[120];
            float v113_data = r1[10];
            r1[10] = (v113_data + (v60_data * v111_data));
            float v116_data = s0[132];
            float v118_data = r1[11];
            r1[11] = (v118_data + (v60_data * v116_data));
          }
          if (v12_lead < 6) {
            float v124_data = r0[1];
            float v125_data = s0[1];
            float v127_data = r1[0];
            r1[0] = (v127_data + (v124_data * v125_data));
            float v130_data = s0[13];
            float v132_data = r1[1];
            r1[1] = (v132_data + (v124_data * v130_data));
            float v135_data = s0[25];
            float v137_data = r1[2];
            r1[2] = (v137_data + (v124_data * v135_data));
            float v140_data = s0[37];
            float v142_data = r1[3];
            r1[3] = (v142_data + (v124_data * v140_data));
            float v145_data = s0[49];
            float v147_data = r1[4];
            r1[4] = (v147_data + (v124_data * v145_data));
            float v150_data = s0[61];
            float v152_data = r1[5];
            r1[5] = (v152_data + (v124_data * v150_data));
            float v155_data = s0[73];
            float v157_data = r1[6];
            r1[6] = (v157_data + (v124_data * v155_data));
            float v160_data = s0[85];
            float v162_data = r1[7];
            r1[7] = (v162_data + (v124_data * v160_data));
            float v165_data = s0[97];
            float v167_data = r1[8];
            r1[8] = (v167_data + (v124_data * v165_data));
            float v170_data = s0[109];
            float v172_data = r1[9];
            r1[9] = (v172_data + (v124_data * v170_data));
            float v175_data = s0[121];
            float v177_data = r1[10];
            r1[10] = (v177_data + (v124_data * v175_data));
            float v180_data = s0[133];
            float v182_data = r1[11];
            r1[11] = (v182_data + (v124_data * v180_data));
          }
          if (v12_lead < 6) {
            float v188_data = r0[2];
            float v189_data = s0[2];
            float v191_data = r1[0];
            r1[0] = (v191_data + (v188_data * v189_data));
            float v194_data = s0[14];
            float v196_data = r1[1];
            r1[1] = (v196_data + (v188_data * v194_data));
            float v199_data = s0[26];
            float v201_data = r1[2];
            r1[2] = (v201_data + (v188_data * v199_data));
            float v204_data = s0[38];
            float v206_data = r1[3];
            r1[3] = (v206_data + (v188_data * v204_data));
            float v209_data = s0[50];
            float v211_data = r1[4];
            r1[4] = (v211_data + (v188_data * v209_data));
            float v214_data = s0[62];
            float v216_data = r1[5];
            r1[5] = (v216_data + (v188_data * v214_data));
            float v219_data = s0[74];
            float v221_data = r1[6];
            r1[6] = (v221_data + (v188_data * v219_data));
            float v224_data = s0[86];
            float v226_data = r1[7];
            r1[7] = (v226_data + (v188_data * v224_data));
            float v229_data = s0[98];
            float v231_data = r1[8];
            r1[8] = (v231_data + (v188_data * v229_data));
            float v234_data = s0[110];
            float v236_data = r1[9];
            r1[9] = (v236_data + (v188_data * v234_data));
            float v239_data = s0[122];
            float v241_data = r1[10];
            r1[10] = (v241_data + (v188_data * v239_data));
            float v244_data = s0[134];
            float v246_data = r1[11];
            r1[11] = (v246_data + (v188_data * v244_data));
          }
          if (v12_lead < 6) {
            float v252_data = r0[3];
            float v253_data = s0[3];
            float v255_data = r1[0];
            r1[0] = (v255_data + (v252_data * v253_data));
            float v258_data = s0[15];
            float v260_data = r1[1];
            r1[1] = (v260_data + (v252_data * v258_data));
            float v263_data = s0[27];
            float v265_data = r1[2];
            r1[2] = (v265_data + (v252_data * v263_data));
            float v268_data = s0[39];
            float v270_data = r1[3];
            r1[3] = (v270_data + (v252_data * v268_data));
            float v273_data = s0[51];
            float v275_data = r1[4];
            r1[4] = (v275_data + (v252_data * v273_data));
            float v278_data = s0[63];
            float v280_data = r1[5];
            r1[5] = (v280_data + (v252_data * v278_data));
            float v283_data = s0[75];
            float v285_data = r1[6];
            r1[6] = (v285_data + (v252_data * v283_data));
            float v288_data = s0[87];
            float v290_data = r1[7];
            r1[7] = (v290_data + (v252_data * v288_data));
            float v293_data = s0[99];
            float v295_data = r1[8];
            r1[8] = (v295_data + (v252_data * v293_data));
            float v298_data = s0[111];
            float v300_data = r1[9];
            r1[9] = (v300_data + (v252_data * v298_data));
            float v303_data = s0[123];
            float v305_data = r1[10];
            r1[10] = (v305_data + (v252_data * v303_data));
            float v308_data = s0[135];
            float v310_data = r1[11];
            r1[11] = (v310_data + (v252_data * v308_data));
          }
          if (v12_lead < 6) {
            float v316_data = r0[4];
            float v317_data = s0[4];
            float v319_data = r1[0];
            r1[0] = (v319_data + (v316_data * v317_data));
            float v322_data = s0[16];
            float v324_data = r1[1];
            r1[1] = (v324_data + (v316_data * v322_data));
            float v327_data = s0[28];
            float v329_data = r1[2];
            r1[2] = (v329_data + (v316_data * v327_data));
            float v332_data = s0[40];
            float v334_data = r1[3];
            r1[3] = (v334_data + (v316_data * v332_data));
            float v337_data = s0[52];
            float v339_data = r1[4];
            r1[4] = (v339_data + (v316_data * v337_data));
            float v342_data = s0[64];
            float v344_data = r1[5];
            r1[5] = (v344_data + (v316_data * v342_data));
            float v347_data = s0[76];
            float v349_data = r1[6];
            r1[6] = (v349_data + (v316_data * v347_data));
            float v352_data = s0[88];
            float v354_data = r1[7];
            r1[7] = (v354_data + (v316_data * v352_data));
            float v357_data = s0[100];
            float v359_data = r1[8];
            r1[8] = (v359_data + (v316_data * v357_data));
            float v362_data = s0[112];
            float v364_data = r1[9];
            r1[9] = (v364_data + (v316_data * v362_data));
            float v367_data = s0[124];
            float v369_data = r1[10];
            r1[10] = (v369_data + (v316_data * v367_data));
            float v372_data = s0[136];
            float v374_data = r1[11];
            r1[11] = (v374_data + (v316_data * v372_data));
          }
          if (v12_lead < 6) {
            float v380_data = r0[5];
            float v381_data = s0[5];
            float v383_data = r1[0];
            r1[0] = (v383_data + (v380_data * v381_data));
            float v386_data = s0[17];
            float v388_data = r1[1];
            r1[1] = (v388_data + (v380_data * v386_data));
            float v391_data = s0[29];
            float v393_data = r1[2];
            r1[2] = (v393_data + (v380_data * v391_data));
            float v396_data = s0[41];
            float v398_data = r1[3];
            r1[3] = (v398_data + (v380_data * v396_data));
            float v401_data = s0[53];
            float v403_data = r1[4];
            r1[4] = (v403_data + (v380_data * v401_data));
            float v406_data = s0[65];
            float v408_data = r1[5];
            r1[5] = (v408_data + (v380_data * v406_data));
            float v411_data = s0[77];
            float v413_data = r1[6];
            r1[6] = (v413_data + (v380_data * v411_data));
            float v416_data = s0[89];
            float v418_data = r1[7];
            r1[7] = (v418_data + (v380_data * v416_data));
            float v421_data = s0[101];
            float v423_data = r1[8];
            r1[8] = (v423_data + (v380_data * v421_data));
            float v426_data = s0[113];
            float v428_data = r1[9];
            r1[9] = (v428_data + (v380_data * v426_data));
            float v431_data = s0[125];
            float v433_data = r1[10];
            r1[10] = (v433_data + (v380_data * v431_data));
            float v436_data = s0[137];
            float v438_data = r1[11];
            r1[11] = (v438_data + (v380_data * v436_data));
          }
          if (v12_lead < 6) {
            float v444_data = r0[6];
            float v445_data = s0[6];
            float v447_data = r1[0];
            r1[0] = (v447_data + (v444_data * v445_data));
            float v450_data = s0[18];
            float v452_data = r1[1];
            r1[1] = (v452_data + (v444_data * v450_data));
            float v455_data = s0[30];
            float v457_data = r1[2];
            r1[2] = (v457_data + (v444_data * v455_data));
            float v460_data = s0[42];
            float v462_data = r1[3];
            r1[3] = (v462_data + (v444_data * v460_data));
            float v465_data = s0[54];
            float v467_data = r1[4];
            r1[4] = (v467_data + (v444_data * v465_data));
            float v470_data = s0[66];
            float v472_data = r1[5];
            r1[5] = (v472_data + (v444_data * v470_data));
            float v475_data = s0[78];
            float v477_data = r1[6];
            r1[6] = (v477_data + (v444_data * v475_data));
            float v480_data = s0[90];
            float v482_data = r1[7];
            r1[7] = (v482_data + (v444_data * v480_data));
            float v485_data = s0[102];
            float v487_data = r1[8];
            r1[8] = (v487_data + (v444_data * v485_data));
            float v490_data = s0[114];
            float v492_data = r1[9];
            r1[9] = (v492_data + (v444_data * v490_data));
            float v495_data = s0[126];
            float v497_data = r1[10];
            r1[10] = (v497_data + (v444_data * v495_data));
            float v500_data = s0[138];
            float v502_data = r1[11];
            r1[11] = (v502_data + (v444_data * v500_data));
          }
          if (v12_lead < 6) {
            float v508_data = r0[7];
            float v509_data = s0[7];
            float v511_data = r1[0];
            r1[0] = (v511_data + (v508_data * v509_data));
            float v514_data = s0[19];
            float v516_data = r1[1];
            r1[1] = (v516_data + (v508_data * v514_data));
            float v519_data = s0[31];
            float v521_data = r1[2];
            r1[2] = (v521_data + (v508_data * v519_data));
            float v524_data = s0[43];
            float v526_data = r1[3];
            r1[3] = (v526_data + (v508_data * v524_data));
            float v529_data = s0[55];
            float v531_data = r1[4];
            r1[4] = (v531_data + (v508_data * v529_data));
            float v534_data = s0[67];
            float v536_data = r1[5];
            r1[5] = (v536_data + (v508_data * v534_data));
            float v539_data = s0[79];
            float v541_data = r1[6];
            r1[6] = (v541_data + (v508_data * v539_data));
            float v544_data = s0[91];
            float v546_data = r1[7];
            r1[7] = (v546_data + (v508_data * v544_data));
            float v549_data = s0[103];
            float v551_data = r1[8];
            r1[8] = (v551_data + (v508_data * v549_data));
            float v554_data = s0[115];
            float v556_data = r1[9];
            r1[9] = (v556_data + (v508_data * v554_data));
            float v559_data = s0[127];
            float v561_data = r1[10];
            r1[10] = (v561_data + (v508_data * v559_data));
            float v564_data = s0[139];
            float v566_data = r1[11];
            r1[11] = (v566_data + (v508_data * v564_data));
          }
          if (v12_lead < 6) {
            float v572_data = r0[8];
            float v573_data = s0[8];
            float v575_data = r1[0];
            r1[0] = (v575_data + (v572_data * v573_data));
            float v578_data = s0[20];
            float v580_data = r1[1];
            r1[1] = (v580_data + (v572_data * v578_data));
            float v583_data = s0[32];
            float v585_data = r1[2];
            r1[2] = (v585_data + (v572_data * v583_data));
            float v588_data = s0[44];
            float v590_data = r1[3];
            r1[3] = (v590_data + (v572_data * v588_data));
            float v593_data = s0[56];
            float v595_data = r1[4];
            r1[4] = (v595_data + (v572_data * v593_data));
            float v598_data = s0[68];
            float v600_data = r1[5];
            r1[5] = (v600_data + (v572_data * v598_data));
            float v603_data = s0[80];
            float v605_data = r1[6];
            r1[6] = (v605_data + (v572_data * v603_data));
            float v608_data = s0[92];
            float v610_data = r1[7];
            r1[7] = (v610_data + (v572_data * v608_data));
            float v613_data = s0[104];
            float v615_data = r1[8];
            r1[8] = (v615_data + (v572_data * v613_data));
            float v618_data = s0[116];
            float v620_data = r1[9];
            r1[9] = (v620_data + (v572_data * v618_data));
            float v623_data = s0[128];
            float v625_data = r1[10];
            r1[10] = (v625_data + (v572_data * v623_data));
            float v628_data = s0[140];
            float v630_data = r1[11];
            r1[11] = (v630_data + (v572_data * v628_data));
          }
          if (v12_lead < 6) {
            float v636_data = r0[9];
            float v637_data = s0[9];
            float v639_data = r1[0];
            r1[0] = (v639_data + (v636_data * v637_data));
            float v642_data = s0[21];
            float v644_data = r1[1];
            r1[1] = (v644_data + (v636_data * v642_data));
            float v647_data = s0[33];
            float v649_data = r1[2];
            r1[2] = (v649_data + (v636_data * v647_data));
            float v652_data = s0[45];
            float v654_data = r1[3];
            r1[3] = (v654_data + (v636_data * v652_data));
            float v657_data = s0[57];
            float v659_data = r1[4];
            r1[4] = (v659_data + (v636_data * v657_data));
            float v662_data = s0[69];
            float v664_data = r1[5];
            r1[5] = (v664_data + (v636_data * v662_data));
            float v667_data = s0[81];
            float v669_data = r1[6];
            r1[6] = (v669_data + (v636_data * v667_data));
            float v672_data = s0[93];
            float v674_data = r1[7];
            r1[7] = (v674_data + (v636_data * v672_data));
            float v677_data = s0[105];
            float v679_data = r1[8];
            r1[8] = (v679_data + (v636_data * v677_data));
            float v682_data = s0[117];
            float v684_data = r1[9];
            r1[9] = (v684_data + (v636_data * v682_data));
            float v687_data = s0[129];
            float v689_data = r1[10];
            r1[10] = (v689_data + (v636_data * v687_data));
            float v692_data = s0[141];
            float v694_data = r1[11];
            r1[11] = (v694_data + (v636_data * v692_data));
          }
          if (v12_lead < 6) {
            float v700_data = r0[10];
            float v701_data = s0[10];
            float v703_data = r1[0];
            r1[0] = (v703_data + (v700_data * v701_data));
            float v706_data = s0[22];
            float v708_data = r1[1];
            r1[1] = (v708_data + (v700_data * v706_data));
            float v711_data = s0[34];
            float v713_data = r1[2];
            r1[2] = (v713_data + (v700_data * v711_data));
            float v716_data = s0[46];
            float v718_data = r1[3];
            r1[3] = (v718_data + (v700_data * v716_data));
            float v721_data = s0[58];
            float v723_data = r1[4];
            r1[4] = (v723_data + (v700_data * v721_data));
            float v726_data = s0[70];
            float v728_data = r1[5];
            r1[5] = (v728_data + (v700_data * v726_data));
            float v731_data = s0[82];
            float v733_data = r1[6];
            r1[6] = (v733_data + (v700_data * v731_data));
            float v736_data = s0[94];
            float v738_data = r1[7];
            r1[7] = (v738_data + (v700_data * v736_data));
            float v741_data = s0[106];
            float v743_data = r1[8];
            r1[8] = (v743_data + (v700_data * v741_data));
            float v746_data = s0[118];
            float v748_data = r1[9];
            r1[9] = (v748_data + (v700_data * v746_data));
            float v751_data = s0[130];
            float v753_data = r1[10];
            r1[10] = (v753_data + (v700_data * v751_data));
            float v756_data = s0[142];
            float v758_data = r1[11];
            r1[11] = (v758_data + (v700_data * v756_data));
          }
          if (v12_lead < 6) {
            float v764_data = r0[11];
            float v765_data = s0[11];
            float v767_data = r1[0];
            r1[0] = (v767_data + (v764_data * v765_data));
            float v770_data = s0[23];
            float v772_data = r1[1];
            r1[1] = (v772_data + (v764_data * v770_data));
            float v775_data = s0[35];
            float v777_data = r1[2];
            r1[2] = (v777_data + (v764_data * v775_data));
            float v780_data = s0[47];
            float v782_data = r1[3];
            r1[3] = (v782_data + (v764_data * v780_data));
            float v785_data = s0[59];
            float v787_data = r1[4];
            r1[4] = (v787_data + (v764_data * v785_data));
            float v790_data = s0[71];
            float v792_data = r1[5];
            r1[5] = (v792_data + (v764_data * v790_data));
            float v795_data = s0[83];
            float v797_data = r1[6];
            r1[6] = (v797_data + (v764_data * v795_data));
            float v800_data = s0[95];
            float v802_data = r1[7];
            r1[7] = (v802_data + (v764_data * v800_data));
            float v805_data = s0[107];
            float v807_data = r1[8];
            r1[8] = (v807_data + (v764_data * v805_data));
            float v810_data = s0[119];
            float v812_data = r1[9];
            r1[9] = (v812_data + (v764_data * v810_data));
            float v815_data = s0[131];
            float v817_data = r1[10];
            r1[10] = (v817_data + (v764_data * v815_data));
            float v820_data = s0[143];
            float v822_data = r1[11];
            r1[11] = (v822_data + (v764_data * v820_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v829_i1 = 0; v829_i1 < 12; ++v829_i1) {
              int32_t v830_a = 0 + v829_i1;
              float v832_data = r1[v829_i1];
              s1[(v12_lead + (v829_i1 * 12))] = v832_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v845_i1 = 0; v845_i1 < 12; ++v845_i1) {
              int32_t v851_a = v845_i1 * 12;
              int32_t v852_a = v12_lead + v851_a;
              float v860_data = __ldcg(&glb_m4[(v12_lead + v851_a)]);
              r4[v845_i1] = v860_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v12_lead < 6) {
            float v868_data = r2[0];
            float v869_data = s0[0];
            float v871_data = ir3[0];
            ir3[0] = (v871_data + (v868_data * v869_data));
            float v874_data = s0[12];
            float v876_data = ir3[1];
            ir3[1] = (v876_data + (v868_data * v874_data));
            float v879_data = s0[24];
            float v881_data = ir3[2];
            ir3[2] = (v881_data + (v868_data * v879_data));
            float v884_data = s0[36];
            float v886_data = ir3[3];
            ir3[3] = (v886_data + (v868_data * v884_data));
            float v889_data = s0[48];
            float v891_data = ir3[4];
            ir3[4] = (v891_data + (v868_data * v889_data));
            float v894_data = s0[60];
            float v896_data = ir3[5];
            ir3[5] = (v896_data + (v868_data * v894_data));
            float v899_data = s0[72];
            float v901_data = ir3[6];
            ir3[6] = (v901_data + (v868_data * v899_data));
            float v904_data = s0[84];
            float v906_data = ir3[7];
            ir3[7] = (v906_data + (v868_data * v904_data));
            float v909_data = s0[96];
            float v911_data = ir3[8];
            ir3[8] = (v911_data + (v868_data * v909_data));
            float v914_data = s0[108];
            float v916_data = ir3[9];
            ir3[9] = (v916_data + (v868_data * v914_data));
            float v919_data = s0[120];
            float v921_data = ir3[10];
            ir3[10] = (v921_data + (v868_data * v919_data));
            float v924_data = s0[132];
            float v926_data = ir3[11];
            ir3[11] = (v926_data + (v868_data * v924_data));
          }
          if (v12_lead < 6) {
            float v932_data = r2[1];
            float v933_data = s0[1];
            float v935_data = ir3[0];
            ir3[0] = (v935_data + (v932_data * v933_data));
            float v938_data = s0[13];
            float v940_data = ir3[1];
            ir3[1] = (v940_data + (v932_data * v938_data));
            float v943_data = s0[25];
            float v945_data = ir3[2];
            ir3[2] = (v945_data + (v932_data * v943_data));
            float v948_data = s0[37];
            float v950_data = ir3[3];
            ir3[3] = (v950_data + (v932_data * v948_data));
            float v953_data = s0[49];
            float v955_data = ir3[4];
            ir3[4] = (v955_data + (v932_data * v953_data));
            float v958_data = s0[61];
            float v960_data = ir3[5];
            ir3[5] = (v960_data + (v932_data * v958_data));
            float v963_data = s0[73];
            float v965_data = ir3[6];
            ir3[6] = (v965_data + (v932_data * v963_data));
            float v968_data = s0[85];
            float v970_data = ir3[7];
            ir3[7] = (v970_data + (v932_data * v968_data));
            float v973_data = s0[97];
            float v975_data = ir3[8];
            ir3[8] = (v975_data + (v932_data * v973_data));
            float v978_data = s0[109];
            float v980_data = ir3[9];
            ir3[9] = (v980_data + (v932_data * v978_data));
            float v983_data = s0[121];
            float v985_data = ir3[10];
            ir3[10] = (v985_data + (v932_data * v983_data));
            float v988_data = s0[133];
            float v990_data = ir3[11];
            ir3[11] = (v990_data + (v932_data * v988_data));
          }
          if (v12_lead < 6) {
            float v996_data = r2[2];
            float v997_data = s0[2];
            float v999_data = ir3[0];
            ir3[0] = (v999_data + (v996_data * v997_data));
            float v1002_data = s0[14];
            float v1004_data = ir3[1];
            ir3[1] = (v1004_data + (v996_data * v1002_data));
            float v1007_data = s0[26];
            float v1009_data = ir3[2];
            ir3[2] = (v1009_data + (v996_data * v1007_data));
            float v1012_data = s0[38];
            float v1014_data = ir3[3];
            ir3[3] = (v1014_data + (v996_data * v1012_data));
            float v1017_data = s0[50];
            float v1019_data = ir3[4];
            ir3[4] = (v1019_data + (v996_data * v1017_data));
            float v1022_data = s0[62];
            float v1024_data = ir3[5];
            ir3[5] = (v1024_data + (v996_data * v1022_data));
            float v1027_data = s0[74];
            float v1029_data = ir3[6];
            ir3[6] = (v1029_data + (v996_data * v1027_data));
            float v1032_data = s0[86];
            float v1034_data = ir3[7];
            ir3[7] = (v1034_data + (v996_data * v1032_data));
            float v1037_data = s0[98];
            float v1039_data = ir3[8];
            ir3[8] = (v1039_data + (v996_data * v1037_data));
            float v1042_data = s0[110];
            float v1044_data = ir3[9];
            ir3[9] = (v1044_data + (v996_data * v1042_data));
            float v1047_data = s0[122];
            float v1049_data = ir3[10];
            ir3[10] = (v1049_data + (v996_data * v1047_data));
            float v1052_data = s0[134];
            float v1054_data = ir3[11];
            ir3[11] = (v1054_data + (v996_data * v1052_data));
          }
          if (v12_lead < 6) {
            float v1060_data = r2[3];
            float v1061_data = s0[3];
            float v1063_data = ir3[0];
            ir3[0] = (v1063_data + (v1060_data * v1061_data));
            float v1066_data = s0[15];
            float v1068_data = ir3[1];
            ir3[1] = (v1068_data + (v1060_data * v1066_data));
            float v1071_data = s0[27];
            float v1073_data = ir3[2];
            ir3[2] = (v1073_data + (v1060_data * v1071_data));
            float v1076_data = s0[39];
            float v1078_data = ir3[3];
            ir3[3] = (v1078_data + (v1060_data * v1076_data));
            float v1081_data = s0[51];
            float v1083_data = ir3[4];
            ir3[4] = (v1083_data + (v1060_data * v1081_data));
            float v1086_data = s0[63];
            float v1088_data = ir3[5];
            ir3[5] = (v1088_data + (v1060_data * v1086_data));
            float v1091_data = s0[75];
            float v1093_data = ir3[6];
            ir3[6] = (v1093_data + (v1060_data * v1091_data));
            float v1096_data = s0[87];
            float v1098_data = ir3[7];
            ir3[7] = (v1098_data + (v1060_data * v1096_data));
            float v1101_data = s0[99];
            float v1103_data = ir3[8];
            ir3[8] = (v1103_data + (v1060_data * v1101_data));
            float v1106_data = s0[111];
            float v1108_data = ir3[9];
            ir3[9] = (v1108_data + (v1060_data * v1106_data));
            float v1111_data = s0[123];
            float v1113_data = ir3[10];
            ir3[10] = (v1113_data + (v1060_data * v1111_data));
            float v1116_data = s0[135];
            float v1118_data = ir3[11];
            ir3[11] = (v1118_data + (v1060_data * v1116_data));
          }
          if (v12_lead < 6) {
            float v1124_data = r2[4];
            float v1125_data = s0[4];
            float v1127_data = ir3[0];
            ir3[0] = (v1127_data + (v1124_data * v1125_data));
            float v1130_data = s0[16];
            float v1132_data = ir3[1];
            ir3[1] = (v1132_data + (v1124_data * v1130_data));
            float v1135_data = s0[28];
            float v1137_data = ir3[2];
            ir3[2] = (v1137_data + (v1124_data * v1135_data));
            float v1140_data = s0[40];
            float v1142_data = ir3[3];
            ir3[3] = (v1142_data + (v1124_data * v1140_data));
            float v1145_data = s0[52];
            float v1147_data = ir3[4];
            ir3[4] = (v1147_data + (v1124_data * v1145_data));
            float v1150_data = s0[64];
            float v1152_data = ir3[5];
            ir3[5] = (v1152_data + (v1124_data * v1150_data));
            float v1155_data = s0[76];
            float v1157_data = ir3[6];
            ir3[6] = (v1157_data + (v1124_data * v1155_data));
            float v1160_data = s0[88];
            float v1162_data = ir3[7];
            ir3[7] = (v1162_data + (v1124_data * v1160_data));
            float v1165_data = s0[100];
            float v1167_data = ir3[8];
            ir3[8] = (v1167_data + (v1124_data * v1165_data));
            float v1170_data = s0[112];
            float v1172_data = ir3[9];
            ir3[9] = (v1172_data + (v1124_data * v1170_data));
            float v1175_data = s0[124];
            float v1177_data = ir3[10];
            ir3[10] = (v1177_data + (v1124_data * v1175_data));
            float v1180_data = s0[136];
            float v1182_data = ir3[11];
            ir3[11] = (v1182_data + (v1124_data * v1180_data));
          }
          if (v12_lead < 6) {
            float v1188_data = r2[5];
            float v1189_data = s0[5];
            float v1191_data = ir3[0];
            ir3[0] = (v1191_data + (v1188_data * v1189_data));
            float v1194_data = s0[17];
            float v1196_data = ir3[1];
            ir3[1] = (v1196_data + (v1188_data * v1194_data));
            float v1199_data = s0[29];
            float v1201_data = ir3[2];
            ir3[2] = (v1201_data + (v1188_data * v1199_data));
            float v1204_data = s0[41];
            float v1206_data = ir3[3];
            ir3[3] = (v1206_data + (v1188_data * v1204_data));
            float v1209_data = s0[53];
            float v1211_data = ir3[4];
            ir3[4] = (v1211_data + (v1188_data * v1209_data));
            float v1214_data = s0[65];
            float v1216_data = ir3[5];
            ir3[5] = (v1216_data + (v1188_data * v1214_data));
            float v1219_data = s0[77];
            float v1221_data = ir3[6];
            ir3[6] = (v1221_data + (v1188_data * v1219_data));
            float v1224_data = s0[89];
            float v1226_data = ir3[7];
            ir3[7] = (v1226_data + (v1188_data * v1224_data));
            float v1229_data = s0[101];
            float v1231_data = ir3[8];
            ir3[8] = (v1231_data + (v1188_data * v1229_data));
            float v1234_data = s0[113];
            float v1236_data = ir3[9];
            ir3[9] = (v1236_data + (v1188_data * v1234_data));
            float v1239_data = s0[125];
            float v1241_data = ir3[10];
            ir3[10] = (v1241_data + (v1188_data * v1239_data));
            float v1244_data = s0[137];
            float v1246_data = ir3[11];
            ir3[11] = (v1246_data + (v1188_data * v1244_data));
          }
          if (v12_lead < 6) {
            float v1252_data = r2[6];
            float v1253_data = s0[6];
            float v1255_data = ir3[0];
            ir3[0] = (v1255_data + (v1252_data * v1253_data));
            float v1258_data = s0[18];
            float v1260_data = ir3[1];
            ir3[1] = (v1260_data + (v1252_data * v1258_data));
            float v1263_data = s0[30];
            float v1265_data = ir3[2];
            ir3[2] = (v1265_data + (v1252_data * v1263_data));
            float v1268_data = s0[42];
            float v1270_data = ir3[3];
            ir3[3] = (v1270_data + (v1252_data * v1268_data));
            float v1273_data = s0[54];
            float v1275_data = ir3[4];
            ir3[4] = (v1275_data + (v1252_data * v1273_data));
            float v1278_data = s0[66];
            float v1280_data = ir3[5];
            ir3[5] = (v1280_data + (v1252_data * v1278_data));
            float v1283_data = s0[78];
            float v1285_data = ir3[6];
            ir3[6] = (v1285_data + (v1252_data * v1283_data));
            float v1288_data = s0[90];
            float v1290_data = ir3[7];
            ir3[7] = (v1290_data + (v1252_data * v1288_data));
            float v1293_data = s0[102];
            float v1295_data = ir3[8];
            ir3[8] = (v1295_data + (v1252_data * v1293_data));
            float v1298_data = s0[114];
            float v1300_data = ir3[9];
            ir3[9] = (v1300_data + (v1252_data * v1298_data));
            float v1303_data = s0[126];
            float v1305_data = ir3[10];
            ir3[10] = (v1305_data + (v1252_data * v1303_data));
            float v1308_data = s0[138];
            float v1310_data = ir3[11];
            ir3[11] = (v1310_data + (v1252_data * v1308_data));
          }
          if (v12_lead < 6) {
            float v1316_data = r2[7];
            float v1317_data = s0[7];
            float v1319_data = ir3[0];
            ir3[0] = (v1319_data + (v1316_data * v1317_data));
            float v1322_data = s0[19];
            float v1324_data = ir3[1];
            ir3[1] = (v1324_data + (v1316_data * v1322_data));
            float v1327_data = s0[31];
            float v1329_data = ir3[2];
            ir3[2] = (v1329_data + (v1316_data * v1327_data));
            float v1332_data = s0[43];
            float v1334_data = ir3[3];
            ir3[3] = (v1334_data + (v1316_data * v1332_data));
            float v1337_data = s0[55];
            float v1339_data = ir3[4];
            ir3[4] = (v1339_data + (v1316_data * v1337_data));
            float v1342_data = s0[67];
            float v1344_data = ir3[5];
            ir3[5] = (v1344_data + (v1316_data * v1342_data));
            float v1347_data = s0[79];
            float v1349_data = ir3[6];
            ir3[6] = (v1349_data + (v1316_data * v1347_data));
            float v1352_data = s0[91];
            float v1354_data = ir3[7];
            ir3[7] = (v1354_data + (v1316_data * v1352_data));
            float v1357_data = s0[103];
            float v1359_data = ir3[8];
            ir3[8] = (v1359_data + (v1316_data * v1357_data));
            float v1362_data = s0[115];
            float v1364_data = ir3[9];
            ir3[9] = (v1364_data + (v1316_data * v1362_data));
            float v1367_data = s0[127];
            float v1369_data = ir3[10];
            ir3[10] = (v1369_data + (v1316_data * v1367_data));
            float v1372_data = s0[139];
            float v1374_data = ir3[11];
            ir3[11] = (v1374_data + (v1316_data * v1372_data));
          }
          if (v12_lead < 6) {
            float v1380_data = r2[8];
            float v1381_data = s0[8];
            float v1383_data = ir3[0];
            ir3[0] = (v1383_data + (v1380_data * v1381_data));
            float v1386_data = s0[20];
            float v1388_data = ir3[1];
            ir3[1] = (v1388_data + (v1380_data * v1386_data));
            float v1391_data = s0[32];
            float v1393_data = ir3[2];
            ir3[2] = (v1393_data + (v1380_data * v1391_data));
            float v1396_data = s0[44];
            float v1398_data = ir3[3];
            ir3[3] = (v1398_data + (v1380_data * v1396_data));
            float v1401_data = s0[56];
            float v1403_data = ir3[4];
            ir3[4] = (v1403_data + (v1380_data * v1401_data));
            float v1406_data = s0[68];
            float v1408_data = ir3[5];
            ir3[5] = (v1408_data + (v1380_data * v1406_data));
            float v1411_data = s0[80];
            float v1413_data = ir3[6];
            ir3[6] = (v1413_data + (v1380_data * v1411_data));
            float v1416_data = s0[92];
            float v1418_data = ir3[7];
            ir3[7] = (v1418_data + (v1380_data * v1416_data));
            float v1421_data = s0[104];
            float v1423_data = ir3[8];
            ir3[8] = (v1423_data + (v1380_data * v1421_data));
            float v1426_data = s0[116];
            float v1428_data = ir3[9];
            ir3[9] = (v1428_data + (v1380_data * v1426_data));
            float v1431_data = s0[128];
            float v1433_data = ir3[10];
            ir3[10] = (v1433_data + (v1380_data * v1431_data));
            float v1436_data = s0[140];
            float v1438_data = ir3[11];
            ir3[11] = (v1438_data + (v1380_data * v1436_data));
          }
          if (v12_lead < 6) {
            float v1444_data = r2[9];
            float v1445_data = s0[9];
            float v1447_data = ir3[0];
            ir3[0] = (v1447_data + (v1444_data * v1445_data));
            float v1450_data = s0[21];
            float v1452_data = ir3[1];
            ir3[1] = (v1452_data + (v1444_data * v1450_data));
            float v1455_data = s0[33];
            float v1457_data = ir3[2];
            ir3[2] = (v1457_data + (v1444_data * v1455_data));
            float v1460_data = s0[45];
            float v1462_data = ir3[3];
            ir3[3] = (v1462_data + (v1444_data * v1460_data));
            float v1465_data = s0[57];
            float v1467_data = ir3[4];
            ir3[4] = (v1467_data + (v1444_data * v1465_data));
            float v1470_data = s0[69];
            float v1472_data = ir3[5];
            ir3[5] = (v1472_data + (v1444_data * v1470_data));
            float v1475_data = s0[81];
            float v1477_data = ir3[6];
            ir3[6] = (v1477_data + (v1444_data * v1475_data));
            float v1480_data = s0[93];
            float v1482_data = ir3[7];
            ir3[7] = (v1482_data + (v1444_data * v1480_data));
            float v1485_data = s0[105];
            float v1487_data = ir3[8];
            ir3[8] = (v1487_data + (v1444_data * v1485_data));
            float v1490_data = s0[117];
            float v1492_data = ir3[9];
            ir3[9] = (v1492_data + (v1444_data * v1490_data));
            float v1495_data = s0[129];
            float v1497_data = ir3[10];
            ir3[10] = (v1497_data + (v1444_data * v1495_data));
            float v1500_data = s0[141];
            float v1502_data = ir3[11];
            ir3[11] = (v1502_data + (v1444_data * v1500_data));
          }
          if (v12_lead < 6) {
            float v1508_data = r2[10];
            float v1509_data = s0[10];
            float v1511_data = ir3[0];
            ir3[0] = (v1511_data + (v1508_data * v1509_data));
            float v1514_data = s0[22];
            float v1516_data = ir3[1];
            ir3[1] = (v1516_data + (v1508_data * v1514_data));
            float v1519_data = s0[34];
            float v1521_data = ir3[2];
            ir3[2] = (v1521_data + (v1508_data * v1519_data));
            float v1524_data = s0[46];
            float v1526_data = ir3[3];
            ir3[3] = (v1526_data + (v1508_data * v1524_data));
            float v1529_data = s0[58];
            float v1531_data = ir3[4];
            ir3[4] = (v1531_data + (v1508_data * v1529_data));
            float v1534_data = s0[70];
            float v1536_data = ir3[5];
            ir3[5] = (v1536_data + (v1508_data * v1534_data));
            float v1539_data = s0[82];
            float v1541_data = ir3[6];
            ir3[6] = (v1541_data + (v1508_data * v1539_data));
            float v1544_data = s0[94];
            float v1546_data = ir3[7];
            ir3[7] = (v1546_data + (v1508_data * v1544_data));
            float v1549_data = s0[106];
            float v1551_data = ir3[8];
            ir3[8] = (v1551_data + (v1508_data * v1549_data));
            float v1554_data = s0[118];
            float v1556_data = ir3[9];
            ir3[9] = (v1556_data + (v1508_data * v1554_data));
            float v1559_data = s0[130];
            float v1561_data = ir3[10];
            ir3[10] = (v1561_data + (v1508_data * v1559_data));
            float v1564_data = s0[142];
            float v1566_data = ir3[11];
            ir3[11] = (v1566_data + (v1508_data * v1564_data));
          }
          if (v12_lead < 6) {
            float v1572_data = r2[11];
            float v1573_data = s0[11];
            float v1575_data = ir3[0];
            ir3[0] = (v1575_data + (v1572_data * v1573_data));
            float v1578_data = s0[23];
            float v1580_data = ir3[1];
            ir3[1] = (v1580_data + (v1572_data * v1578_data));
            float v1583_data = s0[35];
            float v1585_data = ir3[2];
            ir3[2] = (v1585_data + (v1572_data * v1583_data));
            float v1588_data = s0[47];
            float v1590_data = ir3[3];
            ir3[3] = (v1590_data + (v1572_data * v1588_data));
            float v1593_data = s0[59];
            float v1595_data = ir3[4];
            ir3[4] = (v1595_data + (v1572_data * v1593_data));
            float v1598_data = s0[71];
            float v1600_data = ir3[5];
            ir3[5] = (v1600_data + (v1572_data * v1598_data));
            float v1603_data = s0[83];
            float v1605_data = ir3[6];
            ir3[6] = (v1605_data + (v1572_data * v1603_data));
            float v1608_data = s0[95];
            float v1610_data = ir3[7];
            ir3[7] = (v1610_data + (v1572_data * v1608_data));
            float v1613_data = s0[107];
            float v1615_data = ir3[8];
            ir3[8] = (v1615_data + (v1572_data * v1613_data));
            float v1618_data = s0[119];
            float v1620_data = ir3[9];
            ir3[9] = (v1620_data + (v1572_data * v1618_data));
            float v1623_data = s0[131];
            float v1625_data = ir3[10];
            ir3[10] = (v1625_data + (v1572_data * v1623_data));
            float v1628_data = s0[143];
            float v1630_data = ir3[11];
            ir3[11] = (v1630_data + (v1572_data * v1628_data));
          }
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v1636_n1 = 0; v1636_n1 < 12; ++v1636_n1) {
              int32_t v1637_a = 0 + v1636_n1;
              float v1639_data = ir3[v1636_n1];
              r3[v1636_n1] = v1639_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 6) {
            int32_t v1654_off = v12_lead + 6;
            #pragma unroll
            for (int32_t v1645_i1 = 0; v1645_i1 < 12; ++v1645_i1) {
              int32_t v1646_a = 0 + v1645_i1;
              float v1648_data = r3[v1645_i1];
              s1[(v1654_off + (v1645_i1 * 12))] = v1648_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v12_lead < 12) {
            float v1663_data = r4[0];
            float v1664_data = s1[0];
            float v1666_data = ir5[0];
            ir5[0] = (v1666_data + (v1663_data * v1664_data));
            float v1669_data = s1[12];
            float v1671_data = ir5[1];
            ir5[1] = (v1671_data + (v1663_data * v1669_data));
            float v1674_data = s1[24];
            float v1676_data = ir5[2];
            ir5[2] = (v1676_data + (v1663_data * v1674_data));
            float v1679_data = s1[36];
            float v1681_data = ir5[3];
            ir5[3] = (v1681_data + (v1663_data * v1679_data));
            float v1684_data = s1[48];
            float v1686_data = ir5[4];
            ir5[4] = (v1686_data + (v1663_data * v1684_data));
            float v1689_data = s1[60];
            float v1691_data = ir5[5];
            ir5[5] = (v1691_data + (v1663_data * v1689_data));
            float v1694_data = s1[72];
            float v1696_data = ir5[6];
            ir5[6] = (v1696_data + (v1663_data * v1694_data));
            float v1699_data = s1[84];
            float v1701_data = ir5[7];
            ir5[7] = (v1701_data + (v1663_data * v1699_data));
            float v1704_data = s1[96];
            float v1706_data = ir5[8];
            ir5[8] = (v1706_data + (v1663_data * v1704_data));
            float v1709_data = s1[108];
            float v1711_data = ir5[9];
            ir5[9] = (v1711_data + (v1663_data * v1709_data));
            float v1714_data = s1[120];
            float v1716_data = ir5[10];
            ir5[10] = (v1716_data + (v1663_data * v1714_data));
            float v1719_data = s1[132];
            float v1721_data = ir5[11];
            ir5[11] = (v1721_data + (v1663_data * v1719_data));
          }
          if (v12_lead < 12) {
            float v1727_data = r4[1];
            float v1728_data = s1[1];
            float v1730_data = ir5[0];
            ir5[0] = (v1730_data + (v1727_data * v1728_data));
            float v1733_data = s1[13];
            float v1735_data = ir5[1];
            ir5[1] = (v1735_data + (v1727_data * v1733_data));
            float v1738_data = s1[25];
            float v1740_data = ir5[2];
            ir5[2] = (v1740_data + (v1727_data * v1738_data));
            float v1743_data = s1[37];
            float v1745_data = ir5[3];
            ir5[3] = (v1745_data + (v1727_data * v1743_data));
            float v1748_data = s1[49];
            float v1750_data = ir5[4];
            ir5[4] = (v1750_data + (v1727_data * v1748_data));
            float v1753_data = s1[61];
            float v1755_data = ir5[5];
            ir5[5] = (v1755_data + (v1727_data * v1753_data));
            float v1758_data = s1[73];
            float v1760_data = ir5[6];
            ir5[6] = (v1760_data + (v1727_data * v1758_data));
            float v1763_data = s1[85];
            float v1765_data = ir5[7];
            ir5[7] = (v1765_data + (v1727_data * v1763_data));
            float v1768_data = s1[97];
            float v1770_data = ir5[8];
            ir5[8] = (v1770_data + (v1727_data * v1768_data));
            float v1773_data = s1[109];
            float v1775_data = ir5[9];
            ir5[9] = (v1775_data + (v1727_data * v1773_data));
            float v1778_data = s1[121];
            float v1780_data = ir5[10];
            ir5[10] = (v1780_data + (v1727_data * v1778_data));
            float v1783_data = s1[133];
            float v1785_data = ir5[11];
            ir5[11] = (v1785_data + (v1727_data * v1783_data));
          }
          if (v12_lead < 12) {
            float v1791_data = r4[2];
            float v1792_data = s1[2];
            float v1794_data = ir5[0];
            ir5[0] = (v1794_data + (v1791_data * v1792_data));
            float v1797_data = s1[14];
            float v1799_data = ir5[1];
            ir5[1] = (v1799_data + (v1791_data * v1797_data));
            float v1802_data = s1[26];
            float v1804_data = ir5[2];
            ir5[2] = (v1804_data + (v1791_data * v1802_data));
            float v1807_data = s1[38];
            float v1809_data = ir5[3];
            ir5[3] = (v1809_data + (v1791_data * v1807_data));
            float v1812_data = s1[50];
            float v1814_data = ir5[4];
            ir5[4] = (v1814_data + (v1791_data * v1812_data));
            float v1817_data = s1[62];
            float v1819_data = ir5[5];
            ir5[5] = (v1819_data + (v1791_data * v1817_data));
            float v1822_data = s1[74];
            float v1824_data = ir5[6];
            ir5[6] = (v1824_data + (v1791_data * v1822_data));
            float v1827_data = s1[86];
            float v1829_data = ir5[7];
            ir5[7] = (v1829_data + (v1791_data * v1827_data));
            float v1832_data = s1[98];
            float v1834_data = ir5[8];
            ir5[8] = (v1834_data + (v1791_data * v1832_data));
            float v1837_data = s1[110];
            float v1839_data = ir5[9];
            ir5[9] = (v1839_data + (v1791_data * v1837_data));
            float v1842_data = s1[122];
            float v1844_data = ir5[10];
            ir5[10] = (v1844_data + (v1791_data * v1842_data));
            float v1847_data = s1[134];
            float v1849_data = ir5[11];
            ir5[11] = (v1849_data + (v1791_data * v1847_data));
          }
          if (v12_lead < 12) {
            float v1855_data = r4[3];
            float v1856_data = s1[3];
            float v1858_data = ir5[0];
            ir5[0] = (v1858_data + (v1855_data * v1856_data));
            float v1861_data = s1[15];
            float v1863_data = ir5[1];
            ir5[1] = (v1863_data + (v1855_data * v1861_data));
            float v1866_data = s1[27];
            float v1868_data = ir5[2];
            ir5[2] = (v1868_data + (v1855_data * v1866_data));
            float v1871_data = s1[39];
            float v1873_data = ir5[3];
            ir5[3] = (v1873_data + (v1855_data * v1871_data));
            float v1876_data = s1[51];
            float v1878_data = ir5[4];
            ir5[4] = (v1878_data + (v1855_data * v1876_data));
            float v1881_data = s1[63];
            float v1883_data = ir5[5];
            ir5[5] = (v1883_data + (v1855_data * v1881_data));
            float v1886_data = s1[75];
            float v1888_data = ir5[6];
            ir5[6] = (v1888_data + (v1855_data * v1886_data));
            float v1891_data = s1[87];
            float v1893_data = ir5[7];
            ir5[7] = (v1893_data + (v1855_data * v1891_data));
            float v1896_data = s1[99];
            float v1898_data = ir5[8];
            ir5[8] = (v1898_data + (v1855_data * v1896_data));
            float v1901_data = s1[111];
            float v1903_data = ir5[9];
            ir5[9] = (v1903_data + (v1855_data * v1901_data));
            float v1906_data = s1[123];
            float v1908_data = ir5[10];
            ir5[10] = (v1908_data + (v1855_data * v1906_data));
            float v1911_data = s1[135];
            float v1913_data = ir5[11];
            ir5[11] = (v1913_data + (v1855_data * v1911_data));
          }
          if (v12_lead < 12) {
            float v1919_data = r4[4];
            float v1920_data = s1[4];
            float v1922_data = ir5[0];
            ir5[0] = (v1922_data + (v1919_data * v1920_data));
            float v1925_data = s1[16];
            float v1927_data = ir5[1];
            ir5[1] = (v1927_data + (v1919_data * v1925_data));
            float v1930_data = s1[28];
            float v1932_data = ir5[2];
            ir5[2] = (v1932_data + (v1919_data * v1930_data));
            float v1935_data = s1[40];
            float v1937_data = ir5[3];
            ir5[3] = (v1937_data + (v1919_data * v1935_data));
            float v1940_data = s1[52];
            float v1942_data = ir5[4];
            ir5[4] = (v1942_data + (v1919_data * v1940_data));
            float v1945_data = s1[64];
            float v1947_data = ir5[5];
            ir5[5] = (v1947_data + (v1919_data * v1945_data));
            float v1950_data = s1[76];
            float v1952_data = ir5[6];
            ir5[6] = (v1952_data + (v1919_data * v1950_data));
            float v1955_data = s1[88];
            float v1957_data = ir5[7];
            ir5[7] = (v1957_data + (v1919_data * v1955_data));
            float v1960_data = s1[100];
            float v1962_data = ir5[8];
            ir5[8] = (v1962_data + (v1919_data * v1960_data));
            float v1965_data = s1[112];
            float v1967_data = ir5[9];
            ir5[9] = (v1967_data + (v1919_data * v1965_data));
            float v1970_data = s1[124];
            float v1972_data = ir5[10];
            ir5[10] = (v1972_data + (v1919_data * v1970_data));
            float v1975_data = s1[136];
            float v1977_data = ir5[11];
            ir5[11] = (v1977_data + (v1919_data * v1975_data));
          }
          if (v12_lead < 12) {
            float v1983_data = r4[5];
            float v1984_data = s1[5];
            float v1986_data = ir5[0];
            ir5[0] = (v1986_data + (v1983_data * v1984_data));
            float v1989_data = s1[17];
            float v1991_data = ir5[1];
            ir5[1] = (v1991_data + (v1983_data * v1989_data));
            float v1994_data = s1[29];
            float v1996_data = ir5[2];
            ir5[2] = (v1996_data + (v1983_data * v1994_data));
            float v1999_data = s1[41];
            float v2001_data = ir5[3];
            ir5[3] = (v2001_data + (v1983_data * v1999_data));
            float v2004_data = s1[53];
            float v2006_data = ir5[4];
            ir5[4] = (v2006_data + (v1983_data * v2004_data));
            float v2009_data = s1[65];
            float v2011_data = ir5[5];
            ir5[5] = (v2011_data + (v1983_data * v2009_data));
            float v2014_data = s1[77];
            float v2016_data = ir5[6];
            ir5[6] = (v2016_data + (v1983_data * v2014_data));
            float v2019_data = s1[89];
            float v2021_data = ir5[7];
            ir5[7] = (v2021_data + (v1983_data * v2019_data));
            float v2024_data = s1[101];
            float v2026_data = ir5[8];
            ir5[8] = (v2026_data + (v1983_data * v2024_data));
            float v2029_data = s1[113];
            float v2031_data = ir5[9];
            ir5[9] = (v2031_data + (v1983_data * v2029_data));
            float v2034_data = s1[125];
            float v2036_data = ir5[10];
            ir5[10] = (v2036_data + (v1983_data * v2034_data));
            float v2039_data = s1[137];
            float v2041_data = ir5[11];
            ir5[11] = (v2041_data + (v1983_data * v2039_data));
          }
          if (v12_lead < 12) {
            float v2047_data = r4[6];
            float v2048_data = s1[6];
            float v2050_data = ir5[0];
            ir5[0] = (v2050_data + (v2047_data * v2048_data));
            float v2053_data = s1[18];
            float v2055_data = ir5[1];
            ir5[1] = (v2055_data + (v2047_data * v2053_data));
            float v2058_data = s1[30];
            float v2060_data = ir5[2];
            ir5[2] = (v2060_data + (v2047_data * v2058_data));
            float v2063_data = s1[42];
            float v2065_data = ir5[3];
            ir5[3] = (v2065_data + (v2047_data * v2063_data));
            float v2068_data = s1[54];
            float v2070_data = ir5[4];
            ir5[4] = (v2070_data + (v2047_data * v2068_data));
            float v2073_data = s1[66];
            float v2075_data = ir5[5];
            ir5[5] = (v2075_data + (v2047_data * v2073_data));
            float v2078_data = s1[78];
            float v2080_data = ir5[6];
            ir5[6] = (v2080_data + (v2047_data * v2078_data));
            float v2083_data = s1[90];
            float v2085_data = ir5[7];
            ir5[7] = (v2085_data + (v2047_data * v2083_data));
            float v2088_data = s1[102];
            float v2090_data = ir5[8];
            ir5[8] = (v2090_data + (v2047_data * v2088_data));
            float v2093_data = s1[114];
            float v2095_data = ir5[9];
            ir5[9] = (v2095_data + (v2047_data * v2093_data));
            float v2098_data = s1[126];
            float v2100_data = ir5[10];
            ir5[10] = (v2100_data + (v2047_data * v2098_data));
            float v2103_data = s1[138];
            float v2105_data = ir5[11];
            ir5[11] = (v2105_data + (v2047_data * v2103_data));
          }
          if (v12_lead < 12) {
            float v2111_data = r4[7];
            float v2112_data = s1[7];
            float v2114_data = ir5[0];
            ir5[0] = (v2114_data + (v2111_data * v2112_data));
            float v2117_data = s1[19];
            float v2119_data = ir5[1];
            ir5[1] = (v2119_data + (v2111_data * v2117_data));
            float v2122_data = s1[31];
            float v2124_data = ir5[2];
            ir5[2] = (v2124_data + (v2111_data * v2122_data));
            float v2127_data = s1[43];
            float v2129_data = ir5[3];
            ir5[3] = (v2129_data + (v2111_data * v2127_data));
            float v2132_data = s1[55];
            float v2134_data = ir5[4];
            ir5[4] = (v2134_data + (v2111_data * v2132_data));
            float v2137_data = s1[67];
            float v2139_data = ir5[5];
            ir5[5] = (v2139_data + (v2111_data * v2137_data));
            float v2142_data = s1[79];
            float v2144_data = ir5[6];
            ir5[6] = (v2144_data + (v2111_data * v2142_data));
            float v2147_data = s1[91];
            float v2149_data = ir5[7];
            ir5[7] = (v2149_data + (v2111_data * v2147_data));
            float v2152_data = s1[103];
            float v2154_data = ir5[8];
            ir5[8] = (v2154_data + (v2111_data * v2152_data));
            float v2157_data = s1[115];
            float v2159_data = ir5[9];
            ir5[9] = (v2159_data + (v2111_data * v2157_data));
            float v2162_data = s1[127];
            float v2164_data = ir5[10];
            ir5[10] = (v2164_data + (v2111_data * v2162_data));
            float v2167_data = s1[139];
            float v2169_data = ir5[11];
            ir5[11] = (v2169_data + (v2111_data * v2167_data));
          }
          if (v12_lead < 12) {
            float v2175_data = r4[8];
            float v2176_data = s1[8];
            float v2178_data = ir5[0];
            ir5[0] = (v2178_data + (v2175_data * v2176_data));
            float v2181_data = s1[20];
            float v2183_data = ir5[1];
            ir5[1] = (v2183_data + (v2175_data * v2181_data));
            float v2186_data = s1[32];
            float v2188_data = ir5[2];
            ir5[2] = (v2188_data + (v2175_data * v2186_data));
            float v2191_data = s1[44];
            float v2193_data = ir5[3];
            ir5[3] = (v2193_data + (v2175_data * v2191_data));
            float v2196_data = s1[56];
            float v2198_data = ir5[4];
            ir5[4] = (v2198_data + (v2175_data * v2196_data));
            float v2201_data = s1[68];
            float v2203_data = ir5[5];
            ir5[5] = (v2203_data + (v2175_data * v2201_data));
            float v2206_data = s1[80];
            float v2208_data = ir5[6];
            ir5[6] = (v2208_data + (v2175_data * v2206_data));
            float v2211_data = s1[92];
            float v2213_data = ir5[7];
            ir5[7] = (v2213_data + (v2175_data * v2211_data));
            float v2216_data = s1[104];
            float v2218_data = ir5[8];
            ir5[8] = (v2218_data + (v2175_data * v2216_data));
            float v2221_data = s1[116];
            float v2223_data = ir5[9];
            ir5[9] = (v2223_data + (v2175_data * v2221_data));
            float v2226_data = s1[128];
            float v2228_data = ir5[10];
            ir5[10] = (v2228_data + (v2175_data * v2226_data));
            float v2231_data = s1[140];
            float v2233_data = ir5[11];
            ir5[11] = (v2233_data + (v2175_data * v2231_data));
          }
          if (v12_lead < 12) {
            float v2239_data = r4[9];
            float v2240_data = s1[9];
            float v2242_data = ir5[0];
            ir5[0] = (v2242_data + (v2239_data * v2240_data));
            float v2245_data = s1[21];
            float v2247_data = ir5[1];
            ir5[1] = (v2247_data + (v2239_data * v2245_data));
            float v2250_data = s1[33];
            float v2252_data = ir5[2];
            ir5[2] = (v2252_data + (v2239_data * v2250_data));
            float v2255_data = s1[45];
            float v2257_data = ir5[3];
            ir5[3] = (v2257_data + (v2239_data * v2255_data));
            float v2260_data = s1[57];
            float v2262_data = ir5[4];
            ir5[4] = (v2262_data + (v2239_data * v2260_data));
            float v2265_data = s1[69];
            float v2267_data = ir5[5];
            ir5[5] = (v2267_data + (v2239_data * v2265_data));
            float v2270_data = s1[81];
            float v2272_data = ir5[6];
            ir5[6] = (v2272_data + (v2239_data * v2270_data));
            float v2275_data = s1[93];
            float v2277_data = ir5[7];
            ir5[7] = (v2277_data + (v2239_data * v2275_data));
            float v2280_data = s1[105];
            float v2282_data = ir5[8];
            ir5[8] = (v2282_data + (v2239_data * v2280_data));
            float v2285_data = s1[117];
            float v2287_data = ir5[9];
            ir5[9] = (v2287_data + (v2239_data * v2285_data));
            float v2290_data = s1[129];
            float v2292_data = ir5[10];
            ir5[10] = (v2292_data + (v2239_data * v2290_data));
            float v2295_data = s1[141];
            float v2297_data = ir5[11];
            ir5[11] = (v2297_data + (v2239_data * v2295_data));
          }
          if (v12_lead < 12) {
            float v2303_data = r4[10];
            float v2304_data = s1[10];
            float v2306_data = ir5[0];
            ir5[0] = (v2306_data + (v2303_data * v2304_data));
            float v2309_data = s1[22];
            float v2311_data = ir5[1];
            ir5[1] = (v2311_data + (v2303_data * v2309_data));
            float v2314_data = s1[34];
            float v2316_data = ir5[2];
            ir5[2] = (v2316_data + (v2303_data * v2314_data));
            float v2319_data = s1[46];
            float v2321_data = ir5[3];
            ir5[3] = (v2321_data + (v2303_data * v2319_data));
            float v2324_data = s1[58];
            float v2326_data = ir5[4];
            ir5[4] = (v2326_data + (v2303_data * v2324_data));
            float v2329_data = s1[70];
            float v2331_data = ir5[5];
            ir5[5] = (v2331_data + (v2303_data * v2329_data));
            float v2334_data = s1[82];
            float v2336_data = ir5[6];
            ir5[6] = (v2336_data + (v2303_data * v2334_data));
            float v2339_data = s1[94];
            float v2341_data = ir5[7];
            ir5[7] = (v2341_data + (v2303_data * v2339_data));
            float v2344_data = s1[106];
            float v2346_data = ir5[8];
            ir5[8] = (v2346_data + (v2303_data * v2344_data));
            float v2349_data = s1[118];
            float v2351_data = ir5[9];
            ir5[9] = (v2351_data + (v2303_data * v2349_data));
            float v2354_data = s1[130];
            float v2356_data = ir5[10];
            ir5[10] = (v2356_data + (v2303_data * v2354_data));
            float v2359_data = s1[142];
            float v2361_data = ir5[11];
            ir5[11] = (v2361_data + (v2303_data * v2359_data));
          }
          if (v12_lead < 12) {
            float v2367_data = r4[11];
            float v2368_data = s1[11];
            float v2370_data = ir5[0];
            ir5[0] = (v2370_data + (v2367_data * v2368_data));
            float v2373_data = s1[23];
            float v2375_data = ir5[1];
            ir5[1] = (v2375_data + (v2367_data * v2373_data));
            float v2378_data = s1[35];
            float v2380_data = ir5[2];
            ir5[2] = (v2380_data + (v2367_data * v2378_data));
            float v2383_data = s1[47];
            float v2385_data = ir5[3];
            ir5[3] = (v2385_data + (v2367_data * v2383_data));
            float v2388_data = s1[59];
            float v2390_data = ir5[4];
            ir5[4] = (v2390_data + (v2367_data * v2388_data));
            float v2393_data = s1[71];
            float v2395_data = ir5[5];
            ir5[5] = (v2395_data + (v2367_data * v2393_data));
            float v2398_data = s1[83];
            float v2400_data = ir5[6];
            ir5[6] = (v2400_data + (v2367_data * v2398_data));
            float v2403_data = s1[95];
            float v2405_data = ir5[7];
            ir5[7] = (v2405_data + (v2367_data * v2403_data));
            float v2408_data = s1[107];
            float v2410_data = ir5[8];
            ir5[8] = (v2410_data + (v2367_data * v2408_data));
            float v2413_data = s1[119];
            float v2415_data = ir5[9];
            ir5[9] = (v2415_data + (v2367_data * v2413_data));
            float v2418_data = s1[131];
            float v2420_data = ir5[10];
            ir5[10] = (v2420_data + (v2367_data * v2418_data));
            float v2423_data = s1[143];
            float v2425_data = ir5[11];
            ir5[11] = (v2425_data + (v2367_data * v2423_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2431_n1 = 0; v2431_n1 < 12; ++v2431_n1) {
              int32_t v2432_a = 0 + v2431_n1;
              float v2434_data = ir5[v2431_n1];
              r5[v2431_n1] = v2434_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2440_i1 = 0; v2440_i1 < 12; ++v2440_i1) {
              int32_t v2441_a = 0 + v2440_i1;
              float v2443_data = r5[v2440_i1];
              glb_m3[(v12_lead + (v2440_i1 * 12))] = v2443_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

