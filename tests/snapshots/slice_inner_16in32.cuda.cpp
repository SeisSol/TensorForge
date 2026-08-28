// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87f2838a59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_87f2838a59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_87f2838a59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          alignas(16) float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 16;
            int32_t v14_off = (v6_lead + v12_lead) + 8;
            int32_t v22_off = (v6_lead + v12_lead) + 8;
            #pragma unroll
            for (int32_t v8_i1 = 8; v8_i1 < 24; ++v8_i1) {
              int32_t v15_a = v8_i1 * 32;
              int32_t v16_a = v14_off + v15_a;
              float v25_data = __ldcg(&glb_m1[(v22_off + v15_a)]);
              int32_t v27_a = v7_i0 + (v8_i1 - 8);
              r0[v27_a] = v25_data;
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
          // [(0, 16), (0, 8)] [(0, 16)]
          float ir1[8]{};
          float v35_data = r0[0];
          float v36_data = s0[0];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s0[16];
          float v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s0[32];
          float v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s0[48];
          float v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s0[64];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s0[80];
          float v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          float v66_data = s0[96];
          float v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          float v71_data = s0[112];
          float v73_data = ir1[7];
          ir1[7] = (v73_data + (v35_data * v71_data));
          float v78_data = r0[1];
          float v79_data = s0[1];
          float v81_data = ir1[0];
          ir1[0] = (v81_data + (v78_data * v79_data));
          float v84_data = s0[17];
          float v86_data = ir1[1];
          ir1[1] = (v86_data + (v78_data * v84_data));
          float v89_data = s0[33];
          float v91_data = ir1[2];
          ir1[2] = (v91_data + (v78_data * v89_data));
          float v94_data = s0[49];
          float v96_data = ir1[3];
          ir1[3] = (v96_data + (v78_data * v94_data));
          float v99_data = s0[65];
          float v101_data = ir1[4];
          ir1[4] = (v101_data + (v78_data * v99_data));
          float v104_data = s0[81];
          float v106_data = ir1[5];
          ir1[5] = (v106_data + (v78_data * v104_data));
          float v109_data = s0[97];
          float v111_data = ir1[6];
          ir1[6] = (v111_data + (v78_data * v109_data));
          float v114_data = s0[113];
          float v116_data = ir1[7];
          ir1[7] = (v116_data + (v78_data * v114_data));
          float v121_data = r0[2];
          float v122_data = s0[2];
          float v124_data = ir1[0];
          ir1[0] = (v124_data + (v121_data * v122_data));
          float v127_data = s0[18];
          float v129_data = ir1[1];
          ir1[1] = (v129_data + (v121_data * v127_data));
          float v132_data = s0[34];
          float v134_data = ir1[2];
          ir1[2] = (v134_data + (v121_data * v132_data));
          float v137_data = s0[50];
          float v139_data = ir1[3];
          ir1[3] = (v139_data + (v121_data * v137_data));
          float v142_data = s0[66];
          float v144_data = ir1[4];
          ir1[4] = (v144_data + (v121_data * v142_data));
          float v147_data = s0[82];
          float v149_data = ir1[5];
          ir1[5] = (v149_data + (v121_data * v147_data));
          float v152_data = s0[98];
          float v154_data = ir1[6];
          ir1[6] = (v154_data + (v121_data * v152_data));
          float v157_data = s0[114];
          float v159_data = ir1[7];
          ir1[7] = (v159_data + (v121_data * v157_data));
          float v164_data = r0[3];
          float v165_data = s0[3];
          float v167_data = ir1[0];
          ir1[0] = (v167_data + (v164_data * v165_data));
          float v170_data = s0[19];
          float v172_data = ir1[1];
          ir1[1] = (v172_data + (v164_data * v170_data));
          float v175_data = s0[35];
          float v177_data = ir1[2];
          ir1[2] = (v177_data + (v164_data * v175_data));
          float v180_data = s0[51];
          float v182_data = ir1[3];
          ir1[3] = (v182_data + (v164_data * v180_data));
          float v185_data = s0[67];
          float v187_data = ir1[4];
          ir1[4] = (v187_data + (v164_data * v185_data));
          float v190_data = s0[83];
          float v192_data = ir1[5];
          ir1[5] = (v192_data + (v164_data * v190_data));
          float v195_data = s0[99];
          float v197_data = ir1[6];
          ir1[6] = (v197_data + (v164_data * v195_data));
          float v200_data = s0[115];
          float v202_data = ir1[7];
          ir1[7] = (v202_data + (v164_data * v200_data));
          float v207_data = r0[4];
          float v208_data = s0[4];
          float v210_data = ir1[0];
          ir1[0] = (v210_data + (v207_data * v208_data));
          float v213_data = s0[20];
          float v215_data = ir1[1];
          ir1[1] = (v215_data + (v207_data * v213_data));
          float v218_data = s0[36];
          float v220_data = ir1[2];
          ir1[2] = (v220_data + (v207_data * v218_data));
          float v223_data = s0[52];
          float v225_data = ir1[3];
          ir1[3] = (v225_data + (v207_data * v223_data));
          float v228_data = s0[68];
          float v230_data = ir1[4];
          ir1[4] = (v230_data + (v207_data * v228_data));
          float v233_data = s0[84];
          float v235_data = ir1[5];
          ir1[5] = (v235_data + (v207_data * v233_data));
          float v238_data = s0[100];
          float v240_data = ir1[6];
          ir1[6] = (v240_data + (v207_data * v238_data));
          float v243_data = s0[116];
          float v245_data = ir1[7];
          ir1[7] = (v245_data + (v207_data * v243_data));
          float v250_data = r0[5];
          float v251_data = s0[5];
          float v253_data = ir1[0];
          ir1[0] = (v253_data + (v250_data * v251_data));
          float v256_data = s0[21];
          float v258_data = ir1[1];
          ir1[1] = (v258_data + (v250_data * v256_data));
          float v261_data = s0[37];
          float v263_data = ir1[2];
          ir1[2] = (v263_data + (v250_data * v261_data));
          float v266_data = s0[53];
          float v268_data = ir1[3];
          ir1[3] = (v268_data + (v250_data * v266_data));
          float v271_data = s0[69];
          float v273_data = ir1[4];
          ir1[4] = (v273_data + (v250_data * v271_data));
          float v276_data = s0[85];
          float v278_data = ir1[5];
          ir1[5] = (v278_data + (v250_data * v276_data));
          float v281_data = s0[101];
          float v283_data = ir1[6];
          ir1[6] = (v283_data + (v250_data * v281_data));
          float v286_data = s0[117];
          float v288_data = ir1[7];
          ir1[7] = (v288_data + (v250_data * v286_data));
          float v293_data = r0[6];
          float v294_data = s0[6];
          float v296_data = ir1[0];
          ir1[0] = (v296_data + (v293_data * v294_data));
          float v299_data = s0[22];
          float v301_data = ir1[1];
          ir1[1] = (v301_data + (v293_data * v299_data));
          float v304_data = s0[38];
          float v306_data = ir1[2];
          ir1[2] = (v306_data + (v293_data * v304_data));
          float v309_data = s0[54];
          float v311_data = ir1[3];
          ir1[3] = (v311_data + (v293_data * v309_data));
          float v314_data = s0[70];
          float v316_data = ir1[4];
          ir1[4] = (v316_data + (v293_data * v314_data));
          float v319_data = s0[86];
          float v321_data = ir1[5];
          ir1[5] = (v321_data + (v293_data * v319_data));
          float v324_data = s0[102];
          float v326_data = ir1[6];
          ir1[6] = (v326_data + (v293_data * v324_data));
          float v329_data = s0[118];
          float v331_data = ir1[7];
          ir1[7] = (v331_data + (v293_data * v329_data));
          float v336_data = r0[7];
          float v337_data = s0[7];
          float v339_data = ir1[0];
          ir1[0] = (v339_data + (v336_data * v337_data));
          float v342_data = s0[23];
          float v344_data = ir1[1];
          ir1[1] = (v344_data + (v336_data * v342_data));
          float v347_data = s0[39];
          float v349_data = ir1[2];
          ir1[2] = (v349_data + (v336_data * v347_data));
          float v352_data = s0[55];
          float v354_data = ir1[3];
          ir1[3] = (v354_data + (v336_data * v352_data));
          float v357_data = s0[71];
          float v359_data = ir1[4];
          ir1[4] = (v359_data + (v336_data * v357_data));
          float v362_data = s0[87];
          float v364_data = ir1[5];
          ir1[5] = (v364_data + (v336_data * v362_data));
          float v367_data = s0[103];
          float v369_data = ir1[6];
          ir1[6] = (v369_data + (v336_data * v367_data));
          float v372_data = s0[119];
          float v374_data = ir1[7];
          ir1[7] = (v374_data + (v336_data * v372_data));
          float v379_data = r0[8];
          float v380_data = s0[8];
          float v382_data = ir1[0];
          ir1[0] = (v382_data + (v379_data * v380_data));
          float v385_data = s0[24];
          float v387_data = ir1[1];
          ir1[1] = (v387_data + (v379_data * v385_data));
          float v390_data = s0[40];
          float v392_data = ir1[2];
          ir1[2] = (v392_data + (v379_data * v390_data));
          float v395_data = s0[56];
          float v397_data = ir1[3];
          ir1[3] = (v397_data + (v379_data * v395_data));
          float v400_data = s0[72];
          float v402_data = ir1[4];
          ir1[4] = (v402_data + (v379_data * v400_data));
          float v405_data = s0[88];
          float v407_data = ir1[5];
          ir1[5] = (v407_data + (v379_data * v405_data));
          float v410_data = s0[104];
          float v412_data = ir1[6];
          ir1[6] = (v412_data + (v379_data * v410_data));
          float v415_data = s0[120];
          float v417_data = ir1[7];
          ir1[7] = (v417_data + (v379_data * v415_data));
          float v422_data = r0[9];
          float v423_data = s0[9];
          float v425_data = ir1[0];
          ir1[0] = (v425_data + (v422_data * v423_data));
          float v428_data = s0[25];
          float v430_data = ir1[1];
          ir1[1] = (v430_data + (v422_data * v428_data));
          float v433_data = s0[41];
          float v435_data = ir1[2];
          ir1[2] = (v435_data + (v422_data * v433_data));
          float v438_data = s0[57];
          float v440_data = ir1[3];
          ir1[3] = (v440_data + (v422_data * v438_data));
          float v443_data = s0[73];
          float v445_data = ir1[4];
          ir1[4] = (v445_data + (v422_data * v443_data));
          float v448_data = s0[89];
          float v450_data = ir1[5];
          ir1[5] = (v450_data + (v422_data * v448_data));
          float v453_data = s0[105];
          float v455_data = ir1[6];
          ir1[6] = (v455_data + (v422_data * v453_data));
          float v458_data = s0[121];
          float v460_data = ir1[7];
          ir1[7] = (v460_data + (v422_data * v458_data));
          float v465_data = r0[10];
          float v466_data = s0[10];
          float v468_data = ir1[0];
          ir1[0] = (v468_data + (v465_data * v466_data));
          float v471_data = s0[26];
          float v473_data = ir1[1];
          ir1[1] = (v473_data + (v465_data * v471_data));
          float v476_data = s0[42];
          float v478_data = ir1[2];
          ir1[2] = (v478_data + (v465_data * v476_data));
          float v481_data = s0[58];
          float v483_data = ir1[3];
          ir1[3] = (v483_data + (v465_data * v481_data));
          float v486_data = s0[74];
          float v488_data = ir1[4];
          ir1[4] = (v488_data + (v465_data * v486_data));
          float v491_data = s0[90];
          float v493_data = ir1[5];
          ir1[5] = (v493_data + (v465_data * v491_data));
          float v496_data = s0[106];
          float v498_data = ir1[6];
          ir1[6] = (v498_data + (v465_data * v496_data));
          float v501_data = s0[122];
          float v503_data = ir1[7];
          ir1[7] = (v503_data + (v465_data * v501_data));
          float v508_data = r0[11];
          float v509_data = s0[11];
          float v511_data = ir1[0];
          ir1[0] = (v511_data + (v508_data * v509_data));
          float v514_data = s0[27];
          float v516_data = ir1[1];
          ir1[1] = (v516_data + (v508_data * v514_data));
          float v519_data = s0[43];
          float v521_data = ir1[2];
          ir1[2] = (v521_data + (v508_data * v519_data));
          float v524_data = s0[59];
          float v526_data = ir1[3];
          ir1[3] = (v526_data + (v508_data * v524_data));
          float v529_data = s0[75];
          float v531_data = ir1[4];
          ir1[4] = (v531_data + (v508_data * v529_data));
          float v534_data = s0[91];
          float v536_data = ir1[5];
          ir1[5] = (v536_data + (v508_data * v534_data));
          float v539_data = s0[107];
          float v541_data = ir1[6];
          ir1[6] = (v541_data + (v508_data * v539_data));
          float v544_data = s0[123];
          float v546_data = ir1[7];
          ir1[7] = (v546_data + (v508_data * v544_data));
          float v551_data = r0[12];
          float v552_data = s0[12];
          float v554_data = ir1[0];
          ir1[0] = (v554_data + (v551_data * v552_data));
          float v557_data = s0[28];
          float v559_data = ir1[1];
          ir1[1] = (v559_data + (v551_data * v557_data));
          float v562_data = s0[44];
          float v564_data = ir1[2];
          ir1[2] = (v564_data + (v551_data * v562_data));
          float v567_data = s0[60];
          float v569_data = ir1[3];
          ir1[3] = (v569_data + (v551_data * v567_data));
          float v572_data = s0[76];
          float v574_data = ir1[4];
          ir1[4] = (v574_data + (v551_data * v572_data));
          float v577_data = s0[92];
          float v579_data = ir1[5];
          ir1[5] = (v579_data + (v551_data * v577_data));
          float v582_data = s0[108];
          float v584_data = ir1[6];
          ir1[6] = (v584_data + (v551_data * v582_data));
          float v587_data = s0[124];
          float v589_data = ir1[7];
          ir1[7] = (v589_data + (v551_data * v587_data));
          float v594_data = r0[13];
          float v595_data = s0[13];
          float v597_data = ir1[0];
          ir1[0] = (v597_data + (v594_data * v595_data));
          float v600_data = s0[29];
          float v602_data = ir1[1];
          ir1[1] = (v602_data + (v594_data * v600_data));
          float v605_data = s0[45];
          float v607_data = ir1[2];
          ir1[2] = (v607_data + (v594_data * v605_data));
          float v610_data = s0[61];
          float v612_data = ir1[3];
          ir1[3] = (v612_data + (v594_data * v610_data));
          float v615_data = s0[77];
          float v617_data = ir1[4];
          ir1[4] = (v617_data + (v594_data * v615_data));
          float v620_data = s0[93];
          float v622_data = ir1[5];
          ir1[5] = (v622_data + (v594_data * v620_data));
          float v625_data = s0[109];
          float v627_data = ir1[6];
          ir1[6] = (v627_data + (v594_data * v625_data));
          float v630_data = s0[125];
          float v632_data = ir1[7];
          ir1[7] = (v632_data + (v594_data * v630_data));
          float v637_data = r0[14];
          float v638_data = s0[14];
          float v640_data = ir1[0];
          ir1[0] = (v640_data + (v637_data * v638_data));
          float v643_data = s0[30];
          float v645_data = ir1[1];
          ir1[1] = (v645_data + (v637_data * v643_data));
          float v648_data = s0[46];
          float v650_data = ir1[2];
          ir1[2] = (v650_data + (v637_data * v648_data));
          float v653_data = s0[62];
          float v655_data = ir1[3];
          ir1[3] = (v655_data + (v637_data * v653_data));
          float v658_data = s0[78];
          float v660_data = ir1[4];
          ir1[4] = (v660_data + (v637_data * v658_data));
          float v663_data = s0[94];
          float v665_data = ir1[5];
          ir1[5] = (v665_data + (v637_data * v663_data));
          float v668_data = s0[110];
          float v670_data = ir1[6];
          ir1[6] = (v670_data + (v637_data * v668_data));
          float v673_data = s0[126];
          float v675_data = ir1[7];
          ir1[7] = (v675_data + (v637_data * v673_data));
          float v680_data = r0[15];
          float v681_data = s0[15];
          float v683_data = ir1[0];
          ir1[0] = (v683_data + (v680_data * v681_data));
          float v686_data = s0[31];
          float v688_data = ir1[1];
          ir1[1] = (v688_data + (v680_data * v686_data));
          float v691_data = s0[47];
          float v693_data = ir1[2];
          ir1[2] = (v693_data + (v680_data * v691_data));
          float v696_data = s0[63];
          float v698_data = ir1[3];
          ir1[3] = (v698_data + (v680_data * v696_data));
          float v701_data = s0[79];
          float v703_data = ir1[4];
          ir1[4] = (v703_data + (v680_data * v701_data));
          float v706_data = s0[95];
          float v708_data = ir1[5];
          ir1[5] = (v708_data + (v680_data * v706_data));
          float v711_data = s0[111];
          float v713_data = ir1[6];
          ir1[6] = (v713_data + (v680_data * v711_data));
          float v716_data = s0[127];
          float v718_data = ir1[7];
          ir1[7] = (v718_data + (v680_data * v716_data));
          #pragma unroll
          for (int32_t v723_n0 = 0; v723_n0 < 1; ++v723_n0) {
            #pragma unroll
            for (int32_t v724_n1 = 0; v724_n1 < 8; ++v724_n1) {
              int32_t v725_a = v723_n0 + v724_n1;
              int32_t v726_a = v723_n0 + v724_n1;
              float v727_data = ir1[v726_a];
              r1[v726_a] = v727_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v732_i0 = 0; v732_i0 < 1; ++v732_i0) {
            int32_t v741_lead = v6_lead + (v732_i0 * 16);
            #pragma unroll
            for (int32_t v733_i1 = 0; v733_i1 < 8; ++v733_i1) {
              int32_t v734_a = v732_i0 + v733_i1;
              float v736_data = r1[(v732_i0 + v733_i1)];
              glb_m0[(v741_lead + (v733_i1 * 16))] = v736_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

