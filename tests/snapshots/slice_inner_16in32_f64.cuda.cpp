// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d37ccf0b0, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_3d37ccf0b0, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3d37ccf0b0<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          alignas(16) double r0[16]{};
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
              double v25_data = __ldcg(&glb_m1[(v22_off + v15_a)]);
              int32_t v27_a = v7_i0 + (v8_i1 - 8);
              r0[v27_a] = v25_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) double r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double ir1[8]{};
          double v35_data = r0[0];
          double v36_data = s0[0];
          double v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          double v41_data = s0[16];
          double v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          double v46_data = s0[32];
          double v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          double v51_data = s0[48];
          double v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          double v56_data = s0[64];
          double v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          double v61_data = s0[80];
          double v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          double v66_data = s0[96];
          double v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          double v71_data = s0[112];
          double v73_data = ir1[7];
          ir1[7] = (v73_data + (v35_data * v71_data));
          double v78_data = r0[1];
          double v79_data = s0[1];
          double v81_data = ir1[0];
          ir1[0] = (v81_data + (v78_data * v79_data));
          double v84_data = s0[17];
          double v86_data = ir1[1];
          ir1[1] = (v86_data + (v78_data * v84_data));
          double v89_data = s0[33];
          double v91_data = ir1[2];
          ir1[2] = (v91_data + (v78_data * v89_data));
          double v94_data = s0[49];
          double v96_data = ir1[3];
          ir1[3] = (v96_data + (v78_data * v94_data));
          double v99_data = s0[65];
          double v101_data = ir1[4];
          ir1[4] = (v101_data + (v78_data * v99_data));
          double v104_data = s0[81];
          double v106_data = ir1[5];
          ir1[5] = (v106_data + (v78_data * v104_data));
          double v109_data = s0[97];
          double v111_data = ir1[6];
          ir1[6] = (v111_data + (v78_data * v109_data));
          double v114_data = s0[113];
          double v116_data = ir1[7];
          ir1[7] = (v116_data + (v78_data * v114_data));
          double v121_data = r0[2];
          double v122_data = s0[2];
          double v124_data = ir1[0];
          ir1[0] = (v124_data + (v121_data * v122_data));
          double v127_data = s0[18];
          double v129_data = ir1[1];
          ir1[1] = (v129_data + (v121_data * v127_data));
          double v132_data = s0[34];
          double v134_data = ir1[2];
          ir1[2] = (v134_data + (v121_data * v132_data));
          double v137_data = s0[50];
          double v139_data = ir1[3];
          ir1[3] = (v139_data + (v121_data * v137_data));
          double v142_data = s0[66];
          double v144_data = ir1[4];
          ir1[4] = (v144_data + (v121_data * v142_data));
          double v147_data = s0[82];
          double v149_data = ir1[5];
          ir1[5] = (v149_data + (v121_data * v147_data));
          double v152_data = s0[98];
          double v154_data = ir1[6];
          ir1[6] = (v154_data + (v121_data * v152_data));
          double v157_data = s0[114];
          double v159_data = ir1[7];
          ir1[7] = (v159_data + (v121_data * v157_data));
          double v164_data = r0[3];
          double v165_data = s0[3];
          double v167_data = ir1[0];
          ir1[0] = (v167_data + (v164_data * v165_data));
          double v170_data = s0[19];
          double v172_data = ir1[1];
          ir1[1] = (v172_data + (v164_data * v170_data));
          double v175_data = s0[35];
          double v177_data = ir1[2];
          ir1[2] = (v177_data + (v164_data * v175_data));
          double v180_data = s0[51];
          double v182_data = ir1[3];
          ir1[3] = (v182_data + (v164_data * v180_data));
          double v185_data = s0[67];
          double v187_data = ir1[4];
          ir1[4] = (v187_data + (v164_data * v185_data));
          double v190_data = s0[83];
          double v192_data = ir1[5];
          ir1[5] = (v192_data + (v164_data * v190_data));
          double v195_data = s0[99];
          double v197_data = ir1[6];
          ir1[6] = (v197_data + (v164_data * v195_data));
          double v200_data = s0[115];
          double v202_data = ir1[7];
          ir1[7] = (v202_data + (v164_data * v200_data));
          double v207_data = r0[4];
          double v208_data = s0[4];
          double v210_data = ir1[0];
          ir1[0] = (v210_data + (v207_data * v208_data));
          double v213_data = s0[20];
          double v215_data = ir1[1];
          ir1[1] = (v215_data + (v207_data * v213_data));
          double v218_data = s0[36];
          double v220_data = ir1[2];
          ir1[2] = (v220_data + (v207_data * v218_data));
          double v223_data = s0[52];
          double v225_data = ir1[3];
          ir1[3] = (v225_data + (v207_data * v223_data));
          double v228_data = s0[68];
          double v230_data = ir1[4];
          ir1[4] = (v230_data + (v207_data * v228_data));
          double v233_data = s0[84];
          double v235_data = ir1[5];
          ir1[5] = (v235_data + (v207_data * v233_data));
          double v238_data = s0[100];
          double v240_data = ir1[6];
          ir1[6] = (v240_data + (v207_data * v238_data));
          double v243_data = s0[116];
          double v245_data = ir1[7];
          ir1[7] = (v245_data + (v207_data * v243_data));
          double v250_data = r0[5];
          double v251_data = s0[5];
          double v253_data = ir1[0];
          ir1[0] = (v253_data + (v250_data * v251_data));
          double v256_data = s0[21];
          double v258_data = ir1[1];
          ir1[1] = (v258_data + (v250_data * v256_data));
          double v261_data = s0[37];
          double v263_data = ir1[2];
          ir1[2] = (v263_data + (v250_data * v261_data));
          double v266_data = s0[53];
          double v268_data = ir1[3];
          ir1[3] = (v268_data + (v250_data * v266_data));
          double v271_data = s0[69];
          double v273_data = ir1[4];
          ir1[4] = (v273_data + (v250_data * v271_data));
          double v276_data = s0[85];
          double v278_data = ir1[5];
          ir1[5] = (v278_data + (v250_data * v276_data));
          double v281_data = s0[101];
          double v283_data = ir1[6];
          ir1[6] = (v283_data + (v250_data * v281_data));
          double v286_data = s0[117];
          double v288_data = ir1[7];
          ir1[7] = (v288_data + (v250_data * v286_data));
          double v293_data = r0[6];
          double v294_data = s0[6];
          double v296_data = ir1[0];
          ir1[0] = (v296_data + (v293_data * v294_data));
          double v299_data = s0[22];
          double v301_data = ir1[1];
          ir1[1] = (v301_data + (v293_data * v299_data));
          double v304_data = s0[38];
          double v306_data = ir1[2];
          ir1[2] = (v306_data + (v293_data * v304_data));
          double v309_data = s0[54];
          double v311_data = ir1[3];
          ir1[3] = (v311_data + (v293_data * v309_data));
          double v314_data = s0[70];
          double v316_data = ir1[4];
          ir1[4] = (v316_data + (v293_data * v314_data));
          double v319_data = s0[86];
          double v321_data = ir1[5];
          ir1[5] = (v321_data + (v293_data * v319_data));
          double v324_data = s0[102];
          double v326_data = ir1[6];
          ir1[6] = (v326_data + (v293_data * v324_data));
          double v329_data = s0[118];
          double v331_data = ir1[7];
          ir1[7] = (v331_data + (v293_data * v329_data));
          double v336_data = r0[7];
          double v337_data = s0[7];
          double v339_data = ir1[0];
          ir1[0] = (v339_data + (v336_data * v337_data));
          double v342_data = s0[23];
          double v344_data = ir1[1];
          ir1[1] = (v344_data + (v336_data * v342_data));
          double v347_data = s0[39];
          double v349_data = ir1[2];
          ir1[2] = (v349_data + (v336_data * v347_data));
          double v352_data = s0[55];
          double v354_data = ir1[3];
          ir1[3] = (v354_data + (v336_data * v352_data));
          double v357_data = s0[71];
          double v359_data = ir1[4];
          ir1[4] = (v359_data + (v336_data * v357_data));
          double v362_data = s0[87];
          double v364_data = ir1[5];
          ir1[5] = (v364_data + (v336_data * v362_data));
          double v367_data = s0[103];
          double v369_data = ir1[6];
          ir1[6] = (v369_data + (v336_data * v367_data));
          double v372_data = s0[119];
          double v374_data = ir1[7];
          ir1[7] = (v374_data + (v336_data * v372_data));
          double v379_data = r0[8];
          double v380_data = s0[8];
          double v382_data = ir1[0];
          ir1[0] = (v382_data + (v379_data * v380_data));
          double v385_data = s0[24];
          double v387_data = ir1[1];
          ir1[1] = (v387_data + (v379_data * v385_data));
          double v390_data = s0[40];
          double v392_data = ir1[2];
          ir1[2] = (v392_data + (v379_data * v390_data));
          double v395_data = s0[56];
          double v397_data = ir1[3];
          ir1[3] = (v397_data + (v379_data * v395_data));
          double v400_data = s0[72];
          double v402_data = ir1[4];
          ir1[4] = (v402_data + (v379_data * v400_data));
          double v405_data = s0[88];
          double v407_data = ir1[5];
          ir1[5] = (v407_data + (v379_data * v405_data));
          double v410_data = s0[104];
          double v412_data = ir1[6];
          ir1[6] = (v412_data + (v379_data * v410_data));
          double v415_data = s0[120];
          double v417_data = ir1[7];
          ir1[7] = (v417_data + (v379_data * v415_data));
          double v422_data = r0[9];
          double v423_data = s0[9];
          double v425_data = ir1[0];
          ir1[0] = (v425_data + (v422_data * v423_data));
          double v428_data = s0[25];
          double v430_data = ir1[1];
          ir1[1] = (v430_data + (v422_data * v428_data));
          double v433_data = s0[41];
          double v435_data = ir1[2];
          ir1[2] = (v435_data + (v422_data * v433_data));
          double v438_data = s0[57];
          double v440_data = ir1[3];
          ir1[3] = (v440_data + (v422_data * v438_data));
          double v443_data = s0[73];
          double v445_data = ir1[4];
          ir1[4] = (v445_data + (v422_data * v443_data));
          double v448_data = s0[89];
          double v450_data = ir1[5];
          ir1[5] = (v450_data + (v422_data * v448_data));
          double v453_data = s0[105];
          double v455_data = ir1[6];
          ir1[6] = (v455_data + (v422_data * v453_data));
          double v458_data = s0[121];
          double v460_data = ir1[7];
          ir1[7] = (v460_data + (v422_data * v458_data));
          double v465_data = r0[10];
          double v466_data = s0[10];
          double v468_data = ir1[0];
          ir1[0] = (v468_data + (v465_data * v466_data));
          double v471_data = s0[26];
          double v473_data = ir1[1];
          ir1[1] = (v473_data + (v465_data * v471_data));
          double v476_data = s0[42];
          double v478_data = ir1[2];
          ir1[2] = (v478_data + (v465_data * v476_data));
          double v481_data = s0[58];
          double v483_data = ir1[3];
          ir1[3] = (v483_data + (v465_data * v481_data));
          double v486_data = s0[74];
          double v488_data = ir1[4];
          ir1[4] = (v488_data + (v465_data * v486_data));
          double v491_data = s0[90];
          double v493_data = ir1[5];
          ir1[5] = (v493_data + (v465_data * v491_data));
          double v496_data = s0[106];
          double v498_data = ir1[6];
          ir1[6] = (v498_data + (v465_data * v496_data));
          double v501_data = s0[122];
          double v503_data = ir1[7];
          ir1[7] = (v503_data + (v465_data * v501_data));
          double v508_data = r0[11];
          double v509_data = s0[11];
          double v511_data = ir1[0];
          ir1[0] = (v511_data + (v508_data * v509_data));
          double v514_data = s0[27];
          double v516_data = ir1[1];
          ir1[1] = (v516_data + (v508_data * v514_data));
          double v519_data = s0[43];
          double v521_data = ir1[2];
          ir1[2] = (v521_data + (v508_data * v519_data));
          double v524_data = s0[59];
          double v526_data = ir1[3];
          ir1[3] = (v526_data + (v508_data * v524_data));
          double v529_data = s0[75];
          double v531_data = ir1[4];
          ir1[4] = (v531_data + (v508_data * v529_data));
          double v534_data = s0[91];
          double v536_data = ir1[5];
          ir1[5] = (v536_data + (v508_data * v534_data));
          double v539_data = s0[107];
          double v541_data = ir1[6];
          ir1[6] = (v541_data + (v508_data * v539_data));
          double v544_data = s0[123];
          double v546_data = ir1[7];
          ir1[7] = (v546_data + (v508_data * v544_data));
          double v551_data = r0[12];
          double v552_data = s0[12];
          double v554_data = ir1[0];
          ir1[0] = (v554_data + (v551_data * v552_data));
          double v557_data = s0[28];
          double v559_data = ir1[1];
          ir1[1] = (v559_data + (v551_data * v557_data));
          double v562_data = s0[44];
          double v564_data = ir1[2];
          ir1[2] = (v564_data + (v551_data * v562_data));
          double v567_data = s0[60];
          double v569_data = ir1[3];
          ir1[3] = (v569_data + (v551_data * v567_data));
          double v572_data = s0[76];
          double v574_data = ir1[4];
          ir1[4] = (v574_data + (v551_data * v572_data));
          double v577_data = s0[92];
          double v579_data = ir1[5];
          ir1[5] = (v579_data + (v551_data * v577_data));
          double v582_data = s0[108];
          double v584_data = ir1[6];
          ir1[6] = (v584_data + (v551_data * v582_data));
          double v587_data = s0[124];
          double v589_data = ir1[7];
          ir1[7] = (v589_data + (v551_data * v587_data));
          double v594_data = r0[13];
          double v595_data = s0[13];
          double v597_data = ir1[0];
          ir1[0] = (v597_data + (v594_data * v595_data));
          double v600_data = s0[29];
          double v602_data = ir1[1];
          ir1[1] = (v602_data + (v594_data * v600_data));
          double v605_data = s0[45];
          double v607_data = ir1[2];
          ir1[2] = (v607_data + (v594_data * v605_data));
          double v610_data = s0[61];
          double v612_data = ir1[3];
          ir1[3] = (v612_data + (v594_data * v610_data));
          double v615_data = s0[77];
          double v617_data = ir1[4];
          ir1[4] = (v617_data + (v594_data * v615_data));
          double v620_data = s0[93];
          double v622_data = ir1[5];
          ir1[5] = (v622_data + (v594_data * v620_data));
          double v625_data = s0[109];
          double v627_data = ir1[6];
          ir1[6] = (v627_data + (v594_data * v625_data));
          double v630_data = s0[125];
          double v632_data = ir1[7];
          ir1[7] = (v632_data + (v594_data * v630_data));
          double v637_data = r0[14];
          double v638_data = s0[14];
          double v640_data = ir1[0];
          ir1[0] = (v640_data + (v637_data * v638_data));
          double v643_data = s0[30];
          double v645_data = ir1[1];
          ir1[1] = (v645_data + (v637_data * v643_data));
          double v648_data = s0[46];
          double v650_data = ir1[2];
          ir1[2] = (v650_data + (v637_data * v648_data));
          double v653_data = s0[62];
          double v655_data = ir1[3];
          ir1[3] = (v655_data + (v637_data * v653_data));
          double v658_data = s0[78];
          double v660_data = ir1[4];
          ir1[4] = (v660_data + (v637_data * v658_data));
          double v663_data = s0[94];
          double v665_data = ir1[5];
          ir1[5] = (v665_data + (v637_data * v663_data));
          double v668_data = s0[110];
          double v670_data = ir1[6];
          ir1[6] = (v670_data + (v637_data * v668_data));
          double v673_data = s0[126];
          double v675_data = ir1[7];
          ir1[7] = (v675_data + (v637_data * v673_data));
          double v680_data = r0[15];
          double v681_data = s0[15];
          double v683_data = ir1[0];
          ir1[0] = (v683_data + (v680_data * v681_data));
          double v686_data = s0[31];
          double v688_data = ir1[1];
          ir1[1] = (v688_data + (v680_data * v686_data));
          double v691_data = s0[47];
          double v693_data = ir1[2];
          ir1[2] = (v693_data + (v680_data * v691_data));
          double v696_data = s0[63];
          double v698_data = ir1[3];
          ir1[3] = (v698_data + (v680_data * v696_data));
          double v701_data = s0[79];
          double v703_data = ir1[4];
          ir1[4] = (v703_data + (v680_data * v701_data));
          double v706_data = s0[95];
          double v708_data = ir1[5];
          ir1[5] = (v708_data + (v680_data * v706_data));
          double v711_data = s0[111];
          double v713_data = ir1[6];
          ir1[6] = (v713_data + (v680_data * v711_data));
          double v716_data = s0[127];
          double v718_data = ir1[7];
          ir1[7] = (v718_data + (v680_data * v716_data));
          #pragma unroll
          for (int32_t v723_n0 = 0; v723_n0 < 1; ++v723_n0) {
            #pragma unroll
            for (int32_t v724_n1 = 0; v724_n1 < 8; ++v724_n1) {
              int32_t v725_a = v723_n0 + v724_n1;
              int32_t v726_a = v723_n0 + v724_n1;
              double v727_data = ir1[v726_a];
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
              double v736_data = r1[(v732_i0 + v733_i1)];
              glb_m0[(v741_lead + (v733_i1 * 16))] = v736_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

