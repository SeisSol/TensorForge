// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c847f49d, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_16c847f49d, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16c847f49d<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v14_a = v8_i1 * 12;
              int32_t v15_a = v6_lead + v14_a;
              double v23_data = __ldcg(&glb_m1[(v6_lead + v14_a)]);
              int32_t v24_a = 0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<8>(8), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
              int32_t v37_a = v31_i1 * 12;
              int32_t v38_a = v6_lead + v37_a;
              double v46_data = glb_m0[(v6_lead + v37_a)];
              int32_t v47_a = 0 + v31_i1;
              r1[v47_a] = v46_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v6_lead < 12) {
            double v54_data = r0[0];
            double v55_data = s0[0];
            double v57_data = ir2[0];
            ir2[0] = (v57_data + (v54_data * v55_data));
            double v60_data = s0[16];
            double v62_data = ir2[1];
            ir2[1] = (v62_data + (v54_data * v60_data));
            double v65_data = s0[32];
            double v67_data = ir2[2];
            ir2[2] = (v67_data + (v54_data * v65_data));
            double v70_data = s0[48];
            double v72_data = ir2[3];
            ir2[3] = (v72_data + (v54_data * v70_data));
            double v75_data = s0[64];
            double v77_data = ir2[4];
            ir2[4] = (v77_data + (v54_data * v75_data));
            double v80_data = s0[80];
            double v82_data = ir2[5];
            ir2[5] = (v82_data + (v54_data * v80_data));
            double v85_data = s0[96];
            double v87_data = ir2[6];
            ir2[6] = (v87_data + (v54_data * v85_data));
            double v90_data = s0[112];
            double v92_data = ir2[7];
            ir2[7] = (v92_data + (v54_data * v90_data));
          }
          if (v6_lead < 12) {
            double v98_data = r0[1];
            double v99_data = s0[1];
            double v101_data = ir2[0];
            ir2[0] = (v101_data + (v98_data * v99_data));
            double v104_data = s0[17];
            double v106_data = ir2[1];
            ir2[1] = (v106_data + (v98_data * v104_data));
            double v109_data = s0[33];
            double v111_data = ir2[2];
            ir2[2] = (v111_data + (v98_data * v109_data));
            double v114_data = s0[49];
            double v116_data = ir2[3];
            ir2[3] = (v116_data + (v98_data * v114_data));
            double v119_data = s0[65];
            double v121_data = ir2[4];
            ir2[4] = (v121_data + (v98_data * v119_data));
            double v124_data = s0[81];
            double v126_data = ir2[5];
            ir2[5] = (v126_data + (v98_data * v124_data));
            double v129_data = s0[97];
            double v131_data = ir2[6];
            ir2[6] = (v131_data + (v98_data * v129_data));
            double v134_data = s0[113];
            double v136_data = ir2[7];
            ir2[7] = (v136_data + (v98_data * v134_data));
          }
          if (v6_lead < 12) {
            double v142_data = r0[2];
            double v143_data = s0[2];
            double v145_data = ir2[0];
            ir2[0] = (v145_data + (v142_data * v143_data));
            double v148_data = s0[18];
            double v150_data = ir2[1];
            ir2[1] = (v150_data + (v142_data * v148_data));
            double v153_data = s0[34];
            double v155_data = ir2[2];
            ir2[2] = (v155_data + (v142_data * v153_data));
            double v158_data = s0[50];
            double v160_data = ir2[3];
            ir2[3] = (v160_data + (v142_data * v158_data));
            double v163_data = s0[66];
            double v165_data = ir2[4];
            ir2[4] = (v165_data + (v142_data * v163_data));
            double v168_data = s0[82];
            double v170_data = ir2[5];
            ir2[5] = (v170_data + (v142_data * v168_data));
            double v173_data = s0[98];
            double v175_data = ir2[6];
            ir2[6] = (v175_data + (v142_data * v173_data));
            double v178_data = s0[114];
            double v180_data = ir2[7];
            ir2[7] = (v180_data + (v142_data * v178_data));
          }
          if (v6_lead < 12) {
            double v186_data = r0[3];
            double v187_data = s0[3];
            double v189_data = ir2[0];
            ir2[0] = (v189_data + (v186_data * v187_data));
            double v192_data = s0[19];
            double v194_data = ir2[1];
            ir2[1] = (v194_data + (v186_data * v192_data));
            double v197_data = s0[35];
            double v199_data = ir2[2];
            ir2[2] = (v199_data + (v186_data * v197_data));
            double v202_data = s0[51];
            double v204_data = ir2[3];
            ir2[3] = (v204_data + (v186_data * v202_data));
            double v207_data = s0[67];
            double v209_data = ir2[4];
            ir2[4] = (v209_data + (v186_data * v207_data));
            double v212_data = s0[83];
            double v214_data = ir2[5];
            ir2[5] = (v214_data + (v186_data * v212_data));
            double v217_data = s0[99];
            double v219_data = ir2[6];
            ir2[6] = (v219_data + (v186_data * v217_data));
            double v222_data = s0[115];
            double v224_data = ir2[7];
            ir2[7] = (v224_data + (v186_data * v222_data));
          }
          if (v6_lead < 12) {
            double v230_data = r0[4];
            double v231_data = s0[4];
            double v233_data = ir2[0];
            ir2[0] = (v233_data + (v230_data * v231_data));
            double v236_data = s0[20];
            double v238_data = ir2[1];
            ir2[1] = (v238_data + (v230_data * v236_data));
            double v241_data = s0[36];
            double v243_data = ir2[2];
            ir2[2] = (v243_data + (v230_data * v241_data));
            double v246_data = s0[52];
            double v248_data = ir2[3];
            ir2[3] = (v248_data + (v230_data * v246_data));
            double v251_data = s0[68];
            double v253_data = ir2[4];
            ir2[4] = (v253_data + (v230_data * v251_data));
            double v256_data = s0[84];
            double v258_data = ir2[5];
            ir2[5] = (v258_data + (v230_data * v256_data));
            double v261_data = s0[100];
            double v263_data = ir2[6];
            ir2[6] = (v263_data + (v230_data * v261_data));
            double v266_data = s0[116];
            double v268_data = ir2[7];
            ir2[7] = (v268_data + (v230_data * v266_data));
          }
          if (v6_lead < 12) {
            double v274_data = r0[5];
            double v275_data = s0[5];
            double v277_data = ir2[0];
            ir2[0] = (v277_data + (v274_data * v275_data));
            double v280_data = s0[21];
            double v282_data = ir2[1];
            ir2[1] = (v282_data + (v274_data * v280_data));
            double v285_data = s0[37];
            double v287_data = ir2[2];
            ir2[2] = (v287_data + (v274_data * v285_data));
            double v290_data = s0[53];
            double v292_data = ir2[3];
            ir2[3] = (v292_data + (v274_data * v290_data));
            double v295_data = s0[69];
            double v297_data = ir2[4];
            ir2[4] = (v297_data + (v274_data * v295_data));
            double v300_data = s0[85];
            double v302_data = ir2[5];
            ir2[5] = (v302_data + (v274_data * v300_data));
            double v305_data = s0[101];
            double v307_data = ir2[6];
            ir2[6] = (v307_data + (v274_data * v305_data));
            double v310_data = s0[117];
            double v312_data = ir2[7];
            ir2[7] = (v312_data + (v274_data * v310_data));
          }
          if (v6_lead < 12) {
            double v318_data = r0[6];
            double v319_data = s0[6];
            double v321_data = ir2[0];
            ir2[0] = (v321_data + (v318_data * v319_data));
            double v324_data = s0[22];
            double v326_data = ir2[1];
            ir2[1] = (v326_data + (v318_data * v324_data));
            double v329_data = s0[38];
            double v331_data = ir2[2];
            ir2[2] = (v331_data + (v318_data * v329_data));
            double v334_data = s0[54];
            double v336_data = ir2[3];
            ir2[3] = (v336_data + (v318_data * v334_data));
            double v339_data = s0[70];
            double v341_data = ir2[4];
            ir2[4] = (v341_data + (v318_data * v339_data));
            double v344_data = s0[86];
            double v346_data = ir2[5];
            ir2[5] = (v346_data + (v318_data * v344_data));
            double v349_data = s0[102];
            double v351_data = ir2[6];
            ir2[6] = (v351_data + (v318_data * v349_data));
            double v354_data = s0[118];
            double v356_data = ir2[7];
            ir2[7] = (v356_data + (v318_data * v354_data));
          }
          if (v6_lead < 12) {
            double v362_data = r0[7];
            double v363_data = s0[7];
            double v365_data = ir2[0];
            ir2[0] = (v365_data + (v362_data * v363_data));
            double v368_data = s0[23];
            double v370_data = ir2[1];
            ir2[1] = (v370_data + (v362_data * v368_data));
            double v373_data = s0[39];
            double v375_data = ir2[2];
            ir2[2] = (v375_data + (v362_data * v373_data));
            double v378_data = s0[55];
            double v380_data = ir2[3];
            ir2[3] = (v380_data + (v362_data * v378_data));
            double v383_data = s0[71];
            double v385_data = ir2[4];
            ir2[4] = (v385_data + (v362_data * v383_data));
            double v388_data = s0[87];
            double v390_data = ir2[5];
            ir2[5] = (v390_data + (v362_data * v388_data));
            double v393_data = s0[103];
            double v395_data = ir2[6];
            ir2[6] = (v395_data + (v362_data * v393_data));
            double v398_data = s0[119];
            double v400_data = ir2[7];
            ir2[7] = (v400_data + (v362_data * v398_data));
          }
          if (v6_lead < 12) {
            double v406_data = r0[8];
            double v407_data = s0[8];
            double v409_data = ir2[0];
            ir2[0] = (v409_data + (v406_data * v407_data));
            double v412_data = s0[24];
            double v414_data = ir2[1];
            ir2[1] = (v414_data + (v406_data * v412_data));
            double v417_data = s0[40];
            double v419_data = ir2[2];
            ir2[2] = (v419_data + (v406_data * v417_data));
            double v422_data = s0[56];
            double v424_data = ir2[3];
            ir2[3] = (v424_data + (v406_data * v422_data));
            double v427_data = s0[72];
            double v429_data = ir2[4];
            ir2[4] = (v429_data + (v406_data * v427_data));
            double v432_data = s0[88];
            double v434_data = ir2[5];
            ir2[5] = (v434_data + (v406_data * v432_data));
            double v437_data = s0[104];
            double v439_data = ir2[6];
            ir2[6] = (v439_data + (v406_data * v437_data));
            double v442_data = s0[120];
            double v444_data = ir2[7];
            ir2[7] = (v444_data + (v406_data * v442_data));
          }
          if (v6_lead < 12) {
            double v450_data = r0[9];
            double v451_data = s0[9];
            double v453_data = ir2[0];
            ir2[0] = (v453_data + (v450_data * v451_data));
            double v456_data = s0[25];
            double v458_data = ir2[1];
            ir2[1] = (v458_data + (v450_data * v456_data));
            double v461_data = s0[41];
            double v463_data = ir2[2];
            ir2[2] = (v463_data + (v450_data * v461_data));
            double v466_data = s0[57];
            double v468_data = ir2[3];
            ir2[3] = (v468_data + (v450_data * v466_data));
            double v471_data = s0[73];
            double v473_data = ir2[4];
            ir2[4] = (v473_data + (v450_data * v471_data));
            double v476_data = s0[89];
            double v478_data = ir2[5];
            ir2[5] = (v478_data + (v450_data * v476_data));
            double v481_data = s0[105];
            double v483_data = ir2[6];
            ir2[6] = (v483_data + (v450_data * v481_data));
            double v486_data = s0[121];
            double v488_data = ir2[7];
            ir2[7] = (v488_data + (v450_data * v486_data));
          }
          if (v6_lead < 12) {
            double v494_data = r0[10];
            double v495_data = s0[10];
            double v497_data = ir2[0];
            ir2[0] = (v497_data + (v494_data * v495_data));
            double v500_data = s0[26];
            double v502_data = ir2[1];
            ir2[1] = (v502_data + (v494_data * v500_data));
            double v505_data = s0[42];
            double v507_data = ir2[2];
            ir2[2] = (v507_data + (v494_data * v505_data));
            double v510_data = s0[58];
            double v512_data = ir2[3];
            ir2[3] = (v512_data + (v494_data * v510_data));
            double v515_data = s0[74];
            double v517_data = ir2[4];
            ir2[4] = (v517_data + (v494_data * v515_data));
            double v520_data = s0[90];
            double v522_data = ir2[5];
            ir2[5] = (v522_data + (v494_data * v520_data));
            double v525_data = s0[106];
            double v527_data = ir2[6];
            ir2[6] = (v527_data + (v494_data * v525_data));
            double v530_data = s0[122];
            double v532_data = ir2[7];
            ir2[7] = (v532_data + (v494_data * v530_data));
          }
          if (v6_lead < 12) {
            double v538_data = r0[11];
            double v539_data = s0[11];
            double v541_data = ir2[0];
            ir2[0] = (v541_data + (v538_data * v539_data));
            double v544_data = s0[27];
            double v546_data = ir2[1];
            ir2[1] = (v546_data + (v538_data * v544_data));
            double v549_data = s0[43];
            double v551_data = ir2[2];
            ir2[2] = (v551_data + (v538_data * v549_data));
            double v554_data = s0[59];
            double v556_data = ir2[3];
            ir2[3] = (v556_data + (v538_data * v554_data));
            double v559_data = s0[75];
            double v561_data = ir2[4];
            ir2[4] = (v561_data + (v538_data * v559_data));
            double v564_data = s0[91];
            double v566_data = ir2[5];
            ir2[5] = (v566_data + (v538_data * v564_data));
            double v569_data = s0[107];
            double v571_data = ir2[6];
            ir2[6] = (v571_data + (v538_data * v569_data));
            double v574_data = s0[123];
            double v576_data = ir2[7];
            ir2[7] = (v576_data + (v538_data * v574_data));
          }
          if (v6_lead < 12) {
            double v582_data = r0[12];
            double v583_data = s0[12];
            double v585_data = ir2[0];
            ir2[0] = (v585_data + (v582_data * v583_data));
            double v588_data = s0[28];
            double v590_data = ir2[1];
            ir2[1] = (v590_data + (v582_data * v588_data));
            double v593_data = s0[44];
            double v595_data = ir2[2];
            ir2[2] = (v595_data + (v582_data * v593_data));
            double v598_data = s0[60];
            double v600_data = ir2[3];
            ir2[3] = (v600_data + (v582_data * v598_data));
            double v603_data = s0[76];
            double v605_data = ir2[4];
            ir2[4] = (v605_data + (v582_data * v603_data));
            double v608_data = s0[92];
            double v610_data = ir2[5];
            ir2[5] = (v610_data + (v582_data * v608_data));
            double v613_data = s0[108];
            double v615_data = ir2[6];
            ir2[6] = (v615_data + (v582_data * v613_data));
            double v618_data = s0[124];
            double v620_data = ir2[7];
            ir2[7] = (v620_data + (v582_data * v618_data));
          }
          if (v6_lead < 12) {
            double v626_data = r0[13];
            double v627_data = s0[13];
            double v629_data = ir2[0];
            ir2[0] = (v629_data + (v626_data * v627_data));
            double v632_data = s0[29];
            double v634_data = ir2[1];
            ir2[1] = (v634_data + (v626_data * v632_data));
            double v637_data = s0[45];
            double v639_data = ir2[2];
            ir2[2] = (v639_data + (v626_data * v637_data));
            double v642_data = s0[61];
            double v644_data = ir2[3];
            ir2[3] = (v644_data + (v626_data * v642_data));
            double v647_data = s0[77];
            double v649_data = ir2[4];
            ir2[4] = (v649_data + (v626_data * v647_data));
            double v652_data = s0[93];
            double v654_data = ir2[5];
            ir2[5] = (v654_data + (v626_data * v652_data));
            double v657_data = s0[109];
            double v659_data = ir2[6];
            ir2[6] = (v659_data + (v626_data * v657_data));
            double v662_data = s0[125];
            double v664_data = ir2[7];
            ir2[7] = (v664_data + (v626_data * v662_data));
          }
          if (v6_lead < 12) {
            double v670_data = r0[14];
            double v671_data = s0[14];
            double v673_data = ir2[0];
            ir2[0] = (v673_data + (v670_data * v671_data));
            double v676_data = s0[30];
            double v678_data = ir2[1];
            ir2[1] = (v678_data + (v670_data * v676_data));
            double v681_data = s0[46];
            double v683_data = ir2[2];
            ir2[2] = (v683_data + (v670_data * v681_data));
            double v686_data = s0[62];
            double v688_data = ir2[3];
            ir2[3] = (v688_data + (v670_data * v686_data));
            double v691_data = s0[78];
            double v693_data = ir2[4];
            ir2[4] = (v693_data + (v670_data * v691_data));
            double v696_data = s0[94];
            double v698_data = ir2[5];
            ir2[5] = (v698_data + (v670_data * v696_data));
            double v701_data = s0[110];
            double v703_data = ir2[6];
            ir2[6] = (v703_data + (v670_data * v701_data));
            double v706_data = s0[126];
            double v708_data = ir2[7];
            ir2[7] = (v708_data + (v670_data * v706_data));
          }
          if (v6_lead < 12) {
            double v714_data = r0[15];
            double v715_data = s0[15];
            double v717_data = ir2[0];
            ir2[0] = (v717_data + (v714_data * v715_data));
            double v720_data = s0[31];
            double v722_data = ir2[1];
            ir2[1] = (v722_data + (v714_data * v720_data));
            double v725_data = s0[47];
            double v727_data = ir2[2];
            ir2[2] = (v727_data + (v714_data * v725_data));
            double v730_data = s0[63];
            double v732_data = ir2[3];
            ir2[3] = (v732_data + (v714_data * v730_data));
            double v735_data = s0[79];
            double v737_data = ir2[4];
            ir2[4] = (v737_data + (v714_data * v735_data));
            double v740_data = s0[95];
            double v742_data = ir2[5];
            ir2[5] = (v742_data + (v714_data * v740_data));
            double v745_data = s0[111];
            double v747_data = ir2[6];
            ir2[6] = (v747_data + (v714_data * v745_data));
            double v750_data = s0[127];
            double v752_data = ir2[7];
            ir2[7] = (v752_data + (v714_data * v750_data));
          }
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v758_n1 = 0; v758_n1 < 8; ++v758_n1) {
              int32_t v759_a = 0 + v758_n1;
              double v761_data = ir2[v758_n1];
              int32_t v762_a = 0 + v758_n1;
              double v764_data = r1[v758_n1];
              int32_t v766_a = 0 + v758_n1;
              r2[v758_n1] = (v764_data + v761_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v772_i1 = 0; v772_i1 < 8; ++v772_i1) {
              int32_t v773_a = 0 + v772_i1;
              double v775_data = r2[v772_i1];
              int32_t v782_a = v6_lead + (v772_i1 * 12);
              glb_m0[v782_a] = v775_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

