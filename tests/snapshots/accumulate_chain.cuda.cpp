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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
          int32_t v19_lead = threadIdx.x % 16;
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
              float v29_data = __ldcg(&glb_m1[(v19_lead + (v21_i1 * 12))]);
              r0[v21_i1] = v29_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              float v46_data = __ldcg(&glb_m3[(v19_lead + (v38_i1 * 12))]);
              r2[v38_i1] = v46_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v19_lead < 12) {
            float v54_data = r0[0];
            float v55_data = s0[0];
            float v57_data = ir1[0];
            ir1[0] = (v57_data + (v54_data * v55_data));
            float v60_data = s0[12];
            float v62_data = ir1[1];
            ir1[1] = (v62_data + (v54_data * v60_data));
            float v65_data = s0[24];
            float v67_data = ir1[2];
            ir1[2] = (v67_data + (v54_data * v65_data));
            float v70_data = s0[37];
            float v72_data = ir1[3];
            ir1[3] = (v72_data + (v54_data * v70_data));
            float v75_data = s0[49];
            float v77_data = ir1[4];
            ir1[4] = (v77_data + (v54_data * v75_data));
            float v80_data = s0[61];
            float v82_data = ir1[5];
            ir1[5] = (v82_data + (v54_data * v80_data));
            float v85_data = s0[74];
            float v87_data = ir1[6];
            ir1[6] = (v87_data + (v54_data * v85_data));
            float v90_data = s0[86];
            float v92_data = ir1[7];
            ir1[7] = (v92_data + (v54_data * v90_data));
          }
          if (v19_lead < 12) {
            float v98_data = r0[1];
            float v99_data = s0[1];
            float v101_data = ir1[0];
            ir1[0] = (v101_data + (v98_data * v99_data));
            float v104_data = s0[13];
            float v106_data = ir1[1];
            ir1[1] = (v106_data + (v98_data * v104_data));
            float v109_data = s0[25];
            float v111_data = ir1[2];
            ir1[2] = (v111_data + (v98_data * v109_data));
            float v114_data = s0[36];
            float v116_data = ir1[3];
            ir1[3] = (v116_data + (v98_data * v114_data));
            float v119_data = s0[48];
            float v121_data = ir1[4];
            ir1[4] = (v121_data + (v98_data * v119_data));
            float v124_data = s0[60];
            float v126_data = ir1[5];
            ir1[5] = (v126_data + (v98_data * v124_data));
            float v129_data = s0[75];
            float v131_data = ir1[6];
            ir1[6] = (v131_data + (v98_data * v129_data));
            float v134_data = s0[87];
            float v136_data = ir1[7];
            ir1[7] = (v136_data + (v98_data * v134_data));
          }
          if (v19_lead < 12) {
            float v142_data = r0[2];
            float v143_data = s0[2];
            float v145_data = ir1[0];
            ir1[0] = (v145_data + (v142_data * v143_data));
            float v148_data = s0[14];
            float v150_data = ir1[1];
            ir1[1] = (v150_data + (v142_data * v148_data));
            float v153_data = s0[26];
            float v155_data = ir1[2];
            ir1[2] = (v155_data + (v142_data * v153_data));
            float v158_data = s0[39];
            float v160_data = ir1[3];
            ir1[3] = (v160_data + (v142_data * v158_data));
            float v163_data = s0[51];
            float v165_data = ir1[4];
            ir1[4] = (v165_data + (v142_data * v163_data));
            float v168_data = s0[63];
            float v170_data = ir1[5];
            ir1[5] = (v170_data + (v142_data * v168_data));
            float v173_data = s0[72];
            float v175_data = ir1[6];
            ir1[6] = (v175_data + (v142_data * v173_data));
            float v178_data = s0[84];
            float v180_data = ir1[7];
            ir1[7] = (v180_data + (v142_data * v178_data));
          }
          if (v19_lead < 12) {
            float v186_data = r0[3];
            float v187_data = s0[3];
            float v189_data = ir1[0];
            ir1[0] = (v189_data + (v186_data * v187_data));
            float v192_data = s0[15];
            float v194_data = ir1[1];
            ir1[1] = (v194_data + (v186_data * v192_data));
            float v197_data = s0[27];
            float v199_data = ir1[2];
            ir1[2] = (v199_data + (v186_data * v197_data));
            float v202_data = s0[38];
            float v204_data = ir1[3];
            ir1[3] = (v204_data + (v186_data * v202_data));
            float v207_data = s0[50];
            float v209_data = ir1[4];
            ir1[4] = (v209_data + (v186_data * v207_data));
            float v212_data = s0[62];
            float v214_data = ir1[5];
            ir1[5] = (v214_data + (v186_data * v212_data));
            float v217_data = s0[73];
            float v219_data = ir1[6];
            ir1[6] = (v219_data + (v186_data * v217_data));
            float v222_data = s0[85];
            float v224_data = ir1[7];
            ir1[7] = (v224_data + (v186_data * v222_data));
          }
          if (v19_lead < 12) {
            float v230_data = r0[4];
            float v231_data = s0[4];
            float v233_data = ir1[0];
            ir1[0] = (v233_data + (v230_data * v231_data));
            float v236_data = s0[16];
            float v238_data = ir1[1];
            ir1[1] = (v238_data + (v230_data * v236_data));
            float v241_data = s0[28];
            float v243_data = ir1[2];
            ir1[2] = (v243_data + (v230_data * v241_data));
            float v246_data = s0[41];
            float v248_data = ir1[3];
            ir1[3] = (v248_data + (v230_data * v246_data));
            float v251_data = s0[53];
            float v253_data = ir1[4];
            ir1[4] = (v253_data + (v230_data * v251_data));
            float v256_data = s0[66];
            float v258_data = ir1[5];
            ir1[5] = (v258_data + (v230_data * v256_data));
            float v261_data = s0[78];
            float v263_data = ir1[6];
            ir1[6] = (v263_data + (v230_data * v261_data));
            float v266_data = s0[90];
            float v268_data = ir1[7];
            ir1[7] = (v268_data + (v230_data * v266_data));
          }
          if (v19_lead < 12) {
            float v274_data = r0[5];
            float v275_data = s0[5];
            float v277_data = ir1[0];
            ir1[0] = (v277_data + (v274_data * v275_data));
            float v280_data = s0[17];
            float v282_data = ir1[1];
            ir1[1] = (v282_data + (v274_data * v280_data));
            float v285_data = s0[29];
            float v287_data = ir1[2];
            ir1[2] = (v287_data + (v274_data * v285_data));
            float v290_data = s0[40];
            float v292_data = ir1[3];
            ir1[3] = (v292_data + (v274_data * v290_data));
            float v295_data = s0[52];
            float v297_data = ir1[4];
            ir1[4] = (v297_data + (v274_data * v295_data));
            float v300_data = s0[67];
            float v302_data = ir1[5];
            ir1[5] = (v302_data + (v274_data * v300_data));
            float v305_data = s0[79];
            float v307_data = ir1[6];
            ir1[6] = (v307_data + (v274_data * v305_data));
            float v310_data = s0[91];
            float v312_data = ir1[7];
            ir1[7] = (v312_data + (v274_data * v310_data));
          }
          if (v19_lead < 12) {
            float v318_data = r0[6];
            float v319_data = s0[6];
            float v321_data = ir1[0];
            ir1[0] = (v321_data + (v318_data * v319_data));
            float v324_data = s0[18];
            float v326_data = ir1[1];
            ir1[1] = (v326_data + (v318_data * v324_data));
            float v329_data = s0[30];
            float v331_data = ir1[2];
            ir1[2] = (v331_data + (v318_data * v329_data));
            float v334_data = s0[43];
            float v336_data = ir1[3];
            ir1[3] = (v336_data + (v318_data * v334_data));
            float v339_data = s0[55];
            float v341_data = ir1[4];
            ir1[4] = (v341_data + (v318_data * v339_data));
            float v344_data = s0[64];
            float v346_data = ir1[5];
            ir1[5] = (v346_data + (v318_data * v344_data));
            float v349_data = s0[76];
            float v351_data = ir1[6];
            ir1[6] = (v351_data + (v318_data * v349_data));
            float v354_data = s0[88];
            float v356_data = ir1[7];
            ir1[7] = (v356_data + (v318_data * v354_data));
          }
          if (v19_lead < 12) {
            float v362_data = r0[7];
            float v363_data = s0[7];
            float v365_data = ir1[0];
            ir1[0] = (v365_data + (v362_data * v363_data));
            float v368_data = s0[19];
            float v370_data = ir1[1];
            ir1[1] = (v370_data + (v362_data * v368_data));
            float v373_data = s0[31];
            float v375_data = ir1[2];
            ir1[2] = (v375_data + (v362_data * v373_data));
            float v378_data = s0[42];
            float v380_data = ir1[3];
            ir1[3] = (v380_data + (v362_data * v378_data));
            float v383_data = s0[54];
            float v385_data = ir1[4];
            ir1[4] = (v385_data + (v362_data * v383_data));
            float v388_data = s0[65];
            float v390_data = ir1[5];
            ir1[5] = (v390_data + (v362_data * v388_data));
            float v393_data = s0[77];
            float v395_data = ir1[6];
            ir1[6] = (v395_data + (v362_data * v393_data));
            float v398_data = s0[89];
            float v400_data = ir1[7];
            ir1[7] = (v400_data + (v362_data * v398_data));
          }
          if (v19_lead < 12) {
            float v406_data = r0[8];
            float v407_data = s0[8];
            float v409_data = ir1[0];
            ir1[0] = (v409_data + (v406_data * v407_data));
            float v412_data = s0[20];
            float v414_data = ir1[1];
            ir1[1] = (v414_data + (v406_data * v412_data));
            float v417_data = s0[33];
            float v419_data = ir1[2];
            ir1[2] = (v419_data + (v406_data * v417_data));
            float v422_data = s0[45];
            float v424_data = ir1[3];
            ir1[3] = (v424_data + (v406_data * v422_data));
            float v427_data = s0[57];
            float v429_data = ir1[4];
            ir1[4] = (v429_data + (v406_data * v427_data));
            float v432_data = s0[70];
            float v434_data = ir1[5];
            ir1[5] = (v434_data + (v406_data * v432_data));
            float v437_data = s0[82];
            float v439_data = ir1[6];
            ir1[6] = (v439_data + (v406_data * v437_data));
            float v442_data = s0[94];
            float v444_data = ir1[7];
            ir1[7] = (v444_data + (v406_data * v442_data));
          }
          if (v19_lead < 12) {
            float v450_data = r0[9];
            float v451_data = s0[9];
            float v453_data = ir1[0];
            ir1[0] = (v453_data + (v450_data * v451_data));
            float v456_data = s0[21];
            float v458_data = ir1[1];
            ir1[1] = (v458_data + (v450_data * v456_data));
            float v461_data = s0[32];
            float v463_data = ir1[2];
            ir1[2] = (v463_data + (v450_data * v461_data));
            float v466_data = s0[44];
            float v468_data = ir1[3];
            ir1[3] = (v468_data + (v450_data * v466_data));
            float v471_data = s0[56];
            float v473_data = ir1[4];
            ir1[4] = (v473_data + (v450_data * v471_data));
            float v476_data = s0[71];
            float v478_data = ir1[5];
            ir1[5] = (v478_data + (v450_data * v476_data));
            float v481_data = s0[83];
            float v483_data = ir1[6];
            ir1[6] = (v483_data + (v450_data * v481_data));
            float v486_data = s0[95];
            float v488_data = ir1[7];
            ir1[7] = (v488_data + (v450_data * v486_data));
          }
          if (v19_lead < 12) {
            float v494_data = r0[10];
            float v495_data = s0[10];
            float v497_data = ir1[0];
            ir1[0] = (v497_data + (v494_data * v495_data));
            float v500_data = s0[22];
            float v502_data = ir1[1];
            ir1[1] = (v502_data + (v494_data * v500_data));
            float v505_data = s0[35];
            float v507_data = ir1[2];
            ir1[2] = (v507_data + (v494_data * v505_data));
            float v510_data = s0[47];
            float v512_data = ir1[3];
            ir1[3] = (v512_data + (v494_data * v510_data));
            float v515_data = s0[59];
            float v517_data = ir1[4];
            ir1[4] = (v517_data + (v494_data * v515_data));
            float v520_data = s0[68];
            float v522_data = ir1[5];
            ir1[5] = (v522_data + (v494_data * v520_data));
            float v525_data = s0[80];
            float v527_data = ir1[6];
            ir1[6] = (v527_data + (v494_data * v525_data));
            float v530_data = s0[92];
            float v532_data = ir1[7];
            ir1[7] = (v532_data + (v494_data * v530_data));
          }
          if (v19_lead < 12) {
            float v538_data = r0[11];
            float v539_data = s0[11];
            float v541_data = ir1[0];
            ir1[0] = (v541_data + (v538_data * v539_data));
            float v544_data = s0[23];
            float v546_data = ir1[1];
            ir1[1] = (v546_data + (v538_data * v544_data));
            float v549_data = s0[34];
            float v551_data = ir1[2];
            ir1[2] = (v551_data + (v538_data * v549_data));
            float v554_data = s0[46];
            float v556_data = ir1[3];
            ir1[3] = (v556_data + (v538_data * v554_data));
            float v559_data = s0[58];
            float v561_data = ir1[4];
            ir1[4] = (v561_data + (v538_data * v559_data));
            float v564_data = s0[69];
            float v566_data = ir1[5];
            ir1[5] = (v566_data + (v538_data * v564_data));
            float v569_data = s0[81];
            float v571_data = ir1[6];
            ir1[6] = (v571_data + (v538_data * v569_data));
            float v574_data = s0[93];
            float v576_data = ir1[7];
            ir1[7] = (v576_data + (v538_data * v574_data));
          }
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v582_n1 = 0; v582_n1 < 8; ++v582_n1) {
              float v584_data = ir1[v582_n1];
              r1[v582_n1] = v584_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v593_i1 = 0; v593_i1 < 12; ++v593_i1) {
              float v601_data = __ldcg(&glb_m5[(v19_lead + (v593_i1 * 12))]);
              r4[v593_i1] = v601_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v19_lead < 12) {
            float v609_data = r2[0];
            float v610_data = s1[0];
            float v612_data = ir3[0];
            ir3[0] = (v612_data + (v609_data * v610_data));
            float v615_data = s1[12];
            float v617_data = ir3[1];
            ir3[1] = (v617_data + (v609_data * v615_data));
            float v620_data = s1[24];
            float v622_data = ir3[2];
            ir3[2] = (v622_data + (v609_data * v620_data));
            float v625_data = s1[37];
            float v627_data = ir3[3];
            ir3[3] = (v627_data + (v609_data * v625_data));
            float v630_data = s1[49];
            float v632_data = ir3[4];
            ir3[4] = (v632_data + (v609_data * v630_data));
            float v635_data = s1[61];
            float v637_data = ir3[5];
            ir3[5] = (v637_data + (v609_data * v635_data));
            float v640_data = s1[74];
            float v642_data = ir3[6];
            ir3[6] = (v642_data + (v609_data * v640_data));
            float v645_data = s1[86];
            float v647_data = ir3[7];
            ir3[7] = (v647_data + (v609_data * v645_data));
          }
          if (v19_lead < 12) {
            float v653_data = r2[1];
            float v654_data = s1[1];
            float v656_data = ir3[0];
            ir3[0] = (v656_data + (v653_data * v654_data));
            float v659_data = s1[13];
            float v661_data = ir3[1];
            ir3[1] = (v661_data + (v653_data * v659_data));
            float v664_data = s1[25];
            float v666_data = ir3[2];
            ir3[2] = (v666_data + (v653_data * v664_data));
            float v669_data = s1[36];
            float v671_data = ir3[3];
            ir3[3] = (v671_data + (v653_data * v669_data));
            float v674_data = s1[48];
            float v676_data = ir3[4];
            ir3[4] = (v676_data + (v653_data * v674_data));
            float v679_data = s1[60];
            float v681_data = ir3[5];
            ir3[5] = (v681_data + (v653_data * v679_data));
            float v684_data = s1[75];
            float v686_data = ir3[6];
            ir3[6] = (v686_data + (v653_data * v684_data));
            float v689_data = s1[87];
            float v691_data = ir3[7];
            ir3[7] = (v691_data + (v653_data * v689_data));
          }
          if (v19_lead < 12) {
            float v697_data = r2[2];
            float v698_data = s1[2];
            float v700_data = ir3[0];
            ir3[0] = (v700_data + (v697_data * v698_data));
            float v703_data = s1[14];
            float v705_data = ir3[1];
            ir3[1] = (v705_data + (v697_data * v703_data));
            float v708_data = s1[26];
            float v710_data = ir3[2];
            ir3[2] = (v710_data + (v697_data * v708_data));
            float v713_data = s1[39];
            float v715_data = ir3[3];
            ir3[3] = (v715_data + (v697_data * v713_data));
            float v718_data = s1[51];
            float v720_data = ir3[4];
            ir3[4] = (v720_data + (v697_data * v718_data));
            float v723_data = s1[63];
            float v725_data = ir3[5];
            ir3[5] = (v725_data + (v697_data * v723_data));
            float v728_data = s1[72];
            float v730_data = ir3[6];
            ir3[6] = (v730_data + (v697_data * v728_data));
            float v733_data = s1[84];
            float v735_data = ir3[7];
            ir3[7] = (v735_data + (v697_data * v733_data));
          }
          if (v19_lead < 12) {
            float v741_data = r2[3];
            float v742_data = s1[3];
            float v744_data = ir3[0];
            ir3[0] = (v744_data + (v741_data * v742_data));
            float v747_data = s1[15];
            float v749_data = ir3[1];
            ir3[1] = (v749_data + (v741_data * v747_data));
            float v752_data = s1[27];
            float v754_data = ir3[2];
            ir3[2] = (v754_data + (v741_data * v752_data));
            float v757_data = s1[38];
            float v759_data = ir3[3];
            ir3[3] = (v759_data + (v741_data * v757_data));
            float v762_data = s1[50];
            float v764_data = ir3[4];
            ir3[4] = (v764_data + (v741_data * v762_data));
            float v767_data = s1[62];
            float v769_data = ir3[5];
            ir3[5] = (v769_data + (v741_data * v767_data));
            float v772_data = s1[73];
            float v774_data = ir3[6];
            ir3[6] = (v774_data + (v741_data * v772_data));
            float v777_data = s1[85];
            float v779_data = ir3[7];
            ir3[7] = (v779_data + (v741_data * v777_data));
          }
          if (v19_lead < 12) {
            float v785_data = r2[4];
            float v786_data = s1[4];
            float v788_data = ir3[0];
            ir3[0] = (v788_data + (v785_data * v786_data));
            float v791_data = s1[16];
            float v793_data = ir3[1];
            ir3[1] = (v793_data + (v785_data * v791_data));
            float v796_data = s1[28];
            float v798_data = ir3[2];
            ir3[2] = (v798_data + (v785_data * v796_data));
            float v801_data = s1[41];
            float v803_data = ir3[3];
            ir3[3] = (v803_data + (v785_data * v801_data));
            float v806_data = s1[53];
            float v808_data = ir3[4];
            ir3[4] = (v808_data + (v785_data * v806_data));
            float v811_data = s1[66];
            float v813_data = ir3[5];
            ir3[5] = (v813_data + (v785_data * v811_data));
            float v816_data = s1[78];
            float v818_data = ir3[6];
            ir3[6] = (v818_data + (v785_data * v816_data));
            float v821_data = s1[90];
            float v823_data = ir3[7];
            ir3[7] = (v823_data + (v785_data * v821_data));
          }
          if (v19_lead < 12) {
            float v829_data = r2[5];
            float v830_data = s1[5];
            float v832_data = ir3[0];
            ir3[0] = (v832_data + (v829_data * v830_data));
            float v835_data = s1[17];
            float v837_data = ir3[1];
            ir3[1] = (v837_data + (v829_data * v835_data));
            float v840_data = s1[29];
            float v842_data = ir3[2];
            ir3[2] = (v842_data + (v829_data * v840_data));
            float v845_data = s1[40];
            float v847_data = ir3[3];
            ir3[3] = (v847_data + (v829_data * v845_data));
            float v850_data = s1[52];
            float v852_data = ir3[4];
            ir3[4] = (v852_data + (v829_data * v850_data));
            float v855_data = s1[67];
            float v857_data = ir3[5];
            ir3[5] = (v857_data + (v829_data * v855_data));
            float v860_data = s1[79];
            float v862_data = ir3[6];
            ir3[6] = (v862_data + (v829_data * v860_data));
            float v865_data = s1[91];
            float v867_data = ir3[7];
            ir3[7] = (v867_data + (v829_data * v865_data));
          }
          if (v19_lead < 12) {
            float v873_data = r2[6];
            float v874_data = s1[6];
            float v876_data = ir3[0];
            ir3[0] = (v876_data + (v873_data * v874_data));
            float v879_data = s1[18];
            float v881_data = ir3[1];
            ir3[1] = (v881_data + (v873_data * v879_data));
            float v884_data = s1[30];
            float v886_data = ir3[2];
            ir3[2] = (v886_data + (v873_data * v884_data));
            float v889_data = s1[43];
            float v891_data = ir3[3];
            ir3[3] = (v891_data + (v873_data * v889_data));
            float v894_data = s1[55];
            float v896_data = ir3[4];
            ir3[4] = (v896_data + (v873_data * v894_data));
            float v899_data = s1[64];
            float v901_data = ir3[5];
            ir3[5] = (v901_data + (v873_data * v899_data));
            float v904_data = s1[76];
            float v906_data = ir3[6];
            ir3[6] = (v906_data + (v873_data * v904_data));
            float v909_data = s1[88];
            float v911_data = ir3[7];
            ir3[7] = (v911_data + (v873_data * v909_data));
          }
          if (v19_lead < 12) {
            float v917_data = r2[7];
            float v918_data = s1[7];
            float v920_data = ir3[0];
            ir3[0] = (v920_data + (v917_data * v918_data));
            float v923_data = s1[19];
            float v925_data = ir3[1];
            ir3[1] = (v925_data + (v917_data * v923_data));
            float v928_data = s1[31];
            float v930_data = ir3[2];
            ir3[2] = (v930_data + (v917_data * v928_data));
            float v933_data = s1[42];
            float v935_data = ir3[3];
            ir3[3] = (v935_data + (v917_data * v933_data));
            float v938_data = s1[54];
            float v940_data = ir3[4];
            ir3[4] = (v940_data + (v917_data * v938_data));
            float v943_data = s1[65];
            float v945_data = ir3[5];
            ir3[5] = (v945_data + (v917_data * v943_data));
            float v948_data = s1[77];
            float v950_data = ir3[6];
            ir3[6] = (v950_data + (v917_data * v948_data));
            float v953_data = s1[89];
            float v955_data = ir3[7];
            ir3[7] = (v955_data + (v917_data * v953_data));
          }
          if (v19_lead < 12) {
            float v961_data = r2[8];
            float v962_data = s1[8];
            float v964_data = ir3[0];
            ir3[0] = (v964_data + (v961_data * v962_data));
            float v967_data = s1[20];
            float v969_data = ir3[1];
            ir3[1] = (v969_data + (v961_data * v967_data));
            float v972_data = s1[33];
            float v974_data = ir3[2];
            ir3[2] = (v974_data + (v961_data * v972_data));
            float v977_data = s1[45];
            float v979_data = ir3[3];
            ir3[3] = (v979_data + (v961_data * v977_data));
            float v982_data = s1[57];
            float v984_data = ir3[4];
            ir3[4] = (v984_data + (v961_data * v982_data));
            float v987_data = s1[70];
            float v989_data = ir3[5];
            ir3[5] = (v989_data + (v961_data * v987_data));
            float v992_data = s1[82];
            float v994_data = ir3[6];
            ir3[6] = (v994_data + (v961_data * v992_data));
            float v997_data = s1[94];
            float v999_data = ir3[7];
            ir3[7] = (v999_data + (v961_data * v997_data));
          }
          if (v19_lead < 12) {
            float v1005_data = r2[9];
            float v1006_data = s1[9];
            float v1008_data = ir3[0];
            ir3[0] = (v1008_data + (v1005_data * v1006_data));
            float v1011_data = s1[21];
            float v1013_data = ir3[1];
            ir3[1] = (v1013_data + (v1005_data * v1011_data));
            float v1016_data = s1[32];
            float v1018_data = ir3[2];
            ir3[2] = (v1018_data + (v1005_data * v1016_data));
            float v1021_data = s1[44];
            float v1023_data = ir3[3];
            ir3[3] = (v1023_data + (v1005_data * v1021_data));
            float v1026_data = s1[56];
            float v1028_data = ir3[4];
            ir3[4] = (v1028_data + (v1005_data * v1026_data));
            float v1031_data = s1[71];
            float v1033_data = ir3[5];
            ir3[5] = (v1033_data + (v1005_data * v1031_data));
            float v1036_data = s1[83];
            float v1038_data = ir3[6];
            ir3[6] = (v1038_data + (v1005_data * v1036_data));
            float v1041_data = s1[95];
            float v1043_data = ir3[7];
            ir3[7] = (v1043_data + (v1005_data * v1041_data));
          }
          if (v19_lead < 12) {
            float v1049_data = r2[10];
            float v1050_data = s1[10];
            float v1052_data = ir3[0];
            ir3[0] = (v1052_data + (v1049_data * v1050_data));
            float v1055_data = s1[22];
            float v1057_data = ir3[1];
            ir3[1] = (v1057_data + (v1049_data * v1055_data));
            float v1060_data = s1[35];
            float v1062_data = ir3[2];
            ir3[2] = (v1062_data + (v1049_data * v1060_data));
            float v1065_data = s1[47];
            float v1067_data = ir3[3];
            ir3[3] = (v1067_data + (v1049_data * v1065_data));
            float v1070_data = s1[59];
            float v1072_data = ir3[4];
            ir3[4] = (v1072_data + (v1049_data * v1070_data));
            float v1075_data = s1[68];
            float v1077_data = ir3[5];
            ir3[5] = (v1077_data + (v1049_data * v1075_data));
            float v1080_data = s1[80];
            float v1082_data = ir3[6];
            ir3[6] = (v1082_data + (v1049_data * v1080_data));
            float v1085_data = s1[92];
            float v1087_data = ir3[7];
            ir3[7] = (v1087_data + (v1049_data * v1085_data));
          }
          if (v19_lead < 12) {
            float v1093_data = r2[11];
            float v1094_data = s1[11];
            float v1096_data = ir3[0];
            ir3[0] = (v1096_data + (v1093_data * v1094_data));
            float v1099_data = s1[23];
            float v1101_data = ir3[1];
            ir3[1] = (v1101_data + (v1093_data * v1099_data));
            float v1104_data = s1[34];
            float v1106_data = ir3[2];
            ir3[2] = (v1106_data + (v1093_data * v1104_data));
            float v1109_data = s1[46];
            float v1111_data = ir3[3];
            ir3[3] = (v1111_data + (v1093_data * v1109_data));
            float v1114_data = s1[58];
            float v1116_data = ir3[4];
            ir3[4] = (v1116_data + (v1093_data * v1114_data));
            float v1119_data = s1[69];
            float v1121_data = ir3[5];
            ir3[5] = (v1121_data + (v1093_data * v1119_data));
            float v1124_data = s1[81];
            float v1126_data = ir3[6];
            ir3[6] = (v1126_data + (v1093_data * v1124_data));
            float v1129_data = s1[93];
            float v1131_data = ir3[7];
            ir3[7] = (v1131_data + (v1093_data * v1129_data));
          }
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v1137_n1 = 0; v1137_n1 < 8; ++v1137_n1) {
              float v1139_data = ir3[v1137_n1];
              float v1141_data = r1[v1137_n1];
              r3[v1137_n1] = (v1141_data + v1139_data);
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          // s2 = load{g>s}(glb_m6[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v1151_i1 = 0; v1151_i1 < 12; ++v1151_i1) {
              float v1159_data = __ldcg(&glb_m7[(v19_lead + (v1151_i1 * 12))]);
              r6[v1151_i1] = v1159_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v19_lead < 12) {
            float v1167_data = r4[0];
            float v1168_data = s2[0];
            float v1170_data = ir5[0];
            ir5[0] = (v1170_data + (v1167_data * v1168_data));
            float v1173_data = s2[12];
            float v1175_data = ir5[1];
            ir5[1] = (v1175_data + (v1167_data * v1173_data));
            float v1178_data = s2[24];
            float v1180_data = ir5[2];
            ir5[2] = (v1180_data + (v1167_data * v1178_data));
            float v1183_data = s2[37];
            float v1185_data = ir5[3];
            ir5[3] = (v1185_data + (v1167_data * v1183_data));
            float v1188_data = s2[49];
            float v1190_data = ir5[4];
            ir5[4] = (v1190_data + (v1167_data * v1188_data));
            float v1193_data = s2[61];
            float v1195_data = ir5[5];
            ir5[5] = (v1195_data + (v1167_data * v1193_data));
            float v1198_data = s2[74];
            float v1200_data = ir5[6];
            ir5[6] = (v1200_data + (v1167_data * v1198_data));
            float v1203_data = s2[86];
            float v1205_data = ir5[7];
            ir5[7] = (v1205_data + (v1167_data * v1203_data));
          }
          if (v19_lead < 12) {
            float v1211_data = r4[1];
            float v1212_data = s2[1];
            float v1214_data = ir5[0];
            ir5[0] = (v1214_data + (v1211_data * v1212_data));
            float v1217_data = s2[13];
            float v1219_data = ir5[1];
            ir5[1] = (v1219_data + (v1211_data * v1217_data));
            float v1222_data = s2[25];
            float v1224_data = ir5[2];
            ir5[2] = (v1224_data + (v1211_data * v1222_data));
            float v1227_data = s2[36];
            float v1229_data = ir5[3];
            ir5[3] = (v1229_data + (v1211_data * v1227_data));
            float v1232_data = s2[48];
            float v1234_data = ir5[4];
            ir5[4] = (v1234_data + (v1211_data * v1232_data));
            float v1237_data = s2[60];
            float v1239_data = ir5[5];
            ir5[5] = (v1239_data + (v1211_data * v1237_data));
            float v1242_data = s2[75];
            float v1244_data = ir5[6];
            ir5[6] = (v1244_data + (v1211_data * v1242_data));
            float v1247_data = s2[87];
            float v1249_data = ir5[7];
            ir5[7] = (v1249_data + (v1211_data * v1247_data));
          }
          if (v19_lead < 12) {
            float v1255_data = r4[2];
            float v1256_data = s2[2];
            float v1258_data = ir5[0];
            ir5[0] = (v1258_data + (v1255_data * v1256_data));
            float v1261_data = s2[14];
            float v1263_data = ir5[1];
            ir5[1] = (v1263_data + (v1255_data * v1261_data));
            float v1266_data = s2[26];
            float v1268_data = ir5[2];
            ir5[2] = (v1268_data + (v1255_data * v1266_data));
            float v1271_data = s2[39];
            float v1273_data = ir5[3];
            ir5[3] = (v1273_data + (v1255_data * v1271_data));
            float v1276_data = s2[51];
            float v1278_data = ir5[4];
            ir5[4] = (v1278_data + (v1255_data * v1276_data));
            float v1281_data = s2[63];
            float v1283_data = ir5[5];
            ir5[5] = (v1283_data + (v1255_data * v1281_data));
            float v1286_data = s2[72];
            float v1288_data = ir5[6];
            ir5[6] = (v1288_data + (v1255_data * v1286_data));
            float v1291_data = s2[84];
            float v1293_data = ir5[7];
            ir5[7] = (v1293_data + (v1255_data * v1291_data));
          }
          if (v19_lead < 12) {
            float v1299_data = r4[3];
            float v1300_data = s2[3];
            float v1302_data = ir5[0];
            ir5[0] = (v1302_data + (v1299_data * v1300_data));
            float v1305_data = s2[15];
            float v1307_data = ir5[1];
            ir5[1] = (v1307_data + (v1299_data * v1305_data));
            float v1310_data = s2[27];
            float v1312_data = ir5[2];
            ir5[2] = (v1312_data + (v1299_data * v1310_data));
            float v1315_data = s2[38];
            float v1317_data = ir5[3];
            ir5[3] = (v1317_data + (v1299_data * v1315_data));
            float v1320_data = s2[50];
            float v1322_data = ir5[4];
            ir5[4] = (v1322_data + (v1299_data * v1320_data));
            float v1325_data = s2[62];
            float v1327_data = ir5[5];
            ir5[5] = (v1327_data + (v1299_data * v1325_data));
            float v1330_data = s2[73];
            float v1332_data = ir5[6];
            ir5[6] = (v1332_data + (v1299_data * v1330_data));
            float v1335_data = s2[85];
            float v1337_data = ir5[7];
            ir5[7] = (v1337_data + (v1299_data * v1335_data));
          }
          if (v19_lead < 12) {
            float v1343_data = r4[4];
            float v1344_data = s2[4];
            float v1346_data = ir5[0];
            ir5[0] = (v1346_data + (v1343_data * v1344_data));
            float v1349_data = s2[16];
            float v1351_data = ir5[1];
            ir5[1] = (v1351_data + (v1343_data * v1349_data));
            float v1354_data = s2[28];
            float v1356_data = ir5[2];
            ir5[2] = (v1356_data + (v1343_data * v1354_data));
            float v1359_data = s2[41];
            float v1361_data = ir5[3];
            ir5[3] = (v1361_data + (v1343_data * v1359_data));
            float v1364_data = s2[53];
            float v1366_data = ir5[4];
            ir5[4] = (v1366_data + (v1343_data * v1364_data));
            float v1369_data = s2[66];
            float v1371_data = ir5[5];
            ir5[5] = (v1371_data + (v1343_data * v1369_data));
            float v1374_data = s2[78];
            float v1376_data = ir5[6];
            ir5[6] = (v1376_data + (v1343_data * v1374_data));
            float v1379_data = s2[90];
            float v1381_data = ir5[7];
            ir5[7] = (v1381_data + (v1343_data * v1379_data));
          }
          if (v19_lead < 12) {
            float v1387_data = r4[5];
            float v1388_data = s2[5];
            float v1390_data = ir5[0];
            ir5[0] = (v1390_data + (v1387_data * v1388_data));
            float v1393_data = s2[17];
            float v1395_data = ir5[1];
            ir5[1] = (v1395_data + (v1387_data * v1393_data));
            float v1398_data = s2[29];
            float v1400_data = ir5[2];
            ir5[2] = (v1400_data + (v1387_data * v1398_data));
            float v1403_data = s2[40];
            float v1405_data = ir5[3];
            ir5[3] = (v1405_data + (v1387_data * v1403_data));
            float v1408_data = s2[52];
            float v1410_data = ir5[4];
            ir5[4] = (v1410_data + (v1387_data * v1408_data));
            float v1413_data = s2[67];
            float v1415_data = ir5[5];
            ir5[5] = (v1415_data + (v1387_data * v1413_data));
            float v1418_data = s2[79];
            float v1420_data = ir5[6];
            ir5[6] = (v1420_data + (v1387_data * v1418_data));
            float v1423_data = s2[91];
            float v1425_data = ir5[7];
            ir5[7] = (v1425_data + (v1387_data * v1423_data));
          }
          if (v19_lead < 12) {
            float v1431_data = r4[6];
            float v1432_data = s2[6];
            float v1434_data = ir5[0];
            ir5[0] = (v1434_data + (v1431_data * v1432_data));
            float v1437_data = s2[18];
            float v1439_data = ir5[1];
            ir5[1] = (v1439_data + (v1431_data * v1437_data));
            float v1442_data = s2[30];
            float v1444_data = ir5[2];
            ir5[2] = (v1444_data + (v1431_data * v1442_data));
            float v1447_data = s2[43];
            float v1449_data = ir5[3];
            ir5[3] = (v1449_data + (v1431_data * v1447_data));
            float v1452_data = s2[55];
            float v1454_data = ir5[4];
            ir5[4] = (v1454_data + (v1431_data * v1452_data));
            float v1457_data = s2[64];
            float v1459_data = ir5[5];
            ir5[5] = (v1459_data + (v1431_data * v1457_data));
            float v1462_data = s2[76];
            float v1464_data = ir5[6];
            ir5[6] = (v1464_data + (v1431_data * v1462_data));
            float v1467_data = s2[88];
            float v1469_data = ir5[7];
            ir5[7] = (v1469_data + (v1431_data * v1467_data));
          }
          if (v19_lead < 12) {
            float v1475_data = r4[7];
            float v1476_data = s2[7];
            float v1478_data = ir5[0];
            ir5[0] = (v1478_data + (v1475_data * v1476_data));
            float v1481_data = s2[19];
            float v1483_data = ir5[1];
            ir5[1] = (v1483_data + (v1475_data * v1481_data));
            float v1486_data = s2[31];
            float v1488_data = ir5[2];
            ir5[2] = (v1488_data + (v1475_data * v1486_data));
            float v1491_data = s2[42];
            float v1493_data = ir5[3];
            ir5[3] = (v1493_data + (v1475_data * v1491_data));
            float v1496_data = s2[54];
            float v1498_data = ir5[4];
            ir5[4] = (v1498_data + (v1475_data * v1496_data));
            float v1501_data = s2[65];
            float v1503_data = ir5[5];
            ir5[5] = (v1503_data + (v1475_data * v1501_data));
            float v1506_data = s2[77];
            float v1508_data = ir5[6];
            ir5[6] = (v1508_data + (v1475_data * v1506_data));
            float v1511_data = s2[89];
            float v1513_data = ir5[7];
            ir5[7] = (v1513_data + (v1475_data * v1511_data));
          }
          if (v19_lead < 12) {
            float v1519_data = r4[8];
            float v1520_data = s2[8];
            float v1522_data = ir5[0];
            ir5[0] = (v1522_data + (v1519_data * v1520_data));
            float v1525_data = s2[20];
            float v1527_data = ir5[1];
            ir5[1] = (v1527_data + (v1519_data * v1525_data));
            float v1530_data = s2[33];
            float v1532_data = ir5[2];
            ir5[2] = (v1532_data + (v1519_data * v1530_data));
            float v1535_data = s2[45];
            float v1537_data = ir5[3];
            ir5[3] = (v1537_data + (v1519_data * v1535_data));
            float v1540_data = s2[57];
            float v1542_data = ir5[4];
            ir5[4] = (v1542_data + (v1519_data * v1540_data));
            float v1545_data = s2[70];
            float v1547_data = ir5[5];
            ir5[5] = (v1547_data + (v1519_data * v1545_data));
            float v1550_data = s2[82];
            float v1552_data = ir5[6];
            ir5[6] = (v1552_data + (v1519_data * v1550_data));
            float v1555_data = s2[94];
            float v1557_data = ir5[7];
            ir5[7] = (v1557_data + (v1519_data * v1555_data));
          }
          if (v19_lead < 12) {
            float v1563_data = r4[9];
            float v1564_data = s2[9];
            float v1566_data = ir5[0];
            ir5[0] = (v1566_data + (v1563_data * v1564_data));
            float v1569_data = s2[21];
            float v1571_data = ir5[1];
            ir5[1] = (v1571_data + (v1563_data * v1569_data));
            float v1574_data = s2[32];
            float v1576_data = ir5[2];
            ir5[2] = (v1576_data + (v1563_data * v1574_data));
            float v1579_data = s2[44];
            float v1581_data = ir5[3];
            ir5[3] = (v1581_data + (v1563_data * v1579_data));
            float v1584_data = s2[56];
            float v1586_data = ir5[4];
            ir5[4] = (v1586_data + (v1563_data * v1584_data));
            float v1589_data = s2[71];
            float v1591_data = ir5[5];
            ir5[5] = (v1591_data + (v1563_data * v1589_data));
            float v1594_data = s2[83];
            float v1596_data = ir5[6];
            ir5[6] = (v1596_data + (v1563_data * v1594_data));
            float v1599_data = s2[95];
            float v1601_data = ir5[7];
            ir5[7] = (v1601_data + (v1563_data * v1599_data));
          }
          if (v19_lead < 12) {
            float v1607_data = r4[10];
            float v1608_data = s2[10];
            float v1610_data = ir5[0];
            ir5[0] = (v1610_data + (v1607_data * v1608_data));
            float v1613_data = s2[22];
            float v1615_data = ir5[1];
            ir5[1] = (v1615_data + (v1607_data * v1613_data));
            float v1618_data = s2[35];
            float v1620_data = ir5[2];
            ir5[2] = (v1620_data + (v1607_data * v1618_data));
            float v1623_data = s2[47];
            float v1625_data = ir5[3];
            ir5[3] = (v1625_data + (v1607_data * v1623_data));
            float v1628_data = s2[59];
            float v1630_data = ir5[4];
            ir5[4] = (v1630_data + (v1607_data * v1628_data));
            float v1633_data = s2[68];
            float v1635_data = ir5[5];
            ir5[5] = (v1635_data + (v1607_data * v1633_data));
            float v1638_data = s2[80];
            float v1640_data = ir5[6];
            ir5[6] = (v1640_data + (v1607_data * v1638_data));
            float v1643_data = s2[92];
            float v1645_data = ir5[7];
            ir5[7] = (v1645_data + (v1607_data * v1643_data));
          }
          if (v19_lead < 12) {
            float v1651_data = r4[11];
            float v1652_data = s2[11];
            float v1654_data = ir5[0];
            ir5[0] = (v1654_data + (v1651_data * v1652_data));
            float v1657_data = s2[23];
            float v1659_data = ir5[1];
            ir5[1] = (v1659_data + (v1651_data * v1657_data));
            float v1662_data = s2[34];
            float v1664_data = ir5[2];
            ir5[2] = (v1664_data + (v1651_data * v1662_data));
            float v1667_data = s2[46];
            float v1669_data = ir5[3];
            ir5[3] = (v1669_data + (v1651_data * v1667_data));
            float v1672_data = s2[58];
            float v1674_data = ir5[4];
            ir5[4] = (v1674_data + (v1651_data * v1672_data));
            float v1677_data = s2[69];
            float v1679_data = ir5[5];
            ir5[5] = (v1679_data + (v1651_data * v1677_data));
            float v1682_data = s2[81];
            float v1684_data = ir5[6];
            ir5[6] = (v1684_data + (v1651_data * v1682_data));
            float v1687_data = s2[93];
            float v1689_data = ir5[7];
            ir5[7] = (v1689_data + (v1651_data * v1687_data));
          }
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v1695_n1 = 0; v1695_n1 < 8; ++v1695_n1) {
              float v1697_data = ir5[v1695_n1];
              float v1699_data = r3[v1695_n1];
              r5[v1695_n1] = (v1699_data + v1697_data);
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          // s3 = load{g>s}(glb_m8[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v19_lead < 12) {
            float v1710_data = r6[0];
            float v1711_data = s3[0];
            float v1713_data = ir7[0];
            ir7[0] = (v1713_data + (v1710_data * v1711_data));
            float v1716_data = s3[12];
            float v1718_data = ir7[1];
            ir7[1] = (v1718_data + (v1710_data * v1716_data));
            float v1721_data = s3[24];
            float v1723_data = ir7[2];
            ir7[2] = (v1723_data + (v1710_data * v1721_data));
            float v1726_data = s3[37];
            float v1728_data = ir7[3];
            ir7[3] = (v1728_data + (v1710_data * v1726_data));
            float v1731_data = s3[49];
            float v1733_data = ir7[4];
            ir7[4] = (v1733_data + (v1710_data * v1731_data));
            float v1736_data = s3[61];
            float v1738_data = ir7[5];
            ir7[5] = (v1738_data + (v1710_data * v1736_data));
            float v1741_data = s3[74];
            float v1743_data = ir7[6];
            ir7[6] = (v1743_data + (v1710_data * v1741_data));
            float v1746_data = s3[86];
            float v1748_data = ir7[7];
            ir7[7] = (v1748_data + (v1710_data * v1746_data));
          }
          if (v19_lead < 12) {
            float v1754_data = r6[1];
            float v1755_data = s3[1];
            float v1757_data = ir7[0];
            ir7[0] = (v1757_data + (v1754_data * v1755_data));
            float v1760_data = s3[13];
            float v1762_data = ir7[1];
            ir7[1] = (v1762_data + (v1754_data * v1760_data));
            float v1765_data = s3[25];
            float v1767_data = ir7[2];
            ir7[2] = (v1767_data + (v1754_data * v1765_data));
            float v1770_data = s3[36];
            float v1772_data = ir7[3];
            ir7[3] = (v1772_data + (v1754_data * v1770_data));
            float v1775_data = s3[48];
            float v1777_data = ir7[4];
            ir7[4] = (v1777_data + (v1754_data * v1775_data));
            float v1780_data = s3[60];
            float v1782_data = ir7[5];
            ir7[5] = (v1782_data + (v1754_data * v1780_data));
            float v1785_data = s3[75];
            float v1787_data = ir7[6];
            ir7[6] = (v1787_data + (v1754_data * v1785_data));
            float v1790_data = s3[87];
            float v1792_data = ir7[7];
            ir7[7] = (v1792_data + (v1754_data * v1790_data));
          }
          if (v19_lead < 12) {
            float v1798_data = r6[2];
            float v1799_data = s3[2];
            float v1801_data = ir7[0];
            ir7[0] = (v1801_data + (v1798_data * v1799_data));
            float v1804_data = s3[14];
            float v1806_data = ir7[1];
            ir7[1] = (v1806_data + (v1798_data * v1804_data));
            float v1809_data = s3[26];
            float v1811_data = ir7[2];
            ir7[2] = (v1811_data + (v1798_data * v1809_data));
            float v1814_data = s3[39];
            float v1816_data = ir7[3];
            ir7[3] = (v1816_data + (v1798_data * v1814_data));
            float v1819_data = s3[51];
            float v1821_data = ir7[4];
            ir7[4] = (v1821_data + (v1798_data * v1819_data));
            float v1824_data = s3[63];
            float v1826_data = ir7[5];
            ir7[5] = (v1826_data + (v1798_data * v1824_data));
            float v1829_data = s3[72];
            float v1831_data = ir7[6];
            ir7[6] = (v1831_data + (v1798_data * v1829_data));
            float v1834_data = s3[84];
            float v1836_data = ir7[7];
            ir7[7] = (v1836_data + (v1798_data * v1834_data));
          }
          if (v19_lead < 12) {
            float v1842_data = r6[3];
            float v1843_data = s3[3];
            float v1845_data = ir7[0];
            ir7[0] = (v1845_data + (v1842_data * v1843_data));
            float v1848_data = s3[15];
            float v1850_data = ir7[1];
            ir7[1] = (v1850_data + (v1842_data * v1848_data));
            float v1853_data = s3[27];
            float v1855_data = ir7[2];
            ir7[2] = (v1855_data + (v1842_data * v1853_data));
            float v1858_data = s3[38];
            float v1860_data = ir7[3];
            ir7[3] = (v1860_data + (v1842_data * v1858_data));
            float v1863_data = s3[50];
            float v1865_data = ir7[4];
            ir7[4] = (v1865_data + (v1842_data * v1863_data));
            float v1868_data = s3[62];
            float v1870_data = ir7[5];
            ir7[5] = (v1870_data + (v1842_data * v1868_data));
            float v1873_data = s3[73];
            float v1875_data = ir7[6];
            ir7[6] = (v1875_data + (v1842_data * v1873_data));
            float v1878_data = s3[85];
            float v1880_data = ir7[7];
            ir7[7] = (v1880_data + (v1842_data * v1878_data));
          }
          if (v19_lead < 12) {
            float v1886_data = r6[4];
            float v1887_data = s3[4];
            float v1889_data = ir7[0];
            ir7[0] = (v1889_data + (v1886_data * v1887_data));
            float v1892_data = s3[16];
            float v1894_data = ir7[1];
            ir7[1] = (v1894_data + (v1886_data * v1892_data));
            float v1897_data = s3[28];
            float v1899_data = ir7[2];
            ir7[2] = (v1899_data + (v1886_data * v1897_data));
            float v1902_data = s3[41];
            float v1904_data = ir7[3];
            ir7[3] = (v1904_data + (v1886_data * v1902_data));
            float v1907_data = s3[53];
            float v1909_data = ir7[4];
            ir7[4] = (v1909_data + (v1886_data * v1907_data));
            float v1912_data = s3[66];
            float v1914_data = ir7[5];
            ir7[5] = (v1914_data + (v1886_data * v1912_data));
            float v1917_data = s3[78];
            float v1919_data = ir7[6];
            ir7[6] = (v1919_data + (v1886_data * v1917_data));
            float v1922_data = s3[90];
            float v1924_data = ir7[7];
            ir7[7] = (v1924_data + (v1886_data * v1922_data));
          }
          if (v19_lead < 12) {
            float v1930_data = r6[5];
            float v1931_data = s3[5];
            float v1933_data = ir7[0];
            ir7[0] = (v1933_data + (v1930_data * v1931_data));
            float v1936_data = s3[17];
            float v1938_data = ir7[1];
            ir7[1] = (v1938_data + (v1930_data * v1936_data));
            float v1941_data = s3[29];
            float v1943_data = ir7[2];
            ir7[2] = (v1943_data + (v1930_data * v1941_data));
            float v1946_data = s3[40];
            float v1948_data = ir7[3];
            ir7[3] = (v1948_data + (v1930_data * v1946_data));
            float v1951_data = s3[52];
            float v1953_data = ir7[4];
            ir7[4] = (v1953_data + (v1930_data * v1951_data));
            float v1956_data = s3[67];
            float v1958_data = ir7[5];
            ir7[5] = (v1958_data + (v1930_data * v1956_data));
            float v1961_data = s3[79];
            float v1963_data = ir7[6];
            ir7[6] = (v1963_data + (v1930_data * v1961_data));
            float v1966_data = s3[91];
            float v1968_data = ir7[7];
            ir7[7] = (v1968_data + (v1930_data * v1966_data));
          }
          if (v19_lead < 12) {
            float v1974_data = r6[6];
            float v1975_data = s3[6];
            float v1977_data = ir7[0];
            ir7[0] = (v1977_data + (v1974_data * v1975_data));
            float v1980_data = s3[18];
            float v1982_data = ir7[1];
            ir7[1] = (v1982_data + (v1974_data * v1980_data));
            float v1985_data = s3[30];
            float v1987_data = ir7[2];
            ir7[2] = (v1987_data + (v1974_data * v1985_data));
            float v1990_data = s3[43];
            float v1992_data = ir7[3];
            ir7[3] = (v1992_data + (v1974_data * v1990_data));
            float v1995_data = s3[55];
            float v1997_data = ir7[4];
            ir7[4] = (v1997_data + (v1974_data * v1995_data));
            float v2000_data = s3[64];
            float v2002_data = ir7[5];
            ir7[5] = (v2002_data + (v1974_data * v2000_data));
            float v2005_data = s3[76];
            float v2007_data = ir7[6];
            ir7[6] = (v2007_data + (v1974_data * v2005_data));
            float v2010_data = s3[88];
            float v2012_data = ir7[7];
            ir7[7] = (v2012_data + (v1974_data * v2010_data));
          }
          if (v19_lead < 12) {
            float v2018_data = r6[7];
            float v2019_data = s3[7];
            float v2021_data = ir7[0];
            ir7[0] = (v2021_data + (v2018_data * v2019_data));
            float v2024_data = s3[19];
            float v2026_data = ir7[1];
            ir7[1] = (v2026_data + (v2018_data * v2024_data));
            float v2029_data = s3[31];
            float v2031_data = ir7[2];
            ir7[2] = (v2031_data + (v2018_data * v2029_data));
            float v2034_data = s3[42];
            float v2036_data = ir7[3];
            ir7[3] = (v2036_data + (v2018_data * v2034_data));
            float v2039_data = s3[54];
            float v2041_data = ir7[4];
            ir7[4] = (v2041_data + (v2018_data * v2039_data));
            float v2044_data = s3[65];
            float v2046_data = ir7[5];
            ir7[5] = (v2046_data + (v2018_data * v2044_data));
            float v2049_data = s3[77];
            float v2051_data = ir7[6];
            ir7[6] = (v2051_data + (v2018_data * v2049_data));
            float v2054_data = s3[89];
            float v2056_data = ir7[7];
            ir7[7] = (v2056_data + (v2018_data * v2054_data));
          }
          if (v19_lead < 12) {
            float v2062_data = r6[8];
            float v2063_data = s3[8];
            float v2065_data = ir7[0];
            ir7[0] = (v2065_data + (v2062_data * v2063_data));
            float v2068_data = s3[20];
            float v2070_data = ir7[1];
            ir7[1] = (v2070_data + (v2062_data * v2068_data));
            float v2073_data = s3[33];
            float v2075_data = ir7[2];
            ir7[2] = (v2075_data + (v2062_data * v2073_data));
            float v2078_data = s3[45];
            float v2080_data = ir7[3];
            ir7[3] = (v2080_data + (v2062_data * v2078_data));
            float v2083_data = s3[57];
            float v2085_data = ir7[4];
            ir7[4] = (v2085_data + (v2062_data * v2083_data));
            float v2088_data = s3[70];
            float v2090_data = ir7[5];
            ir7[5] = (v2090_data + (v2062_data * v2088_data));
            float v2093_data = s3[82];
            float v2095_data = ir7[6];
            ir7[6] = (v2095_data + (v2062_data * v2093_data));
            float v2098_data = s3[94];
            float v2100_data = ir7[7];
            ir7[7] = (v2100_data + (v2062_data * v2098_data));
          }
          if (v19_lead < 12) {
            float v2106_data = r6[9];
            float v2107_data = s3[9];
            float v2109_data = ir7[0];
            ir7[0] = (v2109_data + (v2106_data * v2107_data));
            float v2112_data = s3[21];
            float v2114_data = ir7[1];
            ir7[1] = (v2114_data + (v2106_data * v2112_data));
            float v2117_data = s3[32];
            float v2119_data = ir7[2];
            ir7[2] = (v2119_data + (v2106_data * v2117_data));
            float v2122_data = s3[44];
            float v2124_data = ir7[3];
            ir7[3] = (v2124_data + (v2106_data * v2122_data));
            float v2127_data = s3[56];
            float v2129_data = ir7[4];
            ir7[4] = (v2129_data + (v2106_data * v2127_data));
            float v2132_data = s3[71];
            float v2134_data = ir7[5];
            ir7[5] = (v2134_data + (v2106_data * v2132_data));
            float v2137_data = s3[83];
            float v2139_data = ir7[6];
            ir7[6] = (v2139_data + (v2106_data * v2137_data));
            float v2142_data = s3[95];
            float v2144_data = ir7[7];
            ir7[7] = (v2144_data + (v2106_data * v2142_data));
          }
          if (v19_lead < 12) {
            float v2150_data = r6[10];
            float v2151_data = s3[10];
            float v2153_data = ir7[0];
            ir7[0] = (v2153_data + (v2150_data * v2151_data));
            float v2156_data = s3[22];
            float v2158_data = ir7[1];
            ir7[1] = (v2158_data + (v2150_data * v2156_data));
            float v2161_data = s3[35];
            float v2163_data = ir7[2];
            ir7[2] = (v2163_data + (v2150_data * v2161_data));
            float v2166_data = s3[47];
            float v2168_data = ir7[3];
            ir7[3] = (v2168_data + (v2150_data * v2166_data));
            float v2171_data = s3[59];
            float v2173_data = ir7[4];
            ir7[4] = (v2173_data + (v2150_data * v2171_data));
            float v2176_data = s3[68];
            float v2178_data = ir7[5];
            ir7[5] = (v2178_data + (v2150_data * v2176_data));
            float v2181_data = s3[80];
            float v2183_data = ir7[6];
            ir7[6] = (v2183_data + (v2150_data * v2181_data));
            float v2186_data = s3[92];
            float v2188_data = ir7[7];
            ir7[7] = (v2188_data + (v2150_data * v2186_data));
          }
          if (v19_lead < 12) {
            float v2194_data = r6[11];
            float v2195_data = s3[11];
            float v2197_data = ir7[0];
            ir7[0] = (v2197_data + (v2194_data * v2195_data));
            float v2200_data = s3[23];
            float v2202_data = ir7[1];
            ir7[1] = (v2202_data + (v2194_data * v2200_data));
            float v2205_data = s3[34];
            float v2207_data = ir7[2];
            ir7[2] = (v2207_data + (v2194_data * v2205_data));
            float v2210_data = s3[46];
            float v2212_data = ir7[3];
            ir7[3] = (v2212_data + (v2194_data * v2210_data));
            float v2215_data = s3[58];
            float v2217_data = ir7[4];
            ir7[4] = (v2217_data + (v2194_data * v2215_data));
            float v2220_data = s3[69];
            float v2222_data = ir7[5];
            ir7[5] = (v2222_data + (v2194_data * v2220_data));
            float v2225_data = s3[81];
            float v2227_data = ir7[6];
            ir7[6] = (v2227_data + (v2194_data * v2225_data));
            float v2230_data = s3[93];
            float v2232_data = ir7[7];
            ir7[7] = (v2232_data + (v2194_data * v2230_data));
          }
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v2238_n1 = 0; v2238_n1 < 8; ++v2238_n1) {
              float v2240_data = ir7[v2238_n1];
              float v2242_data = r5[v2238_n1];
              r7[v2238_n1] = (v2242_data + v2240_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v19_lead < 12) {
            #pragma unroll
            for (int32_t v2249_i1 = 0; v2249_i1 < 8; ++v2249_i1) {
              float v2251_data = r7[v2249_i1];
              glb_m0[(v19_lead + (v2249_i1 * 12))] = v2251_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

