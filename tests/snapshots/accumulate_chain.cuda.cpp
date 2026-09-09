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
    // options: default
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
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
      float* __restrict__ s2 = &localShrMem0[0];
      float* __restrict__ s3 = &localShrMem0[0];
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
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              float v33_data = __ldcg(&glb_m1[(v23_lead + (v25_i1 * 12))]);
              r0[v25_i1] = v33_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
              float v49_data = __ldcg(&glb_m3[(v23_lead + (v41_i1 * 12))]);
              r2[v41_i1] = v49_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v23_lead < 12) {
            float v57_data = r0[0];
            float v58_data = s0[0];
            float v60_data = ir1[0];
            ir1[0] = (v60_data + (v57_data * v58_data));
            float v63_data = s0[12];
            float v65_data = ir1[1];
            ir1[1] = (v65_data + (v57_data * v63_data));
            float v68_data = s0[24];
            float v70_data = ir1[2];
            ir1[2] = (v70_data + (v57_data * v68_data));
            float v73_data = s0[36];
            float v75_data = ir1[3];
            ir1[3] = (v75_data + (v57_data * v73_data));
            float v78_data = s0[48];
            float v80_data = ir1[4];
            ir1[4] = (v80_data + (v57_data * v78_data));
            float v83_data = s0[60];
            float v85_data = ir1[5];
            ir1[5] = (v85_data + (v57_data * v83_data));
            float v88_data = s0[72];
            float v90_data = ir1[6];
            ir1[6] = (v90_data + (v57_data * v88_data));
            float v93_data = s0[84];
            float v95_data = ir1[7];
            ir1[7] = (v95_data + (v57_data * v93_data));
          }
          if (v23_lead < 12) {
            float v101_data = r0[1];
            float v102_data = s0[1];
            float v104_data = ir1[0];
            ir1[0] = (v104_data + (v101_data * v102_data));
            float v107_data = s0[13];
            float v109_data = ir1[1];
            ir1[1] = (v109_data + (v101_data * v107_data));
            float v112_data = s0[25];
            float v114_data = ir1[2];
            ir1[2] = (v114_data + (v101_data * v112_data));
            float v117_data = s0[37];
            float v119_data = ir1[3];
            ir1[3] = (v119_data + (v101_data * v117_data));
            float v122_data = s0[49];
            float v124_data = ir1[4];
            ir1[4] = (v124_data + (v101_data * v122_data));
            float v127_data = s0[61];
            float v129_data = ir1[5];
            ir1[5] = (v129_data + (v101_data * v127_data));
            float v132_data = s0[73];
            float v134_data = ir1[6];
            ir1[6] = (v134_data + (v101_data * v132_data));
            float v137_data = s0[85];
            float v139_data = ir1[7];
            ir1[7] = (v139_data + (v101_data * v137_data));
          }
          if (v23_lead < 12) {
            float v145_data = r0[2];
            float v146_data = s0[2];
            float v148_data = ir1[0];
            ir1[0] = (v148_data + (v145_data * v146_data));
            float v151_data = s0[14];
            float v153_data = ir1[1];
            ir1[1] = (v153_data + (v145_data * v151_data));
            float v156_data = s0[26];
            float v158_data = ir1[2];
            ir1[2] = (v158_data + (v145_data * v156_data));
            float v161_data = s0[38];
            float v163_data = ir1[3];
            ir1[3] = (v163_data + (v145_data * v161_data));
            float v166_data = s0[50];
            float v168_data = ir1[4];
            ir1[4] = (v168_data + (v145_data * v166_data));
            float v171_data = s0[62];
            float v173_data = ir1[5];
            ir1[5] = (v173_data + (v145_data * v171_data));
            float v176_data = s0[74];
            float v178_data = ir1[6];
            ir1[6] = (v178_data + (v145_data * v176_data));
            float v181_data = s0[86];
            float v183_data = ir1[7];
            ir1[7] = (v183_data + (v145_data * v181_data));
          }
          if (v23_lead < 12) {
            float v189_data = r0[3];
            float v190_data = s0[3];
            float v192_data = ir1[0];
            ir1[0] = (v192_data + (v189_data * v190_data));
            float v195_data = s0[15];
            float v197_data = ir1[1];
            ir1[1] = (v197_data + (v189_data * v195_data));
            float v200_data = s0[27];
            float v202_data = ir1[2];
            ir1[2] = (v202_data + (v189_data * v200_data));
            float v205_data = s0[39];
            float v207_data = ir1[3];
            ir1[3] = (v207_data + (v189_data * v205_data));
            float v210_data = s0[51];
            float v212_data = ir1[4];
            ir1[4] = (v212_data + (v189_data * v210_data));
            float v215_data = s0[63];
            float v217_data = ir1[5];
            ir1[5] = (v217_data + (v189_data * v215_data));
            float v220_data = s0[75];
            float v222_data = ir1[6];
            ir1[6] = (v222_data + (v189_data * v220_data));
            float v225_data = s0[87];
            float v227_data = ir1[7];
            ir1[7] = (v227_data + (v189_data * v225_data));
          }
          if (v23_lead < 12) {
            float v233_data = r0[4];
            float v234_data = s0[4];
            float v236_data = ir1[0];
            ir1[0] = (v236_data + (v233_data * v234_data));
            float v239_data = s0[16];
            float v241_data = ir1[1];
            ir1[1] = (v241_data + (v233_data * v239_data));
            float v244_data = s0[28];
            float v246_data = ir1[2];
            ir1[2] = (v246_data + (v233_data * v244_data));
            float v249_data = s0[40];
            float v251_data = ir1[3];
            ir1[3] = (v251_data + (v233_data * v249_data));
            float v254_data = s0[52];
            float v256_data = ir1[4];
            ir1[4] = (v256_data + (v233_data * v254_data));
            float v259_data = s0[64];
            float v261_data = ir1[5];
            ir1[5] = (v261_data + (v233_data * v259_data));
            float v264_data = s0[76];
            float v266_data = ir1[6];
            ir1[6] = (v266_data + (v233_data * v264_data));
            float v269_data = s0[88];
            float v271_data = ir1[7];
            ir1[7] = (v271_data + (v233_data * v269_data));
          }
          if (v23_lead < 12) {
            float v277_data = r0[5];
            float v278_data = s0[5];
            float v280_data = ir1[0];
            ir1[0] = (v280_data + (v277_data * v278_data));
            float v283_data = s0[17];
            float v285_data = ir1[1];
            ir1[1] = (v285_data + (v277_data * v283_data));
            float v288_data = s0[29];
            float v290_data = ir1[2];
            ir1[2] = (v290_data + (v277_data * v288_data));
            float v293_data = s0[41];
            float v295_data = ir1[3];
            ir1[3] = (v295_data + (v277_data * v293_data));
            float v298_data = s0[53];
            float v300_data = ir1[4];
            ir1[4] = (v300_data + (v277_data * v298_data));
            float v303_data = s0[65];
            float v305_data = ir1[5];
            ir1[5] = (v305_data + (v277_data * v303_data));
            float v308_data = s0[77];
            float v310_data = ir1[6];
            ir1[6] = (v310_data + (v277_data * v308_data));
            float v313_data = s0[89];
            float v315_data = ir1[7];
            ir1[7] = (v315_data + (v277_data * v313_data));
          }
          if (v23_lead < 12) {
            float v321_data = r0[6];
            float v322_data = s0[6];
            float v324_data = ir1[0];
            ir1[0] = (v324_data + (v321_data * v322_data));
            float v327_data = s0[18];
            float v329_data = ir1[1];
            ir1[1] = (v329_data + (v321_data * v327_data));
            float v332_data = s0[30];
            float v334_data = ir1[2];
            ir1[2] = (v334_data + (v321_data * v332_data));
            float v337_data = s0[42];
            float v339_data = ir1[3];
            ir1[3] = (v339_data + (v321_data * v337_data));
            float v342_data = s0[54];
            float v344_data = ir1[4];
            ir1[4] = (v344_data + (v321_data * v342_data));
            float v347_data = s0[66];
            float v349_data = ir1[5];
            ir1[5] = (v349_data + (v321_data * v347_data));
            float v352_data = s0[78];
            float v354_data = ir1[6];
            ir1[6] = (v354_data + (v321_data * v352_data));
            float v357_data = s0[90];
            float v359_data = ir1[7];
            ir1[7] = (v359_data + (v321_data * v357_data));
          }
          if (v23_lead < 12) {
            float v365_data = r0[7];
            float v366_data = s0[7];
            float v368_data = ir1[0];
            ir1[0] = (v368_data + (v365_data * v366_data));
            float v371_data = s0[19];
            float v373_data = ir1[1];
            ir1[1] = (v373_data + (v365_data * v371_data));
            float v376_data = s0[31];
            float v378_data = ir1[2];
            ir1[2] = (v378_data + (v365_data * v376_data));
            float v381_data = s0[43];
            float v383_data = ir1[3];
            ir1[3] = (v383_data + (v365_data * v381_data));
            float v386_data = s0[55];
            float v388_data = ir1[4];
            ir1[4] = (v388_data + (v365_data * v386_data));
            float v391_data = s0[67];
            float v393_data = ir1[5];
            ir1[5] = (v393_data + (v365_data * v391_data));
            float v396_data = s0[79];
            float v398_data = ir1[6];
            ir1[6] = (v398_data + (v365_data * v396_data));
            float v401_data = s0[91];
            float v403_data = ir1[7];
            ir1[7] = (v403_data + (v365_data * v401_data));
          }
          if (v23_lead < 12) {
            float v409_data = r0[8];
            float v410_data = s0[8];
            float v412_data = ir1[0];
            ir1[0] = (v412_data + (v409_data * v410_data));
            float v415_data = s0[20];
            float v417_data = ir1[1];
            ir1[1] = (v417_data + (v409_data * v415_data));
            float v420_data = s0[32];
            float v422_data = ir1[2];
            ir1[2] = (v422_data + (v409_data * v420_data));
            float v425_data = s0[44];
            float v427_data = ir1[3];
            ir1[3] = (v427_data + (v409_data * v425_data));
            float v430_data = s0[56];
            float v432_data = ir1[4];
            ir1[4] = (v432_data + (v409_data * v430_data));
            float v435_data = s0[68];
            float v437_data = ir1[5];
            ir1[5] = (v437_data + (v409_data * v435_data));
            float v440_data = s0[80];
            float v442_data = ir1[6];
            ir1[6] = (v442_data + (v409_data * v440_data));
            float v445_data = s0[92];
            float v447_data = ir1[7];
            ir1[7] = (v447_data + (v409_data * v445_data));
          }
          if (v23_lead < 12) {
            float v453_data = r0[9];
            float v454_data = s0[9];
            float v456_data = ir1[0];
            ir1[0] = (v456_data + (v453_data * v454_data));
            float v459_data = s0[21];
            float v461_data = ir1[1];
            ir1[1] = (v461_data + (v453_data * v459_data));
            float v464_data = s0[33];
            float v466_data = ir1[2];
            ir1[2] = (v466_data + (v453_data * v464_data));
            float v469_data = s0[45];
            float v471_data = ir1[3];
            ir1[3] = (v471_data + (v453_data * v469_data));
            float v474_data = s0[57];
            float v476_data = ir1[4];
            ir1[4] = (v476_data + (v453_data * v474_data));
            float v479_data = s0[69];
            float v481_data = ir1[5];
            ir1[5] = (v481_data + (v453_data * v479_data));
            float v484_data = s0[81];
            float v486_data = ir1[6];
            ir1[6] = (v486_data + (v453_data * v484_data));
            float v489_data = s0[93];
            float v491_data = ir1[7];
            ir1[7] = (v491_data + (v453_data * v489_data));
          }
          if (v23_lead < 12) {
            float v497_data = r0[10];
            float v498_data = s0[10];
            float v500_data = ir1[0];
            ir1[0] = (v500_data + (v497_data * v498_data));
            float v503_data = s0[22];
            float v505_data = ir1[1];
            ir1[1] = (v505_data + (v497_data * v503_data));
            float v508_data = s0[34];
            float v510_data = ir1[2];
            ir1[2] = (v510_data + (v497_data * v508_data));
            float v513_data = s0[46];
            float v515_data = ir1[3];
            ir1[3] = (v515_data + (v497_data * v513_data));
            float v518_data = s0[58];
            float v520_data = ir1[4];
            ir1[4] = (v520_data + (v497_data * v518_data));
            float v523_data = s0[70];
            float v525_data = ir1[5];
            ir1[5] = (v525_data + (v497_data * v523_data));
            float v528_data = s0[82];
            float v530_data = ir1[6];
            ir1[6] = (v530_data + (v497_data * v528_data));
            float v533_data = s0[94];
            float v535_data = ir1[7];
            ir1[7] = (v535_data + (v497_data * v533_data));
          }
          if (v23_lead < 12) {
            float v541_data = r0[11];
            float v542_data = s0[11];
            float v544_data = ir1[0];
            ir1[0] = (v544_data + (v541_data * v542_data));
            float v547_data = s0[23];
            float v549_data = ir1[1];
            ir1[1] = (v549_data + (v541_data * v547_data));
            float v552_data = s0[35];
            float v554_data = ir1[2];
            ir1[2] = (v554_data + (v541_data * v552_data));
            float v557_data = s0[47];
            float v559_data = ir1[3];
            ir1[3] = (v559_data + (v541_data * v557_data));
            float v562_data = s0[59];
            float v564_data = ir1[4];
            ir1[4] = (v564_data + (v541_data * v562_data));
            float v567_data = s0[71];
            float v569_data = ir1[5];
            ir1[5] = (v569_data + (v541_data * v567_data));
            float v572_data = s0[83];
            float v574_data = ir1[6];
            ir1[6] = (v574_data + (v541_data * v572_data));
            float v577_data = s0[95];
            float v579_data = ir1[7];
            ir1[7] = (v579_data + (v541_data * v577_data));
          }
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v585_n1 = 0; v585_n1 < 8; ++v585_n1) {
              float v587_data = ir1[v585_n1];
              r1[v585_n1] = v587_data;
            }
          }
          __syncwarp();
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v595_i1 = 0; v595_i1 < 12; ++v595_i1) {
              float v603_data = __ldcg(&glb_m5[(v23_lead + (v595_i1 * 12))]);
              r4[v595_i1] = v603_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v23_lead < 12) {
            float v611_data = r2[0];
            float v612_data = s1[0];
            float v614_data = ir3[0];
            ir3[0] = (v614_data + (v611_data * v612_data));
            float v617_data = s1[12];
            float v619_data = ir3[1];
            ir3[1] = (v619_data + (v611_data * v617_data));
            float v622_data = s1[24];
            float v624_data = ir3[2];
            ir3[2] = (v624_data + (v611_data * v622_data));
            float v627_data = s1[36];
            float v629_data = ir3[3];
            ir3[3] = (v629_data + (v611_data * v627_data));
            float v632_data = s1[48];
            float v634_data = ir3[4];
            ir3[4] = (v634_data + (v611_data * v632_data));
            float v637_data = s1[60];
            float v639_data = ir3[5];
            ir3[5] = (v639_data + (v611_data * v637_data));
            float v642_data = s1[72];
            float v644_data = ir3[6];
            ir3[6] = (v644_data + (v611_data * v642_data));
            float v647_data = s1[84];
            float v649_data = ir3[7];
            ir3[7] = (v649_data + (v611_data * v647_data));
          }
          if (v23_lead < 12) {
            float v655_data = r2[1];
            float v656_data = s1[1];
            float v658_data = ir3[0];
            ir3[0] = (v658_data + (v655_data * v656_data));
            float v661_data = s1[13];
            float v663_data = ir3[1];
            ir3[1] = (v663_data + (v655_data * v661_data));
            float v666_data = s1[25];
            float v668_data = ir3[2];
            ir3[2] = (v668_data + (v655_data * v666_data));
            float v671_data = s1[37];
            float v673_data = ir3[3];
            ir3[3] = (v673_data + (v655_data * v671_data));
            float v676_data = s1[49];
            float v678_data = ir3[4];
            ir3[4] = (v678_data + (v655_data * v676_data));
            float v681_data = s1[61];
            float v683_data = ir3[5];
            ir3[5] = (v683_data + (v655_data * v681_data));
            float v686_data = s1[73];
            float v688_data = ir3[6];
            ir3[6] = (v688_data + (v655_data * v686_data));
            float v691_data = s1[85];
            float v693_data = ir3[7];
            ir3[7] = (v693_data + (v655_data * v691_data));
          }
          if (v23_lead < 12) {
            float v699_data = r2[2];
            float v700_data = s1[2];
            float v702_data = ir3[0];
            ir3[0] = (v702_data + (v699_data * v700_data));
            float v705_data = s1[14];
            float v707_data = ir3[1];
            ir3[1] = (v707_data + (v699_data * v705_data));
            float v710_data = s1[26];
            float v712_data = ir3[2];
            ir3[2] = (v712_data + (v699_data * v710_data));
            float v715_data = s1[38];
            float v717_data = ir3[3];
            ir3[3] = (v717_data + (v699_data * v715_data));
            float v720_data = s1[50];
            float v722_data = ir3[4];
            ir3[4] = (v722_data + (v699_data * v720_data));
            float v725_data = s1[62];
            float v727_data = ir3[5];
            ir3[5] = (v727_data + (v699_data * v725_data));
            float v730_data = s1[74];
            float v732_data = ir3[6];
            ir3[6] = (v732_data + (v699_data * v730_data));
            float v735_data = s1[86];
            float v737_data = ir3[7];
            ir3[7] = (v737_data + (v699_data * v735_data));
          }
          if (v23_lead < 12) {
            float v743_data = r2[3];
            float v744_data = s1[3];
            float v746_data = ir3[0];
            ir3[0] = (v746_data + (v743_data * v744_data));
            float v749_data = s1[15];
            float v751_data = ir3[1];
            ir3[1] = (v751_data + (v743_data * v749_data));
            float v754_data = s1[27];
            float v756_data = ir3[2];
            ir3[2] = (v756_data + (v743_data * v754_data));
            float v759_data = s1[39];
            float v761_data = ir3[3];
            ir3[3] = (v761_data + (v743_data * v759_data));
            float v764_data = s1[51];
            float v766_data = ir3[4];
            ir3[4] = (v766_data + (v743_data * v764_data));
            float v769_data = s1[63];
            float v771_data = ir3[5];
            ir3[5] = (v771_data + (v743_data * v769_data));
            float v774_data = s1[75];
            float v776_data = ir3[6];
            ir3[6] = (v776_data + (v743_data * v774_data));
            float v779_data = s1[87];
            float v781_data = ir3[7];
            ir3[7] = (v781_data + (v743_data * v779_data));
          }
          if (v23_lead < 12) {
            float v787_data = r2[4];
            float v788_data = s1[4];
            float v790_data = ir3[0];
            ir3[0] = (v790_data + (v787_data * v788_data));
            float v793_data = s1[16];
            float v795_data = ir3[1];
            ir3[1] = (v795_data + (v787_data * v793_data));
            float v798_data = s1[28];
            float v800_data = ir3[2];
            ir3[2] = (v800_data + (v787_data * v798_data));
            float v803_data = s1[40];
            float v805_data = ir3[3];
            ir3[3] = (v805_data + (v787_data * v803_data));
            float v808_data = s1[52];
            float v810_data = ir3[4];
            ir3[4] = (v810_data + (v787_data * v808_data));
            float v813_data = s1[64];
            float v815_data = ir3[5];
            ir3[5] = (v815_data + (v787_data * v813_data));
            float v818_data = s1[76];
            float v820_data = ir3[6];
            ir3[6] = (v820_data + (v787_data * v818_data));
            float v823_data = s1[88];
            float v825_data = ir3[7];
            ir3[7] = (v825_data + (v787_data * v823_data));
          }
          if (v23_lead < 12) {
            float v831_data = r2[5];
            float v832_data = s1[5];
            float v834_data = ir3[0];
            ir3[0] = (v834_data + (v831_data * v832_data));
            float v837_data = s1[17];
            float v839_data = ir3[1];
            ir3[1] = (v839_data + (v831_data * v837_data));
            float v842_data = s1[29];
            float v844_data = ir3[2];
            ir3[2] = (v844_data + (v831_data * v842_data));
            float v847_data = s1[41];
            float v849_data = ir3[3];
            ir3[3] = (v849_data + (v831_data * v847_data));
            float v852_data = s1[53];
            float v854_data = ir3[4];
            ir3[4] = (v854_data + (v831_data * v852_data));
            float v857_data = s1[65];
            float v859_data = ir3[5];
            ir3[5] = (v859_data + (v831_data * v857_data));
            float v862_data = s1[77];
            float v864_data = ir3[6];
            ir3[6] = (v864_data + (v831_data * v862_data));
            float v867_data = s1[89];
            float v869_data = ir3[7];
            ir3[7] = (v869_data + (v831_data * v867_data));
          }
          if (v23_lead < 12) {
            float v875_data = r2[6];
            float v876_data = s1[6];
            float v878_data = ir3[0];
            ir3[0] = (v878_data + (v875_data * v876_data));
            float v881_data = s1[18];
            float v883_data = ir3[1];
            ir3[1] = (v883_data + (v875_data * v881_data));
            float v886_data = s1[30];
            float v888_data = ir3[2];
            ir3[2] = (v888_data + (v875_data * v886_data));
            float v891_data = s1[42];
            float v893_data = ir3[3];
            ir3[3] = (v893_data + (v875_data * v891_data));
            float v896_data = s1[54];
            float v898_data = ir3[4];
            ir3[4] = (v898_data + (v875_data * v896_data));
            float v901_data = s1[66];
            float v903_data = ir3[5];
            ir3[5] = (v903_data + (v875_data * v901_data));
            float v906_data = s1[78];
            float v908_data = ir3[6];
            ir3[6] = (v908_data + (v875_data * v906_data));
            float v911_data = s1[90];
            float v913_data = ir3[7];
            ir3[7] = (v913_data + (v875_data * v911_data));
          }
          if (v23_lead < 12) {
            float v919_data = r2[7];
            float v920_data = s1[7];
            float v922_data = ir3[0];
            ir3[0] = (v922_data + (v919_data * v920_data));
            float v925_data = s1[19];
            float v927_data = ir3[1];
            ir3[1] = (v927_data + (v919_data * v925_data));
            float v930_data = s1[31];
            float v932_data = ir3[2];
            ir3[2] = (v932_data + (v919_data * v930_data));
            float v935_data = s1[43];
            float v937_data = ir3[3];
            ir3[3] = (v937_data + (v919_data * v935_data));
            float v940_data = s1[55];
            float v942_data = ir3[4];
            ir3[4] = (v942_data + (v919_data * v940_data));
            float v945_data = s1[67];
            float v947_data = ir3[5];
            ir3[5] = (v947_data + (v919_data * v945_data));
            float v950_data = s1[79];
            float v952_data = ir3[6];
            ir3[6] = (v952_data + (v919_data * v950_data));
            float v955_data = s1[91];
            float v957_data = ir3[7];
            ir3[7] = (v957_data + (v919_data * v955_data));
          }
          if (v23_lead < 12) {
            float v963_data = r2[8];
            float v964_data = s1[8];
            float v966_data = ir3[0];
            ir3[0] = (v966_data + (v963_data * v964_data));
            float v969_data = s1[20];
            float v971_data = ir3[1];
            ir3[1] = (v971_data + (v963_data * v969_data));
            float v974_data = s1[32];
            float v976_data = ir3[2];
            ir3[2] = (v976_data + (v963_data * v974_data));
            float v979_data = s1[44];
            float v981_data = ir3[3];
            ir3[3] = (v981_data + (v963_data * v979_data));
            float v984_data = s1[56];
            float v986_data = ir3[4];
            ir3[4] = (v986_data + (v963_data * v984_data));
            float v989_data = s1[68];
            float v991_data = ir3[5];
            ir3[5] = (v991_data + (v963_data * v989_data));
            float v994_data = s1[80];
            float v996_data = ir3[6];
            ir3[6] = (v996_data + (v963_data * v994_data));
            float v999_data = s1[92];
            float v1001_data = ir3[7];
            ir3[7] = (v1001_data + (v963_data * v999_data));
          }
          if (v23_lead < 12) {
            float v1007_data = r2[9];
            float v1008_data = s1[9];
            float v1010_data = ir3[0];
            ir3[0] = (v1010_data + (v1007_data * v1008_data));
            float v1013_data = s1[21];
            float v1015_data = ir3[1];
            ir3[1] = (v1015_data + (v1007_data * v1013_data));
            float v1018_data = s1[33];
            float v1020_data = ir3[2];
            ir3[2] = (v1020_data + (v1007_data * v1018_data));
            float v1023_data = s1[45];
            float v1025_data = ir3[3];
            ir3[3] = (v1025_data + (v1007_data * v1023_data));
            float v1028_data = s1[57];
            float v1030_data = ir3[4];
            ir3[4] = (v1030_data + (v1007_data * v1028_data));
            float v1033_data = s1[69];
            float v1035_data = ir3[5];
            ir3[5] = (v1035_data + (v1007_data * v1033_data));
            float v1038_data = s1[81];
            float v1040_data = ir3[6];
            ir3[6] = (v1040_data + (v1007_data * v1038_data));
            float v1043_data = s1[93];
            float v1045_data = ir3[7];
            ir3[7] = (v1045_data + (v1007_data * v1043_data));
          }
          if (v23_lead < 12) {
            float v1051_data = r2[10];
            float v1052_data = s1[10];
            float v1054_data = ir3[0];
            ir3[0] = (v1054_data + (v1051_data * v1052_data));
            float v1057_data = s1[22];
            float v1059_data = ir3[1];
            ir3[1] = (v1059_data + (v1051_data * v1057_data));
            float v1062_data = s1[34];
            float v1064_data = ir3[2];
            ir3[2] = (v1064_data + (v1051_data * v1062_data));
            float v1067_data = s1[46];
            float v1069_data = ir3[3];
            ir3[3] = (v1069_data + (v1051_data * v1067_data));
            float v1072_data = s1[58];
            float v1074_data = ir3[4];
            ir3[4] = (v1074_data + (v1051_data * v1072_data));
            float v1077_data = s1[70];
            float v1079_data = ir3[5];
            ir3[5] = (v1079_data + (v1051_data * v1077_data));
            float v1082_data = s1[82];
            float v1084_data = ir3[6];
            ir3[6] = (v1084_data + (v1051_data * v1082_data));
            float v1087_data = s1[94];
            float v1089_data = ir3[7];
            ir3[7] = (v1089_data + (v1051_data * v1087_data));
          }
          if (v23_lead < 12) {
            float v1095_data = r2[11];
            float v1096_data = s1[11];
            float v1098_data = ir3[0];
            ir3[0] = (v1098_data + (v1095_data * v1096_data));
            float v1101_data = s1[23];
            float v1103_data = ir3[1];
            ir3[1] = (v1103_data + (v1095_data * v1101_data));
            float v1106_data = s1[35];
            float v1108_data = ir3[2];
            ir3[2] = (v1108_data + (v1095_data * v1106_data));
            float v1111_data = s1[47];
            float v1113_data = ir3[3];
            ir3[3] = (v1113_data + (v1095_data * v1111_data));
            float v1116_data = s1[59];
            float v1118_data = ir3[4];
            ir3[4] = (v1118_data + (v1095_data * v1116_data));
            float v1121_data = s1[71];
            float v1123_data = ir3[5];
            ir3[5] = (v1123_data + (v1095_data * v1121_data));
            float v1126_data = s1[83];
            float v1128_data = ir3[6];
            ir3[6] = (v1128_data + (v1095_data * v1126_data));
            float v1131_data = s1[95];
            float v1133_data = ir3[7];
            ir3[7] = (v1133_data + (v1095_data * v1131_data));
          }
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v1139_n1 = 0; v1139_n1 < 8; ++v1139_n1) {
              float v1141_data = ir3[v1139_n1];
              float v1143_data = r1[v1139_n1];
              r3[v1139_n1] = (v1143_data + v1141_data);
            }
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m6[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v1152_i1 = 0; v1152_i1 < 12; ++v1152_i1) {
              float v1160_data = __ldcg(&glb_m7[(v23_lead + (v1152_i1 * 12))]);
              r6[v1152_i1] = v1160_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v23_lead < 12) {
            float v1168_data = r4[0];
            float v1169_data = s2[0];
            float v1171_data = ir5[0];
            ir5[0] = (v1171_data + (v1168_data * v1169_data));
            float v1174_data = s2[12];
            float v1176_data = ir5[1];
            ir5[1] = (v1176_data + (v1168_data * v1174_data));
            float v1179_data = s2[24];
            float v1181_data = ir5[2];
            ir5[2] = (v1181_data + (v1168_data * v1179_data));
            float v1184_data = s2[36];
            float v1186_data = ir5[3];
            ir5[3] = (v1186_data + (v1168_data * v1184_data));
            float v1189_data = s2[48];
            float v1191_data = ir5[4];
            ir5[4] = (v1191_data + (v1168_data * v1189_data));
            float v1194_data = s2[60];
            float v1196_data = ir5[5];
            ir5[5] = (v1196_data + (v1168_data * v1194_data));
            float v1199_data = s2[72];
            float v1201_data = ir5[6];
            ir5[6] = (v1201_data + (v1168_data * v1199_data));
            float v1204_data = s2[84];
            float v1206_data = ir5[7];
            ir5[7] = (v1206_data + (v1168_data * v1204_data));
          }
          if (v23_lead < 12) {
            float v1212_data = r4[1];
            float v1213_data = s2[1];
            float v1215_data = ir5[0];
            ir5[0] = (v1215_data + (v1212_data * v1213_data));
            float v1218_data = s2[13];
            float v1220_data = ir5[1];
            ir5[1] = (v1220_data + (v1212_data * v1218_data));
            float v1223_data = s2[25];
            float v1225_data = ir5[2];
            ir5[2] = (v1225_data + (v1212_data * v1223_data));
            float v1228_data = s2[37];
            float v1230_data = ir5[3];
            ir5[3] = (v1230_data + (v1212_data * v1228_data));
            float v1233_data = s2[49];
            float v1235_data = ir5[4];
            ir5[4] = (v1235_data + (v1212_data * v1233_data));
            float v1238_data = s2[61];
            float v1240_data = ir5[5];
            ir5[5] = (v1240_data + (v1212_data * v1238_data));
            float v1243_data = s2[73];
            float v1245_data = ir5[6];
            ir5[6] = (v1245_data + (v1212_data * v1243_data));
            float v1248_data = s2[85];
            float v1250_data = ir5[7];
            ir5[7] = (v1250_data + (v1212_data * v1248_data));
          }
          if (v23_lead < 12) {
            float v1256_data = r4[2];
            float v1257_data = s2[2];
            float v1259_data = ir5[0];
            ir5[0] = (v1259_data + (v1256_data * v1257_data));
            float v1262_data = s2[14];
            float v1264_data = ir5[1];
            ir5[1] = (v1264_data + (v1256_data * v1262_data));
            float v1267_data = s2[26];
            float v1269_data = ir5[2];
            ir5[2] = (v1269_data + (v1256_data * v1267_data));
            float v1272_data = s2[38];
            float v1274_data = ir5[3];
            ir5[3] = (v1274_data + (v1256_data * v1272_data));
            float v1277_data = s2[50];
            float v1279_data = ir5[4];
            ir5[4] = (v1279_data + (v1256_data * v1277_data));
            float v1282_data = s2[62];
            float v1284_data = ir5[5];
            ir5[5] = (v1284_data + (v1256_data * v1282_data));
            float v1287_data = s2[74];
            float v1289_data = ir5[6];
            ir5[6] = (v1289_data + (v1256_data * v1287_data));
            float v1292_data = s2[86];
            float v1294_data = ir5[7];
            ir5[7] = (v1294_data + (v1256_data * v1292_data));
          }
          if (v23_lead < 12) {
            float v1300_data = r4[3];
            float v1301_data = s2[3];
            float v1303_data = ir5[0];
            ir5[0] = (v1303_data + (v1300_data * v1301_data));
            float v1306_data = s2[15];
            float v1308_data = ir5[1];
            ir5[1] = (v1308_data + (v1300_data * v1306_data));
            float v1311_data = s2[27];
            float v1313_data = ir5[2];
            ir5[2] = (v1313_data + (v1300_data * v1311_data));
            float v1316_data = s2[39];
            float v1318_data = ir5[3];
            ir5[3] = (v1318_data + (v1300_data * v1316_data));
            float v1321_data = s2[51];
            float v1323_data = ir5[4];
            ir5[4] = (v1323_data + (v1300_data * v1321_data));
            float v1326_data = s2[63];
            float v1328_data = ir5[5];
            ir5[5] = (v1328_data + (v1300_data * v1326_data));
            float v1331_data = s2[75];
            float v1333_data = ir5[6];
            ir5[6] = (v1333_data + (v1300_data * v1331_data));
            float v1336_data = s2[87];
            float v1338_data = ir5[7];
            ir5[7] = (v1338_data + (v1300_data * v1336_data));
          }
          if (v23_lead < 12) {
            float v1344_data = r4[4];
            float v1345_data = s2[4];
            float v1347_data = ir5[0];
            ir5[0] = (v1347_data + (v1344_data * v1345_data));
            float v1350_data = s2[16];
            float v1352_data = ir5[1];
            ir5[1] = (v1352_data + (v1344_data * v1350_data));
            float v1355_data = s2[28];
            float v1357_data = ir5[2];
            ir5[2] = (v1357_data + (v1344_data * v1355_data));
            float v1360_data = s2[40];
            float v1362_data = ir5[3];
            ir5[3] = (v1362_data + (v1344_data * v1360_data));
            float v1365_data = s2[52];
            float v1367_data = ir5[4];
            ir5[4] = (v1367_data + (v1344_data * v1365_data));
            float v1370_data = s2[64];
            float v1372_data = ir5[5];
            ir5[5] = (v1372_data + (v1344_data * v1370_data));
            float v1375_data = s2[76];
            float v1377_data = ir5[6];
            ir5[6] = (v1377_data + (v1344_data * v1375_data));
            float v1380_data = s2[88];
            float v1382_data = ir5[7];
            ir5[7] = (v1382_data + (v1344_data * v1380_data));
          }
          if (v23_lead < 12) {
            float v1388_data = r4[5];
            float v1389_data = s2[5];
            float v1391_data = ir5[0];
            ir5[0] = (v1391_data + (v1388_data * v1389_data));
            float v1394_data = s2[17];
            float v1396_data = ir5[1];
            ir5[1] = (v1396_data + (v1388_data * v1394_data));
            float v1399_data = s2[29];
            float v1401_data = ir5[2];
            ir5[2] = (v1401_data + (v1388_data * v1399_data));
            float v1404_data = s2[41];
            float v1406_data = ir5[3];
            ir5[3] = (v1406_data + (v1388_data * v1404_data));
            float v1409_data = s2[53];
            float v1411_data = ir5[4];
            ir5[4] = (v1411_data + (v1388_data * v1409_data));
            float v1414_data = s2[65];
            float v1416_data = ir5[5];
            ir5[5] = (v1416_data + (v1388_data * v1414_data));
            float v1419_data = s2[77];
            float v1421_data = ir5[6];
            ir5[6] = (v1421_data + (v1388_data * v1419_data));
            float v1424_data = s2[89];
            float v1426_data = ir5[7];
            ir5[7] = (v1426_data + (v1388_data * v1424_data));
          }
          if (v23_lead < 12) {
            float v1432_data = r4[6];
            float v1433_data = s2[6];
            float v1435_data = ir5[0];
            ir5[0] = (v1435_data + (v1432_data * v1433_data));
            float v1438_data = s2[18];
            float v1440_data = ir5[1];
            ir5[1] = (v1440_data + (v1432_data * v1438_data));
            float v1443_data = s2[30];
            float v1445_data = ir5[2];
            ir5[2] = (v1445_data + (v1432_data * v1443_data));
            float v1448_data = s2[42];
            float v1450_data = ir5[3];
            ir5[3] = (v1450_data + (v1432_data * v1448_data));
            float v1453_data = s2[54];
            float v1455_data = ir5[4];
            ir5[4] = (v1455_data + (v1432_data * v1453_data));
            float v1458_data = s2[66];
            float v1460_data = ir5[5];
            ir5[5] = (v1460_data + (v1432_data * v1458_data));
            float v1463_data = s2[78];
            float v1465_data = ir5[6];
            ir5[6] = (v1465_data + (v1432_data * v1463_data));
            float v1468_data = s2[90];
            float v1470_data = ir5[7];
            ir5[7] = (v1470_data + (v1432_data * v1468_data));
          }
          if (v23_lead < 12) {
            float v1476_data = r4[7];
            float v1477_data = s2[7];
            float v1479_data = ir5[0];
            ir5[0] = (v1479_data + (v1476_data * v1477_data));
            float v1482_data = s2[19];
            float v1484_data = ir5[1];
            ir5[1] = (v1484_data + (v1476_data * v1482_data));
            float v1487_data = s2[31];
            float v1489_data = ir5[2];
            ir5[2] = (v1489_data + (v1476_data * v1487_data));
            float v1492_data = s2[43];
            float v1494_data = ir5[3];
            ir5[3] = (v1494_data + (v1476_data * v1492_data));
            float v1497_data = s2[55];
            float v1499_data = ir5[4];
            ir5[4] = (v1499_data + (v1476_data * v1497_data));
            float v1502_data = s2[67];
            float v1504_data = ir5[5];
            ir5[5] = (v1504_data + (v1476_data * v1502_data));
            float v1507_data = s2[79];
            float v1509_data = ir5[6];
            ir5[6] = (v1509_data + (v1476_data * v1507_data));
            float v1512_data = s2[91];
            float v1514_data = ir5[7];
            ir5[7] = (v1514_data + (v1476_data * v1512_data));
          }
          if (v23_lead < 12) {
            float v1520_data = r4[8];
            float v1521_data = s2[8];
            float v1523_data = ir5[0];
            ir5[0] = (v1523_data + (v1520_data * v1521_data));
            float v1526_data = s2[20];
            float v1528_data = ir5[1];
            ir5[1] = (v1528_data + (v1520_data * v1526_data));
            float v1531_data = s2[32];
            float v1533_data = ir5[2];
            ir5[2] = (v1533_data + (v1520_data * v1531_data));
            float v1536_data = s2[44];
            float v1538_data = ir5[3];
            ir5[3] = (v1538_data + (v1520_data * v1536_data));
            float v1541_data = s2[56];
            float v1543_data = ir5[4];
            ir5[4] = (v1543_data + (v1520_data * v1541_data));
            float v1546_data = s2[68];
            float v1548_data = ir5[5];
            ir5[5] = (v1548_data + (v1520_data * v1546_data));
            float v1551_data = s2[80];
            float v1553_data = ir5[6];
            ir5[6] = (v1553_data + (v1520_data * v1551_data));
            float v1556_data = s2[92];
            float v1558_data = ir5[7];
            ir5[7] = (v1558_data + (v1520_data * v1556_data));
          }
          if (v23_lead < 12) {
            float v1564_data = r4[9];
            float v1565_data = s2[9];
            float v1567_data = ir5[0];
            ir5[0] = (v1567_data + (v1564_data * v1565_data));
            float v1570_data = s2[21];
            float v1572_data = ir5[1];
            ir5[1] = (v1572_data + (v1564_data * v1570_data));
            float v1575_data = s2[33];
            float v1577_data = ir5[2];
            ir5[2] = (v1577_data + (v1564_data * v1575_data));
            float v1580_data = s2[45];
            float v1582_data = ir5[3];
            ir5[3] = (v1582_data + (v1564_data * v1580_data));
            float v1585_data = s2[57];
            float v1587_data = ir5[4];
            ir5[4] = (v1587_data + (v1564_data * v1585_data));
            float v1590_data = s2[69];
            float v1592_data = ir5[5];
            ir5[5] = (v1592_data + (v1564_data * v1590_data));
            float v1595_data = s2[81];
            float v1597_data = ir5[6];
            ir5[6] = (v1597_data + (v1564_data * v1595_data));
            float v1600_data = s2[93];
            float v1602_data = ir5[7];
            ir5[7] = (v1602_data + (v1564_data * v1600_data));
          }
          if (v23_lead < 12) {
            float v1608_data = r4[10];
            float v1609_data = s2[10];
            float v1611_data = ir5[0];
            ir5[0] = (v1611_data + (v1608_data * v1609_data));
            float v1614_data = s2[22];
            float v1616_data = ir5[1];
            ir5[1] = (v1616_data + (v1608_data * v1614_data));
            float v1619_data = s2[34];
            float v1621_data = ir5[2];
            ir5[2] = (v1621_data + (v1608_data * v1619_data));
            float v1624_data = s2[46];
            float v1626_data = ir5[3];
            ir5[3] = (v1626_data + (v1608_data * v1624_data));
            float v1629_data = s2[58];
            float v1631_data = ir5[4];
            ir5[4] = (v1631_data + (v1608_data * v1629_data));
            float v1634_data = s2[70];
            float v1636_data = ir5[5];
            ir5[5] = (v1636_data + (v1608_data * v1634_data));
            float v1639_data = s2[82];
            float v1641_data = ir5[6];
            ir5[6] = (v1641_data + (v1608_data * v1639_data));
            float v1644_data = s2[94];
            float v1646_data = ir5[7];
            ir5[7] = (v1646_data + (v1608_data * v1644_data));
          }
          if (v23_lead < 12) {
            float v1652_data = r4[11];
            float v1653_data = s2[11];
            float v1655_data = ir5[0];
            ir5[0] = (v1655_data + (v1652_data * v1653_data));
            float v1658_data = s2[23];
            float v1660_data = ir5[1];
            ir5[1] = (v1660_data + (v1652_data * v1658_data));
            float v1663_data = s2[35];
            float v1665_data = ir5[2];
            ir5[2] = (v1665_data + (v1652_data * v1663_data));
            float v1668_data = s2[47];
            float v1670_data = ir5[3];
            ir5[3] = (v1670_data + (v1652_data * v1668_data));
            float v1673_data = s2[59];
            float v1675_data = ir5[4];
            ir5[4] = (v1675_data + (v1652_data * v1673_data));
            float v1678_data = s2[71];
            float v1680_data = ir5[5];
            ir5[5] = (v1680_data + (v1652_data * v1678_data));
            float v1683_data = s2[83];
            float v1685_data = ir5[6];
            ir5[6] = (v1685_data + (v1652_data * v1683_data));
            float v1688_data = s2[95];
            float v1690_data = ir5[7];
            ir5[7] = (v1690_data + (v1652_data * v1688_data));
          }
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v1696_n1 = 0; v1696_n1 < 8; ++v1696_n1) {
              float v1698_data = ir5[v1696_n1];
              float v1700_data = r3[v1696_n1];
              r5[v1696_n1] = (v1700_data + v1698_data);
            }
          }
          __syncwarp();
          // s3 = load{g>s}(glb_m8[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v23_lead < 12) {
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
            float v1726_data = s3[36];
            float v1728_data = ir7[3];
            ir7[3] = (v1728_data + (v1710_data * v1726_data));
            float v1731_data = s3[48];
            float v1733_data = ir7[4];
            ir7[4] = (v1733_data + (v1710_data * v1731_data));
            float v1736_data = s3[60];
            float v1738_data = ir7[5];
            ir7[5] = (v1738_data + (v1710_data * v1736_data));
            float v1741_data = s3[72];
            float v1743_data = ir7[6];
            ir7[6] = (v1743_data + (v1710_data * v1741_data));
            float v1746_data = s3[84];
            float v1748_data = ir7[7];
            ir7[7] = (v1748_data + (v1710_data * v1746_data));
          }
          if (v23_lead < 12) {
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
            float v1770_data = s3[37];
            float v1772_data = ir7[3];
            ir7[3] = (v1772_data + (v1754_data * v1770_data));
            float v1775_data = s3[49];
            float v1777_data = ir7[4];
            ir7[4] = (v1777_data + (v1754_data * v1775_data));
            float v1780_data = s3[61];
            float v1782_data = ir7[5];
            ir7[5] = (v1782_data + (v1754_data * v1780_data));
            float v1785_data = s3[73];
            float v1787_data = ir7[6];
            ir7[6] = (v1787_data + (v1754_data * v1785_data));
            float v1790_data = s3[85];
            float v1792_data = ir7[7];
            ir7[7] = (v1792_data + (v1754_data * v1790_data));
          }
          if (v23_lead < 12) {
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
            float v1814_data = s3[38];
            float v1816_data = ir7[3];
            ir7[3] = (v1816_data + (v1798_data * v1814_data));
            float v1819_data = s3[50];
            float v1821_data = ir7[4];
            ir7[4] = (v1821_data + (v1798_data * v1819_data));
            float v1824_data = s3[62];
            float v1826_data = ir7[5];
            ir7[5] = (v1826_data + (v1798_data * v1824_data));
            float v1829_data = s3[74];
            float v1831_data = ir7[6];
            ir7[6] = (v1831_data + (v1798_data * v1829_data));
            float v1834_data = s3[86];
            float v1836_data = ir7[7];
            ir7[7] = (v1836_data + (v1798_data * v1834_data));
          }
          if (v23_lead < 12) {
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
            float v1858_data = s3[39];
            float v1860_data = ir7[3];
            ir7[3] = (v1860_data + (v1842_data * v1858_data));
            float v1863_data = s3[51];
            float v1865_data = ir7[4];
            ir7[4] = (v1865_data + (v1842_data * v1863_data));
            float v1868_data = s3[63];
            float v1870_data = ir7[5];
            ir7[5] = (v1870_data + (v1842_data * v1868_data));
            float v1873_data = s3[75];
            float v1875_data = ir7[6];
            ir7[6] = (v1875_data + (v1842_data * v1873_data));
            float v1878_data = s3[87];
            float v1880_data = ir7[7];
            ir7[7] = (v1880_data + (v1842_data * v1878_data));
          }
          if (v23_lead < 12) {
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
            float v1902_data = s3[40];
            float v1904_data = ir7[3];
            ir7[3] = (v1904_data + (v1886_data * v1902_data));
            float v1907_data = s3[52];
            float v1909_data = ir7[4];
            ir7[4] = (v1909_data + (v1886_data * v1907_data));
            float v1912_data = s3[64];
            float v1914_data = ir7[5];
            ir7[5] = (v1914_data + (v1886_data * v1912_data));
            float v1917_data = s3[76];
            float v1919_data = ir7[6];
            ir7[6] = (v1919_data + (v1886_data * v1917_data));
            float v1922_data = s3[88];
            float v1924_data = ir7[7];
            ir7[7] = (v1924_data + (v1886_data * v1922_data));
          }
          if (v23_lead < 12) {
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
            float v1946_data = s3[41];
            float v1948_data = ir7[3];
            ir7[3] = (v1948_data + (v1930_data * v1946_data));
            float v1951_data = s3[53];
            float v1953_data = ir7[4];
            ir7[4] = (v1953_data + (v1930_data * v1951_data));
            float v1956_data = s3[65];
            float v1958_data = ir7[5];
            ir7[5] = (v1958_data + (v1930_data * v1956_data));
            float v1961_data = s3[77];
            float v1963_data = ir7[6];
            ir7[6] = (v1963_data + (v1930_data * v1961_data));
            float v1966_data = s3[89];
            float v1968_data = ir7[7];
            ir7[7] = (v1968_data + (v1930_data * v1966_data));
          }
          if (v23_lead < 12) {
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
            float v1990_data = s3[42];
            float v1992_data = ir7[3];
            ir7[3] = (v1992_data + (v1974_data * v1990_data));
            float v1995_data = s3[54];
            float v1997_data = ir7[4];
            ir7[4] = (v1997_data + (v1974_data * v1995_data));
            float v2000_data = s3[66];
            float v2002_data = ir7[5];
            ir7[5] = (v2002_data + (v1974_data * v2000_data));
            float v2005_data = s3[78];
            float v2007_data = ir7[6];
            ir7[6] = (v2007_data + (v1974_data * v2005_data));
            float v2010_data = s3[90];
            float v2012_data = ir7[7];
            ir7[7] = (v2012_data + (v1974_data * v2010_data));
          }
          if (v23_lead < 12) {
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
            float v2034_data = s3[43];
            float v2036_data = ir7[3];
            ir7[3] = (v2036_data + (v2018_data * v2034_data));
            float v2039_data = s3[55];
            float v2041_data = ir7[4];
            ir7[4] = (v2041_data + (v2018_data * v2039_data));
            float v2044_data = s3[67];
            float v2046_data = ir7[5];
            ir7[5] = (v2046_data + (v2018_data * v2044_data));
            float v2049_data = s3[79];
            float v2051_data = ir7[6];
            ir7[6] = (v2051_data + (v2018_data * v2049_data));
            float v2054_data = s3[91];
            float v2056_data = ir7[7];
            ir7[7] = (v2056_data + (v2018_data * v2054_data));
          }
          if (v23_lead < 12) {
            float v2062_data = r6[8];
            float v2063_data = s3[8];
            float v2065_data = ir7[0];
            ir7[0] = (v2065_data + (v2062_data * v2063_data));
            float v2068_data = s3[20];
            float v2070_data = ir7[1];
            ir7[1] = (v2070_data + (v2062_data * v2068_data));
            float v2073_data = s3[32];
            float v2075_data = ir7[2];
            ir7[2] = (v2075_data + (v2062_data * v2073_data));
            float v2078_data = s3[44];
            float v2080_data = ir7[3];
            ir7[3] = (v2080_data + (v2062_data * v2078_data));
            float v2083_data = s3[56];
            float v2085_data = ir7[4];
            ir7[4] = (v2085_data + (v2062_data * v2083_data));
            float v2088_data = s3[68];
            float v2090_data = ir7[5];
            ir7[5] = (v2090_data + (v2062_data * v2088_data));
            float v2093_data = s3[80];
            float v2095_data = ir7[6];
            ir7[6] = (v2095_data + (v2062_data * v2093_data));
            float v2098_data = s3[92];
            float v2100_data = ir7[7];
            ir7[7] = (v2100_data + (v2062_data * v2098_data));
          }
          if (v23_lead < 12) {
            float v2106_data = r6[9];
            float v2107_data = s3[9];
            float v2109_data = ir7[0];
            ir7[0] = (v2109_data + (v2106_data * v2107_data));
            float v2112_data = s3[21];
            float v2114_data = ir7[1];
            ir7[1] = (v2114_data + (v2106_data * v2112_data));
            float v2117_data = s3[33];
            float v2119_data = ir7[2];
            ir7[2] = (v2119_data + (v2106_data * v2117_data));
            float v2122_data = s3[45];
            float v2124_data = ir7[3];
            ir7[3] = (v2124_data + (v2106_data * v2122_data));
            float v2127_data = s3[57];
            float v2129_data = ir7[4];
            ir7[4] = (v2129_data + (v2106_data * v2127_data));
            float v2132_data = s3[69];
            float v2134_data = ir7[5];
            ir7[5] = (v2134_data + (v2106_data * v2132_data));
            float v2137_data = s3[81];
            float v2139_data = ir7[6];
            ir7[6] = (v2139_data + (v2106_data * v2137_data));
            float v2142_data = s3[93];
            float v2144_data = ir7[7];
            ir7[7] = (v2144_data + (v2106_data * v2142_data));
          }
          if (v23_lead < 12) {
            float v2150_data = r6[10];
            float v2151_data = s3[10];
            float v2153_data = ir7[0];
            ir7[0] = (v2153_data + (v2150_data * v2151_data));
            float v2156_data = s3[22];
            float v2158_data = ir7[1];
            ir7[1] = (v2158_data + (v2150_data * v2156_data));
            float v2161_data = s3[34];
            float v2163_data = ir7[2];
            ir7[2] = (v2163_data + (v2150_data * v2161_data));
            float v2166_data = s3[46];
            float v2168_data = ir7[3];
            ir7[3] = (v2168_data + (v2150_data * v2166_data));
            float v2171_data = s3[58];
            float v2173_data = ir7[4];
            ir7[4] = (v2173_data + (v2150_data * v2171_data));
            float v2176_data = s3[70];
            float v2178_data = ir7[5];
            ir7[5] = (v2178_data + (v2150_data * v2176_data));
            float v2181_data = s3[82];
            float v2183_data = ir7[6];
            ir7[6] = (v2183_data + (v2150_data * v2181_data));
            float v2186_data = s3[94];
            float v2188_data = ir7[7];
            ir7[7] = (v2188_data + (v2150_data * v2186_data));
          }
          if (v23_lead < 12) {
            float v2194_data = r6[11];
            float v2195_data = s3[11];
            float v2197_data = ir7[0];
            ir7[0] = (v2197_data + (v2194_data * v2195_data));
            float v2200_data = s3[23];
            float v2202_data = ir7[1];
            ir7[1] = (v2202_data + (v2194_data * v2200_data));
            float v2205_data = s3[35];
            float v2207_data = ir7[2];
            ir7[2] = (v2207_data + (v2194_data * v2205_data));
            float v2210_data = s3[47];
            float v2212_data = ir7[3];
            ir7[3] = (v2212_data + (v2194_data * v2210_data));
            float v2215_data = s3[59];
            float v2217_data = ir7[4];
            ir7[4] = (v2217_data + (v2194_data * v2215_data));
            float v2220_data = s3[71];
            float v2222_data = ir7[5];
            ir7[5] = (v2222_data + (v2194_data * v2220_data));
            float v2225_data = s3[83];
            float v2227_data = ir7[6];
            ir7[6] = (v2227_data + (v2194_data * v2225_data));
            float v2230_data = s3[95];
            float v2232_data = ir7[7];
            ir7[7] = (v2232_data + (v2194_data * v2230_data));
          }
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v2238_n1 = 0; v2238_n1 < 8; ++v2238_n1) {
              float v2240_data = ir7[v2238_n1];
              float v2242_data = r5[v2238_n1];
              r7[v2238_n1] = (v2242_data + v2240_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v2249_i1 = 0; v2249_i1 < 8; ++v2249_i1) {
              float v2251_data = r7[v2249_i1];
              glb_m0[(v23_lead + (v2249_i1 * 12))] = v2251_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

