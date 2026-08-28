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
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 12; ++v15_i1) {
              int32_t v21_a = v15_i1 * 12;
              int32_t v22_a = v13_lead + v21_a;
              float v30_data = __ldcg(&glb_m1[(v13_lead + v21_a)]);
              r0[v15_i1] = v30_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              int32_t v45_a = v39_i1 * 12;
              int32_t v46_a = v13_lead + v45_a;
              float v54_data = __ldcg(&glb_m3[(v13_lead + v45_a)]);
              r2[v39_i1] = v54_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v13_lead < 12) {
            float v62_data = r0[0];
            float v63_data = s0[0];
            float v65_data = ir1[0];
            ir1[0] = (v65_data + (v62_data * v63_data));
            float v68_data = s0[12];
            float v70_data = ir1[1];
            ir1[1] = (v70_data + (v62_data * v68_data));
            float v73_data = s0[24];
            float v75_data = ir1[2];
            ir1[2] = (v75_data + (v62_data * v73_data));
            float v78_data = s0[36];
            float v80_data = ir1[3];
            ir1[3] = (v80_data + (v62_data * v78_data));
            float v83_data = s0[48];
            float v85_data = ir1[4];
            ir1[4] = (v85_data + (v62_data * v83_data));
            float v88_data = s0[60];
            float v90_data = ir1[5];
            ir1[5] = (v90_data + (v62_data * v88_data));
            float v93_data = s0[72];
            float v95_data = ir1[6];
            ir1[6] = (v95_data + (v62_data * v93_data));
            float v98_data = s0[84];
            float v100_data = ir1[7];
            ir1[7] = (v100_data + (v62_data * v98_data));
          }
          if (v13_lead < 12) {
            float v106_data = r0[1];
            float v107_data = s0[1];
            float v109_data = ir1[0];
            ir1[0] = (v109_data + (v106_data * v107_data));
            float v112_data = s0[13];
            float v114_data = ir1[1];
            ir1[1] = (v114_data + (v106_data * v112_data));
            float v117_data = s0[25];
            float v119_data = ir1[2];
            ir1[2] = (v119_data + (v106_data * v117_data));
            float v122_data = s0[37];
            float v124_data = ir1[3];
            ir1[3] = (v124_data + (v106_data * v122_data));
            float v127_data = s0[49];
            float v129_data = ir1[4];
            ir1[4] = (v129_data + (v106_data * v127_data));
            float v132_data = s0[61];
            float v134_data = ir1[5];
            ir1[5] = (v134_data + (v106_data * v132_data));
            float v137_data = s0[73];
            float v139_data = ir1[6];
            ir1[6] = (v139_data + (v106_data * v137_data));
            float v142_data = s0[85];
            float v144_data = ir1[7];
            ir1[7] = (v144_data + (v106_data * v142_data));
          }
          if (v13_lead < 12) {
            float v150_data = r0[2];
            float v151_data = s0[2];
            float v153_data = ir1[0];
            ir1[0] = (v153_data + (v150_data * v151_data));
            float v156_data = s0[14];
            float v158_data = ir1[1];
            ir1[1] = (v158_data + (v150_data * v156_data));
            float v161_data = s0[26];
            float v163_data = ir1[2];
            ir1[2] = (v163_data + (v150_data * v161_data));
            float v166_data = s0[38];
            float v168_data = ir1[3];
            ir1[3] = (v168_data + (v150_data * v166_data));
            float v171_data = s0[50];
            float v173_data = ir1[4];
            ir1[4] = (v173_data + (v150_data * v171_data));
            float v176_data = s0[62];
            float v178_data = ir1[5];
            ir1[5] = (v178_data + (v150_data * v176_data));
            float v181_data = s0[74];
            float v183_data = ir1[6];
            ir1[6] = (v183_data + (v150_data * v181_data));
            float v186_data = s0[86];
            float v188_data = ir1[7];
            ir1[7] = (v188_data + (v150_data * v186_data));
          }
          if (v13_lead < 12) {
            float v194_data = r0[3];
            float v195_data = s0[3];
            float v197_data = ir1[0];
            ir1[0] = (v197_data + (v194_data * v195_data));
            float v200_data = s0[15];
            float v202_data = ir1[1];
            ir1[1] = (v202_data + (v194_data * v200_data));
            float v205_data = s0[27];
            float v207_data = ir1[2];
            ir1[2] = (v207_data + (v194_data * v205_data));
            float v210_data = s0[39];
            float v212_data = ir1[3];
            ir1[3] = (v212_data + (v194_data * v210_data));
            float v215_data = s0[51];
            float v217_data = ir1[4];
            ir1[4] = (v217_data + (v194_data * v215_data));
            float v220_data = s0[63];
            float v222_data = ir1[5];
            ir1[5] = (v222_data + (v194_data * v220_data));
            float v225_data = s0[75];
            float v227_data = ir1[6];
            ir1[6] = (v227_data + (v194_data * v225_data));
            float v230_data = s0[87];
            float v232_data = ir1[7];
            ir1[7] = (v232_data + (v194_data * v230_data));
          }
          if (v13_lead < 12) {
            float v238_data = r0[4];
            float v239_data = s0[4];
            float v241_data = ir1[0];
            ir1[0] = (v241_data + (v238_data * v239_data));
            float v244_data = s0[16];
            float v246_data = ir1[1];
            ir1[1] = (v246_data + (v238_data * v244_data));
            float v249_data = s0[28];
            float v251_data = ir1[2];
            ir1[2] = (v251_data + (v238_data * v249_data));
            float v254_data = s0[40];
            float v256_data = ir1[3];
            ir1[3] = (v256_data + (v238_data * v254_data));
            float v259_data = s0[52];
            float v261_data = ir1[4];
            ir1[4] = (v261_data + (v238_data * v259_data));
            float v264_data = s0[64];
            float v266_data = ir1[5];
            ir1[5] = (v266_data + (v238_data * v264_data));
            float v269_data = s0[76];
            float v271_data = ir1[6];
            ir1[6] = (v271_data + (v238_data * v269_data));
            float v274_data = s0[88];
            float v276_data = ir1[7];
            ir1[7] = (v276_data + (v238_data * v274_data));
          }
          if (v13_lead < 12) {
            float v282_data = r0[5];
            float v283_data = s0[5];
            float v285_data = ir1[0];
            ir1[0] = (v285_data + (v282_data * v283_data));
            float v288_data = s0[17];
            float v290_data = ir1[1];
            ir1[1] = (v290_data + (v282_data * v288_data));
            float v293_data = s0[29];
            float v295_data = ir1[2];
            ir1[2] = (v295_data + (v282_data * v293_data));
            float v298_data = s0[41];
            float v300_data = ir1[3];
            ir1[3] = (v300_data + (v282_data * v298_data));
            float v303_data = s0[53];
            float v305_data = ir1[4];
            ir1[4] = (v305_data + (v282_data * v303_data));
            float v308_data = s0[65];
            float v310_data = ir1[5];
            ir1[5] = (v310_data + (v282_data * v308_data));
            float v313_data = s0[77];
            float v315_data = ir1[6];
            ir1[6] = (v315_data + (v282_data * v313_data));
            float v318_data = s0[89];
            float v320_data = ir1[7];
            ir1[7] = (v320_data + (v282_data * v318_data));
          }
          if (v13_lead < 12) {
            float v326_data = r0[6];
            float v327_data = s0[6];
            float v329_data = ir1[0];
            ir1[0] = (v329_data + (v326_data * v327_data));
            float v332_data = s0[18];
            float v334_data = ir1[1];
            ir1[1] = (v334_data + (v326_data * v332_data));
            float v337_data = s0[30];
            float v339_data = ir1[2];
            ir1[2] = (v339_data + (v326_data * v337_data));
            float v342_data = s0[42];
            float v344_data = ir1[3];
            ir1[3] = (v344_data + (v326_data * v342_data));
            float v347_data = s0[54];
            float v349_data = ir1[4];
            ir1[4] = (v349_data + (v326_data * v347_data));
            float v352_data = s0[66];
            float v354_data = ir1[5];
            ir1[5] = (v354_data + (v326_data * v352_data));
            float v357_data = s0[78];
            float v359_data = ir1[6];
            ir1[6] = (v359_data + (v326_data * v357_data));
            float v362_data = s0[90];
            float v364_data = ir1[7];
            ir1[7] = (v364_data + (v326_data * v362_data));
          }
          if (v13_lead < 12) {
            float v370_data = r0[7];
            float v371_data = s0[7];
            float v373_data = ir1[0];
            ir1[0] = (v373_data + (v370_data * v371_data));
            float v376_data = s0[19];
            float v378_data = ir1[1];
            ir1[1] = (v378_data + (v370_data * v376_data));
            float v381_data = s0[31];
            float v383_data = ir1[2];
            ir1[2] = (v383_data + (v370_data * v381_data));
            float v386_data = s0[43];
            float v388_data = ir1[3];
            ir1[3] = (v388_data + (v370_data * v386_data));
            float v391_data = s0[55];
            float v393_data = ir1[4];
            ir1[4] = (v393_data + (v370_data * v391_data));
            float v396_data = s0[67];
            float v398_data = ir1[5];
            ir1[5] = (v398_data + (v370_data * v396_data));
            float v401_data = s0[79];
            float v403_data = ir1[6];
            ir1[6] = (v403_data + (v370_data * v401_data));
            float v406_data = s0[91];
            float v408_data = ir1[7];
            ir1[7] = (v408_data + (v370_data * v406_data));
          }
          if (v13_lead < 12) {
            float v414_data = r0[8];
            float v415_data = s0[8];
            float v417_data = ir1[0];
            ir1[0] = (v417_data + (v414_data * v415_data));
            float v420_data = s0[20];
            float v422_data = ir1[1];
            ir1[1] = (v422_data + (v414_data * v420_data));
            float v425_data = s0[32];
            float v427_data = ir1[2];
            ir1[2] = (v427_data + (v414_data * v425_data));
            float v430_data = s0[44];
            float v432_data = ir1[3];
            ir1[3] = (v432_data + (v414_data * v430_data));
            float v435_data = s0[56];
            float v437_data = ir1[4];
            ir1[4] = (v437_data + (v414_data * v435_data));
            float v440_data = s0[68];
            float v442_data = ir1[5];
            ir1[5] = (v442_data + (v414_data * v440_data));
            float v445_data = s0[80];
            float v447_data = ir1[6];
            ir1[6] = (v447_data + (v414_data * v445_data));
            float v450_data = s0[92];
            float v452_data = ir1[7];
            ir1[7] = (v452_data + (v414_data * v450_data));
          }
          if (v13_lead < 12) {
            float v458_data = r0[9];
            float v459_data = s0[9];
            float v461_data = ir1[0];
            ir1[0] = (v461_data + (v458_data * v459_data));
            float v464_data = s0[21];
            float v466_data = ir1[1];
            ir1[1] = (v466_data + (v458_data * v464_data));
            float v469_data = s0[33];
            float v471_data = ir1[2];
            ir1[2] = (v471_data + (v458_data * v469_data));
            float v474_data = s0[45];
            float v476_data = ir1[3];
            ir1[3] = (v476_data + (v458_data * v474_data));
            float v479_data = s0[57];
            float v481_data = ir1[4];
            ir1[4] = (v481_data + (v458_data * v479_data));
            float v484_data = s0[69];
            float v486_data = ir1[5];
            ir1[5] = (v486_data + (v458_data * v484_data));
            float v489_data = s0[81];
            float v491_data = ir1[6];
            ir1[6] = (v491_data + (v458_data * v489_data));
            float v494_data = s0[93];
            float v496_data = ir1[7];
            ir1[7] = (v496_data + (v458_data * v494_data));
          }
          if (v13_lead < 12) {
            float v502_data = r0[10];
            float v503_data = s0[10];
            float v505_data = ir1[0];
            ir1[0] = (v505_data + (v502_data * v503_data));
            float v508_data = s0[22];
            float v510_data = ir1[1];
            ir1[1] = (v510_data + (v502_data * v508_data));
            float v513_data = s0[34];
            float v515_data = ir1[2];
            ir1[2] = (v515_data + (v502_data * v513_data));
            float v518_data = s0[46];
            float v520_data = ir1[3];
            ir1[3] = (v520_data + (v502_data * v518_data));
            float v523_data = s0[58];
            float v525_data = ir1[4];
            ir1[4] = (v525_data + (v502_data * v523_data));
            float v528_data = s0[70];
            float v530_data = ir1[5];
            ir1[5] = (v530_data + (v502_data * v528_data));
            float v533_data = s0[82];
            float v535_data = ir1[6];
            ir1[6] = (v535_data + (v502_data * v533_data));
            float v538_data = s0[94];
            float v540_data = ir1[7];
            ir1[7] = (v540_data + (v502_data * v538_data));
          }
          if (v13_lead < 12) {
            float v546_data = r0[11];
            float v547_data = s0[11];
            float v549_data = ir1[0];
            ir1[0] = (v549_data + (v546_data * v547_data));
            float v552_data = s0[23];
            float v554_data = ir1[1];
            ir1[1] = (v554_data + (v546_data * v552_data));
            float v557_data = s0[35];
            float v559_data = ir1[2];
            ir1[2] = (v559_data + (v546_data * v557_data));
            float v562_data = s0[47];
            float v564_data = ir1[3];
            ir1[3] = (v564_data + (v546_data * v562_data));
            float v567_data = s0[59];
            float v569_data = ir1[4];
            ir1[4] = (v569_data + (v546_data * v567_data));
            float v572_data = s0[71];
            float v574_data = ir1[5];
            ir1[5] = (v574_data + (v546_data * v572_data));
            float v577_data = s0[83];
            float v579_data = ir1[6];
            ir1[6] = (v579_data + (v546_data * v577_data));
            float v582_data = s0[95];
            float v584_data = ir1[7];
            ir1[7] = (v584_data + (v546_data * v582_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v590_n1 = 0; v590_n1 < 8; ++v590_n1) {
              int32_t v591_a = 0 + v590_n1;
              float v593_data = ir1[v590_n1];
              r1[v590_n1] = v593_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v602_i1 = 0; v602_i1 < 12; ++v602_i1) {
              int32_t v608_a = v602_i1 * 12;
              int32_t v609_a = v13_lead + v608_a;
              float v617_data = __ldcg(&glb_m5[(v13_lead + v608_a)]);
              r4[v602_i1] = v617_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v13_lead < 12) {
            float v625_data = r2[0];
            float v626_data = s1[0];
            float v628_data = ir3[0];
            ir3[0] = (v628_data + (v625_data * v626_data));
            float v631_data = s1[12];
            float v633_data = ir3[1];
            ir3[1] = (v633_data + (v625_data * v631_data));
            float v636_data = s1[24];
            float v638_data = ir3[2];
            ir3[2] = (v638_data + (v625_data * v636_data));
            float v641_data = s1[36];
            float v643_data = ir3[3];
            ir3[3] = (v643_data + (v625_data * v641_data));
            float v646_data = s1[48];
            float v648_data = ir3[4];
            ir3[4] = (v648_data + (v625_data * v646_data));
            float v651_data = s1[60];
            float v653_data = ir3[5];
            ir3[5] = (v653_data + (v625_data * v651_data));
            float v656_data = s1[72];
            float v658_data = ir3[6];
            ir3[6] = (v658_data + (v625_data * v656_data));
            float v661_data = s1[84];
            float v663_data = ir3[7];
            ir3[7] = (v663_data + (v625_data * v661_data));
          }
          if (v13_lead < 12) {
            float v669_data = r2[1];
            float v670_data = s1[1];
            float v672_data = ir3[0];
            ir3[0] = (v672_data + (v669_data * v670_data));
            float v675_data = s1[13];
            float v677_data = ir3[1];
            ir3[1] = (v677_data + (v669_data * v675_data));
            float v680_data = s1[25];
            float v682_data = ir3[2];
            ir3[2] = (v682_data + (v669_data * v680_data));
            float v685_data = s1[37];
            float v687_data = ir3[3];
            ir3[3] = (v687_data + (v669_data * v685_data));
            float v690_data = s1[49];
            float v692_data = ir3[4];
            ir3[4] = (v692_data + (v669_data * v690_data));
            float v695_data = s1[61];
            float v697_data = ir3[5];
            ir3[5] = (v697_data + (v669_data * v695_data));
            float v700_data = s1[73];
            float v702_data = ir3[6];
            ir3[6] = (v702_data + (v669_data * v700_data));
            float v705_data = s1[85];
            float v707_data = ir3[7];
            ir3[7] = (v707_data + (v669_data * v705_data));
          }
          if (v13_lead < 12) {
            float v713_data = r2[2];
            float v714_data = s1[2];
            float v716_data = ir3[0];
            ir3[0] = (v716_data + (v713_data * v714_data));
            float v719_data = s1[14];
            float v721_data = ir3[1];
            ir3[1] = (v721_data + (v713_data * v719_data));
            float v724_data = s1[26];
            float v726_data = ir3[2];
            ir3[2] = (v726_data + (v713_data * v724_data));
            float v729_data = s1[38];
            float v731_data = ir3[3];
            ir3[3] = (v731_data + (v713_data * v729_data));
            float v734_data = s1[50];
            float v736_data = ir3[4];
            ir3[4] = (v736_data + (v713_data * v734_data));
            float v739_data = s1[62];
            float v741_data = ir3[5];
            ir3[5] = (v741_data + (v713_data * v739_data));
            float v744_data = s1[74];
            float v746_data = ir3[6];
            ir3[6] = (v746_data + (v713_data * v744_data));
            float v749_data = s1[86];
            float v751_data = ir3[7];
            ir3[7] = (v751_data + (v713_data * v749_data));
          }
          if (v13_lead < 12) {
            float v757_data = r2[3];
            float v758_data = s1[3];
            float v760_data = ir3[0];
            ir3[0] = (v760_data + (v757_data * v758_data));
            float v763_data = s1[15];
            float v765_data = ir3[1];
            ir3[1] = (v765_data + (v757_data * v763_data));
            float v768_data = s1[27];
            float v770_data = ir3[2];
            ir3[2] = (v770_data + (v757_data * v768_data));
            float v773_data = s1[39];
            float v775_data = ir3[3];
            ir3[3] = (v775_data + (v757_data * v773_data));
            float v778_data = s1[51];
            float v780_data = ir3[4];
            ir3[4] = (v780_data + (v757_data * v778_data));
            float v783_data = s1[63];
            float v785_data = ir3[5];
            ir3[5] = (v785_data + (v757_data * v783_data));
            float v788_data = s1[75];
            float v790_data = ir3[6];
            ir3[6] = (v790_data + (v757_data * v788_data));
            float v793_data = s1[87];
            float v795_data = ir3[7];
            ir3[7] = (v795_data + (v757_data * v793_data));
          }
          if (v13_lead < 12) {
            float v801_data = r2[4];
            float v802_data = s1[4];
            float v804_data = ir3[0];
            ir3[0] = (v804_data + (v801_data * v802_data));
            float v807_data = s1[16];
            float v809_data = ir3[1];
            ir3[1] = (v809_data + (v801_data * v807_data));
            float v812_data = s1[28];
            float v814_data = ir3[2];
            ir3[2] = (v814_data + (v801_data * v812_data));
            float v817_data = s1[40];
            float v819_data = ir3[3];
            ir3[3] = (v819_data + (v801_data * v817_data));
            float v822_data = s1[52];
            float v824_data = ir3[4];
            ir3[4] = (v824_data + (v801_data * v822_data));
            float v827_data = s1[64];
            float v829_data = ir3[5];
            ir3[5] = (v829_data + (v801_data * v827_data));
            float v832_data = s1[76];
            float v834_data = ir3[6];
            ir3[6] = (v834_data + (v801_data * v832_data));
            float v837_data = s1[88];
            float v839_data = ir3[7];
            ir3[7] = (v839_data + (v801_data * v837_data));
          }
          if (v13_lead < 12) {
            float v845_data = r2[5];
            float v846_data = s1[5];
            float v848_data = ir3[0];
            ir3[0] = (v848_data + (v845_data * v846_data));
            float v851_data = s1[17];
            float v853_data = ir3[1];
            ir3[1] = (v853_data + (v845_data * v851_data));
            float v856_data = s1[29];
            float v858_data = ir3[2];
            ir3[2] = (v858_data + (v845_data * v856_data));
            float v861_data = s1[41];
            float v863_data = ir3[3];
            ir3[3] = (v863_data + (v845_data * v861_data));
            float v866_data = s1[53];
            float v868_data = ir3[4];
            ir3[4] = (v868_data + (v845_data * v866_data));
            float v871_data = s1[65];
            float v873_data = ir3[5];
            ir3[5] = (v873_data + (v845_data * v871_data));
            float v876_data = s1[77];
            float v878_data = ir3[6];
            ir3[6] = (v878_data + (v845_data * v876_data));
            float v881_data = s1[89];
            float v883_data = ir3[7];
            ir3[7] = (v883_data + (v845_data * v881_data));
          }
          if (v13_lead < 12) {
            float v889_data = r2[6];
            float v890_data = s1[6];
            float v892_data = ir3[0];
            ir3[0] = (v892_data + (v889_data * v890_data));
            float v895_data = s1[18];
            float v897_data = ir3[1];
            ir3[1] = (v897_data + (v889_data * v895_data));
            float v900_data = s1[30];
            float v902_data = ir3[2];
            ir3[2] = (v902_data + (v889_data * v900_data));
            float v905_data = s1[42];
            float v907_data = ir3[3];
            ir3[3] = (v907_data + (v889_data * v905_data));
            float v910_data = s1[54];
            float v912_data = ir3[4];
            ir3[4] = (v912_data + (v889_data * v910_data));
            float v915_data = s1[66];
            float v917_data = ir3[5];
            ir3[5] = (v917_data + (v889_data * v915_data));
            float v920_data = s1[78];
            float v922_data = ir3[6];
            ir3[6] = (v922_data + (v889_data * v920_data));
            float v925_data = s1[90];
            float v927_data = ir3[7];
            ir3[7] = (v927_data + (v889_data * v925_data));
          }
          if (v13_lead < 12) {
            float v933_data = r2[7];
            float v934_data = s1[7];
            float v936_data = ir3[0];
            ir3[0] = (v936_data + (v933_data * v934_data));
            float v939_data = s1[19];
            float v941_data = ir3[1];
            ir3[1] = (v941_data + (v933_data * v939_data));
            float v944_data = s1[31];
            float v946_data = ir3[2];
            ir3[2] = (v946_data + (v933_data * v944_data));
            float v949_data = s1[43];
            float v951_data = ir3[3];
            ir3[3] = (v951_data + (v933_data * v949_data));
            float v954_data = s1[55];
            float v956_data = ir3[4];
            ir3[4] = (v956_data + (v933_data * v954_data));
            float v959_data = s1[67];
            float v961_data = ir3[5];
            ir3[5] = (v961_data + (v933_data * v959_data));
            float v964_data = s1[79];
            float v966_data = ir3[6];
            ir3[6] = (v966_data + (v933_data * v964_data));
            float v969_data = s1[91];
            float v971_data = ir3[7];
            ir3[7] = (v971_data + (v933_data * v969_data));
          }
          if (v13_lead < 12) {
            float v977_data = r2[8];
            float v978_data = s1[8];
            float v980_data = ir3[0];
            ir3[0] = (v980_data + (v977_data * v978_data));
            float v983_data = s1[20];
            float v985_data = ir3[1];
            ir3[1] = (v985_data + (v977_data * v983_data));
            float v988_data = s1[32];
            float v990_data = ir3[2];
            ir3[2] = (v990_data + (v977_data * v988_data));
            float v993_data = s1[44];
            float v995_data = ir3[3];
            ir3[3] = (v995_data + (v977_data * v993_data));
            float v998_data = s1[56];
            float v1000_data = ir3[4];
            ir3[4] = (v1000_data + (v977_data * v998_data));
            float v1003_data = s1[68];
            float v1005_data = ir3[5];
            ir3[5] = (v1005_data + (v977_data * v1003_data));
            float v1008_data = s1[80];
            float v1010_data = ir3[6];
            ir3[6] = (v1010_data + (v977_data * v1008_data));
            float v1013_data = s1[92];
            float v1015_data = ir3[7];
            ir3[7] = (v1015_data + (v977_data * v1013_data));
          }
          if (v13_lead < 12) {
            float v1021_data = r2[9];
            float v1022_data = s1[9];
            float v1024_data = ir3[0];
            ir3[0] = (v1024_data + (v1021_data * v1022_data));
            float v1027_data = s1[21];
            float v1029_data = ir3[1];
            ir3[1] = (v1029_data + (v1021_data * v1027_data));
            float v1032_data = s1[33];
            float v1034_data = ir3[2];
            ir3[2] = (v1034_data + (v1021_data * v1032_data));
            float v1037_data = s1[45];
            float v1039_data = ir3[3];
            ir3[3] = (v1039_data + (v1021_data * v1037_data));
            float v1042_data = s1[57];
            float v1044_data = ir3[4];
            ir3[4] = (v1044_data + (v1021_data * v1042_data));
            float v1047_data = s1[69];
            float v1049_data = ir3[5];
            ir3[5] = (v1049_data + (v1021_data * v1047_data));
            float v1052_data = s1[81];
            float v1054_data = ir3[6];
            ir3[6] = (v1054_data + (v1021_data * v1052_data));
            float v1057_data = s1[93];
            float v1059_data = ir3[7];
            ir3[7] = (v1059_data + (v1021_data * v1057_data));
          }
          if (v13_lead < 12) {
            float v1065_data = r2[10];
            float v1066_data = s1[10];
            float v1068_data = ir3[0];
            ir3[0] = (v1068_data + (v1065_data * v1066_data));
            float v1071_data = s1[22];
            float v1073_data = ir3[1];
            ir3[1] = (v1073_data + (v1065_data * v1071_data));
            float v1076_data = s1[34];
            float v1078_data = ir3[2];
            ir3[2] = (v1078_data + (v1065_data * v1076_data));
            float v1081_data = s1[46];
            float v1083_data = ir3[3];
            ir3[3] = (v1083_data + (v1065_data * v1081_data));
            float v1086_data = s1[58];
            float v1088_data = ir3[4];
            ir3[4] = (v1088_data + (v1065_data * v1086_data));
            float v1091_data = s1[70];
            float v1093_data = ir3[5];
            ir3[5] = (v1093_data + (v1065_data * v1091_data));
            float v1096_data = s1[82];
            float v1098_data = ir3[6];
            ir3[6] = (v1098_data + (v1065_data * v1096_data));
            float v1101_data = s1[94];
            float v1103_data = ir3[7];
            ir3[7] = (v1103_data + (v1065_data * v1101_data));
          }
          if (v13_lead < 12) {
            float v1109_data = r2[11];
            float v1110_data = s1[11];
            float v1112_data = ir3[0];
            ir3[0] = (v1112_data + (v1109_data * v1110_data));
            float v1115_data = s1[23];
            float v1117_data = ir3[1];
            ir3[1] = (v1117_data + (v1109_data * v1115_data));
            float v1120_data = s1[35];
            float v1122_data = ir3[2];
            ir3[2] = (v1122_data + (v1109_data * v1120_data));
            float v1125_data = s1[47];
            float v1127_data = ir3[3];
            ir3[3] = (v1127_data + (v1109_data * v1125_data));
            float v1130_data = s1[59];
            float v1132_data = ir3[4];
            ir3[4] = (v1132_data + (v1109_data * v1130_data));
            float v1135_data = s1[71];
            float v1137_data = ir3[5];
            ir3[5] = (v1137_data + (v1109_data * v1135_data));
            float v1140_data = s1[83];
            float v1142_data = ir3[6];
            ir3[6] = (v1142_data + (v1109_data * v1140_data));
            float v1145_data = s1[95];
            float v1147_data = ir3[7];
            ir3[7] = (v1147_data + (v1109_data * v1145_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v1153_n1 = 0; v1153_n1 < 8; ++v1153_n1) {
              int32_t v1154_a = 0 + v1153_n1;
              float v1156_data = ir3[v1153_n1];
              int32_t v1157_a = 0 + v1153_n1;
              float v1159_data = r1[v1153_n1];
              r3[v1153_n1] = (v1159_data + v1156_data);
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          {
            // s2 = load{g>s}(glb_m6[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v1169_i1 = 0; v1169_i1 < 12; ++v1169_i1) {
              int32_t v1175_a = v1169_i1 * 12;
              int32_t v1176_a = v13_lead + v1175_a;
              float v1184_data = __ldcg(&glb_m7[(v13_lead + v1175_a)]);
              r6[v1169_i1] = v1184_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v13_lead < 12) {
            float v1192_data = r4[0];
            float v1193_data = s2[0];
            float v1195_data = ir5[0];
            ir5[0] = (v1195_data + (v1192_data * v1193_data));
            float v1198_data = s2[12];
            float v1200_data = ir5[1];
            ir5[1] = (v1200_data + (v1192_data * v1198_data));
            float v1203_data = s2[24];
            float v1205_data = ir5[2];
            ir5[2] = (v1205_data + (v1192_data * v1203_data));
            float v1208_data = s2[36];
            float v1210_data = ir5[3];
            ir5[3] = (v1210_data + (v1192_data * v1208_data));
            float v1213_data = s2[48];
            float v1215_data = ir5[4];
            ir5[4] = (v1215_data + (v1192_data * v1213_data));
            float v1218_data = s2[60];
            float v1220_data = ir5[5];
            ir5[5] = (v1220_data + (v1192_data * v1218_data));
            float v1223_data = s2[72];
            float v1225_data = ir5[6];
            ir5[6] = (v1225_data + (v1192_data * v1223_data));
            float v1228_data = s2[84];
            float v1230_data = ir5[7];
            ir5[7] = (v1230_data + (v1192_data * v1228_data));
          }
          if (v13_lead < 12) {
            float v1236_data = r4[1];
            float v1237_data = s2[1];
            float v1239_data = ir5[0];
            ir5[0] = (v1239_data + (v1236_data * v1237_data));
            float v1242_data = s2[13];
            float v1244_data = ir5[1];
            ir5[1] = (v1244_data + (v1236_data * v1242_data));
            float v1247_data = s2[25];
            float v1249_data = ir5[2];
            ir5[2] = (v1249_data + (v1236_data * v1247_data));
            float v1252_data = s2[37];
            float v1254_data = ir5[3];
            ir5[3] = (v1254_data + (v1236_data * v1252_data));
            float v1257_data = s2[49];
            float v1259_data = ir5[4];
            ir5[4] = (v1259_data + (v1236_data * v1257_data));
            float v1262_data = s2[61];
            float v1264_data = ir5[5];
            ir5[5] = (v1264_data + (v1236_data * v1262_data));
            float v1267_data = s2[73];
            float v1269_data = ir5[6];
            ir5[6] = (v1269_data + (v1236_data * v1267_data));
            float v1272_data = s2[85];
            float v1274_data = ir5[7];
            ir5[7] = (v1274_data + (v1236_data * v1272_data));
          }
          if (v13_lead < 12) {
            float v1280_data = r4[2];
            float v1281_data = s2[2];
            float v1283_data = ir5[0];
            ir5[0] = (v1283_data + (v1280_data * v1281_data));
            float v1286_data = s2[14];
            float v1288_data = ir5[1];
            ir5[1] = (v1288_data + (v1280_data * v1286_data));
            float v1291_data = s2[26];
            float v1293_data = ir5[2];
            ir5[2] = (v1293_data + (v1280_data * v1291_data));
            float v1296_data = s2[38];
            float v1298_data = ir5[3];
            ir5[3] = (v1298_data + (v1280_data * v1296_data));
            float v1301_data = s2[50];
            float v1303_data = ir5[4];
            ir5[4] = (v1303_data + (v1280_data * v1301_data));
            float v1306_data = s2[62];
            float v1308_data = ir5[5];
            ir5[5] = (v1308_data + (v1280_data * v1306_data));
            float v1311_data = s2[74];
            float v1313_data = ir5[6];
            ir5[6] = (v1313_data + (v1280_data * v1311_data));
            float v1316_data = s2[86];
            float v1318_data = ir5[7];
            ir5[7] = (v1318_data + (v1280_data * v1316_data));
          }
          if (v13_lead < 12) {
            float v1324_data = r4[3];
            float v1325_data = s2[3];
            float v1327_data = ir5[0];
            ir5[0] = (v1327_data + (v1324_data * v1325_data));
            float v1330_data = s2[15];
            float v1332_data = ir5[1];
            ir5[1] = (v1332_data + (v1324_data * v1330_data));
            float v1335_data = s2[27];
            float v1337_data = ir5[2];
            ir5[2] = (v1337_data + (v1324_data * v1335_data));
            float v1340_data = s2[39];
            float v1342_data = ir5[3];
            ir5[3] = (v1342_data + (v1324_data * v1340_data));
            float v1345_data = s2[51];
            float v1347_data = ir5[4];
            ir5[4] = (v1347_data + (v1324_data * v1345_data));
            float v1350_data = s2[63];
            float v1352_data = ir5[5];
            ir5[5] = (v1352_data + (v1324_data * v1350_data));
            float v1355_data = s2[75];
            float v1357_data = ir5[6];
            ir5[6] = (v1357_data + (v1324_data * v1355_data));
            float v1360_data = s2[87];
            float v1362_data = ir5[7];
            ir5[7] = (v1362_data + (v1324_data * v1360_data));
          }
          if (v13_lead < 12) {
            float v1368_data = r4[4];
            float v1369_data = s2[4];
            float v1371_data = ir5[0];
            ir5[0] = (v1371_data + (v1368_data * v1369_data));
            float v1374_data = s2[16];
            float v1376_data = ir5[1];
            ir5[1] = (v1376_data + (v1368_data * v1374_data));
            float v1379_data = s2[28];
            float v1381_data = ir5[2];
            ir5[2] = (v1381_data + (v1368_data * v1379_data));
            float v1384_data = s2[40];
            float v1386_data = ir5[3];
            ir5[3] = (v1386_data + (v1368_data * v1384_data));
            float v1389_data = s2[52];
            float v1391_data = ir5[4];
            ir5[4] = (v1391_data + (v1368_data * v1389_data));
            float v1394_data = s2[64];
            float v1396_data = ir5[5];
            ir5[5] = (v1396_data + (v1368_data * v1394_data));
            float v1399_data = s2[76];
            float v1401_data = ir5[6];
            ir5[6] = (v1401_data + (v1368_data * v1399_data));
            float v1404_data = s2[88];
            float v1406_data = ir5[7];
            ir5[7] = (v1406_data + (v1368_data * v1404_data));
          }
          if (v13_lead < 12) {
            float v1412_data = r4[5];
            float v1413_data = s2[5];
            float v1415_data = ir5[0];
            ir5[0] = (v1415_data + (v1412_data * v1413_data));
            float v1418_data = s2[17];
            float v1420_data = ir5[1];
            ir5[1] = (v1420_data + (v1412_data * v1418_data));
            float v1423_data = s2[29];
            float v1425_data = ir5[2];
            ir5[2] = (v1425_data + (v1412_data * v1423_data));
            float v1428_data = s2[41];
            float v1430_data = ir5[3];
            ir5[3] = (v1430_data + (v1412_data * v1428_data));
            float v1433_data = s2[53];
            float v1435_data = ir5[4];
            ir5[4] = (v1435_data + (v1412_data * v1433_data));
            float v1438_data = s2[65];
            float v1440_data = ir5[5];
            ir5[5] = (v1440_data + (v1412_data * v1438_data));
            float v1443_data = s2[77];
            float v1445_data = ir5[6];
            ir5[6] = (v1445_data + (v1412_data * v1443_data));
            float v1448_data = s2[89];
            float v1450_data = ir5[7];
            ir5[7] = (v1450_data + (v1412_data * v1448_data));
          }
          if (v13_lead < 12) {
            float v1456_data = r4[6];
            float v1457_data = s2[6];
            float v1459_data = ir5[0];
            ir5[0] = (v1459_data + (v1456_data * v1457_data));
            float v1462_data = s2[18];
            float v1464_data = ir5[1];
            ir5[1] = (v1464_data + (v1456_data * v1462_data));
            float v1467_data = s2[30];
            float v1469_data = ir5[2];
            ir5[2] = (v1469_data + (v1456_data * v1467_data));
            float v1472_data = s2[42];
            float v1474_data = ir5[3];
            ir5[3] = (v1474_data + (v1456_data * v1472_data));
            float v1477_data = s2[54];
            float v1479_data = ir5[4];
            ir5[4] = (v1479_data + (v1456_data * v1477_data));
            float v1482_data = s2[66];
            float v1484_data = ir5[5];
            ir5[5] = (v1484_data + (v1456_data * v1482_data));
            float v1487_data = s2[78];
            float v1489_data = ir5[6];
            ir5[6] = (v1489_data + (v1456_data * v1487_data));
            float v1492_data = s2[90];
            float v1494_data = ir5[7];
            ir5[7] = (v1494_data + (v1456_data * v1492_data));
          }
          if (v13_lead < 12) {
            float v1500_data = r4[7];
            float v1501_data = s2[7];
            float v1503_data = ir5[0];
            ir5[0] = (v1503_data + (v1500_data * v1501_data));
            float v1506_data = s2[19];
            float v1508_data = ir5[1];
            ir5[1] = (v1508_data + (v1500_data * v1506_data));
            float v1511_data = s2[31];
            float v1513_data = ir5[2];
            ir5[2] = (v1513_data + (v1500_data * v1511_data));
            float v1516_data = s2[43];
            float v1518_data = ir5[3];
            ir5[3] = (v1518_data + (v1500_data * v1516_data));
            float v1521_data = s2[55];
            float v1523_data = ir5[4];
            ir5[4] = (v1523_data + (v1500_data * v1521_data));
            float v1526_data = s2[67];
            float v1528_data = ir5[5];
            ir5[5] = (v1528_data + (v1500_data * v1526_data));
            float v1531_data = s2[79];
            float v1533_data = ir5[6];
            ir5[6] = (v1533_data + (v1500_data * v1531_data));
            float v1536_data = s2[91];
            float v1538_data = ir5[7];
            ir5[7] = (v1538_data + (v1500_data * v1536_data));
          }
          if (v13_lead < 12) {
            float v1544_data = r4[8];
            float v1545_data = s2[8];
            float v1547_data = ir5[0];
            ir5[0] = (v1547_data + (v1544_data * v1545_data));
            float v1550_data = s2[20];
            float v1552_data = ir5[1];
            ir5[1] = (v1552_data + (v1544_data * v1550_data));
            float v1555_data = s2[32];
            float v1557_data = ir5[2];
            ir5[2] = (v1557_data + (v1544_data * v1555_data));
            float v1560_data = s2[44];
            float v1562_data = ir5[3];
            ir5[3] = (v1562_data + (v1544_data * v1560_data));
            float v1565_data = s2[56];
            float v1567_data = ir5[4];
            ir5[4] = (v1567_data + (v1544_data * v1565_data));
            float v1570_data = s2[68];
            float v1572_data = ir5[5];
            ir5[5] = (v1572_data + (v1544_data * v1570_data));
            float v1575_data = s2[80];
            float v1577_data = ir5[6];
            ir5[6] = (v1577_data + (v1544_data * v1575_data));
            float v1580_data = s2[92];
            float v1582_data = ir5[7];
            ir5[7] = (v1582_data + (v1544_data * v1580_data));
          }
          if (v13_lead < 12) {
            float v1588_data = r4[9];
            float v1589_data = s2[9];
            float v1591_data = ir5[0];
            ir5[0] = (v1591_data + (v1588_data * v1589_data));
            float v1594_data = s2[21];
            float v1596_data = ir5[1];
            ir5[1] = (v1596_data + (v1588_data * v1594_data));
            float v1599_data = s2[33];
            float v1601_data = ir5[2];
            ir5[2] = (v1601_data + (v1588_data * v1599_data));
            float v1604_data = s2[45];
            float v1606_data = ir5[3];
            ir5[3] = (v1606_data + (v1588_data * v1604_data));
            float v1609_data = s2[57];
            float v1611_data = ir5[4];
            ir5[4] = (v1611_data + (v1588_data * v1609_data));
            float v1614_data = s2[69];
            float v1616_data = ir5[5];
            ir5[5] = (v1616_data + (v1588_data * v1614_data));
            float v1619_data = s2[81];
            float v1621_data = ir5[6];
            ir5[6] = (v1621_data + (v1588_data * v1619_data));
            float v1624_data = s2[93];
            float v1626_data = ir5[7];
            ir5[7] = (v1626_data + (v1588_data * v1624_data));
          }
          if (v13_lead < 12) {
            float v1632_data = r4[10];
            float v1633_data = s2[10];
            float v1635_data = ir5[0];
            ir5[0] = (v1635_data + (v1632_data * v1633_data));
            float v1638_data = s2[22];
            float v1640_data = ir5[1];
            ir5[1] = (v1640_data + (v1632_data * v1638_data));
            float v1643_data = s2[34];
            float v1645_data = ir5[2];
            ir5[2] = (v1645_data + (v1632_data * v1643_data));
            float v1648_data = s2[46];
            float v1650_data = ir5[3];
            ir5[3] = (v1650_data + (v1632_data * v1648_data));
            float v1653_data = s2[58];
            float v1655_data = ir5[4];
            ir5[4] = (v1655_data + (v1632_data * v1653_data));
            float v1658_data = s2[70];
            float v1660_data = ir5[5];
            ir5[5] = (v1660_data + (v1632_data * v1658_data));
            float v1663_data = s2[82];
            float v1665_data = ir5[6];
            ir5[6] = (v1665_data + (v1632_data * v1663_data));
            float v1668_data = s2[94];
            float v1670_data = ir5[7];
            ir5[7] = (v1670_data + (v1632_data * v1668_data));
          }
          if (v13_lead < 12) {
            float v1676_data = r4[11];
            float v1677_data = s2[11];
            float v1679_data = ir5[0];
            ir5[0] = (v1679_data + (v1676_data * v1677_data));
            float v1682_data = s2[23];
            float v1684_data = ir5[1];
            ir5[1] = (v1684_data + (v1676_data * v1682_data));
            float v1687_data = s2[35];
            float v1689_data = ir5[2];
            ir5[2] = (v1689_data + (v1676_data * v1687_data));
            float v1692_data = s2[47];
            float v1694_data = ir5[3];
            ir5[3] = (v1694_data + (v1676_data * v1692_data));
            float v1697_data = s2[59];
            float v1699_data = ir5[4];
            ir5[4] = (v1699_data + (v1676_data * v1697_data));
            float v1702_data = s2[71];
            float v1704_data = ir5[5];
            ir5[5] = (v1704_data + (v1676_data * v1702_data));
            float v1707_data = s2[83];
            float v1709_data = ir5[6];
            ir5[6] = (v1709_data + (v1676_data * v1707_data));
            float v1712_data = s2[95];
            float v1714_data = ir5[7];
            ir5[7] = (v1714_data + (v1676_data * v1712_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v1720_n1 = 0; v1720_n1 < 8; ++v1720_n1) {
              int32_t v1721_a = 0 + v1720_n1;
              float v1723_data = ir5[v1720_n1];
              int32_t v1724_a = 0 + v1720_n1;
              float v1726_data = r3[v1720_n1];
              r5[v1720_n1] = (v1726_data + v1723_data);
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          {
            // s3 = load{g>s}(glb_m8[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v13_lead < 12) {
            float v1737_data = r6[0];
            float v1738_data = s3[0];
            float v1740_data = ir7[0];
            ir7[0] = (v1740_data + (v1737_data * v1738_data));
            float v1743_data = s3[12];
            float v1745_data = ir7[1];
            ir7[1] = (v1745_data + (v1737_data * v1743_data));
            float v1748_data = s3[24];
            float v1750_data = ir7[2];
            ir7[2] = (v1750_data + (v1737_data * v1748_data));
            float v1753_data = s3[36];
            float v1755_data = ir7[3];
            ir7[3] = (v1755_data + (v1737_data * v1753_data));
            float v1758_data = s3[48];
            float v1760_data = ir7[4];
            ir7[4] = (v1760_data + (v1737_data * v1758_data));
            float v1763_data = s3[60];
            float v1765_data = ir7[5];
            ir7[5] = (v1765_data + (v1737_data * v1763_data));
            float v1768_data = s3[72];
            float v1770_data = ir7[6];
            ir7[6] = (v1770_data + (v1737_data * v1768_data));
            float v1773_data = s3[84];
            float v1775_data = ir7[7];
            ir7[7] = (v1775_data + (v1737_data * v1773_data));
          }
          if (v13_lead < 12) {
            float v1781_data = r6[1];
            float v1782_data = s3[1];
            float v1784_data = ir7[0];
            ir7[0] = (v1784_data + (v1781_data * v1782_data));
            float v1787_data = s3[13];
            float v1789_data = ir7[1];
            ir7[1] = (v1789_data + (v1781_data * v1787_data));
            float v1792_data = s3[25];
            float v1794_data = ir7[2];
            ir7[2] = (v1794_data + (v1781_data * v1792_data));
            float v1797_data = s3[37];
            float v1799_data = ir7[3];
            ir7[3] = (v1799_data + (v1781_data * v1797_data));
            float v1802_data = s3[49];
            float v1804_data = ir7[4];
            ir7[4] = (v1804_data + (v1781_data * v1802_data));
            float v1807_data = s3[61];
            float v1809_data = ir7[5];
            ir7[5] = (v1809_data + (v1781_data * v1807_data));
            float v1812_data = s3[73];
            float v1814_data = ir7[6];
            ir7[6] = (v1814_data + (v1781_data * v1812_data));
            float v1817_data = s3[85];
            float v1819_data = ir7[7];
            ir7[7] = (v1819_data + (v1781_data * v1817_data));
          }
          if (v13_lead < 12) {
            float v1825_data = r6[2];
            float v1826_data = s3[2];
            float v1828_data = ir7[0];
            ir7[0] = (v1828_data + (v1825_data * v1826_data));
            float v1831_data = s3[14];
            float v1833_data = ir7[1];
            ir7[1] = (v1833_data + (v1825_data * v1831_data));
            float v1836_data = s3[26];
            float v1838_data = ir7[2];
            ir7[2] = (v1838_data + (v1825_data * v1836_data));
            float v1841_data = s3[38];
            float v1843_data = ir7[3];
            ir7[3] = (v1843_data + (v1825_data * v1841_data));
            float v1846_data = s3[50];
            float v1848_data = ir7[4];
            ir7[4] = (v1848_data + (v1825_data * v1846_data));
            float v1851_data = s3[62];
            float v1853_data = ir7[5];
            ir7[5] = (v1853_data + (v1825_data * v1851_data));
            float v1856_data = s3[74];
            float v1858_data = ir7[6];
            ir7[6] = (v1858_data + (v1825_data * v1856_data));
            float v1861_data = s3[86];
            float v1863_data = ir7[7];
            ir7[7] = (v1863_data + (v1825_data * v1861_data));
          }
          if (v13_lead < 12) {
            float v1869_data = r6[3];
            float v1870_data = s3[3];
            float v1872_data = ir7[0];
            ir7[0] = (v1872_data + (v1869_data * v1870_data));
            float v1875_data = s3[15];
            float v1877_data = ir7[1];
            ir7[1] = (v1877_data + (v1869_data * v1875_data));
            float v1880_data = s3[27];
            float v1882_data = ir7[2];
            ir7[2] = (v1882_data + (v1869_data * v1880_data));
            float v1885_data = s3[39];
            float v1887_data = ir7[3];
            ir7[3] = (v1887_data + (v1869_data * v1885_data));
            float v1890_data = s3[51];
            float v1892_data = ir7[4];
            ir7[4] = (v1892_data + (v1869_data * v1890_data));
            float v1895_data = s3[63];
            float v1897_data = ir7[5];
            ir7[5] = (v1897_data + (v1869_data * v1895_data));
            float v1900_data = s3[75];
            float v1902_data = ir7[6];
            ir7[6] = (v1902_data + (v1869_data * v1900_data));
            float v1905_data = s3[87];
            float v1907_data = ir7[7];
            ir7[7] = (v1907_data + (v1869_data * v1905_data));
          }
          if (v13_lead < 12) {
            float v1913_data = r6[4];
            float v1914_data = s3[4];
            float v1916_data = ir7[0];
            ir7[0] = (v1916_data + (v1913_data * v1914_data));
            float v1919_data = s3[16];
            float v1921_data = ir7[1];
            ir7[1] = (v1921_data + (v1913_data * v1919_data));
            float v1924_data = s3[28];
            float v1926_data = ir7[2];
            ir7[2] = (v1926_data + (v1913_data * v1924_data));
            float v1929_data = s3[40];
            float v1931_data = ir7[3];
            ir7[3] = (v1931_data + (v1913_data * v1929_data));
            float v1934_data = s3[52];
            float v1936_data = ir7[4];
            ir7[4] = (v1936_data + (v1913_data * v1934_data));
            float v1939_data = s3[64];
            float v1941_data = ir7[5];
            ir7[5] = (v1941_data + (v1913_data * v1939_data));
            float v1944_data = s3[76];
            float v1946_data = ir7[6];
            ir7[6] = (v1946_data + (v1913_data * v1944_data));
            float v1949_data = s3[88];
            float v1951_data = ir7[7];
            ir7[7] = (v1951_data + (v1913_data * v1949_data));
          }
          if (v13_lead < 12) {
            float v1957_data = r6[5];
            float v1958_data = s3[5];
            float v1960_data = ir7[0];
            ir7[0] = (v1960_data + (v1957_data * v1958_data));
            float v1963_data = s3[17];
            float v1965_data = ir7[1];
            ir7[1] = (v1965_data + (v1957_data * v1963_data));
            float v1968_data = s3[29];
            float v1970_data = ir7[2];
            ir7[2] = (v1970_data + (v1957_data * v1968_data));
            float v1973_data = s3[41];
            float v1975_data = ir7[3];
            ir7[3] = (v1975_data + (v1957_data * v1973_data));
            float v1978_data = s3[53];
            float v1980_data = ir7[4];
            ir7[4] = (v1980_data + (v1957_data * v1978_data));
            float v1983_data = s3[65];
            float v1985_data = ir7[5];
            ir7[5] = (v1985_data + (v1957_data * v1983_data));
            float v1988_data = s3[77];
            float v1990_data = ir7[6];
            ir7[6] = (v1990_data + (v1957_data * v1988_data));
            float v1993_data = s3[89];
            float v1995_data = ir7[7];
            ir7[7] = (v1995_data + (v1957_data * v1993_data));
          }
          if (v13_lead < 12) {
            float v2001_data = r6[6];
            float v2002_data = s3[6];
            float v2004_data = ir7[0];
            ir7[0] = (v2004_data + (v2001_data * v2002_data));
            float v2007_data = s3[18];
            float v2009_data = ir7[1];
            ir7[1] = (v2009_data + (v2001_data * v2007_data));
            float v2012_data = s3[30];
            float v2014_data = ir7[2];
            ir7[2] = (v2014_data + (v2001_data * v2012_data));
            float v2017_data = s3[42];
            float v2019_data = ir7[3];
            ir7[3] = (v2019_data + (v2001_data * v2017_data));
            float v2022_data = s3[54];
            float v2024_data = ir7[4];
            ir7[4] = (v2024_data + (v2001_data * v2022_data));
            float v2027_data = s3[66];
            float v2029_data = ir7[5];
            ir7[5] = (v2029_data + (v2001_data * v2027_data));
            float v2032_data = s3[78];
            float v2034_data = ir7[6];
            ir7[6] = (v2034_data + (v2001_data * v2032_data));
            float v2037_data = s3[90];
            float v2039_data = ir7[7];
            ir7[7] = (v2039_data + (v2001_data * v2037_data));
          }
          if (v13_lead < 12) {
            float v2045_data = r6[7];
            float v2046_data = s3[7];
            float v2048_data = ir7[0];
            ir7[0] = (v2048_data + (v2045_data * v2046_data));
            float v2051_data = s3[19];
            float v2053_data = ir7[1];
            ir7[1] = (v2053_data + (v2045_data * v2051_data));
            float v2056_data = s3[31];
            float v2058_data = ir7[2];
            ir7[2] = (v2058_data + (v2045_data * v2056_data));
            float v2061_data = s3[43];
            float v2063_data = ir7[3];
            ir7[3] = (v2063_data + (v2045_data * v2061_data));
            float v2066_data = s3[55];
            float v2068_data = ir7[4];
            ir7[4] = (v2068_data + (v2045_data * v2066_data));
            float v2071_data = s3[67];
            float v2073_data = ir7[5];
            ir7[5] = (v2073_data + (v2045_data * v2071_data));
            float v2076_data = s3[79];
            float v2078_data = ir7[6];
            ir7[6] = (v2078_data + (v2045_data * v2076_data));
            float v2081_data = s3[91];
            float v2083_data = ir7[7];
            ir7[7] = (v2083_data + (v2045_data * v2081_data));
          }
          if (v13_lead < 12) {
            float v2089_data = r6[8];
            float v2090_data = s3[8];
            float v2092_data = ir7[0];
            ir7[0] = (v2092_data + (v2089_data * v2090_data));
            float v2095_data = s3[20];
            float v2097_data = ir7[1];
            ir7[1] = (v2097_data + (v2089_data * v2095_data));
            float v2100_data = s3[32];
            float v2102_data = ir7[2];
            ir7[2] = (v2102_data + (v2089_data * v2100_data));
            float v2105_data = s3[44];
            float v2107_data = ir7[3];
            ir7[3] = (v2107_data + (v2089_data * v2105_data));
            float v2110_data = s3[56];
            float v2112_data = ir7[4];
            ir7[4] = (v2112_data + (v2089_data * v2110_data));
            float v2115_data = s3[68];
            float v2117_data = ir7[5];
            ir7[5] = (v2117_data + (v2089_data * v2115_data));
            float v2120_data = s3[80];
            float v2122_data = ir7[6];
            ir7[6] = (v2122_data + (v2089_data * v2120_data));
            float v2125_data = s3[92];
            float v2127_data = ir7[7];
            ir7[7] = (v2127_data + (v2089_data * v2125_data));
          }
          if (v13_lead < 12) {
            float v2133_data = r6[9];
            float v2134_data = s3[9];
            float v2136_data = ir7[0];
            ir7[0] = (v2136_data + (v2133_data * v2134_data));
            float v2139_data = s3[21];
            float v2141_data = ir7[1];
            ir7[1] = (v2141_data + (v2133_data * v2139_data));
            float v2144_data = s3[33];
            float v2146_data = ir7[2];
            ir7[2] = (v2146_data + (v2133_data * v2144_data));
            float v2149_data = s3[45];
            float v2151_data = ir7[3];
            ir7[3] = (v2151_data + (v2133_data * v2149_data));
            float v2154_data = s3[57];
            float v2156_data = ir7[4];
            ir7[4] = (v2156_data + (v2133_data * v2154_data));
            float v2159_data = s3[69];
            float v2161_data = ir7[5];
            ir7[5] = (v2161_data + (v2133_data * v2159_data));
            float v2164_data = s3[81];
            float v2166_data = ir7[6];
            ir7[6] = (v2166_data + (v2133_data * v2164_data));
            float v2169_data = s3[93];
            float v2171_data = ir7[7];
            ir7[7] = (v2171_data + (v2133_data * v2169_data));
          }
          if (v13_lead < 12) {
            float v2177_data = r6[10];
            float v2178_data = s3[10];
            float v2180_data = ir7[0];
            ir7[0] = (v2180_data + (v2177_data * v2178_data));
            float v2183_data = s3[22];
            float v2185_data = ir7[1];
            ir7[1] = (v2185_data + (v2177_data * v2183_data));
            float v2188_data = s3[34];
            float v2190_data = ir7[2];
            ir7[2] = (v2190_data + (v2177_data * v2188_data));
            float v2193_data = s3[46];
            float v2195_data = ir7[3];
            ir7[3] = (v2195_data + (v2177_data * v2193_data));
            float v2198_data = s3[58];
            float v2200_data = ir7[4];
            ir7[4] = (v2200_data + (v2177_data * v2198_data));
            float v2203_data = s3[70];
            float v2205_data = ir7[5];
            ir7[5] = (v2205_data + (v2177_data * v2203_data));
            float v2208_data = s3[82];
            float v2210_data = ir7[6];
            ir7[6] = (v2210_data + (v2177_data * v2208_data));
            float v2213_data = s3[94];
            float v2215_data = ir7[7];
            ir7[7] = (v2215_data + (v2177_data * v2213_data));
          }
          if (v13_lead < 12) {
            float v2221_data = r6[11];
            float v2222_data = s3[11];
            float v2224_data = ir7[0];
            ir7[0] = (v2224_data + (v2221_data * v2222_data));
            float v2227_data = s3[23];
            float v2229_data = ir7[1];
            ir7[1] = (v2229_data + (v2221_data * v2227_data));
            float v2232_data = s3[35];
            float v2234_data = ir7[2];
            ir7[2] = (v2234_data + (v2221_data * v2232_data));
            float v2237_data = s3[47];
            float v2239_data = ir7[3];
            ir7[3] = (v2239_data + (v2221_data * v2237_data));
            float v2242_data = s3[59];
            float v2244_data = ir7[4];
            ir7[4] = (v2244_data + (v2221_data * v2242_data));
            float v2247_data = s3[71];
            float v2249_data = ir7[5];
            ir7[5] = (v2249_data + (v2221_data * v2247_data));
            float v2252_data = s3[83];
            float v2254_data = ir7[6];
            ir7[6] = (v2254_data + (v2221_data * v2252_data));
            float v2257_data = s3[95];
            float v2259_data = ir7[7];
            ir7[7] = (v2259_data + (v2221_data * v2257_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v2265_n1 = 0; v2265_n1 < 8; ++v2265_n1) {
              int32_t v2266_a = 0 + v2265_n1;
              float v2268_data = ir7[v2265_n1];
              int32_t v2269_a = 0 + v2265_n1;
              float v2271_data = r5[v2265_n1];
              r7[v2265_n1] = (v2271_data + v2268_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v2278_i1 = 0; v2278_i1 < 8; ++v2278_i1) {
              int32_t v2279_a = 0 + v2278_i1;
              float v2281_data = r7[v2278_i1];
              glb_m0[(v13_lead + (v2278_i1 * 12))] = v2281_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

