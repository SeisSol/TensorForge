// === base name ===
kernel_64179564b21da6f6

// === header ===
void launcher_kernel_64179564b21da6f6(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_64179564b21da6f6(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_64179564b21da6f6, block.x * block.y * block.z, 896 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_64179564b21da6f6, cudaFuncAttributeMaxDynamicSharedMemorySize, 896 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_64179564b21da6f6<<<grid,block,896 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_64179564b21da6f6(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×32(12×6) {0..12}×{0..6} strided
    // m1 32×32(6×6) {0..6}×{0..6} strided
    // m2 32×32(12×6) {0..12}×{0..6} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
    // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 16;
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 6; ++v22_i1) {
              float v30_data = __ldcg(&glb_m0[(v20_lead + (v22_i1 * 12))]);
              r0[v22_i1] = v30_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 4) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v48_data = __ldcg(&glb_m3[(v20_lead + (v40_i1 * 12))]);
              r2[v40_i1] = v48_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v20_lead < 12) {
            float v55_data = r0[0];
            float v56_data = s0[0];
            float v58_data = r1[0];
            r1[0] = (v58_data + (v55_data * v56_data));
            float v61_data = s0[6];
            float v63_data = r1[1];
            r1[1] = (v63_data + (v55_data * v61_data));
            float v66_data = s0[12];
            float v68_data = r1[2];
            r1[2] = (v68_data + (v55_data * v66_data));
            float v71_data = s0[18];
            float v73_data = r1[3];
            r1[3] = (v73_data + (v55_data * v71_data));
            float v76_data = s0[24];
            float v78_data = r1[4];
            r1[4] = (v78_data + (v55_data * v76_data));
            float v81_data = s0[30];
            float v83_data = r1[5];
            r1[5] = (v83_data + (v55_data * v81_data));
          }
          if (v20_lead < 12) {
            float v89_data = r0[1];
            float v90_data = s0[1];
            float v92_data = r1[0];
            r1[0] = (v92_data + (v89_data * v90_data));
            float v95_data = s0[7];
            float v97_data = r1[1];
            r1[1] = (v97_data + (v89_data * v95_data));
            float v100_data = s0[13];
            float v102_data = r1[2];
            r1[2] = (v102_data + (v89_data * v100_data));
            float v105_data = s0[19];
            float v107_data = r1[3];
            r1[3] = (v107_data + (v89_data * v105_data));
            float v110_data = s0[25];
            float v112_data = r1[4];
            r1[4] = (v112_data + (v89_data * v110_data));
            float v115_data = s0[31];
            float v117_data = r1[5];
            r1[5] = (v117_data + (v89_data * v115_data));
          }
          if (v20_lead < 12) {
            float v123_data = r0[2];
            float v124_data = s0[2];
            float v126_data = r1[0];
            r1[0] = (v126_data + (v123_data * v124_data));
            float v129_data = s0[8];
            float v131_data = r1[1];
            r1[1] = (v131_data + (v123_data * v129_data));
            float v134_data = s0[14];
            float v136_data = r1[2];
            r1[2] = (v136_data + (v123_data * v134_data));
            float v139_data = s0[20];
            float v141_data = r1[3];
            r1[3] = (v141_data + (v123_data * v139_data));
            float v144_data = s0[26];
            float v146_data = r1[4];
            r1[4] = (v146_data + (v123_data * v144_data));
            float v149_data = s0[32];
            float v151_data = r1[5];
            r1[5] = (v151_data + (v123_data * v149_data));
          }
          if (v20_lead < 12) {
            float v157_data = r0[3];
            float v158_data = s0[3];
            float v160_data = r1[0];
            r1[0] = (v160_data + (v157_data * v158_data));
            float v163_data = s0[9];
            float v165_data = r1[1];
            r1[1] = (v165_data + (v157_data * v163_data));
            float v168_data = s0[15];
            float v170_data = r1[2];
            r1[2] = (v170_data + (v157_data * v168_data));
            float v173_data = s0[21];
            float v175_data = r1[3];
            r1[3] = (v175_data + (v157_data * v173_data));
            float v178_data = s0[27];
            float v180_data = r1[4];
            r1[4] = (v180_data + (v157_data * v178_data));
            float v183_data = s0[33];
            float v185_data = r1[5];
            r1[5] = (v185_data + (v157_data * v183_data));
          }
          if (v20_lead < 12) {
            float v191_data = r0[4];
            float v192_data = s0[4];
            float v194_data = r1[0];
            r1[0] = (v194_data + (v191_data * v192_data));
            float v197_data = s0[10];
            float v199_data = r1[1];
            r1[1] = (v199_data + (v191_data * v197_data));
            float v202_data = s0[16];
            float v204_data = r1[2];
            r1[2] = (v204_data + (v191_data * v202_data));
            float v207_data = s0[22];
            float v209_data = r1[3];
            r1[3] = (v209_data + (v191_data * v207_data));
            float v212_data = s0[28];
            float v214_data = r1[4];
            r1[4] = (v214_data + (v191_data * v212_data));
            float v217_data = s0[34];
            float v219_data = r1[5];
            r1[5] = (v219_data + (v191_data * v217_data));
          }
          if (v20_lead < 12) {
            float v225_data = r0[5];
            float v226_data = s0[5];
            float v228_data = r1[0];
            r1[0] = (v228_data + (v225_data * v226_data));
            float v231_data = s0[11];
            float v233_data = r1[1];
            r1[1] = (v233_data + (v225_data * v231_data));
            float v236_data = s0[17];
            float v238_data = r1[2];
            r1[2] = (v238_data + (v225_data * v236_data));
            float v241_data = s0[23];
            float v243_data = r1[3];
            r1[3] = (v243_data + (v225_data * v241_data));
            float v246_data = s0[29];
            float v248_data = r1[4];
            r1[4] = (v248_data + (v225_data * v246_data));
            float v251_data = s0[35];
            float v253_data = r1[5];
            r1[5] = (v253_data + (v225_data * v251_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = store{r>s}(localShrMem0, r1);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v259_i1 = 0; v259_i1 < 6; ++v259_i1) {
              float v261_data = r1[v259_i1];
              int32_t v268_a = v20_lead + (v259_i1 * 12);
              s1[(v268_a ^ ((v268_a >> 3) & 7))] = v261_data;
            }
          }
          float r3[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v20_lead < 12) {
            float v278_data = r2[0];
            float v279_data = s1[0];
            float v281_data = ir3[0];
            ir3[0] = (v281_data + (v278_data * v279_data));
            float v284_data = s1[13];
            float v286_data = ir3[1];
            ir3[1] = (v286_data + (v278_data * v284_data));
            float v289_data = s1[27];
            float v291_data = ir3[2];
            ir3[2] = (v291_data + (v278_data * v289_data));
            float v294_data = s1[32];
            float v296_data = ir3[3];
            ir3[3] = (v296_data + (v278_data * v294_data));
            float v299_data = s1[54];
            float v301_data = ir3[4];
            ir3[4] = (v301_data + (v278_data * v299_data));
            float v304_data = s1[59];
            float v306_data = ir3[5];
            ir3[5] = (v306_data + (v278_data * v304_data));
          }
          if (v20_lead < 12) {
            float v312_data = r2[1];
            float v313_data = s1[1];
            float v315_data = ir3[0];
            ir3[0] = (v315_data + (v312_data * v313_data));
            float v318_data = s1[12];
            float v320_data = ir3[1];
            ir3[1] = (v320_data + (v312_data * v318_data));
            float v323_data = s1[26];
            float v325_data = ir3[2];
            ir3[2] = (v325_data + (v312_data * v323_data));
            float v328_data = s1[33];
            float v330_data = ir3[3];
            ir3[3] = (v330_data + (v312_data * v328_data));
            float v333_data = s1[55];
            float v335_data = ir3[4];
            ir3[4] = (v335_data + (v312_data * v333_data));
            float v338_data = s1[58];
            float v340_data = ir3[5];
            ir3[5] = (v340_data + (v312_data * v338_data));
          }
          if (v20_lead < 12) {
            float v346_data = r2[2];
            float v347_data = s1[2];
            float v349_data = ir3[0];
            ir3[0] = (v349_data + (v346_data * v347_data));
            float v352_data = s1[15];
            float v354_data = ir3[1];
            ir3[1] = (v354_data + (v346_data * v352_data));
            float v357_data = s1[25];
            float v359_data = ir3[2];
            ir3[2] = (v359_data + (v346_data * v357_data));
            float v362_data = s1[34];
            float v364_data = ir3[3];
            ir3[3] = (v364_data + (v346_data * v362_data));
            float v367_data = s1[52];
            float v369_data = ir3[4];
            ir3[4] = (v369_data + (v346_data * v367_data));
            float v372_data = s1[57];
            float v374_data = ir3[5];
            ir3[5] = (v374_data + (v346_data * v372_data));
          }
          if (v20_lead < 12) {
            float v380_data = r2[3];
            float v381_data = s1[3];
            float v383_data = ir3[0];
            ir3[0] = (v383_data + (v380_data * v381_data));
            float v386_data = s1[14];
            float v388_data = ir3[1];
            ir3[1] = (v388_data + (v380_data * v386_data));
            float v391_data = s1[24];
            float v393_data = ir3[2];
            ir3[2] = (v393_data + (v380_data * v391_data));
            float v396_data = s1[35];
            float v398_data = ir3[3];
            ir3[3] = (v398_data + (v380_data * v396_data));
            float v401_data = s1[53];
            float v403_data = ir3[4];
            ir3[4] = (v403_data + (v380_data * v401_data));
            float v406_data = s1[56];
            float v408_data = ir3[5];
            ir3[5] = (v408_data + (v380_data * v406_data));
          }
          if (v20_lead < 12) {
            float v414_data = r2[4];
            float v415_data = s1[4];
            float v417_data = ir3[0];
            ir3[0] = (v417_data + (v414_data * v415_data));
            float v420_data = s1[18];
            float v422_data = ir3[1];
            ir3[1] = (v422_data + (v414_data * v420_data));
            float v425_data = s1[31];
            float v427_data = ir3[2];
            ir3[2] = (v427_data + (v414_data * v425_data));
            float v430_data = s1[45];
            float v432_data = ir3[3];
            ir3[3] = (v432_data + (v414_data * v430_data));
            float v435_data = s1[50];
            float v437_data = ir3[4];
            ir3[4] = (v437_data + (v414_data * v435_data));
            float v440_data = s1[64];
            float v442_data = ir3[5];
            ir3[5] = (v442_data + (v414_data * v440_data));
          }
          if (v20_lead < 12) {
            float v448_data = r2[5];
            float v449_data = s1[5];
            float v451_data = ir3[0];
            ir3[0] = (v451_data + (v448_data * v449_data));
            float v454_data = s1[19];
            float v456_data = ir3[1];
            ir3[1] = (v456_data + (v448_data * v454_data));
            float v459_data = s1[30];
            float v461_data = ir3[2];
            ir3[2] = (v461_data + (v448_data * v459_data));
            float v464_data = s1[44];
            float v466_data = ir3[3];
            ir3[3] = (v466_data + (v448_data * v464_data));
            float v469_data = s1[51];
            float v471_data = ir3[4];
            ir3[4] = (v471_data + (v448_data * v469_data));
            float v474_data = s1[65];
            float v476_data = ir3[5];
            ir3[5] = (v476_data + (v448_data * v474_data));
          }
          if (v20_lead < 12) {
            float v482_data = r2[6];
            float v483_data = s1[6];
            float v485_data = ir3[0];
            ir3[0] = (v485_data + (v482_data * v483_data));
            float v488_data = s1[16];
            float v490_data = ir3[1];
            ir3[1] = (v490_data + (v482_data * v488_data));
            float v493_data = s1[29];
            float v495_data = ir3[2];
            ir3[2] = (v495_data + (v482_data * v493_data));
            float v498_data = s1[47];
            float v500_data = ir3[3];
            ir3[3] = (v500_data + (v482_data * v498_data));
            float v503_data = s1[48];
            float v505_data = ir3[4];
            ir3[4] = (v505_data + (v482_data * v503_data));
            float v508_data = s1[66];
            float v510_data = ir3[5];
            ir3[5] = (v510_data + (v482_data * v508_data));
          }
          if (v20_lead < 12) {
            float v516_data = r2[7];
            float v517_data = s1[7];
            float v519_data = ir3[0];
            ir3[0] = (v519_data + (v516_data * v517_data));
            float v522_data = s1[17];
            float v524_data = ir3[1];
            ir3[1] = (v524_data + (v516_data * v522_data));
            float v527_data = s1[28];
            float v529_data = ir3[2];
            ir3[2] = (v529_data + (v516_data * v527_data));
            float v532_data = s1[46];
            float v534_data = ir3[3];
            ir3[3] = (v534_data + (v516_data * v532_data));
            float v537_data = s1[49];
            float v539_data = ir3[4];
            ir3[4] = (v539_data + (v516_data * v537_data));
            float v542_data = s1[67];
            float v544_data = ir3[5];
            ir3[5] = (v544_data + (v516_data * v542_data));
          }
          if (v20_lead < 12) {
            float v550_data = r2[8];
            float v551_data = s1[9];
            float v553_data = ir3[0];
            ir3[0] = (v553_data + (v550_data * v551_data));
            float v556_data = s1[22];
            float v558_data = ir3[1];
            ir3[1] = (v558_data + (v550_data * v556_data));
            float v561_data = s1[36];
            float v563_data = ir3[2];
            ir3[2] = (v563_data + (v550_data * v561_data));
            float v566_data = s1[41];
            float v568_data = ir3[3];
            ir3[3] = (v568_data + (v550_data * v566_data));
            float v571_data = s1[63];
            float v573_data = ir3[4];
            ir3[4] = (v573_data + (v550_data * v571_data));
            float v576_data = s1[68];
            float v578_data = ir3[5];
            ir3[5] = (v578_data + (v550_data * v576_data));
          }
          if (v20_lead < 12) {
            float v584_data = r2[9];
            float v585_data = s1[8];
            float v587_data = ir3[0];
            ir3[0] = (v587_data + (v584_data * v585_data));
            float v590_data = s1[23];
            float v592_data = ir3[1];
            ir3[1] = (v592_data + (v584_data * v590_data));
            float v595_data = s1[37];
            float v597_data = ir3[2];
            ir3[2] = (v597_data + (v584_data * v595_data));
            float v600_data = s1[40];
            float v602_data = ir3[3];
            ir3[3] = (v602_data + (v584_data * v600_data));
            float v605_data = s1[62];
            float v607_data = ir3[4];
            ir3[4] = (v607_data + (v584_data * v605_data));
            float v610_data = s1[69];
            float v612_data = ir3[5];
            ir3[5] = (v612_data + (v584_data * v610_data));
          }
          if (v20_lead < 12) {
            float v618_data = r2[10];
            float v619_data = s1[11];
            float v621_data = ir3[0];
            ir3[0] = (v621_data + (v618_data * v619_data));
            float v624_data = s1[20];
            float v626_data = ir3[1];
            ir3[1] = (v626_data + (v618_data * v624_data));
            float v629_data = s1[38];
            float v631_data = ir3[2];
            ir3[2] = (v631_data + (v618_data * v629_data));
            float v634_data = s1[43];
            float v636_data = ir3[3];
            ir3[3] = (v636_data + (v618_data * v634_data));
            float v639_data = s1[61];
            float v641_data = ir3[4];
            ir3[4] = (v641_data + (v618_data * v639_data));
            float v644_data = s1[70];
            float v646_data = ir3[5];
            ir3[5] = (v646_data + (v618_data * v644_data));
          }
          if (v20_lead < 12) {
            float v652_data = r2[11];
            float v653_data = s1[10];
            float v655_data = ir3[0];
            ir3[0] = (v655_data + (v652_data * v653_data));
            float v658_data = s1[21];
            float v660_data = ir3[1];
            ir3[1] = (v660_data + (v652_data * v658_data));
            float v663_data = s1[39];
            float v665_data = ir3[2];
            ir3[2] = (v665_data + (v652_data * v663_data));
            float v668_data = s1[42];
            float v670_data = ir3[3];
            ir3[3] = (v670_data + (v652_data * v668_data));
            float v673_data = s1[60];
            float v675_data = ir3[4];
            ir3[4] = (v675_data + (v652_data * v673_data));
            float v678_data = s1[71];
            float v680_data = ir3[5];
            ir3[5] = (v680_data + (v652_data * v678_data));
          }
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v686_n1 = 0; v686_n1 < 6; ++v686_n1) {
              float v688_data = ir3[v686_n1];
              r3[v686_n1] = v688_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v20_lead < 12) {
            #pragma unroll
            for (int32_t v694_i1 = 0; v694_i1 < 6; ++v694_i1) {
              float v696_data = r3[v694_i1];
              glb_m2[(v20_lead + (v694_i1 * 12))] = v696_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

