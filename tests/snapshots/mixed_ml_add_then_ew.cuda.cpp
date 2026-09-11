// === base name ===
kernel_7648941730faa9f3

// === header ===
void launcher_kernel_7648941730faa9f3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7648941730faa9f3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7648941730faa9f3, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7648941730faa9f3, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7648941730faa9f3<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_7648941730faa9f3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 32;
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v32_data = __ldcg(&glb_m0[(v22_lead + (v24_i1 * 8))]);
              r0[v24_i1] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 8; ++v41_i1) {
              float v49_data = __ldcg(&glb_m2[(v22_lead + (v41_i1 * 8))]);
              r2[v41_i1] = v49_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          if (v22_lead < 8) {
            float v56_data = r0[0];
            float v57_data = s0[0];
            float v59_data = r1[0];
            r1[0] = (v59_data + (v56_data * v57_data));
            float v62_data = s0[8];
            float v64_data = r1[1];
            r1[1] = (v64_data + (v56_data * v62_data));
            float v67_data = s0[16];
            float v69_data = r1[2];
            r1[2] = (v69_data + (v56_data * v67_data));
            float v72_data = s0[24];
            float v74_data = r1[3];
            r1[3] = (v74_data + (v56_data * v72_data));
            float v77_data = s0[32];
            float v79_data = r1[4];
            r1[4] = (v79_data + (v56_data * v77_data));
            float v82_data = s0[40];
            float v84_data = r1[5];
            r1[5] = (v84_data + (v56_data * v82_data));
            float v87_data = s0[48];
            float v89_data = r1[6];
            r1[6] = (v89_data + (v56_data * v87_data));
            float v92_data = s0[56];
            float v94_data = r1[7];
            r1[7] = (v94_data + (v56_data * v92_data));
          }
          if (v22_lead < 8) {
            float v100_data = r0[1];
            float v101_data = s0[1];
            float v103_data = r1[0];
            r1[0] = (v103_data + (v100_data * v101_data));
            float v106_data = s0[9];
            float v108_data = r1[1];
            r1[1] = (v108_data + (v100_data * v106_data));
            float v111_data = s0[17];
            float v113_data = r1[2];
            r1[2] = (v113_data + (v100_data * v111_data));
            float v116_data = s0[25];
            float v118_data = r1[3];
            r1[3] = (v118_data + (v100_data * v116_data));
            float v121_data = s0[33];
            float v123_data = r1[4];
            r1[4] = (v123_data + (v100_data * v121_data));
            float v126_data = s0[41];
            float v128_data = r1[5];
            r1[5] = (v128_data + (v100_data * v126_data));
            float v131_data = s0[49];
            float v133_data = r1[6];
            r1[6] = (v133_data + (v100_data * v131_data));
            float v136_data = s0[57];
            float v138_data = r1[7];
            r1[7] = (v138_data + (v100_data * v136_data));
          }
          if (v22_lead < 8) {
            float v144_data = r0[2];
            float v145_data = s0[2];
            float v147_data = r1[0];
            r1[0] = (v147_data + (v144_data * v145_data));
            float v150_data = s0[10];
            float v152_data = r1[1];
            r1[1] = (v152_data + (v144_data * v150_data));
            float v155_data = s0[18];
            float v157_data = r1[2];
            r1[2] = (v157_data + (v144_data * v155_data));
            float v160_data = s0[26];
            float v162_data = r1[3];
            r1[3] = (v162_data + (v144_data * v160_data));
            float v165_data = s0[34];
            float v167_data = r1[4];
            r1[4] = (v167_data + (v144_data * v165_data));
            float v170_data = s0[42];
            float v172_data = r1[5];
            r1[5] = (v172_data + (v144_data * v170_data));
            float v175_data = s0[50];
            float v177_data = r1[6];
            r1[6] = (v177_data + (v144_data * v175_data));
            float v180_data = s0[58];
            float v182_data = r1[7];
            r1[7] = (v182_data + (v144_data * v180_data));
          }
          if (v22_lead < 8) {
            float v188_data = r0[3];
            float v189_data = s0[3];
            float v191_data = r1[0];
            r1[0] = (v191_data + (v188_data * v189_data));
            float v194_data = s0[11];
            float v196_data = r1[1];
            r1[1] = (v196_data + (v188_data * v194_data));
            float v199_data = s0[19];
            float v201_data = r1[2];
            r1[2] = (v201_data + (v188_data * v199_data));
            float v204_data = s0[27];
            float v206_data = r1[3];
            r1[3] = (v206_data + (v188_data * v204_data));
            float v209_data = s0[35];
            float v211_data = r1[4];
            r1[4] = (v211_data + (v188_data * v209_data));
            float v214_data = s0[43];
            float v216_data = r1[5];
            r1[5] = (v216_data + (v188_data * v214_data));
            float v219_data = s0[51];
            float v221_data = r1[6];
            r1[6] = (v221_data + (v188_data * v219_data));
            float v224_data = s0[59];
            float v226_data = r1[7];
            r1[7] = (v226_data + (v188_data * v224_data));
          }
          if (v22_lead < 8) {
            float v232_data = r0[4];
            float v233_data = s0[4];
            float v235_data = r1[0];
            r1[0] = (v235_data + (v232_data * v233_data));
            float v238_data = s0[12];
            float v240_data = r1[1];
            r1[1] = (v240_data + (v232_data * v238_data));
            float v243_data = s0[20];
            float v245_data = r1[2];
            r1[2] = (v245_data + (v232_data * v243_data));
            float v248_data = s0[28];
            float v250_data = r1[3];
            r1[3] = (v250_data + (v232_data * v248_data));
            float v253_data = s0[36];
            float v255_data = r1[4];
            r1[4] = (v255_data + (v232_data * v253_data));
            float v258_data = s0[44];
            float v260_data = r1[5];
            r1[5] = (v260_data + (v232_data * v258_data));
            float v263_data = s0[52];
            float v265_data = r1[6];
            r1[6] = (v265_data + (v232_data * v263_data));
            float v268_data = s0[60];
            float v270_data = r1[7];
            r1[7] = (v270_data + (v232_data * v268_data));
          }
          if (v22_lead < 8) {
            float v276_data = r0[5];
            float v277_data = s0[5];
            float v279_data = r1[0];
            r1[0] = (v279_data + (v276_data * v277_data));
            float v282_data = s0[13];
            float v284_data = r1[1];
            r1[1] = (v284_data + (v276_data * v282_data));
            float v287_data = s0[21];
            float v289_data = r1[2];
            r1[2] = (v289_data + (v276_data * v287_data));
            float v292_data = s0[29];
            float v294_data = r1[3];
            r1[3] = (v294_data + (v276_data * v292_data));
            float v297_data = s0[37];
            float v299_data = r1[4];
            r1[4] = (v299_data + (v276_data * v297_data));
            float v302_data = s0[45];
            float v304_data = r1[5];
            r1[5] = (v304_data + (v276_data * v302_data));
            float v307_data = s0[53];
            float v309_data = r1[6];
            r1[6] = (v309_data + (v276_data * v307_data));
            float v312_data = s0[61];
            float v314_data = r1[7];
            r1[7] = (v314_data + (v276_data * v312_data));
          }
          if (v22_lead < 8) {
            float v320_data = r0[6];
            float v321_data = s0[6];
            float v323_data = r1[0];
            r1[0] = (v323_data + (v320_data * v321_data));
            float v326_data = s0[14];
            float v328_data = r1[1];
            r1[1] = (v328_data + (v320_data * v326_data));
            float v331_data = s0[22];
            float v333_data = r1[2];
            r1[2] = (v333_data + (v320_data * v331_data));
            float v336_data = s0[30];
            float v338_data = r1[3];
            r1[3] = (v338_data + (v320_data * v336_data));
            float v341_data = s0[38];
            float v343_data = r1[4];
            r1[4] = (v343_data + (v320_data * v341_data));
            float v346_data = s0[46];
            float v348_data = r1[5];
            r1[5] = (v348_data + (v320_data * v346_data));
            float v351_data = s0[54];
            float v353_data = r1[6];
            r1[6] = (v353_data + (v320_data * v351_data));
            float v356_data = s0[62];
            float v358_data = r1[7];
            r1[7] = (v358_data + (v320_data * v356_data));
          }
          if (v22_lead < 8) {
            float v364_data = r0[7];
            float v365_data = s0[7];
            float v367_data = r1[0];
            r1[0] = (v367_data + (v364_data * v365_data));
            float v370_data = s0[15];
            float v372_data = r1[1];
            r1[1] = (v372_data + (v364_data * v370_data));
            float v375_data = s0[23];
            float v377_data = r1[2];
            r1[2] = (v377_data + (v364_data * v375_data));
            float v380_data = s0[31];
            float v382_data = r1[3];
            r1[3] = (v382_data + (v364_data * v380_data));
            float v385_data = s0[39];
            float v387_data = r1[4];
            r1[4] = (v387_data + (v364_data * v385_data));
            float v390_data = s0[47];
            float v392_data = r1[5];
            r1[5] = (v392_data + (v364_data * v390_data));
            float v395_data = s0[55];
            float v397_data = r1[6];
            r1[6] = (v397_data + (v364_data * v395_data));
            float v400_data = s0[63];
            float v402_data = r1[7];
            r1[7] = (v402_data + (v364_data * v400_data));
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m3[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m3[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m3[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          if (v22_lead < 8) {
            float v412_data = r2[0];
            float v413_data = s2[0];
            float v415_data = ir3[0];
            ir3[0] = (v415_data + (v412_data * v413_data));
            float v418_data = s2[8];
            float v420_data = ir3[1];
            ir3[1] = (v420_data + (v412_data * v418_data));
            float v423_data = s2[16];
            float v425_data = ir3[2];
            ir3[2] = (v425_data + (v412_data * v423_data));
            float v428_data = s2[24];
            float v430_data = ir3[3];
            ir3[3] = (v430_data + (v412_data * v428_data));
            float v433_data = s2[32];
            float v435_data = ir3[4];
            ir3[4] = (v435_data + (v412_data * v433_data));
            float v438_data = s2[40];
            float v440_data = ir3[5];
            ir3[5] = (v440_data + (v412_data * v438_data));
            float v443_data = s2[48];
            float v445_data = ir3[6];
            ir3[6] = (v445_data + (v412_data * v443_data));
            float v448_data = s2[56];
            float v450_data = ir3[7];
            ir3[7] = (v450_data + (v412_data * v448_data));
          }
          if (v22_lead < 8) {
            float v456_data = r2[1];
            float v457_data = s2[1];
            float v459_data = ir3[0];
            ir3[0] = (v459_data + (v456_data * v457_data));
            float v462_data = s2[9];
            float v464_data = ir3[1];
            ir3[1] = (v464_data + (v456_data * v462_data));
            float v467_data = s2[17];
            float v469_data = ir3[2];
            ir3[2] = (v469_data + (v456_data * v467_data));
            float v472_data = s2[25];
            float v474_data = ir3[3];
            ir3[3] = (v474_data + (v456_data * v472_data));
            float v477_data = s2[33];
            float v479_data = ir3[4];
            ir3[4] = (v479_data + (v456_data * v477_data));
            float v482_data = s2[41];
            float v484_data = ir3[5];
            ir3[5] = (v484_data + (v456_data * v482_data));
            float v487_data = s2[49];
            float v489_data = ir3[6];
            ir3[6] = (v489_data + (v456_data * v487_data));
            float v492_data = s2[57];
            float v494_data = ir3[7];
            ir3[7] = (v494_data + (v456_data * v492_data));
          }
          if (v22_lead < 8) {
            float v500_data = r2[2];
            float v501_data = s2[2];
            float v503_data = ir3[0];
            ir3[0] = (v503_data + (v500_data * v501_data));
            float v506_data = s2[10];
            float v508_data = ir3[1];
            ir3[1] = (v508_data + (v500_data * v506_data));
            float v511_data = s2[18];
            float v513_data = ir3[2];
            ir3[2] = (v513_data + (v500_data * v511_data));
            float v516_data = s2[26];
            float v518_data = ir3[3];
            ir3[3] = (v518_data + (v500_data * v516_data));
            float v521_data = s2[34];
            float v523_data = ir3[4];
            ir3[4] = (v523_data + (v500_data * v521_data));
            float v526_data = s2[42];
            float v528_data = ir3[5];
            ir3[5] = (v528_data + (v500_data * v526_data));
            float v531_data = s2[50];
            float v533_data = ir3[6];
            ir3[6] = (v533_data + (v500_data * v531_data));
            float v536_data = s2[58];
            float v538_data = ir3[7];
            ir3[7] = (v538_data + (v500_data * v536_data));
          }
          if (v22_lead < 8) {
            float v544_data = r2[3];
            float v545_data = s2[3];
            float v547_data = ir3[0];
            ir3[0] = (v547_data + (v544_data * v545_data));
            float v550_data = s2[11];
            float v552_data = ir3[1];
            ir3[1] = (v552_data + (v544_data * v550_data));
            float v555_data = s2[19];
            float v557_data = ir3[2];
            ir3[2] = (v557_data + (v544_data * v555_data));
            float v560_data = s2[27];
            float v562_data = ir3[3];
            ir3[3] = (v562_data + (v544_data * v560_data));
            float v565_data = s2[35];
            float v567_data = ir3[4];
            ir3[4] = (v567_data + (v544_data * v565_data));
            float v570_data = s2[43];
            float v572_data = ir3[5];
            ir3[5] = (v572_data + (v544_data * v570_data));
            float v575_data = s2[51];
            float v577_data = ir3[6];
            ir3[6] = (v577_data + (v544_data * v575_data));
            float v580_data = s2[59];
            float v582_data = ir3[7];
            ir3[7] = (v582_data + (v544_data * v580_data));
          }
          if (v22_lead < 8) {
            float v588_data = r2[4];
            float v589_data = s2[4];
            float v591_data = ir3[0];
            ir3[0] = (v591_data + (v588_data * v589_data));
            float v594_data = s2[12];
            float v596_data = ir3[1];
            ir3[1] = (v596_data + (v588_data * v594_data));
            float v599_data = s2[20];
            float v601_data = ir3[2];
            ir3[2] = (v601_data + (v588_data * v599_data));
            float v604_data = s2[28];
            float v606_data = ir3[3];
            ir3[3] = (v606_data + (v588_data * v604_data));
            float v609_data = s2[36];
            float v611_data = ir3[4];
            ir3[4] = (v611_data + (v588_data * v609_data));
            float v614_data = s2[44];
            float v616_data = ir3[5];
            ir3[5] = (v616_data + (v588_data * v614_data));
            float v619_data = s2[52];
            float v621_data = ir3[6];
            ir3[6] = (v621_data + (v588_data * v619_data));
            float v624_data = s2[60];
            float v626_data = ir3[7];
            ir3[7] = (v626_data + (v588_data * v624_data));
          }
          if (v22_lead < 8) {
            float v632_data = r2[5];
            float v633_data = s2[5];
            float v635_data = ir3[0];
            ir3[0] = (v635_data + (v632_data * v633_data));
            float v638_data = s2[13];
            float v640_data = ir3[1];
            ir3[1] = (v640_data + (v632_data * v638_data));
            float v643_data = s2[21];
            float v645_data = ir3[2];
            ir3[2] = (v645_data + (v632_data * v643_data));
            float v648_data = s2[29];
            float v650_data = ir3[3];
            ir3[3] = (v650_data + (v632_data * v648_data));
            float v653_data = s2[37];
            float v655_data = ir3[4];
            ir3[4] = (v655_data + (v632_data * v653_data));
            float v658_data = s2[45];
            float v660_data = ir3[5];
            ir3[5] = (v660_data + (v632_data * v658_data));
            float v663_data = s2[53];
            float v665_data = ir3[6];
            ir3[6] = (v665_data + (v632_data * v663_data));
            float v668_data = s2[61];
            float v670_data = ir3[7];
            ir3[7] = (v670_data + (v632_data * v668_data));
          }
          if (v22_lead < 8) {
            float v676_data = r2[6];
            float v677_data = s2[6];
            float v679_data = ir3[0];
            ir3[0] = (v679_data + (v676_data * v677_data));
            float v682_data = s2[14];
            float v684_data = ir3[1];
            ir3[1] = (v684_data + (v676_data * v682_data));
            float v687_data = s2[22];
            float v689_data = ir3[2];
            ir3[2] = (v689_data + (v676_data * v687_data));
            float v692_data = s2[30];
            float v694_data = ir3[3];
            ir3[3] = (v694_data + (v676_data * v692_data));
            float v697_data = s2[38];
            float v699_data = ir3[4];
            ir3[4] = (v699_data + (v676_data * v697_data));
            float v702_data = s2[46];
            float v704_data = ir3[5];
            ir3[5] = (v704_data + (v676_data * v702_data));
            float v707_data = s2[54];
            float v709_data = ir3[6];
            ir3[6] = (v709_data + (v676_data * v707_data));
            float v712_data = s2[62];
            float v714_data = ir3[7];
            ir3[7] = (v714_data + (v676_data * v712_data));
          }
          if (v22_lead < 8) {
            float v720_data = r2[7];
            float v721_data = s2[7];
            float v723_data = ir3[0];
            ir3[0] = (v723_data + (v720_data * v721_data));
            float v726_data = s2[15];
            float v728_data = ir3[1];
            ir3[1] = (v728_data + (v720_data * v726_data));
            float v731_data = s2[23];
            float v733_data = ir3[2];
            ir3[2] = (v733_data + (v720_data * v731_data));
            float v736_data = s2[31];
            float v738_data = ir3[3];
            ir3[3] = (v738_data + (v720_data * v736_data));
            float v741_data = s2[39];
            float v743_data = ir3[4];
            ir3[4] = (v743_data + (v720_data * v741_data));
            float v746_data = s2[47];
            float v748_data = ir3[5];
            ir3[5] = (v748_data + (v720_data * v746_data));
            float v751_data = s2[55];
            float v753_data = ir3[6];
            ir3[6] = (v753_data + (v720_data * v751_data));
            float v756_data = s2[63];
            float v758_data = ir3[7];
            ir3[7] = (v758_data + (v720_data * v756_data));
          }
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v764_n1 = 0; v764_n1 < 8; ++v764_n1) {
              float v766_data = ir3[v764_n1];
              float v768_data = r1[v764_n1];
              r3[v764_n1] = (v768_data + v766_data);
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v775_i1 = 0; v775_i1 < 8; ++v775_i1) {
              float v777_data = r3[v775_i1];
              int32_t v784_a = v22_lead + (v775_i1 * 8);
              s1[(v784_a ^ ((v784_a >> 5) & 31))] = v777_data;
            }
          }
          __syncwarp();
          // glb_m4 = abs(s1)
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v792_k1 = 0; v792_k1 < 8; ++v792_k1) {
              int32_t v798_a = v792_k1 * 8;
              int32_t v799_a = v22_lead + v798_a;
              float v803_data = s1[(v799_a ^ ((v799_a >> 5) & 31))];
              glb_m4[(v22_lead + v798_a)] = (fabsf(v803_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

