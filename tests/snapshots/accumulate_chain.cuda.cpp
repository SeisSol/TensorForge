// === base name ===
kernel_d7426afcadf74359

// === header ===
void launcher_kernel_d7426afcadf74359(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d7426afcadf74359(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d7426afcadf74359, block.x * block.y * block.z, 896 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_d7426afcadf74359, cudaFuncAttributeMaxDynamicSharedMemorySize, 896 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_d7426afcadf74359<<<grid,block,896 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_d7426afcadf74359(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      float * __restrict__ s3 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v7_batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[v7_batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[v7_batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v27_lead = threadIdx.x % 16;
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v37_data = __ldcg(&glb_m1[(v27_lead + (v29_i1 * 12))]);
              r0[v29_i1] = v37_data;
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
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v53_data = __ldcg(&glb_m3[(v27_lead + (v45_i1 * 12))]);
              r2[v45_i1] = v53_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v27_lead < 12) {
            float v61_data = r0[0];
            float v62_data = s0[0];
            float v64_data = ir1[0];
            ir1[0] = (v64_data + (v61_data * v62_data));
            float v67_data = s0[12];
            float v69_data = ir1[1];
            ir1[1] = (v69_data + (v61_data * v67_data));
            float v72_data = s0[24];
            float v74_data = ir1[2];
            ir1[2] = (v74_data + (v61_data * v72_data));
            float v77_data = s0[36];
            float v79_data = ir1[3];
            ir1[3] = (v79_data + (v61_data * v77_data));
            float v82_data = s0[48];
            float v84_data = ir1[4];
            ir1[4] = (v84_data + (v61_data * v82_data));
            float v87_data = s0[60];
            float v89_data = ir1[5];
            ir1[5] = (v89_data + (v61_data * v87_data));
            float v92_data = s0[72];
            float v94_data = ir1[6];
            ir1[6] = (v94_data + (v61_data * v92_data));
            float v97_data = s0[84];
            float v99_data = ir1[7];
            ir1[7] = (v99_data + (v61_data * v97_data));
          }
          if (v27_lead < 12) {
            float v105_data = r0[1];
            float v106_data = s0[1];
            float v108_data = ir1[0];
            ir1[0] = (v108_data + (v105_data * v106_data));
            float v111_data = s0[13];
            float v113_data = ir1[1];
            ir1[1] = (v113_data + (v105_data * v111_data));
            float v116_data = s0[25];
            float v118_data = ir1[2];
            ir1[2] = (v118_data + (v105_data * v116_data));
            float v121_data = s0[37];
            float v123_data = ir1[3];
            ir1[3] = (v123_data + (v105_data * v121_data));
            float v126_data = s0[49];
            float v128_data = ir1[4];
            ir1[4] = (v128_data + (v105_data * v126_data));
            float v131_data = s0[61];
            float v133_data = ir1[5];
            ir1[5] = (v133_data + (v105_data * v131_data));
            float v136_data = s0[73];
            float v138_data = ir1[6];
            ir1[6] = (v138_data + (v105_data * v136_data));
            float v141_data = s0[85];
            float v143_data = ir1[7];
            ir1[7] = (v143_data + (v105_data * v141_data));
          }
          if (v27_lead < 12) {
            float v149_data = r0[2];
            float v150_data = s0[2];
            float v152_data = ir1[0];
            ir1[0] = (v152_data + (v149_data * v150_data));
            float v155_data = s0[14];
            float v157_data = ir1[1];
            ir1[1] = (v157_data + (v149_data * v155_data));
            float v160_data = s0[26];
            float v162_data = ir1[2];
            ir1[2] = (v162_data + (v149_data * v160_data));
            float v165_data = s0[38];
            float v167_data = ir1[3];
            ir1[3] = (v167_data + (v149_data * v165_data));
            float v170_data = s0[50];
            float v172_data = ir1[4];
            ir1[4] = (v172_data + (v149_data * v170_data));
            float v175_data = s0[62];
            float v177_data = ir1[5];
            ir1[5] = (v177_data + (v149_data * v175_data));
            float v180_data = s0[74];
            float v182_data = ir1[6];
            ir1[6] = (v182_data + (v149_data * v180_data));
            float v185_data = s0[86];
            float v187_data = ir1[7];
            ir1[7] = (v187_data + (v149_data * v185_data));
          }
          if (v27_lead < 12) {
            float v193_data = r0[3];
            float v194_data = s0[3];
            float v196_data = ir1[0];
            ir1[0] = (v196_data + (v193_data * v194_data));
            float v199_data = s0[15];
            float v201_data = ir1[1];
            ir1[1] = (v201_data + (v193_data * v199_data));
            float v204_data = s0[27];
            float v206_data = ir1[2];
            ir1[2] = (v206_data + (v193_data * v204_data));
            float v209_data = s0[39];
            float v211_data = ir1[3];
            ir1[3] = (v211_data + (v193_data * v209_data));
            float v214_data = s0[51];
            float v216_data = ir1[4];
            ir1[4] = (v216_data + (v193_data * v214_data));
            float v219_data = s0[63];
            float v221_data = ir1[5];
            ir1[5] = (v221_data + (v193_data * v219_data));
            float v224_data = s0[75];
            float v226_data = ir1[6];
            ir1[6] = (v226_data + (v193_data * v224_data));
            float v229_data = s0[87];
            float v231_data = ir1[7];
            ir1[7] = (v231_data + (v193_data * v229_data));
          }
          if (v27_lead < 12) {
            float v237_data = r0[4];
            float v238_data = s0[4];
            float v240_data = ir1[0];
            ir1[0] = (v240_data + (v237_data * v238_data));
            float v243_data = s0[16];
            float v245_data = ir1[1];
            ir1[1] = (v245_data + (v237_data * v243_data));
            float v248_data = s0[28];
            float v250_data = ir1[2];
            ir1[2] = (v250_data + (v237_data * v248_data));
            float v253_data = s0[40];
            float v255_data = ir1[3];
            ir1[3] = (v255_data + (v237_data * v253_data));
            float v258_data = s0[52];
            float v260_data = ir1[4];
            ir1[4] = (v260_data + (v237_data * v258_data));
            float v263_data = s0[64];
            float v265_data = ir1[5];
            ir1[5] = (v265_data + (v237_data * v263_data));
            float v268_data = s0[76];
            float v270_data = ir1[6];
            ir1[6] = (v270_data + (v237_data * v268_data));
            float v273_data = s0[88];
            float v275_data = ir1[7];
            ir1[7] = (v275_data + (v237_data * v273_data));
          }
          if (v27_lead < 12) {
            float v281_data = r0[5];
            float v282_data = s0[5];
            float v284_data = ir1[0];
            ir1[0] = (v284_data + (v281_data * v282_data));
            float v287_data = s0[17];
            float v289_data = ir1[1];
            ir1[1] = (v289_data + (v281_data * v287_data));
            float v292_data = s0[29];
            float v294_data = ir1[2];
            ir1[2] = (v294_data + (v281_data * v292_data));
            float v297_data = s0[41];
            float v299_data = ir1[3];
            ir1[3] = (v299_data + (v281_data * v297_data));
            float v302_data = s0[53];
            float v304_data = ir1[4];
            ir1[4] = (v304_data + (v281_data * v302_data));
            float v307_data = s0[65];
            float v309_data = ir1[5];
            ir1[5] = (v309_data + (v281_data * v307_data));
            float v312_data = s0[77];
            float v314_data = ir1[6];
            ir1[6] = (v314_data + (v281_data * v312_data));
            float v317_data = s0[89];
            float v319_data = ir1[7];
            ir1[7] = (v319_data + (v281_data * v317_data));
          }
          if (v27_lead < 12) {
            float v325_data = r0[6];
            float v326_data = s0[6];
            float v328_data = ir1[0];
            ir1[0] = (v328_data + (v325_data * v326_data));
            float v331_data = s0[18];
            float v333_data = ir1[1];
            ir1[1] = (v333_data + (v325_data * v331_data));
            float v336_data = s0[30];
            float v338_data = ir1[2];
            ir1[2] = (v338_data + (v325_data * v336_data));
            float v341_data = s0[42];
            float v343_data = ir1[3];
            ir1[3] = (v343_data + (v325_data * v341_data));
            float v346_data = s0[54];
            float v348_data = ir1[4];
            ir1[4] = (v348_data + (v325_data * v346_data));
            float v351_data = s0[66];
            float v353_data = ir1[5];
            ir1[5] = (v353_data + (v325_data * v351_data));
            float v356_data = s0[78];
            float v358_data = ir1[6];
            ir1[6] = (v358_data + (v325_data * v356_data));
            float v361_data = s0[90];
            float v363_data = ir1[7];
            ir1[7] = (v363_data + (v325_data * v361_data));
          }
          if (v27_lead < 12) {
            float v369_data = r0[7];
            float v370_data = s0[7];
            float v372_data = ir1[0];
            ir1[0] = (v372_data + (v369_data * v370_data));
            float v375_data = s0[19];
            float v377_data = ir1[1];
            ir1[1] = (v377_data + (v369_data * v375_data));
            float v380_data = s0[31];
            float v382_data = ir1[2];
            ir1[2] = (v382_data + (v369_data * v380_data));
            float v385_data = s0[43];
            float v387_data = ir1[3];
            ir1[3] = (v387_data + (v369_data * v385_data));
            float v390_data = s0[55];
            float v392_data = ir1[4];
            ir1[4] = (v392_data + (v369_data * v390_data));
            float v395_data = s0[67];
            float v397_data = ir1[5];
            ir1[5] = (v397_data + (v369_data * v395_data));
            float v400_data = s0[79];
            float v402_data = ir1[6];
            ir1[6] = (v402_data + (v369_data * v400_data));
            float v405_data = s0[91];
            float v407_data = ir1[7];
            ir1[7] = (v407_data + (v369_data * v405_data));
          }
          if (v27_lead < 12) {
            float v413_data = r0[8];
            float v414_data = s0[8];
            float v416_data = ir1[0];
            ir1[0] = (v416_data + (v413_data * v414_data));
            float v419_data = s0[20];
            float v421_data = ir1[1];
            ir1[1] = (v421_data + (v413_data * v419_data));
            float v424_data = s0[32];
            float v426_data = ir1[2];
            ir1[2] = (v426_data + (v413_data * v424_data));
            float v429_data = s0[44];
            float v431_data = ir1[3];
            ir1[3] = (v431_data + (v413_data * v429_data));
            float v434_data = s0[56];
            float v436_data = ir1[4];
            ir1[4] = (v436_data + (v413_data * v434_data));
            float v439_data = s0[68];
            float v441_data = ir1[5];
            ir1[5] = (v441_data + (v413_data * v439_data));
            float v444_data = s0[80];
            float v446_data = ir1[6];
            ir1[6] = (v446_data + (v413_data * v444_data));
            float v449_data = s0[92];
            float v451_data = ir1[7];
            ir1[7] = (v451_data + (v413_data * v449_data));
          }
          if (v27_lead < 12) {
            float v457_data = r0[9];
            float v458_data = s0[9];
            float v460_data = ir1[0];
            ir1[0] = (v460_data + (v457_data * v458_data));
            float v463_data = s0[21];
            float v465_data = ir1[1];
            ir1[1] = (v465_data + (v457_data * v463_data));
            float v468_data = s0[33];
            float v470_data = ir1[2];
            ir1[2] = (v470_data + (v457_data * v468_data));
            float v473_data = s0[45];
            float v475_data = ir1[3];
            ir1[3] = (v475_data + (v457_data * v473_data));
            float v478_data = s0[57];
            float v480_data = ir1[4];
            ir1[4] = (v480_data + (v457_data * v478_data));
            float v483_data = s0[69];
            float v485_data = ir1[5];
            ir1[5] = (v485_data + (v457_data * v483_data));
            float v488_data = s0[81];
            float v490_data = ir1[6];
            ir1[6] = (v490_data + (v457_data * v488_data));
            float v493_data = s0[93];
            float v495_data = ir1[7];
            ir1[7] = (v495_data + (v457_data * v493_data));
          }
          if (v27_lead < 12) {
            float v501_data = r0[10];
            float v502_data = s0[10];
            float v504_data = ir1[0];
            ir1[0] = (v504_data + (v501_data * v502_data));
            float v507_data = s0[22];
            float v509_data = ir1[1];
            ir1[1] = (v509_data + (v501_data * v507_data));
            float v512_data = s0[34];
            float v514_data = ir1[2];
            ir1[2] = (v514_data + (v501_data * v512_data));
            float v517_data = s0[46];
            float v519_data = ir1[3];
            ir1[3] = (v519_data + (v501_data * v517_data));
            float v522_data = s0[58];
            float v524_data = ir1[4];
            ir1[4] = (v524_data + (v501_data * v522_data));
            float v527_data = s0[70];
            float v529_data = ir1[5];
            ir1[5] = (v529_data + (v501_data * v527_data));
            float v532_data = s0[82];
            float v534_data = ir1[6];
            ir1[6] = (v534_data + (v501_data * v532_data));
            float v537_data = s0[94];
            float v539_data = ir1[7];
            ir1[7] = (v539_data + (v501_data * v537_data));
          }
          if (v27_lead < 12) {
            float v545_data = r0[11];
            float v546_data = s0[11];
            float v548_data = ir1[0];
            ir1[0] = (v548_data + (v545_data * v546_data));
            float v551_data = s0[23];
            float v553_data = ir1[1];
            ir1[1] = (v553_data + (v545_data * v551_data));
            float v556_data = s0[35];
            float v558_data = ir1[2];
            ir1[2] = (v558_data + (v545_data * v556_data));
            float v561_data = s0[47];
            float v563_data = ir1[3];
            ir1[3] = (v563_data + (v545_data * v561_data));
            float v566_data = s0[59];
            float v568_data = ir1[4];
            ir1[4] = (v568_data + (v545_data * v566_data));
            float v571_data = s0[71];
            float v573_data = ir1[5];
            ir1[5] = (v573_data + (v545_data * v571_data));
            float v576_data = s0[83];
            float v578_data = ir1[6];
            ir1[6] = (v578_data + (v545_data * v576_data));
            float v581_data = s0[95];
            float v583_data = ir1[7];
            ir1[7] = (v583_data + (v545_data * v581_data));
          }
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v589_n1 = 0; v589_n1 < 8; ++v589_n1) {
              float v591_data = ir1[v589_n1];
              r1[v589_n1] = v591_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v599_i1 = 0; v599_i1 < 12; ++v599_i1) {
              float v607_data = __ldcg(&glb_m5[(v27_lead + (v599_i1 * 12))]);
              r4[v599_i1] = v607_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v27_lead < 12) {
            float v615_data = r2[0];
            float v616_data = s1[0];
            float v618_data = ir3[0];
            ir3[0] = (v618_data + (v615_data * v616_data));
            float v621_data = s1[12];
            float v623_data = ir3[1];
            ir3[1] = (v623_data + (v615_data * v621_data));
            float v626_data = s1[24];
            float v628_data = ir3[2];
            ir3[2] = (v628_data + (v615_data * v626_data));
            float v631_data = s1[36];
            float v633_data = ir3[3];
            ir3[3] = (v633_data + (v615_data * v631_data));
            float v636_data = s1[48];
            float v638_data = ir3[4];
            ir3[4] = (v638_data + (v615_data * v636_data));
            float v641_data = s1[60];
            float v643_data = ir3[5];
            ir3[5] = (v643_data + (v615_data * v641_data));
            float v646_data = s1[72];
            float v648_data = ir3[6];
            ir3[6] = (v648_data + (v615_data * v646_data));
            float v651_data = s1[84];
            float v653_data = ir3[7];
            ir3[7] = (v653_data + (v615_data * v651_data));
          }
          if (v27_lead < 12) {
            float v659_data = r2[1];
            float v660_data = s1[1];
            float v662_data = ir3[0];
            ir3[0] = (v662_data + (v659_data * v660_data));
            float v665_data = s1[13];
            float v667_data = ir3[1];
            ir3[1] = (v667_data + (v659_data * v665_data));
            float v670_data = s1[25];
            float v672_data = ir3[2];
            ir3[2] = (v672_data + (v659_data * v670_data));
            float v675_data = s1[37];
            float v677_data = ir3[3];
            ir3[3] = (v677_data + (v659_data * v675_data));
            float v680_data = s1[49];
            float v682_data = ir3[4];
            ir3[4] = (v682_data + (v659_data * v680_data));
            float v685_data = s1[61];
            float v687_data = ir3[5];
            ir3[5] = (v687_data + (v659_data * v685_data));
            float v690_data = s1[73];
            float v692_data = ir3[6];
            ir3[6] = (v692_data + (v659_data * v690_data));
            float v695_data = s1[85];
            float v697_data = ir3[7];
            ir3[7] = (v697_data + (v659_data * v695_data));
          }
          if (v27_lead < 12) {
            float v703_data = r2[2];
            float v704_data = s1[2];
            float v706_data = ir3[0];
            ir3[0] = (v706_data + (v703_data * v704_data));
            float v709_data = s1[14];
            float v711_data = ir3[1];
            ir3[1] = (v711_data + (v703_data * v709_data));
            float v714_data = s1[26];
            float v716_data = ir3[2];
            ir3[2] = (v716_data + (v703_data * v714_data));
            float v719_data = s1[38];
            float v721_data = ir3[3];
            ir3[3] = (v721_data + (v703_data * v719_data));
            float v724_data = s1[50];
            float v726_data = ir3[4];
            ir3[4] = (v726_data + (v703_data * v724_data));
            float v729_data = s1[62];
            float v731_data = ir3[5];
            ir3[5] = (v731_data + (v703_data * v729_data));
            float v734_data = s1[74];
            float v736_data = ir3[6];
            ir3[6] = (v736_data + (v703_data * v734_data));
            float v739_data = s1[86];
            float v741_data = ir3[7];
            ir3[7] = (v741_data + (v703_data * v739_data));
          }
          if (v27_lead < 12) {
            float v747_data = r2[3];
            float v748_data = s1[3];
            float v750_data = ir3[0];
            ir3[0] = (v750_data + (v747_data * v748_data));
            float v753_data = s1[15];
            float v755_data = ir3[1];
            ir3[1] = (v755_data + (v747_data * v753_data));
            float v758_data = s1[27];
            float v760_data = ir3[2];
            ir3[2] = (v760_data + (v747_data * v758_data));
            float v763_data = s1[39];
            float v765_data = ir3[3];
            ir3[3] = (v765_data + (v747_data * v763_data));
            float v768_data = s1[51];
            float v770_data = ir3[4];
            ir3[4] = (v770_data + (v747_data * v768_data));
            float v773_data = s1[63];
            float v775_data = ir3[5];
            ir3[5] = (v775_data + (v747_data * v773_data));
            float v778_data = s1[75];
            float v780_data = ir3[6];
            ir3[6] = (v780_data + (v747_data * v778_data));
            float v783_data = s1[87];
            float v785_data = ir3[7];
            ir3[7] = (v785_data + (v747_data * v783_data));
          }
          if (v27_lead < 12) {
            float v791_data = r2[4];
            float v792_data = s1[4];
            float v794_data = ir3[0];
            ir3[0] = (v794_data + (v791_data * v792_data));
            float v797_data = s1[16];
            float v799_data = ir3[1];
            ir3[1] = (v799_data + (v791_data * v797_data));
            float v802_data = s1[28];
            float v804_data = ir3[2];
            ir3[2] = (v804_data + (v791_data * v802_data));
            float v807_data = s1[40];
            float v809_data = ir3[3];
            ir3[3] = (v809_data + (v791_data * v807_data));
            float v812_data = s1[52];
            float v814_data = ir3[4];
            ir3[4] = (v814_data + (v791_data * v812_data));
            float v817_data = s1[64];
            float v819_data = ir3[5];
            ir3[5] = (v819_data + (v791_data * v817_data));
            float v822_data = s1[76];
            float v824_data = ir3[6];
            ir3[6] = (v824_data + (v791_data * v822_data));
            float v827_data = s1[88];
            float v829_data = ir3[7];
            ir3[7] = (v829_data + (v791_data * v827_data));
          }
          if (v27_lead < 12) {
            float v835_data = r2[5];
            float v836_data = s1[5];
            float v838_data = ir3[0];
            ir3[0] = (v838_data + (v835_data * v836_data));
            float v841_data = s1[17];
            float v843_data = ir3[1];
            ir3[1] = (v843_data + (v835_data * v841_data));
            float v846_data = s1[29];
            float v848_data = ir3[2];
            ir3[2] = (v848_data + (v835_data * v846_data));
            float v851_data = s1[41];
            float v853_data = ir3[3];
            ir3[3] = (v853_data + (v835_data * v851_data));
            float v856_data = s1[53];
            float v858_data = ir3[4];
            ir3[4] = (v858_data + (v835_data * v856_data));
            float v861_data = s1[65];
            float v863_data = ir3[5];
            ir3[5] = (v863_data + (v835_data * v861_data));
            float v866_data = s1[77];
            float v868_data = ir3[6];
            ir3[6] = (v868_data + (v835_data * v866_data));
            float v871_data = s1[89];
            float v873_data = ir3[7];
            ir3[7] = (v873_data + (v835_data * v871_data));
          }
          if (v27_lead < 12) {
            float v879_data = r2[6];
            float v880_data = s1[6];
            float v882_data = ir3[0];
            ir3[0] = (v882_data + (v879_data * v880_data));
            float v885_data = s1[18];
            float v887_data = ir3[1];
            ir3[1] = (v887_data + (v879_data * v885_data));
            float v890_data = s1[30];
            float v892_data = ir3[2];
            ir3[2] = (v892_data + (v879_data * v890_data));
            float v895_data = s1[42];
            float v897_data = ir3[3];
            ir3[3] = (v897_data + (v879_data * v895_data));
            float v900_data = s1[54];
            float v902_data = ir3[4];
            ir3[4] = (v902_data + (v879_data * v900_data));
            float v905_data = s1[66];
            float v907_data = ir3[5];
            ir3[5] = (v907_data + (v879_data * v905_data));
            float v910_data = s1[78];
            float v912_data = ir3[6];
            ir3[6] = (v912_data + (v879_data * v910_data));
            float v915_data = s1[90];
            float v917_data = ir3[7];
            ir3[7] = (v917_data + (v879_data * v915_data));
          }
          if (v27_lead < 12) {
            float v923_data = r2[7];
            float v924_data = s1[7];
            float v926_data = ir3[0];
            ir3[0] = (v926_data + (v923_data * v924_data));
            float v929_data = s1[19];
            float v931_data = ir3[1];
            ir3[1] = (v931_data + (v923_data * v929_data));
            float v934_data = s1[31];
            float v936_data = ir3[2];
            ir3[2] = (v936_data + (v923_data * v934_data));
            float v939_data = s1[43];
            float v941_data = ir3[3];
            ir3[3] = (v941_data + (v923_data * v939_data));
            float v944_data = s1[55];
            float v946_data = ir3[4];
            ir3[4] = (v946_data + (v923_data * v944_data));
            float v949_data = s1[67];
            float v951_data = ir3[5];
            ir3[5] = (v951_data + (v923_data * v949_data));
            float v954_data = s1[79];
            float v956_data = ir3[6];
            ir3[6] = (v956_data + (v923_data * v954_data));
            float v959_data = s1[91];
            float v961_data = ir3[7];
            ir3[7] = (v961_data + (v923_data * v959_data));
          }
          if (v27_lead < 12) {
            float v967_data = r2[8];
            float v968_data = s1[8];
            float v970_data = ir3[0];
            ir3[0] = (v970_data + (v967_data * v968_data));
            float v973_data = s1[20];
            float v975_data = ir3[1];
            ir3[1] = (v975_data + (v967_data * v973_data));
            float v978_data = s1[32];
            float v980_data = ir3[2];
            ir3[2] = (v980_data + (v967_data * v978_data));
            float v983_data = s1[44];
            float v985_data = ir3[3];
            ir3[3] = (v985_data + (v967_data * v983_data));
            float v988_data = s1[56];
            float v990_data = ir3[4];
            ir3[4] = (v990_data + (v967_data * v988_data));
            float v993_data = s1[68];
            float v995_data = ir3[5];
            ir3[5] = (v995_data + (v967_data * v993_data));
            float v998_data = s1[80];
            float v1000_data = ir3[6];
            ir3[6] = (v1000_data + (v967_data * v998_data));
            float v1003_data = s1[92];
            float v1005_data = ir3[7];
            ir3[7] = (v1005_data + (v967_data * v1003_data));
          }
          if (v27_lead < 12) {
            float v1011_data = r2[9];
            float v1012_data = s1[9];
            float v1014_data = ir3[0];
            ir3[0] = (v1014_data + (v1011_data * v1012_data));
            float v1017_data = s1[21];
            float v1019_data = ir3[1];
            ir3[1] = (v1019_data + (v1011_data * v1017_data));
            float v1022_data = s1[33];
            float v1024_data = ir3[2];
            ir3[2] = (v1024_data + (v1011_data * v1022_data));
            float v1027_data = s1[45];
            float v1029_data = ir3[3];
            ir3[3] = (v1029_data + (v1011_data * v1027_data));
            float v1032_data = s1[57];
            float v1034_data = ir3[4];
            ir3[4] = (v1034_data + (v1011_data * v1032_data));
            float v1037_data = s1[69];
            float v1039_data = ir3[5];
            ir3[5] = (v1039_data + (v1011_data * v1037_data));
            float v1042_data = s1[81];
            float v1044_data = ir3[6];
            ir3[6] = (v1044_data + (v1011_data * v1042_data));
            float v1047_data = s1[93];
            float v1049_data = ir3[7];
            ir3[7] = (v1049_data + (v1011_data * v1047_data));
          }
          if (v27_lead < 12) {
            float v1055_data = r2[10];
            float v1056_data = s1[10];
            float v1058_data = ir3[0];
            ir3[0] = (v1058_data + (v1055_data * v1056_data));
            float v1061_data = s1[22];
            float v1063_data = ir3[1];
            ir3[1] = (v1063_data + (v1055_data * v1061_data));
            float v1066_data = s1[34];
            float v1068_data = ir3[2];
            ir3[2] = (v1068_data + (v1055_data * v1066_data));
            float v1071_data = s1[46];
            float v1073_data = ir3[3];
            ir3[3] = (v1073_data + (v1055_data * v1071_data));
            float v1076_data = s1[58];
            float v1078_data = ir3[4];
            ir3[4] = (v1078_data + (v1055_data * v1076_data));
            float v1081_data = s1[70];
            float v1083_data = ir3[5];
            ir3[5] = (v1083_data + (v1055_data * v1081_data));
            float v1086_data = s1[82];
            float v1088_data = ir3[6];
            ir3[6] = (v1088_data + (v1055_data * v1086_data));
            float v1091_data = s1[94];
            float v1093_data = ir3[7];
            ir3[7] = (v1093_data + (v1055_data * v1091_data));
          }
          if (v27_lead < 12) {
            float v1099_data = r2[11];
            float v1100_data = s1[11];
            float v1102_data = ir3[0];
            ir3[0] = (v1102_data + (v1099_data * v1100_data));
            float v1105_data = s1[23];
            float v1107_data = ir3[1];
            ir3[1] = (v1107_data + (v1099_data * v1105_data));
            float v1110_data = s1[35];
            float v1112_data = ir3[2];
            ir3[2] = (v1112_data + (v1099_data * v1110_data));
            float v1115_data = s1[47];
            float v1117_data = ir3[3];
            ir3[3] = (v1117_data + (v1099_data * v1115_data));
            float v1120_data = s1[59];
            float v1122_data = ir3[4];
            ir3[4] = (v1122_data + (v1099_data * v1120_data));
            float v1125_data = s1[71];
            float v1127_data = ir3[5];
            ir3[5] = (v1127_data + (v1099_data * v1125_data));
            float v1130_data = s1[83];
            float v1132_data = ir3[6];
            ir3[6] = (v1132_data + (v1099_data * v1130_data));
            float v1135_data = s1[95];
            float v1137_data = ir3[7];
            ir3[7] = (v1137_data + (v1099_data * v1135_data));
          }
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v1143_n1 = 0; v1143_n1 < 8; ++v1143_n1) {
              float v1145_data = ir3[v1143_n1];
              float v1147_data = r1[v1143_n1];
              r3[v1143_n1] = (v1147_data + v1145_data);
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s2 = load{g>s}(glb_m6[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v1156_i1 = 0; v1156_i1 < 12; ++v1156_i1) {
              float v1164_data = __ldcg(&glb_m7[(v27_lead + (v1156_i1 * 12))]);
              r6[v1156_i1] = v1164_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v27_lead < 12) {
            float v1172_data = r4[0];
            float v1173_data = s2[0];
            float v1175_data = ir5[0];
            ir5[0] = (v1175_data + (v1172_data * v1173_data));
            float v1178_data = s2[12];
            float v1180_data = ir5[1];
            ir5[1] = (v1180_data + (v1172_data * v1178_data));
            float v1183_data = s2[24];
            float v1185_data = ir5[2];
            ir5[2] = (v1185_data + (v1172_data * v1183_data));
            float v1188_data = s2[36];
            float v1190_data = ir5[3];
            ir5[3] = (v1190_data + (v1172_data * v1188_data));
            float v1193_data = s2[48];
            float v1195_data = ir5[4];
            ir5[4] = (v1195_data + (v1172_data * v1193_data));
            float v1198_data = s2[60];
            float v1200_data = ir5[5];
            ir5[5] = (v1200_data + (v1172_data * v1198_data));
            float v1203_data = s2[72];
            float v1205_data = ir5[6];
            ir5[6] = (v1205_data + (v1172_data * v1203_data));
            float v1208_data = s2[84];
            float v1210_data = ir5[7];
            ir5[7] = (v1210_data + (v1172_data * v1208_data));
          }
          if (v27_lead < 12) {
            float v1216_data = r4[1];
            float v1217_data = s2[1];
            float v1219_data = ir5[0];
            ir5[0] = (v1219_data + (v1216_data * v1217_data));
            float v1222_data = s2[13];
            float v1224_data = ir5[1];
            ir5[1] = (v1224_data + (v1216_data * v1222_data));
            float v1227_data = s2[25];
            float v1229_data = ir5[2];
            ir5[2] = (v1229_data + (v1216_data * v1227_data));
            float v1232_data = s2[37];
            float v1234_data = ir5[3];
            ir5[3] = (v1234_data + (v1216_data * v1232_data));
            float v1237_data = s2[49];
            float v1239_data = ir5[4];
            ir5[4] = (v1239_data + (v1216_data * v1237_data));
            float v1242_data = s2[61];
            float v1244_data = ir5[5];
            ir5[5] = (v1244_data + (v1216_data * v1242_data));
            float v1247_data = s2[73];
            float v1249_data = ir5[6];
            ir5[6] = (v1249_data + (v1216_data * v1247_data));
            float v1252_data = s2[85];
            float v1254_data = ir5[7];
            ir5[7] = (v1254_data + (v1216_data * v1252_data));
          }
          if (v27_lead < 12) {
            float v1260_data = r4[2];
            float v1261_data = s2[2];
            float v1263_data = ir5[0];
            ir5[0] = (v1263_data + (v1260_data * v1261_data));
            float v1266_data = s2[14];
            float v1268_data = ir5[1];
            ir5[1] = (v1268_data + (v1260_data * v1266_data));
            float v1271_data = s2[26];
            float v1273_data = ir5[2];
            ir5[2] = (v1273_data + (v1260_data * v1271_data));
            float v1276_data = s2[38];
            float v1278_data = ir5[3];
            ir5[3] = (v1278_data + (v1260_data * v1276_data));
            float v1281_data = s2[50];
            float v1283_data = ir5[4];
            ir5[4] = (v1283_data + (v1260_data * v1281_data));
            float v1286_data = s2[62];
            float v1288_data = ir5[5];
            ir5[5] = (v1288_data + (v1260_data * v1286_data));
            float v1291_data = s2[74];
            float v1293_data = ir5[6];
            ir5[6] = (v1293_data + (v1260_data * v1291_data));
            float v1296_data = s2[86];
            float v1298_data = ir5[7];
            ir5[7] = (v1298_data + (v1260_data * v1296_data));
          }
          if (v27_lead < 12) {
            float v1304_data = r4[3];
            float v1305_data = s2[3];
            float v1307_data = ir5[0];
            ir5[0] = (v1307_data + (v1304_data * v1305_data));
            float v1310_data = s2[15];
            float v1312_data = ir5[1];
            ir5[1] = (v1312_data + (v1304_data * v1310_data));
            float v1315_data = s2[27];
            float v1317_data = ir5[2];
            ir5[2] = (v1317_data + (v1304_data * v1315_data));
            float v1320_data = s2[39];
            float v1322_data = ir5[3];
            ir5[3] = (v1322_data + (v1304_data * v1320_data));
            float v1325_data = s2[51];
            float v1327_data = ir5[4];
            ir5[4] = (v1327_data + (v1304_data * v1325_data));
            float v1330_data = s2[63];
            float v1332_data = ir5[5];
            ir5[5] = (v1332_data + (v1304_data * v1330_data));
            float v1335_data = s2[75];
            float v1337_data = ir5[6];
            ir5[6] = (v1337_data + (v1304_data * v1335_data));
            float v1340_data = s2[87];
            float v1342_data = ir5[7];
            ir5[7] = (v1342_data + (v1304_data * v1340_data));
          }
          if (v27_lead < 12) {
            float v1348_data = r4[4];
            float v1349_data = s2[4];
            float v1351_data = ir5[0];
            ir5[0] = (v1351_data + (v1348_data * v1349_data));
            float v1354_data = s2[16];
            float v1356_data = ir5[1];
            ir5[1] = (v1356_data + (v1348_data * v1354_data));
            float v1359_data = s2[28];
            float v1361_data = ir5[2];
            ir5[2] = (v1361_data + (v1348_data * v1359_data));
            float v1364_data = s2[40];
            float v1366_data = ir5[3];
            ir5[3] = (v1366_data + (v1348_data * v1364_data));
            float v1369_data = s2[52];
            float v1371_data = ir5[4];
            ir5[4] = (v1371_data + (v1348_data * v1369_data));
            float v1374_data = s2[64];
            float v1376_data = ir5[5];
            ir5[5] = (v1376_data + (v1348_data * v1374_data));
            float v1379_data = s2[76];
            float v1381_data = ir5[6];
            ir5[6] = (v1381_data + (v1348_data * v1379_data));
            float v1384_data = s2[88];
            float v1386_data = ir5[7];
            ir5[7] = (v1386_data + (v1348_data * v1384_data));
          }
          if (v27_lead < 12) {
            float v1392_data = r4[5];
            float v1393_data = s2[5];
            float v1395_data = ir5[0];
            ir5[0] = (v1395_data + (v1392_data * v1393_data));
            float v1398_data = s2[17];
            float v1400_data = ir5[1];
            ir5[1] = (v1400_data + (v1392_data * v1398_data));
            float v1403_data = s2[29];
            float v1405_data = ir5[2];
            ir5[2] = (v1405_data + (v1392_data * v1403_data));
            float v1408_data = s2[41];
            float v1410_data = ir5[3];
            ir5[3] = (v1410_data + (v1392_data * v1408_data));
            float v1413_data = s2[53];
            float v1415_data = ir5[4];
            ir5[4] = (v1415_data + (v1392_data * v1413_data));
            float v1418_data = s2[65];
            float v1420_data = ir5[5];
            ir5[5] = (v1420_data + (v1392_data * v1418_data));
            float v1423_data = s2[77];
            float v1425_data = ir5[6];
            ir5[6] = (v1425_data + (v1392_data * v1423_data));
            float v1428_data = s2[89];
            float v1430_data = ir5[7];
            ir5[7] = (v1430_data + (v1392_data * v1428_data));
          }
          if (v27_lead < 12) {
            float v1436_data = r4[6];
            float v1437_data = s2[6];
            float v1439_data = ir5[0];
            ir5[0] = (v1439_data + (v1436_data * v1437_data));
            float v1442_data = s2[18];
            float v1444_data = ir5[1];
            ir5[1] = (v1444_data + (v1436_data * v1442_data));
            float v1447_data = s2[30];
            float v1449_data = ir5[2];
            ir5[2] = (v1449_data + (v1436_data * v1447_data));
            float v1452_data = s2[42];
            float v1454_data = ir5[3];
            ir5[3] = (v1454_data + (v1436_data * v1452_data));
            float v1457_data = s2[54];
            float v1459_data = ir5[4];
            ir5[4] = (v1459_data + (v1436_data * v1457_data));
            float v1462_data = s2[66];
            float v1464_data = ir5[5];
            ir5[5] = (v1464_data + (v1436_data * v1462_data));
            float v1467_data = s2[78];
            float v1469_data = ir5[6];
            ir5[6] = (v1469_data + (v1436_data * v1467_data));
            float v1472_data = s2[90];
            float v1474_data = ir5[7];
            ir5[7] = (v1474_data + (v1436_data * v1472_data));
          }
          if (v27_lead < 12) {
            float v1480_data = r4[7];
            float v1481_data = s2[7];
            float v1483_data = ir5[0];
            ir5[0] = (v1483_data + (v1480_data * v1481_data));
            float v1486_data = s2[19];
            float v1488_data = ir5[1];
            ir5[1] = (v1488_data + (v1480_data * v1486_data));
            float v1491_data = s2[31];
            float v1493_data = ir5[2];
            ir5[2] = (v1493_data + (v1480_data * v1491_data));
            float v1496_data = s2[43];
            float v1498_data = ir5[3];
            ir5[3] = (v1498_data + (v1480_data * v1496_data));
            float v1501_data = s2[55];
            float v1503_data = ir5[4];
            ir5[4] = (v1503_data + (v1480_data * v1501_data));
            float v1506_data = s2[67];
            float v1508_data = ir5[5];
            ir5[5] = (v1508_data + (v1480_data * v1506_data));
            float v1511_data = s2[79];
            float v1513_data = ir5[6];
            ir5[6] = (v1513_data + (v1480_data * v1511_data));
            float v1516_data = s2[91];
            float v1518_data = ir5[7];
            ir5[7] = (v1518_data + (v1480_data * v1516_data));
          }
          if (v27_lead < 12) {
            float v1524_data = r4[8];
            float v1525_data = s2[8];
            float v1527_data = ir5[0];
            ir5[0] = (v1527_data + (v1524_data * v1525_data));
            float v1530_data = s2[20];
            float v1532_data = ir5[1];
            ir5[1] = (v1532_data + (v1524_data * v1530_data));
            float v1535_data = s2[32];
            float v1537_data = ir5[2];
            ir5[2] = (v1537_data + (v1524_data * v1535_data));
            float v1540_data = s2[44];
            float v1542_data = ir5[3];
            ir5[3] = (v1542_data + (v1524_data * v1540_data));
            float v1545_data = s2[56];
            float v1547_data = ir5[4];
            ir5[4] = (v1547_data + (v1524_data * v1545_data));
            float v1550_data = s2[68];
            float v1552_data = ir5[5];
            ir5[5] = (v1552_data + (v1524_data * v1550_data));
            float v1555_data = s2[80];
            float v1557_data = ir5[6];
            ir5[6] = (v1557_data + (v1524_data * v1555_data));
            float v1560_data = s2[92];
            float v1562_data = ir5[7];
            ir5[7] = (v1562_data + (v1524_data * v1560_data));
          }
          if (v27_lead < 12) {
            float v1568_data = r4[9];
            float v1569_data = s2[9];
            float v1571_data = ir5[0];
            ir5[0] = (v1571_data + (v1568_data * v1569_data));
            float v1574_data = s2[21];
            float v1576_data = ir5[1];
            ir5[1] = (v1576_data + (v1568_data * v1574_data));
            float v1579_data = s2[33];
            float v1581_data = ir5[2];
            ir5[2] = (v1581_data + (v1568_data * v1579_data));
            float v1584_data = s2[45];
            float v1586_data = ir5[3];
            ir5[3] = (v1586_data + (v1568_data * v1584_data));
            float v1589_data = s2[57];
            float v1591_data = ir5[4];
            ir5[4] = (v1591_data + (v1568_data * v1589_data));
            float v1594_data = s2[69];
            float v1596_data = ir5[5];
            ir5[5] = (v1596_data + (v1568_data * v1594_data));
            float v1599_data = s2[81];
            float v1601_data = ir5[6];
            ir5[6] = (v1601_data + (v1568_data * v1599_data));
            float v1604_data = s2[93];
            float v1606_data = ir5[7];
            ir5[7] = (v1606_data + (v1568_data * v1604_data));
          }
          if (v27_lead < 12) {
            float v1612_data = r4[10];
            float v1613_data = s2[10];
            float v1615_data = ir5[0];
            ir5[0] = (v1615_data + (v1612_data * v1613_data));
            float v1618_data = s2[22];
            float v1620_data = ir5[1];
            ir5[1] = (v1620_data + (v1612_data * v1618_data));
            float v1623_data = s2[34];
            float v1625_data = ir5[2];
            ir5[2] = (v1625_data + (v1612_data * v1623_data));
            float v1628_data = s2[46];
            float v1630_data = ir5[3];
            ir5[3] = (v1630_data + (v1612_data * v1628_data));
            float v1633_data = s2[58];
            float v1635_data = ir5[4];
            ir5[4] = (v1635_data + (v1612_data * v1633_data));
            float v1638_data = s2[70];
            float v1640_data = ir5[5];
            ir5[5] = (v1640_data + (v1612_data * v1638_data));
            float v1643_data = s2[82];
            float v1645_data = ir5[6];
            ir5[6] = (v1645_data + (v1612_data * v1643_data));
            float v1648_data = s2[94];
            float v1650_data = ir5[7];
            ir5[7] = (v1650_data + (v1612_data * v1648_data));
          }
          if (v27_lead < 12) {
            float v1656_data = r4[11];
            float v1657_data = s2[11];
            float v1659_data = ir5[0];
            ir5[0] = (v1659_data + (v1656_data * v1657_data));
            float v1662_data = s2[23];
            float v1664_data = ir5[1];
            ir5[1] = (v1664_data + (v1656_data * v1662_data));
            float v1667_data = s2[35];
            float v1669_data = ir5[2];
            ir5[2] = (v1669_data + (v1656_data * v1667_data));
            float v1672_data = s2[47];
            float v1674_data = ir5[3];
            ir5[3] = (v1674_data + (v1656_data * v1672_data));
            float v1677_data = s2[59];
            float v1679_data = ir5[4];
            ir5[4] = (v1679_data + (v1656_data * v1677_data));
            float v1682_data = s2[71];
            float v1684_data = ir5[5];
            ir5[5] = (v1684_data + (v1656_data * v1682_data));
            float v1687_data = s2[83];
            float v1689_data = ir5[6];
            ir5[6] = (v1689_data + (v1656_data * v1687_data));
            float v1692_data = s2[95];
            float v1694_data = ir5[7];
            ir5[7] = (v1694_data + (v1656_data * v1692_data));
          }
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v1700_n1 = 0; v1700_n1 < 8; ++v1700_n1) {
              float v1702_data = ir5[v1700_n1];
              float v1704_data = r3[v1700_n1];
              r5[v1700_n1] = (v1704_data + v1702_data);
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
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
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v27_lead < 12) {
            float v1714_data = r6[0];
            float v1715_data = s3[0];
            float v1717_data = ir7[0];
            ir7[0] = (v1717_data + (v1714_data * v1715_data));
            float v1720_data = s3[12];
            float v1722_data = ir7[1];
            ir7[1] = (v1722_data + (v1714_data * v1720_data));
            float v1725_data = s3[24];
            float v1727_data = ir7[2];
            ir7[2] = (v1727_data + (v1714_data * v1725_data));
            float v1730_data = s3[36];
            float v1732_data = ir7[3];
            ir7[3] = (v1732_data + (v1714_data * v1730_data));
            float v1735_data = s3[48];
            float v1737_data = ir7[4];
            ir7[4] = (v1737_data + (v1714_data * v1735_data));
            float v1740_data = s3[60];
            float v1742_data = ir7[5];
            ir7[5] = (v1742_data + (v1714_data * v1740_data));
            float v1745_data = s3[72];
            float v1747_data = ir7[6];
            ir7[6] = (v1747_data + (v1714_data * v1745_data));
            float v1750_data = s3[84];
            float v1752_data = ir7[7];
            ir7[7] = (v1752_data + (v1714_data * v1750_data));
          }
          if (v27_lead < 12) {
            float v1758_data = r6[1];
            float v1759_data = s3[1];
            float v1761_data = ir7[0];
            ir7[0] = (v1761_data + (v1758_data * v1759_data));
            float v1764_data = s3[13];
            float v1766_data = ir7[1];
            ir7[1] = (v1766_data + (v1758_data * v1764_data));
            float v1769_data = s3[25];
            float v1771_data = ir7[2];
            ir7[2] = (v1771_data + (v1758_data * v1769_data));
            float v1774_data = s3[37];
            float v1776_data = ir7[3];
            ir7[3] = (v1776_data + (v1758_data * v1774_data));
            float v1779_data = s3[49];
            float v1781_data = ir7[4];
            ir7[4] = (v1781_data + (v1758_data * v1779_data));
            float v1784_data = s3[61];
            float v1786_data = ir7[5];
            ir7[5] = (v1786_data + (v1758_data * v1784_data));
            float v1789_data = s3[73];
            float v1791_data = ir7[6];
            ir7[6] = (v1791_data + (v1758_data * v1789_data));
            float v1794_data = s3[85];
            float v1796_data = ir7[7];
            ir7[7] = (v1796_data + (v1758_data * v1794_data));
          }
          if (v27_lead < 12) {
            float v1802_data = r6[2];
            float v1803_data = s3[2];
            float v1805_data = ir7[0];
            ir7[0] = (v1805_data + (v1802_data * v1803_data));
            float v1808_data = s3[14];
            float v1810_data = ir7[1];
            ir7[1] = (v1810_data + (v1802_data * v1808_data));
            float v1813_data = s3[26];
            float v1815_data = ir7[2];
            ir7[2] = (v1815_data + (v1802_data * v1813_data));
            float v1818_data = s3[38];
            float v1820_data = ir7[3];
            ir7[3] = (v1820_data + (v1802_data * v1818_data));
            float v1823_data = s3[50];
            float v1825_data = ir7[4];
            ir7[4] = (v1825_data + (v1802_data * v1823_data));
            float v1828_data = s3[62];
            float v1830_data = ir7[5];
            ir7[5] = (v1830_data + (v1802_data * v1828_data));
            float v1833_data = s3[74];
            float v1835_data = ir7[6];
            ir7[6] = (v1835_data + (v1802_data * v1833_data));
            float v1838_data = s3[86];
            float v1840_data = ir7[7];
            ir7[7] = (v1840_data + (v1802_data * v1838_data));
          }
          if (v27_lead < 12) {
            float v1846_data = r6[3];
            float v1847_data = s3[3];
            float v1849_data = ir7[0];
            ir7[0] = (v1849_data + (v1846_data * v1847_data));
            float v1852_data = s3[15];
            float v1854_data = ir7[1];
            ir7[1] = (v1854_data + (v1846_data * v1852_data));
            float v1857_data = s3[27];
            float v1859_data = ir7[2];
            ir7[2] = (v1859_data + (v1846_data * v1857_data));
            float v1862_data = s3[39];
            float v1864_data = ir7[3];
            ir7[3] = (v1864_data + (v1846_data * v1862_data));
            float v1867_data = s3[51];
            float v1869_data = ir7[4];
            ir7[4] = (v1869_data + (v1846_data * v1867_data));
            float v1872_data = s3[63];
            float v1874_data = ir7[5];
            ir7[5] = (v1874_data + (v1846_data * v1872_data));
            float v1877_data = s3[75];
            float v1879_data = ir7[6];
            ir7[6] = (v1879_data + (v1846_data * v1877_data));
            float v1882_data = s3[87];
            float v1884_data = ir7[7];
            ir7[7] = (v1884_data + (v1846_data * v1882_data));
          }
          if (v27_lead < 12) {
            float v1890_data = r6[4];
            float v1891_data = s3[4];
            float v1893_data = ir7[0];
            ir7[0] = (v1893_data + (v1890_data * v1891_data));
            float v1896_data = s3[16];
            float v1898_data = ir7[1];
            ir7[1] = (v1898_data + (v1890_data * v1896_data));
            float v1901_data = s3[28];
            float v1903_data = ir7[2];
            ir7[2] = (v1903_data + (v1890_data * v1901_data));
            float v1906_data = s3[40];
            float v1908_data = ir7[3];
            ir7[3] = (v1908_data + (v1890_data * v1906_data));
            float v1911_data = s3[52];
            float v1913_data = ir7[4];
            ir7[4] = (v1913_data + (v1890_data * v1911_data));
            float v1916_data = s3[64];
            float v1918_data = ir7[5];
            ir7[5] = (v1918_data + (v1890_data * v1916_data));
            float v1921_data = s3[76];
            float v1923_data = ir7[6];
            ir7[6] = (v1923_data + (v1890_data * v1921_data));
            float v1926_data = s3[88];
            float v1928_data = ir7[7];
            ir7[7] = (v1928_data + (v1890_data * v1926_data));
          }
          if (v27_lead < 12) {
            float v1934_data = r6[5];
            float v1935_data = s3[5];
            float v1937_data = ir7[0];
            ir7[0] = (v1937_data + (v1934_data * v1935_data));
            float v1940_data = s3[17];
            float v1942_data = ir7[1];
            ir7[1] = (v1942_data + (v1934_data * v1940_data));
            float v1945_data = s3[29];
            float v1947_data = ir7[2];
            ir7[2] = (v1947_data + (v1934_data * v1945_data));
            float v1950_data = s3[41];
            float v1952_data = ir7[3];
            ir7[3] = (v1952_data + (v1934_data * v1950_data));
            float v1955_data = s3[53];
            float v1957_data = ir7[4];
            ir7[4] = (v1957_data + (v1934_data * v1955_data));
            float v1960_data = s3[65];
            float v1962_data = ir7[5];
            ir7[5] = (v1962_data + (v1934_data * v1960_data));
            float v1965_data = s3[77];
            float v1967_data = ir7[6];
            ir7[6] = (v1967_data + (v1934_data * v1965_data));
            float v1970_data = s3[89];
            float v1972_data = ir7[7];
            ir7[7] = (v1972_data + (v1934_data * v1970_data));
          }
          if (v27_lead < 12) {
            float v1978_data = r6[6];
            float v1979_data = s3[6];
            float v1981_data = ir7[0];
            ir7[0] = (v1981_data + (v1978_data * v1979_data));
            float v1984_data = s3[18];
            float v1986_data = ir7[1];
            ir7[1] = (v1986_data + (v1978_data * v1984_data));
            float v1989_data = s3[30];
            float v1991_data = ir7[2];
            ir7[2] = (v1991_data + (v1978_data * v1989_data));
            float v1994_data = s3[42];
            float v1996_data = ir7[3];
            ir7[3] = (v1996_data + (v1978_data * v1994_data));
            float v1999_data = s3[54];
            float v2001_data = ir7[4];
            ir7[4] = (v2001_data + (v1978_data * v1999_data));
            float v2004_data = s3[66];
            float v2006_data = ir7[5];
            ir7[5] = (v2006_data + (v1978_data * v2004_data));
            float v2009_data = s3[78];
            float v2011_data = ir7[6];
            ir7[6] = (v2011_data + (v1978_data * v2009_data));
            float v2014_data = s3[90];
            float v2016_data = ir7[7];
            ir7[7] = (v2016_data + (v1978_data * v2014_data));
          }
          if (v27_lead < 12) {
            float v2022_data = r6[7];
            float v2023_data = s3[7];
            float v2025_data = ir7[0];
            ir7[0] = (v2025_data + (v2022_data * v2023_data));
            float v2028_data = s3[19];
            float v2030_data = ir7[1];
            ir7[1] = (v2030_data + (v2022_data * v2028_data));
            float v2033_data = s3[31];
            float v2035_data = ir7[2];
            ir7[2] = (v2035_data + (v2022_data * v2033_data));
            float v2038_data = s3[43];
            float v2040_data = ir7[3];
            ir7[3] = (v2040_data + (v2022_data * v2038_data));
            float v2043_data = s3[55];
            float v2045_data = ir7[4];
            ir7[4] = (v2045_data + (v2022_data * v2043_data));
            float v2048_data = s3[67];
            float v2050_data = ir7[5];
            ir7[5] = (v2050_data + (v2022_data * v2048_data));
            float v2053_data = s3[79];
            float v2055_data = ir7[6];
            ir7[6] = (v2055_data + (v2022_data * v2053_data));
            float v2058_data = s3[91];
            float v2060_data = ir7[7];
            ir7[7] = (v2060_data + (v2022_data * v2058_data));
          }
          if (v27_lead < 12) {
            float v2066_data = r6[8];
            float v2067_data = s3[8];
            float v2069_data = ir7[0];
            ir7[0] = (v2069_data + (v2066_data * v2067_data));
            float v2072_data = s3[20];
            float v2074_data = ir7[1];
            ir7[1] = (v2074_data + (v2066_data * v2072_data));
            float v2077_data = s3[32];
            float v2079_data = ir7[2];
            ir7[2] = (v2079_data + (v2066_data * v2077_data));
            float v2082_data = s3[44];
            float v2084_data = ir7[3];
            ir7[3] = (v2084_data + (v2066_data * v2082_data));
            float v2087_data = s3[56];
            float v2089_data = ir7[4];
            ir7[4] = (v2089_data + (v2066_data * v2087_data));
            float v2092_data = s3[68];
            float v2094_data = ir7[5];
            ir7[5] = (v2094_data + (v2066_data * v2092_data));
            float v2097_data = s3[80];
            float v2099_data = ir7[6];
            ir7[6] = (v2099_data + (v2066_data * v2097_data));
            float v2102_data = s3[92];
            float v2104_data = ir7[7];
            ir7[7] = (v2104_data + (v2066_data * v2102_data));
          }
          if (v27_lead < 12) {
            float v2110_data = r6[9];
            float v2111_data = s3[9];
            float v2113_data = ir7[0];
            ir7[0] = (v2113_data + (v2110_data * v2111_data));
            float v2116_data = s3[21];
            float v2118_data = ir7[1];
            ir7[1] = (v2118_data + (v2110_data * v2116_data));
            float v2121_data = s3[33];
            float v2123_data = ir7[2];
            ir7[2] = (v2123_data + (v2110_data * v2121_data));
            float v2126_data = s3[45];
            float v2128_data = ir7[3];
            ir7[3] = (v2128_data + (v2110_data * v2126_data));
            float v2131_data = s3[57];
            float v2133_data = ir7[4];
            ir7[4] = (v2133_data + (v2110_data * v2131_data));
            float v2136_data = s3[69];
            float v2138_data = ir7[5];
            ir7[5] = (v2138_data + (v2110_data * v2136_data));
            float v2141_data = s3[81];
            float v2143_data = ir7[6];
            ir7[6] = (v2143_data + (v2110_data * v2141_data));
            float v2146_data = s3[93];
            float v2148_data = ir7[7];
            ir7[7] = (v2148_data + (v2110_data * v2146_data));
          }
          if (v27_lead < 12) {
            float v2154_data = r6[10];
            float v2155_data = s3[10];
            float v2157_data = ir7[0];
            ir7[0] = (v2157_data + (v2154_data * v2155_data));
            float v2160_data = s3[22];
            float v2162_data = ir7[1];
            ir7[1] = (v2162_data + (v2154_data * v2160_data));
            float v2165_data = s3[34];
            float v2167_data = ir7[2];
            ir7[2] = (v2167_data + (v2154_data * v2165_data));
            float v2170_data = s3[46];
            float v2172_data = ir7[3];
            ir7[3] = (v2172_data + (v2154_data * v2170_data));
            float v2175_data = s3[58];
            float v2177_data = ir7[4];
            ir7[4] = (v2177_data + (v2154_data * v2175_data));
            float v2180_data = s3[70];
            float v2182_data = ir7[5];
            ir7[5] = (v2182_data + (v2154_data * v2180_data));
            float v2185_data = s3[82];
            float v2187_data = ir7[6];
            ir7[6] = (v2187_data + (v2154_data * v2185_data));
            float v2190_data = s3[94];
            float v2192_data = ir7[7];
            ir7[7] = (v2192_data + (v2154_data * v2190_data));
          }
          if (v27_lead < 12) {
            float v2198_data = r6[11];
            float v2199_data = s3[11];
            float v2201_data = ir7[0];
            ir7[0] = (v2201_data + (v2198_data * v2199_data));
            float v2204_data = s3[23];
            float v2206_data = ir7[1];
            ir7[1] = (v2206_data + (v2198_data * v2204_data));
            float v2209_data = s3[35];
            float v2211_data = ir7[2];
            ir7[2] = (v2211_data + (v2198_data * v2209_data));
            float v2214_data = s3[47];
            float v2216_data = ir7[3];
            ir7[3] = (v2216_data + (v2198_data * v2214_data));
            float v2219_data = s3[59];
            float v2221_data = ir7[4];
            ir7[4] = (v2221_data + (v2198_data * v2219_data));
            float v2224_data = s3[71];
            float v2226_data = ir7[5];
            ir7[5] = (v2226_data + (v2198_data * v2224_data));
            float v2229_data = s3[83];
            float v2231_data = ir7[6];
            ir7[6] = (v2231_data + (v2198_data * v2229_data));
            float v2234_data = s3[95];
            float v2236_data = ir7[7];
            ir7[7] = (v2236_data + (v2198_data * v2234_data));
          }
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v2242_n1 = 0; v2242_n1 < 8; ++v2242_n1) {
              float v2244_data = ir7[v2242_n1];
              float v2246_data = r5[v2242_n1];
              r7[v2242_n1] = (v2246_data + v2244_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v27_lead < 12) {
            #pragma unroll
            for (int32_t v2253_i1 = 0; v2253_i1 < 8; ++v2253_i1) {
              float v2255_data = r7[v2253_i1];
              glb_m0[(v27_lead + (v2253_i1 * 12))] = v2255_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

