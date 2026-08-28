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
          alignas(16) float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 16;
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 12; ++v14_i1) {
              int32_t v20_a = v14_i1 * 12;
              int32_t v21_a = v12_lead + v20_a;
              float v29_data = __ldcg(&glb_m1[(v12_lead + v20_a)]);
              int32_t v30_a = 0 + v14_i1;
              r0[v30_a] = v29_data;
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
          alignas(16) float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              int32_t v44_a = v38_i1 * 12;
              int32_t v45_a = v12_lead + v44_a;
              float v53_data = __ldcg(&glb_m3[(v12_lead + v44_a)]);
              int32_t v54_a = 0 + v38_i1;
              r2[v54_a] = v53_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
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
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v589_n1 = 0; v589_n1 < 8; ++v589_n1) {
              int32_t v590_a = 0 + v589_n1;
              float v592_data = ir1[v589_n1];
              r1[v589_n1] = v592_data;
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
          alignas(16) float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v601_i1 = 0; v601_i1 < 12; ++v601_i1) {
              int32_t v607_a = v601_i1 * 12;
              int32_t v608_a = v12_lead + v607_a;
              float v616_data = __ldcg(&glb_m5[(v12_lead + v607_a)]);
              int32_t v617_a = 0 + v601_i1;
              r4[v617_a] = v616_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v12_lead < 12) {
            float v624_data = r2[0];
            float v625_data = s1[0];
            float v627_data = ir3[0];
            ir3[0] = (v627_data + (v624_data * v625_data));
            float v630_data = s1[12];
            float v632_data = ir3[1];
            ir3[1] = (v632_data + (v624_data * v630_data));
            float v635_data = s1[24];
            float v637_data = ir3[2];
            ir3[2] = (v637_data + (v624_data * v635_data));
            float v640_data = s1[36];
            float v642_data = ir3[3];
            ir3[3] = (v642_data + (v624_data * v640_data));
            float v645_data = s1[48];
            float v647_data = ir3[4];
            ir3[4] = (v647_data + (v624_data * v645_data));
            float v650_data = s1[60];
            float v652_data = ir3[5];
            ir3[5] = (v652_data + (v624_data * v650_data));
            float v655_data = s1[72];
            float v657_data = ir3[6];
            ir3[6] = (v657_data + (v624_data * v655_data));
            float v660_data = s1[84];
            float v662_data = ir3[7];
            ir3[7] = (v662_data + (v624_data * v660_data));
          }
          if (v12_lead < 12) {
            float v668_data = r2[1];
            float v669_data = s1[1];
            float v671_data = ir3[0];
            ir3[0] = (v671_data + (v668_data * v669_data));
            float v674_data = s1[13];
            float v676_data = ir3[1];
            ir3[1] = (v676_data + (v668_data * v674_data));
            float v679_data = s1[25];
            float v681_data = ir3[2];
            ir3[2] = (v681_data + (v668_data * v679_data));
            float v684_data = s1[37];
            float v686_data = ir3[3];
            ir3[3] = (v686_data + (v668_data * v684_data));
            float v689_data = s1[49];
            float v691_data = ir3[4];
            ir3[4] = (v691_data + (v668_data * v689_data));
            float v694_data = s1[61];
            float v696_data = ir3[5];
            ir3[5] = (v696_data + (v668_data * v694_data));
            float v699_data = s1[73];
            float v701_data = ir3[6];
            ir3[6] = (v701_data + (v668_data * v699_data));
            float v704_data = s1[85];
            float v706_data = ir3[7];
            ir3[7] = (v706_data + (v668_data * v704_data));
          }
          if (v12_lead < 12) {
            float v712_data = r2[2];
            float v713_data = s1[2];
            float v715_data = ir3[0];
            ir3[0] = (v715_data + (v712_data * v713_data));
            float v718_data = s1[14];
            float v720_data = ir3[1];
            ir3[1] = (v720_data + (v712_data * v718_data));
            float v723_data = s1[26];
            float v725_data = ir3[2];
            ir3[2] = (v725_data + (v712_data * v723_data));
            float v728_data = s1[38];
            float v730_data = ir3[3];
            ir3[3] = (v730_data + (v712_data * v728_data));
            float v733_data = s1[50];
            float v735_data = ir3[4];
            ir3[4] = (v735_data + (v712_data * v733_data));
            float v738_data = s1[62];
            float v740_data = ir3[5];
            ir3[5] = (v740_data + (v712_data * v738_data));
            float v743_data = s1[74];
            float v745_data = ir3[6];
            ir3[6] = (v745_data + (v712_data * v743_data));
            float v748_data = s1[86];
            float v750_data = ir3[7];
            ir3[7] = (v750_data + (v712_data * v748_data));
          }
          if (v12_lead < 12) {
            float v756_data = r2[3];
            float v757_data = s1[3];
            float v759_data = ir3[0];
            ir3[0] = (v759_data + (v756_data * v757_data));
            float v762_data = s1[15];
            float v764_data = ir3[1];
            ir3[1] = (v764_data + (v756_data * v762_data));
            float v767_data = s1[27];
            float v769_data = ir3[2];
            ir3[2] = (v769_data + (v756_data * v767_data));
            float v772_data = s1[39];
            float v774_data = ir3[3];
            ir3[3] = (v774_data + (v756_data * v772_data));
            float v777_data = s1[51];
            float v779_data = ir3[4];
            ir3[4] = (v779_data + (v756_data * v777_data));
            float v782_data = s1[63];
            float v784_data = ir3[5];
            ir3[5] = (v784_data + (v756_data * v782_data));
            float v787_data = s1[75];
            float v789_data = ir3[6];
            ir3[6] = (v789_data + (v756_data * v787_data));
            float v792_data = s1[87];
            float v794_data = ir3[7];
            ir3[7] = (v794_data + (v756_data * v792_data));
          }
          if (v12_lead < 12) {
            float v800_data = r2[4];
            float v801_data = s1[4];
            float v803_data = ir3[0];
            ir3[0] = (v803_data + (v800_data * v801_data));
            float v806_data = s1[16];
            float v808_data = ir3[1];
            ir3[1] = (v808_data + (v800_data * v806_data));
            float v811_data = s1[28];
            float v813_data = ir3[2];
            ir3[2] = (v813_data + (v800_data * v811_data));
            float v816_data = s1[40];
            float v818_data = ir3[3];
            ir3[3] = (v818_data + (v800_data * v816_data));
            float v821_data = s1[52];
            float v823_data = ir3[4];
            ir3[4] = (v823_data + (v800_data * v821_data));
            float v826_data = s1[64];
            float v828_data = ir3[5];
            ir3[5] = (v828_data + (v800_data * v826_data));
            float v831_data = s1[76];
            float v833_data = ir3[6];
            ir3[6] = (v833_data + (v800_data * v831_data));
            float v836_data = s1[88];
            float v838_data = ir3[7];
            ir3[7] = (v838_data + (v800_data * v836_data));
          }
          if (v12_lead < 12) {
            float v844_data = r2[5];
            float v845_data = s1[5];
            float v847_data = ir3[0];
            ir3[0] = (v847_data + (v844_data * v845_data));
            float v850_data = s1[17];
            float v852_data = ir3[1];
            ir3[1] = (v852_data + (v844_data * v850_data));
            float v855_data = s1[29];
            float v857_data = ir3[2];
            ir3[2] = (v857_data + (v844_data * v855_data));
            float v860_data = s1[41];
            float v862_data = ir3[3];
            ir3[3] = (v862_data + (v844_data * v860_data));
            float v865_data = s1[53];
            float v867_data = ir3[4];
            ir3[4] = (v867_data + (v844_data * v865_data));
            float v870_data = s1[65];
            float v872_data = ir3[5];
            ir3[5] = (v872_data + (v844_data * v870_data));
            float v875_data = s1[77];
            float v877_data = ir3[6];
            ir3[6] = (v877_data + (v844_data * v875_data));
            float v880_data = s1[89];
            float v882_data = ir3[7];
            ir3[7] = (v882_data + (v844_data * v880_data));
          }
          if (v12_lead < 12) {
            float v888_data = r2[6];
            float v889_data = s1[6];
            float v891_data = ir3[0];
            ir3[0] = (v891_data + (v888_data * v889_data));
            float v894_data = s1[18];
            float v896_data = ir3[1];
            ir3[1] = (v896_data + (v888_data * v894_data));
            float v899_data = s1[30];
            float v901_data = ir3[2];
            ir3[2] = (v901_data + (v888_data * v899_data));
            float v904_data = s1[42];
            float v906_data = ir3[3];
            ir3[3] = (v906_data + (v888_data * v904_data));
            float v909_data = s1[54];
            float v911_data = ir3[4];
            ir3[4] = (v911_data + (v888_data * v909_data));
            float v914_data = s1[66];
            float v916_data = ir3[5];
            ir3[5] = (v916_data + (v888_data * v914_data));
            float v919_data = s1[78];
            float v921_data = ir3[6];
            ir3[6] = (v921_data + (v888_data * v919_data));
            float v924_data = s1[90];
            float v926_data = ir3[7];
            ir3[7] = (v926_data + (v888_data * v924_data));
          }
          if (v12_lead < 12) {
            float v932_data = r2[7];
            float v933_data = s1[7];
            float v935_data = ir3[0];
            ir3[0] = (v935_data + (v932_data * v933_data));
            float v938_data = s1[19];
            float v940_data = ir3[1];
            ir3[1] = (v940_data + (v932_data * v938_data));
            float v943_data = s1[31];
            float v945_data = ir3[2];
            ir3[2] = (v945_data + (v932_data * v943_data));
            float v948_data = s1[43];
            float v950_data = ir3[3];
            ir3[3] = (v950_data + (v932_data * v948_data));
            float v953_data = s1[55];
            float v955_data = ir3[4];
            ir3[4] = (v955_data + (v932_data * v953_data));
            float v958_data = s1[67];
            float v960_data = ir3[5];
            ir3[5] = (v960_data + (v932_data * v958_data));
            float v963_data = s1[79];
            float v965_data = ir3[6];
            ir3[6] = (v965_data + (v932_data * v963_data));
            float v968_data = s1[91];
            float v970_data = ir3[7];
            ir3[7] = (v970_data + (v932_data * v968_data));
          }
          if (v12_lead < 12) {
            float v976_data = r2[8];
            float v977_data = s1[8];
            float v979_data = ir3[0];
            ir3[0] = (v979_data + (v976_data * v977_data));
            float v982_data = s1[20];
            float v984_data = ir3[1];
            ir3[1] = (v984_data + (v976_data * v982_data));
            float v987_data = s1[32];
            float v989_data = ir3[2];
            ir3[2] = (v989_data + (v976_data * v987_data));
            float v992_data = s1[44];
            float v994_data = ir3[3];
            ir3[3] = (v994_data + (v976_data * v992_data));
            float v997_data = s1[56];
            float v999_data = ir3[4];
            ir3[4] = (v999_data + (v976_data * v997_data));
            float v1002_data = s1[68];
            float v1004_data = ir3[5];
            ir3[5] = (v1004_data + (v976_data * v1002_data));
            float v1007_data = s1[80];
            float v1009_data = ir3[6];
            ir3[6] = (v1009_data + (v976_data * v1007_data));
            float v1012_data = s1[92];
            float v1014_data = ir3[7];
            ir3[7] = (v1014_data + (v976_data * v1012_data));
          }
          if (v12_lead < 12) {
            float v1020_data = r2[9];
            float v1021_data = s1[9];
            float v1023_data = ir3[0];
            ir3[0] = (v1023_data + (v1020_data * v1021_data));
            float v1026_data = s1[21];
            float v1028_data = ir3[1];
            ir3[1] = (v1028_data + (v1020_data * v1026_data));
            float v1031_data = s1[33];
            float v1033_data = ir3[2];
            ir3[2] = (v1033_data + (v1020_data * v1031_data));
            float v1036_data = s1[45];
            float v1038_data = ir3[3];
            ir3[3] = (v1038_data + (v1020_data * v1036_data));
            float v1041_data = s1[57];
            float v1043_data = ir3[4];
            ir3[4] = (v1043_data + (v1020_data * v1041_data));
            float v1046_data = s1[69];
            float v1048_data = ir3[5];
            ir3[5] = (v1048_data + (v1020_data * v1046_data));
            float v1051_data = s1[81];
            float v1053_data = ir3[6];
            ir3[6] = (v1053_data + (v1020_data * v1051_data));
            float v1056_data = s1[93];
            float v1058_data = ir3[7];
            ir3[7] = (v1058_data + (v1020_data * v1056_data));
          }
          if (v12_lead < 12) {
            float v1064_data = r2[10];
            float v1065_data = s1[10];
            float v1067_data = ir3[0];
            ir3[0] = (v1067_data + (v1064_data * v1065_data));
            float v1070_data = s1[22];
            float v1072_data = ir3[1];
            ir3[1] = (v1072_data + (v1064_data * v1070_data));
            float v1075_data = s1[34];
            float v1077_data = ir3[2];
            ir3[2] = (v1077_data + (v1064_data * v1075_data));
            float v1080_data = s1[46];
            float v1082_data = ir3[3];
            ir3[3] = (v1082_data + (v1064_data * v1080_data));
            float v1085_data = s1[58];
            float v1087_data = ir3[4];
            ir3[4] = (v1087_data + (v1064_data * v1085_data));
            float v1090_data = s1[70];
            float v1092_data = ir3[5];
            ir3[5] = (v1092_data + (v1064_data * v1090_data));
            float v1095_data = s1[82];
            float v1097_data = ir3[6];
            ir3[6] = (v1097_data + (v1064_data * v1095_data));
            float v1100_data = s1[94];
            float v1102_data = ir3[7];
            ir3[7] = (v1102_data + (v1064_data * v1100_data));
          }
          if (v12_lead < 12) {
            float v1108_data = r2[11];
            float v1109_data = s1[11];
            float v1111_data = ir3[0];
            ir3[0] = (v1111_data + (v1108_data * v1109_data));
            float v1114_data = s1[23];
            float v1116_data = ir3[1];
            ir3[1] = (v1116_data + (v1108_data * v1114_data));
            float v1119_data = s1[35];
            float v1121_data = ir3[2];
            ir3[2] = (v1121_data + (v1108_data * v1119_data));
            float v1124_data = s1[47];
            float v1126_data = ir3[3];
            ir3[3] = (v1126_data + (v1108_data * v1124_data));
            float v1129_data = s1[59];
            float v1131_data = ir3[4];
            ir3[4] = (v1131_data + (v1108_data * v1129_data));
            float v1134_data = s1[71];
            float v1136_data = ir3[5];
            ir3[5] = (v1136_data + (v1108_data * v1134_data));
            float v1139_data = s1[83];
            float v1141_data = ir3[6];
            ir3[6] = (v1141_data + (v1108_data * v1139_data));
            float v1144_data = s1[95];
            float v1146_data = ir3[7];
            ir3[7] = (v1146_data + (v1108_data * v1144_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1152_n1 = 0; v1152_n1 < 8; ++v1152_n1) {
              int32_t v1153_a = 0 + v1152_n1;
              float v1155_data = ir3[v1152_n1];
              int32_t v1156_a = 0 + v1152_n1;
              float v1158_data = r1[v1152_n1];
              r3[v1152_n1] = (v1158_data + v1155_data);
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
          alignas(16) float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1168_i1 = 0; v1168_i1 < 12; ++v1168_i1) {
              int32_t v1174_a = v1168_i1 * 12;
              int32_t v1175_a = v12_lead + v1174_a;
              float v1183_data = __ldcg(&glb_m7[(v12_lead + v1174_a)]);
              int32_t v1184_a = 0 + v1168_i1;
              r6[v1184_a] = v1183_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v12_lead < 12) {
            float v1191_data = r4[0];
            float v1192_data = s2[0];
            float v1194_data = ir5[0];
            ir5[0] = (v1194_data + (v1191_data * v1192_data));
            float v1197_data = s2[12];
            float v1199_data = ir5[1];
            ir5[1] = (v1199_data + (v1191_data * v1197_data));
            float v1202_data = s2[24];
            float v1204_data = ir5[2];
            ir5[2] = (v1204_data + (v1191_data * v1202_data));
            float v1207_data = s2[36];
            float v1209_data = ir5[3];
            ir5[3] = (v1209_data + (v1191_data * v1207_data));
            float v1212_data = s2[48];
            float v1214_data = ir5[4];
            ir5[4] = (v1214_data + (v1191_data * v1212_data));
            float v1217_data = s2[60];
            float v1219_data = ir5[5];
            ir5[5] = (v1219_data + (v1191_data * v1217_data));
            float v1222_data = s2[72];
            float v1224_data = ir5[6];
            ir5[6] = (v1224_data + (v1191_data * v1222_data));
            float v1227_data = s2[84];
            float v1229_data = ir5[7];
            ir5[7] = (v1229_data + (v1191_data * v1227_data));
          }
          if (v12_lead < 12) {
            float v1235_data = r4[1];
            float v1236_data = s2[1];
            float v1238_data = ir5[0];
            ir5[0] = (v1238_data + (v1235_data * v1236_data));
            float v1241_data = s2[13];
            float v1243_data = ir5[1];
            ir5[1] = (v1243_data + (v1235_data * v1241_data));
            float v1246_data = s2[25];
            float v1248_data = ir5[2];
            ir5[2] = (v1248_data + (v1235_data * v1246_data));
            float v1251_data = s2[37];
            float v1253_data = ir5[3];
            ir5[3] = (v1253_data + (v1235_data * v1251_data));
            float v1256_data = s2[49];
            float v1258_data = ir5[4];
            ir5[4] = (v1258_data + (v1235_data * v1256_data));
            float v1261_data = s2[61];
            float v1263_data = ir5[5];
            ir5[5] = (v1263_data + (v1235_data * v1261_data));
            float v1266_data = s2[73];
            float v1268_data = ir5[6];
            ir5[6] = (v1268_data + (v1235_data * v1266_data));
            float v1271_data = s2[85];
            float v1273_data = ir5[7];
            ir5[7] = (v1273_data + (v1235_data * v1271_data));
          }
          if (v12_lead < 12) {
            float v1279_data = r4[2];
            float v1280_data = s2[2];
            float v1282_data = ir5[0];
            ir5[0] = (v1282_data + (v1279_data * v1280_data));
            float v1285_data = s2[14];
            float v1287_data = ir5[1];
            ir5[1] = (v1287_data + (v1279_data * v1285_data));
            float v1290_data = s2[26];
            float v1292_data = ir5[2];
            ir5[2] = (v1292_data + (v1279_data * v1290_data));
            float v1295_data = s2[38];
            float v1297_data = ir5[3];
            ir5[3] = (v1297_data + (v1279_data * v1295_data));
            float v1300_data = s2[50];
            float v1302_data = ir5[4];
            ir5[4] = (v1302_data + (v1279_data * v1300_data));
            float v1305_data = s2[62];
            float v1307_data = ir5[5];
            ir5[5] = (v1307_data + (v1279_data * v1305_data));
            float v1310_data = s2[74];
            float v1312_data = ir5[6];
            ir5[6] = (v1312_data + (v1279_data * v1310_data));
            float v1315_data = s2[86];
            float v1317_data = ir5[7];
            ir5[7] = (v1317_data + (v1279_data * v1315_data));
          }
          if (v12_lead < 12) {
            float v1323_data = r4[3];
            float v1324_data = s2[3];
            float v1326_data = ir5[0];
            ir5[0] = (v1326_data + (v1323_data * v1324_data));
            float v1329_data = s2[15];
            float v1331_data = ir5[1];
            ir5[1] = (v1331_data + (v1323_data * v1329_data));
            float v1334_data = s2[27];
            float v1336_data = ir5[2];
            ir5[2] = (v1336_data + (v1323_data * v1334_data));
            float v1339_data = s2[39];
            float v1341_data = ir5[3];
            ir5[3] = (v1341_data + (v1323_data * v1339_data));
            float v1344_data = s2[51];
            float v1346_data = ir5[4];
            ir5[4] = (v1346_data + (v1323_data * v1344_data));
            float v1349_data = s2[63];
            float v1351_data = ir5[5];
            ir5[5] = (v1351_data + (v1323_data * v1349_data));
            float v1354_data = s2[75];
            float v1356_data = ir5[6];
            ir5[6] = (v1356_data + (v1323_data * v1354_data));
            float v1359_data = s2[87];
            float v1361_data = ir5[7];
            ir5[7] = (v1361_data + (v1323_data * v1359_data));
          }
          if (v12_lead < 12) {
            float v1367_data = r4[4];
            float v1368_data = s2[4];
            float v1370_data = ir5[0];
            ir5[0] = (v1370_data + (v1367_data * v1368_data));
            float v1373_data = s2[16];
            float v1375_data = ir5[1];
            ir5[1] = (v1375_data + (v1367_data * v1373_data));
            float v1378_data = s2[28];
            float v1380_data = ir5[2];
            ir5[2] = (v1380_data + (v1367_data * v1378_data));
            float v1383_data = s2[40];
            float v1385_data = ir5[3];
            ir5[3] = (v1385_data + (v1367_data * v1383_data));
            float v1388_data = s2[52];
            float v1390_data = ir5[4];
            ir5[4] = (v1390_data + (v1367_data * v1388_data));
            float v1393_data = s2[64];
            float v1395_data = ir5[5];
            ir5[5] = (v1395_data + (v1367_data * v1393_data));
            float v1398_data = s2[76];
            float v1400_data = ir5[6];
            ir5[6] = (v1400_data + (v1367_data * v1398_data));
            float v1403_data = s2[88];
            float v1405_data = ir5[7];
            ir5[7] = (v1405_data + (v1367_data * v1403_data));
          }
          if (v12_lead < 12) {
            float v1411_data = r4[5];
            float v1412_data = s2[5];
            float v1414_data = ir5[0];
            ir5[0] = (v1414_data + (v1411_data * v1412_data));
            float v1417_data = s2[17];
            float v1419_data = ir5[1];
            ir5[1] = (v1419_data + (v1411_data * v1417_data));
            float v1422_data = s2[29];
            float v1424_data = ir5[2];
            ir5[2] = (v1424_data + (v1411_data * v1422_data));
            float v1427_data = s2[41];
            float v1429_data = ir5[3];
            ir5[3] = (v1429_data + (v1411_data * v1427_data));
            float v1432_data = s2[53];
            float v1434_data = ir5[4];
            ir5[4] = (v1434_data + (v1411_data * v1432_data));
            float v1437_data = s2[65];
            float v1439_data = ir5[5];
            ir5[5] = (v1439_data + (v1411_data * v1437_data));
            float v1442_data = s2[77];
            float v1444_data = ir5[6];
            ir5[6] = (v1444_data + (v1411_data * v1442_data));
            float v1447_data = s2[89];
            float v1449_data = ir5[7];
            ir5[7] = (v1449_data + (v1411_data * v1447_data));
          }
          if (v12_lead < 12) {
            float v1455_data = r4[6];
            float v1456_data = s2[6];
            float v1458_data = ir5[0];
            ir5[0] = (v1458_data + (v1455_data * v1456_data));
            float v1461_data = s2[18];
            float v1463_data = ir5[1];
            ir5[1] = (v1463_data + (v1455_data * v1461_data));
            float v1466_data = s2[30];
            float v1468_data = ir5[2];
            ir5[2] = (v1468_data + (v1455_data * v1466_data));
            float v1471_data = s2[42];
            float v1473_data = ir5[3];
            ir5[3] = (v1473_data + (v1455_data * v1471_data));
            float v1476_data = s2[54];
            float v1478_data = ir5[4];
            ir5[4] = (v1478_data + (v1455_data * v1476_data));
            float v1481_data = s2[66];
            float v1483_data = ir5[5];
            ir5[5] = (v1483_data + (v1455_data * v1481_data));
            float v1486_data = s2[78];
            float v1488_data = ir5[6];
            ir5[6] = (v1488_data + (v1455_data * v1486_data));
            float v1491_data = s2[90];
            float v1493_data = ir5[7];
            ir5[7] = (v1493_data + (v1455_data * v1491_data));
          }
          if (v12_lead < 12) {
            float v1499_data = r4[7];
            float v1500_data = s2[7];
            float v1502_data = ir5[0];
            ir5[0] = (v1502_data + (v1499_data * v1500_data));
            float v1505_data = s2[19];
            float v1507_data = ir5[1];
            ir5[1] = (v1507_data + (v1499_data * v1505_data));
            float v1510_data = s2[31];
            float v1512_data = ir5[2];
            ir5[2] = (v1512_data + (v1499_data * v1510_data));
            float v1515_data = s2[43];
            float v1517_data = ir5[3];
            ir5[3] = (v1517_data + (v1499_data * v1515_data));
            float v1520_data = s2[55];
            float v1522_data = ir5[4];
            ir5[4] = (v1522_data + (v1499_data * v1520_data));
            float v1525_data = s2[67];
            float v1527_data = ir5[5];
            ir5[5] = (v1527_data + (v1499_data * v1525_data));
            float v1530_data = s2[79];
            float v1532_data = ir5[6];
            ir5[6] = (v1532_data + (v1499_data * v1530_data));
            float v1535_data = s2[91];
            float v1537_data = ir5[7];
            ir5[7] = (v1537_data + (v1499_data * v1535_data));
          }
          if (v12_lead < 12) {
            float v1543_data = r4[8];
            float v1544_data = s2[8];
            float v1546_data = ir5[0];
            ir5[0] = (v1546_data + (v1543_data * v1544_data));
            float v1549_data = s2[20];
            float v1551_data = ir5[1];
            ir5[1] = (v1551_data + (v1543_data * v1549_data));
            float v1554_data = s2[32];
            float v1556_data = ir5[2];
            ir5[2] = (v1556_data + (v1543_data * v1554_data));
            float v1559_data = s2[44];
            float v1561_data = ir5[3];
            ir5[3] = (v1561_data + (v1543_data * v1559_data));
            float v1564_data = s2[56];
            float v1566_data = ir5[4];
            ir5[4] = (v1566_data + (v1543_data * v1564_data));
            float v1569_data = s2[68];
            float v1571_data = ir5[5];
            ir5[5] = (v1571_data + (v1543_data * v1569_data));
            float v1574_data = s2[80];
            float v1576_data = ir5[6];
            ir5[6] = (v1576_data + (v1543_data * v1574_data));
            float v1579_data = s2[92];
            float v1581_data = ir5[7];
            ir5[7] = (v1581_data + (v1543_data * v1579_data));
          }
          if (v12_lead < 12) {
            float v1587_data = r4[9];
            float v1588_data = s2[9];
            float v1590_data = ir5[0];
            ir5[0] = (v1590_data + (v1587_data * v1588_data));
            float v1593_data = s2[21];
            float v1595_data = ir5[1];
            ir5[1] = (v1595_data + (v1587_data * v1593_data));
            float v1598_data = s2[33];
            float v1600_data = ir5[2];
            ir5[2] = (v1600_data + (v1587_data * v1598_data));
            float v1603_data = s2[45];
            float v1605_data = ir5[3];
            ir5[3] = (v1605_data + (v1587_data * v1603_data));
            float v1608_data = s2[57];
            float v1610_data = ir5[4];
            ir5[4] = (v1610_data + (v1587_data * v1608_data));
            float v1613_data = s2[69];
            float v1615_data = ir5[5];
            ir5[5] = (v1615_data + (v1587_data * v1613_data));
            float v1618_data = s2[81];
            float v1620_data = ir5[6];
            ir5[6] = (v1620_data + (v1587_data * v1618_data));
            float v1623_data = s2[93];
            float v1625_data = ir5[7];
            ir5[7] = (v1625_data + (v1587_data * v1623_data));
          }
          if (v12_lead < 12) {
            float v1631_data = r4[10];
            float v1632_data = s2[10];
            float v1634_data = ir5[0];
            ir5[0] = (v1634_data + (v1631_data * v1632_data));
            float v1637_data = s2[22];
            float v1639_data = ir5[1];
            ir5[1] = (v1639_data + (v1631_data * v1637_data));
            float v1642_data = s2[34];
            float v1644_data = ir5[2];
            ir5[2] = (v1644_data + (v1631_data * v1642_data));
            float v1647_data = s2[46];
            float v1649_data = ir5[3];
            ir5[3] = (v1649_data + (v1631_data * v1647_data));
            float v1652_data = s2[58];
            float v1654_data = ir5[4];
            ir5[4] = (v1654_data + (v1631_data * v1652_data));
            float v1657_data = s2[70];
            float v1659_data = ir5[5];
            ir5[5] = (v1659_data + (v1631_data * v1657_data));
            float v1662_data = s2[82];
            float v1664_data = ir5[6];
            ir5[6] = (v1664_data + (v1631_data * v1662_data));
            float v1667_data = s2[94];
            float v1669_data = ir5[7];
            ir5[7] = (v1669_data + (v1631_data * v1667_data));
          }
          if (v12_lead < 12) {
            float v1675_data = r4[11];
            float v1676_data = s2[11];
            float v1678_data = ir5[0];
            ir5[0] = (v1678_data + (v1675_data * v1676_data));
            float v1681_data = s2[23];
            float v1683_data = ir5[1];
            ir5[1] = (v1683_data + (v1675_data * v1681_data));
            float v1686_data = s2[35];
            float v1688_data = ir5[2];
            ir5[2] = (v1688_data + (v1675_data * v1686_data));
            float v1691_data = s2[47];
            float v1693_data = ir5[3];
            ir5[3] = (v1693_data + (v1675_data * v1691_data));
            float v1696_data = s2[59];
            float v1698_data = ir5[4];
            ir5[4] = (v1698_data + (v1675_data * v1696_data));
            float v1701_data = s2[71];
            float v1703_data = ir5[5];
            ir5[5] = (v1703_data + (v1675_data * v1701_data));
            float v1706_data = s2[83];
            float v1708_data = ir5[6];
            ir5[6] = (v1708_data + (v1675_data * v1706_data));
            float v1711_data = s2[95];
            float v1713_data = ir5[7];
            ir5[7] = (v1713_data + (v1675_data * v1711_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1719_n1 = 0; v1719_n1 < 8; ++v1719_n1) {
              int32_t v1720_a = 0 + v1719_n1;
              float v1722_data = ir5[v1719_n1];
              int32_t v1723_a = 0 + v1719_n1;
              float v1725_data = r3[v1719_n1];
              r5[v1719_n1] = (v1725_data + v1722_data);
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
          alignas(16) float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v12_lead < 12) {
            float v1736_data = r6[0];
            float v1737_data = s3[0];
            float v1739_data = ir7[0];
            ir7[0] = (v1739_data + (v1736_data * v1737_data));
            float v1742_data = s3[12];
            float v1744_data = ir7[1];
            ir7[1] = (v1744_data + (v1736_data * v1742_data));
            float v1747_data = s3[24];
            float v1749_data = ir7[2];
            ir7[2] = (v1749_data + (v1736_data * v1747_data));
            float v1752_data = s3[36];
            float v1754_data = ir7[3];
            ir7[3] = (v1754_data + (v1736_data * v1752_data));
            float v1757_data = s3[48];
            float v1759_data = ir7[4];
            ir7[4] = (v1759_data + (v1736_data * v1757_data));
            float v1762_data = s3[60];
            float v1764_data = ir7[5];
            ir7[5] = (v1764_data + (v1736_data * v1762_data));
            float v1767_data = s3[72];
            float v1769_data = ir7[6];
            ir7[6] = (v1769_data + (v1736_data * v1767_data));
            float v1772_data = s3[84];
            float v1774_data = ir7[7];
            ir7[7] = (v1774_data + (v1736_data * v1772_data));
          }
          if (v12_lead < 12) {
            float v1780_data = r6[1];
            float v1781_data = s3[1];
            float v1783_data = ir7[0];
            ir7[0] = (v1783_data + (v1780_data * v1781_data));
            float v1786_data = s3[13];
            float v1788_data = ir7[1];
            ir7[1] = (v1788_data + (v1780_data * v1786_data));
            float v1791_data = s3[25];
            float v1793_data = ir7[2];
            ir7[2] = (v1793_data + (v1780_data * v1791_data));
            float v1796_data = s3[37];
            float v1798_data = ir7[3];
            ir7[3] = (v1798_data + (v1780_data * v1796_data));
            float v1801_data = s3[49];
            float v1803_data = ir7[4];
            ir7[4] = (v1803_data + (v1780_data * v1801_data));
            float v1806_data = s3[61];
            float v1808_data = ir7[5];
            ir7[5] = (v1808_data + (v1780_data * v1806_data));
            float v1811_data = s3[73];
            float v1813_data = ir7[6];
            ir7[6] = (v1813_data + (v1780_data * v1811_data));
            float v1816_data = s3[85];
            float v1818_data = ir7[7];
            ir7[7] = (v1818_data + (v1780_data * v1816_data));
          }
          if (v12_lead < 12) {
            float v1824_data = r6[2];
            float v1825_data = s3[2];
            float v1827_data = ir7[0];
            ir7[0] = (v1827_data + (v1824_data * v1825_data));
            float v1830_data = s3[14];
            float v1832_data = ir7[1];
            ir7[1] = (v1832_data + (v1824_data * v1830_data));
            float v1835_data = s3[26];
            float v1837_data = ir7[2];
            ir7[2] = (v1837_data + (v1824_data * v1835_data));
            float v1840_data = s3[38];
            float v1842_data = ir7[3];
            ir7[3] = (v1842_data + (v1824_data * v1840_data));
            float v1845_data = s3[50];
            float v1847_data = ir7[4];
            ir7[4] = (v1847_data + (v1824_data * v1845_data));
            float v1850_data = s3[62];
            float v1852_data = ir7[5];
            ir7[5] = (v1852_data + (v1824_data * v1850_data));
            float v1855_data = s3[74];
            float v1857_data = ir7[6];
            ir7[6] = (v1857_data + (v1824_data * v1855_data));
            float v1860_data = s3[86];
            float v1862_data = ir7[7];
            ir7[7] = (v1862_data + (v1824_data * v1860_data));
          }
          if (v12_lead < 12) {
            float v1868_data = r6[3];
            float v1869_data = s3[3];
            float v1871_data = ir7[0];
            ir7[0] = (v1871_data + (v1868_data * v1869_data));
            float v1874_data = s3[15];
            float v1876_data = ir7[1];
            ir7[1] = (v1876_data + (v1868_data * v1874_data));
            float v1879_data = s3[27];
            float v1881_data = ir7[2];
            ir7[2] = (v1881_data + (v1868_data * v1879_data));
            float v1884_data = s3[39];
            float v1886_data = ir7[3];
            ir7[3] = (v1886_data + (v1868_data * v1884_data));
            float v1889_data = s3[51];
            float v1891_data = ir7[4];
            ir7[4] = (v1891_data + (v1868_data * v1889_data));
            float v1894_data = s3[63];
            float v1896_data = ir7[5];
            ir7[5] = (v1896_data + (v1868_data * v1894_data));
            float v1899_data = s3[75];
            float v1901_data = ir7[6];
            ir7[6] = (v1901_data + (v1868_data * v1899_data));
            float v1904_data = s3[87];
            float v1906_data = ir7[7];
            ir7[7] = (v1906_data + (v1868_data * v1904_data));
          }
          if (v12_lead < 12) {
            float v1912_data = r6[4];
            float v1913_data = s3[4];
            float v1915_data = ir7[0];
            ir7[0] = (v1915_data + (v1912_data * v1913_data));
            float v1918_data = s3[16];
            float v1920_data = ir7[1];
            ir7[1] = (v1920_data + (v1912_data * v1918_data));
            float v1923_data = s3[28];
            float v1925_data = ir7[2];
            ir7[2] = (v1925_data + (v1912_data * v1923_data));
            float v1928_data = s3[40];
            float v1930_data = ir7[3];
            ir7[3] = (v1930_data + (v1912_data * v1928_data));
            float v1933_data = s3[52];
            float v1935_data = ir7[4];
            ir7[4] = (v1935_data + (v1912_data * v1933_data));
            float v1938_data = s3[64];
            float v1940_data = ir7[5];
            ir7[5] = (v1940_data + (v1912_data * v1938_data));
            float v1943_data = s3[76];
            float v1945_data = ir7[6];
            ir7[6] = (v1945_data + (v1912_data * v1943_data));
            float v1948_data = s3[88];
            float v1950_data = ir7[7];
            ir7[7] = (v1950_data + (v1912_data * v1948_data));
          }
          if (v12_lead < 12) {
            float v1956_data = r6[5];
            float v1957_data = s3[5];
            float v1959_data = ir7[0];
            ir7[0] = (v1959_data + (v1956_data * v1957_data));
            float v1962_data = s3[17];
            float v1964_data = ir7[1];
            ir7[1] = (v1964_data + (v1956_data * v1962_data));
            float v1967_data = s3[29];
            float v1969_data = ir7[2];
            ir7[2] = (v1969_data + (v1956_data * v1967_data));
            float v1972_data = s3[41];
            float v1974_data = ir7[3];
            ir7[3] = (v1974_data + (v1956_data * v1972_data));
            float v1977_data = s3[53];
            float v1979_data = ir7[4];
            ir7[4] = (v1979_data + (v1956_data * v1977_data));
            float v1982_data = s3[65];
            float v1984_data = ir7[5];
            ir7[5] = (v1984_data + (v1956_data * v1982_data));
            float v1987_data = s3[77];
            float v1989_data = ir7[6];
            ir7[6] = (v1989_data + (v1956_data * v1987_data));
            float v1992_data = s3[89];
            float v1994_data = ir7[7];
            ir7[7] = (v1994_data + (v1956_data * v1992_data));
          }
          if (v12_lead < 12) {
            float v2000_data = r6[6];
            float v2001_data = s3[6];
            float v2003_data = ir7[0];
            ir7[0] = (v2003_data + (v2000_data * v2001_data));
            float v2006_data = s3[18];
            float v2008_data = ir7[1];
            ir7[1] = (v2008_data + (v2000_data * v2006_data));
            float v2011_data = s3[30];
            float v2013_data = ir7[2];
            ir7[2] = (v2013_data + (v2000_data * v2011_data));
            float v2016_data = s3[42];
            float v2018_data = ir7[3];
            ir7[3] = (v2018_data + (v2000_data * v2016_data));
            float v2021_data = s3[54];
            float v2023_data = ir7[4];
            ir7[4] = (v2023_data + (v2000_data * v2021_data));
            float v2026_data = s3[66];
            float v2028_data = ir7[5];
            ir7[5] = (v2028_data + (v2000_data * v2026_data));
            float v2031_data = s3[78];
            float v2033_data = ir7[6];
            ir7[6] = (v2033_data + (v2000_data * v2031_data));
            float v2036_data = s3[90];
            float v2038_data = ir7[7];
            ir7[7] = (v2038_data + (v2000_data * v2036_data));
          }
          if (v12_lead < 12) {
            float v2044_data = r6[7];
            float v2045_data = s3[7];
            float v2047_data = ir7[0];
            ir7[0] = (v2047_data + (v2044_data * v2045_data));
            float v2050_data = s3[19];
            float v2052_data = ir7[1];
            ir7[1] = (v2052_data + (v2044_data * v2050_data));
            float v2055_data = s3[31];
            float v2057_data = ir7[2];
            ir7[2] = (v2057_data + (v2044_data * v2055_data));
            float v2060_data = s3[43];
            float v2062_data = ir7[3];
            ir7[3] = (v2062_data + (v2044_data * v2060_data));
            float v2065_data = s3[55];
            float v2067_data = ir7[4];
            ir7[4] = (v2067_data + (v2044_data * v2065_data));
            float v2070_data = s3[67];
            float v2072_data = ir7[5];
            ir7[5] = (v2072_data + (v2044_data * v2070_data));
            float v2075_data = s3[79];
            float v2077_data = ir7[6];
            ir7[6] = (v2077_data + (v2044_data * v2075_data));
            float v2080_data = s3[91];
            float v2082_data = ir7[7];
            ir7[7] = (v2082_data + (v2044_data * v2080_data));
          }
          if (v12_lead < 12) {
            float v2088_data = r6[8];
            float v2089_data = s3[8];
            float v2091_data = ir7[0];
            ir7[0] = (v2091_data + (v2088_data * v2089_data));
            float v2094_data = s3[20];
            float v2096_data = ir7[1];
            ir7[1] = (v2096_data + (v2088_data * v2094_data));
            float v2099_data = s3[32];
            float v2101_data = ir7[2];
            ir7[2] = (v2101_data + (v2088_data * v2099_data));
            float v2104_data = s3[44];
            float v2106_data = ir7[3];
            ir7[3] = (v2106_data + (v2088_data * v2104_data));
            float v2109_data = s3[56];
            float v2111_data = ir7[4];
            ir7[4] = (v2111_data + (v2088_data * v2109_data));
            float v2114_data = s3[68];
            float v2116_data = ir7[5];
            ir7[5] = (v2116_data + (v2088_data * v2114_data));
            float v2119_data = s3[80];
            float v2121_data = ir7[6];
            ir7[6] = (v2121_data + (v2088_data * v2119_data));
            float v2124_data = s3[92];
            float v2126_data = ir7[7];
            ir7[7] = (v2126_data + (v2088_data * v2124_data));
          }
          if (v12_lead < 12) {
            float v2132_data = r6[9];
            float v2133_data = s3[9];
            float v2135_data = ir7[0];
            ir7[0] = (v2135_data + (v2132_data * v2133_data));
            float v2138_data = s3[21];
            float v2140_data = ir7[1];
            ir7[1] = (v2140_data + (v2132_data * v2138_data));
            float v2143_data = s3[33];
            float v2145_data = ir7[2];
            ir7[2] = (v2145_data + (v2132_data * v2143_data));
            float v2148_data = s3[45];
            float v2150_data = ir7[3];
            ir7[3] = (v2150_data + (v2132_data * v2148_data));
            float v2153_data = s3[57];
            float v2155_data = ir7[4];
            ir7[4] = (v2155_data + (v2132_data * v2153_data));
            float v2158_data = s3[69];
            float v2160_data = ir7[5];
            ir7[5] = (v2160_data + (v2132_data * v2158_data));
            float v2163_data = s3[81];
            float v2165_data = ir7[6];
            ir7[6] = (v2165_data + (v2132_data * v2163_data));
            float v2168_data = s3[93];
            float v2170_data = ir7[7];
            ir7[7] = (v2170_data + (v2132_data * v2168_data));
          }
          if (v12_lead < 12) {
            float v2176_data = r6[10];
            float v2177_data = s3[10];
            float v2179_data = ir7[0];
            ir7[0] = (v2179_data + (v2176_data * v2177_data));
            float v2182_data = s3[22];
            float v2184_data = ir7[1];
            ir7[1] = (v2184_data + (v2176_data * v2182_data));
            float v2187_data = s3[34];
            float v2189_data = ir7[2];
            ir7[2] = (v2189_data + (v2176_data * v2187_data));
            float v2192_data = s3[46];
            float v2194_data = ir7[3];
            ir7[3] = (v2194_data + (v2176_data * v2192_data));
            float v2197_data = s3[58];
            float v2199_data = ir7[4];
            ir7[4] = (v2199_data + (v2176_data * v2197_data));
            float v2202_data = s3[70];
            float v2204_data = ir7[5];
            ir7[5] = (v2204_data + (v2176_data * v2202_data));
            float v2207_data = s3[82];
            float v2209_data = ir7[6];
            ir7[6] = (v2209_data + (v2176_data * v2207_data));
            float v2212_data = s3[94];
            float v2214_data = ir7[7];
            ir7[7] = (v2214_data + (v2176_data * v2212_data));
          }
          if (v12_lead < 12) {
            float v2220_data = r6[11];
            float v2221_data = s3[11];
            float v2223_data = ir7[0];
            ir7[0] = (v2223_data + (v2220_data * v2221_data));
            float v2226_data = s3[23];
            float v2228_data = ir7[1];
            ir7[1] = (v2228_data + (v2220_data * v2226_data));
            float v2231_data = s3[35];
            float v2233_data = ir7[2];
            ir7[2] = (v2233_data + (v2220_data * v2231_data));
            float v2236_data = s3[47];
            float v2238_data = ir7[3];
            ir7[3] = (v2238_data + (v2220_data * v2236_data));
            float v2241_data = s3[59];
            float v2243_data = ir7[4];
            ir7[4] = (v2243_data + (v2220_data * v2241_data));
            float v2246_data = s3[71];
            float v2248_data = ir7[5];
            ir7[5] = (v2248_data + (v2220_data * v2246_data));
            float v2251_data = s3[83];
            float v2253_data = ir7[6];
            ir7[6] = (v2253_data + (v2220_data * v2251_data));
            float v2256_data = s3[95];
            float v2258_data = ir7[7];
            ir7[7] = (v2258_data + (v2220_data * v2256_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2264_n1 = 0; v2264_n1 < 8; ++v2264_n1) {
              int32_t v2265_a = 0 + v2264_n1;
              float v2267_data = ir7[v2264_n1];
              int32_t v2268_a = 0 + v2264_n1;
              float v2270_data = r5[v2264_n1];
              r7[v2264_n1] = (v2270_data + v2267_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2277_i1 = 0; v2277_i1 < 8; ++v2277_i1) {
              int32_t v2278_a = 0 + v2277_i1;
              float v2280_data = r7[v2277_i1];
              glb_m0[(v12_lead + (v2277_i1 * 12))] = v2280_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

