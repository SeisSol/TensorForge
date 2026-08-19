// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_671a350836, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_671a350836, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_671a350836<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 64×13(64×13) {0..64}×{0..13} pointer_based
    // m1 6(6) {0..6} none
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
    // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      const float *const __restrict__ glb_m1 = &m1[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 2; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 64);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m0[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 * 2);
              r0[v14_a] = v12_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          int32_t v17_lead = threadIdx.x % 32;
          if (v17_lead >= 20) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 1; ++v19_i1) {
              int32_t v29_a = v17_lead + ((v19_i1 + 12) * 64);
              int32_t v32_a = v19_i1 * 2;
              #pragma unroll
              for (int32_t v20_i2 = 0; v20_i2 < 6; ++v20_i2) {
                int32_t v30_a = v29_a + (v20_i2 * 832);
                float v31_data;
                {
                  v31_data = glb_m2[v30_a];
                }
                int32_t v35_a = v32_a + (v20_i2 * 2);
                r2[v35_a] = v31_data;
              }
            }
          }
          if (v17_lead < 3) {
            int32_t v43_lead = v17_lead + 32_i32;
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 1; ++v37_i1) {
              int32_t v47_a = v43_lead + ((v37_i1 + 12) * 64);
              int32_t v52_a = 1 + (v37_i1 * 2);
              #pragma unroll
              for (int32_t v38_i2 = 0; v38_i2 < 6; ++v38_i2) {
                int32_t v48_a = v47_a + (v38_i2 * 832);
                float v49_data;
                {
                  v49_data = glb_m2[v48_a];
                }
                int32_t v53_a = v52_a + (v38_i2 * 2);
                r2[v53_a] = v49_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v57_data = r0[0];
          float v58_data;
          {
            v58_data = glb_m1[0];
          }
          float v60_data = ir1[0];
          ir1[0] = (v60_data + (v57_data * v58_data));
          float v63_data;
          {
            v63_data = glb_m1[1];
          }
          float v65_data = ir1[26];
          ir1[26] = (v65_data + (v57_data * v63_data));
          float v68_data;
          {
            v68_data = glb_m1[2];
          }
          float v70_data = ir1[52];
          ir1[52] = (v70_data + (v57_data * v68_data));
          float v73_data;
          {
            v73_data = glb_m1[3];
          }
          float v75_data = ir1[78];
          ir1[78] = (v75_data + (v57_data * v73_data));
          float v78_data;
          {
            v78_data = glb_m1[4];
          }
          float v80_data = ir1[104];
          ir1[104] = (v80_data + (v57_data * v78_data));
          float v83_data;
          {
            v83_data = glb_m1[5];
          }
          float v85_data = ir1[130];
          ir1[130] = (v85_data + (v57_data * v83_data));
          float v87_data = r0[2];
          float v88_data;
          {
            v88_data = glb_m1[0];
          }
          float v90_data = ir1[2];
          ir1[2] = (v90_data + (v87_data * v88_data));
          float v93_data;
          {
            v93_data = glb_m1[1];
          }
          float v95_data = ir1[28];
          ir1[28] = (v95_data + (v87_data * v93_data));
          float v98_data;
          {
            v98_data = glb_m1[2];
          }
          float v100_data = ir1[54];
          ir1[54] = (v100_data + (v87_data * v98_data));
          float v103_data;
          {
            v103_data = glb_m1[3];
          }
          float v105_data = ir1[80];
          ir1[80] = (v105_data + (v87_data * v103_data));
          float v108_data;
          {
            v108_data = glb_m1[4];
          }
          float v110_data = ir1[106];
          ir1[106] = (v110_data + (v87_data * v108_data));
          float v113_data;
          {
            v113_data = glb_m1[5];
          }
          float v115_data = ir1[132];
          ir1[132] = (v115_data + (v87_data * v113_data));
          float v117_data = r0[4];
          float v118_data;
          {
            v118_data = glb_m1[0];
          }
          float v120_data = ir1[4];
          ir1[4] = (v120_data + (v117_data * v118_data));
          float v123_data;
          {
            v123_data = glb_m1[1];
          }
          float v125_data = ir1[30];
          ir1[30] = (v125_data + (v117_data * v123_data));
          float v128_data;
          {
            v128_data = glb_m1[2];
          }
          float v130_data = ir1[56];
          ir1[56] = (v130_data + (v117_data * v128_data));
          float v133_data;
          {
            v133_data = glb_m1[3];
          }
          float v135_data = ir1[82];
          ir1[82] = (v135_data + (v117_data * v133_data));
          float v138_data;
          {
            v138_data = glb_m1[4];
          }
          float v140_data = ir1[108];
          ir1[108] = (v140_data + (v117_data * v138_data));
          float v143_data;
          {
            v143_data = glb_m1[5];
          }
          float v145_data = ir1[134];
          ir1[134] = (v145_data + (v117_data * v143_data));
          float v147_data = r0[6];
          float v148_data;
          {
            v148_data = glb_m1[0];
          }
          float v150_data = ir1[6];
          ir1[6] = (v150_data + (v147_data * v148_data));
          float v153_data;
          {
            v153_data = glb_m1[1];
          }
          float v155_data = ir1[32];
          ir1[32] = (v155_data + (v147_data * v153_data));
          float v158_data;
          {
            v158_data = glb_m1[2];
          }
          float v160_data = ir1[58];
          ir1[58] = (v160_data + (v147_data * v158_data));
          float v163_data;
          {
            v163_data = glb_m1[3];
          }
          float v165_data = ir1[84];
          ir1[84] = (v165_data + (v147_data * v163_data));
          float v168_data;
          {
            v168_data = glb_m1[4];
          }
          float v170_data = ir1[110];
          ir1[110] = (v170_data + (v147_data * v168_data));
          float v173_data;
          {
            v173_data = glb_m1[5];
          }
          float v175_data = ir1[136];
          ir1[136] = (v175_data + (v147_data * v173_data));
          float v177_data = r0[8];
          float v178_data;
          {
            v178_data = glb_m1[0];
          }
          float v180_data = ir1[8];
          ir1[8] = (v180_data + (v177_data * v178_data));
          float v183_data;
          {
            v183_data = glb_m1[1];
          }
          float v185_data = ir1[34];
          ir1[34] = (v185_data + (v177_data * v183_data));
          float v188_data;
          {
            v188_data = glb_m1[2];
          }
          float v190_data = ir1[60];
          ir1[60] = (v190_data + (v177_data * v188_data));
          float v193_data;
          {
            v193_data = glb_m1[3];
          }
          float v195_data = ir1[86];
          ir1[86] = (v195_data + (v177_data * v193_data));
          float v198_data;
          {
            v198_data = glb_m1[4];
          }
          float v200_data = ir1[112];
          ir1[112] = (v200_data + (v177_data * v198_data));
          float v203_data;
          {
            v203_data = glb_m1[5];
          }
          float v205_data = ir1[138];
          ir1[138] = (v205_data + (v177_data * v203_data));
          float v207_data = r0[10];
          float v208_data;
          {
            v208_data = glb_m1[0];
          }
          float v210_data = ir1[10];
          ir1[10] = (v210_data + (v207_data * v208_data));
          float v213_data;
          {
            v213_data = glb_m1[1];
          }
          float v215_data = ir1[36];
          ir1[36] = (v215_data + (v207_data * v213_data));
          float v218_data;
          {
            v218_data = glb_m1[2];
          }
          float v220_data = ir1[62];
          ir1[62] = (v220_data + (v207_data * v218_data));
          float v223_data;
          {
            v223_data = glb_m1[3];
          }
          float v225_data = ir1[88];
          ir1[88] = (v225_data + (v207_data * v223_data));
          float v228_data;
          {
            v228_data = glb_m1[4];
          }
          float v230_data = ir1[114];
          ir1[114] = (v230_data + (v207_data * v228_data));
          float v233_data;
          {
            v233_data = glb_m1[5];
          }
          float v235_data = ir1[140];
          ir1[140] = (v235_data + (v207_data * v233_data));
          float v237_data = r0[12];
          float v238_data;
          {
            v238_data = glb_m1[0];
          }
          float v240_data = ir1[12];
          ir1[12] = (v240_data + (v237_data * v238_data));
          float v243_data;
          {
            v243_data = glb_m1[1];
          }
          float v245_data = ir1[38];
          ir1[38] = (v245_data + (v237_data * v243_data));
          float v248_data;
          {
            v248_data = glb_m1[2];
          }
          float v250_data = ir1[64];
          ir1[64] = (v250_data + (v237_data * v248_data));
          float v253_data;
          {
            v253_data = glb_m1[3];
          }
          float v255_data = ir1[90];
          ir1[90] = (v255_data + (v237_data * v253_data));
          float v258_data;
          {
            v258_data = glb_m1[4];
          }
          float v260_data = ir1[116];
          ir1[116] = (v260_data + (v237_data * v258_data));
          float v263_data;
          {
            v263_data = glb_m1[5];
          }
          float v265_data = ir1[142];
          ir1[142] = (v265_data + (v237_data * v263_data));
          float v267_data = r0[14];
          float v268_data;
          {
            v268_data = glb_m1[0];
          }
          float v270_data = ir1[14];
          ir1[14] = (v270_data + (v267_data * v268_data));
          float v273_data;
          {
            v273_data = glb_m1[1];
          }
          float v275_data = ir1[40];
          ir1[40] = (v275_data + (v267_data * v273_data));
          float v278_data;
          {
            v278_data = glb_m1[2];
          }
          float v280_data = ir1[66];
          ir1[66] = (v280_data + (v267_data * v278_data));
          float v283_data;
          {
            v283_data = glb_m1[3];
          }
          float v285_data = ir1[92];
          ir1[92] = (v285_data + (v267_data * v283_data));
          float v288_data;
          {
            v288_data = glb_m1[4];
          }
          float v290_data = ir1[118];
          ir1[118] = (v290_data + (v267_data * v288_data));
          float v293_data;
          {
            v293_data = glb_m1[5];
          }
          float v295_data = ir1[144];
          ir1[144] = (v295_data + (v267_data * v293_data));
          float v297_data = r0[16];
          float v298_data;
          {
            v298_data = glb_m1[0];
          }
          float v300_data = ir1[16];
          ir1[16] = (v300_data + (v297_data * v298_data));
          float v303_data;
          {
            v303_data = glb_m1[1];
          }
          float v305_data = ir1[42];
          ir1[42] = (v305_data + (v297_data * v303_data));
          float v308_data;
          {
            v308_data = glb_m1[2];
          }
          float v310_data = ir1[68];
          ir1[68] = (v310_data + (v297_data * v308_data));
          float v313_data;
          {
            v313_data = glb_m1[3];
          }
          float v315_data = ir1[94];
          ir1[94] = (v315_data + (v297_data * v313_data));
          float v318_data;
          {
            v318_data = glb_m1[4];
          }
          float v320_data = ir1[120];
          ir1[120] = (v320_data + (v297_data * v318_data));
          float v323_data;
          {
            v323_data = glb_m1[5];
          }
          float v325_data = ir1[146];
          ir1[146] = (v325_data + (v297_data * v323_data));
          float v327_data = r0[18];
          float v328_data;
          {
            v328_data = glb_m1[0];
          }
          float v330_data = ir1[18];
          ir1[18] = (v330_data + (v327_data * v328_data));
          float v333_data;
          {
            v333_data = glb_m1[1];
          }
          float v335_data = ir1[44];
          ir1[44] = (v335_data + (v327_data * v333_data));
          float v338_data;
          {
            v338_data = glb_m1[2];
          }
          float v340_data = ir1[70];
          ir1[70] = (v340_data + (v327_data * v338_data));
          float v343_data;
          {
            v343_data = glb_m1[3];
          }
          float v345_data = ir1[96];
          ir1[96] = (v345_data + (v327_data * v343_data));
          float v348_data;
          {
            v348_data = glb_m1[4];
          }
          float v350_data = ir1[122];
          ir1[122] = (v350_data + (v327_data * v348_data));
          float v353_data;
          {
            v353_data = glb_m1[5];
          }
          float v355_data = ir1[148];
          ir1[148] = (v355_data + (v327_data * v353_data));
          float v357_data = r0[20];
          float v358_data;
          {
            v358_data = glb_m1[0];
          }
          float v360_data = ir1[20];
          ir1[20] = (v360_data + (v357_data * v358_data));
          float v363_data;
          {
            v363_data = glb_m1[1];
          }
          float v365_data = ir1[46];
          ir1[46] = (v365_data + (v357_data * v363_data));
          float v368_data;
          {
            v368_data = glb_m1[2];
          }
          float v370_data = ir1[72];
          ir1[72] = (v370_data + (v357_data * v368_data));
          float v373_data;
          {
            v373_data = glb_m1[3];
          }
          float v375_data = ir1[98];
          ir1[98] = (v375_data + (v357_data * v373_data));
          float v378_data;
          {
            v378_data = glb_m1[4];
          }
          float v380_data = ir1[124];
          ir1[124] = (v380_data + (v357_data * v378_data));
          float v383_data;
          {
            v383_data = glb_m1[5];
          }
          float v385_data = ir1[150];
          ir1[150] = (v385_data + (v357_data * v383_data));
          float v387_data = r0[22];
          float v388_data;
          {
            v388_data = glb_m1[0];
          }
          float v390_data = ir1[22];
          ir1[22] = (v390_data + (v387_data * v388_data));
          float v393_data;
          {
            v393_data = glb_m1[1];
          }
          float v395_data = ir1[48];
          ir1[48] = (v395_data + (v387_data * v393_data));
          float v398_data;
          {
            v398_data = glb_m1[2];
          }
          float v400_data = ir1[74];
          ir1[74] = (v400_data + (v387_data * v398_data));
          float v403_data;
          {
            v403_data = glb_m1[3];
          }
          float v405_data = ir1[100];
          ir1[100] = (v405_data + (v387_data * v403_data));
          float v408_data;
          {
            v408_data = glb_m1[4];
          }
          float v410_data = ir1[126];
          ir1[126] = (v410_data + (v387_data * v408_data));
          float v413_data;
          {
            v413_data = glb_m1[5];
          }
          float v415_data = ir1[152];
          ir1[152] = (v415_data + (v387_data * v413_data));
          float v417_data = r0[24];
          float v418_data;
          {
            v418_data = glb_m1[0];
          }
          float v420_data = ir1[24];
          ir1[24] = (v420_data + (v417_data * v418_data));
          float v423_data;
          {
            v423_data = glb_m1[1];
          }
          float v425_data = ir1[50];
          ir1[50] = (v425_data + (v417_data * v423_data));
          float v428_data;
          {
            v428_data = glb_m1[2];
          }
          float v430_data = ir1[76];
          ir1[76] = (v430_data + (v417_data * v428_data));
          float v433_data;
          {
            v433_data = glb_m1[3];
          }
          float v435_data = ir1[102];
          ir1[102] = (v435_data + (v417_data * v433_data));
          float v438_data;
          {
            v438_data = glb_m1[4];
          }
          float v440_data = ir1[128];
          ir1[128] = (v440_data + (v417_data * v438_data));
          float v443_data;
          {
            v443_data = glb_m1[5];
          }
          float v445_data = ir1[154];
          ir1[154] = (v445_data + (v417_data * v443_data));
          float v447_data = r0[1];
          float v448_data;
          {
            v448_data = glb_m1[0];
          }
          float v450_data = ir1[1];
          ir1[1] = (v450_data + (v447_data * v448_data));
          float v453_data;
          {
            v453_data = glb_m1[1];
          }
          float v455_data = ir1[27];
          ir1[27] = (v455_data + (v447_data * v453_data));
          float v458_data;
          {
            v458_data = glb_m1[2];
          }
          float v460_data = ir1[53];
          ir1[53] = (v460_data + (v447_data * v458_data));
          float v463_data;
          {
            v463_data = glb_m1[3];
          }
          float v465_data = ir1[79];
          ir1[79] = (v465_data + (v447_data * v463_data));
          float v468_data;
          {
            v468_data = glb_m1[4];
          }
          float v470_data = ir1[105];
          ir1[105] = (v470_data + (v447_data * v468_data));
          float v473_data;
          {
            v473_data = glb_m1[5];
          }
          float v475_data = ir1[131];
          ir1[131] = (v475_data + (v447_data * v473_data));
          float v477_data = r0[3];
          float v478_data;
          {
            v478_data = glb_m1[0];
          }
          float v480_data = ir1[3];
          ir1[3] = (v480_data + (v477_data * v478_data));
          float v483_data;
          {
            v483_data = glb_m1[1];
          }
          float v485_data = ir1[29];
          ir1[29] = (v485_data + (v477_data * v483_data));
          float v488_data;
          {
            v488_data = glb_m1[2];
          }
          float v490_data = ir1[55];
          ir1[55] = (v490_data + (v477_data * v488_data));
          float v493_data;
          {
            v493_data = glb_m1[3];
          }
          float v495_data = ir1[81];
          ir1[81] = (v495_data + (v477_data * v493_data));
          float v498_data;
          {
            v498_data = glb_m1[4];
          }
          float v500_data = ir1[107];
          ir1[107] = (v500_data + (v477_data * v498_data));
          float v503_data;
          {
            v503_data = glb_m1[5];
          }
          float v505_data = ir1[133];
          ir1[133] = (v505_data + (v477_data * v503_data));
          float v507_data = r0[5];
          float v508_data;
          {
            v508_data = glb_m1[0];
          }
          float v510_data = ir1[5];
          ir1[5] = (v510_data + (v507_data * v508_data));
          float v513_data;
          {
            v513_data = glb_m1[1];
          }
          float v515_data = ir1[31];
          ir1[31] = (v515_data + (v507_data * v513_data));
          float v518_data;
          {
            v518_data = glb_m1[2];
          }
          float v520_data = ir1[57];
          ir1[57] = (v520_data + (v507_data * v518_data));
          float v523_data;
          {
            v523_data = glb_m1[3];
          }
          float v525_data = ir1[83];
          ir1[83] = (v525_data + (v507_data * v523_data));
          float v528_data;
          {
            v528_data = glb_m1[4];
          }
          float v530_data = ir1[109];
          ir1[109] = (v530_data + (v507_data * v528_data));
          float v533_data;
          {
            v533_data = glb_m1[5];
          }
          float v535_data = ir1[135];
          ir1[135] = (v535_data + (v507_data * v533_data));
          float v537_data = r0[7];
          float v538_data;
          {
            v538_data = glb_m1[0];
          }
          float v540_data = ir1[7];
          ir1[7] = (v540_data + (v537_data * v538_data));
          float v543_data;
          {
            v543_data = glb_m1[1];
          }
          float v545_data = ir1[33];
          ir1[33] = (v545_data + (v537_data * v543_data));
          float v548_data;
          {
            v548_data = glb_m1[2];
          }
          float v550_data = ir1[59];
          ir1[59] = (v550_data + (v537_data * v548_data));
          float v553_data;
          {
            v553_data = glb_m1[3];
          }
          float v555_data = ir1[85];
          ir1[85] = (v555_data + (v537_data * v553_data));
          float v558_data;
          {
            v558_data = glb_m1[4];
          }
          float v560_data = ir1[111];
          ir1[111] = (v560_data + (v537_data * v558_data));
          float v563_data;
          {
            v563_data = glb_m1[5];
          }
          float v565_data = ir1[137];
          ir1[137] = (v565_data + (v537_data * v563_data));
          float v567_data = r0[9];
          float v568_data;
          {
            v568_data = glb_m1[0];
          }
          float v570_data = ir1[9];
          ir1[9] = (v570_data + (v567_data * v568_data));
          float v573_data;
          {
            v573_data = glb_m1[1];
          }
          float v575_data = ir1[35];
          ir1[35] = (v575_data + (v567_data * v573_data));
          float v578_data;
          {
            v578_data = glb_m1[2];
          }
          float v580_data = ir1[61];
          ir1[61] = (v580_data + (v567_data * v578_data));
          float v583_data;
          {
            v583_data = glb_m1[3];
          }
          float v585_data = ir1[87];
          ir1[87] = (v585_data + (v567_data * v583_data));
          float v588_data;
          {
            v588_data = glb_m1[4];
          }
          float v590_data = ir1[113];
          ir1[113] = (v590_data + (v567_data * v588_data));
          float v593_data;
          {
            v593_data = glb_m1[5];
          }
          float v595_data = ir1[139];
          ir1[139] = (v595_data + (v567_data * v593_data));
          float v597_data = r0[11];
          float v598_data;
          {
            v598_data = glb_m1[0];
          }
          float v600_data = ir1[11];
          ir1[11] = (v600_data + (v597_data * v598_data));
          float v603_data;
          {
            v603_data = glb_m1[1];
          }
          float v605_data = ir1[37];
          ir1[37] = (v605_data + (v597_data * v603_data));
          float v608_data;
          {
            v608_data = glb_m1[2];
          }
          float v610_data = ir1[63];
          ir1[63] = (v610_data + (v597_data * v608_data));
          float v613_data;
          {
            v613_data = glb_m1[3];
          }
          float v615_data = ir1[89];
          ir1[89] = (v615_data + (v597_data * v613_data));
          float v618_data;
          {
            v618_data = glb_m1[4];
          }
          float v620_data = ir1[115];
          ir1[115] = (v620_data + (v597_data * v618_data));
          float v623_data;
          {
            v623_data = glb_m1[5];
          }
          float v625_data = ir1[141];
          ir1[141] = (v625_data + (v597_data * v623_data));
          float v627_data = r0[13];
          float v628_data;
          {
            v628_data = glb_m1[0];
          }
          float v630_data = ir1[13];
          ir1[13] = (v630_data + (v627_data * v628_data));
          float v633_data;
          {
            v633_data = glb_m1[1];
          }
          float v635_data = ir1[39];
          ir1[39] = (v635_data + (v627_data * v633_data));
          float v638_data;
          {
            v638_data = glb_m1[2];
          }
          float v640_data = ir1[65];
          ir1[65] = (v640_data + (v627_data * v638_data));
          float v643_data;
          {
            v643_data = glb_m1[3];
          }
          float v645_data = ir1[91];
          ir1[91] = (v645_data + (v627_data * v643_data));
          float v648_data;
          {
            v648_data = glb_m1[4];
          }
          float v650_data = ir1[117];
          ir1[117] = (v650_data + (v627_data * v648_data));
          float v653_data;
          {
            v653_data = glb_m1[5];
          }
          float v655_data = ir1[143];
          ir1[143] = (v655_data + (v627_data * v653_data));
          float v657_data = r0[15];
          float v658_data;
          {
            v658_data = glb_m1[0];
          }
          float v660_data = ir1[15];
          ir1[15] = (v660_data + (v657_data * v658_data));
          float v663_data;
          {
            v663_data = glb_m1[1];
          }
          float v665_data = ir1[41];
          ir1[41] = (v665_data + (v657_data * v663_data));
          float v668_data;
          {
            v668_data = glb_m1[2];
          }
          float v670_data = ir1[67];
          ir1[67] = (v670_data + (v657_data * v668_data));
          float v673_data;
          {
            v673_data = glb_m1[3];
          }
          float v675_data = ir1[93];
          ir1[93] = (v675_data + (v657_data * v673_data));
          float v678_data;
          {
            v678_data = glb_m1[4];
          }
          float v680_data = ir1[119];
          ir1[119] = (v680_data + (v657_data * v678_data));
          float v683_data;
          {
            v683_data = glb_m1[5];
          }
          float v685_data = ir1[145];
          ir1[145] = (v685_data + (v657_data * v683_data));
          float v687_data = r0[17];
          float v688_data;
          {
            v688_data = glb_m1[0];
          }
          float v690_data = ir1[17];
          ir1[17] = (v690_data + (v687_data * v688_data));
          float v693_data;
          {
            v693_data = glb_m1[1];
          }
          float v695_data = ir1[43];
          ir1[43] = (v695_data + (v687_data * v693_data));
          float v698_data;
          {
            v698_data = glb_m1[2];
          }
          float v700_data = ir1[69];
          ir1[69] = (v700_data + (v687_data * v698_data));
          float v703_data;
          {
            v703_data = glb_m1[3];
          }
          float v705_data = ir1[95];
          ir1[95] = (v705_data + (v687_data * v703_data));
          float v708_data;
          {
            v708_data = glb_m1[4];
          }
          float v710_data = ir1[121];
          ir1[121] = (v710_data + (v687_data * v708_data));
          float v713_data;
          {
            v713_data = glb_m1[5];
          }
          float v715_data = ir1[147];
          ir1[147] = (v715_data + (v687_data * v713_data));
          float v717_data = r0[19];
          float v718_data;
          {
            v718_data = glb_m1[0];
          }
          float v720_data = ir1[19];
          ir1[19] = (v720_data + (v717_data * v718_data));
          float v723_data;
          {
            v723_data = glb_m1[1];
          }
          float v725_data = ir1[45];
          ir1[45] = (v725_data + (v717_data * v723_data));
          float v728_data;
          {
            v728_data = glb_m1[2];
          }
          float v730_data = ir1[71];
          ir1[71] = (v730_data + (v717_data * v728_data));
          float v733_data;
          {
            v733_data = glb_m1[3];
          }
          float v735_data = ir1[97];
          ir1[97] = (v735_data + (v717_data * v733_data));
          float v738_data;
          {
            v738_data = glb_m1[4];
          }
          float v740_data = ir1[123];
          ir1[123] = (v740_data + (v717_data * v738_data));
          float v743_data;
          {
            v743_data = glb_m1[5];
          }
          float v745_data = ir1[149];
          ir1[149] = (v745_data + (v717_data * v743_data));
          float v747_data = r0[21];
          float v748_data;
          {
            v748_data = glb_m1[0];
          }
          float v750_data = ir1[21];
          ir1[21] = (v750_data + (v747_data * v748_data));
          float v753_data;
          {
            v753_data = glb_m1[1];
          }
          float v755_data = ir1[47];
          ir1[47] = (v755_data + (v747_data * v753_data));
          float v758_data;
          {
            v758_data = glb_m1[2];
          }
          float v760_data = ir1[73];
          ir1[73] = (v760_data + (v747_data * v758_data));
          float v763_data;
          {
            v763_data = glb_m1[3];
          }
          float v765_data = ir1[99];
          ir1[99] = (v765_data + (v747_data * v763_data));
          float v768_data;
          {
            v768_data = glb_m1[4];
          }
          float v770_data = ir1[125];
          ir1[125] = (v770_data + (v747_data * v768_data));
          float v773_data;
          {
            v773_data = glb_m1[5];
          }
          float v775_data = ir1[151];
          ir1[151] = (v775_data + (v747_data * v773_data));
          float v777_data = r0[23];
          float v778_data;
          {
            v778_data = glb_m1[0];
          }
          float v780_data = ir1[23];
          ir1[23] = (v780_data + (v777_data * v778_data));
          float v783_data;
          {
            v783_data = glb_m1[1];
          }
          float v785_data = ir1[49];
          ir1[49] = (v785_data + (v777_data * v783_data));
          float v788_data;
          {
            v788_data = glb_m1[2];
          }
          float v790_data = ir1[75];
          ir1[75] = (v790_data + (v777_data * v788_data));
          float v793_data;
          {
            v793_data = glb_m1[3];
          }
          float v795_data = ir1[101];
          ir1[101] = (v795_data + (v777_data * v793_data));
          float v798_data;
          {
            v798_data = glb_m1[4];
          }
          float v800_data = ir1[127];
          ir1[127] = (v800_data + (v777_data * v798_data));
          float v803_data;
          {
            v803_data = glb_m1[5];
          }
          float v805_data = ir1[153];
          ir1[153] = (v805_data + (v777_data * v803_data));
          float v807_data = r0[25];
          float v808_data;
          {
            v808_data = glb_m1[0];
          }
          float v810_data = ir1[25];
          ir1[25] = (v810_data + (v807_data * v808_data));
          float v813_data;
          {
            v813_data = glb_m1[1];
          }
          float v815_data = ir1[51];
          ir1[51] = (v815_data + (v807_data * v813_data));
          float v818_data;
          {
            v818_data = glb_m1[2];
          }
          float v820_data = ir1[77];
          ir1[77] = (v820_data + (v807_data * v818_data));
          float v823_data;
          {
            v823_data = glb_m1[3];
          }
          float v825_data = ir1[103];
          ir1[103] = (v825_data + (v807_data * v823_data));
          float v828_data;
          {
            v828_data = glb_m1[4];
          }
          float v830_data = ir1[129];
          ir1[129] = (v830_data + (v807_data * v828_data));
          float v833_data;
          {
            v833_data = glb_m1[5];
          }
          float v835_data = ir1[155];
          ir1[155] = (v835_data + (v807_data * v833_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          {
            // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
            // [(20, 35), (0, 1), (0, 6)] []
            float ir3[12]{};
            int32_t v839_lead = threadIdx.x % 32;
            if (v839_lead >= 20) {
              float v841_data = r1[24];
              float v842_data = ir3[0];
              ir3[0] = (v842_data + v841_data);
              float v844_data = r1[50];
              float v845_data = ir3[2];
              ir3[2] = (v845_data + v844_data);
              float v847_data = r1[76];
              float v848_data = ir3[4];
              ir3[4] = (v848_data + v847_data);
              float v850_data = r1[102];
              float v851_data = ir3[6];
              ir3[6] = (v851_data + v850_data);
              float v853_data = r1[128];
              float v854_data = ir3[8];
              ir3[8] = (v854_data + v853_data);
              float v856_data = r1[154];
              float v857_data = ir3[10];
              ir3[10] = (v857_data + v856_data);
            }
            if (v839_lead < 3) {
              float v860_data = r1[25];
              float v861_data = ir3[1];
              ir3[1] = (v861_data + v860_data);
              float v863_data = r1[51];
              float v864_data = ir3[3];
              ir3[3] = (v864_data + v863_data);
              float v866_data = r1[77];
              float v867_data = ir3[5];
              ir3[5] = (v867_data + v866_data);
              float v869_data = r1[103];
              float v870_data = ir3[7];
              ir3[7] = (v870_data + v869_data);
              float v872_data = r1[129];
              float v873_data = ir3[9];
              ir3[9] = (v873_data + v872_data);
              float v875_data = r1[155];
              float v876_data = ir3[11];
              ir3[11] = (v876_data + v875_data);
            }
            if (v839_lead >= 20) {
              #pragma unroll
              for (int32_t v882_n1 = 0; v882_n1 < 1; ++v882_n1) {
                int32_t v884_a = v882_n1 * 2;
                #pragma unroll
                for (int32_t v883_n2 = 0; v883_n2 < 6; ++v883_n2) {
                  int32_t v885_a = v883_n2 * 2;
                  int32_t v887_a = v884_a + v885_a;
                  int32_t v891_a = v884_a + v885_a;
                  float v892_data = ir3[v891_a];
                  int32_t v896_a = v884_a + v885_a;
                  float v901_data = r2[v891_a];
                  int32_t v906_a = v884_a + v885_a;
                  r3[v891_a] = (v901_data + v892_data);
                }
              }
            }
            if (v839_lead < 3) {
              #pragma unroll
              for (int32_t v912_n1 = 0; v912_n1 < 1; ++v912_n1) {
                int32_t v916_a = 1 + (v912_n1 * 2);
                #pragma unroll
                for (int32_t v913_n2 = 0; v913_n2 < 6; ++v913_n2) {
                  int32_t v915_a = v913_n2 * 2;
                  int32_t v917_a = v916_a + v915_a;
                  float v922_data = ir3[(v916_a + v915_a)];
                  int32_t v926_a = v916_a + v915_a;
                  float v931_data = r2[(v916_a + v915_a)];
                  int32_t v936_a = v916_a + v915_a;
                  r3[(v916_a + v915_a)] = (v931_data + v922_data);
                }
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          int32_t v943_lead = threadIdx.x % 32;
          if (v943_lead >= 20) {
            #pragma unroll
            for (int32_t v945_i1 = 0; v945_i1 < 1; ++v945_i1) {
              int32_t v947_a = v945_i1 * 2;
              int32_t v964_a = v943_lead + ((v945_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v946_i2 = 0; v946_i2 < 6; ++v946_i2) {
                int32_t v948_a = v946_i2 * 2;
                int32_t v950_a = v947_a + v948_a;
                float v955_data = r3[(v947_a + v948_a)];
                int32_t v965_a = v964_a + (v946_i2 * 832);
                glb_m2[v965_a] = v955_data;
              }
            }
          }
          if (v943_lead < 3) {
            int32_t v982_lead = v943_lead + 32_i32;
            #pragma unroll
            for (int32_t v967_i1 = 0; v967_i1 < 1; ++v967_i1) {
              int32_t v971_a = 1 + (v967_i1 * 2);
              int32_t v986_a = v982_lead + ((v967_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v968_i2 = 0; v968_i2 < 6; ++v968_i2) {
                int32_t v970_a = v968_i2 * 2;
                int32_t v972_a = v971_a + v970_a;
                float v977_data = r3[(v971_a + v970_a)];
                int32_t v987_a = v986_a + (v968_i2 * 832);
                glb_m2[v987_a] = v977_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

