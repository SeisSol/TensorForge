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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 2; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 64;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __ldcg(&glb_m0[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 * 2);
              r0[v22_a] = v20_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v3_lead >= 20) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 1; ++v28_i1) {
              int32_t v36_a = (v28_i1 + 12) * 64;
              int32_t v38_a = v3_lead + v36_a;
              int32_t v48_a = v3_lead + v36_a;
              int32_t v51_a = v28_i1 * 2;
              #pragma unroll
              for (int32_t v29_i2 = 0; v29_i2 < 6; ++v29_i2) {
                int32_t v37_a = v29_i2 * 832;
                int32_t v39_a = v38_a + v37_a;
                float v50_data = glb_m2[(v48_a + v37_a)];
                int32_t v54_a = v51_a + (v29_i2 * 2);
                r2[v54_a] = v50_data;
              }
            }
          }
          if (v3_lead < 3) {
            int32_t v62_lead = v3_lead + 32_i32;
            int32_t v72_lead = v3_lead + 32_i32;
            #pragma unroll
            for (int32_t v56_i1 = 0; v56_i1 < 1; ++v56_i1) {
              int32_t v64_a = (v56_i1 + 12) * 64;
              int32_t v66_a = v62_lead + v64_a;
              int32_t v76_a = v72_lead + v64_a;
              int32_t v81_a = 1 + (v56_i1 * 2);
              #pragma unroll
              for (int32_t v57_i2 = 0; v57_i2 < 6; ++v57_i2) {
                int32_t v65_a = v57_i2 * 832;
                int32_t v67_a = v66_a + v65_a;
                float v78_data = glb_m2[(v76_a + v65_a)];
                int32_t v82_a = v81_a + (v57_i2 * 2);
                r2[v82_a] = v78_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v87_data = r0[0];
          float v88_data = glb_m1[0];
          float v90_data = ir1[0];
          ir1[0] = (v90_data + (v87_data * v88_data));
          float v93_data = glb_m1[1];
          float v95_data = ir1[26];
          ir1[26] = (v95_data + (v87_data * v93_data));
          float v98_data = glb_m1[2];
          float v100_data = ir1[52];
          ir1[52] = (v100_data + (v87_data * v98_data));
          float v103_data = glb_m1[3];
          float v105_data = ir1[78];
          ir1[78] = (v105_data + (v87_data * v103_data));
          float v108_data = glb_m1[4];
          float v110_data = ir1[104];
          ir1[104] = (v110_data + (v87_data * v108_data));
          float v113_data = glb_m1[5];
          float v115_data = ir1[130];
          ir1[130] = (v115_data + (v87_data * v113_data));
          float v117_data = r0[2];
          float v120_data = ir1[2];
          ir1[2] = (v120_data + (v117_data * v88_data));
          float v125_data = ir1[28];
          ir1[28] = (v125_data + (v117_data * v93_data));
          float v130_data = ir1[54];
          ir1[54] = (v130_data + (v117_data * v98_data));
          float v135_data = ir1[80];
          ir1[80] = (v135_data + (v117_data * v103_data));
          float v140_data = ir1[106];
          ir1[106] = (v140_data + (v117_data * v108_data));
          float v145_data = ir1[132];
          ir1[132] = (v145_data + (v117_data * v113_data));
          float v147_data = r0[4];
          float v150_data = ir1[4];
          ir1[4] = (v150_data + (v147_data * v88_data));
          float v155_data = ir1[30];
          ir1[30] = (v155_data + (v147_data * v93_data));
          float v160_data = ir1[56];
          ir1[56] = (v160_data + (v147_data * v98_data));
          float v165_data = ir1[82];
          ir1[82] = (v165_data + (v147_data * v103_data));
          float v170_data = ir1[108];
          ir1[108] = (v170_data + (v147_data * v108_data));
          float v175_data = ir1[134];
          ir1[134] = (v175_data + (v147_data * v113_data));
          float v177_data = r0[6];
          float v180_data = ir1[6];
          ir1[6] = (v180_data + (v177_data * v88_data));
          float v185_data = ir1[32];
          ir1[32] = (v185_data + (v177_data * v93_data));
          float v190_data = ir1[58];
          ir1[58] = (v190_data + (v177_data * v98_data));
          float v195_data = ir1[84];
          ir1[84] = (v195_data + (v177_data * v103_data));
          float v200_data = ir1[110];
          ir1[110] = (v200_data + (v177_data * v108_data));
          float v205_data = ir1[136];
          ir1[136] = (v205_data + (v177_data * v113_data));
          float v207_data = r0[8];
          float v210_data = ir1[8];
          ir1[8] = (v210_data + (v207_data * v88_data));
          float v215_data = ir1[34];
          ir1[34] = (v215_data + (v207_data * v93_data));
          float v220_data = ir1[60];
          ir1[60] = (v220_data + (v207_data * v98_data));
          float v225_data = ir1[86];
          ir1[86] = (v225_data + (v207_data * v103_data));
          float v230_data = ir1[112];
          ir1[112] = (v230_data + (v207_data * v108_data));
          float v235_data = ir1[138];
          ir1[138] = (v235_data + (v207_data * v113_data));
          float v237_data = r0[10];
          float v240_data = ir1[10];
          ir1[10] = (v240_data + (v237_data * v88_data));
          float v245_data = ir1[36];
          ir1[36] = (v245_data + (v237_data * v93_data));
          float v250_data = ir1[62];
          ir1[62] = (v250_data + (v237_data * v98_data));
          float v255_data = ir1[88];
          ir1[88] = (v255_data + (v237_data * v103_data));
          float v260_data = ir1[114];
          ir1[114] = (v260_data + (v237_data * v108_data));
          float v265_data = ir1[140];
          ir1[140] = (v265_data + (v237_data * v113_data));
          float v267_data = r0[12];
          float v270_data = ir1[12];
          ir1[12] = (v270_data + (v267_data * v88_data));
          float v275_data = ir1[38];
          ir1[38] = (v275_data + (v267_data * v93_data));
          float v280_data = ir1[64];
          ir1[64] = (v280_data + (v267_data * v98_data));
          float v285_data = ir1[90];
          ir1[90] = (v285_data + (v267_data * v103_data));
          float v290_data = ir1[116];
          ir1[116] = (v290_data + (v267_data * v108_data));
          float v295_data = ir1[142];
          ir1[142] = (v295_data + (v267_data * v113_data));
          float v297_data = r0[14];
          float v300_data = ir1[14];
          ir1[14] = (v300_data + (v297_data * v88_data));
          float v305_data = ir1[40];
          ir1[40] = (v305_data + (v297_data * v93_data));
          float v310_data = ir1[66];
          ir1[66] = (v310_data + (v297_data * v98_data));
          float v315_data = ir1[92];
          ir1[92] = (v315_data + (v297_data * v103_data));
          float v320_data = ir1[118];
          ir1[118] = (v320_data + (v297_data * v108_data));
          float v325_data = ir1[144];
          ir1[144] = (v325_data + (v297_data * v113_data));
          float v327_data = r0[16];
          float v330_data = ir1[16];
          ir1[16] = (v330_data + (v327_data * v88_data));
          float v335_data = ir1[42];
          ir1[42] = (v335_data + (v327_data * v93_data));
          float v340_data = ir1[68];
          ir1[68] = (v340_data + (v327_data * v98_data));
          float v345_data = ir1[94];
          ir1[94] = (v345_data + (v327_data * v103_data));
          float v350_data = ir1[120];
          ir1[120] = (v350_data + (v327_data * v108_data));
          float v355_data = ir1[146];
          ir1[146] = (v355_data + (v327_data * v113_data));
          float v357_data = r0[18];
          float v360_data = ir1[18];
          ir1[18] = (v360_data + (v357_data * v88_data));
          float v365_data = ir1[44];
          ir1[44] = (v365_data + (v357_data * v93_data));
          float v370_data = ir1[70];
          ir1[70] = (v370_data + (v357_data * v98_data));
          float v375_data = ir1[96];
          ir1[96] = (v375_data + (v357_data * v103_data));
          float v380_data = ir1[122];
          ir1[122] = (v380_data + (v357_data * v108_data));
          float v385_data = ir1[148];
          ir1[148] = (v385_data + (v357_data * v113_data));
          float v387_data = r0[20];
          float v390_data = ir1[20];
          ir1[20] = (v390_data + (v387_data * v88_data));
          float v395_data = ir1[46];
          ir1[46] = (v395_data + (v387_data * v93_data));
          float v400_data = ir1[72];
          ir1[72] = (v400_data + (v387_data * v98_data));
          float v405_data = ir1[98];
          ir1[98] = (v405_data + (v387_data * v103_data));
          float v410_data = ir1[124];
          ir1[124] = (v410_data + (v387_data * v108_data));
          float v415_data = ir1[150];
          ir1[150] = (v415_data + (v387_data * v113_data));
          float v417_data = r0[22];
          float v420_data = ir1[22];
          ir1[22] = (v420_data + (v417_data * v88_data));
          float v425_data = ir1[48];
          ir1[48] = (v425_data + (v417_data * v93_data));
          float v430_data = ir1[74];
          ir1[74] = (v430_data + (v417_data * v98_data));
          float v435_data = ir1[100];
          ir1[100] = (v435_data + (v417_data * v103_data));
          float v440_data = ir1[126];
          ir1[126] = (v440_data + (v417_data * v108_data));
          float v445_data = ir1[152];
          ir1[152] = (v445_data + (v417_data * v113_data));
          float v447_data = r0[24];
          float v450_data = ir1[24];
          ir1[24] = (v450_data + (v447_data * v88_data));
          float v455_data = ir1[50];
          ir1[50] = (v455_data + (v447_data * v93_data));
          float v460_data = ir1[76];
          ir1[76] = (v460_data + (v447_data * v98_data));
          float v465_data = ir1[102];
          ir1[102] = (v465_data + (v447_data * v103_data));
          float v470_data = ir1[128];
          ir1[128] = (v470_data + (v447_data * v108_data));
          float v475_data = ir1[154];
          ir1[154] = (v475_data + (v447_data * v113_data));
          float v477_data = r0[1];
          float v480_data = ir1[1];
          ir1[1] = (v480_data + (v477_data * v88_data));
          float v485_data = ir1[27];
          ir1[27] = (v485_data + (v477_data * v93_data));
          float v490_data = ir1[53];
          ir1[53] = (v490_data + (v477_data * v98_data));
          float v495_data = ir1[79];
          ir1[79] = (v495_data + (v477_data * v103_data));
          float v500_data = ir1[105];
          ir1[105] = (v500_data + (v477_data * v108_data));
          float v505_data = ir1[131];
          ir1[131] = (v505_data + (v477_data * v113_data));
          float v507_data = r0[3];
          float v510_data = ir1[3];
          ir1[3] = (v510_data + (v507_data * v88_data));
          float v515_data = ir1[29];
          ir1[29] = (v515_data + (v507_data * v93_data));
          float v520_data = ir1[55];
          ir1[55] = (v520_data + (v507_data * v98_data));
          float v525_data = ir1[81];
          ir1[81] = (v525_data + (v507_data * v103_data));
          float v530_data = ir1[107];
          ir1[107] = (v530_data + (v507_data * v108_data));
          float v535_data = ir1[133];
          ir1[133] = (v535_data + (v507_data * v113_data));
          float v537_data = r0[5];
          float v540_data = ir1[5];
          ir1[5] = (v540_data + (v537_data * v88_data));
          float v545_data = ir1[31];
          ir1[31] = (v545_data + (v537_data * v93_data));
          float v550_data = ir1[57];
          ir1[57] = (v550_data + (v537_data * v98_data));
          float v555_data = ir1[83];
          ir1[83] = (v555_data + (v537_data * v103_data));
          float v560_data = ir1[109];
          ir1[109] = (v560_data + (v537_data * v108_data));
          float v565_data = ir1[135];
          ir1[135] = (v565_data + (v537_data * v113_data));
          float v567_data = r0[7];
          float v570_data = ir1[7];
          ir1[7] = (v570_data + (v567_data * v88_data));
          float v575_data = ir1[33];
          ir1[33] = (v575_data + (v567_data * v93_data));
          float v580_data = ir1[59];
          ir1[59] = (v580_data + (v567_data * v98_data));
          float v585_data = ir1[85];
          ir1[85] = (v585_data + (v567_data * v103_data));
          float v590_data = ir1[111];
          ir1[111] = (v590_data + (v567_data * v108_data));
          float v595_data = ir1[137];
          ir1[137] = (v595_data + (v567_data * v113_data));
          float v597_data = r0[9];
          float v600_data = ir1[9];
          ir1[9] = (v600_data + (v597_data * v88_data));
          float v605_data = ir1[35];
          ir1[35] = (v605_data + (v597_data * v93_data));
          float v610_data = ir1[61];
          ir1[61] = (v610_data + (v597_data * v98_data));
          float v615_data = ir1[87];
          ir1[87] = (v615_data + (v597_data * v103_data));
          float v620_data = ir1[113];
          ir1[113] = (v620_data + (v597_data * v108_data));
          float v625_data = ir1[139];
          ir1[139] = (v625_data + (v597_data * v113_data));
          float v627_data = r0[11];
          float v630_data = ir1[11];
          ir1[11] = (v630_data + (v627_data * v88_data));
          float v635_data = ir1[37];
          ir1[37] = (v635_data + (v627_data * v93_data));
          float v640_data = ir1[63];
          ir1[63] = (v640_data + (v627_data * v98_data));
          float v645_data = ir1[89];
          ir1[89] = (v645_data + (v627_data * v103_data));
          float v650_data = ir1[115];
          ir1[115] = (v650_data + (v627_data * v108_data));
          float v655_data = ir1[141];
          ir1[141] = (v655_data + (v627_data * v113_data));
          float v657_data = r0[13];
          float v660_data = ir1[13];
          ir1[13] = (v660_data + (v657_data * v88_data));
          float v665_data = ir1[39];
          ir1[39] = (v665_data + (v657_data * v93_data));
          float v670_data = ir1[65];
          ir1[65] = (v670_data + (v657_data * v98_data));
          float v675_data = ir1[91];
          ir1[91] = (v675_data + (v657_data * v103_data));
          float v680_data = ir1[117];
          ir1[117] = (v680_data + (v657_data * v108_data));
          float v685_data = ir1[143];
          ir1[143] = (v685_data + (v657_data * v113_data));
          float v687_data = r0[15];
          float v690_data = ir1[15];
          ir1[15] = (v690_data + (v687_data * v88_data));
          float v695_data = ir1[41];
          ir1[41] = (v695_data + (v687_data * v93_data));
          float v700_data = ir1[67];
          ir1[67] = (v700_data + (v687_data * v98_data));
          float v705_data = ir1[93];
          ir1[93] = (v705_data + (v687_data * v103_data));
          float v710_data = ir1[119];
          ir1[119] = (v710_data + (v687_data * v108_data));
          float v715_data = ir1[145];
          ir1[145] = (v715_data + (v687_data * v113_data));
          float v717_data = r0[17];
          float v720_data = ir1[17];
          ir1[17] = (v720_data + (v717_data * v88_data));
          float v725_data = ir1[43];
          ir1[43] = (v725_data + (v717_data * v93_data));
          float v730_data = ir1[69];
          ir1[69] = (v730_data + (v717_data * v98_data));
          float v735_data = ir1[95];
          ir1[95] = (v735_data + (v717_data * v103_data));
          float v740_data = ir1[121];
          ir1[121] = (v740_data + (v717_data * v108_data));
          float v745_data = ir1[147];
          ir1[147] = (v745_data + (v717_data * v113_data));
          float v747_data = r0[19];
          float v750_data = ir1[19];
          ir1[19] = (v750_data + (v747_data * v88_data));
          float v755_data = ir1[45];
          ir1[45] = (v755_data + (v747_data * v93_data));
          float v760_data = ir1[71];
          ir1[71] = (v760_data + (v747_data * v98_data));
          float v765_data = ir1[97];
          ir1[97] = (v765_data + (v747_data * v103_data));
          float v770_data = ir1[123];
          ir1[123] = (v770_data + (v747_data * v108_data));
          float v775_data = ir1[149];
          ir1[149] = (v775_data + (v747_data * v113_data));
          float v777_data = r0[21];
          float v780_data = ir1[21];
          ir1[21] = (v780_data + (v777_data * v88_data));
          float v785_data = ir1[47];
          ir1[47] = (v785_data + (v777_data * v93_data));
          float v790_data = ir1[73];
          ir1[73] = (v790_data + (v777_data * v98_data));
          float v795_data = ir1[99];
          ir1[99] = (v795_data + (v777_data * v103_data));
          float v800_data = ir1[125];
          ir1[125] = (v800_data + (v777_data * v108_data));
          float v805_data = ir1[151];
          ir1[151] = (v805_data + (v777_data * v113_data));
          float v807_data = r0[23];
          float v810_data = ir1[23];
          ir1[23] = (v810_data + (v807_data * v88_data));
          float v815_data = ir1[49];
          ir1[49] = (v815_data + (v807_data * v93_data));
          float v820_data = ir1[75];
          ir1[75] = (v820_data + (v807_data * v98_data));
          float v825_data = ir1[101];
          ir1[101] = (v825_data + (v807_data * v103_data));
          float v830_data = ir1[127];
          ir1[127] = (v830_data + (v807_data * v108_data));
          float v835_data = ir1[153];
          ir1[153] = (v835_data + (v807_data * v113_data));
          float v837_data = r0[25];
          float v840_data = ir1[25];
          ir1[25] = (v840_data + (v837_data * v88_data));
          float v845_data = ir1[51];
          ir1[51] = (v845_data + (v837_data * v93_data));
          float v850_data = ir1[77];
          ir1[77] = (v850_data + (v837_data * v98_data));
          float v855_data = ir1[103];
          ir1[103] = (v855_data + (v837_data * v103_data));
          float v860_data = ir1[129];
          ir1[129] = (v860_data + (v837_data * v108_data));
          float v865_data = ir1[155];
          ir1[155] = (v865_data + (v837_data * v113_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          {
            // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
            // [(20, 35), (0, 1), (0, 6)] []
            float ir3[12]{};
            if (v3_lead >= 20) {
              float v872_data = r1[24];
              float v873_data = ir3[0];
              ir3[0] = (v873_data + v872_data);
              float v875_data = r1[50];
              float v876_data = ir3[2];
              ir3[2] = (v876_data + v875_data);
              float v878_data = r1[76];
              float v879_data = ir3[4];
              ir3[4] = (v879_data + v878_data);
              float v881_data = r1[102];
              float v882_data = ir3[6];
              ir3[6] = (v882_data + v881_data);
              float v884_data = r1[128];
              float v885_data = ir3[8];
              ir3[8] = (v885_data + v884_data);
              float v887_data = r1[154];
              float v888_data = ir3[10];
              ir3[10] = (v888_data + v887_data);
            }
            if (v3_lead < 3) {
              float v891_data = r1[25];
              float v892_data = ir3[1];
              ir3[1] = (v892_data + v891_data);
              float v894_data = r1[51];
              float v895_data = ir3[3];
              ir3[3] = (v895_data + v894_data);
              float v897_data = r1[77];
              float v898_data = ir3[5];
              ir3[5] = (v898_data + v897_data);
              float v900_data = r1[103];
              float v901_data = ir3[7];
              ir3[7] = (v901_data + v900_data);
              float v903_data = r1[129];
              float v904_data = ir3[9];
              ir3[9] = (v904_data + v903_data);
              float v906_data = r1[155];
              float v907_data = ir3[11];
              ir3[11] = (v907_data + v906_data);
            }
            if (v3_lead >= 20) {
              #pragma unroll
              for (int32_t v913_n1 = 0; v913_n1 < 1; ++v913_n1) {
                int32_t v915_a = v913_n1 * 2;
                #pragma unroll
                for (int32_t v914_n2 = 0; v914_n2 < 6; ++v914_n2) {
                  int32_t v916_a = v914_n2 * 2;
                  int32_t v918_a = v915_a + v916_a;
                  int32_t v922_a = v915_a + v916_a;
                  float v923_data = ir3[v922_a];
                  int32_t v927_a = v915_a + v916_a;
                  float v932_data = r2[v922_a];
                  int32_t v937_a = v915_a + v916_a;
                  r3[v922_a] = (v932_data + v923_data);
                }
              }
            }
            if (v3_lead < 3) {
              #pragma unroll
              for (int32_t v943_n1 = 0; v943_n1 < 1; ++v943_n1) {
                int32_t v947_a = 1 + (v943_n1 * 2);
                #pragma unroll
                for (int32_t v944_n2 = 0; v944_n2 < 6; ++v944_n2) {
                  int32_t v946_a = v944_n2 * 2;
                  int32_t v948_a = v947_a + v946_a;
                  float v953_data = ir3[(v947_a + v946_a)];
                  int32_t v957_a = v947_a + v946_a;
                  float v962_data = r2[(v947_a + v946_a)];
                  int32_t v967_a = v947_a + v946_a;
                  r3[(v947_a + v946_a)] = (v962_data + v953_data);
                }
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v3_lead >= 20) {
            #pragma unroll
            for (int32_t v976_i1 = 0; v976_i1 < 1; ++v976_i1) {
              int32_t v978_a = v976_i1 * 2;
              int32_t v995_a = v3_lead + ((v976_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v977_i2 = 0; v977_i2 < 6; ++v977_i2) {
                int32_t v979_a = v977_i2 * 2;
                int32_t v981_a = v978_a + v979_a;
                float v986_data = r3[(v978_a + v979_a)];
                int32_t v996_a = v995_a + (v977_i2 * 832);
                glb_m2[v996_a] = v986_data;
              }
            }
          }
          if (v3_lead < 3) {
            int32_t v1013_lead = v3_lead + 32_i32;
            #pragma unroll
            for (int32_t v998_i1 = 0; v998_i1 < 1; ++v998_i1) {
              int32_t v1002_a = 1 + (v998_i1 * 2);
              int32_t v1017_a = v1013_lead + ((v998_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v999_i2 = 0; v999_i2 < 6; ++v999_i2) {
                int32_t v1001_a = v999_i2 * 2;
                int32_t v1003_a = v1002_a + v1001_a;
                float v1008_data = r3[(v1002_a + v1001_a)];
                int32_t v1018_a = v1017_a + (v999_i2 * 832);
                glb_m2[v1018_a] = v1008_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

