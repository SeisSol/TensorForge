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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 2; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 13; ++v12_i1) {
              float v20_data = __ldcg(&glb_m0[(v17_lead + (v12_i1 * 64))]);
              r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 1; ++v28_i1) {
              int32_t v38_a = v10_lead + ((v28_i1 + 12) * 64);
              int32_t v41_a = v28_i1 * 2;
              #pragma unroll
              for (int32_t v29_i2 = 0; v29_i2 < 6; ++v29_i2) {
                float v40_data = glb_m2[(v38_a + (v29_i2 * 832))];
                r2[(v41_a + (v29_i2 * 2))] = v40_data;
              }
            }
          }
          if (v10_lead < 3) {
            int32_t v52_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 1; ++v46_i1) {
              int32_t v56_a = v52_lead + ((v46_i1 + 12) * 64);
              int32_t v61_a = 1 + (v46_i1 * 2);
              #pragma unroll
              for (int32_t v47_i2 = 0; v47_i2 < 6; ++v47_i2) {
                float v58_data = glb_m2[(v56_a + (v47_i2 * 832))];
                r2[(v61_a + (v47_i2 * 2))] = v58_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v67_data = r0[0];
          float v68_data = glb_m1[0];
          float v70_data = r1[0];
          r1[0] = (v70_data + (v67_data * v68_data));
          float v73_data = glb_m1[1];
          float v75_data = r1[26];
          r1[26] = (v75_data + (v67_data * v73_data));
          float v78_data = glb_m1[2];
          float v80_data = r1[52];
          r1[52] = (v80_data + (v67_data * v78_data));
          float v83_data = glb_m1[3];
          float v85_data = r1[78];
          r1[78] = (v85_data + (v67_data * v83_data));
          float v88_data = glb_m1[4];
          float v90_data = r1[104];
          r1[104] = (v90_data + (v67_data * v88_data));
          float v93_data = glb_m1[5];
          float v95_data = r1[130];
          r1[130] = (v95_data + (v67_data * v93_data));
          float v97_data = r0[2];
          float v100_data = r1[2];
          r1[2] = (v100_data + (v97_data * v68_data));
          float v105_data = r1[28];
          r1[28] = (v105_data + (v97_data * v73_data));
          float v110_data = r1[54];
          r1[54] = (v110_data + (v97_data * v78_data));
          float v115_data = r1[80];
          r1[80] = (v115_data + (v97_data * v83_data));
          float v120_data = r1[106];
          r1[106] = (v120_data + (v97_data * v88_data));
          float v125_data = r1[132];
          r1[132] = (v125_data + (v97_data * v93_data));
          float v127_data = r0[4];
          float v130_data = r1[4];
          r1[4] = (v130_data + (v127_data * v68_data));
          float v135_data = r1[30];
          r1[30] = (v135_data + (v127_data * v73_data));
          float v140_data = r1[56];
          r1[56] = (v140_data + (v127_data * v78_data));
          float v145_data = r1[82];
          r1[82] = (v145_data + (v127_data * v83_data));
          float v150_data = r1[108];
          r1[108] = (v150_data + (v127_data * v88_data));
          float v155_data = r1[134];
          r1[134] = (v155_data + (v127_data * v93_data));
          float v157_data = r0[6];
          float v160_data = r1[6];
          r1[6] = (v160_data + (v157_data * v68_data));
          float v165_data = r1[32];
          r1[32] = (v165_data + (v157_data * v73_data));
          float v170_data = r1[58];
          r1[58] = (v170_data + (v157_data * v78_data));
          float v175_data = r1[84];
          r1[84] = (v175_data + (v157_data * v83_data));
          float v180_data = r1[110];
          r1[110] = (v180_data + (v157_data * v88_data));
          float v185_data = r1[136];
          r1[136] = (v185_data + (v157_data * v93_data));
          float v187_data = r0[8];
          float v190_data = r1[8];
          r1[8] = (v190_data + (v187_data * v68_data));
          float v195_data = r1[34];
          r1[34] = (v195_data + (v187_data * v73_data));
          float v200_data = r1[60];
          r1[60] = (v200_data + (v187_data * v78_data));
          float v205_data = r1[86];
          r1[86] = (v205_data + (v187_data * v83_data));
          float v210_data = r1[112];
          r1[112] = (v210_data + (v187_data * v88_data));
          float v215_data = r1[138];
          r1[138] = (v215_data + (v187_data * v93_data));
          float v217_data = r0[10];
          float v220_data = r1[10];
          r1[10] = (v220_data + (v217_data * v68_data));
          float v225_data = r1[36];
          r1[36] = (v225_data + (v217_data * v73_data));
          float v230_data = r1[62];
          r1[62] = (v230_data + (v217_data * v78_data));
          float v235_data = r1[88];
          r1[88] = (v235_data + (v217_data * v83_data));
          float v240_data = r1[114];
          r1[114] = (v240_data + (v217_data * v88_data));
          float v245_data = r1[140];
          r1[140] = (v245_data + (v217_data * v93_data));
          float v247_data = r0[12];
          float v250_data = r1[12];
          r1[12] = (v250_data + (v247_data * v68_data));
          float v255_data = r1[38];
          r1[38] = (v255_data + (v247_data * v73_data));
          float v260_data = r1[64];
          r1[64] = (v260_data + (v247_data * v78_data));
          float v265_data = r1[90];
          r1[90] = (v265_data + (v247_data * v83_data));
          float v270_data = r1[116];
          r1[116] = (v270_data + (v247_data * v88_data));
          float v275_data = r1[142];
          r1[142] = (v275_data + (v247_data * v93_data));
          float v277_data = r0[14];
          float v280_data = r1[14];
          r1[14] = (v280_data + (v277_data * v68_data));
          float v285_data = r1[40];
          r1[40] = (v285_data + (v277_data * v73_data));
          float v290_data = r1[66];
          r1[66] = (v290_data + (v277_data * v78_data));
          float v295_data = r1[92];
          r1[92] = (v295_data + (v277_data * v83_data));
          float v300_data = r1[118];
          r1[118] = (v300_data + (v277_data * v88_data));
          float v305_data = r1[144];
          r1[144] = (v305_data + (v277_data * v93_data));
          float v307_data = r0[16];
          float v310_data = r1[16];
          r1[16] = (v310_data + (v307_data * v68_data));
          float v315_data = r1[42];
          r1[42] = (v315_data + (v307_data * v73_data));
          float v320_data = r1[68];
          r1[68] = (v320_data + (v307_data * v78_data));
          float v325_data = r1[94];
          r1[94] = (v325_data + (v307_data * v83_data));
          float v330_data = r1[120];
          r1[120] = (v330_data + (v307_data * v88_data));
          float v335_data = r1[146];
          r1[146] = (v335_data + (v307_data * v93_data));
          float v337_data = r0[18];
          float v340_data = r1[18];
          r1[18] = (v340_data + (v337_data * v68_data));
          float v345_data = r1[44];
          r1[44] = (v345_data + (v337_data * v73_data));
          float v350_data = r1[70];
          r1[70] = (v350_data + (v337_data * v78_data));
          float v355_data = r1[96];
          r1[96] = (v355_data + (v337_data * v83_data));
          float v360_data = r1[122];
          r1[122] = (v360_data + (v337_data * v88_data));
          float v365_data = r1[148];
          r1[148] = (v365_data + (v337_data * v93_data));
          float v367_data = r0[20];
          float v370_data = r1[20];
          r1[20] = (v370_data + (v367_data * v68_data));
          float v375_data = r1[46];
          r1[46] = (v375_data + (v367_data * v73_data));
          float v380_data = r1[72];
          r1[72] = (v380_data + (v367_data * v78_data));
          float v385_data = r1[98];
          r1[98] = (v385_data + (v367_data * v83_data));
          float v390_data = r1[124];
          r1[124] = (v390_data + (v367_data * v88_data));
          float v395_data = r1[150];
          r1[150] = (v395_data + (v367_data * v93_data));
          float v397_data = r0[22];
          float v400_data = r1[22];
          r1[22] = (v400_data + (v397_data * v68_data));
          float v405_data = r1[48];
          r1[48] = (v405_data + (v397_data * v73_data));
          float v410_data = r1[74];
          r1[74] = (v410_data + (v397_data * v78_data));
          float v415_data = r1[100];
          r1[100] = (v415_data + (v397_data * v83_data));
          float v420_data = r1[126];
          r1[126] = (v420_data + (v397_data * v88_data));
          float v425_data = r1[152];
          r1[152] = (v425_data + (v397_data * v93_data));
          float v427_data = r0[24];
          float v430_data = r1[24];
          r1[24] = (v430_data + (v427_data * v68_data));
          float v435_data = r1[50];
          r1[50] = (v435_data + (v427_data * v73_data));
          float v440_data = r1[76];
          r1[76] = (v440_data + (v427_data * v78_data));
          float v445_data = r1[102];
          r1[102] = (v445_data + (v427_data * v83_data));
          float v450_data = r1[128];
          r1[128] = (v450_data + (v427_data * v88_data));
          float v455_data = r1[154];
          r1[154] = (v455_data + (v427_data * v93_data));
          float v457_data = r0[1];
          float v460_data = r1[1];
          r1[1] = (v460_data + (v457_data * v68_data));
          float v465_data = r1[27];
          r1[27] = (v465_data + (v457_data * v73_data));
          float v470_data = r1[53];
          r1[53] = (v470_data + (v457_data * v78_data));
          float v475_data = r1[79];
          r1[79] = (v475_data + (v457_data * v83_data));
          float v480_data = r1[105];
          r1[105] = (v480_data + (v457_data * v88_data));
          float v485_data = r1[131];
          r1[131] = (v485_data + (v457_data * v93_data));
          float v487_data = r0[3];
          float v490_data = r1[3];
          r1[3] = (v490_data + (v487_data * v68_data));
          float v495_data = r1[29];
          r1[29] = (v495_data + (v487_data * v73_data));
          float v500_data = r1[55];
          r1[55] = (v500_data + (v487_data * v78_data));
          float v505_data = r1[81];
          r1[81] = (v505_data + (v487_data * v83_data));
          float v510_data = r1[107];
          r1[107] = (v510_data + (v487_data * v88_data));
          float v515_data = r1[133];
          r1[133] = (v515_data + (v487_data * v93_data));
          float v517_data = r0[5];
          float v520_data = r1[5];
          r1[5] = (v520_data + (v517_data * v68_data));
          float v525_data = r1[31];
          r1[31] = (v525_data + (v517_data * v73_data));
          float v530_data = r1[57];
          r1[57] = (v530_data + (v517_data * v78_data));
          float v535_data = r1[83];
          r1[83] = (v535_data + (v517_data * v83_data));
          float v540_data = r1[109];
          r1[109] = (v540_data + (v517_data * v88_data));
          float v545_data = r1[135];
          r1[135] = (v545_data + (v517_data * v93_data));
          float v547_data = r0[7];
          float v550_data = r1[7];
          r1[7] = (v550_data + (v547_data * v68_data));
          float v555_data = r1[33];
          r1[33] = (v555_data + (v547_data * v73_data));
          float v560_data = r1[59];
          r1[59] = (v560_data + (v547_data * v78_data));
          float v565_data = r1[85];
          r1[85] = (v565_data + (v547_data * v83_data));
          float v570_data = r1[111];
          r1[111] = (v570_data + (v547_data * v88_data));
          float v575_data = r1[137];
          r1[137] = (v575_data + (v547_data * v93_data));
          float v577_data = r0[9];
          float v580_data = r1[9];
          r1[9] = (v580_data + (v577_data * v68_data));
          float v585_data = r1[35];
          r1[35] = (v585_data + (v577_data * v73_data));
          float v590_data = r1[61];
          r1[61] = (v590_data + (v577_data * v78_data));
          float v595_data = r1[87];
          r1[87] = (v595_data + (v577_data * v83_data));
          float v600_data = r1[113];
          r1[113] = (v600_data + (v577_data * v88_data));
          float v605_data = r1[139];
          r1[139] = (v605_data + (v577_data * v93_data));
          float v607_data = r0[11];
          float v610_data = r1[11];
          r1[11] = (v610_data + (v607_data * v68_data));
          float v615_data = r1[37];
          r1[37] = (v615_data + (v607_data * v73_data));
          float v620_data = r1[63];
          r1[63] = (v620_data + (v607_data * v78_data));
          float v625_data = r1[89];
          r1[89] = (v625_data + (v607_data * v83_data));
          float v630_data = r1[115];
          r1[115] = (v630_data + (v607_data * v88_data));
          float v635_data = r1[141];
          r1[141] = (v635_data + (v607_data * v93_data));
          float v637_data = r0[13];
          float v640_data = r1[13];
          r1[13] = (v640_data + (v637_data * v68_data));
          float v645_data = r1[39];
          r1[39] = (v645_data + (v637_data * v73_data));
          float v650_data = r1[65];
          r1[65] = (v650_data + (v637_data * v78_data));
          float v655_data = r1[91];
          r1[91] = (v655_data + (v637_data * v83_data));
          float v660_data = r1[117];
          r1[117] = (v660_data + (v637_data * v88_data));
          float v665_data = r1[143];
          r1[143] = (v665_data + (v637_data * v93_data));
          float v667_data = r0[15];
          float v670_data = r1[15];
          r1[15] = (v670_data + (v667_data * v68_data));
          float v675_data = r1[41];
          r1[41] = (v675_data + (v667_data * v73_data));
          float v680_data = r1[67];
          r1[67] = (v680_data + (v667_data * v78_data));
          float v685_data = r1[93];
          r1[93] = (v685_data + (v667_data * v83_data));
          float v690_data = r1[119];
          r1[119] = (v690_data + (v667_data * v88_data));
          float v695_data = r1[145];
          r1[145] = (v695_data + (v667_data * v93_data));
          float v697_data = r0[17];
          float v700_data = r1[17];
          r1[17] = (v700_data + (v697_data * v68_data));
          float v705_data = r1[43];
          r1[43] = (v705_data + (v697_data * v73_data));
          float v710_data = r1[69];
          r1[69] = (v710_data + (v697_data * v78_data));
          float v715_data = r1[95];
          r1[95] = (v715_data + (v697_data * v83_data));
          float v720_data = r1[121];
          r1[121] = (v720_data + (v697_data * v88_data));
          float v725_data = r1[147];
          r1[147] = (v725_data + (v697_data * v93_data));
          float v727_data = r0[19];
          float v730_data = r1[19];
          r1[19] = (v730_data + (v727_data * v68_data));
          float v735_data = r1[45];
          r1[45] = (v735_data + (v727_data * v73_data));
          float v740_data = r1[71];
          r1[71] = (v740_data + (v727_data * v78_data));
          float v745_data = r1[97];
          r1[97] = (v745_data + (v727_data * v83_data));
          float v750_data = r1[123];
          r1[123] = (v750_data + (v727_data * v88_data));
          float v755_data = r1[149];
          r1[149] = (v755_data + (v727_data * v93_data));
          float v757_data = r0[21];
          float v760_data = r1[21];
          r1[21] = (v760_data + (v757_data * v68_data));
          float v765_data = r1[47];
          r1[47] = (v765_data + (v757_data * v73_data));
          float v770_data = r1[73];
          r1[73] = (v770_data + (v757_data * v78_data));
          float v775_data = r1[99];
          r1[99] = (v775_data + (v757_data * v83_data));
          float v780_data = r1[125];
          r1[125] = (v780_data + (v757_data * v88_data));
          float v785_data = r1[151];
          r1[151] = (v785_data + (v757_data * v93_data));
          float v787_data = r0[23];
          float v790_data = r1[23];
          r1[23] = (v790_data + (v787_data * v68_data));
          float v795_data = r1[49];
          r1[49] = (v795_data + (v787_data * v73_data));
          float v800_data = r1[75];
          r1[75] = (v800_data + (v787_data * v78_data));
          float v805_data = r1[101];
          r1[101] = (v805_data + (v787_data * v83_data));
          float v810_data = r1[127];
          r1[127] = (v810_data + (v787_data * v88_data));
          float v815_data = r1[153];
          r1[153] = (v815_data + (v787_data * v93_data));
          float v817_data = r0[25];
          float v820_data = r1[25];
          r1[25] = (v820_data + (v817_data * v68_data));
          float v825_data = r1[51];
          r1[51] = (v825_data + (v817_data * v73_data));
          float v830_data = r1[77];
          r1[77] = (v830_data + (v817_data * v78_data));
          float v835_data = r1[103];
          r1[103] = (v835_data + (v817_data * v83_data));
          float v840_data = r1[129];
          r1[129] = (v840_data + (v817_data * v88_data));
          float v845_data = r1[155];
          r1[155] = (v845_data + (v817_data * v93_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
          // [(20, 35), (0, 1), (0, 6)] []
          float ir3[12]{};
          if (v10_lead >= 20) {
            float v853_data = r1[24];
            float v854_data = ir3[0];
            ir3[0] = (v854_data + v853_data);
            float v856_data = r1[50];
            float v857_data = ir3[2];
            ir3[2] = (v857_data + v856_data);
            float v859_data = r1[76];
            float v860_data = ir3[4];
            ir3[4] = (v860_data + v859_data);
            float v862_data = r1[102];
            float v863_data = ir3[6];
            ir3[6] = (v863_data + v862_data);
            float v865_data = r1[128];
            float v866_data = ir3[8];
            ir3[8] = (v866_data + v865_data);
            float v868_data = r1[154];
            float v869_data = ir3[10];
            ir3[10] = (v869_data + v868_data);
          }
          if (v10_lead < 3) {
            float v872_data = r1[25];
            float v873_data = ir3[1];
            ir3[1] = (v873_data + v872_data);
            float v875_data = r1[51];
            float v876_data = ir3[3];
            ir3[3] = (v876_data + v875_data);
            float v878_data = r1[77];
            float v879_data = ir3[5];
            ir3[5] = (v879_data + v878_data);
            float v881_data = r1[103];
            float v882_data = ir3[7];
            ir3[7] = (v882_data + v881_data);
            float v884_data = r1[129];
            float v885_data = ir3[9];
            ir3[9] = (v885_data + v884_data);
            float v887_data = r1[155];
            float v888_data = ir3[11];
            ir3[11] = (v888_data + v887_data);
          }
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v894_n1 = 0; v894_n1 < 1; ++v894_n1) {
              int32_t v896_a = v894_n1 * 2;
              #pragma unroll
              for (int32_t v895_n2 = 0; v895_n2 < 6; ++v895_n2) {
                int32_t v899_a = v896_a + (v895_n2 * 2);
                float v900_data = ir3[v899_a];
                float v905_data = r2[v899_a];
                r3[v899_a] = (v905_data + v900_data);
              }
            }
          }
          if (v10_lead < 3) {
            #pragma unroll
            for (int32_t v912_n1 = 0; v912_n1 < 1; ++v912_n1) {
              int32_t v916_a = 1 + (v912_n1 * 2);
              #pragma unroll
              for (int32_t v913_n2 = 0; v913_n2 < 6; ++v913_n2) {
                int32_t v915_a = v913_n2 * 2;
                float v918_data = ir3[(v916_a + v915_a)];
                float v923_data = r2[(v916_a + v915_a)];
                r3[(v916_a + v915_a)] = (v923_data + v918_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v933_i1 = 0; v933_i1 < 1; ++v933_i1) {
              int32_t v935_a = v933_i1 * 2;
              int32_t v948_a = v10_lead + ((v933_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v934_i2 = 0; v934_i2 < 6; ++v934_i2) {
                float v939_data = r3[(v935_a + (v934_i2 * 2))];
                glb_m2[(v948_a + (v934_i2 * 832))] = v939_data;
              }
            }
          }
          if (v10_lead < 3) {
            int32_t v962_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v951_i1 = 0; v951_i1 < 1; ++v951_i1) {
              int32_t v955_a = 1 + (v951_i1 * 2);
              int32_t v966_a = v962_lead + ((v951_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v952_i2 = 0; v952_i2 < 6; ++v952_i2) {
                float v957_data = r3[(v955_a + (v952_i2 * 2))];
                glb_m2[(v966_a + (v952_i2 * 832))] = v957_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

