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
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 13; ++v12_i1) {
              int32_t v18_a = v12_i1 * 64;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __ldcg(&glb_m0[(v24_lead + v18_a)]);
              r0[(v11_i0 + (v12_i1 * 2))] = v27_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 1; ++v35_i1) {
              int32_t v43_a = (v35_i1 + 12) * 64;
              int32_t v45_a = v10_lead + v43_a;
              int32_t v55_a = v10_lead + v43_a;
              int32_t v58_a = v35_i1 * 2;
              #pragma unroll
              for (int32_t v36_i2 = 0; v36_i2 < 6; ++v36_i2) {
                int32_t v44_a = v36_i2 * 832;
                int32_t v46_a = v45_a + v44_a;
                float v57_data = glb_m2[(v55_a + v44_a)];
                r2[(v58_a + (v36_i2 * 2))] = v57_data;
              }
            }
          }
          if (v10_lead < 3) {
            int32_t v69_lead = v10_lead + 32_i32;
            int32_t v79_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 1; ++v63_i1) {
              int32_t v71_a = (v63_i1 + 12) * 64;
              int32_t v73_a = v69_lead + v71_a;
              int32_t v83_a = v79_lead + v71_a;
              int32_t v88_a = 1 + (v63_i1 * 2);
              #pragma unroll
              for (int32_t v64_i2 = 0; v64_i2 < 6; ++v64_i2) {
                int32_t v72_a = v64_i2 * 832;
                int32_t v74_a = v73_a + v72_a;
                float v85_data = glb_m2[(v83_a + v72_a)];
                r2[(v88_a + (v64_i2 * 2))] = v85_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v94_data = r0[0];
          float v95_data = glb_m1[0];
          float v97_data = r1[0];
          r1[0] = (v97_data + (v94_data * v95_data));
          float v100_data = glb_m1[1];
          float v102_data = r1[26];
          r1[26] = (v102_data + (v94_data * v100_data));
          float v105_data = glb_m1[2];
          float v107_data = r1[52];
          r1[52] = (v107_data + (v94_data * v105_data));
          float v110_data = glb_m1[3];
          float v112_data = r1[78];
          r1[78] = (v112_data + (v94_data * v110_data));
          float v115_data = glb_m1[4];
          float v117_data = r1[104];
          r1[104] = (v117_data + (v94_data * v115_data));
          float v120_data = glb_m1[5];
          float v122_data = r1[130];
          r1[130] = (v122_data + (v94_data * v120_data));
          float v124_data = r0[2];
          float v127_data = r1[2];
          r1[2] = (v127_data + (v124_data * v95_data));
          float v132_data = r1[28];
          r1[28] = (v132_data + (v124_data * v100_data));
          float v137_data = r1[54];
          r1[54] = (v137_data + (v124_data * v105_data));
          float v142_data = r1[80];
          r1[80] = (v142_data + (v124_data * v110_data));
          float v147_data = r1[106];
          r1[106] = (v147_data + (v124_data * v115_data));
          float v152_data = r1[132];
          r1[132] = (v152_data + (v124_data * v120_data));
          float v154_data = r0[4];
          float v157_data = r1[4];
          r1[4] = (v157_data + (v154_data * v95_data));
          float v162_data = r1[30];
          r1[30] = (v162_data + (v154_data * v100_data));
          float v167_data = r1[56];
          r1[56] = (v167_data + (v154_data * v105_data));
          float v172_data = r1[82];
          r1[82] = (v172_data + (v154_data * v110_data));
          float v177_data = r1[108];
          r1[108] = (v177_data + (v154_data * v115_data));
          float v182_data = r1[134];
          r1[134] = (v182_data + (v154_data * v120_data));
          float v184_data = r0[6];
          float v187_data = r1[6];
          r1[6] = (v187_data + (v184_data * v95_data));
          float v192_data = r1[32];
          r1[32] = (v192_data + (v184_data * v100_data));
          float v197_data = r1[58];
          r1[58] = (v197_data + (v184_data * v105_data));
          float v202_data = r1[84];
          r1[84] = (v202_data + (v184_data * v110_data));
          float v207_data = r1[110];
          r1[110] = (v207_data + (v184_data * v115_data));
          float v212_data = r1[136];
          r1[136] = (v212_data + (v184_data * v120_data));
          float v214_data = r0[8];
          float v217_data = r1[8];
          r1[8] = (v217_data + (v214_data * v95_data));
          float v222_data = r1[34];
          r1[34] = (v222_data + (v214_data * v100_data));
          float v227_data = r1[60];
          r1[60] = (v227_data + (v214_data * v105_data));
          float v232_data = r1[86];
          r1[86] = (v232_data + (v214_data * v110_data));
          float v237_data = r1[112];
          r1[112] = (v237_data + (v214_data * v115_data));
          float v242_data = r1[138];
          r1[138] = (v242_data + (v214_data * v120_data));
          float v244_data = r0[10];
          float v247_data = r1[10];
          r1[10] = (v247_data + (v244_data * v95_data));
          float v252_data = r1[36];
          r1[36] = (v252_data + (v244_data * v100_data));
          float v257_data = r1[62];
          r1[62] = (v257_data + (v244_data * v105_data));
          float v262_data = r1[88];
          r1[88] = (v262_data + (v244_data * v110_data));
          float v267_data = r1[114];
          r1[114] = (v267_data + (v244_data * v115_data));
          float v272_data = r1[140];
          r1[140] = (v272_data + (v244_data * v120_data));
          float v274_data = r0[12];
          float v277_data = r1[12];
          r1[12] = (v277_data + (v274_data * v95_data));
          float v282_data = r1[38];
          r1[38] = (v282_data + (v274_data * v100_data));
          float v287_data = r1[64];
          r1[64] = (v287_data + (v274_data * v105_data));
          float v292_data = r1[90];
          r1[90] = (v292_data + (v274_data * v110_data));
          float v297_data = r1[116];
          r1[116] = (v297_data + (v274_data * v115_data));
          float v302_data = r1[142];
          r1[142] = (v302_data + (v274_data * v120_data));
          float v304_data = r0[14];
          float v307_data = r1[14];
          r1[14] = (v307_data + (v304_data * v95_data));
          float v312_data = r1[40];
          r1[40] = (v312_data + (v304_data * v100_data));
          float v317_data = r1[66];
          r1[66] = (v317_data + (v304_data * v105_data));
          float v322_data = r1[92];
          r1[92] = (v322_data + (v304_data * v110_data));
          float v327_data = r1[118];
          r1[118] = (v327_data + (v304_data * v115_data));
          float v332_data = r1[144];
          r1[144] = (v332_data + (v304_data * v120_data));
          float v334_data = r0[16];
          float v337_data = r1[16];
          r1[16] = (v337_data + (v334_data * v95_data));
          float v342_data = r1[42];
          r1[42] = (v342_data + (v334_data * v100_data));
          float v347_data = r1[68];
          r1[68] = (v347_data + (v334_data * v105_data));
          float v352_data = r1[94];
          r1[94] = (v352_data + (v334_data * v110_data));
          float v357_data = r1[120];
          r1[120] = (v357_data + (v334_data * v115_data));
          float v362_data = r1[146];
          r1[146] = (v362_data + (v334_data * v120_data));
          float v364_data = r0[18];
          float v367_data = r1[18];
          r1[18] = (v367_data + (v364_data * v95_data));
          float v372_data = r1[44];
          r1[44] = (v372_data + (v364_data * v100_data));
          float v377_data = r1[70];
          r1[70] = (v377_data + (v364_data * v105_data));
          float v382_data = r1[96];
          r1[96] = (v382_data + (v364_data * v110_data));
          float v387_data = r1[122];
          r1[122] = (v387_data + (v364_data * v115_data));
          float v392_data = r1[148];
          r1[148] = (v392_data + (v364_data * v120_data));
          float v394_data = r0[20];
          float v397_data = r1[20];
          r1[20] = (v397_data + (v394_data * v95_data));
          float v402_data = r1[46];
          r1[46] = (v402_data + (v394_data * v100_data));
          float v407_data = r1[72];
          r1[72] = (v407_data + (v394_data * v105_data));
          float v412_data = r1[98];
          r1[98] = (v412_data + (v394_data * v110_data));
          float v417_data = r1[124];
          r1[124] = (v417_data + (v394_data * v115_data));
          float v422_data = r1[150];
          r1[150] = (v422_data + (v394_data * v120_data));
          float v424_data = r0[22];
          float v427_data = r1[22];
          r1[22] = (v427_data + (v424_data * v95_data));
          float v432_data = r1[48];
          r1[48] = (v432_data + (v424_data * v100_data));
          float v437_data = r1[74];
          r1[74] = (v437_data + (v424_data * v105_data));
          float v442_data = r1[100];
          r1[100] = (v442_data + (v424_data * v110_data));
          float v447_data = r1[126];
          r1[126] = (v447_data + (v424_data * v115_data));
          float v452_data = r1[152];
          r1[152] = (v452_data + (v424_data * v120_data));
          float v454_data = r0[24];
          float v457_data = r1[24];
          r1[24] = (v457_data + (v454_data * v95_data));
          float v462_data = r1[50];
          r1[50] = (v462_data + (v454_data * v100_data));
          float v467_data = r1[76];
          r1[76] = (v467_data + (v454_data * v105_data));
          float v472_data = r1[102];
          r1[102] = (v472_data + (v454_data * v110_data));
          float v477_data = r1[128];
          r1[128] = (v477_data + (v454_data * v115_data));
          float v482_data = r1[154];
          r1[154] = (v482_data + (v454_data * v120_data));
          float v484_data = r0[1];
          float v487_data = r1[1];
          r1[1] = (v487_data + (v484_data * v95_data));
          float v492_data = r1[27];
          r1[27] = (v492_data + (v484_data * v100_data));
          float v497_data = r1[53];
          r1[53] = (v497_data + (v484_data * v105_data));
          float v502_data = r1[79];
          r1[79] = (v502_data + (v484_data * v110_data));
          float v507_data = r1[105];
          r1[105] = (v507_data + (v484_data * v115_data));
          float v512_data = r1[131];
          r1[131] = (v512_data + (v484_data * v120_data));
          float v514_data = r0[3];
          float v517_data = r1[3];
          r1[3] = (v517_data + (v514_data * v95_data));
          float v522_data = r1[29];
          r1[29] = (v522_data + (v514_data * v100_data));
          float v527_data = r1[55];
          r1[55] = (v527_data + (v514_data * v105_data));
          float v532_data = r1[81];
          r1[81] = (v532_data + (v514_data * v110_data));
          float v537_data = r1[107];
          r1[107] = (v537_data + (v514_data * v115_data));
          float v542_data = r1[133];
          r1[133] = (v542_data + (v514_data * v120_data));
          float v544_data = r0[5];
          float v547_data = r1[5];
          r1[5] = (v547_data + (v544_data * v95_data));
          float v552_data = r1[31];
          r1[31] = (v552_data + (v544_data * v100_data));
          float v557_data = r1[57];
          r1[57] = (v557_data + (v544_data * v105_data));
          float v562_data = r1[83];
          r1[83] = (v562_data + (v544_data * v110_data));
          float v567_data = r1[109];
          r1[109] = (v567_data + (v544_data * v115_data));
          float v572_data = r1[135];
          r1[135] = (v572_data + (v544_data * v120_data));
          float v574_data = r0[7];
          float v577_data = r1[7];
          r1[7] = (v577_data + (v574_data * v95_data));
          float v582_data = r1[33];
          r1[33] = (v582_data + (v574_data * v100_data));
          float v587_data = r1[59];
          r1[59] = (v587_data + (v574_data * v105_data));
          float v592_data = r1[85];
          r1[85] = (v592_data + (v574_data * v110_data));
          float v597_data = r1[111];
          r1[111] = (v597_data + (v574_data * v115_data));
          float v602_data = r1[137];
          r1[137] = (v602_data + (v574_data * v120_data));
          float v604_data = r0[9];
          float v607_data = r1[9];
          r1[9] = (v607_data + (v604_data * v95_data));
          float v612_data = r1[35];
          r1[35] = (v612_data + (v604_data * v100_data));
          float v617_data = r1[61];
          r1[61] = (v617_data + (v604_data * v105_data));
          float v622_data = r1[87];
          r1[87] = (v622_data + (v604_data * v110_data));
          float v627_data = r1[113];
          r1[113] = (v627_data + (v604_data * v115_data));
          float v632_data = r1[139];
          r1[139] = (v632_data + (v604_data * v120_data));
          float v634_data = r0[11];
          float v637_data = r1[11];
          r1[11] = (v637_data + (v634_data * v95_data));
          float v642_data = r1[37];
          r1[37] = (v642_data + (v634_data * v100_data));
          float v647_data = r1[63];
          r1[63] = (v647_data + (v634_data * v105_data));
          float v652_data = r1[89];
          r1[89] = (v652_data + (v634_data * v110_data));
          float v657_data = r1[115];
          r1[115] = (v657_data + (v634_data * v115_data));
          float v662_data = r1[141];
          r1[141] = (v662_data + (v634_data * v120_data));
          float v664_data = r0[13];
          float v667_data = r1[13];
          r1[13] = (v667_data + (v664_data * v95_data));
          float v672_data = r1[39];
          r1[39] = (v672_data + (v664_data * v100_data));
          float v677_data = r1[65];
          r1[65] = (v677_data + (v664_data * v105_data));
          float v682_data = r1[91];
          r1[91] = (v682_data + (v664_data * v110_data));
          float v687_data = r1[117];
          r1[117] = (v687_data + (v664_data * v115_data));
          float v692_data = r1[143];
          r1[143] = (v692_data + (v664_data * v120_data));
          float v694_data = r0[15];
          float v697_data = r1[15];
          r1[15] = (v697_data + (v694_data * v95_data));
          float v702_data = r1[41];
          r1[41] = (v702_data + (v694_data * v100_data));
          float v707_data = r1[67];
          r1[67] = (v707_data + (v694_data * v105_data));
          float v712_data = r1[93];
          r1[93] = (v712_data + (v694_data * v110_data));
          float v717_data = r1[119];
          r1[119] = (v717_data + (v694_data * v115_data));
          float v722_data = r1[145];
          r1[145] = (v722_data + (v694_data * v120_data));
          float v724_data = r0[17];
          float v727_data = r1[17];
          r1[17] = (v727_data + (v724_data * v95_data));
          float v732_data = r1[43];
          r1[43] = (v732_data + (v724_data * v100_data));
          float v737_data = r1[69];
          r1[69] = (v737_data + (v724_data * v105_data));
          float v742_data = r1[95];
          r1[95] = (v742_data + (v724_data * v110_data));
          float v747_data = r1[121];
          r1[121] = (v747_data + (v724_data * v115_data));
          float v752_data = r1[147];
          r1[147] = (v752_data + (v724_data * v120_data));
          float v754_data = r0[19];
          float v757_data = r1[19];
          r1[19] = (v757_data + (v754_data * v95_data));
          float v762_data = r1[45];
          r1[45] = (v762_data + (v754_data * v100_data));
          float v767_data = r1[71];
          r1[71] = (v767_data + (v754_data * v105_data));
          float v772_data = r1[97];
          r1[97] = (v772_data + (v754_data * v110_data));
          float v777_data = r1[123];
          r1[123] = (v777_data + (v754_data * v115_data));
          float v782_data = r1[149];
          r1[149] = (v782_data + (v754_data * v120_data));
          float v784_data = r0[21];
          float v787_data = r1[21];
          r1[21] = (v787_data + (v784_data * v95_data));
          float v792_data = r1[47];
          r1[47] = (v792_data + (v784_data * v100_data));
          float v797_data = r1[73];
          r1[73] = (v797_data + (v784_data * v105_data));
          float v802_data = r1[99];
          r1[99] = (v802_data + (v784_data * v110_data));
          float v807_data = r1[125];
          r1[125] = (v807_data + (v784_data * v115_data));
          float v812_data = r1[151];
          r1[151] = (v812_data + (v784_data * v120_data));
          float v814_data = r0[23];
          float v817_data = r1[23];
          r1[23] = (v817_data + (v814_data * v95_data));
          float v822_data = r1[49];
          r1[49] = (v822_data + (v814_data * v100_data));
          float v827_data = r1[75];
          r1[75] = (v827_data + (v814_data * v105_data));
          float v832_data = r1[101];
          r1[101] = (v832_data + (v814_data * v110_data));
          float v837_data = r1[127];
          r1[127] = (v837_data + (v814_data * v115_data));
          float v842_data = r1[153];
          r1[153] = (v842_data + (v814_data * v120_data));
          float v844_data = r0[25];
          float v847_data = r1[25];
          r1[25] = (v847_data + (v844_data * v95_data));
          float v852_data = r1[51];
          r1[51] = (v852_data + (v844_data * v100_data));
          float v857_data = r1[77];
          r1[77] = (v857_data + (v844_data * v105_data));
          float v862_data = r1[103];
          r1[103] = (v862_data + (v844_data * v110_data));
          float v867_data = r1[129];
          r1[129] = (v867_data + (v844_data * v115_data));
          float v872_data = r1[155];
          r1[155] = (v872_data + (v844_data * v120_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
          // [(20, 35), (0, 1), (0, 6)] []
          float ir3[12]{};
          if (v10_lead >= 20) {
            float v880_data = r1[24];
            float v881_data = ir3[0];
            ir3[0] = (v881_data + v880_data);
            float v883_data = r1[50];
            float v884_data = ir3[2];
            ir3[2] = (v884_data + v883_data);
            float v886_data = r1[76];
            float v887_data = ir3[4];
            ir3[4] = (v887_data + v886_data);
            float v889_data = r1[102];
            float v890_data = ir3[6];
            ir3[6] = (v890_data + v889_data);
            float v892_data = r1[128];
            float v893_data = ir3[8];
            ir3[8] = (v893_data + v892_data);
            float v895_data = r1[154];
            float v896_data = ir3[10];
            ir3[10] = (v896_data + v895_data);
          }
          if (v10_lead < 3) {
            float v899_data = r1[25];
            float v900_data = ir3[1];
            ir3[1] = (v900_data + v899_data);
            float v902_data = r1[51];
            float v903_data = ir3[3];
            ir3[3] = (v903_data + v902_data);
            float v905_data = r1[77];
            float v906_data = ir3[5];
            ir3[5] = (v906_data + v905_data);
            float v908_data = r1[103];
            float v909_data = ir3[7];
            ir3[7] = (v909_data + v908_data);
            float v911_data = r1[129];
            float v912_data = ir3[9];
            ir3[9] = (v912_data + v911_data);
            float v914_data = r1[155];
            float v915_data = ir3[11];
            ir3[11] = (v915_data + v914_data);
          }
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v921_n1 = 0; v921_n1 < 1; ++v921_n1) {
              int32_t v923_a = v921_n1 * 2;
              #pragma unroll
              for (int32_t v922_n2 = 0; v922_n2 < 6; ++v922_n2) {
                int32_t v924_a = v922_n2 * 2;
                int32_t v926_a = v923_a + v924_a;
                int32_t v930_a = v923_a + v924_a;
                float v931_data = ir3[v930_a];
                int32_t v935_a = v923_a + v924_a;
                float v940_data = r2[v930_a];
                r3[v930_a] = (v940_data + v931_data);
              }
            }
          }
          if (v10_lead < 3) {
            #pragma unroll
            for (int32_t v947_n1 = 0; v947_n1 < 1; ++v947_n1) {
              int32_t v951_a = 1 + (v947_n1 * 2);
              #pragma unroll
              for (int32_t v948_n2 = 0; v948_n2 < 6; ++v948_n2) {
                int32_t v950_a = v948_n2 * 2;
                int32_t v952_a = v951_a + v950_a;
                float v957_data = ir3[(v951_a + v950_a)];
                int32_t v961_a = v951_a + v950_a;
                float v966_data = r2[(v951_a + v950_a)];
                r3[(v951_a + v950_a)] = (v966_data + v957_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v10_lead >= 20) {
            #pragma unroll
            for (int32_t v976_i1 = 0; v976_i1 < 1; ++v976_i1) {
              int32_t v978_a = v976_i1 * 2;
              int32_t v995_a = v10_lead + ((v976_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v977_i2 = 0; v977_i2 < 6; ++v977_i2) {
                int32_t v979_a = v977_i2 * 2;
                int32_t v981_a = v978_a + v979_a;
                float v986_data = r3[(v978_a + v979_a)];
                glb_m2[(v995_a + (v977_i2 * 832))] = v986_data;
              }
            }
          }
          if (v10_lead < 3) {
            int32_t v1013_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v998_i1 = 0; v998_i1 < 1; ++v998_i1) {
              int32_t v1002_a = 1 + (v998_i1 * 2);
              int32_t v1017_a = v1013_lead + ((v998_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v999_i2 = 0; v999_i2 < 6; ++v999_i2) {
                int32_t v1001_a = v999_i2 * 2;
                int32_t v1003_a = v1002_a + v1001_a;
                float v1008_data = r3[(v1002_a + v1001_a)];
                glb_m2[(v1017_a + (v999_i2 * 832))] = v1008_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

