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
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 64;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __ldcg(&glb_m0[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 * 2);
              r0[v21_a] = v19_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v2_lead >= 20) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 1; ++v26_i1) {
              int32_t v34_a = (v26_i1 + 12) * 64;
              int32_t v36_a = v2_lead + v34_a;
              int32_t v46_a = v2_lead + v34_a;
              int32_t v49_a = v26_i1 * 2;
              #pragma unroll
              for (int32_t v27_i2 = 0; v27_i2 < 6; ++v27_i2) {
                int32_t v35_a = v27_i2 * 832;
                int32_t v37_a = v36_a + v35_a;
                float v48_data = glb_m2[(v46_a + v35_a)];
                int32_t v52_a = v49_a + (v27_i2 * 2);
                r2[v52_a] = v48_data;
              }
            }
          }
          if (v2_lead < 3) {
            int32_t v60_lead = v2_lead + 32_i32;
            int32_t v70_lead = v2_lead + 32_i32;
            #pragma unroll
            for (int32_t v54_i1 = 0; v54_i1 < 1; ++v54_i1) {
              int32_t v62_a = (v54_i1 + 12) * 64;
              int32_t v64_a = v60_lead + v62_a;
              int32_t v74_a = v70_lead + v62_a;
              int32_t v79_a = 1 + (v54_i1 * 2);
              #pragma unroll
              for (int32_t v55_i2 = 0; v55_i2 < 6; ++v55_i2) {
                int32_t v63_a = v55_i2 * 832;
                int32_t v65_a = v64_a + v63_a;
                float v76_data = glb_m2[(v74_a + v63_a)];
                int32_t v80_a = v79_a + (v55_i2 * 2);
                r2[v80_a] = v76_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v84_data = r0[0];
          float v85_data = glb_m1[0];
          float v87_data = ir1[0];
          ir1[0] = (v87_data + (v84_data * v85_data));
          float v90_data = glb_m1[1];
          float v92_data = ir1[26];
          ir1[26] = (v92_data + (v84_data * v90_data));
          float v95_data = glb_m1[2];
          float v97_data = ir1[52];
          ir1[52] = (v97_data + (v84_data * v95_data));
          float v100_data = glb_m1[3];
          float v102_data = ir1[78];
          ir1[78] = (v102_data + (v84_data * v100_data));
          float v105_data = glb_m1[4];
          float v107_data = ir1[104];
          ir1[104] = (v107_data + (v84_data * v105_data));
          float v110_data = glb_m1[5];
          float v112_data = ir1[130];
          ir1[130] = (v112_data + (v84_data * v110_data));
          float v114_data = r0[2];
          float v117_data = ir1[2];
          ir1[2] = (v117_data + (v114_data * v85_data));
          float v122_data = ir1[28];
          ir1[28] = (v122_data + (v114_data * v90_data));
          float v127_data = ir1[54];
          ir1[54] = (v127_data + (v114_data * v95_data));
          float v132_data = ir1[80];
          ir1[80] = (v132_data + (v114_data * v100_data));
          float v137_data = ir1[106];
          ir1[106] = (v137_data + (v114_data * v105_data));
          float v142_data = ir1[132];
          ir1[132] = (v142_data + (v114_data * v110_data));
          float v144_data = r0[4];
          float v147_data = ir1[4];
          ir1[4] = (v147_data + (v144_data * v85_data));
          float v152_data = ir1[30];
          ir1[30] = (v152_data + (v144_data * v90_data));
          float v157_data = ir1[56];
          ir1[56] = (v157_data + (v144_data * v95_data));
          float v162_data = ir1[82];
          ir1[82] = (v162_data + (v144_data * v100_data));
          float v167_data = ir1[108];
          ir1[108] = (v167_data + (v144_data * v105_data));
          float v172_data = ir1[134];
          ir1[134] = (v172_data + (v144_data * v110_data));
          float v174_data = r0[6];
          float v177_data = ir1[6];
          ir1[6] = (v177_data + (v174_data * v85_data));
          float v182_data = ir1[32];
          ir1[32] = (v182_data + (v174_data * v90_data));
          float v187_data = ir1[58];
          ir1[58] = (v187_data + (v174_data * v95_data));
          float v192_data = ir1[84];
          ir1[84] = (v192_data + (v174_data * v100_data));
          float v197_data = ir1[110];
          ir1[110] = (v197_data + (v174_data * v105_data));
          float v202_data = ir1[136];
          ir1[136] = (v202_data + (v174_data * v110_data));
          float v204_data = r0[8];
          float v207_data = ir1[8];
          ir1[8] = (v207_data + (v204_data * v85_data));
          float v212_data = ir1[34];
          ir1[34] = (v212_data + (v204_data * v90_data));
          float v217_data = ir1[60];
          ir1[60] = (v217_data + (v204_data * v95_data));
          float v222_data = ir1[86];
          ir1[86] = (v222_data + (v204_data * v100_data));
          float v227_data = ir1[112];
          ir1[112] = (v227_data + (v204_data * v105_data));
          float v232_data = ir1[138];
          ir1[138] = (v232_data + (v204_data * v110_data));
          float v234_data = r0[10];
          float v237_data = ir1[10];
          ir1[10] = (v237_data + (v234_data * v85_data));
          float v242_data = ir1[36];
          ir1[36] = (v242_data + (v234_data * v90_data));
          float v247_data = ir1[62];
          ir1[62] = (v247_data + (v234_data * v95_data));
          float v252_data = ir1[88];
          ir1[88] = (v252_data + (v234_data * v100_data));
          float v257_data = ir1[114];
          ir1[114] = (v257_data + (v234_data * v105_data));
          float v262_data = ir1[140];
          ir1[140] = (v262_data + (v234_data * v110_data));
          float v264_data = r0[12];
          float v267_data = ir1[12];
          ir1[12] = (v267_data + (v264_data * v85_data));
          float v272_data = ir1[38];
          ir1[38] = (v272_data + (v264_data * v90_data));
          float v277_data = ir1[64];
          ir1[64] = (v277_data + (v264_data * v95_data));
          float v282_data = ir1[90];
          ir1[90] = (v282_data + (v264_data * v100_data));
          float v287_data = ir1[116];
          ir1[116] = (v287_data + (v264_data * v105_data));
          float v292_data = ir1[142];
          ir1[142] = (v292_data + (v264_data * v110_data));
          float v294_data = r0[14];
          float v297_data = ir1[14];
          ir1[14] = (v297_data + (v294_data * v85_data));
          float v302_data = ir1[40];
          ir1[40] = (v302_data + (v294_data * v90_data));
          float v307_data = ir1[66];
          ir1[66] = (v307_data + (v294_data * v95_data));
          float v312_data = ir1[92];
          ir1[92] = (v312_data + (v294_data * v100_data));
          float v317_data = ir1[118];
          ir1[118] = (v317_data + (v294_data * v105_data));
          float v322_data = ir1[144];
          ir1[144] = (v322_data + (v294_data * v110_data));
          float v324_data = r0[16];
          float v327_data = ir1[16];
          ir1[16] = (v327_data + (v324_data * v85_data));
          float v332_data = ir1[42];
          ir1[42] = (v332_data + (v324_data * v90_data));
          float v337_data = ir1[68];
          ir1[68] = (v337_data + (v324_data * v95_data));
          float v342_data = ir1[94];
          ir1[94] = (v342_data + (v324_data * v100_data));
          float v347_data = ir1[120];
          ir1[120] = (v347_data + (v324_data * v105_data));
          float v352_data = ir1[146];
          ir1[146] = (v352_data + (v324_data * v110_data));
          float v354_data = r0[18];
          float v357_data = ir1[18];
          ir1[18] = (v357_data + (v354_data * v85_data));
          float v362_data = ir1[44];
          ir1[44] = (v362_data + (v354_data * v90_data));
          float v367_data = ir1[70];
          ir1[70] = (v367_data + (v354_data * v95_data));
          float v372_data = ir1[96];
          ir1[96] = (v372_data + (v354_data * v100_data));
          float v377_data = ir1[122];
          ir1[122] = (v377_data + (v354_data * v105_data));
          float v382_data = ir1[148];
          ir1[148] = (v382_data + (v354_data * v110_data));
          float v384_data = r0[20];
          float v387_data = ir1[20];
          ir1[20] = (v387_data + (v384_data * v85_data));
          float v392_data = ir1[46];
          ir1[46] = (v392_data + (v384_data * v90_data));
          float v397_data = ir1[72];
          ir1[72] = (v397_data + (v384_data * v95_data));
          float v402_data = ir1[98];
          ir1[98] = (v402_data + (v384_data * v100_data));
          float v407_data = ir1[124];
          ir1[124] = (v407_data + (v384_data * v105_data));
          float v412_data = ir1[150];
          ir1[150] = (v412_data + (v384_data * v110_data));
          float v414_data = r0[22];
          float v417_data = ir1[22];
          ir1[22] = (v417_data + (v414_data * v85_data));
          float v422_data = ir1[48];
          ir1[48] = (v422_data + (v414_data * v90_data));
          float v427_data = ir1[74];
          ir1[74] = (v427_data + (v414_data * v95_data));
          float v432_data = ir1[100];
          ir1[100] = (v432_data + (v414_data * v100_data));
          float v437_data = ir1[126];
          ir1[126] = (v437_data + (v414_data * v105_data));
          float v442_data = ir1[152];
          ir1[152] = (v442_data + (v414_data * v110_data));
          float v444_data = r0[24];
          float v447_data = ir1[24];
          ir1[24] = (v447_data + (v444_data * v85_data));
          float v452_data = ir1[50];
          ir1[50] = (v452_data + (v444_data * v90_data));
          float v457_data = ir1[76];
          ir1[76] = (v457_data + (v444_data * v95_data));
          float v462_data = ir1[102];
          ir1[102] = (v462_data + (v444_data * v100_data));
          float v467_data = ir1[128];
          ir1[128] = (v467_data + (v444_data * v105_data));
          float v472_data = ir1[154];
          ir1[154] = (v472_data + (v444_data * v110_data));
          float v474_data = r0[1];
          float v477_data = ir1[1];
          ir1[1] = (v477_data + (v474_data * v85_data));
          float v482_data = ir1[27];
          ir1[27] = (v482_data + (v474_data * v90_data));
          float v487_data = ir1[53];
          ir1[53] = (v487_data + (v474_data * v95_data));
          float v492_data = ir1[79];
          ir1[79] = (v492_data + (v474_data * v100_data));
          float v497_data = ir1[105];
          ir1[105] = (v497_data + (v474_data * v105_data));
          float v502_data = ir1[131];
          ir1[131] = (v502_data + (v474_data * v110_data));
          float v504_data = r0[3];
          float v507_data = ir1[3];
          ir1[3] = (v507_data + (v504_data * v85_data));
          float v512_data = ir1[29];
          ir1[29] = (v512_data + (v504_data * v90_data));
          float v517_data = ir1[55];
          ir1[55] = (v517_data + (v504_data * v95_data));
          float v522_data = ir1[81];
          ir1[81] = (v522_data + (v504_data * v100_data));
          float v527_data = ir1[107];
          ir1[107] = (v527_data + (v504_data * v105_data));
          float v532_data = ir1[133];
          ir1[133] = (v532_data + (v504_data * v110_data));
          float v534_data = r0[5];
          float v537_data = ir1[5];
          ir1[5] = (v537_data + (v534_data * v85_data));
          float v542_data = ir1[31];
          ir1[31] = (v542_data + (v534_data * v90_data));
          float v547_data = ir1[57];
          ir1[57] = (v547_data + (v534_data * v95_data));
          float v552_data = ir1[83];
          ir1[83] = (v552_data + (v534_data * v100_data));
          float v557_data = ir1[109];
          ir1[109] = (v557_data + (v534_data * v105_data));
          float v562_data = ir1[135];
          ir1[135] = (v562_data + (v534_data * v110_data));
          float v564_data = r0[7];
          float v567_data = ir1[7];
          ir1[7] = (v567_data + (v564_data * v85_data));
          float v572_data = ir1[33];
          ir1[33] = (v572_data + (v564_data * v90_data));
          float v577_data = ir1[59];
          ir1[59] = (v577_data + (v564_data * v95_data));
          float v582_data = ir1[85];
          ir1[85] = (v582_data + (v564_data * v100_data));
          float v587_data = ir1[111];
          ir1[111] = (v587_data + (v564_data * v105_data));
          float v592_data = ir1[137];
          ir1[137] = (v592_data + (v564_data * v110_data));
          float v594_data = r0[9];
          float v597_data = ir1[9];
          ir1[9] = (v597_data + (v594_data * v85_data));
          float v602_data = ir1[35];
          ir1[35] = (v602_data + (v594_data * v90_data));
          float v607_data = ir1[61];
          ir1[61] = (v607_data + (v594_data * v95_data));
          float v612_data = ir1[87];
          ir1[87] = (v612_data + (v594_data * v100_data));
          float v617_data = ir1[113];
          ir1[113] = (v617_data + (v594_data * v105_data));
          float v622_data = ir1[139];
          ir1[139] = (v622_data + (v594_data * v110_data));
          float v624_data = r0[11];
          float v627_data = ir1[11];
          ir1[11] = (v627_data + (v624_data * v85_data));
          float v632_data = ir1[37];
          ir1[37] = (v632_data + (v624_data * v90_data));
          float v637_data = ir1[63];
          ir1[63] = (v637_data + (v624_data * v95_data));
          float v642_data = ir1[89];
          ir1[89] = (v642_data + (v624_data * v100_data));
          float v647_data = ir1[115];
          ir1[115] = (v647_data + (v624_data * v105_data));
          float v652_data = ir1[141];
          ir1[141] = (v652_data + (v624_data * v110_data));
          float v654_data = r0[13];
          float v657_data = ir1[13];
          ir1[13] = (v657_data + (v654_data * v85_data));
          float v662_data = ir1[39];
          ir1[39] = (v662_data + (v654_data * v90_data));
          float v667_data = ir1[65];
          ir1[65] = (v667_data + (v654_data * v95_data));
          float v672_data = ir1[91];
          ir1[91] = (v672_data + (v654_data * v100_data));
          float v677_data = ir1[117];
          ir1[117] = (v677_data + (v654_data * v105_data));
          float v682_data = ir1[143];
          ir1[143] = (v682_data + (v654_data * v110_data));
          float v684_data = r0[15];
          float v687_data = ir1[15];
          ir1[15] = (v687_data + (v684_data * v85_data));
          float v692_data = ir1[41];
          ir1[41] = (v692_data + (v684_data * v90_data));
          float v697_data = ir1[67];
          ir1[67] = (v697_data + (v684_data * v95_data));
          float v702_data = ir1[93];
          ir1[93] = (v702_data + (v684_data * v100_data));
          float v707_data = ir1[119];
          ir1[119] = (v707_data + (v684_data * v105_data));
          float v712_data = ir1[145];
          ir1[145] = (v712_data + (v684_data * v110_data));
          float v714_data = r0[17];
          float v717_data = ir1[17];
          ir1[17] = (v717_data + (v714_data * v85_data));
          float v722_data = ir1[43];
          ir1[43] = (v722_data + (v714_data * v90_data));
          float v727_data = ir1[69];
          ir1[69] = (v727_data + (v714_data * v95_data));
          float v732_data = ir1[95];
          ir1[95] = (v732_data + (v714_data * v100_data));
          float v737_data = ir1[121];
          ir1[121] = (v737_data + (v714_data * v105_data));
          float v742_data = ir1[147];
          ir1[147] = (v742_data + (v714_data * v110_data));
          float v744_data = r0[19];
          float v747_data = ir1[19];
          ir1[19] = (v747_data + (v744_data * v85_data));
          float v752_data = ir1[45];
          ir1[45] = (v752_data + (v744_data * v90_data));
          float v757_data = ir1[71];
          ir1[71] = (v757_data + (v744_data * v95_data));
          float v762_data = ir1[97];
          ir1[97] = (v762_data + (v744_data * v100_data));
          float v767_data = ir1[123];
          ir1[123] = (v767_data + (v744_data * v105_data));
          float v772_data = ir1[149];
          ir1[149] = (v772_data + (v744_data * v110_data));
          float v774_data = r0[21];
          float v777_data = ir1[21];
          ir1[21] = (v777_data + (v774_data * v85_data));
          float v782_data = ir1[47];
          ir1[47] = (v782_data + (v774_data * v90_data));
          float v787_data = ir1[73];
          ir1[73] = (v787_data + (v774_data * v95_data));
          float v792_data = ir1[99];
          ir1[99] = (v792_data + (v774_data * v100_data));
          float v797_data = ir1[125];
          ir1[125] = (v797_data + (v774_data * v105_data));
          float v802_data = ir1[151];
          ir1[151] = (v802_data + (v774_data * v110_data));
          float v804_data = r0[23];
          float v807_data = ir1[23];
          ir1[23] = (v807_data + (v804_data * v85_data));
          float v812_data = ir1[49];
          ir1[49] = (v812_data + (v804_data * v90_data));
          float v817_data = ir1[75];
          ir1[75] = (v817_data + (v804_data * v95_data));
          float v822_data = ir1[101];
          ir1[101] = (v822_data + (v804_data * v100_data));
          float v827_data = ir1[127];
          ir1[127] = (v827_data + (v804_data * v105_data));
          float v832_data = ir1[153];
          ir1[153] = (v832_data + (v804_data * v110_data));
          float v834_data = r0[25];
          float v837_data = ir1[25];
          ir1[25] = (v837_data + (v834_data * v85_data));
          float v842_data = ir1[51];
          ir1[51] = (v842_data + (v834_data * v90_data));
          float v847_data = ir1[77];
          ir1[77] = (v847_data + (v834_data * v95_data));
          float v852_data = ir1[103];
          ir1[103] = (v852_data + (v834_data * v100_data));
          float v857_data = ir1[129];
          ir1[129] = (v857_data + (v834_data * v105_data));
          float v862_data = ir1[155];
          ir1[155] = (v862_data + (v834_data * v110_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          {
            // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
            // [(20, 35), (0, 1), (0, 6)] []
            float ir3[12]{};
            if (v2_lead >= 20) {
              float v868_data = r1[24];
              float v869_data = ir3[0];
              ir3[0] = (v869_data + v868_data);
              float v871_data = r1[50];
              float v872_data = ir3[2];
              ir3[2] = (v872_data + v871_data);
              float v874_data = r1[76];
              float v875_data = ir3[4];
              ir3[4] = (v875_data + v874_data);
              float v877_data = r1[102];
              float v878_data = ir3[6];
              ir3[6] = (v878_data + v877_data);
              float v880_data = r1[128];
              float v881_data = ir3[8];
              ir3[8] = (v881_data + v880_data);
              float v883_data = r1[154];
              float v884_data = ir3[10];
              ir3[10] = (v884_data + v883_data);
            }
            if (v2_lead < 3) {
              float v887_data = r1[25];
              float v888_data = ir3[1];
              ir3[1] = (v888_data + v887_data);
              float v890_data = r1[51];
              float v891_data = ir3[3];
              ir3[3] = (v891_data + v890_data);
              float v893_data = r1[77];
              float v894_data = ir3[5];
              ir3[5] = (v894_data + v893_data);
              float v896_data = r1[103];
              float v897_data = ir3[7];
              ir3[7] = (v897_data + v896_data);
              float v899_data = r1[129];
              float v900_data = ir3[9];
              ir3[9] = (v900_data + v899_data);
              float v902_data = r1[155];
              float v903_data = ir3[11];
              ir3[11] = (v903_data + v902_data);
            }
            if (v2_lead >= 20) {
              #pragma unroll
              for (int32_t v909_n1 = 0; v909_n1 < 1; ++v909_n1) {
                int32_t v911_a = v909_n1 * 2;
                #pragma unroll
                for (int32_t v910_n2 = 0; v910_n2 < 6; ++v910_n2) {
                  int32_t v912_a = v910_n2 * 2;
                  int32_t v914_a = v911_a + v912_a;
                  int32_t v918_a = v911_a + v912_a;
                  float v919_data = ir3[v918_a];
                  int32_t v923_a = v911_a + v912_a;
                  float v928_data = r2[v918_a];
                  int32_t v933_a = v911_a + v912_a;
                  r3[v918_a] = (v928_data + v919_data);
                }
              }
            }
            if (v2_lead < 3) {
              #pragma unroll
              for (int32_t v939_n1 = 0; v939_n1 < 1; ++v939_n1) {
                int32_t v943_a = 1 + (v939_n1 * 2);
                #pragma unroll
                for (int32_t v940_n2 = 0; v940_n2 < 6; ++v940_n2) {
                  int32_t v942_a = v940_n2 * 2;
                  int32_t v944_a = v943_a + v942_a;
                  float v949_data = ir3[(v943_a + v942_a)];
                  int32_t v953_a = v943_a + v942_a;
                  float v958_data = r2[(v943_a + v942_a)];
                  int32_t v963_a = v943_a + v942_a;
                  r3[(v943_a + v942_a)] = (v958_data + v949_data);
                }
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v2_lead >= 20) {
            #pragma unroll
            for (int32_t v972_i1 = 0; v972_i1 < 1; ++v972_i1) {
              int32_t v974_a = v972_i1 * 2;
              int32_t v991_a = v2_lead + ((v972_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v973_i2 = 0; v973_i2 < 6; ++v973_i2) {
                int32_t v975_a = v973_i2 * 2;
                int32_t v977_a = v974_a + v975_a;
                float v982_data = r3[(v974_a + v975_a)];
                int32_t v992_a = v991_a + (v973_i2 * 832);
                glb_m2[v992_a] = v982_data;
              }
            }
          }
          if (v2_lead < 3) {
            int32_t v1009_lead = v2_lead + 32_i32;
            #pragma unroll
            for (int32_t v994_i1 = 0; v994_i1 < 1; ++v994_i1) {
              int32_t v998_a = 1 + (v994_i1 * 2);
              int32_t v1013_a = v1009_lead + ((v994_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v995_i2 = 0; v995_i2 < 6; ++v995_i2) {
                int32_t v997_a = v995_i2 * 2;
                int32_t v999_a = v998_a + v997_a;
                float v1004_data = r3[(v998_a + v997_a)];
                int32_t v1014_a = v1013_a + (v995_i2 * 832);
                glb_m2[v1014_a] = v1004_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

