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
          int32_t v6_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 2; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 32;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 13; ++v8_i1) {
              int32_t v14_a = v8_i1 * 64;
              int32_t v15_a = v13_lead + v14_a;
              float v23_data = __ldcg(&glb_m0[(v20_lead + v14_a)]);
              int32_t v25_a = v7_i0 + (v8_i1 * 2);
              r0[v25_a] = v23_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v6_lead >= 20) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 1; ++v31_i1) {
              int32_t v39_a = (v31_i1 + 12) * 64;
              int32_t v41_a = v6_lead + v39_a;
              int32_t v51_a = v6_lead + v39_a;
              int32_t v54_a = v31_i1 * 2;
              #pragma unroll
              for (int32_t v32_i2 = 0; v32_i2 < 6; ++v32_i2) {
                int32_t v40_a = v32_i2 * 832;
                int32_t v42_a = v41_a + v40_a;
                float v53_data = glb_m2[(v51_a + v40_a)];
                int32_t v57_a = v54_a + (v32_i2 * 2);
                r2[v57_a] = v53_data;
              }
            }
          }
          if (v6_lead < 3) {
            int32_t v65_lead = v6_lead + 32_i32;
            int32_t v75_lead = v6_lead + 32_i32;
            #pragma unroll
            for (int32_t v59_i1 = 0; v59_i1 < 1; ++v59_i1) {
              int32_t v67_a = (v59_i1 + 12) * 64;
              int32_t v69_a = v65_lead + v67_a;
              int32_t v79_a = v75_lead + v67_a;
              int32_t v84_a = 1 + (v59_i1 * 2);
              #pragma unroll
              for (int32_t v60_i2 = 0; v60_i2 < 6; ++v60_i2) {
                int32_t v68_a = v60_i2 * 832;
                int32_t v70_a = v69_a + v68_a;
                float v81_data = glb_m2[(v79_a + v68_a)];
                int32_t v85_a = v84_a + (v60_i2 * 2);
                r2[v85_a] = v81_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v90_data = r0[0];
          float v91_data = glb_m1[0];
          float v93_data = ir1[0];
          ir1[0] = (v93_data + (v90_data * v91_data));
          float v96_data = glb_m1[1];
          float v98_data = ir1[26];
          ir1[26] = (v98_data + (v90_data * v96_data));
          float v101_data = glb_m1[2];
          float v103_data = ir1[52];
          ir1[52] = (v103_data + (v90_data * v101_data));
          float v106_data = glb_m1[3];
          float v108_data = ir1[78];
          ir1[78] = (v108_data + (v90_data * v106_data));
          float v111_data = glb_m1[4];
          float v113_data = ir1[104];
          ir1[104] = (v113_data + (v90_data * v111_data));
          float v116_data = glb_m1[5];
          float v118_data = ir1[130];
          ir1[130] = (v118_data + (v90_data * v116_data));
          float v120_data = r0[2];
          float v123_data = ir1[2];
          ir1[2] = (v123_data + (v120_data * v91_data));
          float v128_data = ir1[28];
          ir1[28] = (v128_data + (v120_data * v96_data));
          float v133_data = ir1[54];
          ir1[54] = (v133_data + (v120_data * v101_data));
          float v138_data = ir1[80];
          ir1[80] = (v138_data + (v120_data * v106_data));
          float v143_data = ir1[106];
          ir1[106] = (v143_data + (v120_data * v111_data));
          float v148_data = ir1[132];
          ir1[132] = (v148_data + (v120_data * v116_data));
          float v150_data = r0[4];
          float v153_data = ir1[4];
          ir1[4] = (v153_data + (v150_data * v91_data));
          float v158_data = ir1[30];
          ir1[30] = (v158_data + (v150_data * v96_data));
          float v163_data = ir1[56];
          ir1[56] = (v163_data + (v150_data * v101_data));
          float v168_data = ir1[82];
          ir1[82] = (v168_data + (v150_data * v106_data));
          float v173_data = ir1[108];
          ir1[108] = (v173_data + (v150_data * v111_data));
          float v178_data = ir1[134];
          ir1[134] = (v178_data + (v150_data * v116_data));
          float v180_data = r0[6];
          float v183_data = ir1[6];
          ir1[6] = (v183_data + (v180_data * v91_data));
          float v188_data = ir1[32];
          ir1[32] = (v188_data + (v180_data * v96_data));
          float v193_data = ir1[58];
          ir1[58] = (v193_data + (v180_data * v101_data));
          float v198_data = ir1[84];
          ir1[84] = (v198_data + (v180_data * v106_data));
          float v203_data = ir1[110];
          ir1[110] = (v203_data + (v180_data * v111_data));
          float v208_data = ir1[136];
          ir1[136] = (v208_data + (v180_data * v116_data));
          float v210_data = r0[8];
          float v213_data = ir1[8];
          ir1[8] = (v213_data + (v210_data * v91_data));
          float v218_data = ir1[34];
          ir1[34] = (v218_data + (v210_data * v96_data));
          float v223_data = ir1[60];
          ir1[60] = (v223_data + (v210_data * v101_data));
          float v228_data = ir1[86];
          ir1[86] = (v228_data + (v210_data * v106_data));
          float v233_data = ir1[112];
          ir1[112] = (v233_data + (v210_data * v111_data));
          float v238_data = ir1[138];
          ir1[138] = (v238_data + (v210_data * v116_data));
          float v240_data = r0[10];
          float v243_data = ir1[10];
          ir1[10] = (v243_data + (v240_data * v91_data));
          float v248_data = ir1[36];
          ir1[36] = (v248_data + (v240_data * v96_data));
          float v253_data = ir1[62];
          ir1[62] = (v253_data + (v240_data * v101_data));
          float v258_data = ir1[88];
          ir1[88] = (v258_data + (v240_data * v106_data));
          float v263_data = ir1[114];
          ir1[114] = (v263_data + (v240_data * v111_data));
          float v268_data = ir1[140];
          ir1[140] = (v268_data + (v240_data * v116_data));
          float v270_data = r0[12];
          float v273_data = ir1[12];
          ir1[12] = (v273_data + (v270_data * v91_data));
          float v278_data = ir1[38];
          ir1[38] = (v278_data + (v270_data * v96_data));
          float v283_data = ir1[64];
          ir1[64] = (v283_data + (v270_data * v101_data));
          float v288_data = ir1[90];
          ir1[90] = (v288_data + (v270_data * v106_data));
          float v293_data = ir1[116];
          ir1[116] = (v293_data + (v270_data * v111_data));
          float v298_data = ir1[142];
          ir1[142] = (v298_data + (v270_data * v116_data));
          float v300_data = r0[14];
          float v303_data = ir1[14];
          ir1[14] = (v303_data + (v300_data * v91_data));
          float v308_data = ir1[40];
          ir1[40] = (v308_data + (v300_data * v96_data));
          float v313_data = ir1[66];
          ir1[66] = (v313_data + (v300_data * v101_data));
          float v318_data = ir1[92];
          ir1[92] = (v318_data + (v300_data * v106_data));
          float v323_data = ir1[118];
          ir1[118] = (v323_data + (v300_data * v111_data));
          float v328_data = ir1[144];
          ir1[144] = (v328_data + (v300_data * v116_data));
          float v330_data = r0[16];
          float v333_data = ir1[16];
          ir1[16] = (v333_data + (v330_data * v91_data));
          float v338_data = ir1[42];
          ir1[42] = (v338_data + (v330_data * v96_data));
          float v343_data = ir1[68];
          ir1[68] = (v343_data + (v330_data * v101_data));
          float v348_data = ir1[94];
          ir1[94] = (v348_data + (v330_data * v106_data));
          float v353_data = ir1[120];
          ir1[120] = (v353_data + (v330_data * v111_data));
          float v358_data = ir1[146];
          ir1[146] = (v358_data + (v330_data * v116_data));
          float v360_data = r0[18];
          float v363_data = ir1[18];
          ir1[18] = (v363_data + (v360_data * v91_data));
          float v368_data = ir1[44];
          ir1[44] = (v368_data + (v360_data * v96_data));
          float v373_data = ir1[70];
          ir1[70] = (v373_data + (v360_data * v101_data));
          float v378_data = ir1[96];
          ir1[96] = (v378_data + (v360_data * v106_data));
          float v383_data = ir1[122];
          ir1[122] = (v383_data + (v360_data * v111_data));
          float v388_data = ir1[148];
          ir1[148] = (v388_data + (v360_data * v116_data));
          float v390_data = r0[20];
          float v393_data = ir1[20];
          ir1[20] = (v393_data + (v390_data * v91_data));
          float v398_data = ir1[46];
          ir1[46] = (v398_data + (v390_data * v96_data));
          float v403_data = ir1[72];
          ir1[72] = (v403_data + (v390_data * v101_data));
          float v408_data = ir1[98];
          ir1[98] = (v408_data + (v390_data * v106_data));
          float v413_data = ir1[124];
          ir1[124] = (v413_data + (v390_data * v111_data));
          float v418_data = ir1[150];
          ir1[150] = (v418_data + (v390_data * v116_data));
          float v420_data = r0[22];
          float v423_data = ir1[22];
          ir1[22] = (v423_data + (v420_data * v91_data));
          float v428_data = ir1[48];
          ir1[48] = (v428_data + (v420_data * v96_data));
          float v433_data = ir1[74];
          ir1[74] = (v433_data + (v420_data * v101_data));
          float v438_data = ir1[100];
          ir1[100] = (v438_data + (v420_data * v106_data));
          float v443_data = ir1[126];
          ir1[126] = (v443_data + (v420_data * v111_data));
          float v448_data = ir1[152];
          ir1[152] = (v448_data + (v420_data * v116_data));
          float v450_data = r0[24];
          float v453_data = ir1[24];
          ir1[24] = (v453_data + (v450_data * v91_data));
          float v458_data = ir1[50];
          ir1[50] = (v458_data + (v450_data * v96_data));
          float v463_data = ir1[76];
          ir1[76] = (v463_data + (v450_data * v101_data));
          float v468_data = ir1[102];
          ir1[102] = (v468_data + (v450_data * v106_data));
          float v473_data = ir1[128];
          ir1[128] = (v473_data + (v450_data * v111_data));
          float v478_data = ir1[154];
          ir1[154] = (v478_data + (v450_data * v116_data));
          float v480_data = r0[1];
          float v483_data = ir1[1];
          ir1[1] = (v483_data + (v480_data * v91_data));
          float v488_data = ir1[27];
          ir1[27] = (v488_data + (v480_data * v96_data));
          float v493_data = ir1[53];
          ir1[53] = (v493_data + (v480_data * v101_data));
          float v498_data = ir1[79];
          ir1[79] = (v498_data + (v480_data * v106_data));
          float v503_data = ir1[105];
          ir1[105] = (v503_data + (v480_data * v111_data));
          float v508_data = ir1[131];
          ir1[131] = (v508_data + (v480_data * v116_data));
          float v510_data = r0[3];
          float v513_data = ir1[3];
          ir1[3] = (v513_data + (v510_data * v91_data));
          float v518_data = ir1[29];
          ir1[29] = (v518_data + (v510_data * v96_data));
          float v523_data = ir1[55];
          ir1[55] = (v523_data + (v510_data * v101_data));
          float v528_data = ir1[81];
          ir1[81] = (v528_data + (v510_data * v106_data));
          float v533_data = ir1[107];
          ir1[107] = (v533_data + (v510_data * v111_data));
          float v538_data = ir1[133];
          ir1[133] = (v538_data + (v510_data * v116_data));
          float v540_data = r0[5];
          float v543_data = ir1[5];
          ir1[5] = (v543_data + (v540_data * v91_data));
          float v548_data = ir1[31];
          ir1[31] = (v548_data + (v540_data * v96_data));
          float v553_data = ir1[57];
          ir1[57] = (v553_data + (v540_data * v101_data));
          float v558_data = ir1[83];
          ir1[83] = (v558_data + (v540_data * v106_data));
          float v563_data = ir1[109];
          ir1[109] = (v563_data + (v540_data * v111_data));
          float v568_data = ir1[135];
          ir1[135] = (v568_data + (v540_data * v116_data));
          float v570_data = r0[7];
          float v573_data = ir1[7];
          ir1[7] = (v573_data + (v570_data * v91_data));
          float v578_data = ir1[33];
          ir1[33] = (v578_data + (v570_data * v96_data));
          float v583_data = ir1[59];
          ir1[59] = (v583_data + (v570_data * v101_data));
          float v588_data = ir1[85];
          ir1[85] = (v588_data + (v570_data * v106_data));
          float v593_data = ir1[111];
          ir1[111] = (v593_data + (v570_data * v111_data));
          float v598_data = ir1[137];
          ir1[137] = (v598_data + (v570_data * v116_data));
          float v600_data = r0[9];
          float v603_data = ir1[9];
          ir1[9] = (v603_data + (v600_data * v91_data));
          float v608_data = ir1[35];
          ir1[35] = (v608_data + (v600_data * v96_data));
          float v613_data = ir1[61];
          ir1[61] = (v613_data + (v600_data * v101_data));
          float v618_data = ir1[87];
          ir1[87] = (v618_data + (v600_data * v106_data));
          float v623_data = ir1[113];
          ir1[113] = (v623_data + (v600_data * v111_data));
          float v628_data = ir1[139];
          ir1[139] = (v628_data + (v600_data * v116_data));
          float v630_data = r0[11];
          float v633_data = ir1[11];
          ir1[11] = (v633_data + (v630_data * v91_data));
          float v638_data = ir1[37];
          ir1[37] = (v638_data + (v630_data * v96_data));
          float v643_data = ir1[63];
          ir1[63] = (v643_data + (v630_data * v101_data));
          float v648_data = ir1[89];
          ir1[89] = (v648_data + (v630_data * v106_data));
          float v653_data = ir1[115];
          ir1[115] = (v653_data + (v630_data * v111_data));
          float v658_data = ir1[141];
          ir1[141] = (v658_data + (v630_data * v116_data));
          float v660_data = r0[13];
          float v663_data = ir1[13];
          ir1[13] = (v663_data + (v660_data * v91_data));
          float v668_data = ir1[39];
          ir1[39] = (v668_data + (v660_data * v96_data));
          float v673_data = ir1[65];
          ir1[65] = (v673_data + (v660_data * v101_data));
          float v678_data = ir1[91];
          ir1[91] = (v678_data + (v660_data * v106_data));
          float v683_data = ir1[117];
          ir1[117] = (v683_data + (v660_data * v111_data));
          float v688_data = ir1[143];
          ir1[143] = (v688_data + (v660_data * v116_data));
          float v690_data = r0[15];
          float v693_data = ir1[15];
          ir1[15] = (v693_data + (v690_data * v91_data));
          float v698_data = ir1[41];
          ir1[41] = (v698_data + (v690_data * v96_data));
          float v703_data = ir1[67];
          ir1[67] = (v703_data + (v690_data * v101_data));
          float v708_data = ir1[93];
          ir1[93] = (v708_data + (v690_data * v106_data));
          float v713_data = ir1[119];
          ir1[119] = (v713_data + (v690_data * v111_data));
          float v718_data = ir1[145];
          ir1[145] = (v718_data + (v690_data * v116_data));
          float v720_data = r0[17];
          float v723_data = ir1[17];
          ir1[17] = (v723_data + (v720_data * v91_data));
          float v728_data = ir1[43];
          ir1[43] = (v728_data + (v720_data * v96_data));
          float v733_data = ir1[69];
          ir1[69] = (v733_data + (v720_data * v101_data));
          float v738_data = ir1[95];
          ir1[95] = (v738_data + (v720_data * v106_data));
          float v743_data = ir1[121];
          ir1[121] = (v743_data + (v720_data * v111_data));
          float v748_data = ir1[147];
          ir1[147] = (v748_data + (v720_data * v116_data));
          float v750_data = r0[19];
          float v753_data = ir1[19];
          ir1[19] = (v753_data + (v750_data * v91_data));
          float v758_data = ir1[45];
          ir1[45] = (v758_data + (v750_data * v96_data));
          float v763_data = ir1[71];
          ir1[71] = (v763_data + (v750_data * v101_data));
          float v768_data = ir1[97];
          ir1[97] = (v768_data + (v750_data * v106_data));
          float v773_data = ir1[123];
          ir1[123] = (v773_data + (v750_data * v111_data));
          float v778_data = ir1[149];
          ir1[149] = (v778_data + (v750_data * v116_data));
          float v780_data = r0[21];
          float v783_data = ir1[21];
          ir1[21] = (v783_data + (v780_data * v91_data));
          float v788_data = ir1[47];
          ir1[47] = (v788_data + (v780_data * v96_data));
          float v793_data = ir1[73];
          ir1[73] = (v793_data + (v780_data * v101_data));
          float v798_data = ir1[99];
          ir1[99] = (v798_data + (v780_data * v106_data));
          float v803_data = ir1[125];
          ir1[125] = (v803_data + (v780_data * v111_data));
          float v808_data = ir1[151];
          ir1[151] = (v808_data + (v780_data * v116_data));
          float v810_data = r0[23];
          float v813_data = ir1[23];
          ir1[23] = (v813_data + (v810_data * v91_data));
          float v818_data = ir1[49];
          ir1[49] = (v818_data + (v810_data * v96_data));
          float v823_data = ir1[75];
          ir1[75] = (v823_data + (v810_data * v101_data));
          float v828_data = ir1[101];
          ir1[101] = (v828_data + (v810_data * v106_data));
          float v833_data = ir1[127];
          ir1[127] = (v833_data + (v810_data * v111_data));
          float v838_data = ir1[153];
          ir1[153] = (v838_data + (v810_data * v116_data));
          float v840_data = r0[25];
          float v843_data = ir1[25];
          ir1[25] = (v843_data + (v840_data * v91_data));
          float v848_data = ir1[51];
          ir1[51] = (v848_data + (v840_data * v96_data));
          float v853_data = ir1[77];
          ir1[77] = (v853_data + (v840_data * v101_data));
          float v858_data = ir1[103];
          ir1[103] = (v858_data + (v840_data * v106_data));
          float v863_data = ir1[129];
          ir1[129] = (v863_data + (v840_data * v111_data));
          float v868_data = ir1[155];
          ir1[155] = (v868_data + (v840_data * v116_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
          // [(20, 35), (0, 1), (0, 6)] []
          float ir3[12]{};
          if (v6_lead >= 20) {
            float v876_data = r1[24];
            float v877_data = ir3[0];
            ir3[0] = (v877_data + v876_data);
            float v879_data = r1[50];
            float v880_data = ir3[2];
            ir3[2] = (v880_data + v879_data);
            float v882_data = r1[76];
            float v883_data = ir3[4];
            ir3[4] = (v883_data + v882_data);
            float v885_data = r1[102];
            float v886_data = ir3[6];
            ir3[6] = (v886_data + v885_data);
            float v888_data = r1[128];
            float v889_data = ir3[8];
            ir3[8] = (v889_data + v888_data);
            float v891_data = r1[154];
            float v892_data = ir3[10];
            ir3[10] = (v892_data + v891_data);
          }
          if (v6_lead < 3) {
            float v895_data = r1[25];
            float v896_data = ir3[1];
            ir3[1] = (v896_data + v895_data);
            float v898_data = r1[51];
            float v899_data = ir3[3];
            ir3[3] = (v899_data + v898_data);
            float v901_data = r1[77];
            float v902_data = ir3[5];
            ir3[5] = (v902_data + v901_data);
            float v904_data = r1[103];
            float v905_data = ir3[7];
            ir3[7] = (v905_data + v904_data);
            float v907_data = r1[129];
            float v908_data = ir3[9];
            ir3[9] = (v908_data + v907_data);
            float v910_data = r1[155];
            float v911_data = ir3[11];
            ir3[11] = (v911_data + v910_data);
          }
          if (v6_lead >= 20) {
            #pragma unroll
            for (int32_t v917_n1 = 0; v917_n1 < 1; ++v917_n1) {
              int32_t v919_a = v917_n1 * 2;
              #pragma unroll
              for (int32_t v918_n2 = 0; v918_n2 < 6; ++v918_n2) {
                int32_t v920_a = v918_n2 * 2;
                int32_t v922_a = v919_a + v920_a;
                int32_t v926_a = v919_a + v920_a;
                float v927_data = ir3[v926_a];
                int32_t v931_a = v919_a + v920_a;
                float v936_data = r2[v926_a];
                int32_t v941_a = v919_a + v920_a;
                r3[v926_a] = (v936_data + v927_data);
              }
            }
          }
          if (v6_lead < 3) {
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
                int32_t v971_a = v951_a + v950_a;
                r3[(v951_a + v950_a)] = (v966_data + v957_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v6_lead >= 20) {
            #pragma unroll
            for (int32_t v980_i1 = 0; v980_i1 < 1; ++v980_i1) {
              int32_t v982_a = v980_i1 * 2;
              int32_t v999_a = v6_lead + ((v980_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v981_i2 = 0; v981_i2 < 6; ++v981_i2) {
                int32_t v983_a = v981_i2 * 2;
                int32_t v985_a = v982_a + v983_a;
                float v990_data = r3[(v982_a + v983_a)];
                int32_t v1000_a = v999_a + (v981_i2 * 832);
                glb_m2[v1000_a] = v990_data;
              }
            }
          }
          if (v6_lead < 3) {
            int32_t v1017_lead = v6_lead + 32_i32;
            #pragma unroll
            for (int32_t v1002_i1 = 0; v1002_i1 < 1; ++v1002_i1) {
              int32_t v1006_a = 1 + (v1002_i1 * 2);
              int32_t v1021_a = v1017_lead + ((v1002_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v1003_i2 = 0; v1003_i2 < 6; ++v1003_i2) {
                int32_t v1005_a = v1003_i2 * 2;
                int32_t v1007_a = v1006_a + v1005_a;
                float v1012_data = r3[(v1006_a + v1005_a)];
                int32_t v1022_a = v1021_a + (v1003_i2 * 832);
                glb_m2[v1022_a] = v1012_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

