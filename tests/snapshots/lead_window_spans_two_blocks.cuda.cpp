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
          float v90_data = r0[0];
          float v91_data = glb_m1[0];
          float v93_data = r1[0];
          r1[0] = (v93_data + (v90_data * v91_data));
          float v96_data = glb_m1[1];
          float v98_data = r1[26];
          r1[26] = (v98_data + (v90_data * v96_data));
          float v101_data = glb_m1[2];
          float v103_data = r1[52];
          r1[52] = (v103_data + (v90_data * v101_data));
          float v106_data = glb_m1[3];
          float v108_data = r1[78];
          r1[78] = (v108_data + (v90_data * v106_data));
          float v111_data = glb_m1[4];
          float v113_data = r1[104];
          r1[104] = (v113_data + (v90_data * v111_data));
          float v116_data = glb_m1[5];
          float v118_data = r1[130];
          r1[130] = (v118_data + (v90_data * v116_data));
          float v120_data = r0[2];
          float v123_data = r1[2];
          r1[2] = (v123_data + (v120_data * v91_data));
          float v128_data = r1[28];
          r1[28] = (v128_data + (v120_data * v96_data));
          float v133_data = r1[54];
          r1[54] = (v133_data + (v120_data * v101_data));
          float v138_data = r1[80];
          r1[80] = (v138_data + (v120_data * v106_data));
          float v143_data = r1[106];
          r1[106] = (v143_data + (v120_data * v111_data));
          float v148_data = r1[132];
          r1[132] = (v148_data + (v120_data * v116_data));
          float v150_data = r0[4];
          float v153_data = r1[4];
          r1[4] = (v153_data + (v150_data * v91_data));
          float v158_data = r1[30];
          r1[30] = (v158_data + (v150_data * v96_data));
          float v163_data = r1[56];
          r1[56] = (v163_data + (v150_data * v101_data));
          float v168_data = r1[82];
          r1[82] = (v168_data + (v150_data * v106_data));
          float v173_data = r1[108];
          r1[108] = (v173_data + (v150_data * v111_data));
          float v178_data = r1[134];
          r1[134] = (v178_data + (v150_data * v116_data));
          float v180_data = r0[6];
          float v183_data = r1[6];
          r1[6] = (v183_data + (v180_data * v91_data));
          float v188_data = r1[32];
          r1[32] = (v188_data + (v180_data * v96_data));
          float v193_data = r1[58];
          r1[58] = (v193_data + (v180_data * v101_data));
          float v198_data = r1[84];
          r1[84] = (v198_data + (v180_data * v106_data));
          float v203_data = r1[110];
          r1[110] = (v203_data + (v180_data * v111_data));
          float v208_data = r1[136];
          r1[136] = (v208_data + (v180_data * v116_data));
          float v210_data = r0[8];
          float v213_data = r1[8];
          r1[8] = (v213_data + (v210_data * v91_data));
          float v218_data = r1[34];
          r1[34] = (v218_data + (v210_data * v96_data));
          float v223_data = r1[60];
          r1[60] = (v223_data + (v210_data * v101_data));
          float v228_data = r1[86];
          r1[86] = (v228_data + (v210_data * v106_data));
          float v233_data = r1[112];
          r1[112] = (v233_data + (v210_data * v111_data));
          float v238_data = r1[138];
          r1[138] = (v238_data + (v210_data * v116_data));
          float v240_data = r0[10];
          float v243_data = r1[10];
          r1[10] = (v243_data + (v240_data * v91_data));
          float v248_data = r1[36];
          r1[36] = (v248_data + (v240_data * v96_data));
          float v253_data = r1[62];
          r1[62] = (v253_data + (v240_data * v101_data));
          float v258_data = r1[88];
          r1[88] = (v258_data + (v240_data * v106_data));
          float v263_data = r1[114];
          r1[114] = (v263_data + (v240_data * v111_data));
          float v268_data = r1[140];
          r1[140] = (v268_data + (v240_data * v116_data));
          float v270_data = r0[12];
          float v273_data = r1[12];
          r1[12] = (v273_data + (v270_data * v91_data));
          float v278_data = r1[38];
          r1[38] = (v278_data + (v270_data * v96_data));
          float v283_data = r1[64];
          r1[64] = (v283_data + (v270_data * v101_data));
          float v288_data = r1[90];
          r1[90] = (v288_data + (v270_data * v106_data));
          float v293_data = r1[116];
          r1[116] = (v293_data + (v270_data * v111_data));
          float v298_data = r1[142];
          r1[142] = (v298_data + (v270_data * v116_data));
          float v300_data = r0[14];
          float v303_data = r1[14];
          r1[14] = (v303_data + (v300_data * v91_data));
          float v308_data = r1[40];
          r1[40] = (v308_data + (v300_data * v96_data));
          float v313_data = r1[66];
          r1[66] = (v313_data + (v300_data * v101_data));
          float v318_data = r1[92];
          r1[92] = (v318_data + (v300_data * v106_data));
          float v323_data = r1[118];
          r1[118] = (v323_data + (v300_data * v111_data));
          float v328_data = r1[144];
          r1[144] = (v328_data + (v300_data * v116_data));
          float v330_data = r0[16];
          float v333_data = r1[16];
          r1[16] = (v333_data + (v330_data * v91_data));
          float v338_data = r1[42];
          r1[42] = (v338_data + (v330_data * v96_data));
          float v343_data = r1[68];
          r1[68] = (v343_data + (v330_data * v101_data));
          float v348_data = r1[94];
          r1[94] = (v348_data + (v330_data * v106_data));
          float v353_data = r1[120];
          r1[120] = (v353_data + (v330_data * v111_data));
          float v358_data = r1[146];
          r1[146] = (v358_data + (v330_data * v116_data));
          float v360_data = r0[18];
          float v363_data = r1[18];
          r1[18] = (v363_data + (v360_data * v91_data));
          float v368_data = r1[44];
          r1[44] = (v368_data + (v360_data * v96_data));
          float v373_data = r1[70];
          r1[70] = (v373_data + (v360_data * v101_data));
          float v378_data = r1[96];
          r1[96] = (v378_data + (v360_data * v106_data));
          float v383_data = r1[122];
          r1[122] = (v383_data + (v360_data * v111_data));
          float v388_data = r1[148];
          r1[148] = (v388_data + (v360_data * v116_data));
          float v390_data = r0[20];
          float v393_data = r1[20];
          r1[20] = (v393_data + (v390_data * v91_data));
          float v398_data = r1[46];
          r1[46] = (v398_data + (v390_data * v96_data));
          float v403_data = r1[72];
          r1[72] = (v403_data + (v390_data * v101_data));
          float v408_data = r1[98];
          r1[98] = (v408_data + (v390_data * v106_data));
          float v413_data = r1[124];
          r1[124] = (v413_data + (v390_data * v111_data));
          float v418_data = r1[150];
          r1[150] = (v418_data + (v390_data * v116_data));
          float v420_data = r0[22];
          float v423_data = r1[22];
          r1[22] = (v423_data + (v420_data * v91_data));
          float v428_data = r1[48];
          r1[48] = (v428_data + (v420_data * v96_data));
          float v433_data = r1[74];
          r1[74] = (v433_data + (v420_data * v101_data));
          float v438_data = r1[100];
          r1[100] = (v438_data + (v420_data * v106_data));
          float v443_data = r1[126];
          r1[126] = (v443_data + (v420_data * v111_data));
          float v448_data = r1[152];
          r1[152] = (v448_data + (v420_data * v116_data));
          float v450_data = r0[24];
          float v453_data = r1[24];
          r1[24] = (v453_data + (v450_data * v91_data));
          float v458_data = r1[50];
          r1[50] = (v458_data + (v450_data * v96_data));
          float v463_data = r1[76];
          r1[76] = (v463_data + (v450_data * v101_data));
          float v468_data = r1[102];
          r1[102] = (v468_data + (v450_data * v106_data));
          float v473_data = r1[128];
          r1[128] = (v473_data + (v450_data * v111_data));
          float v478_data = r1[154];
          r1[154] = (v478_data + (v450_data * v116_data));
          float v480_data = r0[1];
          float v483_data = r1[1];
          r1[1] = (v483_data + (v480_data * v91_data));
          float v488_data = r1[27];
          r1[27] = (v488_data + (v480_data * v96_data));
          float v493_data = r1[53];
          r1[53] = (v493_data + (v480_data * v101_data));
          float v498_data = r1[79];
          r1[79] = (v498_data + (v480_data * v106_data));
          float v503_data = r1[105];
          r1[105] = (v503_data + (v480_data * v111_data));
          float v508_data = r1[131];
          r1[131] = (v508_data + (v480_data * v116_data));
          float v510_data = r0[3];
          float v513_data = r1[3];
          r1[3] = (v513_data + (v510_data * v91_data));
          float v518_data = r1[29];
          r1[29] = (v518_data + (v510_data * v96_data));
          float v523_data = r1[55];
          r1[55] = (v523_data + (v510_data * v101_data));
          float v528_data = r1[81];
          r1[81] = (v528_data + (v510_data * v106_data));
          float v533_data = r1[107];
          r1[107] = (v533_data + (v510_data * v111_data));
          float v538_data = r1[133];
          r1[133] = (v538_data + (v510_data * v116_data));
          float v540_data = r0[5];
          float v543_data = r1[5];
          r1[5] = (v543_data + (v540_data * v91_data));
          float v548_data = r1[31];
          r1[31] = (v548_data + (v540_data * v96_data));
          float v553_data = r1[57];
          r1[57] = (v553_data + (v540_data * v101_data));
          float v558_data = r1[83];
          r1[83] = (v558_data + (v540_data * v106_data));
          float v563_data = r1[109];
          r1[109] = (v563_data + (v540_data * v111_data));
          float v568_data = r1[135];
          r1[135] = (v568_data + (v540_data * v116_data));
          float v570_data = r0[7];
          float v573_data = r1[7];
          r1[7] = (v573_data + (v570_data * v91_data));
          float v578_data = r1[33];
          r1[33] = (v578_data + (v570_data * v96_data));
          float v583_data = r1[59];
          r1[59] = (v583_data + (v570_data * v101_data));
          float v588_data = r1[85];
          r1[85] = (v588_data + (v570_data * v106_data));
          float v593_data = r1[111];
          r1[111] = (v593_data + (v570_data * v111_data));
          float v598_data = r1[137];
          r1[137] = (v598_data + (v570_data * v116_data));
          float v600_data = r0[9];
          float v603_data = r1[9];
          r1[9] = (v603_data + (v600_data * v91_data));
          float v608_data = r1[35];
          r1[35] = (v608_data + (v600_data * v96_data));
          float v613_data = r1[61];
          r1[61] = (v613_data + (v600_data * v101_data));
          float v618_data = r1[87];
          r1[87] = (v618_data + (v600_data * v106_data));
          float v623_data = r1[113];
          r1[113] = (v623_data + (v600_data * v111_data));
          float v628_data = r1[139];
          r1[139] = (v628_data + (v600_data * v116_data));
          float v630_data = r0[11];
          float v633_data = r1[11];
          r1[11] = (v633_data + (v630_data * v91_data));
          float v638_data = r1[37];
          r1[37] = (v638_data + (v630_data * v96_data));
          float v643_data = r1[63];
          r1[63] = (v643_data + (v630_data * v101_data));
          float v648_data = r1[89];
          r1[89] = (v648_data + (v630_data * v106_data));
          float v653_data = r1[115];
          r1[115] = (v653_data + (v630_data * v111_data));
          float v658_data = r1[141];
          r1[141] = (v658_data + (v630_data * v116_data));
          float v660_data = r0[13];
          float v663_data = r1[13];
          r1[13] = (v663_data + (v660_data * v91_data));
          float v668_data = r1[39];
          r1[39] = (v668_data + (v660_data * v96_data));
          float v673_data = r1[65];
          r1[65] = (v673_data + (v660_data * v101_data));
          float v678_data = r1[91];
          r1[91] = (v678_data + (v660_data * v106_data));
          float v683_data = r1[117];
          r1[117] = (v683_data + (v660_data * v111_data));
          float v688_data = r1[143];
          r1[143] = (v688_data + (v660_data * v116_data));
          float v690_data = r0[15];
          float v693_data = r1[15];
          r1[15] = (v693_data + (v690_data * v91_data));
          float v698_data = r1[41];
          r1[41] = (v698_data + (v690_data * v96_data));
          float v703_data = r1[67];
          r1[67] = (v703_data + (v690_data * v101_data));
          float v708_data = r1[93];
          r1[93] = (v708_data + (v690_data * v106_data));
          float v713_data = r1[119];
          r1[119] = (v713_data + (v690_data * v111_data));
          float v718_data = r1[145];
          r1[145] = (v718_data + (v690_data * v116_data));
          float v720_data = r0[17];
          float v723_data = r1[17];
          r1[17] = (v723_data + (v720_data * v91_data));
          float v728_data = r1[43];
          r1[43] = (v728_data + (v720_data * v96_data));
          float v733_data = r1[69];
          r1[69] = (v733_data + (v720_data * v101_data));
          float v738_data = r1[95];
          r1[95] = (v738_data + (v720_data * v106_data));
          float v743_data = r1[121];
          r1[121] = (v743_data + (v720_data * v111_data));
          float v748_data = r1[147];
          r1[147] = (v748_data + (v720_data * v116_data));
          float v750_data = r0[19];
          float v753_data = r1[19];
          r1[19] = (v753_data + (v750_data * v91_data));
          float v758_data = r1[45];
          r1[45] = (v758_data + (v750_data * v96_data));
          float v763_data = r1[71];
          r1[71] = (v763_data + (v750_data * v101_data));
          float v768_data = r1[97];
          r1[97] = (v768_data + (v750_data * v106_data));
          float v773_data = r1[123];
          r1[123] = (v773_data + (v750_data * v111_data));
          float v778_data = r1[149];
          r1[149] = (v778_data + (v750_data * v116_data));
          float v780_data = r0[21];
          float v783_data = r1[21];
          r1[21] = (v783_data + (v780_data * v91_data));
          float v788_data = r1[47];
          r1[47] = (v788_data + (v780_data * v96_data));
          float v793_data = r1[73];
          r1[73] = (v793_data + (v780_data * v101_data));
          float v798_data = r1[99];
          r1[99] = (v798_data + (v780_data * v106_data));
          float v803_data = r1[125];
          r1[125] = (v803_data + (v780_data * v111_data));
          float v808_data = r1[151];
          r1[151] = (v808_data + (v780_data * v116_data));
          float v810_data = r0[23];
          float v813_data = r1[23];
          r1[23] = (v813_data + (v810_data * v91_data));
          float v818_data = r1[49];
          r1[49] = (v818_data + (v810_data * v96_data));
          float v823_data = r1[75];
          r1[75] = (v823_data + (v810_data * v101_data));
          float v828_data = r1[101];
          r1[101] = (v828_data + (v810_data * v106_data));
          float v833_data = r1[127];
          r1[127] = (v833_data + (v810_data * v111_data));
          float v838_data = r1[153];
          r1[153] = (v838_data + (v810_data * v116_data));
          float v840_data = r0[25];
          float v843_data = r1[25];
          r1[25] = (v843_data + (v840_data * v91_data));
          float v848_data = r1[51];
          r1[51] = (v848_data + (v840_data * v96_data));
          float v853_data = r1[77];
          r1[77] = (v853_data + (v840_data * v101_data));
          float v858_data = r1[103];
          r1[103] = (v858_data + (v840_data * v106_data));
          float v863_data = r1[129];
          r1[129] = (v863_data + (v840_data * v111_data));
          float v868_data = r1[155];
          r1[155] = (v868_data + (v840_data * v116_data));
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
                r3[v926_a] = (v936_data + v927_data);
              }
            }
          }
          if (v6_lead < 3) {
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
                r3[(v947_a + v946_a)] = (v962_data + v953_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v6_lead >= 20) {
            #pragma unroll
            for (int32_t v972_i1 = 0; v972_i1 < 1; ++v972_i1) {
              int32_t v974_a = v972_i1 * 2;
              int32_t v991_a = v6_lead + ((v972_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v973_i2 = 0; v973_i2 < 6; ++v973_i2) {
                int32_t v975_a = v973_i2 * 2;
                int32_t v977_a = v974_a + v975_a;
                float v982_data = r3[(v974_a + v975_a)];
                glb_m2[(v991_a + (v973_i2 * 832))] = v982_data;
              }
            }
          }
          if (v6_lead < 3) {
            int32_t v1009_lead = v6_lead + 32_i32;
            #pragma unroll
            for (int32_t v994_i1 = 0; v994_i1 < 1; ++v994_i1) {
              int32_t v998_a = 1 + (v994_i1 * 2);
              int32_t v1013_a = v1009_lead + ((v994_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v995_i2 = 0; v995_i2 < 6; ++v995_i2) {
                int32_t v997_a = v995_i2 * 2;
                int32_t v999_a = v998_a + v997_a;
                float v1004_data = r3[(v998_a + v997_a)];
                glb_m2[(v1013_a + (v995_i2 * 832))] = v1004_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

