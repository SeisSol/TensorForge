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
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 2; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 13; ++v9_i1) {
              int32_t v15_a = v9_i1 * 64;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __ldcg(&glb_m0[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 * 2))] = v24_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v7_lead >= 20) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 1; ++v32_i1) {
              int32_t v40_a = (v32_i1 + 12) * 64;
              int32_t v42_a = v7_lead + v40_a;
              int32_t v52_a = v7_lead + v40_a;
              int32_t v55_a = v32_i1 * 2;
              #pragma unroll
              for (int32_t v33_i2 = 0; v33_i2 < 6; ++v33_i2) {
                int32_t v41_a = v33_i2 * 832;
                int32_t v43_a = v42_a + v41_a;
                float v54_data = glb_m2[(v52_a + v41_a)];
                r2[(v55_a + (v33_i2 * 2))] = v54_data;
              }
            }
          }
          if (v7_lead < 3) {
            int32_t v66_lead = v7_lead + 32_i32;
            int32_t v76_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v60_i1 = 0; v60_i1 < 1; ++v60_i1) {
              int32_t v68_a = (v60_i1 + 12) * 64;
              int32_t v70_a = v66_lead + v68_a;
              int32_t v80_a = v76_lead + v68_a;
              int32_t v85_a = 1 + (v60_i1 * 2);
              #pragma unroll
              for (int32_t v61_i2 = 0; v61_i2 < 6; ++v61_i2) {
                int32_t v69_a = v61_i2 * 832;
                int32_t v71_a = v70_a + v69_a;
                float v82_data = glb_m2[(v80_a + v69_a)];
                r2[(v85_a + (v61_i2 * 2))] = v82_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v91_data = r0[0];
          float v92_data = glb_m1[0];
          float v94_data = r1[0];
          r1[0] = (v94_data + (v91_data * v92_data));
          float v97_data = glb_m1[1];
          float v99_data = r1[26];
          r1[26] = (v99_data + (v91_data * v97_data));
          float v102_data = glb_m1[2];
          float v104_data = r1[52];
          r1[52] = (v104_data + (v91_data * v102_data));
          float v107_data = glb_m1[3];
          float v109_data = r1[78];
          r1[78] = (v109_data + (v91_data * v107_data));
          float v112_data = glb_m1[4];
          float v114_data = r1[104];
          r1[104] = (v114_data + (v91_data * v112_data));
          float v117_data = glb_m1[5];
          float v119_data = r1[130];
          r1[130] = (v119_data + (v91_data * v117_data));
          float v121_data = r0[2];
          float v124_data = r1[2];
          r1[2] = (v124_data + (v121_data * v92_data));
          float v129_data = r1[28];
          r1[28] = (v129_data + (v121_data * v97_data));
          float v134_data = r1[54];
          r1[54] = (v134_data + (v121_data * v102_data));
          float v139_data = r1[80];
          r1[80] = (v139_data + (v121_data * v107_data));
          float v144_data = r1[106];
          r1[106] = (v144_data + (v121_data * v112_data));
          float v149_data = r1[132];
          r1[132] = (v149_data + (v121_data * v117_data));
          float v151_data = r0[4];
          float v154_data = r1[4];
          r1[4] = (v154_data + (v151_data * v92_data));
          float v159_data = r1[30];
          r1[30] = (v159_data + (v151_data * v97_data));
          float v164_data = r1[56];
          r1[56] = (v164_data + (v151_data * v102_data));
          float v169_data = r1[82];
          r1[82] = (v169_data + (v151_data * v107_data));
          float v174_data = r1[108];
          r1[108] = (v174_data + (v151_data * v112_data));
          float v179_data = r1[134];
          r1[134] = (v179_data + (v151_data * v117_data));
          float v181_data = r0[6];
          float v184_data = r1[6];
          r1[6] = (v184_data + (v181_data * v92_data));
          float v189_data = r1[32];
          r1[32] = (v189_data + (v181_data * v97_data));
          float v194_data = r1[58];
          r1[58] = (v194_data + (v181_data * v102_data));
          float v199_data = r1[84];
          r1[84] = (v199_data + (v181_data * v107_data));
          float v204_data = r1[110];
          r1[110] = (v204_data + (v181_data * v112_data));
          float v209_data = r1[136];
          r1[136] = (v209_data + (v181_data * v117_data));
          float v211_data = r0[8];
          float v214_data = r1[8];
          r1[8] = (v214_data + (v211_data * v92_data));
          float v219_data = r1[34];
          r1[34] = (v219_data + (v211_data * v97_data));
          float v224_data = r1[60];
          r1[60] = (v224_data + (v211_data * v102_data));
          float v229_data = r1[86];
          r1[86] = (v229_data + (v211_data * v107_data));
          float v234_data = r1[112];
          r1[112] = (v234_data + (v211_data * v112_data));
          float v239_data = r1[138];
          r1[138] = (v239_data + (v211_data * v117_data));
          float v241_data = r0[10];
          float v244_data = r1[10];
          r1[10] = (v244_data + (v241_data * v92_data));
          float v249_data = r1[36];
          r1[36] = (v249_data + (v241_data * v97_data));
          float v254_data = r1[62];
          r1[62] = (v254_data + (v241_data * v102_data));
          float v259_data = r1[88];
          r1[88] = (v259_data + (v241_data * v107_data));
          float v264_data = r1[114];
          r1[114] = (v264_data + (v241_data * v112_data));
          float v269_data = r1[140];
          r1[140] = (v269_data + (v241_data * v117_data));
          float v271_data = r0[12];
          float v274_data = r1[12];
          r1[12] = (v274_data + (v271_data * v92_data));
          float v279_data = r1[38];
          r1[38] = (v279_data + (v271_data * v97_data));
          float v284_data = r1[64];
          r1[64] = (v284_data + (v271_data * v102_data));
          float v289_data = r1[90];
          r1[90] = (v289_data + (v271_data * v107_data));
          float v294_data = r1[116];
          r1[116] = (v294_data + (v271_data * v112_data));
          float v299_data = r1[142];
          r1[142] = (v299_data + (v271_data * v117_data));
          float v301_data = r0[14];
          float v304_data = r1[14];
          r1[14] = (v304_data + (v301_data * v92_data));
          float v309_data = r1[40];
          r1[40] = (v309_data + (v301_data * v97_data));
          float v314_data = r1[66];
          r1[66] = (v314_data + (v301_data * v102_data));
          float v319_data = r1[92];
          r1[92] = (v319_data + (v301_data * v107_data));
          float v324_data = r1[118];
          r1[118] = (v324_data + (v301_data * v112_data));
          float v329_data = r1[144];
          r1[144] = (v329_data + (v301_data * v117_data));
          float v331_data = r0[16];
          float v334_data = r1[16];
          r1[16] = (v334_data + (v331_data * v92_data));
          float v339_data = r1[42];
          r1[42] = (v339_data + (v331_data * v97_data));
          float v344_data = r1[68];
          r1[68] = (v344_data + (v331_data * v102_data));
          float v349_data = r1[94];
          r1[94] = (v349_data + (v331_data * v107_data));
          float v354_data = r1[120];
          r1[120] = (v354_data + (v331_data * v112_data));
          float v359_data = r1[146];
          r1[146] = (v359_data + (v331_data * v117_data));
          float v361_data = r0[18];
          float v364_data = r1[18];
          r1[18] = (v364_data + (v361_data * v92_data));
          float v369_data = r1[44];
          r1[44] = (v369_data + (v361_data * v97_data));
          float v374_data = r1[70];
          r1[70] = (v374_data + (v361_data * v102_data));
          float v379_data = r1[96];
          r1[96] = (v379_data + (v361_data * v107_data));
          float v384_data = r1[122];
          r1[122] = (v384_data + (v361_data * v112_data));
          float v389_data = r1[148];
          r1[148] = (v389_data + (v361_data * v117_data));
          float v391_data = r0[20];
          float v394_data = r1[20];
          r1[20] = (v394_data + (v391_data * v92_data));
          float v399_data = r1[46];
          r1[46] = (v399_data + (v391_data * v97_data));
          float v404_data = r1[72];
          r1[72] = (v404_data + (v391_data * v102_data));
          float v409_data = r1[98];
          r1[98] = (v409_data + (v391_data * v107_data));
          float v414_data = r1[124];
          r1[124] = (v414_data + (v391_data * v112_data));
          float v419_data = r1[150];
          r1[150] = (v419_data + (v391_data * v117_data));
          float v421_data = r0[22];
          float v424_data = r1[22];
          r1[22] = (v424_data + (v421_data * v92_data));
          float v429_data = r1[48];
          r1[48] = (v429_data + (v421_data * v97_data));
          float v434_data = r1[74];
          r1[74] = (v434_data + (v421_data * v102_data));
          float v439_data = r1[100];
          r1[100] = (v439_data + (v421_data * v107_data));
          float v444_data = r1[126];
          r1[126] = (v444_data + (v421_data * v112_data));
          float v449_data = r1[152];
          r1[152] = (v449_data + (v421_data * v117_data));
          float v451_data = r0[24];
          float v454_data = r1[24];
          r1[24] = (v454_data + (v451_data * v92_data));
          float v459_data = r1[50];
          r1[50] = (v459_data + (v451_data * v97_data));
          float v464_data = r1[76];
          r1[76] = (v464_data + (v451_data * v102_data));
          float v469_data = r1[102];
          r1[102] = (v469_data + (v451_data * v107_data));
          float v474_data = r1[128];
          r1[128] = (v474_data + (v451_data * v112_data));
          float v479_data = r1[154];
          r1[154] = (v479_data + (v451_data * v117_data));
          float v481_data = r0[1];
          float v484_data = r1[1];
          r1[1] = (v484_data + (v481_data * v92_data));
          float v489_data = r1[27];
          r1[27] = (v489_data + (v481_data * v97_data));
          float v494_data = r1[53];
          r1[53] = (v494_data + (v481_data * v102_data));
          float v499_data = r1[79];
          r1[79] = (v499_data + (v481_data * v107_data));
          float v504_data = r1[105];
          r1[105] = (v504_data + (v481_data * v112_data));
          float v509_data = r1[131];
          r1[131] = (v509_data + (v481_data * v117_data));
          float v511_data = r0[3];
          float v514_data = r1[3];
          r1[3] = (v514_data + (v511_data * v92_data));
          float v519_data = r1[29];
          r1[29] = (v519_data + (v511_data * v97_data));
          float v524_data = r1[55];
          r1[55] = (v524_data + (v511_data * v102_data));
          float v529_data = r1[81];
          r1[81] = (v529_data + (v511_data * v107_data));
          float v534_data = r1[107];
          r1[107] = (v534_data + (v511_data * v112_data));
          float v539_data = r1[133];
          r1[133] = (v539_data + (v511_data * v117_data));
          float v541_data = r0[5];
          float v544_data = r1[5];
          r1[5] = (v544_data + (v541_data * v92_data));
          float v549_data = r1[31];
          r1[31] = (v549_data + (v541_data * v97_data));
          float v554_data = r1[57];
          r1[57] = (v554_data + (v541_data * v102_data));
          float v559_data = r1[83];
          r1[83] = (v559_data + (v541_data * v107_data));
          float v564_data = r1[109];
          r1[109] = (v564_data + (v541_data * v112_data));
          float v569_data = r1[135];
          r1[135] = (v569_data + (v541_data * v117_data));
          float v571_data = r0[7];
          float v574_data = r1[7];
          r1[7] = (v574_data + (v571_data * v92_data));
          float v579_data = r1[33];
          r1[33] = (v579_data + (v571_data * v97_data));
          float v584_data = r1[59];
          r1[59] = (v584_data + (v571_data * v102_data));
          float v589_data = r1[85];
          r1[85] = (v589_data + (v571_data * v107_data));
          float v594_data = r1[111];
          r1[111] = (v594_data + (v571_data * v112_data));
          float v599_data = r1[137];
          r1[137] = (v599_data + (v571_data * v117_data));
          float v601_data = r0[9];
          float v604_data = r1[9];
          r1[9] = (v604_data + (v601_data * v92_data));
          float v609_data = r1[35];
          r1[35] = (v609_data + (v601_data * v97_data));
          float v614_data = r1[61];
          r1[61] = (v614_data + (v601_data * v102_data));
          float v619_data = r1[87];
          r1[87] = (v619_data + (v601_data * v107_data));
          float v624_data = r1[113];
          r1[113] = (v624_data + (v601_data * v112_data));
          float v629_data = r1[139];
          r1[139] = (v629_data + (v601_data * v117_data));
          float v631_data = r0[11];
          float v634_data = r1[11];
          r1[11] = (v634_data + (v631_data * v92_data));
          float v639_data = r1[37];
          r1[37] = (v639_data + (v631_data * v97_data));
          float v644_data = r1[63];
          r1[63] = (v644_data + (v631_data * v102_data));
          float v649_data = r1[89];
          r1[89] = (v649_data + (v631_data * v107_data));
          float v654_data = r1[115];
          r1[115] = (v654_data + (v631_data * v112_data));
          float v659_data = r1[141];
          r1[141] = (v659_data + (v631_data * v117_data));
          float v661_data = r0[13];
          float v664_data = r1[13];
          r1[13] = (v664_data + (v661_data * v92_data));
          float v669_data = r1[39];
          r1[39] = (v669_data + (v661_data * v97_data));
          float v674_data = r1[65];
          r1[65] = (v674_data + (v661_data * v102_data));
          float v679_data = r1[91];
          r1[91] = (v679_data + (v661_data * v107_data));
          float v684_data = r1[117];
          r1[117] = (v684_data + (v661_data * v112_data));
          float v689_data = r1[143];
          r1[143] = (v689_data + (v661_data * v117_data));
          float v691_data = r0[15];
          float v694_data = r1[15];
          r1[15] = (v694_data + (v691_data * v92_data));
          float v699_data = r1[41];
          r1[41] = (v699_data + (v691_data * v97_data));
          float v704_data = r1[67];
          r1[67] = (v704_data + (v691_data * v102_data));
          float v709_data = r1[93];
          r1[93] = (v709_data + (v691_data * v107_data));
          float v714_data = r1[119];
          r1[119] = (v714_data + (v691_data * v112_data));
          float v719_data = r1[145];
          r1[145] = (v719_data + (v691_data * v117_data));
          float v721_data = r0[17];
          float v724_data = r1[17];
          r1[17] = (v724_data + (v721_data * v92_data));
          float v729_data = r1[43];
          r1[43] = (v729_data + (v721_data * v97_data));
          float v734_data = r1[69];
          r1[69] = (v734_data + (v721_data * v102_data));
          float v739_data = r1[95];
          r1[95] = (v739_data + (v721_data * v107_data));
          float v744_data = r1[121];
          r1[121] = (v744_data + (v721_data * v112_data));
          float v749_data = r1[147];
          r1[147] = (v749_data + (v721_data * v117_data));
          float v751_data = r0[19];
          float v754_data = r1[19];
          r1[19] = (v754_data + (v751_data * v92_data));
          float v759_data = r1[45];
          r1[45] = (v759_data + (v751_data * v97_data));
          float v764_data = r1[71];
          r1[71] = (v764_data + (v751_data * v102_data));
          float v769_data = r1[97];
          r1[97] = (v769_data + (v751_data * v107_data));
          float v774_data = r1[123];
          r1[123] = (v774_data + (v751_data * v112_data));
          float v779_data = r1[149];
          r1[149] = (v779_data + (v751_data * v117_data));
          float v781_data = r0[21];
          float v784_data = r1[21];
          r1[21] = (v784_data + (v781_data * v92_data));
          float v789_data = r1[47];
          r1[47] = (v789_data + (v781_data * v97_data));
          float v794_data = r1[73];
          r1[73] = (v794_data + (v781_data * v102_data));
          float v799_data = r1[99];
          r1[99] = (v799_data + (v781_data * v107_data));
          float v804_data = r1[125];
          r1[125] = (v804_data + (v781_data * v112_data));
          float v809_data = r1[151];
          r1[151] = (v809_data + (v781_data * v117_data));
          float v811_data = r0[23];
          float v814_data = r1[23];
          r1[23] = (v814_data + (v811_data * v92_data));
          float v819_data = r1[49];
          r1[49] = (v819_data + (v811_data * v97_data));
          float v824_data = r1[75];
          r1[75] = (v824_data + (v811_data * v102_data));
          float v829_data = r1[101];
          r1[101] = (v829_data + (v811_data * v107_data));
          float v834_data = r1[127];
          r1[127] = (v834_data + (v811_data * v112_data));
          float v839_data = r1[153];
          r1[153] = (v839_data + (v811_data * v117_data));
          float v841_data = r0[25];
          float v844_data = r1[25];
          r1[25] = (v844_data + (v841_data * v92_data));
          float v849_data = r1[51];
          r1[51] = (v849_data + (v841_data * v97_data));
          float v854_data = r1[77];
          r1[77] = (v854_data + (v841_data * v102_data));
          float v859_data = r1[103];
          r1[103] = (v859_data + (v841_data * v107_data));
          float v864_data = r1[129];
          r1[129] = (v864_data + (v841_data * v112_data));
          float v869_data = r1[155];
          r1[155] = (v869_data + (v841_data * v117_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
          // [(20, 35), (0, 1), (0, 6)] []
          float ir3[12]{};
          if (v7_lead >= 20) {
            float v877_data = r1[24];
            float v878_data = ir3[0];
            ir3[0] = (v878_data + v877_data);
            float v880_data = r1[50];
            float v881_data = ir3[2];
            ir3[2] = (v881_data + v880_data);
            float v883_data = r1[76];
            float v884_data = ir3[4];
            ir3[4] = (v884_data + v883_data);
            float v886_data = r1[102];
            float v887_data = ir3[6];
            ir3[6] = (v887_data + v886_data);
            float v889_data = r1[128];
            float v890_data = ir3[8];
            ir3[8] = (v890_data + v889_data);
            float v892_data = r1[154];
            float v893_data = ir3[10];
            ir3[10] = (v893_data + v892_data);
          }
          if (v7_lead < 3) {
            float v896_data = r1[25];
            float v897_data = ir3[1];
            ir3[1] = (v897_data + v896_data);
            float v899_data = r1[51];
            float v900_data = ir3[3];
            ir3[3] = (v900_data + v899_data);
            float v902_data = r1[77];
            float v903_data = ir3[5];
            ir3[5] = (v903_data + v902_data);
            float v905_data = r1[103];
            float v906_data = ir3[7];
            ir3[7] = (v906_data + v905_data);
            float v908_data = r1[129];
            float v909_data = ir3[9];
            ir3[9] = (v909_data + v908_data);
            float v911_data = r1[155];
            float v912_data = ir3[11];
            ir3[11] = (v912_data + v911_data);
          }
          if (v7_lead >= 20) {
            #pragma unroll
            for (int32_t v918_n1 = 0; v918_n1 < 1; ++v918_n1) {
              int32_t v920_a = v918_n1 * 2;
              #pragma unroll
              for (int32_t v919_n2 = 0; v919_n2 < 6; ++v919_n2) {
                int32_t v921_a = v919_n2 * 2;
                int32_t v923_a = v920_a + v921_a;
                int32_t v927_a = v920_a + v921_a;
                float v928_data = ir3[v927_a];
                int32_t v932_a = v920_a + v921_a;
                float v937_data = r2[v927_a];
                r3[v927_a] = (v937_data + v928_data);
              }
            }
          }
          if (v7_lead < 3) {
            #pragma unroll
            for (int32_t v944_n1 = 0; v944_n1 < 1; ++v944_n1) {
              int32_t v948_a = 1 + (v944_n1 * 2);
              #pragma unroll
              for (int32_t v945_n2 = 0; v945_n2 < 6; ++v945_n2) {
                int32_t v947_a = v945_n2 * 2;
                int32_t v949_a = v948_a + v947_a;
                float v954_data = ir3[(v948_a + v947_a)];
                int32_t v958_a = v948_a + v947_a;
                float v963_data = r2[(v948_a + v947_a)];
                r3[(v948_a + v947_a)] = (v963_data + v954_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v7_lead >= 20) {
            #pragma unroll
            for (int32_t v973_i1 = 0; v973_i1 < 1; ++v973_i1) {
              int32_t v975_a = v973_i1 * 2;
              int32_t v992_a = v7_lead + ((v973_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v974_i2 = 0; v974_i2 < 6; ++v974_i2) {
                int32_t v976_a = v974_i2 * 2;
                int32_t v978_a = v975_a + v976_a;
                float v983_data = r3[(v975_a + v976_a)];
                glb_m2[(v992_a + (v974_i2 * 832))] = v983_data;
              }
            }
          }
          if (v7_lead < 3) {
            int32_t v1010_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v995_i1 = 0; v995_i1 < 1; ++v995_i1) {
              int32_t v999_a = 1 + (v995_i1 * 2);
              int32_t v1014_a = v1010_lead + ((v995_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v996_i2 = 0; v996_i2 < 6; ++v996_i2) {
                int32_t v998_a = v996_i2 * 2;
                int32_t v1000_a = v999_a + v998_a;
                float v1005_data = r3[(v999_a + v998_a)];
                glb_m2[(v1014_a + (v996_i2 * 832))] = v1005_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

