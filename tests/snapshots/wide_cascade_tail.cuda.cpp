// === base name ===
kernel_8162b17515

// === header ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8162b17515, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_8162b17515, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8162b17515<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 24×9(24×9) {0..24}×{0..9} strided
    // m1 24×24(24×24) {0..24}×{0..24} strided
    // m2 24×9(24×9) {0..24}×{0..9} strided
    // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[224 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[224];
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 216 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 576 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v14_lead = threadIdx.x % 32;
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 24; ++v16_i1) {
              float v24_data = __ldcg(&glb_m1[(v14_lead + (v16_i1 * 24))]);
              r0[v16_i1] = v24_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 2 * threadIdx.x + 128], &glb_m2[0 + 0 + 2 * threadIdx.x + 128], 8);
          if (threadIdx.x < 24) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 192], &glb_m2[0 + 0 + 1 * threadIdx.x + 192], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float ir1[9]{};
          if (v14_lead < 24) {
            float v35_data = r0[0];
            float v36_data = s0[0];
            float v38_data = ir1[0];
            ir1[0] = (v38_data + (v35_data * v36_data));
            float v41_data = s0[24];
            float v43_data = ir1[1];
            ir1[1] = (v43_data + (v35_data * v41_data));
            float v46_data = s0[48];
            float v48_data = ir1[2];
            ir1[2] = (v48_data + (v35_data * v46_data));
            float v51_data = s0[72];
            float v53_data = ir1[3];
            ir1[3] = (v53_data + (v35_data * v51_data));
            float v56_data = s0[96];
            float v58_data = ir1[4];
            ir1[4] = (v58_data + (v35_data * v56_data));
            float v61_data = s0[120];
            float v63_data = ir1[5];
            ir1[5] = (v63_data + (v35_data * v61_data));
            float v66_data = s0[144];
            float v68_data = ir1[6];
            ir1[6] = (v68_data + (v35_data * v66_data));
            float v71_data = s0[168];
            float v73_data = ir1[7];
            ir1[7] = (v73_data + (v35_data * v71_data));
            float v76_data = s0[192];
            float v78_data = ir1[8];
            ir1[8] = (v78_data + (v35_data * v76_data));
          }
          if (v14_lead < 24) {
            float v84_data = r0[1];
            float v85_data = s0[1];
            float v87_data = ir1[0];
            ir1[0] = (v87_data + (v84_data * v85_data));
            float v90_data = s0[25];
            float v92_data = ir1[1];
            ir1[1] = (v92_data + (v84_data * v90_data));
            float v95_data = s0[49];
            float v97_data = ir1[2];
            ir1[2] = (v97_data + (v84_data * v95_data));
            float v100_data = s0[73];
            float v102_data = ir1[3];
            ir1[3] = (v102_data + (v84_data * v100_data));
            float v105_data = s0[97];
            float v107_data = ir1[4];
            ir1[4] = (v107_data + (v84_data * v105_data));
            float v110_data = s0[121];
            float v112_data = ir1[5];
            ir1[5] = (v112_data + (v84_data * v110_data));
            float v115_data = s0[145];
            float v117_data = ir1[6];
            ir1[6] = (v117_data + (v84_data * v115_data));
            float v120_data = s0[169];
            float v122_data = ir1[7];
            ir1[7] = (v122_data + (v84_data * v120_data));
            float v125_data = s0[193];
            float v127_data = ir1[8];
            ir1[8] = (v127_data + (v84_data * v125_data));
          }
          if (v14_lead < 24) {
            float v133_data = r0[2];
            float v134_data = s0[2];
            float v136_data = ir1[0];
            ir1[0] = (v136_data + (v133_data * v134_data));
            float v139_data = s0[26];
            float v141_data = ir1[1];
            ir1[1] = (v141_data + (v133_data * v139_data));
            float v144_data = s0[50];
            float v146_data = ir1[2];
            ir1[2] = (v146_data + (v133_data * v144_data));
            float v149_data = s0[74];
            float v151_data = ir1[3];
            ir1[3] = (v151_data + (v133_data * v149_data));
            float v154_data = s0[98];
            float v156_data = ir1[4];
            ir1[4] = (v156_data + (v133_data * v154_data));
            float v159_data = s0[122];
            float v161_data = ir1[5];
            ir1[5] = (v161_data + (v133_data * v159_data));
            float v164_data = s0[146];
            float v166_data = ir1[6];
            ir1[6] = (v166_data + (v133_data * v164_data));
            float v169_data = s0[170];
            float v171_data = ir1[7];
            ir1[7] = (v171_data + (v133_data * v169_data));
            float v174_data = s0[194];
            float v176_data = ir1[8];
            ir1[8] = (v176_data + (v133_data * v174_data));
          }
          if (v14_lead < 24) {
            float v182_data = r0[3];
            float v183_data = s0[3];
            float v185_data = ir1[0];
            ir1[0] = (v185_data + (v182_data * v183_data));
            float v188_data = s0[27];
            float v190_data = ir1[1];
            ir1[1] = (v190_data + (v182_data * v188_data));
            float v193_data = s0[51];
            float v195_data = ir1[2];
            ir1[2] = (v195_data + (v182_data * v193_data));
            float v198_data = s0[75];
            float v200_data = ir1[3];
            ir1[3] = (v200_data + (v182_data * v198_data));
            float v203_data = s0[99];
            float v205_data = ir1[4];
            ir1[4] = (v205_data + (v182_data * v203_data));
            float v208_data = s0[123];
            float v210_data = ir1[5];
            ir1[5] = (v210_data + (v182_data * v208_data));
            float v213_data = s0[147];
            float v215_data = ir1[6];
            ir1[6] = (v215_data + (v182_data * v213_data));
            float v218_data = s0[171];
            float v220_data = ir1[7];
            ir1[7] = (v220_data + (v182_data * v218_data));
            float v223_data = s0[195];
            float v225_data = ir1[8];
            ir1[8] = (v225_data + (v182_data * v223_data));
          }
          if (v14_lead < 24) {
            float v231_data = r0[4];
            float v232_data = s0[4];
            float v234_data = ir1[0];
            ir1[0] = (v234_data + (v231_data * v232_data));
            float v237_data = s0[28];
            float v239_data = ir1[1];
            ir1[1] = (v239_data + (v231_data * v237_data));
            float v242_data = s0[52];
            float v244_data = ir1[2];
            ir1[2] = (v244_data + (v231_data * v242_data));
            float v247_data = s0[76];
            float v249_data = ir1[3];
            ir1[3] = (v249_data + (v231_data * v247_data));
            float v252_data = s0[100];
            float v254_data = ir1[4];
            ir1[4] = (v254_data + (v231_data * v252_data));
            float v257_data = s0[124];
            float v259_data = ir1[5];
            ir1[5] = (v259_data + (v231_data * v257_data));
            float v262_data = s0[148];
            float v264_data = ir1[6];
            ir1[6] = (v264_data + (v231_data * v262_data));
            float v267_data = s0[172];
            float v269_data = ir1[7];
            ir1[7] = (v269_data + (v231_data * v267_data));
            float v272_data = s0[196];
            float v274_data = ir1[8];
            ir1[8] = (v274_data + (v231_data * v272_data));
          }
          if (v14_lead < 24) {
            float v280_data = r0[5];
            float v281_data = s0[5];
            float v283_data = ir1[0];
            ir1[0] = (v283_data + (v280_data * v281_data));
            float v286_data = s0[29];
            float v288_data = ir1[1];
            ir1[1] = (v288_data + (v280_data * v286_data));
            float v291_data = s0[53];
            float v293_data = ir1[2];
            ir1[2] = (v293_data + (v280_data * v291_data));
            float v296_data = s0[77];
            float v298_data = ir1[3];
            ir1[3] = (v298_data + (v280_data * v296_data));
            float v301_data = s0[101];
            float v303_data = ir1[4];
            ir1[4] = (v303_data + (v280_data * v301_data));
            float v306_data = s0[125];
            float v308_data = ir1[5];
            ir1[5] = (v308_data + (v280_data * v306_data));
            float v311_data = s0[149];
            float v313_data = ir1[6];
            ir1[6] = (v313_data + (v280_data * v311_data));
            float v316_data = s0[173];
            float v318_data = ir1[7];
            ir1[7] = (v318_data + (v280_data * v316_data));
            float v321_data = s0[197];
            float v323_data = ir1[8];
            ir1[8] = (v323_data + (v280_data * v321_data));
          }
          if (v14_lead < 24) {
            float v329_data = r0[6];
            float v330_data = s0[6];
            float v332_data = ir1[0];
            ir1[0] = (v332_data + (v329_data * v330_data));
            float v335_data = s0[30];
            float v337_data = ir1[1];
            ir1[1] = (v337_data + (v329_data * v335_data));
            float v340_data = s0[54];
            float v342_data = ir1[2];
            ir1[2] = (v342_data + (v329_data * v340_data));
            float v345_data = s0[78];
            float v347_data = ir1[3];
            ir1[3] = (v347_data + (v329_data * v345_data));
            float v350_data = s0[102];
            float v352_data = ir1[4];
            ir1[4] = (v352_data + (v329_data * v350_data));
            float v355_data = s0[126];
            float v357_data = ir1[5];
            ir1[5] = (v357_data + (v329_data * v355_data));
            float v360_data = s0[150];
            float v362_data = ir1[6];
            ir1[6] = (v362_data + (v329_data * v360_data));
            float v365_data = s0[174];
            float v367_data = ir1[7];
            ir1[7] = (v367_data + (v329_data * v365_data));
            float v370_data = s0[198];
            float v372_data = ir1[8];
            ir1[8] = (v372_data + (v329_data * v370_data));
          }
          if (v14_lead < 24) {
            float v378_data = r0[7];
            float v379_data = s0[7];
            float v381_data = ir1[0];
            ir1[0] = (v381_data + (v378_data * v379_data));
            float v384_data = s0[31];
            float v386_data = ir1[1];
            ir1[1] = (v386_data + (v378_data * v384_data));
            float v389_data = s0[55];
            float v391_data = ir1[2];
            ir1[2] = (v391_data + (v378_data * v389_data));
            float v394_data = s0[79];
            float v396_data = ir1[3];
            ir1[3] = (v396_data + (v378_data * v394_data));
            float v399_data = s0[103];
            float v401_data = ir1[4];
            ir1[4] = (v401_data + (v378_data * v399_data));
            float v404_data = s0[127];
            float v406_data = ir1[5];
            ir1[5] = (v406_data + (v378_data * v404_data));
            float v409_data = s0[151];
            float v411_data = ir1[6];
            ir1[6] = (v411_data + (v378_data * v409_data));
            float v414_data = s0[175];
            float v416_data = ir1[7];
            ir1[7] = (v416_data + (v378_data * v414_data));
            float v419_data = s0[199];
            float v421_data = ir1[8];
            ir1[8] = (v421_data + (v378_data * v419_data));
          }
          if (v14_lead < 24) {
            float v427_data = r0[8];
            float v428_data = s0[8];
            float v430_data = ir1[0];
            ir1[0] = (v430_data + (v427_data * v428_data));
            float v433_data = s0[32];
            float v435_data = ir1[1];
            ir1[1] = (v435_data + (v427_data * v433_data));
            float v438_data = s0[56];
            float v440_data = ir1[2];
            ir1[2] = (v440_data + (v427_data * v438_data));
            float v443_data = s0[80];
            float v445_data = ir1[3];
            ir1[3] = (v445_data + (v427_data * v443_data));
            float v448_data = s0[104];
            float v450_data = ir1[4];
            ir1[4] = (v450_data + (v427_data * v448_data));
            float v453_data = s0[128];
            float v455_data = ir1[5];
            ir1[5] = (v455_data + (v427_data * v453_data));
            float v458_data = s0[152];
            float v460_data = ir1[6];
            ir1[6] = (v460_data + (v427_data * v458_data));
            float v463_data = s0[176];
            float v465_data = ir1[7];
            ir1[7] = (v465_data + (v427_data * v463_data));
            float v468_data = s0[200];
            float v470_data = ir1[8];
            ir1[8] = (v470_data + (v427_data * v468_data));
          }
          if (v14_lead < 24) {
            float v476_data = r0[9];
            float v477_data = s0[9];
            float v479_data = ir1[0];
            ir1[0] = (v479_data + (v476_data * v477_data));
            float v482_data = s0[33];
            float v484_data = ir1[1];
            ir1[1] = (v484_data + (v476_data * v482_data));
            float v487_data = s0[57];
            float v489_data = ir1[2];
            ir1[2] = (v489_data + (v476_data * v487_data));
            float v492_data = s0[81];
            float v494_data = ir1[3];
            ir1[3] = (v494_data + (v476_data * v492_data));
            float v497_data = s0[105];
            float v499_data = ir1[4];
            ir1[4] = (v499_data + (v476_data * v497_data));
            float v502_data = s0[129];
            float v504_data = ir1[5];
            ir1[5] = (v504_data + (v476_data * v502_data));
            float v507_data = s0[153];
            float v509_data = ir1[6];
            ir1[6] = (v509_data + (v476_data * v507_data));
            float v512_data = s0[177];
            float v514_data = ir1[7];
            ir1[7] = (v514_data + (v476_data * v512_data));
            float v517_data = s0[201];
            float v519_data = ir1[8];
            ir1[8] = (v519_data + (v476_data * v517_data));
          }
          if (v14_lead < 24) {
            float v525_data = r0[10];
            float v526_data = s0[10];
            float v528_data = ir1[0];
            ir1[0] = (v528_data + (v525_data * v526_data));
            float v531_data = s0[34];
            float v533_data = ir1[1];
            ir1[1] = (v533_data + (v525_data * v531_data));
            float v536_data = s0[58];
            float v538_data = ir1[2];
            ir1[2] = (v538_data + (v525_data * v536_data));
            float v541_data = s0[82];
            float v543_data = ir1[3];
            ir1[3] = (v543_data + (v525_data * v541_data));
            float v546_data = s0[106];
            float v548_data = ir1[4];
            ir1[4] = (v548_data + (v525_data * v546_data));
            float v551_data = s0[130];
            float v553_data = ir1[5];
            ir1[5] = (v553_data + (v525_data * v551_data));
            float v556_data = s0[154];
            float v558_data = ir1[6];
            ir1[6] = (v558_data + (v525_data * v556_data));
            float v561_data = s0[178];
            float v563_data = ir1[7];
            ir1[7] = (v563_data + (v525_data * v561_data));
            float v566_data = s0[202];
            float v568_data = ir1[8];
            ir1[8] = (v568_data + (v525_data * v566_data));
          }
          if (v14_lead < 24) {
            float v574_data = r0[11];
            float v575_data = s0[11];
            float v577_data = ir1[0];
            ir1[0] = (v577_data + (v574_data * v575_data));
            float v580_data = s0[35];
            float v582_data = ir1[1];
            ir1[1] = (v582_data + (v574_data * v580_data));
            float v585_data = s0[59];
            float v587_data = ir1[2];
            ir1[2] = (v587_data + (v574_data * v585_data));
            float v590_data = s0[83];
            float v592_data = ir1[3];
            ir1[3] = (v592_data + (v574_data * v590_data));
            float v595_data = s0[107];
            float v597_data = ir1[4];
            ir1[4] = (v597_data + (v574_data * v595_data));
            float v600_data = s0[131];
            float v602_data = ir1[5];
            ir1[5] = (v602_data + (v574_data * v600_data));
            float v605_data = s0[155];
            float v607_data = ir1[6];
            ir1[6] = (v607_data + (v574_data * v605_data));
            float v610_data = s0[179];
            float v612_data = ir1[7];
            ir1[7] = (v612_data + (v574_data * v610_data));
            float v615_data = s0[203];
            float v617_data = ir1[8];
            ir1[8] = (v617_data + (v574_data * v615_data));
          }
          if (v14_lead < 24) {
            float v623_data = r0[12];
            float v624_data = s0[12];
            float v626_data = ir1[0];
            ir1[0] = (v626_data + (v623_data * v624_data));
            float v629_data = s0[36];
            float v631_data = ir1[1];
            ir1[1] = (v631_data + (v623_data * v629_data));
            float v634_data = s0[60];
            float v636_data = ir1[2];
            ir1[2] = (v636_data + (v623_data * v634_data));
            float v639_data = s0[84];
            float v641_data = ir1[3];
            ir1[3] = (v641_data + (v623_data * v639_data));
            float v644_data = s0[108];
            float v646_data = ir1[4];
            ir1[4] = (v646_data + (v623_data * v644_data));
            float v649_data = s0[132];
            float v651_data = ir1[5];
            ir1[5] = (v651_data + (v623_data * v649_data));
            float v654_data = s0[156];
            float v656_data = ir1[6];
            ir1[6] = (v656_data + (v623_data * v654_data));
            float v659_data = s0[180];
            float v661_data = ir1[7];
            ir1[7] = (v661_data + (v623_data * v659_data));
            float v664_data = s0[204];
            float v666_data = ir1[8];
            ir1[8] = (v666_data + (v623_data * v664_data));
          }
          if (v14_lead < 24) {
            float v672_data = r0[13];
            float v673_data = s0[13];
            float v675_data = ir1[0];
            ir1[0] = (v675_data + (v672_data * v673_data));
            float v678_data = s0[37];
            float v680_data = ir1[1];
            ir1[1] = (v680_data + (v672_data * v678_data));
            float v683_data = s0[61];
            float v685_data = ir1[2];
            ir1[2] = (v685_data + (v672_data * v683_data));
            float v688_data = s0[85];
            float v690_data = ir1[3];
            ir1[3] = (v690_data + (v672_data * v688_data));
            float v693_data = s0[109];
            float v695_data = ir1[4];
            ir1[4] = (v695_data + (v672_data * v693_data));
            float v698_data = s0[133];
            float v700_data = ir1[5];
            ir1[5] = (v700_data + (v672_data * v698_data));
            float v703_data = s0[157];
            float v705_data = ir1[6];
            ir1[6] = (v705_data + (v672_data * v703_data));
            float v708_data = s0[181];
            float v710_data = ir1[7];
            ir1[7] = (v710_data + (v672_data * v708_data));
            float v713_data = s0[205];
            float v715_data = ir1[8];
            ir1[8] = (v715_data + (v672_data * v713_data));
          }
          if (v14_lead < 24) {
            float v721_data = r0[14];
            float v722_data = s0[14];
            float v724_data = ir1[0];
            ir1[0] = (v724_data + (v721_data * v722_data));
            float v727_data = s0[38];
            float v729_data = ir1[1];
            ir1[1] = (v729_data + (v721_data * v727_data));
            float v732_data = s0[62];
            float v734_data = ir1[2];
            ir1[2] = (v734_data + (v721_data * v732_data));
            float v737_data = s0[86];
            float v739_data = ir1[3];
            ir1[3] = (v739_data + (v721_data * v737_data));
            float v742_data = s0[110];
            float v744_data = ir1[4];
            ir1[4] = (v744_data + (v721_data * v742_data));
            float v747_data = s0[134];
            float v749_data = ir1[5];
            ir1[5] = (v749_data + (v721_data * v747_data));
            float v752_data = s0[158];
            float v754_data = ir1[6];
            ir1[6] = (v754_data + (v721_data * v752_data));
            float v757_data = s0[182];
            float v759_data = ir1[7];
            ir1[7] = (v759_data + (v721_data * v757_data));
            float v762_data = s0[206];
            float v764_data = ir1[8];
            ir1[8] = (v764_data + (v721_data * v762_data));
          }
          if (v14_lead < 24) {
            float v770_data = r0[15];
            float v771_data = s0[15];
            float v773_data = ir1[0];
            ir1[0] = (v773_data + (v770_data * v771_data));
            float v776_data = s0[39];
            float v778_data = ir1[1];
            ir1[1] = (v778_data + (v770_data * v776_data));
            float v781_data = s0[63];
            float v783_data = ir1[2];
            ir1[2] = (v783_data + (v770_data * v781_data));
            float v786_data = s0[87];
            float v788_data = ir1[3];
            ir1[3] = (v788_data + (v770_data * v786_data));
            float v791_data = s0[111];
            float v793_data = ir1[4];
            ir1[4] = (v793_data + (v770_data * v791_data));
            float v796_data = s0[135];
            float v798_data = ir1[5];
            ir1[5] = (v798_data + (v770_data * v796_data));
            float v801_data = s0[159];
            float v803_data = ir1[6];
            ir1[6] = (v803_data + (v770_data * v801_data));
            float v806_data = s0[183];
            float v808_data = ir1[7];
            ir1[7] = (v808_data + (v770_data * v806_data));
            float v811_data = s0[207];
            float v813_data = ir1[8];
            ir1[8] = (v813_data + (v770_data * v811_data));
          }
          if (v14_lead < 24) {
            float v819_data = r0[16];
            float v820_data = s0[16];
            float v822_data = ir1[0];
            ir1[0] = (v822_data + (v819_data * v820_data));
            float v825_data = s0[40];
            float v827_data = ir1[1];
            ir1[1] = (v827_data + (v819_data * v825_data));
            float v830_data = s0[64];
            float v832_data = ir1[2];
            ir1[2] = (v832_data + (v819_data * v830_data));
            float v835_data = s0[88];
            float v837_data = ir1[3];
            ir1[3] = (v837_data + (v819_data * v835_data));
            float v840_data = s0[112];
            float v842_data = ir1[4];
            ir1[4] = (v842_data + (v819_data * v840_data));
            float v845_data = s0[136];
            float v847_data = ir1[5];
            ir1[5] = (v847_data + (v819_data * v845_data));
            float v850_data = s0[160];
            float v852_data = ir1[6];
            ir1[6] = (v852_data + (v819_data * v850_data));
            float v855_data = s0[184];
            float v857_data = ir1[7];
            ir1[7] = (v857_data + (v819_data * v855_data));
            float v860_data = s0[208];
            float v862_data = ir1[8];
            ir1[8] = (v862_data + (v819_data * v860_data));
          }
          if (v14_lead < 24) {
            float v868_data = r0[17];
            float v869_data = s0[17];
            float v871_data = ir1[0];
            ir1[0] = (v871_data + (v868_data * v869_data));
            float v874_data = s0[41];
            float v876_data = ir1[1];
            ir1[1] = (v876_data + (v868_data * v874_data));
            float v879_data = s0[65];
            float v881_data = ir1[2];
            ir1[2] = (v881_data + (v868_data * v879_data));
            float v884_data = s0[89];
            float v886_data = ir1[3];
            ir1[3] = (v886_data + (v868_data * v884_data));
            float v889_data = s0[113];
            float v891_data = ir1[4];
            ir1[4] = (v891_data + (v868_data * v889_data));
            float v894_data = s0[137];
            float v896_data = ir1[5];
            ir1[5] = (v896_data + (v868_data * v894_data));
            float v899_data = s0[161];
            float v901_data = ir1[6];
            ir1[6] = (v901_data + (v868_data * v899_data));
            float v904_data = s0[185];
            float v906_data = ir1[7];
            ir1[7] = (v906_data + (v868_data * v904_data));
            float v909_data = s0[209];
            float v911_data = ir1[8];
            ir1[8] = (v911_data + (v868_data * v909_data));
          }
          if (v14_lead < 24) {
            float v917_data = r0[18];
            float v918_data = s0[18];
            float v920_data = ir1[0];
            ir1[0] = (v920_data + (v917_data * v918_data));
            float v923_data = s0[42];
            float v925_data = ir1[1];
            ir1[1] = (v925_data + (v917_data * v923_data));
            float v928_data = s0[66];
            float v930_data = ir1[2];
            ir1[2] = (v930_data + (v917_data * v928_data));
            float v933_data = s0[90];
            float v935_data = ir1[3];
            ir1[3] = (v935_data + (v917_data * v933_data));
            float v938_data = s0[114];
            float v940_data = ir1[4];
            ir1[4] = (v940_data + (v917_data * v938_data));
            float v943_data = s0[138];
            float v945_data = ir1[5];
            ir1[5] = (v945_data + (v917_data * v943_data));
            float v948_data = s0[162];
            float v950_data = ir1[6];
            ir1[6] = (v950_data + (v917_data * v948_data));
            float v953_data = s0[186];
            float v955_data = ir1[7];
            ir1[7] = (v955_data + (v917_data * v953_data));
            float v958_data = s0[210];
            float v960_data = ir1[8];
            ir1[8] = (v960_data + (v917_data * v958_data));
          }
          if (v14_lead < 24) {
            float v966_data = r0[19];
            float v967_data = s0[19];
            float v969_data = ir1[0];
            ir1[0] = (v969_data + (v966_data * v967_data));
            float v972_data = s0[43];
            float v974_data = ir1[1];
            ir1[1] = (v974_data + (v966_data * v972_data));
            float v977_data = s0[67];
            float v979_data = ir1[2];
            ir1[2] = (v979_data + (v966_data * v977_data));
            float v982_data = s0[91];
            float v984_data = ir1[3];
            ir1[3] = (v984_data + (v966_data * v982_data));
            float v987_data = s0[115];
            float v989_data = ir1[4];
            ir1[4] = (v989_data + (v966_data * v987_data));
            float v992_data = s0[139];
            float v994_data = ir1[5];
            ir1[5] = (v994_data + (v966_data * v992_data));
            float v997_data = s0[163];
            float v999_data = ir1[6];
            ir1[6] = (v999_data + (v966_data * v997_data));
            float v1002_data = s0[187];
            float v1004_data = ir1[7];
            ir1[7] = (v1004_data + (v966_data * v1002_data));
            float v1007_data = s0[211];
            float v1009_data = ir1[8];
            ir1[8] = (v1009_data + (v966_data * v1007_data));
          }
          if (v14_lead < 24) {
            float v1015_data = r0[20];
            float v1016_data = s0[20];
            float v1018_data = ir1[0];
            ir1[0] = (v1018_data + (v1015_data * v1016_data));
            float v1021_data = s0[44];
            float v1023_data = ir1[1];
            ir1[1] = (v1023_data + (v1015_data * v1021_data));
            float v1026_data = s0[68];
            float v1028_data = ir1[2];
            ir1[2] = (v1028_data + (v1015_data * v1026_data));
            float v1031_data = s0[92];
            float v1033_data = ir1[3];
            ir1[3] = (v1033_data + (v1015_data * v1031_data));
            float v1036_data = s0[116];
            float v1038_data = ir1[4];
            ir1[4] = (v1038_data + (v1015_data * v1036_data));
            float v1041_data = s0[140];
            float v1043_data = ir1[5];
            ir1[5] = (v1043_data + (v1015_data * v1041_data));
            float v1046_data = s0[164];
            float v1048_data = ir1[6];
            ir1[6] = (v1048_data + (v1015_data * v1046_data));
            float v1051_data = s0[188];
            float v1053_data = ir1[7];
            ir1[7] = (v1053_data + (v1015_data * v1051_data));
            float v1056_data = s0[212];
            float v1058_data = ir1[8];
            ir1[8] = (v1058_data + (v1015_data * v1056_data));
          }
          if (v14_lead < 24) {
            float v1064_data = r0[21];
            float v1065_data = s0[21];
            float v1067_data = ir1[0];
            ir1[0] = (v1067_data + (v1064_data * v1065_data));
            float v1070_data = s0[45];
            float v1072_data = ir1[1];
            ir1[1] = (v1072_data + (v1064_data * v1070_data));
            float v1075_data = s0[69];
            float v1077_data = ir1[2];
            ir1[2] = (v1077_data + (v1064_data * v1075_data));
            float v1080_data = s0[93];
            float v1082_data = ir1[3];
            ir1[3] = (v1082_data + (v1064_data * v1080_data));
            float v1085_data = s0[117];
            float v1087_data = ir1[4];
            ir1[4] = (v1087_data + (v1064_data * v1085_data));
            float v1090_data = s0[141];
            float v1092_data = ir1[5];
            ir1[5] = (v1092_data + (v1064_data * v1090_data));
            float v1095_data = s0[165];
            float v1097_data = ir1[6];
            ir1[6] = (v1097_data + (v1064_data * v1095_data));
            float v1100_data = s0[189];
            float v1102_data = ir1[7];
            ir1[7] = (v1102_data + (v1064_data * v1100_data));
            float v1105_data = s0[213];
            float v1107_data = ir1[8];
            ir1[8] = (v1107_data + (v1064_data * v1105_data));
          }
          if (v14_lead < 24) {
            float v1113_data = r0[22];
            float v1114_data = s0[22];
            float v1116_data = ir1[0];
            ir1[0] = (v1116_data + (v1113_data * v1114_data));
            float v1119_data = s0[46];
            float v1121_data = ir1[1];
            ir1[1] = (v1121_data + (v1113_data * v1119_data));
            float v1124_data = s0[70];
            float v1126_data = ir1[2];
            ir1[2] = (v1126_data + (v1113_data * v1124_data));
            float v1129_data = s0[94];
            float v1131_data = ir1[3];
            ir1[3] = (v1131_data + (v1113_data * v1129_data));
            float v1134_data = s0[118];
            float v1136_data = ir1[4];
            ir1[4] = (v1136_data + (v1113_data * v1134_data));
            float v1139_data = s0[142];
            float v1141_data = ir1[5];
            ir1[5] = (v1141_data + (v1113_data * v1139_data));
            float v1144_data = s0[166];
            float v1146_data = ir1[6];
            ir1[6] = (v1146_data + (v1113_data * v1144_data));
            float v1149_data = s0[190];
            float v1151_data = ir1[7];
            ir1[7] = (v1151_data + (v1113_data * v1149_data));
            float v1154_data = s0[214];
            float v1156_data = ir1[8];
            ir1[8] = (v1156_data + (v1113_data * v1154_data));
          }
          if (v14_lead < 24) {
            float v1162_data = r0[23];
            float v1163_data = s0[23];
            float v1165_data = ir1[0];
            ir1[0] = (v1165_data + (v1162_data * v1163_data));
            float v1168_data = s0[47];
            float v1170_data = ir1[1];
            ir1[1] = (v1170_data + (v1162_data * v1168_data));
            float v1173_data = s0[71];
            float v1175_data = ir1[2];
            ir1[2] = (v1175_data + (v1162_data * v1173_data));
            float v1178_data = s0[95];
            float v1180_data = ir1[3];
            ir1[3] = (v1180_data + (v1162_data * v1178_data));
            float v1183_data = s0[119];
            float v1185_data = ir1[4];
            ir1[4] = (v1185_data + (v1162_data * v1183_data));
            float v1188_data = s0[143];
            float v1190_data = ir1[5];
            ir1[5] = (v1190_data + (v1162_data * v1188_data));
            float v1193_data = s0[167];
            float v1195_data = ir1[6];
            ir1[6] = (v1195_data + (v1162_data * v1193_data));
            float v1198_data = s0[191];
            float v1200_data = ir1[7];
            ir1[7] = (v1200_data + (v1162_data * v1198_data));
            float v1203_data = s0[215];
            float v1205_data = ir1[8];
            ir1[8] = (v1205_data + (v1162_data * v1203_data));
          }
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v1211_n1 = 0; v1211_n1 < 9; ++v1211_n1) {
              float v1213_data = ir1[v1211_n1];
              r1[v1211_n1] = v1213_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v1219_i1 = 0; v1219_i1 < 9; ++v1219_i1) {
              float v1221_data = r1[v1219_i1];
              glb_m0[(v14_lead + (v1219_i1 * 24))] = v1221_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

