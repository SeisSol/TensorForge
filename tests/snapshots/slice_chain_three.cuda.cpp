// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08703cce1d, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08703cce1d, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08703cce1d<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(12×6) {0..12}×{0..6} strided
    // m1 32×32(6×6) {0..6}×{0..6} strided
    // m2 32×32(12×6) {0..12}×{0..6} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
    // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          alignas(16) float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 6; ++v9_i1) {
              int32_t v15_a = v9_i1 * 12;
              int32_t v16_a = v7_lead + v15_a;
              float v24_data = __ldcg(&glb_m0[(v7_lead + v15_a)]);
              int32_t v25_a = 0 + v9_i1;
              r0[v25_a] = v24_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_commit();
          if (threadIdx.x < 4) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m0););
          alignas(16) float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              int32_t v41_a = v35_i1 * 12;
              int32_t v42_a = v7_lead + v41_a;
              float v50_data = __ldcg(&glb_m3[(v7_lead + v41_a)]);
              int32_t v51_a = 0 + v35_i1;
              r2[v51_a] = v50_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v7_lead < 12) {
            float v57_data = r0[0];
            float v58_data = s0[0];
            float v60_data = r1[0];
            r1[0] = (v60_data + (v57_data * v58_data));
            float v63_data = s0[6];
            float v65_data = r1[1];
            r1[1] = (v65_data + (v57_data * v63_data));
            float v68_data = s0[12];
            float v70_data = r1[2];
            r1[2] = (v70_data + (v57_data * v68_data));
            float v73_data = s0[18];
            float v75_data = r1[3];
            r1[3] = (v75_data + (v57_data * v73_data));
            float v78_data = s0[24];
            float v80_data = r1[4];
            r1[4] = (v80_data + (v57_data * v78_data));
            float v83_data = s0[30];
            float v85_data = r1[5];
            r1[5] = (v85_data + (v57_data * v83_data));
          }
          if (v7_lead < 12) {
            float v91_data = r0[1];
            float v92_data = s0[1];
            float v94_data = r1[0];
            r1[0] = (v94_data + (v91_data * v92_data));
            float v97_data = s0[7];
            float v99_data = r1[1];
            r1[1] = (v99_data + (v91_data * v97_data));
            float v102_data = s0[13];
            float v104_data = r1[2];
            r1[2] = (v104_data + (v91_data * v102_data));
            float v107_data = s0[19];
            float v109_data = r1[3];
            r1[3] = (v109_data + (v91_data * v107_data));
            float v112_data = s0[25];
            float v114_data = r1[4];
            r1[4] = (v114_data + (v91_data * v112_data));
            float v117_data = s0[31];
            float v119_data = r1[5];
            r1[5] = (v119_data + (v91_data * v117_data));
          }
          if (v7_lead < 12) {
            float v125_data = r0[2];
            float v126_data = s0[2];
            float v128_data = r1[0];
            r1[0] = (v128_data + (v125_data * v126_data));
            float v131_data = s0[8];
            float v133_data = r1[1];
            r1[1] = (v133_data + (v125_data * v131_data));
            float v136_data = s0[14];
            float v138_data = r1[2];
            r1[2] = (v138_data + (v125_data * v136_data));
            float v141_data = s0[20];
            float v143_data = r1[3];
            r1[3] = (v143_data + (v125_data * v141_data));
            float v146_data = s0[26];
            float v148_data = r1[4];
            r1[4] = (v148_data + (v125_data * v146_data));
            float v151_data = s0[32];
            float v153_data = r1[5];
            r1[5] = (v153_data + (v125_data * v151_data));
          }
          if (v7_lead < 12) {
            float v159_data = r0[3];
            float v160_data = s0[3];
            float v162_data = r1[0];
            r1[0] = (v162_data + (v159_data * v160_data));
            float v165_data = s0[9];
            float v167_data = r1[1];
            r1[1] = (v167_data + (v159_data * v165_data));
            float v170_data = s0[15];
            float v172_data = r1[2];
            r1[2] = (v172_data + (v159_data * v170_data));
            float v175_data = s0[21];
            float v177_data = r1[3];
            r1[3] = (v177_data + (v159_data * v175_data));
            float v180_data = s0[27];
            float v182_data = r1[4];
            r1[4] = (v182_data + (v159_data * v180_data));
            float v185_data = s0[33];
            float v187_data = r1[5];
            r1[5] = (v187_data + (v159_data * v185_data));
          }
          if (v7_lead < 12) {
            float v193_data = r0[4];
            float v194_data = s0[4];
            float v196_data = r1[0];
            r1[0] = (v196_data + (v193_data * v194_data));
            float v199_data = s0[10];
            float v201_data = r1[1];
            r1[1] = (v201_data + (v193_data * v199_data));
            float v204_data = s0[16];
            float v206_data = r1[2];
            r1[2] = (v206_data + (v193_data * v204_data));
            float v209_data = s0[22];
            float v211_data = r1[3];
            r1[3] = (v211_data + (v193_data * v209_data));
            float v214_data = s0[28];
            float v216_data = r1[4];
            r1[4] = (v216_data + (v193_data * v214_data));
            float v219_data = s0[34];
            float v221_data = r1[5];
            r1[5] = (v221_data + (v193_data * v219_data));
          }
          if (v7_lead < 12) {
            float v227_data = r0[5];
            float v228_data = s0[5];
            float v230_data = r1[0];
            r1[0] = (v230_data + (v227_data * v228_data));
            float v233_data = s0[11];
            float v235_data = r1[1];
            r1[1] = (v235_data + (v227_data * v233_data));
            float v238_data = s0[17];
            float v240_data = r1[2];
            r1[2] = (v240_data + (v227_data * v238_data));
            float v243_data = s0[23];
            float v245_data = r1[3];
            r1[3] = (v245_data + (v227_data * v243_data));
            float v248_data = s0[29];
            float v250_data = r1[4];
            r1[4] = (v250_data + (v227_data * v248_data));
            float v253_data = s0[35];
            float v255_data = r1[5];
            r1[5] = (v255_data + (v227_data * v253_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v262_i1 = 0; v262_i1 < 6; ++v262_i1) {
              int32_t v263_a = 0 + v262_i1;
              float v265_data = r1[v262_i1];
              int32_t v272_a = v7_lead + (v262_i1 * 12);
              s1[v272_a] = v265_data;
            }
          }
          alignas(16) float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v7_lead < 12) {
            float v279_data = r2[0];
            float v280_data = s1[0];
            float v282_data = ir3[0];
            ir3[0] = (v282_data + (v279_data * v280_data));
            float v285_data = s1[12];
            float v287_data = ir3[1];
            ir3[1] = (v287_data + (v279_data * v285_data));
            float v290_data = s1[24];
            float v292_data = ir3[2];
            ir3[2] = (v292_data + (v279_data * v290_data));
            float v295_data = s1[36];
            float v297_data = ir3[3];
            ir3[3] = (v297_data + (v279_data * v295_data));
            float v300_data = s1[48];
            float v302_data = ir3[4];
            ir3[4] = (v302_data + (v279_data * v300_data));
            float v305_data = s1[60];
            float v307_data = ir3[5];
            ir3[5] = (v307_data + (v279_data * v305_data));
          }
          if (v7_lead < 12) {
            float v313_data = r2[1];
            float v314_data = s1[1];
            float v316_data = ir3[0];
            ir3[0] = (v316_data + (v313_data * v314_data));
            float v319_data = s1[13];
            float v321_data = ir3[1];
            ir3[1] = (v321_data + (v313_data * v319_data));
            float v324_data = s1[25];
            float v326_data = ir3[2];
            ir3[2] = (v326_data + (v313_data * v324_data));
            float v329_data = s1[37];
            float v331_data = ir3[3];
            ir3[3] = (v331_data + (v313_data * v329_data));
            float v334_data = s1[49];
            float v336_data = ir3[4];
            ir3[4] = (v336_data + (v313_data * v334_data));
            float v339_data = s1[61];
            float v341_data = ir3[5];
            ir3[5] = (v341_data + (v313_data * v339_data));
          }
          if (v7_lead < 12) {
            float v347_data = r2[2];
            float v348_data = s1[2];
            float v350_data = ir3[0];
            ir3[0] = (v350_data + (v347_data * v348_data));
            float v353_data = s1[14];
            float v355_data = ir3[1];
            ir3[1] = (v355_data + (v347_data * v353_data));
            float v358_data = s1[26];
            float v360_data = ir3[2];
            ir3[2] = (v360_data + (v347_data * v358_data));
            float v363_data = s1[38];
            float v365_data = ir3[3];
            ir3[3] = (v365_data + (v347_data * v363_data));
            float v368_data = s1[50];
            float v370_data = ir3[4];
            ir3[4] = (v370_data + (v347_data * v368_data));
            float v373_data = s1[62];
            float v375_data = ir3[5];
            ir3[5] = (v375_data + (v347_data * v373_data));
          }
          if (v7_lead < 12) {
            float v381_data = r2[3];
            float v382_data = s1[3];
            float v384_data = ir3[0];
            ir3[0] = (v384_data + (v381_data * v382_data));
            float v387_data = s1[15];
            float v389_data = ir3[1];
            ir3[1] = (v389_data + (v381_data * v387_data));
            float v392_data = s1[27];
            float v394_data = ir3[2];
            ir3[2] = (v394_data + (v381_data * v392_data));
            float v397_data = s1[39];
            float v399_data = ir3[3];
            ir3[3] = (v399_data + (v381_data * v397_data));
            float v402_data = s1[51];
            float v404_data = ir3[4];
            ir3[4] = (v404_data + (v381_data * v402_data));
            float v407_data = s1[63];
            float v409_data = ir3[5];
            ir3[5] = (v409_data + (v381_data * v407_data));
          }
          if (v7_lead < 12) {
            float v415_data = r2[4];
            float v416_data = s1[4];
            float v418_data = ir3[0];
            ir3[0] = (v418_data + (v415_data * v416_data));
            float v421_data = s1[16];
            float v423_data = ir3[1];
            ir3[1] = (v423_data + (v415_data * v421_data));
            float v426_data = s1[28];
            float v428_data = ir3[2];
            ir3[2] = (v428_data + (v415_data * v426_data));
            float v431_data = s1[40];
            float v433_data = ir3[3];
            ir3[3] = (v433_data + (v415_data * v431_data));
            float v436_data = s1[52];
            float v438_data = ir3[4];
            ir3[4] = (v438_data + (v415_data * v436_data));
            float v441_data = s1[64];
            float v443_data = ir3[5];
            ir3[5] = (v443_data + (v415_data * v441_data));
          }
          if (v7_lead < 12) {
            float v449_data = r2[5];
            float v450_data = s1[5];
            float v452_data = ir3[0];
            ir3[0] = (v452_data + (v449_data * v450_data));
            float v455_data = s1[17];
            float v457_data = ir3[1];
            ir3[1] = (v457_data + (v449_data * v455_data));
            float v460_data = s1[29];
            float v462_data = ir3[2];
            ir3[2] = (v462_data + (v449_data * v460_data));
            float v465_data = s1[41];
            float v467_data = ir3[3];
            ir3[3] = (v467_data + (v449_data * v465_data));
            float v470_data = s1[53];
            float v472_data = ir3[4];
            ir3[4] = (v472_data + (v449_data * v470_data));
            float v475_data = s1[65];
            float v477_data = ir3[5];
            ir3[5] = (v477_data + (v449_data * v475_data));
          }
          if (v7_lead < 12) {
            float v483_data = r2[6];
            float v484_data = s1[6];
            float v486_data = ir3[0];
            ir3[0] = (v486_data + (v483_data * v484_data));
            float v489_data = s1[18];
            float v491_data = ir3[1];
            ir3[1] = (v491_data + (v483_data * v489_data));
            float v494_data = s1[30];
            float v496_data = ir3[2];
            ir3[2] = (v496_data + (v483_data * v494_data));
            float v499_data = s1[42];
            float v501_data = ir3[3];
            ir3[3] = (v501_data + (v483_data * v499_data));
            float v504_data = s1[54];
            float v506_data = ir3[4];
            ir3[4] = (v506_data + (v483_data * v504_data));
            float v509_data = s1[66];
            float v511_data = ir3[5];
            ir3[5] = (v511_data + (v483_data * v509_data));
          }
          if (v7_lead < 12) {
            float v517_data = r2[7];
            float v518_data = s1[7];
            float v520_data = ir3[0];
            ir3[0] = (v520_data + (v517_data * v518_data));
            float v523_data = s1[19];
            float v525_data = ir3[1];
            ir3[1] = (v525_data + (v517_data * v523_data));
            float v528_data = s1[31];
            float v530_data = ir3[2];
            ir3[2] = (v530_data + (v517_data * v528_data));
            float v533_data = s1[43];
            float v535_data = ir3[3];
            ir3[3] = (v535_data + (v517_data * v533_data));
            float v538_data = s1[55];
            float v540_data = ir3[4];
            ir3[4] = (v540_data + (v517_data * v538_data));
            float v543_data = s1[67];
            float v545_data = ir3[5];
            ir3[5] = (v545_data + (v517_data * v543_data));
          }
          if (v7_lead < 12) {
            float v551_data = r2[8];
            float v552_data = s1[8];
            float v554_data = ir3[0];
            ir3[0] = (v554_data + (v551_data * v552_data));
            float v557_data = s1[20];
            float v559_data = ir3[1];
            ir3[1] = (v559_data + (v551_data * v557_data));
            float v562_data = s1[32];
            float v564_data = ir3[2];
            ir3[2] = (v564_data + (v551_data * v562_data));
            float v567_data = s1[44];
            float v569_data = ir3[3];
            ir3[3] = (v569_data + (v551_data * v567_data));
            float v572_data = s1[56];
            float v574_data = ir3[4];
            ir3[4] = (v574_data + (v551_data * v572_data));
            float v577_data = s1[68];
            float v579_data = ir3[5];
            ir3[5] = (v579_data + (v551_data * v577_data));
          }
          if (v7_lead < 12) {
            float v585_data = r2[9];
            float v586_data = s1[9];
            float v588_data = ir3[0];
            ir3[0] = (v588_data + (v585_data * v586_data));
            float v591_data = s1[21];
            float v593_data = ir3[1];
            ir3[1] = (v593_data + (v585_data * v591_data));
            float v596_data = s1[33];
            float v598_data = ir3[2];
            ir3[2] = (v598_data + (v585_data * v596_data));
            float v601_data = s1[45];
            float v603_data = ir3[3];
            ir3[3] = (v603_data + (v585_data * v601_data));
            float v606_data = s1[57];
            float v608_data = ir3[4];
            ir3[4] = (v608_data + (v585_data * v606_data));
            float v611_data = s1[69];
            float v613_data = ir3[5];
            ir3[5] = (v613_data + (v585_data * v611_data));
          }
          if (v7_lead < 12) {
            float v619_data = r2[10];
            float v620_data = s1[10];
            float v622_data = ir3[0];
            ir3[0] = (v622_data + (v619_data * v620_data));
            float v625_data = s1[22];
            float v627_data = ir3[1];
            ir3[1] = (v627_data + (v619_data * v625_data));
            float v630_data = s1[34];
            float v632_data = ir3[2];
            ir3[2] = (v632_data + (v619_data * v630_data));
            float v635_data = s1[46];
            float v637_data = ir3[3];
            ir3[3] = (v637_data + (v619_data * v635_data));
            float v640_data = s1[58];
            float v642_data = ir3[4];
            ir3[4] = (v642_data + (v619_data * v640_data));
            float v645_data = s1[70];
            float v647_data = ir3[5];
            ir3[5] = (v647_data + (v619_data * v645_data));
          }
          if (v7_lead < 12) {
            float v653_data = r2[11];
            float v654_data = s1[11];
            float v656_data = ir3[0];
            ir3[0] = (v656_data + (v653_data * v654_data));
            float v659_data = s1[23];
            float v661_data = ir3[1];
            ir3[1] = (v661_data + (v653_data * v659_data));
            float v664_data = s1[35];
            float v666_data = ir3[2];
            ir3[2] = (v666_data + (v653_data * v664_data));
            float v669_data = s1[47];
            float v671_data = ir3[3];
            ir3[3] = (v671_data + (v653_data * v669_data));
            float v674_data = s1[59];
            float v676_data = ir3[4];
            ir3[4] = (v676_data + (v653_data * v674_data));
            float v679_data = s1[71];
            float v681_data = ir3[5];
            ir3[5] = (v681_data + (v653_data * v679_data));
          }
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v687_n1 = 0; v687_n1 < 6; ++v687_n1) {
              int32_t v688_a = 0 + v687_n1;
              float v690_data = ir3[v687_n1];
              r3[v687_n1] = v690_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v696_i1 = 0; v696_i1 < 6; ++v696_i1) {
              int32_t v697_a = 0 + v696_i1;
              float v699_data = r3[v696_i1];
              glb_m2[(v7_lead + (v696_i1 * 12))] = v699_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

