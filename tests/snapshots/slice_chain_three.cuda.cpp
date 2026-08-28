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
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v8_lead = threadIdx.x % 16;
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 6; ++v10_i1) {
              int32_t v16_a = v10_i1 * 12;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __ldcg(&glb_m0[(v8_lead + v16_a)]);
              r0[v10_i1] = v25_data;
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
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              int32_t v42_a = v36_i1 * 12;
              int32_t v43_a = v8_lead + v42_a;
              float v51_data = __ldcg(&glb_m3[(v8_lead + v42_a)]);
              r2[v36_i1] = v51_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v8_lead < 12) {
            float v58_data = r0[0];
            float v59_data = s0[0];
            float v61_data = r1[0];
            r1[0] = (v61_data + (v58_data * v59_data));
            float v64_data = s0[6];
            float v66_data = r1[1];
            r1[1] = (v66_data + (v58_data * v64_data));
            float v69_data = s0[12];
            float v71_data = r1[2];
            r1[2] = (v71_data + (v58_data * v69_data));
            float v74_data = s0[18];
            float v76_data = r1[3];
            r1[3] = (v76_data + (v58_data * v74_data));
            float v79_data = s0[24];
            float v81_data = r1[4];
            r1[4] = (v81_data + (v58_data * v79_data));
            float v84_data = s0[30];
            float v86_data = r1[5];
            r1[5] = (v86_data + (v58_data * v84_data));
          }
          if (v8_lead < 12) {
            float v92_data = r0[1];
            float v93_data = s0[1];
            float v95_data = r1[0];
            r1[0] = (v95_data + (v92_data * v93_data));
            float v98_data = s0[7];
            float v100_data = r1[1];
            r1[1] = (v100_data + (v92_data * v98_data));
            float v103_data = s0[13];
            float v105_data = r1[2];
            r1[2] = (v105_data + (v92_data * v103_data));
            float v108_data = s0[19];
            float v110_data = r1[3];
            r1[3] = (v110_data + (v92_data * v108_data));
            float v113_data = s0[25];
            float v115_data = r1[4];
            r1[4] = (v115_data + (v92_data * v113_data));
            float v118_data = s0[31];
            float v120_data = r1[5];
            r1[5] = (v120_data + (v92_data * v118_data));
          }
          if (v8_lead < 12) {
            float v126_data = r0[2];
            float v127_data = s0[2];
            float v129_data = r1[0];
            r1[0] = (v129_data + (v126_data * v127_data));
            float v132_data = s0[8];
            float v134_data = r1[1];
            r1[1] = (v134_data + (v126_data * v132_data));
            float v137_data = s0[14];
            float v139_data = r1[2];
            r1[2] = (v139_data + (v126_data * v137_data));
            float v142_data = s0[20];
            float v144_data = r1[3];
            r1[3] = (v144_data + (v126_data * v142_data));
            float v147_data = s0[26];
            float v149_data = r1[4];
            r1[4] = (v149_data + (v126_data * v147_data));
            float v152_data = s0[32];
            float v154_data = r1[5];
            r1[5] = (v154_data + (v126_data * v152_data));
          }
          if (v8_lead < 12) {
            float v160_data = r0[3];
            float v161_data = s0[3];
            float v163_data = r1[0];
            r1[0] = (v163_data + (v160_data * v161_data));
            float v166_data = s0[9];
            float v168_data = r1[1];
            r1[1] = (v168_data + (v160_data * v166_data));
            float v171_data = s0[15];
            float v173_data = r1[2];
            r1[2] = (v173_data + (v160_data * v171_data));
            float v176_data = s0[21];
            float v178_data = r1[3];
            r1[3] = (v178_data + (v160_data * v176_data));
            float v181_data = s0[27];
            float v183_data = r1[4];
            r1[4] = (v183_data + (v160_data * v181_data));
            float v186_data = s0[33];
            float v188_data = r1[5];
            r1[5] = (v188_data + (v160_data * v186_data));
          }
          if (v8_lead < 12) {
            float v194_data = r0[4];
            float v195_data = s0[4];
            float v197_data = r1[0];
            r1[0] = (v197_data + (v194_data * v195_data));
            float v200_data = s0[10];
            float v202_data = r1[1];
            r1[1] = (v202_data + (v194_data * v200_data));
            float v205_data = s0[16];
            float v207_data = r1[2];
            r1[2] = (v207_data + (v194_data * v205_data));
            float v210_data = s0[22];
            float v212_data = r1[3];
            r1[3] = (v212_data + (v194_data * v210_data));
            float v215_data = s0[28];
            float v217_data = r1[4];
            r1[4] = (v217_data + (v194_data * v215_data));
            float v220_data = s0[34];
            float v222_data = r1[5];
            r1[5] = (v222_data + (v194_data * v220_data));
          }
          if (v8_lead < 12) {
            float v228_data = r0[5];
            float v229_data = s0[5];
            float v231_data = r1[0];
            r1[0] = (v231_data + (v228_data * v229_data));
            float v234_data = s0[11];
            float v236_data = r1[1];
            r1[1] = (v236_data + (v228_data * v234_data));
            float v239_data = s0[17];
            float v241_data = r1[2];
            r1[2] = (v241_data + (v228_data * v239_data));
            float v244_data = s0[23];
            float v246_data = r1[3];
            r1[3] = (v246_data + (v228_data * v244_data));
            float v249_data = s0[29];
            float v251_data = r1[4];
            r1[4] = (v251_data + (v228_data * v249_data));
            float v254_data = s0[35];
            float v256_data = r1[5];
            r1[5] = (v256_data + (v228_data * v254_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v263_i1 = 0; v263_i1 < 6; ++v263_i1) {
              int32_t v264_a = 0 + v263_i1;
              float v266_data = r1[v263_i1];
              s1[(v8_lead + (v263_i1 * 12))] = v266_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v8_lead < 12) {
            float v280_data = r2[0];
            float v281_data = s1[0];
            float v283_data = ir3[0];
            ir3[0] = (v283_data + (v280_data * v281_data));
            float v286_data = s1[12];
            float v288_data = ir3[1];
            ir3[1] = (v288_data + (v280_data * v286_data));
            float v291_data = s1[24];
            float v293_data = ir3[2];
            ir3[2] = (v293_data + (v280_data * v291_data));
            float v296_data = s1[36];
            float v298_data = ir3[3];
            ir3[3] = (v298_data + (v280_data * v296_data));
            float v301_data = s1[48];
            float v303_data = ir3[4];
            ir3[4] = (v303_data + (v280_data * v301_data));
            float v306_data = s1[60];
            float v308_data = ir3[5];
            ir3[5] = (v308_data + (v280_data * v306_data));
          }
          if (v8_lead < 12) {
            float v314_data = r2[1];
            float v315_data = s1[1];
            float v317_data = ir3[0];
            ir3[0] = (v317_data + (v314_data * v315_data));
            float v320_data = s1[13];
            float v322_data = ir3[1];
            ir3[1] = (v322_data + (v314_data * v320_data));
            float v325_data = s1[25];
            float v327_data = ir3[2];
            ir3[2] = (v327_data + (v314_data * v325_data));
            float v330_data = s1[37];
            float v332_data = ir3[3];
            ir3[3] = (v332_data + (v314_data * v330_data));
            float v335_data = s1[49];
            float v337_data = ir3[4];
            ir3[4] = (v337_data + (v314_data * v335_data));
            float v340_data = s1[61];
            float v342_data = ir3[5];
            ir3[5] = (v342_data + (v314_data * v340_data));
          }
          if (v8_lead < 12) {
            float v348_data = r2[2];
            float v349_data = s1[2];
            float v351_data = ir3[0];
            ir3[0] = (v351_data + (v348_data * v349_data));
            float v354_data = s1[14];
            float v356_data = ir3[1];
            ir3[1] = (v356_data + (v348_data * v354_data));
            float v359_data = s1[26];
            float v361_data = ir3[2];
            ir3[2] = (v361_data + (v348_data * v359_data));
            float v364_data = s1[38];
            float v366_data = ir3[3];
            ir3[3] = (v366_data + (v348_data * v364_data));
            float v369_data = s1[50];
            float v371_data = ir3[4];
            ir3[4] = (v371_data + (v348_data * v369_data));
            float v374_data = s1[62];
            float v376_data = ir3[5];
            ir3[5] = (v376_data + (v348_data * v374_data));
          }
          if (v8_lead < 12) {
            float v382_data = r2[3];
            float v383_data = s1[3];
            float v385_data = ir3[0];
            ir3[0] = (v385_data + (v382_data * v383_data));
            float v388_data = s1[15];
            float v390_data = ir3[1];
            ir3[1] = (v390_data + (v382_data * v388_data));
            float v393_data = s1[27];
            float v395_data = ir3[2];
            ir3[2] = (v395_data + (v382_data * v393_data));
            float v398_data = s1[39];
            float v400_data = ir3[3];
            ir3[3] = (v400_data + (v382_data * v398_data));
            float v403_data = s1[51];
            float v405_data = ir3[4];
            ir3[4] = (v405_data + (v382_data * v403_data));
            float v408_data = s1[63];
            float v410_data = ir3[5];
            ir3[5] = (v410_data + (v382_data * v408_data));
          }
          if (v8_lead < 12) {
            float v416_data = r2[4];
            float v417_data = s1[4];
            float v419_data = ir3[0];
            ir3[0] = (v419_data + (v416_data * v417_data));
            float v422_data = s1[16];
            float v424_data = ir3[1];
            ir3[1] = (v424_data + (v416_data * v422_data));
            float v427_data = s1[28];
            float v429_data = ir3[2];
            ir3[2] = (v429_data + (v416_data * v427_data));
            float v432_data = s1[40];
            float v434_data = ir3[3];
            ir3[3] = (v434_data + (v416_data * v432_data));
            float v437_data = s1[52];
            float v439_data = ir3[4];
            ir3[4] = (v439_data + (v416_data * v437_data));
            float v442_data = s1[64];
            float v444_data = ir3[5];
            ir3[5] = (v444_data + (v416_data * v442_data));
          }
          if (v8_lead < 12) {
            float v450_data = r2[5];
            float v451_data = s1[5];
            float v453_data = ir3[0];
            ir3[0] = (v453_data + (v450_data * v451_data));
            float v456_data = s1[17];
            float v458_data = ir3[1];
            ir3[1] = (v458_data + (v450_data * v456_data));
            float v461_data = s1[29];
            float v463_data = ir3[2];
            ir3[2] = (v463_data + (v450_data * v461_data));
            float v466_data = s1[41];
            float v468_data = ir3[3];
            ir3[3] = (v468_data + (v450_data * v466_data));
            float v471_data = s1[53];
            float v473_data = ir3[4];
            ir3[4] = (v473_data + (v450_data * v471_data));
            float v476_data = s1[65];
            float v478_data = ir3[5];
            ir3[5] = (v478_data + (v450_data * v476_data));
          }
          if (v8_lead < 12) {
            float v484_data = r2[6];
            float v485_data = s1[6];
            float v487_data = ir3[0];
            ir3[0] = (v487_data + (v484_data * v485_data));
            float v490_data = s1[18];
            float v492_data = ir3[1];
            ir3[1] = (v492_data + (v484_data * v490_data));
            float v495_data = s1[30];
            float v497_data = ir3[2];
            ir3[2] = (v497_data + (v484_data * v495_data));
            float v500_data = s1[42];
            float v502_data = ir3[3];
            ir3[3] = (v502_data + (v484_data * v500_data));
            float v505_data = s1[54];
            float v507_data = ir3[4];
            ir3[4] = (v507_data + (v484_data * v505_data));
            float v510_data = s1[66];
            float v512_data = ir3[5];
            ir3[5] = (v512_data + (v484_data * v510_data));
          }
          if (v8_lead < 12) {
            float v518_data = r2[7];
            float v519_data = s1[7];
            float v521_data = ir3[0];
            ir3[0] = (v521_data + (v518_data * v519_data));
            float v524_data = s1[19];
            float v526_data = ir3[1];
            ir3[1] = (v526_data + (v518_data * v524_data));
            float v529_data = s1[31];
            float v531_data = ir3[2];
            ir3[2] = (v531_data + (v518_data * v529_data));
            float v534_data = s1[43];
            float v536_data = ir3[3];
            ir3[3] = (v536_data + (v518_data * v534_data));
            float v539_data = s1[55];
            float v541_data = ir3[4];
            ir3[4] = (v541_data + (v518_data * v539_data));
            float v544_data = s1[67];
            float v546_data = ir3[5];
            ir3[5] = (v546_data + (v518_data * v544_data));
          }
          if (v8_lead < 12) {
            float v552_data = r2[8];
            float v553_data = s1[8];
            float v555_data = ir3[0];
            ir3[0] = (v555_data + (v552_data * v553_data));
            float v558_data = s1[20];
            float v560_data = ir3[1];
            ir3[1] = (v560_data + (v552_data * v558_data));
            float v563_data = s1[32];
            float v565_data = ir3[2];
            ir3[2] = (v565_data + (v552_data * v563_data));
            float v568_data = s1[44];
            float v570_data = ir3[3];
            ir3[3] = (v570_data + (v552_data * v568_data));
            float v573_data = s1[56];
            float v575_data = ir3[4];
            ir3[4] = (v575_data + (v552_data * v573_data));
            float v578_data = s1[68];
            float v580_data = ir3[5];
            ir3[5] = (v580_data + (v552_data * v578_data));
          }
          if (v8_lead < 12) {
            float v586_data = r2[9];
            float v587_data = s1[9];
            float v589_data = ir3[0];
            ir3[0] = (v589_data + (v586_data * v587_data));
            float v592_data = s1[21];
            float v594_data = ir3[1];
            ir3[1] = (v594_data + (v586_data * v592_data));
            float v597_data = s1[33];
            float v599_data = ir3[2];
            ir3[2] = (v599_data + (v586_data * v597_data));
            float v602_data = s1[45];
            float v604_data = ir3[3];
            ir3[3] = (v604_data + (v586_data * v602_data));
            float v607_data = s1[57];
            float v609_data = ir3[4];
            ir3[4] = (v609_data + (v586_data * v607_data));
            float v612_data = s1[69];
            float v614_data = ir3[5];
            ir3[5] = (v614_data + (v586_data * v612_data));
          }
          if (v8_lead < 12) {
            float v620_data = r2[10];
            float v621_data = s1[10];
            float v623_data = ir3[0];
            ir3[0] = (v623_data + (v620_data * v621_data));
            float v626_data = s1[22];
            float v628_data = ir3[1];
            ir3[1] = (v628_data + (v620_data * v626_data));
            float v631_data = s1[34];
            float v633_data = ir3[2];
            ir3[2] = (v633_data + (v620_data * v631_data));
            float v636_data = s1[46];
            float v638_data = ir3[3];
            ir3[3] = (v638_data + (v620_data * v636_data));
            float v641_data = s1[58];
            float v643_data = ir3[4];
            ir3[4] = (v643_data + (v620_data * v641_data));
            float v646_data = s1[70];
            float v648_data = ir3[5];
            ir3[5] = (v648_data + (v620_data * v646_data));
          }
          if (v8_lead < 12) {
            float v654_data = r2[11];
            float v655_data = s1[11];
            float v657_data = ir3[0];
            ir3[0] = (v657_data + (v654_data * v655_data));
            float v660_data = s1[23];
            float v662_data = ir3[1];
            ir3[1] = (v662_data + (v654_data * v660_data));
            float v665_data = s1[35];
            float v667_data = ir3[2];
            ir3[2] = (v667_data + (v654_data * v665_data));
            float v670_data = s1[47];
            float v672_data = ir3[3];
            ir3[3] = (v672_data + (v654_data * v670_data));
            float v675_data = s1[59];
            float v677_data = ir3[4];
            ir3[4] = (v677_data + (v654_data * v675_data));
            float v680_data = s1[71];
            float v682_data = ir3[5];
            ir3[5] = (v682_data + (v654_data * v680_data));
          }
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v688_n1 = 0; v688_n1 < 6; ++v688_n1) {
              int32_t v689_a = 0 + v688_n1;
              float v691_data = ir3[v688_n1];
              r3[v688_n1] = v691_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v697_i1 = 0; v697_i1 < 6; ++v697_i1) {
              int32_t v698_a = 0 + v697_i1;
              float v700_data = r3[v697_i1];
              glb_m2[(v8_lead + (v697_i1 * 12))] = v700_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

