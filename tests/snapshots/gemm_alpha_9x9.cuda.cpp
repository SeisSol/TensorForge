// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08a27dccde, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08a27dccde, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08a27dccde<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 9; ++v9_i1) {
              int32_t v15_a = v9_i1 * 9;
              int32_t v16_a = v7_lead + v15_a;
              float v24_data = __ldcg(&glb_m1[(v7_lead + v15_a)]);
              r0[v9_i1] = v24_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          if (v7_lead < 9) {
            float v35_data = r0[0];
            float v36_data = s0[0];
            float v38_data = ir1[0];
            ir1[0] = (v38_data + (v35_data * v36_data));
            float v41_data = s0[9];
            float v43_data = ir1[1];
            ir1[1] = (v43_data + (v35_data * v41_data));
            float v46_data = s0[18];
            float v48_data = ir1[2];
            ir1[2] = (v48_data + (v35_data * v46_data));
            float v51_data = s0[27];
            float v53_data = ir1[3];
            ir1[3] = (v53_data + (v35_data * v51_data));
            float v56_data = s0[36];
            float v58_data = ir1[4];
            ir1[4] = (v58_data + (v35_data * v56_data));
            float v61_data = s0[45];
            float v63_data = ir1[5];
            ir1[5] = (v63_data + (v35_data * v61_data));
            float v66_data = s0[54];
            float v68_data = ir1[6];
            ir1[6] = (v68_data + (v35_data * v66_data));
            float v71_data = s0[63];
            float v73_data = ir1[7];
            ir1[7] = (v73_data + (v35_data * v71_data));
            float v76_data = s0[72];
            float v78_data = ir1[8];
            ir1[8] = (v78_data + (v35_data * v76_data));
          }
          if (v7_lead < 9) {
            float v84_data = r0[1];
            float v85_data = s0[1];
            float v87_data = ir1[0];
            ir1[0] = (v87_data + (v84_data * v85_data));
            float v90_data = s0[10];
            float v92_data = ir1[1];
            ir1[1] = (v92_data + (v84_data * v90_data));
            float v95_data = s0[19];
            float v97_data = ir1[2];
            ir1[2] = (v97_data + (v84_data * v95_data));
            float v100_data = s0[28];
            float v102_data = ir1[3];
            ir1[3] = (v102_data + (v84_data * v100_data));
            float v105_data = s0[37];
            float v107_data = ir1[4];
            ir1[4] = (v107_data + (v84_data * v105_data));
            float v110_data = s0[46];
            float v112_data = ir1[5];
            ir1[5] = (v112_data + (v84_data * v110_data));
            float v115_data = s0[55];
            float v117_data = ir1[6];
            ir1[6] = (v117_data + (v84_data * v115_data));
            float v120_data = s0[64];
            float v122_data = ir1[7];
            ir1[7] = (v122_data + (v84_data * v120_data));
            float v125_data = s0[73];
            float v127_data = ir1[8];
            ir1[8] = (v127_data + (v84_data * v125_data));
          }
          if (v7_lead < 9) {
            float v133_data = r0[2];
            float v134_data = s0[2];
            float v136_data = ir1[0];
            ir1[0] = (v136_data + (v133_data * v134_data));
            float v139_data = s0[11];
            float v141_data = ir1[1];
            ir1[1] = (v141_data + (v133_data * v139_data));
            float v144_data = s0[20];
            float v146_data = ir1[2];
            ir1[2] = (v146_data + (v133_data * v144_data));
            float v149_data = s0[29];
            float v151_data = ir1[3];
            ir1[3] = (v151_data + (v133_data * v149_data));
            float v154_data = s0[38];
            float v156_data = ir1[4];
            ir1[4] = (v156_data + (v133_data * v154_data));
            float v159_data = s0[47];
            float v161_data = ir1[5];
            ir1[5] = (v161_data + (v133_data * v159_data));
            float v164_data = s0[56];
            float v166_data = ir1[6];
            ir1[6] = (v166_data + (v133_data * v164_data));
            float v169_data = s0[65];
            float v171_data = ir1[7];
            ir1[7] = (v171_data + (v133_data * v169_data));
            float v174_data = s0[74];
            float v176_data = ir1[8];
            ir1[8] = (v176_data + (v133_data * v174_data));
          }
          if (v7_lead < 9) {
            float v182_data = r0[3];
            float v183_data = s0[3];
            float v185_data = ir1[0];
            ir1[0] = (v185_data + (v182_data * v183_data));
            float v188_data = s0[12];
            float v190_data = ir1[1];
            ir1[1] = (v190_data + (v182_data * v188_data));
            float v193_data = s0[21];
            float v195_data = ir1[2];
            ir1[2] = (v195_data + (v182_data * v193_data));
            float v198_data = s0[30];
            float v200_data = ir1[3];
            ir1[3] = (v200_data + (v182_data * v198_data));
            float v203_data = s0[39];
            float v205_data = ir1[4];
            ir1[4] = (v205_data + (v182_data * v203_data));
            float v208_data = s0[48];
            float v210_data = ir1[5];
            ir1[5] = (v210_data + (v182_data * v208_data));
            float v213_data = s0[57];
            float v215_data = ir1[6];
            ir1[6] = (v215_data + (v182_data * v213_data));
            float v218_data = s0[66];
            float v220_data = ir1[7];
            ir1[7] = (v220_data + (v182_data * v218_data));
            float v223_data = s0[75];
            float v225_data = ir1[8];
            ir1[8] = (v225_data + (v182_data * v223_data));
          }
          if (v7_lead < 9) {
            float v231_data = r0[4];
            float v232_data = s0[4];
            float v234_data = ir1[0];
            ir1[0] = (v234_data + (v231_data * v232_data));
            float v237_data = s0[13];
            float v239_data = ir1[1];
            ir1[1] = (v239_data + (v231_data * v237_data));
            float v242_data = s0[22];
            float v244_data = ir1[2];
            ir1[2] = (v244_data + (v231_data * v242_data));
            float v247_data = s0[31];
            float v249_data = ir1[3];
            ir1[3] = (v249_data + (v231_data * v247_data));
            float v252_data = s0[40];
            float v254_data = ir1[4];
            ir1[4] = (v254_data + (v231_data * v252_data));
            float v257_data = s0[49];
            float v259_data = ir1[5];
            ir1[5] = (v259_data + (v231_data * v257_data));
            float v262_data = s0[58];
            float v264_data = ir1[6];
            ir1[6] = (v264_data + (v231_data * v262_data));
            float v267_data = s0[67];
            float v269_data = ir1[7];
            ir1[7] = (v269_data + (v231_data * v267_data));
            float v272_data = s0[76];
            float v274_data = ir1[8];
            ir1[8] = (v274_data + (v231_data * v272_data));
          }
          if (v7_lead < 9) {
            float v280_data = r0[5];
            float v281_data = s0[5];
            float v283_data = ir1[0];
            ir1[0] = (v283_data + (v280_data * v281_data));
            float v286_data = s0[14];
            float v288_data = ir1[1];
            ir1[1] = (v288_data + (v280_data * v286_data));
            float v291_data = s0[23];
            float v293_data = ir1[2];
            ir1[2] = (v293_data + (v280_data * v291_data));
            float v296_data = s0[32];
            float v298_data = ir1[3];
            ir1[3] = (v298_data + (v280_data * v296_data));
            float v301_data = s0[41];
            float v303_data = ir1[4];
            ir1[4] = (v303_data + (v280_data * v301_data));
            float v306_data = s0[50];
            float v308_data = ir1[5];
            ir1[5] = (v308_data + (v280_data * v306_data));
            float v311_data = s0[59];
            float v313_data = ir1[6];
            ir1[6] = (v313_data + (v280_data * v311_data));
            float v316_data = s0[68];
            float v318_data = ir1[7];
            ir1[7] = (v318_data + (v280_data * v316_data));
            float v321_data = s0[77];
            float v323_data = ir1[8];
            ir1[8] = (v323_data + (v280_data * v321_data));
          }
          if (v7_lead < 9) {
            float v329_data = r0[6];
            float v330_data = s0[6];
            float v332_data = ir1[0];
            ir1[0] = (v332_data + (v329_data * v330_data));
            float v335_data = s0[15];
            float v337_data = ir1[1];
            ir1[1] = (v337_data + (v329_data * v335_data));
            float v340_data = s0[24];
            float v342_data = ir1[2];
            ir1[2] = (v342_data + (v329_data * v340_data));
            float v345_data = s0[33];
            float v347_data = ir1[3];
            ir1[3] = (v347_data + (v329_data * v345_data));
            float v350_data = s0[42];
            float v352_data = ir1[4];
            ir1[4] = (v352_data + (v329_data * v350_data));
            float v355_data = s0[51];
            float v357_data = ir1[5];
            ir1[5] = (v357_data + (v329_data * v355_data));
            float v360_data = s0[60];
            float v362_data = ir1[6];
            ir1[6] = (v362_data + (v329_data * v360_data));
            float v365_data = s0[69];
            float v367_data = ir1[7];
            ir1[7] = (v367_data + (v329_data * v365_data));
            float v370_data = s0[78];
            float v372_data = ir1[8];
            ir1[8] = (v372_data + (v329_data * v370_data));
          }
          if (v7_lead < 9) {
            float v378_data = r0[7];
            float v379_data = s0[7];
            float v381_data = ir1[0];
            ir1[0] = (v381_data + (v378_data * v379_data));
            float v384_data = s0[16];
            float v386_data = ir1[1];
            ir1[1] = (v386_data + (v378_data * v384_data));
            float v389_data = s0[25];
            float v391_data = ir1[2];
            ir1[2] = (v391_data + (v378_data * v389_data));
            float v394_data = s0[34];
            float v396_data = ir1[3];
            ir1[3] = (v396_data + (v378_data * v394_data));
            float v399_data = s0[43];
            float v401_data = ir1[4];
            ir1[4] = (v401_data + (v378_data * v399_data));
            float v404_data = s0[52];
            float v406_data = ir1[5];
            ir1[5] = (v406_data + (v378_data * v404_data));
            float v409_data = s0[61];
            float v411_data = ir1[6];
            ir1[6] = (v411_data + (v378_data * v409_data));
            float v414_data = s0[70];
            float v416_data = ir1[7];
            ir1[7] = (v416_data + (v378_data * v414_data));
            float v419_data = s0[79];
            float v421_data = ir1[8];
            ir1[8] = (v421_data + (v378_data * v419_data));
          }
          if (v7_lead < 9) {
            float v427_data = r0[8];
            float v428_data = s0[8];
            float v430_data = ir1[0];
            ir1[0] = (v430_data + (v427_data * v428_data));
            float v433_data = s0[17];
            float v435_data = ir1[1];
            ir1[1] = (v435_data + (v427_data * v433_data));
            float v438_data = s0[26];
            float v440_data = ir1[2];
            ir1[2] = (v440_data + (v427_data * v438_data));
            float v443_data = s0[35];
            float v445_data = ir1[3];
            ir1[3] = (v445_data + (v427_data * v443_data));
            float v448_data = s0[44];
            float v450_data = ir1[4];
            ir1[4] = (v450_data + (v427_data * v448_data));
            float v453_data = s0[53];
            float v455_data = ir1[5];
            ir1[5] = (v455_data + (v427_data * v453_data));
            float v458_data = s0[62];
            float v460_data = ir1[6];
            ir1[6] = (v460_data + (v427_data * v458_data));
            float v463_data = s0[71];
            float v465_data = ir1[7];
            ir1[7] = (v465_data + (v427_data * v463_data));
            float v468_data = s0[80];
            float v470_data = ir1[8];
            ir1[8] = (v470_data + (v427_data * v468_data));
          }
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v477_n1 = 0; v477_n1 < 9; ++v477_n1) {
              int32_t v478_a = 0 + v477_n1;
              float v480_data = ir1[v477_n1];
              r1[v477_n1] = (v480_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v487_i1 = 0; v487_i1 < 9; ++v487_i1) {
              int32_t v488_a = 0 + v487_i1;
              float v490_data = r1[v487_i1];
              glb_m0[(v7_lead + (v487_i1 * 9))] = v490_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

