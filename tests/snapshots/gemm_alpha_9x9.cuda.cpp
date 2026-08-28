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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 9; ++v12_i1) {
              int32_t v18_a = v12_i1 * 9;
              int32_t v19_a = v10_lead + v18_a;
              float v27_data = __ldcg(&glb_m1[(v10_lead + v18_a)]);
              r0[v12_i1] = v27_data;
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
          if (v10_lead < 9) {
            float v38_data = r0[0];
            float v39_data = s0[0];
            float v41_data = ir1[0];
            ir1[0] = (v41_data + (v38_data * v39_data));
            float v44_data = s0[9];
            float v46_data = ir1[1];
            ir1[1] = (v46_data + (v38_data * v44_data));
            float v49_data = s0[18];
            float v51_data = ir1[2];
            ir1[2] = (v51_data + (v38_data * v49_data));
            float v54_data = s0[27];
            float v56_data = ir1[3];
            ir1[3] = (v56_data + (v38_data * v54_data));
            float v59_data = s0[36];
            float v61_data = ir1[4];
            ir1[4] = (v61_data + (v38_data * v59_data));
            float v64_data = s0[45];
            float v66_data = ir1[5];
            ir1[5] = (v66_data + (v38_data * v64_data));
            float v69_data = s0[54];
            float v71_data = ir1[6];
            ir1[6] = (v71_data + (v38_data * v69_data));
            float v74_data = s0[63];
            float v76_data = ir1[7];
            ir1[7] = (v76_data + (v38_data * v74_data));
            float v79_data = s0[72];
            float v81_data = ir1[8];
            ir1[8] = (v81_data + (v38_data * v79_data));
          }
          if (v10_lead < 9) {
            float v87_data = r0[1];
            float v88_data = s0[1];
            float v90_data = ir1[0];
            ir1[0] = (v90_data + (v87_data * v88_data));
            float v93_data = s0[10];
            float v95_data = ir1[1];
            ir1[1] = (v95_data + (v87_data * v93_data));
            float v98_data = s0[19];
            float v100_data = ir1[2];
            ir1[2] = (v100_data + (v87_data * v98_data));
            float v103_data = s0[28];
            float v105_data = ir1[3];
            ir1[3] = (v105_data + (v87_data * v103_data));
            float v108_data = s0[37];
            float v110_data = ir1[4];
            ir1[4] = (v110_data + (v87_data * v108_data));
            float v113_data = s0[46];
            float v115_data = ir1[5];
            ir1[5] = (v115_data + (v87_data * v113_data));
            float v118_data = s0[55];
            float v120_data = ir1[6];
            ir1[6] = (v120_data + (v87_data * v118_data));
            float v123_data = s0[64];
            float v125_data = ir1[7];
            ir1[7] = (v125_data + (v87_data * v123_data));
            float v128_data = s0[73];
            float v130_data = ir1[8];
            ir1[8] = (v130_data + (v87_data * v128_data));
          }
          if (v10_lead < 9) {
            float v136_data = r0[2];
            float v137_data = s0[2];
            float v139_data = ir1[0];
            ir1[0] = (v139_data + (v136_data * v137_data));
            float v142_data = s0[11];
            float v144_data = ir1[1];
            ir1[1] = (v144_data + (v136_data * v142_data));
            float v147_data = s0[20];
            float v149_data = ir1[2];
            ir1[2] = (v149_data + (v136_data * v147_data));
            float v152_data = s0[29];
            float v154_data = ir1[3];
            ir1[3] = (v154_data + (v136_data * v152_data));
            float v157_data = s0[38];
            float v159_data = ir1[4];
            ir1[4] = (v159_data + (v136_data * v157_data));
            float v162_data = s0[47];
            float v164_data = ir1[5];
            ir1[5] = (v164_data + (v136_data * v162_data));
            float v167_data = s0[56];
            float v169_data = ir1[6];
            ir1[6] = (v169_data + (v136_data * v167_data));
            float v172_data = s0[65];
            float v174_data = ir1[7];
            ir1[7] = (v174_data + (v136_data * v172_data));
            float v177_data = s0[74];
            float v179_data = ir1[8];
            ir1[8] = (v179_data + (v136_data * v177_data));
          }
          if (v10_lead < 9) {
            float v185_data = r0[3];
            float v186_data = s0[3];
            float v188_data = ir1[0];
            ir1[0] = (v188_data + (v185_data * v186_data));
            float v191_data = s0[12];
            float v193_data = ir1[1];
            ir1[1] = (v193_data + (v185_data * v191_data));
            float v196_data = s0[21];
            float v198_data = ir1[2];
            ir1[2] = (v198_data + (v185_data * v196_data));
            float v201_data = s0[30];
            float v203_data = ir1[3];
            ir1[3] = (v203_data + (v185_data * v201_data));
            float v206_data = s0[39];
            float v208_data = ir1[4];
            ir1[4] = (v208_data + (v185_data * v206_data));
            float v211_data = s0[48];
            float v213_data = ir1[5];
            ir1[5] = (v213_data + (v185_data * v211_data));
            float v216_data = s0[57];
            float v218_data = ir1[6];
            ir1[6] = (v218_data + (v185_data * v216_data));
            float v221_data = s0[66];
            float v223_data = ir1[7];
            ir1[7] = (v223_data + (v185_data * v221_data));
            float v226_data = s0[75];
            float v228_data = ir1[8];
            ir1[8] = (v228_data + (v185_data * v226_data));
          }
          if (v10_lead < 9) {
            float v234_data = r0[4];
            float v235_data = s0[4];
            float v237_data = ir1[0];
            ir1[0] = (v237_data + (v234_data * v235_data));
            float v240_data = s0[13];
            float v242_data = ir1[1];
            ir1[1] = (v242_data + (v234_data * v240_data));
            float v245_data = s0[22];
            float v247_data = ir1[2];
            ir1[2] = (v247_data + (v234_data * v245_data));
            float v250_data = s0[31];
            float v252_data = ir1[3];
            ir1[3] = (v252_data + (v234_data * v250_data));
            float v255_data = s0[40];
            float v257_data = ir1[4];
            ir1[4] = (v257_data + (v234_data * v255_data));
            float v260_data = s0[49];
            float v262_data = ir1[5];
            ir1[5] = (v262_data + (v234_data * v260_data));
            float v265_data = s0[58];
            float v267_data = ir1[6];
            ir1[6] = (v267_data + (v234_data * v265_data));
            float v270_data = s0[67];
            float v272_data = ir1[7];
            ir1[7] = (v272_data + (v234_data * v270_data));
            float v275_data = s0[76];
            float v277_data = ir1[8];
            ir1[8] = (v277_data + (v234_data * v275_data));
          }
          if (v10_lead < 9) {
            float v283_data = r0[5];
            float v284_data = s0[5];
            float v286_data = ir1[0];
            ir1[0] = (v286_data + (v283_data * v284_data));
            float v289_data = s0[14];
            float v291_data = ir1[1];
            ir1[1] = (v291_data + (v283_data * v289_data));
            float v294_data = s0[23];
            float v296_data = ir1[2];
            ir1[2] = (v296_data + (v283_data * v294_data));
            float v299_data = s0[32];
            float v301_data = ir1[3];
            ir1[3] = (v301_data + (v283_data * v299_data));
            float v304_data = s0[41];
            float v306_data = ir1[4];
            ir1[4] = (v306_data + (v283_data * v304_data));
            float v309_data = s0[50];
            float v311_data = ir1[5];
            ir1[5] = (v311_data + (v283_data * v309_data));
            float v314_data = s0[59];
            float v316_data = ir1[6];
            ir1[6] = (v316_data + (v283_data * v314_data));
            float v319_data = s0[68];
            float v321_data = ir1[7];
            ir1[7] = (v321_data + (v283_data * v319_data));
            float v324_data = s0[77];
            float v326_data = ir1[8];
            ir1[8] = (v326_data + (v283_data * v324_data));
          }
          if (v10_lead < 9) {
            float v332_data = r0[6];
            float v333_data = s0[6];
            float v335_data = ir1[0];
            ir1[0] = (v335_data + (v332_data * v333_data));
            float v338_data = s0[15];
            float v340_data = ir1[1];
            ir1[1] = (v340_data + (v332_data * v338_data));
            float v343_data = s0[24];
            float v345_data = ir1[2];
            ir1[2] = (v345_data + (v332_data * v343_data));
            float v348_data = s0[33];
            float v350_data = ir1[3];
            ir1[3] = (v350_data + (v332_data * v348_data));
            float v353_data = s0[42];
            float v355_data = ir1[4];
            ir1[4] = (v355_data + (v332_data * v353_data));
            float v358_data = s0[51];
            float v360_data = ir1[5];
            ir1[5] = (v360_data + (v332_data * v358_data));
            float v363_data = s0[60];
            float v365_data = ir1[6];
            ir1[6] = (v365_data + (v332_data * v363_data));
            float v368_data = s0[69];
            float v370_data = ir1[7];
            ir1[7] = (v370_data + (v332_data * v368_data));
            float v373_data = s0[78];
            float v375_data = ir1[8];
            ir1[8] = (v375_data + (v332_data * v373_data));
          }
          if (v10_lead < 9) {
            float v381_data = r0[7];
            float v382_data = s0[7];
            float v384_data = ir1[0];
            ir1[0] = (v384_data + (v381_data * v382_data));
            float v387_data = s0[16];
            float v389_data = ir1[1];
            ir1[1] = (v389_data + (v381_data * v387_data));
            float v392_data = s0[25];
            float v394_data = ir1[2];
            ir1[2] = (v394_data + (v381_data * v392_data));
            float v397_data = s0[34];
            float v399_data = ir1[3];
            ir1[3] = (v399_data + (v381_data * v397_data));
            float v402_data = s0[43];
            float v404_data = ir1[4];
            ir1[4] = (v404_data + (v381_data * v402_data));
            float v407_data = s0[52];
            float v409_data = ir1[5];
            ir1[5] = (v409_data + (v381_data * v407_data));
            float v412_data = s0[61];
            float v414_data = ir1[6];
            ir1[6] = (v414_data + (v381_data * v412_data));
            float v417_data = s0[70];
            float v419_data = ir1[7];
            ir1[7] = (v419_data + (v381_data * v417_data));
            float v422_data = s0[79];
            float v424_data = ir1[8];
            ir1[8] = (v424_data + (v381_data * v422_data));
          }
          if (v10_lead < 9) {
            float v430_data = r0[8];
            float v431_data = s0[8];
            float v433_data = ir1[0];
            ir1[0] = (v433_data + (v430_data * v431_data));
            float v436_data = s0[17];
            float v438_data = ir1[1];
            ir1[1] = (v438_data + (v430_data * v436_data));
            float v441_data = s0[26];
            float v443_data = ir1[2];
            ir1[2] = (v443_data + (v430_data * v441_data));
            float v446_data = s0[35];
            float v448_data = ir1[3];
            ir1[3] = (v448_data + (v430_data * v446_data));
            float v451_data = s0[44];
            float v453_data = ir1[4];
            ir1[4] = (v453_data + (v430_data * v451_data));
            float v456_data = s0[53];
            float v458_data = ir1[5];
            ir1[5] = (v458_data + (v430_data * v456_data));
            float v461_data = s0[62];
            float v463_data = ir1[6];
            ir1[6] = (v463_data + (v430_data * v461_data));
            float v466_data = s0[71];
            float v468_data = ir1[7];
            ir1[7] = (v468_data + (v430_data * v466_data));
            float v471_data = s0[80];
            float v473_data = ir1[8];
            ir1[8] = (v473_data + (v430_data * v471_data));
          }
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v480_n1 = 0; v480_n1 < 9; ++v480_n1) {
              int32_t v481_a = 0 + v480_n1;
              float v483_data = ir1[v480_n1];
              r1[v480_n1] = (v483_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v490_i1 = 0; v490_i1 < 9; ++v490_i1) {
              int32_t v491_a = 0 + v490_i1;
              float v493_data = r1[v490_i1];
              glb_m0[(v10_lead + (v490_i1 * 9))] = v493_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

