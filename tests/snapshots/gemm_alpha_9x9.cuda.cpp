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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 9;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            if (threadIdx.x < 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          if (v3_lead < 9) {
            float v28_data = r0[0];
            float v29_data = s0[0];
            float v31_data = ir1[0];
            ir1[0] = (v31_data + (v28_data * v29_data));
            float v34_data = s0[9];
            float v36_data = ir1[1];
            ir1[1] = (v36_data + (v28_data * v34_data));
            float v39_data = s0[18];
            float v41_data = ir1[2];
            ir1[2] = (v41_data + (v28_data * v39_data));
            float v44_data = s0[27];
            float v46_data = ir1[3];
            ir1[3] = (v46_data + (v28_data * v44_data));
            float v49_data = s0[36];
            float v51_data = ir1[4];
            ir1[4] = (v51_data + (v28_data * v49_data));
            float v54_data = s0[45];
            float v56_data = ir1[5];
            ir1[5] = (v56_data + (v28_data * v54_data));
            float v59_data = s0[54];
            float v61_data = ir1[6];
            ir1[6] = (v61_data + (v28_data * v59_data));
            float v64_data = s0[63];
            float v66_data = ir1[7];
            ir1[7] = (v66_data + (v28_data * v64_data));
            float v69_data = s0[72];
            float v71_data = ir1[8];
            ir1[8] = (v71_data + (v28_data * v69_data));
          }
          if (v3_lead < 9) {
            float v77_data = r0[1];
            float v78_data = s0[1];
            float v80_data = ir1[0];
            ir1[0] = (v80_data + (v77_data * v78_data));
            float v83_data = s0[10];
            float v85_data = ir1[1];
            ir1[1] = (v85_data + (v77_data * v83_data));
            float v88_data = s0[19];
            float v90_data = ir1[2];
            ir1[2] = (v90_data + (v77_data * v88_data));
            float v93_data = s0[28];
            float v95_data = ir1[3];
            ir1[3] = (v95_data + (v77_data * v93_data));
            float v98_data = s0[37];
            float v100_data = ir1[4];
            ir1[4] = (v100_data + (v77_data * v98_data));
            float v103_data = s0[46];
            float v105_data = ir1[5];
            ir1[5] = (v105_data + (v77_data * v103_data));
            float v108_data = s0[55];
            float v110_data = ir1[6];
            ir1[6] = (v110_data + (v77_data * v108_data));
            float v113_data = s0[64];
            float v115_data = ir1[7];
            ir1[7] = (v115_data + (v77_data * v113_data));
            float v118_data = s0[73];
            float v120_data = ir1[8];
            ir1[8] = (v120_data + (v77_data * v118_data));
          }
          if (v3_lead < 9) {
            float v126_data = r0[2];
            float v127_data = s0[2];
            float v129_data = ir1[0];
            ir1[0] = (v129_data + (v126_data * v127_data));
            float v132_data = s0[11];
            float v134_data = ir1[1];
            ir1[1] = (v134_data + (v126_data * v132_data));
            float v137_data = s0[20];
            float v139_data = ir1[2];
            ir1[2] = (v139_data + (v126_data * v137_data));
            float v142_data = s0[29];
            float v144_data = ir1[3];
            ir1[3] = (v144_data + (v126_data * v142_data));
            float v147_data = s0[38];
            float v149_data = ir1[4];
            ir1[4] = (v149_data + (v126_data * v147_data));
            float v152_data = s0[47];
            float v154_data = ir1[5];
            ir1[5] = (v154_data + (v126_data * v152_data));
            float v157_data = s0[56];
            float v159_data = ir1[6];
            ir1[6] = (v159_data + (v126_data * v157_data));
            float v162_data = s0[65];
            float v164_data = ir1[7];
            ir1[7] = (v164_data + (v126_data * v162_data));
            float v167_data = s0[74];
            float v169_data = ir1[8];
            ir1[8] = (v169_data + (v126_data * v167_data));
          }
          if (v3_lead < 9) {
            float v175_data = r0[3];
            float v176_data = s0[3];
            float v178_data = ir1[0];
            ir1[0] = (v178_data + (v175_data * v176_data));
            float v181_data = s0[12];
            float v183_data = ir1[1];
            ir1[1] = (v183_data + (v175_data * v181_data));
            float v186_data = s0[21];
            float v188_data = ir1[2];
            ir1[2] = (v188_data + (v175_data * v186_data));
            float v191_data = s0[30];
            float v193_data = ir1[3];
            ir1[3] = (v193_data + (v175_data * v191_data));
            float v196_data = s0[39];
            float v198_data = ir1[4];
            ir1[4] = (v198_data + (v175_data * v196_data));
            float v201_data = s0[48];
            float v203_data = ir1[5];
            ir1[5] = (v203_data + (v175_data * v201_data));
            float v206_data = s0[57];
            float v208_data = ir1[6];
            ir1[6] = (v208_data + (v175_data * v206_data));
            float v211_data = s0[66];
            float v213_data = ir1[7];
            ir1[7] = (v213_data + (v175_data * v211_data));
            float v216_data = s0[75];
            float v218_data = ir1[8];
            ir1[8] = (v218_data + (v175_data * v216_data));
          }
          if (v3_lead < 9) {
            float v224_data = r0[4];
            float v225_data = s0[4];
            float v227_data = ir1[0];
            ir1[0] = (v227_data + (v224_data * v225_data));
            float v230_data = s0[13];
            float v232_data = ir1[1];
            ir1[1] = (v232_data + (v224_data * v230_data));
            float v235_data = s0[22];
            float v237_data = ir1[2];
            ir1[2] = (v237_data + (v224_data * v235_data));
            float v240_data = s0[31];
            float v242_data = ir1[3];
            ir1[3] = (v242_data + (v224_data * v240_data));
            float v245_data = s0[40];
            float v247_data = ir1[4];
            ir1[4] = (v247_data + (v224_data * v245_data));
            float v250_data = s0[49];
            float v252_data = ir1[5];
            ir1[5] = (v252_data + (v224_data * v250_data));
            float v255_data = s0[58];
            float v257_data = ir1[6];
            ir1[6] = (v257_data + (v224_data * v255_data));
            float v260_data = s0[67];
            float v262_data = ir1[7];
            ir1[7] = (v262_data + (v224_data * v260_data));
            float v265_data = s0[76];
            float v267_data = ir1[8];
            ir1[8] = (v267_data + (v224_data * v265_data));
          }
          if (v3_lead < 9) {
            float v273_data = r0[5];
            float v274_data = s0[5];
            float v276_data = ir1[0];
            ir1[0] = (v276_data + (v273_data * v274_data));
            float v279_data = s0[14];
            float v281_data = ir1[1];
            ir1[1] = (v281_data + (v273_data * v279_data));
            float v284_data = s0[23];
            float v286_data = ir1[2];
            ir1[2] = (v286_data + (v273_data * v284_data));
            float v289_data = s0[32];
            float v291_data = ir1[3];
            ir1[3] = (v291_data + (v273_data * v289_data));
            float v294_data = s0[41];
            float v296_data = ir1[4];
            ir1[4] = (v296_data + (v273_data * v294_data));
            float v299_data = s0[50];
            float v301_data = ir1[5];
            ir1[5] = (v301_data + (v273_data * v299_data));
            float v304_data = s0[59];
            float v306_data = ir1[6];
            ir1[6] = (v306_data + (v273_data * v304_data));
            float v309_data = s0[68];
            float v311_data = ir1[7];
            ir1[7] = (v311_data + (v273_data * v309_data));
            float v314_data = s0[77];
            float v316_data = ir1[8];
            ir1[8] = (v316_data + (v273_data * v314_data));
          }
          if (v3_lead < 9) {
            float v322_data = r0[6];
            float v323_data = s0[6];
            float v325_data = ir1[0];
            ir1[0] = (v325_data + (v322_data * v323_data));
            float v328_data = s0[15];
            float v330_data = ir1[1];
            ir1[1] = (v330_data + (v322_data * v328_data));
            float v333_data = s0[24];
            float v335_data = ir1[2];
            ir1[2] = (v335_data + (v322_data * v333_data));
            float v338_data = s0[33];
            float v340_data = ir1[3];
            ir1[3] = (v340_data + (v322_data * v338_data));
            float v343_data = s0[42];
            float v345_data = ir1[4];
            ir1[4] = (v345_data + (v322_data * v343_data));
            float v348_data = s0[51];
            float v350_data = ir1[5];
            ir1[5] = (v350_data + (v322_data * v348_data));
            float v353_data = s0[60];
            float v355_data = ir1[6];
            ir1[6] = (v355_data + (v322_data * v353_data));
            float v358_data = s0[69];
            float v360_data = ir1[7];
            ir1[7] = (v360_data + (v322_data * v358_data));
            float v363_data = s0[78];
            float v365_data = ir1[8];
            ir1[8] = (v365_data + (v322_data * v363_data));
          }
          if (v3_lead < 9) {
            float v371_data = r0[7];
            float v372_data = s0[7];
            float v374_data = ir1[0];
            ir1[0] = (v374_data + (v371_data * v372_data));
            float v377_data = s0[16];
            float v379_data = ir1[1];
            ir1[1] = (v379_data + (v371_data * v377_data));
            float v382_data = s0[25];
            float v384_data = ir1[2];
            ir1[2] = (v384_data + (v371_data * v382_data));
            float v387_data = s0[34];
            float v389_data = ir1[3];
            ir1[3] = (v389_data + (v371_data * v387_data));
            float v392_data = s0[43];
            float v394_data = ir1[4];
            ir1[4] = (v394_data + (v371_data * v392_data));
            float v397_data = s0[52];
            float v399_data = ir1[5];
            ir1[5] = (v399_data + (v371_data * v397_data));
            float v402_data = s0[61];
            float v404_data = ir1[6];
            ir1[6] = (v404_data + (v371_data * v402_data));
            float v407_data = s0[70];
            float v409_data = ir1[7];
            ir1[7] = (v409_data + (v371_data * v407_data));
            float v412_data = s0[79];
            float v414_data = ir1[8];
            ir1[8] = (v414_data + (v371_data * v412_data));
          }
          if (v3_lead < 9) {
            float v420_data = r0[8];
            float v421_data = s0[8];
            float v423_data = ir1[0];
            ir1[0] = (v423_data + (v420_data * v421_data));
            float v426_data = s0[17];
            float v428_data = ir1[1];
            ir1[1] = (v428_data + (v420_data * v426_data));
            float v431_data = s0[26];
            float v433_data = ir1[2];
            ir1[2] = (v433_data + (v420_data * v431_data));
            float v436_data = s0[35];
            float v438_data = ir1[3];
            ir1[3] = (v438_data + (v420_data * v436_data));
            float v441_data = s0[44];
            float v443_data = ir1[4];
            ir1[4] = (v443_data + (v420_data * v441_data));
            float v446_data = s0[53];
            float v448_data = ir1[5];
            ir1[5] = (v448_data + (v420_data * v446_data));
            float v451_data = s0[62];
            float v453_data = ir1[6];
            ir1[6] = (v453_data + (v420_data * v451_data));
            float v456_data = s0[71];
            float v458_data = ir1[7];
            ir1[7] = (v458_data + (v420_data * v456_data));
            float v461_data = s0[80];
            float v463_data = ir1[8];
            ir1[8] = (v463_data + (v420_data * v461_data));
          }
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v470_n1 = 0; v470_n1 < 9; ++v470_n1) {
              int32_t v471_a = 0 + v470_n1;
              float v473_data = ir1[v470_n1];
              int32_t v475_a = 0 + v470_n1;
              r1[v470_n1] = (v473_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v481_i1 = 0; v481_i1 < 9; ++v481_i1) {
              int32_t v482_a = 0 + v481_i1;
              float v484_data = r1[v481_i1];
              int32_t v491_a = v3_lead + (v481_i1 * 9);
              glb_m0[v491_a] = v484_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

