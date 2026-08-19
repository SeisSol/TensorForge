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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 9) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 9);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 9), (0, 9)] [(0, 9)]
            float ir1[9]{};
            int32_t v16_lead = threadIdx.x % 16;
            if (v16_lead < 9) {
              float v18_data = r0[0];
              float v19_data = s0[0];
              float v21_data = ir1[0];
              ir1[0] = (v21_data + (v18_data * v19_data));
              float v24_data = s0[9];
              float v26_data = ir1[1];
              ir1[1] = (v26_data + (v18_data * v24_data));
              float v29_data = s0[18];
              float v31_data = ir1[2];
              ir1[2] = (v31_data + (v18_data * v29_data));
              float v34_data = s0[27];
              float v36_data = ir1[3];
              ir1[3] = (v36_data + (v18_data * v34_data));
              float v39_data = s0[36];
              float v41_data = ir1[4];
              ir1[4] = (v41_data + (v18_data * v39_data));
              float v44_data = s0[45];
              float v46_data = ir1[5];
              ir1[5] = (v46_data + (v18_data * v44_data));
              float v49_data = s0[54];
              float v51_data = ir1[6];
              ir1[6] = (v51_data + (v18_data * v49_data));
              float v54_data = s0[63];
              float v56_data = ir1[7];
              ir1[7] = (v56_data + (v18_data * v54_data));
              float v59_data = s0[72];
              float v61_data = ir1[8];
              ir1[8] = (v61_data + (v18_data * v59_data));
            }
            if (v16_lead < 9) {
              float v67_data = r0[1];
              float v68_data = s0[1];
              float v70_data = ir1[0];
              ir1[0] = (v70_data + (v67_data * v68_data));
              float v73_data = s0[10];
              float v75_data = ir1[1];
              ir1[1] = (v75_data + (v67_data * v73_data));
              float v78_data = s0[19];
              float v80_data = ir1[2];
              ir1[2] = (v80_data + (v67_data * v78_data));
              float v83_data = s0[28];
              float v85_data = ir1[3];
              ir1[3] = (v85_data + (v67_data * v83_data));
              float v88_data = s0[37];
              float v90_data = ir1[4];
              ir1[4] = (v90_data + (v67_data * v88_data));
              float v93_data = s0[46];
              float v95_data = ir1[5];
              ir1[5] = (v95_data + (v67_data * v93_data));
              float v98_data = s0[55];
              float v100_data = ir1[6];
              ir1[6] = (v100_data + (v67_data * v98_data));
              float v103_data = s0[64];
              float v105_data = ir1[7];
              ir1[7] = (v105_data + (v67_data * v103_data));
              float v108_data = s0[73];
              float v110_data = ir1[8];
              ir1[8] = (v110_data + (v67_data * v108_data));
            }
            if (v16_lead < 9) {
              float v116_data = r0[2];
              float v117_data = s0[2];
              float v119_data = ir1[0];
              ir1[0] = (v119_data + (v116_data * v117_data));
              float v122_data = s0[11];
              float v124_data = ir1[1];
              ir1[1] = (v124_data + (v116_data * v122_data));
              float v127_data = s0[20];
              float v129_data = ir1[2];
              ir1[2] = (v129_data + (v116_data * v127_data));
              float v132_data = s0[29];
              float v134_data = ir1[3];
              ir1[3] = (v134_data + (v116_data * v132_data));
              float v137_data = s0[38];
              float v139_data = ir1[4];
              ir1[4] = (v139_data + (v116_data * v137_data));
              float v142_data = s0[47];
              float v144_data = ir1[5];
              ir1[5] = (v144_data + (v116_data * v142_data));
              float v147_data = s0[56];
              float v149_data = ir1[6];
              ir1[6] = (v149_data + (v116_data * v147_data));
              float v152_data = s0[65];
              float v154_data = ir1[7];
              ir1[7] = (v154_data + (v116_data * v152_data));
              float v157_data = s0[74];
              float v159_data = ir1[8];
              ir1[8] = (v159_data + (v116_data * v157_data));
            }
            if (v16_lead < 9) {
              float v165_data = r0[3];
              float v166_data = s0[3];
              float v168_data = ir1[0];
              ir1[0] = (v168_data + (v165_data * v166_data));
              float v171_data = s0[12];
              float v173_data = ir1[1];
              ir1[1] = (v173_data + (v165_data * v171_data));
              float v176_data = s0[21];
              float v178_data = ir1[2];
              ir1[2] = (v178_data + (v165_data * v176_data));
              float v181_data = s0[30];
              float v183_data = ir1[3];
              ir1[3] = (v183_data + (v165_data * v181_data));
              float v186_data = s0[39];
              float v188_data = ir1[4];
              ir1[4] = (v188_data + (v165_data * v186_data));
              float v191_data = s0[48];
              float v193_data = ir1[5];
              ir1[5] = (v193_data + (v165_data * v191_data));
              float v196_data = s0[57];
              float v198_data = ir1[6];
              ir1[6] = (v198_data + (v165_data * v196_data));
              float v201_data = s0[66];
              float v203_data = ir1[7];
              ir1[7] = (v203_data + (v165_data * v201_data));
              float v206_data = s0[75];
              float v208_data = ir1[8];
              ir1[8] = (v208_data + (v165_data * v206_data));
            }
            if (v16_lead < 9) {
              float v214_data = r0[4];
              float v215_data = s0[4];
              float v217_data = ir1[0];
              ir1[0] = (v217_data + (v214_data * v215_data));
              float v220_data = s0[13];
              float v222_data = ir1[1];
              ir1[1] = (v222_data + (v214_data * v220_data));
              float v225_data = s0[22];
              float v227_data = ir1[2];
              ir1[2] = (v227_data + (v214_data * v225_data));
              float v230_data = s0[31];
              float v232_data = ir1[3];
              ir1[3] = (v232_data + (v214_data * v230_data));
              float v235_data = s0[40];
              float v237_data = ir1[4];
              ir1[4] = (v237_data + (v214_data * v235_data));
              float v240_data = s0[49];
              float v242_data = ir1[5];
              ir1[5] = (v242_data + (v214_data * v240_data));
              float v245_data = s0[58];
              float v247_data = ir1[6];
              ir1[6] = (v247_data + (v214_data * v245_data));
              float v250_data = s0[67];
              float v252_data = ir1[7];
              ir1[7] = (v252_data + (v214_data * v250_data));
              float v255_data = s0[76];
              float v257_data = ir1[8];
              ir1[8] = (v257_data + (v214_data * v255_data));
            }
            if (v16_lead < 9) {
              float v263_data = r0[5];
              float v264_data = s0[5];
              float v266_data = ir1[0];
              ir1[0] = (v266_data + (v263_data * v264_data));
              float v269_data = s0[14];
              float v271_data = ir1[1];
              ir1[1] = (v271_data + (v263_data * v269_data));
              float v274_data = s0[23];
              float v276_data = ir1[2];
              ir1[2] = (v276_data + (v263_data * v274_data));
              float v279_data = s0[32];
              float v281_data = ir1[3];
              ir1[3] = (v281_data + (v263_data * v279_data));
              float v284_data = s0[41];
              float v286_data = ir1[4];
              ir1[4] = (v286_data + (v263_data * v284_data));
              float v289_data = s0[50];
              float v291_data = ir1[5];
              ir1[5] = (v291_data + (v263_data * v289_data));
              float v294_data = s0[59];
              float v296_data = ir1[6];
              ir1[6] = (v296_data + (v263_data * v294_data));
              float v299_data = s0[68];
              float v301_data = ir1[7];
              ir1[7] = (v301_data + (v263_data * v299_data));
              float v304_data = s0[77];
              float v306_data = ir1[8];
              ir1[8] = (v306_data + (v263_data * v304_data));
            }
            if (v16_lead < 9) {
              float v312_data = r0[6];
              float v313_data = s0[6];
              float v315_data = ir1[0];
              ir1[0] = (v315_data + (v312_data * v313_data));
              float v318_data = s0[15];
              float v320_data = ir1[1];
              ir1[1] = (v320_data + (v312_data * v318_data));
              float v323_data = s0[24];
              float v325_data = ir1[2];
              ir1[2] = (v325_data + (v312_data * v323_data));
              float v328_data = s0[33];
              float v330_data = ir1[3];
              ir1[3] = (v330_data + (v312_data * v328_data));
              float v333_data = s0[42];
              float v335_data = ir1[4];
              ir1[4] = (v335_data + (v312_data * v333_data));
              float v338_data = s0[51];
              float v340_data = ir1[5];
              ir1[5] = (v340_data + (v312_data * v338_data));
              float v343_data = s0[60];
              float v345_data = ir1[6];
              ir1[6] = (v345_data + (v312_data * v343_data));
              float v348_data = s0[69];
              float v350_data = ir1[7];
              ir1[7] = (v350_data + (v312_data * v348_data));
              float v353_data = s0[78];
              float v355_data = ir1[8];
              ir1[8] = (v355_data + (v312_data * v353_data));
            }
            if (v16_lead < 9) {
              float v361_data = r0[7];
              float v362_data = s0[7];
              float v364_data = ir1[0];
              ir1[0] = (v364_data + (v361_data * v362_data));
              float v367_data = s0[16];
              float v369_data = ir1[1];
              ir1[1] = (v369_data + (v361_data * v367_data));
              float v372_data = s0[25];
              float v374_data = ir1[2];
              ir1[2] = (v374_data + (v361_data * v372_data));
              float v377_data = s0[34];
              float v379_data = ir1[3];
              ir1[3] = (v379_data + (v361_data * v377_data));
              float v382_data = s0[43];
              float v384_data = ir1[4];
              ir1[4] = (v384_data + (v361_data * v382_data));
              float v387_data = s0[52];
              float v389_data = ir1[5];
              ir1[5] = (v389_data + (v361_data * v387_data));
              float v392_data = s0[61];
              float v394_data = ir1[6];
              ir1[6] = (v394_data + (v361_data * v392_data));
              float v397_data = s0[70];
              float v399_data = ir1[7];
              ir1[7] = (v399_data + (v361_data * v397_data));
              float v402_data = s0[79];
              float v404_data = ir1[8];
              ir1[8] = (v404_data + (v361_data * v402_data));
            }
            if (v16_lead < 9) {
              float v410_data = r0[8];
              float v411_data = s0[8];
              float v413_data = ir1[0];
              ir1[0] = (v413_data + (v410_data * v411_data));
              float v416_data = s0[17];
              float v418_data = ir1[1];
              ir1[1] = (v418_data + (v410_data * v416_data));
              float v421_data = s0[26];
              float v423_data = ir1[2];
              ir1[2] = (v423_data + (v410_data * v421_data));
              float v426_data = s0[35];
              float v428_data = ir1[3];
              ir1[3] = (v428_data + (v410_data * v426_data));
              float v431_data = s0[44];
              float v433_data = ir1[4];
              ir1[4] = (v433_data + (v410_data * v431_data));
              float v436_data = s0[53];
              float v438_data = ir1[5];
              ir1[5] = (v438_data + (v410_data * v436_data));
              float v441_data = s0[62];
              float v443_data = ir1[6];
              ir1[6] = (v443_data + (v410_data * v441_data));
              float v446_data = s0[71];
              float v448_data = ir1[7];
              ir1[7] = (v448_data + (v410_data * v446_data));
              float v451_data = s0[80];
              float v453_data = ir1[8];
              ir1[8] = (v453_data + (v410_data * v451_data));
            }
            float v455_data;
            {
              v455_data = 0.0f;
              v455_data = 13.0f;
            }
            if (v16_lead < 9) {
              #pragma unroll
              for (int32_t v460_n1 = 0; v460_n1 < 9; ++v460_n1) {
                int32_t v461_a = 0 + v460_n1;
                float v463_data = ir1[v460_n1];
                int32_t v465_a = 0 + v460_n1;
                r1[v460_n1] = (v463_data * v455_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v469_lead = threadIdx.x % 16;
          if (v469_lead < 9) {
            #pragma unroll
            for (int32_t v471_i1 = 0; v471_i1 < 9; ++v471_i1) {
              int32_t v472_a = 0 + v471_i1;
              float v474_data = r1[v471_i1];
              int32_t v481_a = v469_lead + (v471_i1 * 9);
              glb_m0[v481_a] = v474_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

