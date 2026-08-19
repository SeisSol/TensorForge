// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 3072 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7ab185b978, cudaFuncAttributeMaxDynamicSharedMemorySize, 3072 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7ab185b978<<<grid,block,3072 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m0[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 16);
              float v26_data;
              {
                v26_data = __ldcg(&glb_m1[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r2[v27_a] = v26_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v31_data = r0[0];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + v31_data);
          float v34_data = r0[1];
          float v35_data = ir1[1];
          ir1[1] = (v35_data + v34_data);
          float v37_data = r0[2];
          float v38_data = ir1[2];
          ir1[2] = (v38_data + v37_data);
          float v40_data = r0[3];
          float v41_data = ir1[3];
          ir1[3] = (v41_data + v40_data);
          float v43_data = r0[4];
          float v44_data = ir1[4];
          ir1[4] = (v44_data + v43_data);
          float v46_data = r0[5];
          float v47_data = ir1[5];
          ir1[5] = (v47_data + v46_data);
          float v49_data = r0[6];
          float v50_data = ir1[6];
          ir1[6] = (v50_data + v49_data);
          float v52_data = r0[7];
          float v53_data = ir1[7];
          ir1[7] = (v53_data + v52_data);
          float v55_data = r0[8];
          float v56_data = ir1[8];
          ir1[8] = (v56_data + v55_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          int32_t v60_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
            int32_t v69_lead = v60_lead + (v61_i0 * 32);
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
              int32_t v63_a = v61_i0 + v62_i1;
              float v64_data = r1[v63_a];
              int32_t v71_a = v69_lead + (v62_i1 * 32);
              s0[v71_a] = v64_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          int32_t v74_lead = threadIdx.x % 32;
          if (v74_lead < 16) {
            #pragma unroll
            for (int32_t v76_i1 = 0; v76_i1 < 9; ++v76_i1) {
              int32_t v83_a = v74_lead + (v76_i1 * 16);
              float v84_data;
              {
                v84_data = __ldcg(&glb_m2[v83_a]);
              }
              int32_t v85_a = 0 + v76_i1;
              r4[v85_a] = v84_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          {
            // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir3[9]{};
            int32_t v88_lead = threadIdx.x % 32;
            if (v88_lead < 16) {
              float v90_data = r2[0];
              float v91_data = ir3[0];
              ir3[0] = (v91_data + v90_data);
              float v93_data = r2[1];
              float v94_data = ir3[1];
              ir3[1] = (v94_data + v93_data);
              float v96_data = r2[2];
              float v97_data = ir3[2];
              ir3[2] = (v97_data + v96_data);
              float v99_data = r2[3];
              float v100_data = ir3[3];
              ir3[3] = (v100_data + v99_data);
              float v102_data = r2[4];
              float v103_data = ir3[4];
              ir3[4] = (v103_data + v102_data);
              float v105_data = r2[5];
              float v106_data = ir3[5];
              ir3[5] = (v106_data + v105_data);
              float v108_data = r2[6];
              float v109_data = ir3[6];
              ir3[6] = (v109_data + v108_data);
              float v111_data = r2[7];
              float v112_data = ir3[7];
              ir3[7] = (v112_data + v111_data);
              float v114_data = r2[8];
              float v115_data = ir3[8];
              ir3[8] = (v115_data + v114_data);
            }
            if (v88_lead < 16) {
              #pragma unroll
              for (int32_t v121_n1 = 0; v121_n1 < 9; ++v121_n1) {
                int32_t v122_a = 0 + v121_n1;
                float v123_data = ir3[v122_a];
                int32_t v130_a = v88_lead + (v121_n1 * 32);
                float v131_data = s0[v130_a];
                int32_t v133_a = 0 + v121_n1;
                r3[v133_a] = (v131_data + v123_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          int32_t v136_lead = threadIdx.x % 32;
          if (v136_lead < 16) {
            #pragma unroll
            for (int32_t v138_i1 = 0; v138_i1 < 9; ++v138_i1) {
              int32_t v139_a = 0 + v138_i1;
              float v140_data = r3[v139_a];
              int32_t v147_a = v136_lead + (v138_i1 * 32);
              s0[v147_a] = v140_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          {
            // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir5[9]{};
            int32_t v150_lead = threadIdx.x % 32;
            if (v150_lead < 16) {
              float v152_data = r4[0];
              float v153_data = ir5[0];
              ir5[0] = (v153_data + v152_data);
              float v155_data = r4[1];
              float v156_data = ir5[1];
              ir5[1] = (v156_data + v155_data);
              float v158_data = r4[2];
              float v159_data = ir5[2];
              ir5[2] = (v159_data + v158_data);
              float v161_data = r4[3];
              float v162_data = ir5[3];
              ir5[3] = (v162_data + v161_data);
              float v164_data = r4[4];
              float v165_data = ir5[4];
              ir5[4] = (v165_data + v164_data);
              float v167_data = r4[5];
              float v168_data = ir5[5];
              ir5[5] = (v168_data + v167_data);
              float v170_data = r4[6];
              float v171_data = ir5[6];
              ir5[6] = (v171_data + v170_data);
              float v173_data = r4[7];
              float v174_data = ir5[7];
              ir5[7] = (v174_data + v173_data);
              float v176_data = r4[8];
              float v177_data = ir5[8];
              ir5[8] = (v177_data + v176_data);
            }
            if (v150_lead < 16) {
              #pragma unroll
              for (int32_t v183_n1 = 0; v183_n1 < 9; ++v183_n1) {
                int32_t v184_a = 0 + v183_n1;
                float v185_data = ir5[v184_a];
                int32_t v192_a = v150_lead + (v183_n1 * 32);
                float v193_data = s0[v192_a];
                int32_t v195_a = 0 + v183_n1;
                r5[v195_a] = (v193_data + v185_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          int32_t v198_lead = threadIdx.x % 32;
          if (v198_lead < 16) {
            #pragma unroll
            for (int32_t v200_i1 = 0; v200_i1 < 9; ++v200_i1) {
              int32_t v201_a = 0 + v200_i1;
              float v202_data = r5[v201_a];
              int32_t v209_a = v198_lead + (v200_i1 * 32);
              s0[v209_a] = v202_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<4>(4), pipeline);
          if (threadIdx.x < 17) {
            cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r6[9]{};
          __syncwarp();
          {
            // r6 = +(s0 * s1) + None
            // [(0, 32), (0, 9)] [(0, 9)]
            float ir6[9]{};
            int32_t v212_lead = threadIdx.x % 32;
            int32_t v218_a = v212_lead + 0;
            float v219_data = s0[v218_a];
            float v220_data = s1[0];
            float v222_data = ir6[0];
            ir6[0] = (v222_data + (v219_data * v220_data));
            int32_t v229_a = v212_lead + 0;
            float v230_data = s0[v229_a];
            float v231_data = s1[9];
            float v233_data = ir6[1];
            ir6[1] = (v233_data + (v230_data * v231_data));
            int32_t v240_a = v212_lead + 0;
            float v241_data = s0[v240_a];
            float v242_data = s1[18];
            float v244_data = ir6[2];
            ir6[2] = (v244_data + (v241_data * v242_data));
            int32_t v251_a = v212_lead + 0;
            float v252_data = s0[v251_a];
            float v253_data = s1[27];
            float v255_data = ir6[3];
            ir6[3] = (v255_data + (v252_data * v253_data));
            int32_t v262_a = v212_lead + 0;
            float v263_data = s0[v262_a];
            float v264_data = s1[36];
            float v266_data = ir6[4];
            ir6[4] = (v266_data + (v263_data * v264_data));
            int32_t v273_a = v212_lead + 0;
            float v274_data = s0[v273_a];
            float v275_data = s1[45];
            float v277_data = ir6[5];
            ir6[5] = (v277_data + (v274_data * v275_data));
            int32_t v284_a = v212_lead + 0;
            float v285_data = s0[v284_a];
            float v286_data = s1[54];
            float v288_data = ir6[6];
            ir6[6] = (v288_data + (v285_data * v286_data));
            int32_t v295_a = v212_lead + 0;
            float v296_data = s0[v295_a];
            float v297_data = s1[63];
            float v299_data = ir6[7];
            ir6[7] = (v299_data + (v296_data * v297_data));
            int32_t v306_a = v212_lead + 0;
            float v307_data = s0[v306_a];
            float v308_data = s1[72];
            float v310_data = ir6[8];
            ir6[8] = (v310_data + (v307_data * v308_data));
            int32_t v320_a = v212_lead + 32;
            float v321_data = s0[v320_a];
            float v322_data = s1[1];
            float v324_data = ir6[0];
            ir6[0] = (v324_data + (v321_data * v322_data));
            int32_t v331_a = v212_lead + 32;
            float v332_data = s0[v331_a];
            float v333_data = s1[10];
            float v335_data = ir6[1];
            ir6[1] = (v335_data + (v332_data * v333_data));
            int32_t v342_a = v212_lead + 32;
            float v343_data = s0[v342_a];
            float v344_data = s1[19];
            float v346_data = ir6[2];
            ir6[2] = (v346_data + (v343_data * v344_data));
            int32_t v353_a = v212_lead + 32;
            float v354_data = s0[v353_a];
            float v355_data = s1[28];
            float v357_data = ir6[3];
            ir6[3] = (v357_data + (v354_data * v355_data));
            int32_t v364_a = v212_lead + 32;
            float v365_data = s0[v364_a];
            float v366_data = s1[37];
            float v368_data = ir6[4];
            ir6[4] = (v368_data + (v365_data * v366_data));
            int32_t v375_a = v212_lead + 32;
            float v376_data = s0[v375_a];
            float v377_data = s1[46];
            float v379_data = ir6[5];
            ir6[5] = (v379_data + (v376_data * v377_data));
            int32_t v386_a = v212_lead + 32;
            float v387_data = s0[v386_a];
            float v388_data = s1[55];
            float v390_data = ir6[6];
            ir6[6] = (v390_data + (v387_data * v388_data));
            int32_t v397_a = v212_lead + 32;
            float v398_data = s0[v397_a];
            float v399_data = s1[64];
            float v401_data = ir6[7];
            ir6[7] = (v401_data + (v398_data * v399_data));
            int32_t v408_a = v212_lead + 32;
            float v409_data = s0[v408_a];
            float v410_data = s1[73];
            float v412_data = ir6[8];
            ir6[8] = (v412_data + (v409_data * v410_data));
            int32_t v422_a = v212_lead + 64;
            float v423_data = s0[v422_a];
            float v424_data = s1[2];
            float v426_data = ir6[0];
            ir6[0] = (v426_data + (v423_data * v424_data));
            int32_t v433_a = v212_lead + 64;
            float v434_data = s0[v433_a];
            float v435_data = s1[11];
            float v437_data = ir6[1];
            ir6[1] = (v437_data + (v434_data * v435_data));
            int32_t v444_a = v212_lead + 64;
            float v445_data = s0[v444_a];
            float v446_data = s1[20];
            float v448_data = ir6[2];
            ir6[2] = (v448_data + (v445_data * v446_data));
            int32_t v455_a = v212_lead + 64;
            float v456_data = s0[v455_a];
            float v457_data = s1[29];
            float v459_data = ir6[3];
            ir6[3] = (v459_data + (v456_data * v457_data));
            int32_t v466_a = v212_lead + 64;
            float v467_data = s0[v466_a];
            float v468_data = s1[38];
            float v470_data = ir6[4];
            ir6[4] = (v470_data + (v467_data * v468_data));
            int32_t v477_a = v212_lead + 64;
            float v478_data = s0[v477_a];
            float v479_data = s1[47];
            float v481_data = ir6[5];
            ir6[5] = (v481_data + (v478_data * v479_data));
            int32_t v488_a = v212_lead + 64;
            float v489_data = s0[v488_a];
            float v490_data = s1[56];
            float v492_data = ir6[6];
            ir6[6] = (v492_data + (v489_data * v490_data));
            int32_t v499_a = v212_lead + 64;
            float v500_data = s0[v499_a];
            float v501_data = s1[65];
            float v503_data = ir6[7];
            ir6[7] = (v503_data + (v500_data * v501_data));
            int32_t v510_a = v212_lead + 64;
            float v511_data = s0[v510_a];
            float v512_data = s1[74];
            float v514_data = ir6[8];
            ir6[8] = (v514_data + (v511_data * v512_data));
            int32_t v524_a = v212_lead + 96;
            float v525_data = s0[v524_a];
            float v526_data = s1[3];
            float v528_data = ir6[0];
            ir6[0] = (v528_data + (v525_data * v526_data));
            int32_t v535_a = v212_lead + 96;
            float v536_data = s0[v535_a];
            float v537_data = s1[12];
            float v539_data = ir6[1];
            ir6[1] = (v539_data + (v536_data * v537_data));
            int32_t v546_a = v212_lead + 96;
            float v547_data = s0[v546_a];
            float v548_data = s1[21];
            float v550_data = ir6[2];
            ir6[2] = (v550_data + (v547_data * v548_data));
            int32_t v557_a = v212_lead + 96;
            float v558_data = s0[v557_a];
            float v559_data = s1[30];
            float v561_data = ir6[3];
            ir6[3] = (v561_data + (v558_data * v559_data));
            int32_t v568_a = v212_lead + 96;
            float v569_data = s0[v568_a];
            float v570_data = s1[39];
            float v572_data = ir6[4];
            ir6[4] = (v572_data + (v569_data * v570_data));
            int32_t v579_a = v212_lead + 96;
            float v580_data = s0[v579_a];
            float v581_data = s1[48];
            float v583_data = ir6[5];
            ir6[5] = (v583_data + (v580_data * v581_data));
            int32_t v590_a = v212_lead + 96;
            float v591_data = s0[v590_a];
            float v592_data = s1[57];
            float v594_data = ir6[6];
            ir6[6] = (v594_data + (v591_data * v592_data));
            int32_t v601_a = v212_lead + 96;
            float v602_data = s0[v601_a];
            float v603_data = s1[66];
            float v605_data = ir6[7];
            ir6[7] = (v605_data + (v602_data * v603_data));
            int32_t v612_a = v212_lead + 96;
            float v613_data = s0[v612_a];
            float v614_data = s1[75];
            float v616_data = ir6[8];
            ir6[8] = (v616_data + (v613_data * v614_data));
            int32_t v626_a = v212_lead + 128;
            float v627_data = s0[v626_a];
            float v628_data = s1[4];
            float v630_data = ir6[0];
            ir6[0] = (v630_data + (v627_data * v628_data));
            int32_t v637_a = v212_lead + 128;
            float v638_data = s0[v637_a];
            float v639_data = s1[13];
            float v641_data = ir6[1];
            ir6[1] = (v641_data + (v638_data * v639_data));
            int32_t v648_a = v212_lead + 128;
            float v649_data = s0[v648_a];
            float v650_data = s1[22];
            float v652_data = ir6[2];
            ir6[2] = (v652_data + (v649_data * v650_data));
            int32_t v659_a = v212_lead + 128;
            float v660_data = s0[v659_a];
            float v661_data = s1[31];
            float v663_data = ir6[3];
            ir6[3] = (v663_data + (v660_data * v661_data));
            int32_t v670_a = v212_lead + 128;
            float v671_data = s0[v670_a];
            float v672_data = s1[40];
            float v674_data = ir6[4];
            ir6[4] = (v674_data + (v671_data * v672_data));
            int32_t v681_a = v212_lead + 128;
            float v682_data = s0[v681_a];
            float v683_data = s1[49];
            float v685_data = ir6[5];
            ir6[5] = (v685_data + (v682_data * v683_data));
            int32_t v692_a = v212_lead + 128;
            float v693_data = s0[v692_a];
            float v694_data = s1[58];
            float v696_data = ir6[6];
            ir6[6] = (v696_data + (v693_data * v694_data));
            int32_t v703_a = v212_lead + 128;
            float v704_data = s0[v703_a];
            float v705_data = s1[67];
            float v707_data = ir6[7];
            ir6[7] = (v707_data + (v704_data * v705_data));
            int32_t v714_a = v212_lead + 128;
            float v715_data = s0[v714_a];
            float v716_data = s1[76];
            float v718_data = ir6[8];
            ir6[8] = (v718_data + (v715_data * v716_data));
            int32_t v728_a = v212_lead + 160;
            float v729_data = s0[v728_a];
            float v730_data = s1[5];
            float v732_data = ir6[0];
            ir6[0] = (v732_data + (v729_data * v730_data));
            int32_t v739_a = v212_lead + 160;
            float v740_data = s0[v739_a];
            float v741_data = s1[14];
            float v743_data = ir6[1];
            ir6[1] = (v743_data + (v740_data * v741_data));
            int32_t v750_a = v212_lead + 160;
            float v751_data = s0[v750_a];
            float v752_data = s1[23];
            float v754_data = ir6[2];
            ir6[2] = (v754_data + (v751_data * v752_data));
            int32_t v761_a = v212_lead + 160;
            float v762_data = s0[v761_a];
            float v763_data = s1[32];
            float v765_data = ir6[3];
            ir6[3] = (v765_data + (v762_data * v763_data));
            int32_t v772_a = v212_lead + 160;
            float v773_data = s0[v772_a];
            float v774_data = s1[41];
            float v776_data = ir6[4];
            ir6[4] = (v776_data + (v773_data * v774_data));
            int32_t v783_a = v212_lead + 160;
            float v784_data = s0[v783_a];
            float v785_data = s1[50];
            float v787_data = ir6[5];
            ir6[5] = (v787_data + (v784_data * v785_data));
            int32_t v794_a = v212_lead + 160;
            float v795_data = s0[v794_a];
            float v796_data = s1[59];
            float v798_data = ir6[6];
            ir6[6] = (v798_data + (v795_data * v796_data));
            int32_t v805_a = v212_lead + 160;
            float v806_data = s0[v805_a];
            float v807_data = s1[68];
            float v809_data = ir6[7];
            ir6[7] = (v809_data + (v806_data * v807_data));
            int32_t v816_a = v212_lead + 160;
            float v817_data = s0[v816_a];
            float v818_data = s1[77];
            float v820_data = ir6[8];
            ir6[8] = (v820_data + (v817_data * v818_data));
            int32_t v830_a = v212_lead + 192;
            float v831_data = s0[v830_a];
            float v832_data = s1[6];
            float v834_data = ir6[0];
            ir6[0] = (v834_data + (v831_data * v832_data));
            int32_t v841_a = v212_lead + 192;
            float v842_data = s0[v841_a];
            float v843_data = s1[15];
            float v845_data = ir6[1];
            ir6[1] = (v845_data + (v842_data * v843_data));
            int32_t v852_a = v212_lead + 192;
            float v853_data = s0[v852_a];
            float v854_data = s1[24];
            float v856_data = ir6[2];
            ir6[2] = (v856_data + (v853_data * v854_data));
            int32_t v863_a = v212_lead + 192;
            float v864_data = s0[v863_a];
            float v865_data = s1[33];
            float v867_data = ir6[3];
            ir6[3] = (v867_data + (v864_data * v865_data));
            int32_t v874_a = v212_lead + 192;
            float v875_data = s0[v874_a];
            float v876_data = s1[42];
            float v878_data = ir6[4];
            ir6[4] = (v878_data + (v875_data * v876_data));
            int32_t v885_a = v212_lead + 192;
            float v886_data = s0[v885_a];
            float v887_data = s1[51];
            float v889_data = ir6[5];
            ir6[5] = (v889_data + (v886_data * v887_data));
            int32_t v896_a = v212_lead + 192;
            float v897_data = s0[v896_a];
            float v898_data = s1[60];
            float v900_data = ir6[6];
            ir6[6] = (v900_data + (v897_data * v898_data));
            int32_t v907_a = v212_lead + 192;
            float v908_data = s0[v907_a];
            float v909_data = s1[69];
            float v911_data = ir6[7];
            ir6[7] = (v911_data + (v908_data * v909_data));
            int32_t v918_a = v212_lead + 192;
            float v919_data = s0[v918_a];
            float v920_data = s1[78];
            float v922_data = ir6[8];
            ir6[8] = (v922_data + (v919_data * v920_data));
            int32_t v932_a = v212_lead + 224;
            float v933_data = s0[v932_a];
            float v934_data = s1[7];
            float v936_data = ir6[0];
            ir6[0] = (v936_data + (v933_data * v934_data));
            int32_t v943_a = v212_lead + 224;
            float v944_data = s0[v943_a];
            float v945_data = s1[16];
            float v947_data = ir6[1];
            ir6[1] = (v947_data + (v944_data * v945_data));
            int32_t v954_a = v212_lead + 224;
            float v955_data = s0[v954_a];
            float v956_data = s1[25];
            float v958_data = ir6[2];
            ir6[2] = (v958_data + (v955_data * v956_data));
            int32_t v965_a = v212_lead + 224;
            float v966_data = s0[v965_a];
            float v967_data = s1[34];
            float v969_data = ir6[3];
            ir6[3] = (v969_data + (v966_data * v967_data));
            int32_t v976_a = v212_lead + 224;
            float v977_data = s0[v976_a];
            float v978_data = s1[43];
            float v980_data = ir6[4];
            ir6[4] = (v980_data + (v977_data * v978_data));
            int32_t v987_a = v212_lead + 224;
            float v988_data = s0[v987_a];
            float v989_data = s1[52];
            float v991_data = ir6[5];
            ir6[5] = (v991_data + (v988_data * v989_data));
            int32_t v998_a = v212_lead + 224;
            float v999_data = s0[v998_a];
            float v1000_data = s1[61];
            float v1002_data = ir6[6];
            ir6[6] = (v1002_data + (v999_data * v1000_data));
            int32_t v1009_a = v212_lead + 224;
            float v1010_data = s0[v1009_a];
            float v1011_data = s1[70];
            float v1013_data = ir6[7];
            ir6[7] = (v1013_data + (v1010_data * v1011_data));
            int32_t v1020_a = v212_lead + 224;
            float v1021_data = s0[v1020_a];
            float v1022_data = s1[79];
            float v1024_data = ir6[8];
            ir6[8] = (v1024_data + (v1021_data * v1022_data));
            int32_t v1034_a = v212_lead + 256;
            float v1035_data = s0[v1034_a];
            float v1036_data = s1[8];
            float v1038_data = ir6[0];
            ir6[0] = (v1038_data + (v1035_data * v1036_data));
            int32_t v1045_a = v212_lead + 256;
            float v1046_data = s0[v1045_a];
            float v1047_data = s1[17];
            float v1049_data = ir6[1];
            ir6[1] = (v1049_data + (v1046_data * v1047_data));
            int32_t v1056_a = v212_lead + 256;
            float v1057_data = s0[v1056_a];
            float v1058_data = s1[26];
            float v1060_data = ir6[2];
            ir6[2] = (v1060_data + (v1057_data * v1058_data));
            int32_t v1067_a = v212_lead + 256;
            float v1068_data = s0[v1067_a];
            float v1069_data = s1[35];
            float v1071_data = ir6[3];
            ir6[3] = (v1071_data + (v1068_data * v1069_data));
            int32_t v1078_a = v212_lead + 256;
            float v1079_data = s0[v1078_a];
            float v1080_data = s1[44];
            float v1082_data = ir6[4];
            ir6[4] = (v1082_data + (v1079_data * v1080_data));
            int32_t v1089_a = v212_lead + 256;
            float v1090_data = s0[v1089_a];
            float v1091_data = s1[53];
            float v1093_data = ir6[5];
            ir6[5] = (v1093_data + (v1090_data * v1091_data));
            int32_t v1100_a = v212_lead + 256;
            float v1101_data = s0[v1100_a];
            float v1102_data = s1[62];
            float v1104_data = ir6[6];
            ir6[6] = (v1104_data + (v1101_data * v1102_data));
            int32_t v1111_a = v212_lead + 256;
            float v1112_data = s0[v1111_a];
            float v1113_data = s1[71];
            float v1115_data = ir6[7];
            ir6[7] = (v1115_data + (v1112_data * v1113_data));
            int32_t v1122_a = v212_lead + 256;
            float v1123_data = s0[v1122_a];
            float v1124_data = s1[80];
            float v1126_data = ir6[8];
            ir6[8] = (v1126_data + (v1123_data * v1124_data));
            #pragma unroll
            for (int32_t v1131_n0 = 0; v1131_n0 < 1; ++v1131_n0) {
              #pragma unroll
              for (int32_t v1132_n1 = 0; v1132_n1 < 9; ++v1132_n1) {
                int32_t v1133_a = v1131_n0 + v1132_n1;
                float v1134_data = ir6[v1133_a];
                int32_t v1135_a = v1131_n0 + v1132_n1;
                r6[v1135_a] = v1134_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r6);
          int32_t v1138_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v1139_i0 = 0; v1139_i0 < 1; ++v1139_i0) {
            int32_t v1147_lead = v1138_lead + (v1139_i0 * 32);
            #pragma unroll
            for (int32_t v1140_i1 = 0; v1140_i1 < 9; ++v1140_i1) {
              int32_t v1141_a = v1139_i0 + v1140_i1;
              float v1142_data = r6[v1141_a];
              int32_t v1149_a = v1147_lead + (v1140_i1 * 32);
              glb_m3[v1149_a] = v1142_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

