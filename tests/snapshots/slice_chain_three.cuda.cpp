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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 6; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m0[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], cuda::aligned_size_t<4>(4), pipeline);
          if (threadIdx.x < 4) {
            cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              int32_t v33_a = v27_i1 * 12;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __ldcg(&glb_m3[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          auto& ir1 = r1;
          if (v3_lead < 12) {
            float v49_data = r0[0];
            float v50_data = s0[0];
            float v52_data = ir1[0];
            ir1[0] = (v52_data + (v49_data * v50_data));
            float v55_data = s0[6];
            float v57_data = ir1[1];
            ir1[1] = (v57_data + (v49_data * v55_data));
            float v60_data = s0[12];
            float v62_data = ir1[2];
            ir1[2] = (v62_data + (v49_data * v60_data));
            float v65_data = s0[18];
            float v67_data = ir1[3];
            ir1[3] = (v67_data + (v49_data * v65_data));
            float v70_data = s0[24];
            float v72_data = ir1[4];
            ir1[4] = (v72_data + (v49_data * v70_data));
            float v75_data = s0[30];
            float v77_data = ir1[5];
            ir1[5] = (v77_data + (v49_data * v75_data));
          }
          if (v3_lead < 12) {
            float v83_data = r0[1];
            float v84_data = s0[1];
            float v86_data = ir1[0];
            ir1[0] = (v86_data + (v83_data * v84_data));
            float v89_data = s0[7];
            float v91_data = ir1[1];
            ir1[1] = (v91_data + (v83_data * v89_data));
            float v94_data = s0[13];
            float v96_data = ir1[2];
            ir1[2] = (v96_data + (v83_data * v94_data));
            float v99_data = s0[19];
            float v101_data = ir1[3];
            ir1[3] = (v101_data + (v83_data * v99_data));
            float v104_data = s0[25];
            float v106_data = ir1[4];
            ir1[4] = (v106_data + (v83_data * v104_data));
            float v109_data = s0[31];
            float v111_data = ir1[5];
            ir1[5] = (v111_data + (v83_data * v109_data));
          }
          if (v3_lead < 12) {
            float v117_data = r0[2];
            float v118_data = s0[2];
            float v120_data = ir1[0];
            ir1[0] = (v120_data + (v117_data * v118_data));
            float v123_data = s0[8];
            float v125_data = ir1[1];
            ir1[1] = (v125_data + (v117_data * v123_data));
            float v128_data = s0[14];
            float v130_data = ir1[2];
            ir1[2] = (v130_data + (v117_data * v128_data));
            float v133_data = s0[20];
            float v135_data = ir1[3];
            ir1[3] = (v135_data + (v117_data * v133_data));
            float v138_data = s0[26];
            float v140_data = ir1[4];
            ir1[4] = (v140_data + (v117_data * v138_data));
            float v143_data = s0[32];
            float v145_data = ir1[5];
            ir1[5] = (v145_data + (v117_data * v143_data));
          }
          if (v3_lead < 12) {
            float v151_data = r0[3];
            float v152_data = s0[3];
            float v154_data = ir1[0];
            ir1[0] = (v154_data + (v151_data * v152_data));
            float v157_data = s0[9];
            float v159_data = ir1[1];
            ir1[1] = (v159_data + (v151_data * v157_data));
            float v162_data = s0[15];
            float v164_data = ir1[2];
            ir1[2] = (v164_data + (v151_data * v162_data));
            float v167_data = s0[21];
            float v169_data = ir1[3];
            ir1[3] = (v169_data + (v151_data * v167_data));
            float v172_data = s0[27];
            float v174_data = ir1[4];
            ir1[4] = (v174_data + (v151_data * v172_data));
            float v177_data = s0[33];
            float v179_data = ir1[5];
            ir1[5] = (v179_data + (v151_data * v177_data));
          }
          if (v3_lead < 12) {
            float v185_data = r0[4];
            float v186_data = s0[4];
            float v188_data = ir1[0];
            ir1[0] = (v188_data + (v185_data * v186_data));
            float v191_data = s0[10];
            float v193_data = ir1[1];
            ir1[1] = (v193_data + (v185_data * v191_data));
            float v196_data = s0[16];
            float v198_data = ir1[2];
            ir1[2] = (v198_data + (v185_data * v196_data));
            float v201_data = s0[22];
            float v203_data = ir1[3];
            ir1[3] = (v203_data + (v185_data * v201_data));
            float v206_data = s0[28];
            float v208_data = ir1[4];
            ir1[4] = (v208_data + (v185_data * v206_data));
            float v211_data = s0[34];
            float v213_data = ir1[5];
            ir1[5] = (v213_data + (v185_data * v211_data));
          }
          if (v3_lead < 12) {
            float v219_data = r0[5];
            float v220_data = s0[5];
            float v222_data = ir1[0];
            ir1[0] = (v222_data + (v219_data * v220_data));
            float v225_data = s0[11];
            float v227_data = ir1[1];
            ir1[1] = (v227_data + (v219_data * v225_data));
            float v230_data = s0[17];
            float v232_data = ir1[2];
            ir1[2] = (v232_data + (v219_data * v230_data));
            float v235_data = s0[23];
            float v237_data = ir1[3];
            ir1[3] = (v237_data + (v219_data * v235_data));
            float v240_data = s0[29];
            float v242_data = ir1[4];
            ir1[4] = (v242_data + (v219_data * v240_data));
            float v245_data = s0[35];
            float v247_data = ir1[5];
            ir1[5] = (v247_data + (v219_data * v245_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v253_i1 = 0; v253_i1 < 6; ++v253_i1) {
              int32_t v254_a = 0 + v253_i1;
              float v256_data = r1[v253_i1];
              int32_t v263_a = v3_lead + (v253_i1 * 12);
              s1[v263_a] = v256_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v3_lead < 12) {
            float v270_data = r2[0];
            float v271_data = s1[0];
            float v273_data = ir3[0];
            ir3[0] = (v273_data + (v270_data * v271_data));
            float v276_data = s1[12];
            float v278_data = ir3[1];
            ir3[1] = (v278_data + (v270_data * v276_data));
            float v281_data = s1[24];
            float v283_data = ir3[2];
            ir3[2] = (v283_data + (v270_data * v281_data));
            float v286_data = s1[36];
            float v288_data = ir3[3];
            ir3[3] = (v288_data + (v270_data * v286_data));
            float v291_data = s1[48];
            float v293_data = ir3[4];
            ir3[4] = (v293_data + (v270_data * v291_data));
            float v296_data = s1[60];
            float v298_data = ir3[5];
            ir3[5] = (v298_data + (v270_data * v296_data));
          }
          if (v3_lead < 12) {
            float v304_data = r2[1];
            float v305_data = s1[1];
            float v307_data = ir3[0];
            ir3[0] = (v307_data + (v304_data * v305_data));
            float v310_data = s1[13];
            float v312_data = ir3[1];
            ir3[1] = (v312_data + (v304_data * v310_data));
            float v315_data = s1[25];
            float v317_data = ir3[2];
            ir3[2] = (v317_data + (v304_data * v315_data));
            float v320_data = s1[37];
            float v322_data = ir3[3];
            ir3[3] = (v322_data + (v304_data * v320_data));
            float v325_data = s1[49];
            float v327_data = ir3[4];
            ir3[4] = (v327_data + (v304_data * v325_data));
            float v330_data = s1[61];
            float v332_data = ir3[5];
            ir3[5] = (v332_data + (v304_data * v330_data));
          }
          if (v3_lead < 12) {
            float v338_data = r2[2];
            float v339_data = s1[2];
            float v341_data = ir3[0];
            ir3[0] = (v341_data + (v338_data * v339_data));
            float v344_data = s1[14];
            float v346_data = ir3[1];
            ir3[1] = (v346_data + (v338_data * v344_data));
            float v349_data = s1[26];
            float v351_data = ir3[2];
            ir3[2] = (v351_data + (v338_data * v349_data));
            float v354_data = s1[38];
            float v356_data = ir3[3];
            ir3[3] = (v356_data + (v338_data * v354_data));
            float v359_data = s1[50];
            float v361_data = ir3[4];
            ir3[4] = (v361_data + (v338_data * v359_data));
            float v364_data = s1[62];
            float v366_data = ir3[5];
            ir3[5] = (v366_data + (v338_data * v364_data));
          }
          if (v3_lead < 12) {
            float v372_data = r2[3];
            float v373_data = s1[3];
            float v375_data = ir3[0];
            ir3[0] = (v375_data + (v372_data * v373_data));
            float v378_data = s1[15];
            float v380_data = ir3[1];
            ir3[1] = (v380_data + (v372_data * v378_data));
            float v383_data = s1[27];
            float v385_data = ir3[2];
            ir3[2] = (v385_data + (v372_data * v383_data));
            float v388_data = s1[39];
            float v390_data = ir3[3];
            ir3[3] = (v390_data + (v372_data * v388_data));
            float v393_data = s1[51];
            float v395_data = ir3[4];
            ir3[4] = (v395_data + (v372_data * v393_data));
            float v398_data = s1[63];
            float v400_data = ir3[5];
            ir3[5] = (v400_data + (v372_data * v398_data));
          }
          if (v3_lead < 12) {
            float v406_data = r2[4];
            float v407_data = s1[4];
            float v409_data = ir3[0];
            ir3[0] = (v409_data + (v406_data * v407_data));
            float v412_data = s1[16];
            float v414_data = ir3[1];
            ir3[1] = (v414_data + (v406_data * v412_data));
            float v417_data = s1[28];
            float v419_data = ir3[2];
            ir3[2] = (v419_data + (v406_data * v417_data));
            float v422_data = s1[40];
            float v424_data = ir3[3];
            ir3[3] = (v424_data + (v406_data * v422_data));
            float v427_data = s1[52];
            float v429_data = ir3[4];
            ir3[4] = (v429_data + (v406_data * v427_data));
            float v432_data = s1[64];
            float v434_data = ir3[5];
            ir3[5] = (v434_data + (v406_data * v432_data));
          }
          if (v3_lead < 12) {
            float v440_data = r2[5];
            float v441_data = s1[5];
            float v443_data = ir3[0];
            ir3[0] = (v443_data + (v440_data * v441_data));
            float v446_data = s1[17];
            float v448_data = ir3[1];
            ir3[1] = (v448_data + (v440_data * v446_data));
            float v451_data = s1[29];
            float v453_data = ir3[2];
            ir3[2] = (v453_data + (v440_data * v451_data));
            float v456_data = s1[41];
            float v458_data = ir3[3];
            ir3[3] = (v458_data + (v440_data * v456_data));
            float v461_data = s1[53];
            float v463_data = ir3[4];
            ir3[4] = (v463_data + (v440_data * v461_data));
            float v466_data = s1[65];
            float v468_data = ir3[5];
            ir3[5] = (v468_data + (v440_data * v466_data));
          }
          if (v3_lead < 12) {
            float v474_data = r2[6];
            float v475_data = s1[6];
            float v477_data = ir3[0];
            ir3[0] = (v477_data + (v474_data * v475_data));
            float v480_data = s1[18];
            float v482_data = ir3[1];
            ir3[1] = (v482_data + (v474_data * v480_data));
            float v485_data = s1[30];
            float v487_data = ir3[2];
            ir3[2] = (v487_data + (v474_data * v485_data));
            float v490_data = s1[42];
            float v492_data = ir3[3];
            ir3[3] = (v492_data + (v474_data * v490_data));
            float v495_data = s1[54];
            float v497_data = ir3[4];
            ir3[4] = (v497_data + (v474_data * v495_data));
            float v500_data = s1[66];
            float v502_data = ir3[5];
            ir3[5] = (v502_data + (v474_data * v500_data));
          }
          if (v3_lead < 12) {
            float v508_data = r2[7];
            float v509_data = s1[7];
            float v511_data = ir3[0];
            ir3[0] = (v511_data + (v508_data * v509_data));
            float v514_data = s1[19];
            float v516_data = ir3[1];
            ir3[1] = (v516_data + (v508_data * v514_data));
            float v519_data = s1[31];
            float v521_data = ir3[2];
            ir3[2] = (v521_data + (v508_data * v519_data));
            float v524_data = s1[43];
            float v526_data = ir3[3];
            ir3[3] = (v526_data + (v508_data * v524_data));
            float v529_data = s1[55];
            float v531_data = ir3[4];
            ir3[4] = (v531_data + (v508_data * v529_data));
            float v534_data = s1[67];
            float v536_data = ir3[5];
            ir3[5] = (v536_data + (v508_data * v534_data));
          }
          if (v3_lead < 12) {
            float v542_data = r2[8];
            float v543_data = s1[8];
            float v545_data = ir3[0];
            ir3[0] = (v545_data + (v542_data * v543_data));
            float v548_data = s1[20];
            float v550_data = ir3[1];
            ir3[1] = (v550_data + (v542_data * v548_data));
            float v553_data = s1[32];
            float v555_data = ir3[2];
            ir3[2] = (v555_data + (v542_data * v553_data));
            float v558_data = s1[44];
            float v560_data = ir3[3];
            ir3[3] = (v560_data + (v542_data * v558_data));
            float v563_data = s1[56];
            float v565_data = ir3[4];
            ir3[4] = (v565_data + (v542_data * v563_data));
            float v568_data = s1[68];
            float v570_data = ir3[5];
            ir3[5] = (v570_data + (v542_data * v568_data));
          }
          if (v3_lead < 12) {
            float v576_data = r2[9];
            float v577_data = s1[9];
            float v579_data = ir3[0];
            ir3[0] = (v579_data + (v576_data * v577_data));
            float v582_data = s1[21];
            float v584_data = ir3[1];
            ir3[1] = (v584_data + (v576_data * v582_data));
            float v587_data = s1[33];
            float v589_data = ir3[2];
            ir3[2] = (v589_data + (v576_data * v587_data));
            float v592_data = s1[45];
            float v594_data = ir3[3];
            ir3[3] = (v594_data + (v576_data * v592_data));
            float v597_data = s1[57];
            float v599_data = ir3[4];
            ir3[4] = (v599_data + (v576_data * v597_data));
            float v602_data = s1[69];
            float v604_data = ir3[5];
            ir3[5] = (v604_data + (v576_data * v602_data));
          }
          if (v3_lead < 12) {
            float v610_data = r2[10];
            float v611_data = s1[10];
            float v613_data = ir3[0];
            ir3[0] = (v613_data + (v610_data * v611_data));
            float v616_data = s1[22];
            float v618_data = ir3[1];
            ir3[1] = (v618_data + (v610_data * v616_data));
            float v621_data = s1[34];
            float v623_data = ir3[2];
            ir3[2] = (v623_data + (v610_data * v621_data));
            float v626_data = s1[46];
            float v628_data = ir3[3];
            ir3[3] = (v628_data + (v610_data * v626_data));
            float v631_data = s1[58];
            float v633_data = ir3[4];
            ir3[4] = (v633_data + (v610_data * v631_data));
            float v636_data = s1[70];
            float v638_data = ir3[5];
            ir3[5] = (v638_data + (v610_data * v636_data));
          }
          if (v3_lead < 12) {
            float v644_data = r2[11];
            float v645_data = s1[11];
            float v647_data = ir3[0];
            ir3[0] = (v647_data + (v644_data * v645_data));
            float v650_data = s1[23];
            float v652_data = ir3[1];
            ir3[1] = (v652_data + (v644_data * v650_data));
            float v655_data = s1[35];
            float v657_data = ir3[2];
            ir3[2] = (v657_data + (v644_data * v655_data));
            float v660_data = s1[47];
            float v662_data = ir3[3];
            ir3[3] = (v662_data + (v644_data * v660_data));
            float v665_data = s1[59];
            float v667_data = ir3[4];
            ir3[4] = (v667_data + (v644_data * v665_data));
            float v670_data = s1[71];
            float v672_data = ir3[5];
            ir3[5] = (v672_data + (v644_data * v670_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v678_n1 = 0; v678_n1 < 6; ++v678_n1) {
              int32_t v679_a = 0 + v678_n1;
              float v681_data = ir3[v678_n1];
              int32_t v682_a = 0 + v678_n1;
              r3[v678_n1] = v681_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v688_i1 = 0; v688_i1 < 6; ++v688_i1) {
              int32_t v689_a = 0 + v688_i1;
              float v691_data = r3[v688_i1];
              int32_t v698_a = v3_lead + (v688_i1 * 12);
              glb_m2[v698_a] = v691_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

