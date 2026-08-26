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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 6; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __ldcg(&glb_m0[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 12;
              int32_t v32_a = v23_lead + v31_a;
              float v40_data = __ldcg(&glb_m3[(v23_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r2[v41_a] = v40_data;
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
          int32_t v44_lead = threadIdx.x % 16;
          if (v44_lead < 12) {
            float v46_data = r0[0];
            float v47_data = s0[0];
            float v49_data = ir1[0];
            ir1[0] = (v49_data + (v46_data * v47_data));
            float v52_data = s0[6];
            float v54_data = ir1[1];
            ir1[1] = (v54_data + (v46_data * v52_data));
            float v57_data = s0[12];
            float v59_data = ir1[2];
            ir1[2] = (v59_data + (v46_data * v57_data));
            float v62_data = s0[18];
            float v64_data = ir1[3];
            ir1[3] = (v64_data + (v46_data * v62_data));
            float v67_data = s0[24];
            float v69_data = ir1[4];
            ir1[4] = (v69_data + (v46_data * v67_data));
            float v72_data = s0[30];
            float v74_data = ir1[5];
            ir1[5] = (v74_data + (v46_data * v72_data));
          }
          if (v44_lead < 12) {
            float v80_data = r0[1];
            float v81_data = s0[1];
            float v83_data = ir1[0];
            ir1[0] = (v83_data + (v80_data * v81_data));
            float v86_data = s0[7];
            float v88_data = ir1[1];
            ir1[1] = (v88_data + (v80_data * v86_data));
            float v91_data = s0[13];
            float v93_data = ir1[2];
            ir1[2] = (v93_data + (v80_data * v91_data));
            float v96_data = s0[19];
            float v98_data = ir1[3];
            ir1[3] = (v98_data + (v80_data * v96_data));
            float v101_data = s0[25];
            float v103_data = ir1[4];
            ir1[4] = (v103_data + (v80_data * v101_data));
            float v106_data = s0[31];
            float v108_data = ir1[5];
            ir1[5] = (v108_data + (v80_data * v106_data));
          }
          if (v44_lead < 12) {
            float v114_data = r0[2];
            float v115_data = s0[2];
            float v117_data = ir1[0];
            ir1[0] = (v117_data + (v114_data * v115_data));
            float v120_data = s0[8];
            float v122_data = ir1[1];
            ir1[1] = (v122_data + (v114_data * v120_data));
            float v125_data = s0[14];
            float v127_data = ir1[2];
            ir1[2] = (v127_data + (v114_data * v125_data));
            float v130_data = s0[20];
            float v132_data = ir1[3];
            ir1[3] = (v132_data + (v114_data * v130_data));
            float v135_data = s0[26];
            float v137_data = ir1[4];
            ir1[4] = (v137_data + (v114_data * v135_data));
            float v140_data = s0[32];
            float v142_data = ir1[5];
            ir1[5] = (v142_data + (v114_data * v140_data));
          }
          if (v44_lead < 12) {
            float v148_data = r0[3];
            float v149_data = s0[3];
            float v151_data = ir1[0];
            ir1[0] = (v151_data + (v148_data * v149_data));
            float v154_data = s0[9];
            float v156_data = ir1[1];
            ir1[1] = (v156_data + (v148_data * v154_data));
            float v159_data = s0[15];
            float v161_data = ir1[2];
            ir1[2] = (v161_data + (v148_data * v159_data));
            float v164_data = s0[21];
            float v166_data = ir1[3];
            ir1[3] = (v166_data + (v148_data * v164_data));
            float v169_data = s0[27];
            float v171_data = ir1[4];
            ir1[4] = (v171_data + (v148_data * v169_data));
            float v174_data = s0[33];
            float v176_data = ir1[5];
            ir1[5] = (v176_data + (v148_data * v174_data));
          }
          if (v44_lead < 12) {
            float v182_data = r0[4];
            float v183_data = s0[4];
            float v185_data = ir1[0];
            ir1[0] = (v185_data + (v182_data * v183_data));
            float v188_data = s0[10];
            float v190_data = ir1[1];
            ir1[1] = (v190_data + (v182_data * v188_data));
            float v193_data = s0[16];
            float v195_data = ir1[2];
            ir1[2] = (v195_data + (v182_data * v193_data));
            float v198_data = s0[22];
            float v200_data = ir1[3];
            ir1[3] = (v200_data + (v182_data * v198_data));
            float v203_data = s0[28];
            float v205_data = ir1[4];
            ir1[4] = (v205_data + (v182_data * v203_data));
            float v208_data = s0[34];
            float v210_data = ir1[5];
            ir1[5] = (v210_data + (v182_data * v208_data));
          }
          if (v44_lead < 12) {
            float v216_data = r0[5];
            float v217_data = s0[5];
            float v219_data = ir1[0];
            ir1[0] = (v219_data + (v216_data * v217_data));
            float v222_data = s0[11];
            float v224_data = ir1[1];
            ir1[1] = (v224_data + (v216_data * v222_data));
            float v227_data = s0[17];
            float v229_data = ir1[2];
            ir1[2] = (v229_data + (v216_data * v227_data));
            float v232_data = s0[23];
            float v234_data = ir1[3];
            ir1[3] = (v234_data + (v216_data * v232_data));
            float v237_data = s0[29];
            float v239_data = ir1[4];
            ir1[4] = (v239_data + (v216_data * v237_data));
            float v242_data = s0[35];
            float v244_data = ir1[5];
            ir1[5] = (v244_data + (v216_data * v242_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          int32_t v248_lead = threadIdx.x % 16;
          if (v248_lead < 12) {
            #pragma unroll
            for (int32_t v250_i1 = 0; v250_i1 < 6; ++v250_i1) {
              int32_t v251_a = 0 + v250_i1;
              float v253_data = r1[v250_i1];
              int32_t v260_a = v248_lead + (v250_i1 * 12);
              s1[v260_a] = v253_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + None
            // [(0, 12), (0, 6)] [(0, 12)]
            float ir3[6]{};
            int32_t v263_lead = threadIdx.x % 16;
            if (v263_lead < 12) {
              float v265_data = r2[0];
              float v266_data = s1[0];
              float v268_data = ir3[0];
              ir3[0] = (v268_data + (v265_data * v266_data));
              float v271_data = s1[12];
              float v273_data = ir3[1];
              ir3[1] = (v273_data + (v265_data * v271_data));
              float v276_data = s1[24];
              float v278_data = ir3[2];
              ir3[2] = (v278_data + (v265_data * v276_data));
              float v281_data = s1[36];
              float v283_data = ir3[3];
              ir3[3] = (v283_data + (v265_data * v281_data));
              float v286_data = s1[48];
              float v288_data = ir3[4];
              ir3[4] = (v288_data + (v265_data * v286_data));
              float v291_data = s1[60];
              float v293_data = ir3[5];
              ir3[5] = (v293_data + (v265_data * v291_data));
            }
            if (v263_lead < 12) {
              float v299_data = r2[1];
              float v300_data = s1[1];
              float v302_data = ir3[0];
              ir3[0] = (v302_data + (v299_data * v300_data));
              float v305_data = s1[13];
              float v307_data = ir3[1];
              ir3[1] = (v307_data + (v299_data * v305_data));
              float v310_data = s1[25];
              float v312_data = ir3[2];
              ir3[2] = (v312_data + (v299_data * v310_data));
              float v315_data = s1[37];
              float v317_data = ir3[3];
              ir3[3] = (v317_data + (v299_data * v315_data));
              float v320_data = s1[49];
              float v322_data = ir3[4];
              ir3[4] = (v322_data + (v299_data * v320_data));
              float v325_data = s1[61];
              float v327_data = ir3[5];
              ir3[5] = (v327_data + (v299_data * v325_data));
            }
            if (v263_lead < 12) {
              float v333_data = r2[2];
              float v334_data = s1[2];
              float v336_data = ir3[0];
              ir3[0] = (v336_data + (v333_data * v334_data));
              float v339_data = s1[14];
              float v341_data = ir3[1];
              ir3[1] = (v341_data + (v333_data * v339_data));
              float v344_data = s1[26];
              float v346_data = ir3[2];
              ir3[2] = (v346_data + (v333_data * v344_data));
              float v349_data = s1[38];
              float v351_data = ir3[3];
              ir3[3] = (v351_data + (v333_data * v349_data));
              float v354_data = s1[50];
              float v356_data = ir3[4];
              ir3[4] = (v356_data + (v333_data * v354_data));
              float v359_data = s1[62];
              float v361_data = ir3[5];
              ir3[5] = (v361_data + (v333_data * v359_data));
            }
            if (v263_lead < 12) {
              float v367_data = r2[3];
              float v368_data = s1[3];
              float v370_data = ir3[0];
              ir3[0] = (v370_data + (v367_data * v368_data));
              float v373_data = s1[15];
              float v375_data = ir3[1];
              ir3[1] = (v375_data + (v367_data * v373_data));
              float v378_data = s1[27];
              float v380_data = ir3[2];
              ir3[2] = (v380_data + (v367_data * v378_data));
              float v383_data = s1[39];
              float v385_data = ir3[3];
              ir3[3] = (v385_data + (v367_data * v383_data));
              float v388_data = s1[51];
              float v390_data = ir3[4];
              ir3[4] = (v390_data + (v367_data * v388_data));
              float v393_data = s1[63];
              float v395_data = ir3[5];
              ir3[5] = (v395_data + (v367_data * v393_data));
            }
            if (v263_lead < 12) {
              float v401_data = r2[4];
              float v402_data = s1[4];
              float v404_data = ir3[0];
              ir3[0] = (v404_data + (v401_data * v402_data));
              float v407_data = s1[16];
              float v409_data = ir3[1];
              ir3[1] = (v409_data + (v401_data * v407_data));
              float v412_data = s1[28];
              float v414_data = ir3[2];
              ir3[2] = (v414_data + (v401_data * v412_data));
              float v417_data = s1[40];
              float v419_data = ir3[3];
              ir3[3] = (v419_data + (v401_data * v417_data));
              float v422_data = s1[52];
              float v424_data = ir3[4];
              ir3[4] = (v424_data + (v401_data * v422_data));
              float v427_data = s1[64];
              float v429_data = ir3[5];
              ir3[5] = (v429_data + (v401_data * v427_data));
            }
            if (v263_lead < 12) {
              float v435_data = r2[5];
              float v436_data = s1[5];
              float v438_data = ir3[0];
              ir3[0] = (v438_data + (v435_data * v436_data));
              float v441_data = s1[17];
              float v443_data = ir3[1];
              ir3[1] = (v443_data + (v435_data * v441_data));
              float v446_data = s1[29];
              float v448_data = ir3[2];
              ir3[2] = (v448_data + (v435_data * v446_data));
              float v451_data = s1[41];
              float v453_data = ir3[3];
              ir3[3] = (v453_data + (v435_data * v451_data));
              float v456_data = s1[53];
              float v458_data = ir3[4];
              ir3[4] = (v458_data + (v435_data * v456_data));
              float v461_data = s1[65];
              float v463_data = ir3[5];
              ir3[5] = (v463_data + (v435_data * v461_data));
            }
            if (v263_lead < 12) {
              float v469_data = r2[6];
              float v470_data = s1[6];
              float v472_data = ir3[0];
              ir3[0] = (v472_data + (v469_data * v470_data));
              float v475_data = s1[18];
              float v477_data = ir3[1];
              ir3[1] = (v477_data + (v469_data * v475_data));
              float v480_data = s1[30];
              float v482_data = ir3[2];
              ir3[2] = (v482_data + (v469_data * v480_data));
              float v485_data = s1[42];
              float v487_data = ir3[3];
              ir3[3] = (v487_data + (v469_data * v485_data));
              float v490_data = s1[54];
              float v492_data = ir3[4];
              ir3[4] = (v492_data + (v469_data * v490_data));
              float v495_data = s1[66];
              float v497_data = ir3[5];
              ir3[5] = (v497_data + (v469_data * v495_data));
            }
            if (v263_lead < 12) {
              float v503_data = r2[7];
              float v504_data = s1[7];
              float v506_data = ir3[0];
              ir3[0] = (v506_data + (v503_data * v504_data));
              float v509_data = s1[19];
              float v511_data = ir3[1];
              ir3[1] = (v511_data + (v503_data * v509_data));
              float v514_data = s1[31];
              float v516_data = ir3[2];
              ir3[2] = (v516_data + (v503_data * v514_data));
              float v519_data = s1[43];
              float v521_data = ir3[3];
              ir3[3] = (v521_data + (v503_data * v519_data));
              float v524_data = s1[55];
              float v526_data = ir3[4];
              ir3[4] = (v526_data + (v503_data * v524_data));
              float v529_data = s1[67];
              float v531_data = ir3[5];
              ir3[5] = (v531_data + (v503_data * v529_data));
            }
            if (v263_lead < 12) {
              float v537_data = r2[8];
              float v538_data = s1[8];
              float v540_data = ir3[0];
              ir3[0] = (v540_data + (v537_data * v538_data));
              float v543_data = s1[20];
              float v545_data = ir3[1];
              ir3[1] = (v545_data + (v537_data * v543_data));
              float v548_data = s1[32];
              float v550_data = ir3[2];
              ir3[2] = (v550_data + (v537_data * v548_data));
              float v553_data = s1[44];
              float v555_data = ir3[3];
              ir3[3] = (v555_data + (v537_data * v553_data));
              float v558_data = s1[56];
              float v560_data = ir3[4];
              ir3[4] = (v560_data + (v537_data * v558_data));
              float v563_data = s1[68];
              float v565_data = ir3[5];
              ir3[5] = (v565_data + (v537_data * v563_data));
            }
            if (v263_lead < 12) {
              float v571_data = r2[9];
              float v572_data = s1[9];
              float v574_data = ir3[0];
              ir3[0] = (v574_data + (v571_data * v572_data));
              float v577_data = s1[21];
              float v579_data = ir3[1];
              ir3[1] = (v579_data + (v571_data * v577_data));
              float v582_data = s1[33];
              float v584_data = ir3[2];
              ir3[2] = (v584_data + (v571_data * v582_data));
              float v587_data = s1[45];
              float v589_data = ir3[3];
              ir3[3] = (v589_data + (v571_data * v587_data));
              float v592_data = s1[57];
              float v594_data = ir3[4];
              ir3[4] = (v594_data + (v571_data * v592_data));
              float v597_data = s1[69];
              float v599_data = ir3[5];
              ir3[5] = (v599_data + (v571_data * v597_data));
            }
            if (v263_lead < 12) {
              float v605_data = r2[10];
              float v606_data = s1[10];
              float v608_data = ir3[0];
              ir3[0] = (v608_data + (v605_data * v606_data));
              float v611_data = s1[22];
              float v613_data = ir3[1];
              ir3[1] = (v613_data + (v605_data * v611_data));
              float v616_data = s1[34];
              float v618_data = ir3[2];
              ir3[2] = (v618_data + (v605_data * v616_data));
              float v621_data = s1[46];
              float v623_data = ir3[3];
              ir3[3] = (v623_data + (v605_data * v621_data));
              float v626_data = s1[58];
              float v628_data = ir3[4];
              ir3[4] = (v628_data + (v605_data * v626_data));
              float v631_data = s1[70];
              float v633_data = ir3[5];
              ir3[5] = (v633_data + (v605_data * v631_data));
            }
            if (v263_lead < 12) {
              float v639_data = r2[11];
              float v640_data = s1[11];
              float v642_data = ir3[0];
              ir3[0] = (v642_data + (v639_data * v640_data));
              float v645_data = s1[23];
              float v647_data = ir3[1];
              ir3[1] = (v647_data + (v639_data * v645_data));
              float v650_data = s1[35];
              float v652_data = ir3[2];
              ir3[2] = (v652_data + (v639_data * v650_data));
              float v655_data = s1[47];
              float v657_data = ir3[3];
              ir3[3] = (v657_data + (v639_data * v655_data));
              float v660_data = s1[59];
              float v662_data = ir3[4];
              ir3[4] = (v662_data + (v639_data * v660_data));
              float v665_data = s1[71];
              float v667_data = ir3[5];
              ir3[5] = (v667_data + (v639_data * v665_data));
            }
            if (v263_lead < 12) {
              #pragma unroll
              for (int32_t v673_n1 = 0; v673_n1 < 6; ++v673_n1) {
                int32_t v674_a = 0 + v673_n1;
                float v676_data = ir3[v673_n1];
                int32_t v677_a = 0 + v673_n1;
                r3[v673_n1] = v676_data;
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          int32_t v681_lead = threadIdx.x % 16;
          if (v681_lead < 12) {
            #pragma unroll
            for (int32_t v683_i1 = 0; v683_i1 < 6; ++v683_i1) {
              int32_t v684_a = 0 + v683_i1;
              float v686_data = r3[v683_i1];
              int32_t v693_a = v681_lead + (v683_i1 * 12);
              glb_m2[v693_a] = v686_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

