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
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              int32_t v38_a = v32_i1 * 12;
              int32_t v39_a = v7_lead + v38_a;
              float v47_data = __ldcg(&glb_m3[(v7_lead + v38_a)]);
              int32_t v48_a = 0 + v32_i1;
              r2[v48_a] = v47_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v7_lead < 12) {
            float v54_data = r0[0];
            float v55_data = s0[0];
            float v57_data = r1[0];
            r1[0] = (v57_data + (v54_data * v55_data));
            float v60_data = s0[6];
            float v62_data = r1[1];
            r1[1] = (v62_data + (v54_data * v60_data));
            float v65_data = s0[12];
            float v67_data = r1[2];
            r1[2] = (v67_data + (v54_data * v65_data));
            float v70_data = s0[18];
            float v72_data = r1[3];
            r1[3] = (v72_data + (v54_data * v70_data));
            float v75_data = s0[24];
            float v77_data = r1[4];
            r1[4] = (v77_data + (v54_data * v75_data));
            float v80_data = s0[30];
            float v82_data = r1[5];
            r1[5] = (v82_data + (v54_data * v80_data));
          }
          if (v7_lead < 12) {
            float v88_data = r0[1];
            float v89_data = s0[1];
            float v91_data = r1[0];
            r1[0] = (v91_data + (v88_data * v89_data));
            float v94_data = s0[7];
            float v96_data = r1[1];
            r1[1] = (v96_data + (v88_data * v94_data));
            float v99_data = s0[13];
            float v101_data = r1[2];
            r1[2] = (v101_data + (v88_data * v99_data));
            float v104_data = s0[19];
            float v106_data = r1[3];
            r1[3] = (v106_data + (v88_data * v104_data));
            float v109_data = s0[25];
            float v111_data = r1[4];
            r1[4] = (v111_data + (v88_data * v109_data));
            float v114_data = s0[31];
            float v116_data = r1[5];
            r1[5] = (v116_data + (v88_data * v114_data));
          }
          if (v7_lead < 12) {
            float v122_data = r0[2];
            float v123_data = s0[2];
            float v125_data = r1[0];
            r1[0] = (v125_data + (v122_data * v123_data));
            float v128_data = s0[8];
            float v130_data = r1[1];
            r1[1] = (v130_data + (v122_data * v128_data));
            float v133_data = s0[14];
            float v135_data = r1[2];
            r1[2] = (v135_data + (v122_data * v133_data));
            float v138_data = s0[20];
            float v140_data = r1[3];
            r1[3] = (v140_data + (v122_data * v138_data));
            float v143_data = s0[26];
            float v145_data = r1[4];
            r1[4] = (v145_data + (v122_data * v143_data));
            float v148_data = s0[32];
            float v150_data = r1[5];
            r1[5] = (v150_data + (v122_data * v148_data));
          }
          if (v7_lead < 12) {
            float v156_data = r0[3];
            float v157_data = s0[3];
            float v159_data = r1[0];
            r1[0] = (v159_data + (v156_data * v157_data));
            float v162_data = s0[9];
            float v164_data = r1[1];
            r1[1] = (v164_data + (v156_data * v162_data));
            float v167_data = s0[15];
            float v169_data = r1[2];
            r1[2] = (v169_data + (v156_data * v167_data));
            float v172_data = s0[21];
            float v174_data = r1[3];
            r1[3] = (v174_data + (v156_data * v172_data));
            float v177_data = s0[27];
            float v179_data = r1[4];
            r1[4] = (v179_data + (v156_data * v177_data));
            float v182_data = s0[33];
            float v184_data = r1[5];
            r1[5] = (v184_data + (v156_data * v182_data));
          }
          if (v7_lead < 12) {
            float v190_data = r0[4];
            float v191_data = s0[4];
            float v193_data = r1[0];
            r1[0] = (v193_data + (v190_data * v191_data));
            float v196_data = s0[10];
            float v198_data = r1[1];
            r1[1] = (v198_data + (v190_data * v196_data));
            float v201_data = s0[16];
            float v203_data = r1[2];
            r1[2] = (v203_data + (v190_data * v201_data));
            float v206_data = s0[22];
            float v208_data = r1[3];
            r1[3] = (v208_data + (v190_data * v206_data));
            float v211_data = s0[28];
            float v213_data = r1[4];
            r1[4] = (v213_data + (v190_data * v211_data));
            float v216_data = s0[34];
            float v218_data = r1[5];
            r1[5] = (v218_data + (v190_data * v216_data));
          }
          if (v7_lead < 12) {
            float v224_data = r0[5];
            float v225_data = s0[5];
            float v227_data = r1[0];
            r1[0] = (v227_data + (v224_data * v225_data));
            float v230_data = s0[11];
            float v232_data = r1[1];
            r1[1] = (v232_data + (v224_data * v230_data));
            float v235_data = s0[17];
            float v237_data = r1[2];
            r1[2] = (v237_data + (v224_data * v235_data));
            float v240_data = s0[23];
            float v242_data = r1[3];
            r1[3] = (v242_data + (v224_data * v240_data));
            float v245_data = s0[29];
            float v247_data = r1[4];
            r1[4] = (v247_data + (v224_data * v245_data));
            float v250_data = s0[35];
            float v252_data = r1[5];
            r1[5] = (v252_data + (v224_data * v250_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v259_i1 = 0; v259_i1 < 6; ++v259_i1) {
              int32_t v260_a = 0 + v259_i1;
              float v262_data = r1[v259_i1];
              int32_t v269_a = v7_lead + (v259_i1 * 12);
              s1[v269_a] = v262_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v7_lead < 12) {
            float v276_data = r2[0];
            float v277_data = s1[0];
            float v279_data = ir3[0];
            ir3[0] = (v279_data + (v276_data * v277_data));
            float v282_data = s1[12];
            float v284_data = ir3[1];
            ir3[1] = (v284_data + (v276_data * v282_data));
            float v287_data = s1[24];
            float v289_data = ir3[2];
            ir3[2] = (v289_data + (v276_data * v287_data));
            float v292_data = s1[36];
            float v294_data = ir3[3];
            ir3[3] = (v294_data + (v276_data * v292_data));
            float v297_data = s1[48];
            float v299_data = ir3[4];
            ir3[4] = (v299_data + (v276_data * v297_data));
            float v302_data = s1[60];
            float v304_data = ir3[5];
            ir3[5] = (v304_data + (v276_data * v302_data));
          }
          if (v7_lead < 12) {
            float v310_data = r2[1];
            float v311_data = s1[1];
            float v313_data = ir3[0];
            ir3[0] = (v313_data + (v310_data * v311_data));
            float v316_data = s1[13];
            float v318_data = ir3[1];
            ir3[1] = (v318_data + (v310_data * v316_data));
            float v321_data = s1[25];
            float v323_data = ir3[2];
            ir3[2] = (v323_data + (v310_data * v321_data));
            float v326_data = s1[37];
            float v328_data = ir3[3];
            ir3[3] = (v328_data + (v310_data * v326_data));
            float v331_data = s1[49];
            float v333_data = ir3[4];
            ir3[4] = (v333_data + (v310_data * v331_data));
            float v336_data = s1[61];
            float v338_data = ir3[5];
            ir3[5] = (v338_data + (v310_data * v336_data));
          }
          if (v7_lead < 12) {
            float v344_data = r2[2];
            float v345_data = s1[2];
            float v347_data = ir3[0];
            ir3[0] = (v347_data + (v344_data * v345_data));
            float v350_data = s1[14];
            float v352_data = ir3[1];
            ir3[1] = (v352_data + (v344_data * v350_data));
            float v355_data = s1[26];
            float v357_data = ir3[2];
            ir3[2] = (v357_data + (v344_data * v355_data));
            float v360_data = s1[38];
            float v362_data = ir3[3];
            ir3[3] = (v362_data + (v344_data * v360_data));
            float v365_data = s1[50];
            float v367_data = ir3[4];
            ir3[4] = (v367_data + (v344_data * v365_data));
            float v370_data = s1[62];
            float v372_data = ir3[5];
            ir3[5] = (v372_data + (v344_data * v370_data));
          }
          if (v7_lead < 12) {
            float v378_data = r2[3];
            float v379_data = s1[3];
            float v381_data = ir3[0];
            ir3[0] = (v381_data + (v378_data * v379_data));
            float v384_data = s1[15];
            float v386_data = ir3[1];
            ir3[1] = (v386_data + (v378_data * v384_data));
            float v389_data = s1[27];
            float v391_data = ir3[2];
            ir3[2] = (v391_data + (v378_data * v389_data));
            float v394_data = s1[39];
            float v396_data = ir3[3];
            ir3[3] = (v396_data + (v378_data * v394_data));
            float v399_data = s1[51];
            float v401_data = ir3[4];
            ir3[4] = (v401_data + (v378_data * v399_data));
            float v404_data = s1[63];
            float v406_data = ir3[5];
            ir3[5] = (v406_data + (v378_data * v404_data));
          }
          if (v7_lead < 12) {
            float v412_data = r2[4];
            float v413_data = s1[4];
            float v415_data = ir3[0];
            ir3[0] = (v415_data + (v412_data * v413_data));
            float v418_data = s1[16];
            float v420_data = ir3[1];
            ir3[1] = (v420_data + (v412_data * v418_data));
            float v423_data = s1[28];
            float v425_data = ir3[2];
            ir3[2] = (v425_data + (v412_data * v423_data));
            float v428_data = s1[40];
            float v430_data = ir3[3];
            ir3[3] = (v430_data + (v412_data * v428_data));
            float v433_data = s1[52];
            float v435_data = ir3[4];
            ir3[4] = (v435_data + (v412_data * v433_data));
            float v438_data = s1[64];
            float v440_data = ir3[5];
            ir3[5] = (v440_data + (v412_data * v438_data));
          }
          if (v7_lead < 12) {
            float v446_data = r2[5];
            float v447_data = s1[5];
            float v449_data = ir3[0];
            ir3[0] = (v449_data + (v446_data * v447_data));
            float v452_data = s1[17];
            float v454_data = ir3[1];
            ir3[1] = (v454_data + (v446_data * v452_data));
            float v457_data = s1[29];
            float v459_data = ir3[2];
            ir3[2] = (v459_data + (v446_data * v457_data));
            float v462_data = s1[41];
            float v464_data = ir3[3];
            ir3[3] = (v464_data + (v446_data * v462_data));
            float v467_data = s1[53];
            float v469_data = ir3[4];
            ir3[4] = (v469_data + (v446_data * v467_data));
            float v472_data = s1[65];
            float v474_data = ir3[5];
            ir3[5] = (v474_data + (v446_data * v472_data));
          }
          if (v7_lead < 12) {
            float v480_data = r2[6];
            float v481_data = s1[6];
            float v483_data = ir3[0];
            ir3[0] = (v483_data + (v480_data * v481_data));
            float v486_data = s1[18];
            float v488_data = ir3[1];
            ir3[1] = (v488_data + (v480_data * v486_data));
            float v491_data = s1[30];
            float v493_data = ir3[2];
            ir3[2] = (v493_data + (v480_data * v491_data));
            float v496_data = s1[42];
            float v498_data = ir3[3];
            ir3[3] = (v498_data + (v480_data * v496_data));
            float v501_data = s1[54];
            float v503_data = ir3[4];
            ir3[4] = (v503_data + (v480_data * v501_data));
            float v506_data = s1[66];
            float v508_data = ir3[5];
            ir3[5] = (v508_data + (v480_data * v506_data));
          }
          if (v7_lead < 12) {
            float v514_data = r2[7];
            float v515_data = s1[7];
            float v517_data = ir3[0];
            ir3[0] = (v517_data + (v514_data * v515_data));
            float v520_data = s1[19];
            float v522_data = ir3[1];
            ir3[1] = (v522_data + (v514_data * v520_data));
            float v525_data = s1[31];
            float v527_data = ir3[2];
            ir3[2] = (v527_data + (v514_data * v525_data));
            float v530_data = s1[43];
            float v532_data = ir3[3];
            ir3[3] = (v532_data + (v514_data * v530_data));
            float v535_data = s1[55];
            float v537_data = ir3[4];
            ir3[4] = (v537_data + (v514_data * v535_data));
            float v540_data = s1[67];
            float v542_data = ir3[5];
            ir3[5] = (v542_data + (v514_data * v540_data));
          }
          if (v7_lead < 12) {
            float v548_data = r2[8];
            float v549_data = s1[8];
            float v551_data = ir3[0];
            ir3[0] = (v551_data + (v548_data * v549_data));
            float v554_data = s1[20];
            float v556_data = ir3[1];
            ir3[1] = (v556_data + (v548_data * v554_data));
            float v559_data = s1[32];
            float v561_data = ir3[2];
            ir3[2] = (v561_data + (v548_data * v559_data));
            float v564_data = s1[44];
            float v566_data = ir3[3];
            ir3[3] = (v566_data + (v548_data * v564_data));
            float v569_data = s1[56];
            float v571_data = ir3[4];
            ir3[4] = (v571_data + (v548_data * v569_data));
            float v574_data = s1[68];
            float v576_data = ir3[5];
            ir3[5] = (v576_data + (v548_data * v574_data));
          }
          if (v7_lead < 12) {
            float v582_data = r2[9];
            float v583_data = s1[9];
            float v585_data = ir3[0];
            ir3[0] = (v585_data + (v582_data * v583_data));
            float v588_data = s1[21];
            float v590_data = ir3[1];
            ir3[1] = (v590_data + (v582_data * v588_data));
            float v593_data = s1[33];
            float v595_data = ir3[2];
            ir3[2] = (v595_data + (v582_data * v593_data));
            float v598_data = s1[45];
            float v600_data = ir3[3];
            ir3[3] = (v600_data + (v582_data * v598_data));
            float v603_data = s1[57];
            float v605_data = ir3[4];
            ir3[4] = (v605_data + (v582_data * v603_data));
            float v608_data = s1[69];
            float v610_data = ir3[5];
            ir3[5] = (v610_data + (v582_data * v608_data));
          }
          if (v7_lead < 12) {
            float v616_data = r2[10];
            float v617_data = s1[10];
            float v619_data = ir3[0];
            ir3[0] = (v619_data + (v616_data * v617_data));
            float v622_data = s1[22];
            float v624_data = ir3[1];
            ir3[1] = (v624_data + (v616_data * v622_data));
            float v627_data = s1[34];
            float v629_data = ir3[2];
            ir3[2] = (v629_data + (v616_data * v627_data));
            float v632_data = s1[46];
            float v634_data = ir3[3];
            ir3[3] = (v634_data + (v616_data * v632_data));
            float v637_data = s1[58];
            float v639_data = ir3[4];
            ir3[4] = (v639_data + (v616_data * v637_data));
            float v642_data = s1[70];
            float v644_data = ir3[5];
            ir3[5] = (v644_data + (v616_data * v642_data));
          }
          if (v7_lead < 12) {
            float v650_data = r2[11];
            float v651_data = s1[11];
            float v653_data = ir3[0];
            ir3[0] = (v653_data + (v650_data * v651_data));
            float v656_data = s1[23];
            float v658_data = ir3[1];
            ir3[1] = (v658_data + (v650_data * v656_data));
            float v661_data = s1[35];
            float v663_data = ir3[2];
            ir3[2] = (v663_data + (v650_data * v661_data));
            float v666_data = s1[47];
            float v668_data = ir3[3];
            ir3[3] = (v668_data + (v650_data * v666_data));
            float v671_data = s1[59];
            float v673_data = ir3[4];
            ir3[4] = (v673_data + (v650_data * v671_data));
            float v676_data = s1[71];
            float v678_data = ir3[5];
            ir3[5] = (v678_data + (v650_data * v676_data));
          }
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v684_n1 = 0; v684_n1 < 6; ++v684_n1) {
              int32_t v685_a = 0 + v684_n1;
              float v687_data = ir3[v684_n1];
              int32_t v688_a = 0 + v684_n1;
              r3[v684_n1] = v687_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v694_i1 = 0; v694_i1 < 6; ++v694_i1) {
              int32_t v695_a = 0 + v694_i1;
              float v697_data = r3[v694_i1];
              int32_t v704_a = v7_lead + (v694_i1 * 12);
              glb_m2[v704_a] = v697_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

