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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v11_lead = threadIdx.x % 16;
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 6; ++v13_i1) {
              int32_t v19_a = v13_i1 * 12;
              int32_t v20_a = v11_lead + v19_a;
              float v28_data = __ldcg(&glb_m0[(v11_lead + v19_a)]);
              r0[v13_i1] = v28_data;
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
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              int32_t v45_a = v39_i1 * 12;
              int32_t v46_a = v11_lead + v45_a;
              float v54_data = __ldcg(&glb_m3[(v11_lead + v45_a)]);
              r2[v39_i1] = v54_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v11_lead < 12) {
            float v61_data = r0[0];
            float v62_data = s0[0];
            float v64_data = r1[0];
            r1[0] = (v64_data + (v61_data * v62_data));
            float v67_data = s0[6];
            float v69_data = r1[1];
            r1[1] = (v69_data + (v61_data * v67_data));
            float v72_data = s0[12];
            float v74_data = r1[2];
            r1[2] = (v74_data + (v61_data * v72_data));
            float v77_data = s0[18];
            float v79_data = r1[3];
            r1[3] = (v79_data + (v61_data * v77_data));
            float v82_data = s0[24];
            float v84_data = r1[4];
            r1[4] = (v84_data + (v61_data * v82_data));
            float v87_data = s0[30];
            float v89_data = r1[5];
            r1[5] = (v89_data + (v61_data * v87_data));
          }
          if (v11_lead < 12) {
            float v95_data = r0[1];
            float v96_data = s0[1];
            float v98_data = r1[0];
            r1[0] = (v98_data + (v95_data * v96_data));
            float v101_data = s0[7];
            float v103_data = r1[1];
            r1[1] = (v103_data + (v95_data * v101_data));
            float v106_data = s0[13];
            float v108_data = r1[2];
            r1[2] = (v108_data + (v95_data * v106_data));
            float v111_data = s0[19];
            float v113_data = r1[3];
            r1[3] = (v113_data + (v95_data * v111_data));
            float v116_data = s0[25];
            float v118_data = r1[4];
            r1[4] = (v118_data + (v95_data * v116_data));
            float v121_data = s0[31];
            float v123_data = r1[5];
            r1[5] = (v123_data + (v95_data * v121_data));
          }
          if (v11_lead < 12) {
            float v129_data = r0[2];
            float v130_data = s0[2];
            float v132_data = r1[0];
            r1[0] = (v132_data + (v129_data * v130_data));
            float v135_data = s0[8];
            float v137_data = r1[1];
            r1[1] = (v137_data + (v129_data * v135_data));
            float v140_data = s0[14];
            float v142_data = r1[2];
            r1[2] = (v142_data + (v129_data * v140_data));
            float v145_data = s0[20];
            float v147_data = r1[3];
            r1[3] = (v147_data + (v129_data * v145_data));
            float v150_data = s0[26];
            float v152_data = r1[4];
            r1[4] = (v152_data + (v129_data * v150_data));
            float v155_data = s0[32];
            float v157_data = r1[5];
            r1[5] = (v157_data + (v129_data * v155_data));
          }
          if (v11_lead < 12) {
            float v163_data = r0[3];
            float v164_data = s0[3];
            float v166_data = r1[0];
            r1[0] = (v166_data + (v163_data * v164_data));
            float v169_data = s0[9];
            float v171_data = r1[1];
            r1[1] = (v171_data + (v163_data * v169_data));
            float v174_data = s0[15];
            float v176_data = r1[2];
            r1[2] = (v176_data + (v163_data * v174_data));
            float v179_data = s0[21];
            float v181_data = r1[3];
            r1[3] = (v181_data + (v163_data * v179_data));
            float v184_data = s0[27];
            float v186_data = r1[4];
            r1[4] = (v186_data + (v163_data * v184_data));
            float v189_data = s0[33];
            float v191_data = r1[5];
            r1[5] = (v191_data + (v163_data * v189_data));
          }
          if (v11_lead < 12) {
            float v197_data = r0[4];
            float v198_data = s0[4];
            float v200_data = r1[0];
            r1[0] = (v200_data + (v197_data * v198_data));
            float v203_data = s0[10];
            float v205_data = r1[1];
            r1[1] = (v205_data + (v197_data * v203_data));
            float v208_data = s0[16];
            float v210_data = r1[2];
            r1[2] = (v210_data + (v197_data * v208_data));
            float v213_data = s0[22];
            float v215_data = r1[3];
            r1[3] = (v215_data + (v197_data * v213_data));
            float v218_data = s0[28];
            float v220_data = r1[4];
            r1[4] = (v220_data + (v197_data * v218_data));
            float v223_data = s0[34];
            float v225_data = r1[5];
            r1[5] = (v225_data + (v197_data * v223_data));
          }
          if (v11_lead < 12) {
            float v231_data = r0[5];
            float v232_data = s0[5];
            float v234_data = r1[0];
            r1[0] = (v234_data + (v231_data * v232_data));
            float v237_data = s0[11];
            float v239_data = r1[1];
            r1[1] = (v239_data + (v231_data * v237_data));
            float v242_data = s0[17];
            float v244_data = r1[2];
            r1[2] = (v244_data + (v231_data * v242_data));
            float v247_data = s0[23];
            float v249_data = r1[3];
            r1[3] = (v249_data + (v231_data * v247_data));
            float v252_data = s0[29];
            float v254_data = r1[4];
            r1[4] = (v254_data + (v231_data * v252_data));
            float v257_data = s0[35];
            float v259_data = r1[5];
            r1[5] = (v259_data + (v231_data * v257_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v266_i1 = 0; v266_i1 < 6; ++v266_i1) {
              int32_t v267_a = 0 + v266_i1;
              float v269_data = r1[v266_i1];
              s1[(v11_lead + (v266_i1 * 12))] = v269_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v11_lead < 12) {
            float v283_data = r2[0];
            float v284_data = s1[0];
            float v286_data = ir3[0];
            ir3[0] = (v286_data + (v283_data * v284_data));
            float v289_data = s1[12];
            float v291_data = ir3[1];
            ir3[1] = (v291_data + (v283_data * v289_data));
            float v294_data = s1[24];
            float v296_data = ir3[2];
            ir3[2] = (v296_data + (v283_data * v294_data));
            float v299_data = s1[36];
            float v301_data = ir3[3];
            ir3[3] = (v301_data + (v283_data * v299_data));
            float v304_data = s1[48];
            float v306_data = ir3[4];
            ir3[4] = (v306_data + (v283_data * v304_data));
            float v309_data = s1[60];
            float v311_data = ir3[5];
            ir3[5] = (v311_data + (v283_data * v309_data));
          }
          if (v11_lead < 12) {
            float v317_data = r2[1];
            float v318_data = s1[1];
            float v320_data = ir3[0];
            ir3[0] = (v320_data + (v317_data * v318_data));
            float v323_data = s1[13];
            float v325_data = ir3[1];
            ir3[1] = (v325_data + (v317_data * v323_data));
            float v328_data = s1[25];
            float v330_data = ir3[2];
            ir3[2] = (v330_data + (v317_data * v328_data));
            float v333_data = s1[37];
            float v335_data = ir3[3];
            ir3[3] = (v335_data + (v317_data * v333_data));
            float v338_data = s1[49];
            float v340_data = ir3[4];
            ir3[4] = (v340_data + (v317_data * v338_data));
            float v343_data = s1[61];
            float v345_data = ir3[5];
            ir3[5] = (v345_data + (v317_data * v343_data));
          }
          if (v11_lead < 12) {
            float v351_data = r2[2];
            float v352_data = s1[2];
            float v354_data = ir3[0];
            ir3[0] = (v354_data + (v351_data * v352_data));
            float v357_data = s1[14];
            float v359_data = ir3[1];
            ir3[1] = (v359_data + (v351_data * v357_data));
            float v362_data = s1[26];
            float v364_data = ir3[2];
            ir3[2] = (v364_data + (v351_data * v362_data));
            float v367_data = s1[38];
            float v369_data = ir3[3];
            ir3[3] = (v369_data + (v351_data * v367_data));
            float v372_data = s1[50];
            float v374_data = ir3[4];
            ir3[4] = (v374_data + (v351_data * v372_data));
            float v377_data = s1[62];
            float v379_data = ir3[5];
            ir3[5] = (v379_data + (v351_data * v377_data));
          }
          if (v11_lead < 12) {
            float v385_data = r2[3];
            float v386_data = s1[3];
            float v388_data = ir3[0];
            ir3[0] = (v388_data + (v385_data * v386_data));
            float v391_data = s1[15];
            float v393_data = ir3[1];
            ir3[1] = (v393_data + (v385_data * v391_data));
            float v396_data = s1[27];
            float v398_data = ir3[2];
            ir3[2] = (v398_data + (v385_data * v396_data));
            float v401_data = s1[39];
            float v403_data = ir3[3];
            ir3[3] = (v403_data + (v385_data * v401_data));
            float v406_data = s1[51];
            float v408_data = ir3[4];
            ir3[4] = (v408_data + (v385_data * v406_data));
            float v411_data = s1[63];
            float v413_data = ir3[5];
            ir3[5] = (v413_data + (v385_data * v411_data));
          }
          if (v11_lead < 12) {
            float v419_data = r2[4];
            float v420_data = s1[4];
            float v422_data = ir3[0];
            ir3[0] = (v422_data + (v419_data * v420_data));
            float v425_data = s1[16];
            float v427_data = ir3[1];
            ir3[1] = (v427_data + (v419_data * v425_data));
            float v430_data = s1[28];
            float v432_data = ir3[2];
            ir3[2] = (v432_data + (v419_data * v430_data));
            float v435_data = s1[40];
            float v437_data = ir3[3];
            ir3[3] = (v437_data + (v419_data * v435_data));
            float v440_data = s1[52];
            float v442_data = ir3[4];
            ir3[4] = (v442_data + (v419_data * v440_data));
            float v445_data = s1[64];
            float v447_data = ir3[5];
            ir3[5] = (v447_data + (v419_data * v445_data));
          }
          if (v11_lead < 12) {
            float v453_data = r2[5];
            float v454_data = s1[5];
            float v456_data = ir3[0];
            ir3[0] = (v456_data + (v453_data * v454_data));
            float v459_data = s1[17];
            float v461_data = ir3[1];
            ir3[1] = (v461_data + (v453_data * v459_data));
            float v464_data = s1[29];
            float v466_data = ir3[2];
            ir3[2] = (v466_data + (v453_data * v464_data));
            float v469_data = s1[41];
            float v471_data = ir3[3];
            ir3[3] = (v471_data + (v453_data * v469_data));
            float v474_data = s1[53];
            float v476_data = ir3[4];
            ir3[4] = (v476_data + (v453_data * v474_data));
            float v479_data = s1[65];
            float v481_data = ir3[5];
            ir3[5] = (v481_data + (v453_data * v479_data));
          }
          if (v11_lead < 12) {
            float v487_data = r2[6];
            float v488_data = s1[6];
            float v490_data = ir3[0];
            ir3[0] = (v490_data + (v487_data * v488_data));
            float v493_data = s1[18];
            float v495_data = ir3[1];
            ir3[1] = (v495_data + (v487_data * v493_data));
            float v498_data = s1[30];
            float v500_data = ir3[2];
            ir3[2] = (v500_data + (v487_data * v498_data));
            float v503_data = s1[42];
            float v505_data = ir3[3];
            ir3[3] = (v505_data + (v487_data * v503_data));
            float v508_data = s1[54];
            float v510_data = ir3[4];
            ir3[4] = (v510_data + (v487_data * v508_data));
            float v513_data = s1[66];
            float v515_data = ir3[5];
            ir3[5] = (v515_data + (v487_data * v513_data));
          }
          if (v11_lead < 12) {
            float v521_data = r2[7];
            float v522_data = s1[7];
            float v524_data = ir3[0];
            ir3[0] = (v524_data + (v521_data * v522_data));
            float v527_data = s1[19];
            float v529_data = ir3[1];
            ir3[1] = (v529_data + (v521_data * v527_data));
            float v532_data = s1[31];
            float v534_data = ir3[2];
            ir3[2] = (v534_data + (v521_data * v532_data));
            float v537_data = s1[43];
            float v539_data = ir3[3];
            ir3[3] = (v539_data + (v521_data * v537_data));
            float v542_data = s1[55];
            float v544_data = ir3[4];
            ir3[4] = (v544_data + (v521_data * v542_data));
            float v547_data = s1[67];
            float v549_data = ir3[5];
            ir3[5] = (v549_data + (v521_data * v547_data));
          }
          if (v11_lead < 12) {
            float v555_data = r2[8];
            float v556_data = s1[8];
            float v558_data = ir3[0];
            ir3[0] = (v558_data + (v555_data * v556_data));
            float v561_data = s1[20];
            float v563_data = ir3[1];
            ir3[1] = (v563_data + (v555_data * v561_data));
            float v566_data = s1[32];
            float v568_data = ir3[2];
            ir3[2] = (v568_data + (v555_data * v566_data));
            float v571_data = s1[44];
            float v573_data = ir3[3];
            ir3[3] = (v573_data + (v555_data * v571_data));
            float v576_data = s1[56];
            float v578_data = ir3[4];
            ir3[4] = (v578_data + (v555_data * v576_data));
            float v581_data = s1[68];
            float v583_data = ir3[5];
            ir3[5] = (v583_data + (v555_data * v581_data));
          }
          if (v11_lead < 12) {
            float v589_data = r2[9];
            float v590_data = s1[9];
            float v592_data = ir3[0];
            ir3[0] = (v592_data + (v589_data * v590_data));
            float v595_data = s1[21];
            float v597_data = ir3[1];
            ir3[1] = (v597_data + (v589_data * v595_data));
            float v600_data = s1[33];
            float v602_data = ir3[2];
            ir3[2] = (v602_data + (v589_data * v600_data));
            float v605_data = s1[45];
            float v607_data = ir3[3];
            ir3[3] = (v607_data + (v589_data * v605_data));
            float v610_data = s1[57];
            float v612_data = ir3[4];
            ir3[4] = (v612_data + (v589_data * v610_data));
            float v615_data = s1[69];
            float v617_data = ir3[5];
            ir3[5] = (v617_data + (v589_data * v615_data));
          }
          if (v11_lead < 12) {
            float v623_data = r2[10];
            float v624_data = s1[10];
            float v626_data = ir3[0];
            ir3[0] = (v626_data + (v623_data * v624_data));
            float v629_data = s1[22];
            float v631_data = ir3[1];
            ir3[1] = (v631_data + (v623_data * v629_data));
            float v634_data = s1[34];
            float v636_data = ir3[2];
            ir3[2] = (v636_data + (v623_data * v634_data));
            float v639_data = s1[46];
            float v641_data = ir3[3];
            ir3[3] = (v641_data + (v623_data * v639_data));
            float v644_data = s1[58];
            float v646_data = ir3[4];
            ir3[4] = (v646_data + (v623_data * v644_data));
            float v649_data = s1[70];
            float v651_data = ir3[5];
            ir3[5] = (v651_data + (v623_data * v649_data));
          }
          if (v11_lead < 12) {
            float v657_data = r2[11];
            float v658_data = s1[11];
            float v660_data = ir3[0];
            ir3[0] = (v660_data + (v657_data * v658_data));
            float v663_data = s1[23];
            float v665_data = ir3[1];
            ir3[1] = (v665_data + (v657_data * v663_data));
            float v668_data = s1[35];
            float v670_data = ir3[2];
            ir3[2] = (v670_data + (v657_data * v668_data));
            float v673_data = s1[47];
            float v675_data = ir3[3];
            ir3[3] = (v675_data + (v657_data * v673_data));
            float v678_data = s1[59];
            float v680_data = ir3[4];
            ir3[4] = (v680_data + (v657_data * v678_data));
            float v683_data = s1[71];
            float v685_data = ir3[5];
            ir3[5] = (v685_data + (v657_data * v683_data));
          }
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v691_n1 = 0; v691_n1 < 6; ++v691_n1) {
              int32_t v692_a = 0 + v691_n1;
              float v694_data = ir3[v691_n1];
              r3[v691_n1] = v694_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v700_i1 = 0; v700_i1 < 6; ++v700_i1) {
              int32_t v701_a = 0 + v700_i1;
              float v703_data = r3[v700_i1];
              glb_m2[(v11_lead + (v700_i1 * 12))] = v703_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

