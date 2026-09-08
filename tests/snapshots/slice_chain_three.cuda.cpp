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
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 6; ++v18_i1) {
              float v26_data = __ldcg(&glb_m0[(v16_lead + (v18_i1 * 12))]);
              r0[v18_i1] = v26_data;
            }
          }
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
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v44_data = __ldcg(&glb_m3[(v16_lead + (v36_i1 * 12))]);
              r2[v36_i1] = v44_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v16_lead < 12) {
            float v51_data = r0[0];
            float v52_data = s0[0];
            float v54_data = r1[0];
            r1[0] = (v54_data + (v51_data * v52_data));
            float v57_data = s0[6];
            float v59_data = r1[1];
            r1[1] = (v59_data + (v51_data * v57_data));
            float v62_data = s0[12];
            float v64_data = r1[2];
            r1[2] = (v64_data + (v51_data * v62_data));
            float v67_data = s0[18];
            float v69_data = r1[3];
            r1[3] = (v69_data + (v51_data * v67_data));
            float v72_data = s0[24];
            float v74_data = r1[4];
            r1[4] = (v74_data + (v51_data * v72_data));
            float v77_data = s0[30];
            float v79_data = r1[5];
            r1[5] = (v79_data + (v51_data * v77_data));
          }
          if (v16_lead < 12) {
            float v85_data = r0[1];
            float v86_data = s0[1];
            float v88_data = r1[0];
            r1[0] = (v88_data + (v85_data * v86_data));
            float v91_data = s0[7];
            float v93_data = r1[1];
            r1[1] = (v93_data + (v85_data * v91_data));
            float v96_data = s0[13];
            float v98_data = r1[2];
            r1[2] = (v98_data + (v85_data * v96_data));
            float v101_data = s0[19];
            float v103_data = r1[3];
            r1[3] = (v103_data + (v85_data * v101_data));
            float v106_data = s0[25];
            float v108_data = r1[4];
            r1[4] = (v108_data + (v85_data * v106_data));
            float v111_data = s0[31];
            float v113_data = r1[5];
            r1[5] = (v113_data + (v85_data * v111_data));
          }
          if (v16_lead < 12) {
            float v119_data = r0[2];
            float v120_data = s0[2];
            float v122_data = r1[0];
            r1[0] = (v122_data + (v119_data * v120_data));
            float v125_data = s0[8];
            float v127_data = r1[1];
            r1[1] = (v127_data + (v119_data * v125_data));
            float v130_data = s0[14];
            float v132_data = r1[2];
            r1[2] = (v132_data + (v119_data * v130_data));
            float v135_data = s0[20];
            float v137_data = r1[3];
            r1[3] = (v137_data + (v119_data * v135_data));
            float v140_data = s0[26];
            float v142_data = r1[4];
            r1[4] = (v142_data + (v119_data * v140_data));
            float v145_data = s0[32];
            float v147_data = r1[5];
            r1[5] = (v147_data + (v119_data * v145_data));
          }
          if (v16_lead < 12) {
            float v153_data = r0[3];
            float v154_data = s0[3];
            float v156_data = r1[0];
            r1[0] = (v156_data + (v153_data * v154_data));
            float v159_data = s0[9];
            float v161_data = r1[1];
            r1[1] = (v161_data + (v153_data * v159_data));
            float v164_data = s0[15];
            float v166_data = r1[2];
            r1[2] = (v166_data + (v153_data * v164_data));
            float v169_data = s0[21];
            float v171_data = r1[3];
            r1[3] = (v171_data + (v153_data * v169_data));
            float v174_data = s0[27];
            float v176_data = r1[4];
            r1[4] = (v176_data + (v153_data * v174_data));
            float v179_data = s0[33];
            float v181_data = r1[5];
            r1[5] = (v181_data + (v153_data * v179_data));
          }
          if (v16_lead < 12) {
            float v187_data = r0[4];
            float v188_data = s0[4];
            float v190_data = r1[0];
            r1[0] = (v190_data + (v187_data * v188_data));
            float v193_data = s0[10];
            float v195_data = r1[1];
            r1[1] = (v195_data + (v187_data * v193_data));
            float v198_data = s0[16];
            float v200_data = r1[2];
            r1[2] = (v200_data + (v187_data * v198_data));
            float v203_data = s0[22];
            float v205_data = r1[3];
            r1[3] = (v205_data + (v187_data * v203_data));
            float v208_data = s0[28];
            float v210_data = r1[4];
            r1[4] = (v210_data + (v187_data * v208_data));
            float v213_data = s0[34];
            float v215_data = r1[5];
            r1[5] = (v215_data + (v187_data * v213_data));
          }
          if (v16_lead < 12) {
            float v221_data = r0[5];
            float v222_data = s0[5];
            float v224_data = r1[0];
            r1[0] = (v224_data + (v221_data * v222_data));
            float v227_data = s0[11];
            float v229_data = r1[1];
            r1[1] = (v229_data + (v221_data * v227_data));
            float v232_data = s0[17];
            float v234_data = r1[2];
            r1[2] = (v234_data + (v221_data * v232_data));
            float v237_data = s0[23];
            float v239_data = r1[3];
            r1[3] = (v239_data + (v221_data * v237_data));
            float v242_data = s0[29];
            float v244_data = r1[4];
            r1[4] = (v244_data + (v221_data * v242_data));
            float v247_data = s0[35];
            float v249_data = r1[5];
            r1[5] = (v249_data + (v221_data * v247_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 6; ++v255_i1) {
              float v257_data = r1[v255_i1];
              int32_t v264_a = v16_lead + (v255_i1 * 12);
              s1[(v264_a ^ ((v264_a >> 3) & 7))] = v257_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v16_lead < 12) {
            float v274_data = r2[0];
            float v275_data = s1[0];
            float v277_data = ir3[0];
            ir3[0] = (v277_data + (v274_data * v275_data));
            float v280_data = s1[13];
            float v282_data = ir3[1];
            ir3[1] = (v282_data + (v274_data * v280_data));
            float v285_data = s1[27];
            float v287_data = ir3[2];
            ir3[2] = (v287_data + (v274_data * v285_data));
            float v290_data = s1[32];
            float v292_data = ir3[3];
            ir3[3] = (v292_data + (v274_data * v290_data));
            float v295_data = s1[54];
            float v297_data = ir3[4];
            ir3[4] = (v297_data + (v274_data * v295_data));
            float v300_data = s1[59];
            float v302_data = ir3[5];
            ir3[5] = (v302_data + (v274_data * v300_data));
          }
          if (v16_lead < 12) {
            float v308_data = r2[1];
            float v309_data = s1[1];
            float v311_data = ir3[0];
            ir3[0] = (v311_data + (v308_data * v309_data));
            float v314_data = s1[12];
            float v316_data = ir3[1];
            ir3[1] = (v316_data + (v308_data * v314_data));
            float v319_data = s1[26];
            float v321_data = ir3[2];
            ir3[2] = (v321_data + (v308_data * v319_data));
            float v324_data = s1[33];
            float v326_data = ir3[3];
            ir3[3] = (v326_data + (v308_data * v324_data));
            float v329_data = s1[55];
            float v331_data = ir3[4];
            ir3[4] = (v331_data + (v308_data * v329_data));
            float v334_data = s1[58];
            float v336_data = ir3[5];
            ir3[5] = (v336_data + (v308_data * v334_data));
          }
          if (v16_lead < 12) {
            float v342_data = r2[2];
            float v343_data = s1[2];
            float v345_data = ir3[0];
            ir3[0] = (v345_data + (v342_data * v343_data));
            float v348_data = s1[15];
            float v350_data = ir3[1];
            ir3[1] = (v350_data + (v342_data * v348_data));
            float v353_data = s1[25];
            float v355_data = ir3[2];
            ir3[2] = (v355_data + (v342_data * v353_data));
            float v358_data = s1[34];
            float v360_data = ir3[3];
            ir3[3] = (v360_data + (v342_data * v358_data));
            float v363_data = s1[52];
            float v365_data = ir3[4];
            ir3[4] = (v365_data + (v342_data * v363_data));
            float v368_data = s1[57];
            float v370_data = ir3[5];
            ir3[5] = (v370_data + (v342_data * v368_data));
          }
          if (v16_lead < 12) {
            float v376_data = r2[3];
            float v377_data = s1[3];
            float v379_data = ir3[0];
            ir3[0] = (v379_data + (v376_data * v377_data));
            float v382_data = s1[14];
            float v384_data = ir3[1];
            ir3[1] = (v384_data + (v376_data * v382_data));
            float v387_data = s1[24];
            float v389_data = ir3[2];
            ir3[2] = (v389_data + (v376_data * v387_data));
            float v392_data = s1[35];
            float v394_data = ir3[3];
            ir3[3] = (v394_data + (v376_data * v392_data));
            float v397_data = s1[53];
            float v399_data = ir3[4];
            ir3[4] = (v399_data + (v376_data * v397_data));
            float v402_data = s1[56];
            float v404_data = ir3[5];
            ir3[5] = (v404_data + (v376_data * v402_data));
          }
          if (v16_lead < 12) {
            float v410_data = r2[4];
            float v411_data = s1[4];
            float v413_data = ir3[0];
            ir3[0] = (v413_data + (v410_data * v411_data));
            float v416_data = s1[18];
            float v418_data = ir3[1];
            ir3[1] = (v418_data + (v410_data * v416_data));
            float v421_data = s1[31];
            float v423_data = ir3[2];
            ir3[2] = (v423_data + (v410_data * v421_data));
            float v426_data = s1[45];
            float v428_data = ir3[3];
            ir3[3] = (v428_data + (v410_data * v426_data));
            float v431_data = s1[50];
            float v433_data = ir3[4];
            ir3[4] = (v433_data + (v410_data * v431_data));
            float v436_data = s1[64];
            float v438_data = ir3[5];
            ir3[5] = (v438_data + (v410_data * v436_data));
          }
          if (v16_lead < 12) {
            float v444_data = r2[5];
            float v445_data = s1[5];
            float v447_data = ir3[0];
            ir3[0] = (v447_data + (v444_data * v445_data));
            float v450_data = s1[19];
            float v452_data = ir3[1];
            ir3[1] = (v452_data + (v444_data * v450_data));
            float v455_data = s1[30];
            float v457_data = ir3[2];
            ir3[2] = (v457_data + (v444_data * v455_data));
            float v460_data = s1[44];
            float v462_data = ir3[3];
            ir3[3] = (v462_data + (v444_data * v460_data));
            float v465_data = s1[51];
            float v467_data = ir3[4];
            ir3[4] = (v467_data + (v444_data * v465_data));
            float v470_data = s1[65];
            float v472_data = ir3[5];
            ir3[5] = (v472_data + (v444_data * v470_data));
          }
          if (v16_lead < 12) {
            float v478_data = r2[6];
            float v479_data = s1[6];
            float v481_data = ir3[0];
            ir3[0] = (v481_data + (v478_data * v479_data));
            float v484_data = s1[16];
            float v486_data = ir3[1];
            ir3[1] = (v486_data + (v478_data * v484_data));
            float v489_data = s1[29];
            float v491_data = ir3[2];
            ir3[2] = (v491_data + (v478_data * v489_data));
            float v494_data = s1[47];
            float v496_data = ir3[3];
            ir3[3] = (v496_data + (v478_data * v494_data));
            float v499_data = s1[48];
            float v501_data = ir3[4];
            ir3[4] = (v501_data + (v478_data * v499_data));
            float v504_data = s1[66];
            float v506_data = ir3[5];
            ir3[5] = (v506_data + (v478_data * v504_data));
          }
          if (v16_lead < 12) {
            float v512_data = r2[7];
            float v513_data = s1[7];
            float v515_data = ir3[0];
            ir3[0] = (v515_data + (v512_data * v513_data));
            float v518_data = s1[17];
            float v520_data = ir3[1];
            ir3[1] = (v520_data + (v512_data * v518_data));
            float v523_data = s1[28];
            float v525_data = ir3[2];
            ir3[2] = (v525_data + (v512_data * v523_data));
            float v528_data = s1[46];
            float v530_data = ir3[3];
            ir3[3] = (v530_data + (v512_data * v528_data));
            float v533_data = s1[49];
            float v535_data = ir3[4];
            ir3[4] = (v535_data + (v512_data * v533_data));
            float v538_data = s1[67];
            float v540_data = ir3[5];
            ir3[5] = (v540_data + (v512_data * v538_data));
          }
          if (v16_lead < 12) {
            float v546_data = r2[8];
            float v547_data = s1[9];
            float v549_data = ir3[0];
            ir3[0] = (v549_data + (v546_data * v547_data));
            float v552_data = s1[22];
            float v554_data = ir3[1];
            ir3[1] = (v554_data + (v546_data * v552_data));
            float v557_data = s1[36];
            float v559_data = ir3[2];
            ir3[2] = (v559_data + (v546_data * v557_data));
            float v562_data = s1[41];
            float v564_data = ir3[3];
            ir3[3] = (v564_data + (v546_data * v562_data));
            float v567_data = s1[63];
            float v569_data = ir3[4];
            ir3[4] = (v569_data + (v546_data * v567_data));
            float v572_data = s1[68];
            float v574_data = ir3[5];
            ir3[5] = (v574_data + (v546_data * v572_data));
          }
          if (v16_lead < 12) {
            float v580_data = r2[9];
            float v581_data = s1[8];
            float v583_data = ir3[0];
            ir3[0] = (v583_data + (v580_data * v581_data));
            float v586_data = s1[23];
            float v588_data = ir3[1];
            ir3[1] = (v588_data + (v580_data * v586_data));
            float v591_data = s1[37];
            float v593_data = ir3[2];
            ir3[2] = (v593_data + (v580_data * v591_data));
            float v596_data = s1[40];
            float v598_data = ir3[3];
            ir3[3] = (v598_data + (v580_data * v596_data));
            float v601_data = s1[62];
            float v603_data = ir3[4];
            ir3[4] = (v603_data + (v580_data * v601_data));
            float v606_data = s1[69];
            float v608_data = ir3[5];
            ir3[5] = (v608_data + (v580_data * v606_data));
          }
          if (v16_lead < 12) {
            float v614_data = r2[10];
            float v615_data = s1[11];
            float v617_data = ir3[0];
            ir3[0] = (v617_data + (v614_data * v615_data));
            float v620_data = s1[20];
            float v622_data = ir3[1];
            ir3[1] = (v622_data + (v614_data * v620_data));
            float v625_data = s1[38];
            float v627_data = ir3[2];
            ir3[2] = (v627_data + (v614_data * v625_data));
            float v630_data = s1[43];
            float v632_data = ir3[3];
            ir3[3] = (v632_data + (v614_data * v630_data));
            float v635_data = s1[61];
            float v637_data = ir3[4];
            ir3[4] = (v637_data + (v614_data * v635_data));
            float v640_data = s1[70];
            float v642_data = ir3[5];
            ir3[5] = (v642_data + (v614_data * v640_data));
          }
          if (v16_lead < 12) {
            float v648_data = r2[11];
            float v649_data = s1[10];
            float v651_data = ir3[0];
            ir3[0] = (v651_data + (v648_data * v649_data));
            float v654_data = s1[21];
            float v656_data = ir3[1];
            ir3[1] = (v656_data + (v648_data * v654_data));
            float v659_data = s1[39];
            float v661_data = ir3[2];
            ir3[2] = (v661_data + (v648_data * v659_data));
            float v664_data = s1[42];
            float v666_data = ir3[3];
            ir3[3] = (v666_data + (v648_data * v664_data));
            float v669_data = s1[60];
            float v671_data = ir3[4];
            ir3[4] = (v671_data + (v648_data * v669_data));
            float v674_data = s1[71];
            float v676_data = ir3[5];
            ir3[5] = (v676_data + (v648_data * v674_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v682_n1 = 0; v682_n1 < 6; ++v682_n1) {
              float v684_data = ir3[v682_n1];
              r3[v682_n1] = v684_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v690_i1 = 0; v690_i1 < 6; ++v690_i1) {
              float v692_data = r3[v690_i1];
              glb_m2[(v16_lead + (v690_i1 * 12))] = v692_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

