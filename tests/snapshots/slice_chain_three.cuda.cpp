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
          {
            // r3 = +(r2 * s1) + None
            // [(0, 12), (0, 6)] [(0, 12)]
            float ir3[6]{};
            if (v3_lead < 12) {
              float v269_data = r2[0];
              float v270_data = s1[0];
              float v272_data = ir3[0];
              ir3[0] = (v272_data + (v269_data * v270_data));
              float v275_data = s1[12];
              float v277_data = ir3[1];
              ir3[1] = (v277_data + (v269_data * v275_data));
              float v280_data = s1[24];
              float v282_data = ir3[2];
              ir3[2] = (v282_data + (v269_data * v280_data));
              float v285_data = s1[36];
              float v287_data = ir3[3];
              ir3[3] = (v287_data + (v269_data * v285_data));
              float v290_data = s1[48];
              float v292_data = ir3[4];
              ir3[4] = (v292_data + (v269_data * v290_data));
              float v295_data = s1[60];
              float v297_data = ir3[5];
              ir3[5] = (v297_data + (v269_data * v295_data));
            }
            if (v3_lead < 12) {
              float v303_data = r2[1];
              float v304_data = s1[1];
              float v306_data = ir3[0];
              ir3[0] = (v306_data + (v303_data * v304_data));
              float v309_data = s1[13];
              float v311_data = ir3[1];
              ir3[1] = (v311_data + (v303_data * v309_data));
              float v314_data = s1[25];
              float v316_data = ir3[2];
              ir3[2] = (v316_data + (v303_data * v314_data));
              float v319_data = s1[37];
              float v321_data = ir3[3];
              ir3[3] = (v321_data + (v303_data * v319_data));
              float v324_data = s1[49];
              float v326_data = ir3[4];
              ir3[4] = (v326_data + (v303_data * v324_data));
              float v329_data = s1[61];
              float v331_data = ir3[5];
              ir3[5] = (v331_data + (v303_data * v329_data));
            }
            if (v3_lead < 12) {
              float v337_data = r2[2];
              float v338_data = s1[2];
              float v340_data = ir3[0];
              ir3[0] = (v340_data + (v337_data * v338_data));
              float v343_data = s1[14];
              float v345_data = ir3[1];
              ir3[1] = (v345_data + (v337_data * v343_data));
              float v348_data = s1[26];
              float v350_data = ir3[2];
              ir3[2] = (v350_data + (v337_data * v348_data));
              float v353_data = s1[38];
              float v355_data = ir3[3];
              ir3[3] = (v355_data + (v337_data * v353_data));
              float v358_data = s1[50];
              float v360_data = ir3[4];
              ir3[4] = (v360_data + (v337_data * v358_data));
              float v363_data = s1[62];
              float v365_data = ir3[5];
              ir3[5] = (v365_data + (v337_data * v363_data));
            }
            if (v3_lead < 12) {
              float v371_data = r2[3];
              float v372_data = s1[3];
              float v374_data = ir3[0];
              ir3[0] = (v374_data + (v371_data * v372_data));
              float v377_data = s1[15];
              float v379_data = ir3[1];
              ir3[1] = (v379_data + (v371_data * v377_data));
              float v382_data = s1[27];
              float v384_data = ir3[2];
              ir3[2] = (v384_data + (v371_data * v382_data));
              float v387_data = s1[39];
              float v389_data = ir3[3];
              ir3[3] = (v389_data + (v371_data * v387_data));
              float v392_data = s1[51];
              float v394_data = ir3[4];
              ir3[4] = (v394_data + (v371_data * v392_data));
              float v397_data = s1[63];
              float v399_data = ir3[5];
              ir3[5] = (v399_data + (v371_data * v397_data));
            }
            if (v3_lead < 12) {
              float v405_data = r2[4];
              float v406_data = s1[4];
              float v408_data = ir3[0];
              ir3[0] = (v408_data + (v405_data * v406_data));
              float v411_data = s1[16];
              float v413_data = ir3[1];
              ir3[1] = (v413_data + (v405_data * v411_data));
              float v416_data = s1[28];
              float v418_data = ir3[2];
              ir3[2] = (v418_data + (v405_data * v416_data));
              float v421_data = s1[40];
              float v423_data = ir3[3];
              ir3[3] = (v423_data + (v405_data * v421_data));
              float v426_data = s1[52];
              float v428_data = ir3[4];
              ir3[4] = (v428_data + (v405_data * v426_data));
              float v431_data = s1[64];
              float v433_data = ir3[5];
              ir3[5] = (v433_data + (v405_data * v431_data));
            }
            if (v3_lead < 12) {
              float v439_data = r2[5];
              float v440_data = s1[5];
              float v442_data = ir3[0];
              ir3[0] = (v442_data + (v439_data * v440_data));
              float v445_data = s1[17];
              float v447_data = ir3[1];
              ir3[1] = (v447_data + (v439_data * v445_data));
              float v450_data = s1[29];
              float v452_data = ir3[2];
              ir3[2] = (v452_data + (v439_data * v450_data));
              float v455_data = s1[41];
              float v457_data = ir3[3];
              ir3[3] = (v457_data + (v439_data * v455_data));
              float v460_data = s1[53];
              float v462_data = ir3[4];
              ir3[4] = (v462_data + (v439_data * v460_data));
              float v465_data = s1[65];
              float v467_data = ir3[5];
              ir3[5] = (v467_data + (v439_data * v465_data));
            }
            if (v3_lead < 12) {
              float v473_data = r2[6];
              float v474_data = s1[6];
              float v476_data = ir3[0];
              ir3[0] = (v476_data + (v473_data * v474_data));
              float v479_data = s1[18];
              float v481_data = ir3[1];
              ir3[1] = (v481_data + (v473_data * v479_data));
              float v484_data = s1[30];
              float v486_data = ir3[2];
              ir3[2] = (v486_data + (v473_data * v484_data));
              float v489_data = s1[42];
              float v491_data = ir3[3];
              ir3[3] = (v491_data + (v473_data * v489_data));
              float v494_data = s1[54];
              float v496_data = ir3[4];
              ir3[4] = (v496_data + (v473_data * v494_data));
              float v499_data = s1[66];
              float v501_data = ir3[5];
              ir3[5] = (v501_data + (v473_data * v499_data));
            }
            if (v3_lead < 12) {
              float v507_data = r2[7];
              float v508_data = s1[7];
              float v510_data = ir3[0];
              ir3[0] = (v510_data + (v507_data * v508_data));
              float v513_data = s1[19];
              float v515_data = ir3[1];
              ir3[1] = (v515_data + (v507_data * v513_data));
              float v518_data = s1[31];
              float v520_data = ir3[2];
              ir3[2] = (v520_data + (v507_data * v518_data));
              float v523_data = s1[43];
              float v525_data = ir3[3];
              ir3[3] = (v525_data + (v507_data * v523_data));
              float v528_data = s1[55];
              float v530_data = ir3[4];
              ir3[4] = (v530_data + (v507_data * v528_data));
              float v533_data = s1[67];
              float v535_data = ir3[5];
              ir3[5] = (v535_data + (v507_data * v533_data));
            }
            if (v3_lead < 12) {
              float v541_data = r2[8];
              float v542_data = s1[8];
              float v544_data = ir3[0];
              ir3[0] = (v544_data + (v541_data * v542_data));
              float v547_data = s1[20];
              float v549_data = ir3[1];
              ir3[1] = (v549_data + (v541_data * v547_data));
              float v552_data = s1[32];
              float v554_data = ir3[2];
              ir3[2] = (v554_data + (v541_data * v552_data));
              float v557_data = s1[44];
              float v559_data = ir3[3];
              ir3[3] = (v559_data + (v541_data * v557_data));
              float v562_data = s1[56];
              float v564_data = ir3[4];
              ir3[4] = (v564_data + (v541_data * v562_data));
              float v567_data = s1[68];
              float v569_data = ir3[5];
              ir3[5] = (v569_data + (v541_data * v567_data));
            }
            if (v3_lead < 12) {
              float v575_data = r2[9];
              float v576_data = s1[9];
              float v578_data = ir3[0];
              ir3[0] = (v578_data + (v575_data * v576_data));
              float v581_data = s1[21];
              float v583_data = ir3[1];
              ir3[1] = (v583_data + (v575_data * v581_data));
              float v586_data = s1[33];
              float v588_data = ir3[2];
              ir3[2] = (v588_data + (v575_data * v586_data));
              float v591_data = s1[45];
              float v593_data = ir3[3];
              ir3[3] = (v593_data + (v575_data * v591_data));
              float v596_data = s1[57];
              float v598_data = ir3[4];
              ir3[4] = (v598_data + (v575_data * v596_data));
              float v601_data = s1[69];
              float v603_data = ir3[5];
              ir3[5] = (v603_data + (v575_data * v601_data));
            }
            if (v3_lead < 12) {
              float v609_data = r2[10];
              float v610_data = s1[10];
              float v612_data = ir3[0];
              ir3[0] = (v612_data + (v609_data * v610_data));
              float v615_data = s1[22];
              float v617_data = ir3[1];
              ir3[1] = (v617_data + (v609_data * v615_data));
              float v620_data = s1[34];
              float v622_data = ir3[2];
              ir3[2] = (v622_data + (v609_data * v620_data));
              float v625_data = s1[46];
              float v627_data = ir3[3];
              ir3[3] = (v627_data + (v609_data * v625_data));
              float v630_data = s1[58];
              float v632_data = ir3[4];
              ir3[4] = (v632_data + (v609_data * v630_data));
              float v635_data = s1[70];
              float v637_data = ir3[5];
              ir3[5] = (v637_data + (v609_data * v635_data));
            }
            if (v3_lead < 12) {
              float v643_data = r2[11];
              float v644_data = s1[11];
              float v646_data = ir3[0];
              ir3[0] = (v646_data + (v643_data * v644_data));
              float v649_data = s1[23];
              float v651_data = ir3[1];
              ir3[1] = (v651_data + (v643_data * v649_data));
              float v654_data = s1[35];
              float v656_data = ir3[2];
              ir3[2] = (v656_data + (v643_data * v654_data));
              float v659_data = s1[47];
              float v661_data = ir3[3];
              ir3[3] = (v661_data + (v643_data * v659_data));
              float v664_data = s1[59];
              float v666_data = ir3[4];
              ir3[4] = (v666_data + (v643_data * v664_data));
              float v669_data = s1[71];
              float v671_data = ir3[5];
              ir3[5] = (v671_data + (v643_data * v669_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v677_n1 = 0; v677_n1 < 6; ++v677_n1) {
                int32_t v678_a = 0 + v677_n1;
                float v680_data = ir3[v677_n1];
                int32_t v681_a = 0 + v677_n1;
                r3[v677_n1] = v680_data;
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v687_i1 = 0; v687_i1 < 6; ++v687_i1) {
              int32_t v688_a = 0 + v687_i1;
              float v690_data = r3[v687_i1];
              int32_t v697_a = v3_lead + (v687_i1 * 12);
              glb_m2[v697_a] = v690_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

