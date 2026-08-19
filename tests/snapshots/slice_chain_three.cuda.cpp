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
              int32_t v11_a = v2_lead + (v4_i1 * 12);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m0[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 12);
              float v26_data;
              {
                v26_data = __ldcg(&glb_m3[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r2[v27_a] = v26_data;
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
          int32_t v30_lead = threadIdx.x % 16;
          if (v30_lead < 12) {
            float v32_data = r0[0];
            float v33_data = s0[0];
            float v35_data = ir1[0];
            ir1[0] = (v35_data + (v32_data * v33_data));
            float v38_data = s0[6];
            float v40_data = ir1[1];
            ir1[1] = (v40_data + (v32_data * v38_data));
            float v43_data = s0[12];
            float v45_data = ir1[2];
            ir1[2] = (v45_data + (v32_data * v43_data));
            float v48_data = s0[18];
            float v50_data = ir1[3];
            ir1[3] = (v50_data + (v32_data * v48_data));
            float v53_data = s0[24];
            float v55_data = ir1[4];
            ir1[4] = (v55_data + (v32_data * v53_data));
            float v58_data = s0[30];
            float v60_data = ir1[5];
            ir1[5] = (v60_data + (v32_data * v58_data));
          }
          if (v30_lead < 12) {
            float v66_data = r0[1];
            float v67_data = s0[1];
            float v69_data = ir1[0];
            ir1[0] = (v69_data + (v66_data * v67_data));
            float v72_data = s0[7];
            float v74_data = ir1[1];
            ir1[1] = (v74_data + (v66_data * v72_data));
            float v77_data = s0[13];
            float v79_data = ir1[2];
            ir1[2] = (v79_data + (v66_data * v77_data));
            float v82_data = s0[19];
            float v84_data = ir1[3];
            ir1[3] = (v84_data + (v66_data * v82_data));
            float v87_data = s0[25];
            float v89_data = ir1[4];
            ir1[4] = (v89_data + (v66_data * v87_data));
            float v92_data = s0[31];
            float v94_data = ir1[5];
            ir1[5] = (v94_data + (v66_data * v92_data));
          }
          if (v30_lead < 12) {
            float v100_data = r0[2];
            float v101_data = s0[2];
            float v103_data = ir1[0];
            ir1[0] = (v103_data + (v100_data * v101_data));
            float v106_data = s0[8];
            float v108_data = ir1[1];
            ir1[1] = (v108_data + (v100_data * v106_data));
            float v111_data = s0[14];
            float v113_data = ir1[2];
            ir1[2] = (v113_data + (v100_data * v111_data));
            float v116_data = s0[20];
            float v118_data = ir1[3];
            ir1[3] = (v118_data + (v100_data * v116_data));
            float v121_data = s0[26];
            float v123_data = ir1[4];
            ir1[4] = (v123_data + (v100_data * v121_data));
            float v126_data = s0[32];
            float v128_data = ir1[5];
            ir1[5] = (v128_data + (v100_data * v126_data));
          }
          if (v30_lead < 12) {
            float v134_data = r0[3];
            float v135_data = s0[3];
            float v137_data = ir1[0];
            ir1[0] = (v137_data + (v134_data * v135_data));
            float v140_data = s0[9];
            float v142_data = ir1[1];
            ir1[1] = (v142_data + (v134_data * v140_data));
            float v145_data = s0[15];
            float v147_data = ir1[2];
            ir1[2] = (v147_data + (v134_data * v145_data));
            float v150_data = s0[21];
            float v152_data = ir1[3];
            ir1[3] = (v152_data + (v134_data * v150_data));
            float v155_data = s0[27];
            float v157_data = ir1[4];
            ir1[4] = (v157_data + (v134_data * v155_data));
            float v160_data = s0[33];
            float v162_data = ir1[5];
            ir1[5] = (v162_data + (v134_data * v160_data));
          }
          if (v30_lead < 12) {
            float v168_data = r0[4];
            float v169_data = s0[4];
            float v171_data = ir1[0];
            ir1[0] = (v171_data + (v168_data * v169_data));
            float v174_data = s0[10];
            float v176_data = ir1[1];
            ir1[1] = (v176_data + (v168_data * v174_data));
            float v179_data = s0[16];
            float v181_data = ir1[2];
            ir1[2] = (v181_data + (v168_data * v179_data));
            float v184_data = s0[22];
            float v186_data = ir1[3];
            ir1[3] = (v186_data + (v168_data * v184_data));
            float v189_data = s0[28];
            float v191_data = ir1[4];
            ir1[4] = (v191_data + (v168_data * v189_data));
            float v194_data = s0[34];
            float v196_data = ir1[5];
            ir1[5] = (v196_data + (v168_data * v194_data));
          }
          if (v30_lead < 12) {
            float v202_data = r0[5];
            float v203_data = s0[5];
            float v205_data = ir1[0];
            ir1[0] = (v205_data + (v202_data * v203_data));
            float v208_data = s0[11];
            float v210_data = ir1[1];
            ir1[1] = (v210_data + (v202_data * v208_data));
            float v213_data = s0[17];
            float v215_data = ir1[2];
            ir1[2] = (v215_data + (v202_data * v213_data));
            float v218_data = s0[23];
            float v220_data = ir1[3];
            ir1[3] = (v220_data + (v202_data * v218_data));
            float v223_data = s0[29];
            float v225_data = ir1[4];
            ir1[4] = (v225_data + (v202_data * v223_data));
            float v228_data = s0[35];
            float v230_data = ir1[5];
            ir1[5] = (v230_data + (v202_data * v228_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          int32_t v234_lead = threadIdx.x % 16;
          if (v234_lead < 12) {
            #pragma unroll
            for (int32_t v236_i1 = 0; v236_i1 < 6; ++v236_i1) {
              int32_t v237_a = 0 + v236_i1;
              float v238_data = r1[v237_a];
              int32_t v245_a = v234_lead + (v236_i1 * 12);
              s1[v245_a] = v238_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + None
            // [(0, 12), (0, 6)] [(0, 12)]
            float ir3[6]{};
            int32_t v248_lead = threadIdx.x % 16;
            if (v248_lead < 12) {
              float v250_data = r2[0];
              float v251_data = s1[0];
              float v253_data = ir3[0];
              ir3[0] = (v253_data + (v250_data * v251_data));
              float v256_data = s1[12];
              float v258_data = ir3[1];
              ir3[1] = (v258_data + (v250_data * v256_data));
              float v261_data = s1[24];
              float v263_data = ir3[2];
              ir3[2] = (v263_data + (v250_data * v261_data));
              float v266_data = s1[36];
              float v268_data = ir3[3];
              ir3[3] = (v268_data + (v250_data * v266_data));
              float v271_data = s1[48];
              float v273_data = ir3[4];
              ir3[4] = (v273_data + (v250_data * v271_data));
              float v276_data = s1[60];
              float v278_data = ir3[5];
              ir3[5] = (v278_data + (v250_data * v276_data));
            }
            if (v248_lead < 12) {
              float v284_data = r2[1];
              float v285_data = s1[1];
              float v287_data = ir3[0];
              ir3[0] = (v287_data + (v284_data * v285_data));
              float v290_data = s1[13];
              float v292_data = ir3[1];
              ir3[1] = (v292_data + (v284_data * v290_data));
              float v295_data = s1[25];
              float v297_data = ir3[2];
              ir3[2] = (v297_data + (v284_data * v295_data));
              float v300_data = s1[37];
              float v302_data = ir3[3];
              ir3[3] = (v302_data + (v284_data * v300_data));
              float v305_data = s1[49];
              float v307_data = ir3[4];
              ir3[4] = (v307_data + (v284_data * v305_data));
              float v310_data = s1[61];
              float v312_data = ir3[5];
              ir3[5] = (v312_data + (v284_data * v310_data));
            }
            if (v248_lead < 12) {
              float v318_data = r2[2];
              float v319_data = s1[2];
              float v321_data = ir3[0];
              ir3[0] = (v321_data + (v318_data * v319_data));
              float v324_data = s1[14];
              float v326_data = ir3[1];
              ir3[1] = (v326_data + (v318_data * v324_data));
              float v329_data = s1[26];
              float v331_data = ir3[2];
              ir3[2] = (v331_data + (v318_data * v329_data));
              float v334_data = s1[38];
              float v336_data = ir3[3];
              ir3[3] = (v336_data + (v318_data * v334_data));
              float v339_data = s1[50];
              float v341_data = ir3[4];
              ir3[4] = (v341_data + (v318_data * v339_data));
              float v344_data = s1[62];
              float v346_data = ir3[5];
              ir3[5] = (v346_data + (v318_data * v344_data));
            }
            if (v248_lead < 12) {
              float v352_data = r2[3];
              float v353_data = s1[3];
              float v355_data = ir3[0];
              ir3[0] = (v355_data + (v352_data * v353_data));
              float v358_data = s1[15];
              float v360_data = ir3[1];
              ir3[1] = (v360_data + (v352_data * v358_data));
              float v363_data = s1[27];
              float v365_data = ir3[2];
              ir3[2] = (v365_data + (v352_data * v363_data));
              float v368_data = s1[39];
              float v370_data = ir3[3];
              ir3[3] = (v370_data + (v352_data * v368_data));
              float v373_data = s1[51];
              float v375_data = ir3[4];
              ir3[4] = (v375_data + (v352_data * v373_data));
              float v378_data = s1[63];
              float v380_data = ir3[5];
              ir3[5] = (v380_data + (v352_data * v378_data));
            }
            if (v248_lead < 12) {
              float v386_data = r2[4];
              float v387_data = s1[4];
              float v389_data = ir3[0];
              ir3[0] = (v389_data + (v386_data * v387_data));
              float v392_data = s1[16];
              float v394_data = ir3[1];
              ir3[1] = (v394_data + (v386_data * v392_data));
              float v397_data = s1[28];
              float v399_data = ir3[2];
              ir3[2] = (v399_data + (v386_data * v397_data));
              float v402_data = s1[40];
              float v404_data = ir3[3];
              ir3[3] = (v404_data + (v386_data * v402_data));
              float v407_data = s1[52];
              float v409_data = ir3[4];
              ir3[4] = (v409_data + (v386_data * v407_data));
              float v412_data = s1[64];
              float v414_data = ir3[5];
              ir3[5] = (v414_data + (v386_data * v412_data));
            }
            if (v248_lead < 12) {
              float v420_data = r2[5];
              float v421_data = s1[5];
              float v423_data = ir3[0];
              ir3[0] = (v423_data + (v420_data * v421_data));
              float v426_data = s1[17];
              float v428_data = ir3[1];
              ir3[1] = (v428_data + (v420_data * v426_data));
              float v431_data = s1[29];
              float v433_data = ir3[2];
              ir3[2] = (v433_data + (v420_data * v431_data));
              float v436_data = s1[41];
              float v438_data = ir3[3];
              ir3[3] = (v438_data + (v420_data * v436_data));
              float v441_data = s1[53];
              float v443_data = ir3[4];
              ir3[4] = (v443_data + (v420_data * v441_data));
              float v446_data = s1[65];
              float v448_data = ir3[5];
              ir3[5] = (v448_data + (v420_data * v446_data));
            }
            if (v248_lead < 12) {
              float v454_data = r2[6];
              float v455_data = s1[6];
              float v457_data = ir3[0];
              ir3[0] = (v457_data + (v454_data * v455_data));
              float v460_data = s1[18];
              float v462_data = ir3[1];
              ir3[1] = (v462_data + (v454_data * v460_data));
              float v465_data = s1[30];
              float v467_data = ir3[2];
              ir3[2] = (v467_data + (v454_data * v465_data));
              float v470_data = s1[42];
              float v472_data = ir3[3];
              ir3[3] = (v472_data + (v454_data * v470_data));
              float v475_data = s1[54];
              float v477_data = ir3[4];
              ir3[4] = (v477_data + (v454_data * v475_data));
              float v480_data = s1[66];
              float v482_data = ir3[5];
              ir3[5] = (v482_data + (v454_data * v480_data));
            }
            if (v248_lead < 12) {
              float v488_data = r2[7];
              float v489_data = s1[7];
              float v491_data = ir3[0];
              ir3[0] = (v491_data + (v488_data * v489_data));
              float v494_data = s1[19];
              float v496_data = ir3[1];
              ir3[1] = (v496_data + (v488_data * v494_data));
              float v499_data = s1[31];
              float v501_data = ir3[2];
              ir3[2] = (v501_data + (v488_data * v499_data));
              float v504_data = s1[43];
              float v506_data = ir3[3];
              ir3[3] = (v506_data + (v488_data * v504_data));
              float v509_data = s1[55];
              float v511_data = ir3[4];
              ir3[4] = (v511_data + (v488_data * v509_data));
              float v514_data = s1[67];
              float v516_data = ir3[5];
              ir3[5] = (v516_data + (v488_data * v514_data));
            }
            if (v248_lead < 12) {
              float v522_data = r2[8];
              float v523_data = s1[8];
              float v525_data = ir3[0];
              ir3[0] = (v525_data + (v522_data * v523_data));
              float v528_data = s1[20];
              float v530_data = ir3[1];
              ir3[1] = (v530_data + (v522_data * v528_data));
              float v533_data = s1[32];
              float v535_data = ir3[2];
              ir3[2] = (v535_data + (v522_data * v533_data));
              float v538_data = s1[44];
              float v540_data = ir3[3];
              ir3[3] = (v540_data + (v522_data * v538_data));
              float v543_data = s1[56];
              float v545_data = ir3[4];
              ir3[4] = (v545_data + (v522_data * v543_data));
              float v548_data = s1[68];
              float v550_data = ir3[5];
              ir3[5] = (v550_data + (v522_data * v548_data));
            }
            if (v248_lead < 12) {
              float v556_data = r2[9];
              float v557_data = s1[9];
              float v559_data = ir3[0];
              ir3[0] = (v559_data + (v556_data * v557_data));
              float v562_data = s1[21];
              float v564_data = ir3[1];
              ir3[1] = (v564_data + (v556_data * v562_data));
              float v567_data = s1[33];
              float v569_data = ir3[2];
              ir3[2] = (v569_data + (v556_data * v567_data));
              float v572_data = s1[45];
              float v574_data = ir3[3];
              ir3[3] = (v574_data + (v556_data * v572_data));
              float v577_data = s1[57];
              float v579_data = ir3[4];
              ir3[4] = (v579_data + (v556_data * v577_data));
              float v582_data = s1[69];
              float v584_data = ir3[5];
              ir3[5] = (v584_data + (v556_data * v582_data));
            }
            if (v248_lead < 12) {
              float v590_data = r2[10];
              float v591_data = s1[10];
              float v593_data = ir3[0];
              ir3[0] = (v593_data + (v590_data * v591_data));
              float v596_data = s1[22];
              float v598_data = ir3[1];
              ir3[1] = (v598_data + (v590_data * v596_data));
              float v601_data = s1[34];
              float v603_data = ir3[2];
              ir3[2] = (v603_data + (v590_data * v601_data));
              float v606_data = s1[46];
              float v608_data = ir3[3];
              ir3[3] = (v608_data + (v590_data * v606_data));
              float v611_data = s1[58];
              float v613_data = ir3[4];
              ir3[4] = (v613_data + (v590_data * v611_data));
              float v616_data = s1[70];
              float v618_data = ir3[5];
              ir3[5] = (v618_data + (v590_data * v616_data));
            }
            if (v248_lead < 12) {
              float v624_data = r2[11];
              float v625_data = s1[11];
              float v627_data = ir3[0];
              ir3[0] = (v627_data + (v624_data * v625_data));
              float v630_data = s1[23];
              float v632_data = ir3[1];
              ir3[1] = (v632_data + (v624_data * v630_data));
              float v635_data = s1[35];
              float v637_data = ir3[2];
              ir3[2] = (v637_data + (v624_data * v635_data));
              float v640_data = s1[47];
              float v642_data = ir3[3];
              ir3[3] = (v642_data + (v624_data * v640_data));
              float v645_data = s1[59];
              float v647_data = ir3[4];
              ir3[4] = (v647_data + (v624_data * v645_data));
              float v650_data = s1[71];
              float v652_data = ir3[5];
              ir3[5] = (v652_data + (v624_data * v650_data));
            }
            if (v248_lead < 12) {
              #pragma unroll
              for (int32_t v658_n1 = 0; v658_n1 < 6; ++v658_n1) {
                int32_t v659_a = 0 + v658_n1;
                float v660_data = ir3[v659_a];
                int32_t v661_a = 0 + v658_n1;
                r3[v661_a] = v660_data;
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          int32_t v664_lead = threadIdx.x % 16;
          if (v664_lead < 12) {
            #pragma unroll
            for (int32_t v666_i1 = 0; v666_i1 < 6; ++v666_i1) {
              int32_t v667_a = 0 + v666_i1;
              float v668_data = r3[v667_a];
              int32_t v675_a = v664_lead + (v666_i1 * 12);
              glb_m2[v675_a] = v668_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

