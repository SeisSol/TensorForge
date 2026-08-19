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
              float v239_data = r1[v236_i1];
              int32_t v246_a = v234_lead + (v236_i1 * 12);
              s1[v246_a] = v239_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + None
            // [(0, 12), (0, 6)] [(0, 12)]
            float ir3[6]{};
            int32_t v249_lead = threadIdx.x % 16;
            if (v249_lead < 12) {
              float v251_data = r2[0];
              float v252_data = s1[0];
              float v254_data = ir3[0];
              ir3[0] = (v254_data + (v251_data * v252_data));
              float v257_data = s1[12];
              float v259_data = ir3[1];
              ir3[1] = (v259_data + (v251_data * v257_data));
              float v262_data = s1[24];
              float v264_data = ir3[2];
              ir3[2] = (v264_data + (v251_data * v262_data));
              float v267_data = s1[36];
              float v269_data = ir3[3];
              ir3[3] = (v269_data + (v251_data * v267_data));
              float v272_data = s1[48];
              float v274_data = ir3[4];
              ir3[4] = (v274_data + (v251_data * v272_data));
              float v277_data = s1[60];
              float v279_data = ir3[5];
              ir3[5] = (v279_data + (v251_data * v277_data));
            }
            if (v249_lead < 12) {
              float v285_data = r2[1];
              float v286_data = s1[1];
              float v288_data = ir3[0];
              ir3[0] = (v288_data + (v285_data * v286_data));
              float v291_data = s1[13];
              float v293_data = ir3[1];
              ir3[1] = (v293_data + (v285_data * v291_data));
              float v296_data = s1[25];
              float v298_data = ir3[2];
              ir3[2] = (v298_data + (v285_data * v296_data));
              float v301_data = s1[37];
              float v303_data = ir3[3];
              ir3[3] = (v303_data + (v285_data * v301_data));
              float v306_data = s1[49];
              float v308_data = ir3[4];
              ir3[4] = (v308_data + (v285_data * v306_data));
              float v311_data = s1[61];
              float v313_data = ir3[5];
              ir3[5] = (v313_data + (v285_data * v311_data));
            }
            if (v249_lead < 12) {
              float v319_data = r2[2];
              float v320_data = s1[2];
              float v322_data = ir3[0];
              ir3[0] = (v322_data + (v319_data * v320_data));
              float v325_data = s1[14];
              float v327_data = ir3[1];
              ir3[1] = (v327_data + (v319_data * v325_data));
              float v330_data = s1[26];
              float v332_data = ir3[2];
              ir3[2] = (v332_data + (v319_data * v330_data));
              float v335_data = s1[38];
              float v337_data = ir3[3];
              ir3[3] = (v337_data + (v319_data * v335_data));
              float v340_data = s1[50];
              float v342_data = ir3[4];
              ir3[4] = (v342_data + (v319_data * v340_data));
              float v345_data = s1[62];
              float v347_data = ir3[5];
              ir3[5] = (v347_data + (v319_data * v345_data));
            }
            if (v249_lead < 12) {
              float v353_data = r2[3];
              float v354_data = s1[3];
              float v356_data = ir3[0];
              ir3[0] = (v356_data + (v353_data * v354_data));
              float v359_data = s1[15];
              float v361_data = ir3[1];
              ir3[1] = (v361_data + (v353_data * v359_data));
              float v364_data = s1[27];
              float v366_data = ir3[2];
              ir3[2] = (v366_data + (v353_data * v364_data));
              float v369_data = s1[39];
              float v371_data = ir3[3];
              ir3[3] = (v371_data + (v353_data * v369_data));
              float v374_data = s1[51];
              float v376_data = ir3[4];
              ir3[4] = (v376_data + (v353_data * v374_data));
              float v379_data = s1[63];
              float v381_data = ir3[5];
              ir3[5] = (v381_data + (v353_data * v379_data));
            }
            if (v249_lead < 12) {
              float v387_data = r2[4];
              float v388_data = s1[4];
              float v390_data = ir3[0];
              ir3[0] = (v390_data + (v387_data * v388_data));
              float v393_data = s1[16];
              float v395_data = ir3[1];
              ir3[1] = (v395_data + (v387_data * v393_data));
              float v398_data = s1[28];
              float v400_data = ir3[2];
              ir3[2] = (v400_data + (v387_data * v398_data));
              float v403_data = s1[40];
              float v405_data = ir3[3];
              ir3[3] = (v405_data + (v387_data * v403_data));
              float v408_data = s1[52];
              float v410_data = ir3[4];
              ir3[4] = (v410_data + (v387_data * v408_data));
              float v413_data = s1[64];
              float v415_data = ir3[5];
              ir3[5] = (v415_data + (v387_data * v413_data));
            }
            if (v249_lead < 12) {
              float v421_data = r2[5];
              float v422_data = s1[5];
              float v424_data = ir3[0];
              ir3[0] = (v424_data + (v421_data * v422_data));
              float v427_data = s1[17];
              float v429_data = ir3[1];
              ir3[1] = (v429_data + (v421_data * v427_data));
              float v432_data = s1[29];
              float v434_data = ir3[2];
              ir3[2] = (v434_data + (v421_data * v432_data));
              float v437_data = s1[41];
              float v439_data = ir3[3];
              ir3[3] = (v439_data + (v421_data * v437_data));
              float v442_data = s1[53];
              float v444_data = ir3[4];
              ir3[4] = (v444_data + (v421_data * v442_data));
              float v447_data = s1[65];
              float v449_data = ir3[5];
              ir3[5] = (v449_data + (v421_data * v447_data));
            }
            if (v249_lead < 12) {
              float v455_data = r2[6];
              float v456_data = s1[6];
              float v458_data = ir3[0];
              ir3[0] = (v458_data + (v455_data * v456_data));
              float v461_data = s1[18];
              float v463_data = ir3[1];
              ir3[1] = (v463_data + (v455_data * v461_data));
              float v466_data = s1[30];
              float v468_data = ir3[2];
              ir3[2] = (v468_data + (v455_data * v466_data));
              float v471_data = s1[42];
              float v473_data = ir3[3];
              ir3[3] = (v473_data + (v455_data * v471_data));
              float v476_data = s1[54];
              float v478_data = ir3[4];
              ir3[4] = (v478_data + (v455_data * v476_data));
              float v481_data = s1[66];
              float v483_data = ir3[5];
              ir3[5] = (v483_data + (v455_data * v481_data));
            }
            if (v249_lead < 12) {
              float v489_data = r2[7];
              float v490_data = s1[7];
              float v492_data = ir3[0];
              ir3[0] = (v492_data + (v489_data * v490_data));
              float v495_data = s1[19];
              float v497_data = ir3[1];
              ir3[1] = (v497_data + (v489_data * v495_data));
              float v500_data = s1[31];
              float v502_data = ir3[2];
              ir3[2] = (v502_data + (v489_data * v500_data));
              float v505_data = s1[43];
              float v507_data = ir3[3];
              ir3[3] = (v507_data + (v489_data * v505_data));
              float v510_data = s1[55];
              float v512_data = ir3[4];
              ir3[4] = (v512_data + (v489_data * v510_data));
              float v515_data = s1[67];
              float v517_data = ir3[5];
              ir3[5] = (v517_data + (v489_data * v515_data));
            }
            if (v249_lead < 12) {
              float v523_data = r2[8];
              float v524_data = s1[8];
              float v526_data = ir3[0];
              ir3[0] = (v526_data + (v523_data * v524_data));
              float v529_data = s1[20];
              float v531_data = ir3[1];
              ir3[1] = (v531_data + (v523_data * v529_data));
              float v534_data = s1[32];
              float v536_data = ir3[2];
              ir3[2] = (v536_data + (v523_data * v534_data));
              float v539_data = s1[44];
              float v541_data = ir3[3];
              ir3[3] = (v541_data + (v523_data * v539_data));
              float v544_data = s1[56];
              float v546_data = ir3[4];
              ir3[4] = (v546_data + (v523_data * v544_data));
              float v549_data = s1[68];
              float v551_data = ir3[5];
              ir3[5] = (v551_data + (v523_data * v549_data));
            }
            if (v249_lead < 12) {
              float v557_data = r2[9];
              float v558_data = s1[9];
              float v560_data = ir3[0];
              ir3[0] = (v560_data + (v557_data * v558_data));
              float v563_data = s1[21];
              float v565_data = ir3[1];
              ir3[1] = (v565_data + (v557_data * v563_data));
              float v568_data = s1[33];
              float v570_data = ir3[2];
              ir3[2] = (v570_data + (v557_data * v568_data));
              float v573_data = s1[45];
              float v575_data = ir3[3];
              ir3[3] = (v575_data + (v557_data * v573_data));
              float v578_data = s1[57];
              float v580_data = ir3[4];
              ir3[4] = (v580_data + (v557_data * v578_data));
              float v583_data = s1[69];
              float v585_data = ir3[5];
              ir3[5] = (v585_data + (v557_data * v583_data));
            }
            if (v249_lead < 12) {
              float v591_data = r2[10];
              float v592_data = s1[10];
              float v594_data = ir3[0];
              ir3[0] = (v594_data + (v591_data * v592_data));
              float v597_data = s1[22];
              float v599_data = ir3[1];
              ir3[1] = (v599_data + (v591_data * v597_data));
              float v602_data = s1[34];
              float v604_data = ir3[2];
              ir3[2] = (v604_data + (v591_data * v602_data));
              float v607_data = s1[46];
              float v609_data = ir3[3];
              ir3[3] = (v609_data + (v591_data * v607_data));
              float v612_data = s1[58];
              float v614_data = ir3[4];
              ir3[4] = (v614_data + (v591_data * v612_data));
              float v617_data = s1[70];
              float v619_data = ir3[5];
              ir3[5] = (v619_data + (v591_data * v617_data));
            }
            if (v249_lead < 12) {
              float v625_data = r2[11];
              float v626_data = s1[11];
              float v628_data = ir3[0];
              ir3[0] = (v628_data + (v625_data * v626_data));
              float v631_data = s1[23];
              float v633_data = ir3[1];
              ir3[1] = (v633_data + (v625_data * v631_data));
              float v636_data = s1[35];
              float v638_data = ir3[2];
              ir3[2] = (v638_data + (v625_data * v636_data));
              float v641_data = s1[47];
              float v643_data = ir3[3];
              ir3[3] = (v643_data + (v625_data * v641_data));
              float v646_data = s1[59];
              float v648_data = ir3[4];
              ir3[4] = (v648_data + (v625_data * v646_data));
              float v651_data = s1[71];
              float v653_data = ir3[5];
              ir3[5] = (v653_data + (v625_data * v651_data));
            }
            if (v249_lead < 12) {
              #pragma unroll
              for (int32_t v659_n1 = 0; v659_n1 < 6; ++v659_n1) {
                int32_t v660_a = 0 + v659_n1;
                float v662_data = ir3[v659_n1];
                int32_t v663_a = 0 + v659_n1;
                r3[v663_a] = v662_data;
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          int32_t v666_lead = threadIdx.x % 16;
          if (v666_lead < 12) {
            #pragma unroll
            for (int32_t v668_i1 = 0; v668_i1 < 6; ++v668_i1) {
              int32_t v669_a = 0 + v668_i1;
              float v671_data = r3[v668_i1];
              int32_t v678_a = v666_lead + (v668_i1 * 12);
              glb_m2[v678_a] = v671_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

