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
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              int32_t v34_a = v28_i1 * 12;
              int32_t v35_a = v3_lead + v34_a;
              float v43_data = __ldcg(&glb_m3[(v3_lead + v34_a)]);
              int32_t v44_a = 0 + v28_i1;
              r2[v44_a] = v43_data;
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
            float v50_data = r0[0];
            float v51_data = s0[0];
            float v53_data = ir1[0];
            ir1[0] = (v53_data + (v50_data * v51_data));
            float v56_data = s0[6];
            float v58_data = ir1[1];
            ir1[1] = (v58_data + (v50_data * v56_data));
            float v61_data = s0[12];
            float v63_data = ir1[2];
            ir1[2] = (v63_data + (v50_data * v61_data));
            float v66_data = s0[18];
            float v68_data = ir1[3];
            ir1[3] = (v68_data + (v50_data * v66_data));
            float v71_data = s0[24];
            float v73_data = ir1[4];
            ir1[4] = (v73_data + (v50_data * v71_data));
            float v76_data = s0[30];
            float v78_data = ir1[5];
            ir1[5] = (v78_data + (v50_data * v76_data));
          }
          if (v3_lead < 12) {
            float v84_data = r0[1];
            float v85_data = s0[1];
            float v87_data = ir1[0];
            ir1[0] = (v87_data + (v84_data * v85_data));
            float v90_data = s0[7];
            float v92_data = ir1[1];
            ir1[1] = (v92_data + (v84_data * v90_data));
            float v95_data = s0[13];
            float v97_data = ir1[2];
            ir1[2] = (v97_data + (v84_data * v95_data));
            float v100_data = s0[19];
            float v102_data = ir1[3];
            ir1[3] = (v102_data + (v84_data * v100_data));
            float v105_data = s0[25];
            float v107_data = ir1[4];
            ir1[4] = (v107_data + (v84_data * v105_data));
            float v110_data = s0[31];
            float v112_data = ir1[5];
            ir1[5] = (v112_data + (v84_data * v110_data));
          }
          if (v3_lead < 12) {
            float v118_data = r0[2];
            float v119_data = s0[2];
            float v121_data = ir1[0];
            ir1[0] = (v121_data + (v118_data * v119_data));
            float v124_data = s0[8];
            float v126_data = ir1[1];
            ir1[1] = (v126_data + (v118_data * v124_data));
            float v129_data = s0[14];
            float v131_data = ir1[2];
            ir1[2] = (v131_data + (v118_data * v129_data));
            float v134_data = s0[20];
            float v136_data = ir1[3];
            ir1[3] = (v136_data + (v118_data * v134_data));
            float v139_data = s0[26];
            float v141_data = ir1[4];
            ir1[4] = (v141_data + (v118_data * v139_data));
            float v144_data = s0[32];
            float v146_data = ir1[5];
            ir1[5] = (v146_data + (v118_data * v144_data));
          }
          if (v3_lead < 12) {
            float v152_data = r0[3];
            float v153_data = s0[3];
            float v155_data = ir1[0];
            ir1[0] = (v155_data + (v152_data * v153_data));
            float v158_data = s0[9];
            float v160_data = ir1[1];
            ir1[1] = (v160_data + (v152_data * v158_data));
            float v163_data = s0[15];
            float v165_data = ir1[2];
            ir1[2] = (v165_data + (v152_data * v163_data));
            float v168_data = s0[21];
            float v170_data = ir1[3];
            ir1[3] = (v170_data + (v152_data * v168_data));
            float v173_data = s0[27];
            float v175_data = ir1[4];
            ir1[4] = (v175_data + (v152_data * v173_data));
            float v178_data = s0[33];
            float v180_data = ir1[5];
            ir1[5] = (v180_data + (v152_data * v178_data));
          }
          if (v3_lead < 12) {
            float v186_data = r0[4];
            float v187_data = s0[4];
            float v189_data = ir1[0];
            ir1[0] = (v189_data + (v186_data * v187_data));
            float v192_data = s0[10];
            float v194_data = ir1[1];
            ir1[1] = (v194_data + (v186_data * v192_data));
            float v197_data = s0[16];
            float v199_data = ir1[2];
            ir1[2] = (v199_data + (v186_data * v197_data));
            float v202_data = s0[22];
            float v204_data = ir1[3];
            ir1[3] = (v204_data + (v186_data * v202_data));
            float v207_data = s0[28];
            float v209_data = ir1[4];
            ir1[4] = (v209_data + (v186_data * v207_data));
            float v212_data = s0[34];
            float v214_data = ir1[5];
            ir1[5] = (v214_data + (v186_data * v212_data));
          }
          if (v3_lead < 12) {
            float v220_data = r0[5];
            float v221_data = s0[5];
            float v223_data = ir1[0];
            ir1[0] = (v223_data + (v220_data * v221_data));
            float v226_data = s0[11];
            float v228_data = ir1[1];
            ir1[1] = (v228_data + (v220_data * v226_data));
            float v231_data = s0[17];
            float v233_data = ir1[2];
            ir1[2] = (v233_data + (v220_data * v231_data));
            float v236_data = s0[23];
            float v238_data = ir1[3];
            ir1[3] = (v238_data + (v220_data * v236_data));
            float v241_data = s0[29];
            float v243_data = ir1[4];
            ir1[4] = (v243_data + (v220_data * v241_data));
            float v246_data = s0[35];
            float v248_data = ir1[5];
            ir1[5] = (v248_data + (v220_data * v246_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 6; ++v255_i1) {
              int32_t v256_a = 0 + v255_i1;
              float v258_data = r1[v255_i1];
              int32_t v265_a = v3_lead + (v255_i1 * 12);
              s1[v265_a] = v258_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v3_lead < 12) {
            float v272_data = r2[0];
            float v273_data = s1[0];
            float v275_data = ir3[0];
            ir3[0] = (v275_data + (v272_data * v273_data));
            float v278_data = s1[12];
            float v280_data = ir3[1];
            ir3[1] = (v280_data + (v272_data * v278_data));
            float v283_data = s1[24];
            float v285_data = ir3[2];
            ir3[2] = (v285_data + (v272_data * v283_data));
            float v288_data = s1[36];
            float v290_data = ir3[3];
            ir3[3] = (v290_data + (v272_data * v288_data));
            float v293_data = s1[48];
            float v295_data = ir3[4];
            ir3[4] = (v295_data + (v272_data * v293_data));
            float v298_data = s1[60];
            float v300_data = ir3[5];
            ir3[5] = (v300_data + (v272_data * v298_data));
          }
          if (v3_lead < 12) {
            float v306_data = r2[1];
            float v307_data = s1[1];
            float v309_data = ir3[0];
            ir3[0] = (v309_data + (v306_data * v307_data));
            float v312_data = s1[13];
            float v314_data = ir3[1];
            ir3[1] = (v314_data + (v306_data * v312_data));
            float v317_data = s1[25];
            float v319_data = ir3[2];
            ir3[2] = (v319_data + (v306_data * v317_data));
            float v322_data = s1[37];
            float v324_data = ir3[3];
            ir3[3] = (v324_data + (v306_data * v322_data));
            float v327_data = s1[49];
            float v329_data = ir3[4];
            ir3[4] = (v329_data + (v306_data * v327_data));
            float v332_data = s1[61];
            float v334_data = ir3[5];
            ir3[5] = (v334_data + (v306_data * v332_data));
          }
          if (v3_lead < 12) {
            float v340_data = r2[2];
            float v341_data = s1[2];
            float v343_data = ir3[0];
            ir3[0] = (v343_data + (v340_data * v341_data));
            float v346_data = s1[14];
            float v348_data = ir3[1];
            ir3[1] = (v348_data + (v340_data * v346_data));
            float v351_data = s1[26];
            float v353_data = ir3[2];
            ir3[2] = (v353_data + (v340_data * v351_data));
            float v356_data = s1[38];
            float v358_data = ir3[3];
            ir3[3] = (v358_data + (v340_data * v356_data));
            float v361_data = s1[50];
            float v363_data = ir3[4];
            ir3[4] = (v363_data + (v340_data * v361_data));
            float v366_data = s1[62];
            float v368_data = ir3[5];
            ir3[5] = (v368_data + (v340_data * v366_data));
          }
          if (v3_lead < 12) {
            float v374_data = r2[3];
            float v375_data = s1[3];
            float v377_data = ir3[0];
            ir3[0] = (v377_data + (v374_data * v375_data));
            float v380_data = s1[15];
            float v382_data = ir3[1];
            ir3[1] = (v382_data + (v374_data * v380_data));
            float v385_data = s1[27];
            float v387_data = ir3[2];
            ir3[2] = (v387_data + (v374_data * v385_data));
            float v390_data = s1[39];
            float v392_data = ir3[3];
            ir3[3] = (v392_data + (v374_data * v390_data));
            float v395_data = s1[51];
            float v397_data = ir3[4];
            ir3[4] = (v397_data + (v374_data * v395_data));
            float v400_data = s1[63];
            float v402_data = ir3[5];
            ir3[5] = (v402_data + (v374_data * v400_data));
          }
          if (v3_lead < 12) {
            float v408_data = r2[4];
            float v409_data = s1[4];
            float v411_data = ir3[0];
            ir3[0] = (v411_data + (v408_data * v409_data));
            float v414_data = s1[16];
            float v416_data = ir3[1];
            ir3[1] = (v416_data + (v408_data * v414_data));
            float v419_data = s1[28];
            float v421_data = ir3[2];
            ir3[2] = (v421_data + (v408_data * v419_data));
            float v424_data = s1[40];
            float v426_data = ir3[3];
            ir3[3] = (v426_data + (v408_data * v424_data));
            float v429_data = s1[52];
            float v431_data = ir3[4];
            ir3[4] = (v431_data + (v408_data * v429_data));
            float v434_data = s1[64];
            float v436_data = ir3[5];
            ir3[5] = (v436_data + (v408_data * v434_data));
          }
          if (v3_lead < 12) {
            float v442_data = r2[5];
            float v443_data = s1[5];
            float v445_data = ir3[0];
            ir3[0] = (v445_data + (v442_data * v443_data));
            float v448_data = s1[17];
            float v450_data = ir3[1];
            ir3[1] = (v450_data + (v442_data * v448_data));
            float v453_data = s1[29];
            float v455_data = ir3[2];
            ir3[2] = (v455_data + (v442_data * v453_data));
            float v458_data = s1[41];
            float v460_data = ir3[3];
            ir3[3] = (v460_data + (v442_data * v458_data));
            float v463_data = s1[53];
            float v465_data = ir3[4];
            ir3[4] = (v465_data + (v442_data * v463_data));
            float v468_data = s1[65];
            float v470_data = ir3[5];
            ir3[5] = (v470_data + (v442_data * v468_data));
          }
          if (v3_lead < 12) {
            float v476_data = r2[6];
            float v477_data = s1[6];
            float v479_data = ir3[0];
            ir3[0] = (v479_data + (v476_data * v477_data));
            float v482_data = s1[18];
            float v484_data = ir3[1];
            ir3[1] = (v484_data + (v476_data * v482_data));
            float v487_data = s1[30];
            float v489_data = ir3[2];
            ir3[2] = (v489_data + (v476_data * v487_data));
            float v492_data = s1[42];
            float v494_data = ir3[3];
            ir3[3] = (v494_data + (v476_data * v492_data));
            float v497_data = s1[54];
            float v499_data = ir3[4];
            ir3[4] = (v499_data + (v476_data * v497_data));
            float v502_data = s1[66];
            float v504_data = ir3[5];
            ir3[5] = (v504_data + (v476_data * v502_data));
          }
          if (v3_lead < 12) {
            float v510_data = r2[7];
            float v511_data = s1[7];
            float v513_data = ir3[0];
            ir3[0] = (v513_data + (v510_data * v511_data));
            float v516_data = s1[19];
            float v518_data = ir3[1];
            ir3[1] = (v518_data + (v510_data * v516_data));
            float v521_data = s1[31];
            float v523_data = ir3[2];
            ir3[2] = (v523_data + (v510_data * v521_data));
            float v526_data = s1[43];
            float v528_data = ir3[3];
            ir3[3] = (v528_data + (v510_data * v526_data));
            float v531_data = s1[55];
            float v533_data = ir3[4];
            ir3[4] = (v533_data + (v510_data * v531_data));
            float v536_data = s1[67];
            float v538_data = ir3[5];
            ir3[5] = (v538_data + (v510_data * v536_data));
          }
          if (v3_lead < 12) {
            float v544_data = r2[8];
            float v545_data = s1[8];
            float v547_data = ir3[0];
            ir3[0] = (v547_data + (v544_data * v545_data));
            float v550_data = s1[20];
            float v552_data = ir3[1];
            ir3[1] = (v552_data + (v544_data * v550_data));
            float v555_data = s1[32];
            float v557_data = ir3[2];
            ir3[2] = (v557_data + (v544_data * v555_data));
            float v560_data = s1[44];
            float v562_data = ir3[3];
            ir3[3] = (v562_data + (v544_data * v560_data));
            float v565_data = s1[56];
            float v567_data = ir3[4];
            ir3[4] = (v567_data + (v544_data * v565_data));
            float v570_data = s1[68];
            float v572_data = ir3[5];
            ir3[5] = (v572_data + (v544_data * v570_data));
          }
          if (v3_lead < 12) {
            float v578_data = r2[9];
            float v579_data = s1[9];
            float v581_data = ir3[0];
            ir3[0] = (v581_data + (v578_data * v579_data));
            float v584_data = s1[21];
            float v586_data = ir3[1];
            ir3[1] = (v586_data + (v578_data * v584_data));
            float v589_data = s1[33];
            float v591_data = ir3[2];
            ir3[2] = (v591_data + (v578_data * v589_data));
            float v594_data = s1[45];
            float v596_data = ir3[3];
            ir3[3] = (v596_data + (v578_data * v594_data));
            float v599_data = s1[57];
            float v601_data = ir3[4];
            ir3[4] = (v601_data + (v578_data * v599_data));
            float v604_data = s1[69];
            float v606_data = ir3[5];
            ir3[5] = (v606_data + (v578_data * v604_data));
          }
          if (v3_lead < 12) {
            float v612_data = r2[10];
            float v613_data = s1[10];
            float v615_data = ir3[0];
            ir3[0] = (v615_data + (v612_data * v613_data));
            float v618_data = s1[22];
            float v620_data = ir3[1];
            ir3[1] = (v620_data + (v612_data * v618_data));
            float v623_data = s1[34];
            float v625_data = ir3[2];
            ir3[2] = (v625_data + (v612_data * v623_data));
            float v628_data = s1[46];
            float v630_data = ir3[3];
            ir3[3] = (v630_data + (v612_data * v628_data));
            float v633_data = s1[58];
            float v635_data = ir3[4];
            ir3[4] = (v635_data + (v612_data * v633_data));
            float v638_data = s1[70];
            float v640_data = ir3[5];
            ir3[5] = (v640_data + (v612_data * v638_data));
          }
          if (v3_lead < 12) {
            float v646_data = r2[11];
            float v647_data = s1[11];
            float v649_data = ir3[0];
            ir3[0] = (v649_data + (v646_data * v647_data));
            float v652_data = s1[23];
            float v654_data = ir3[1];
            ir3[1] = (v654_data + (v646_data * v652_data));
            float v657_data = s1[35];
            float v659_data = ir3[2];
            ir3[2] = (v659_data + (v646_data * v657_data));
            float v662_data = s1[47];
            float v664_data = ir3[3];
            ir3[3] = (v664_data + (v646_data * v662_data));
            float v667_data = s1[59];
            float v669_data = ir3[4];
            ir3[4] = (v669_data + (v646_data * v667_data));
            float v672_data = s1[71];
            float v674_data = ir3[5];
            ir3[5] = (v674_data + (v646_data * v672_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v680_n1 = 0; v680_n1 < 6; ++v680_n1) {
              int32_t v681_a = 0 + v680_n1;
              float v683_data = ir3[v680_n1];
              int32_t v684_a = 0 + v680_n1;
              r3[v680_n1] = v683_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v690_i1 = 0; v690_i1 < 6; ++v690_i1) {
              int32_t v691_a = 0 + v690_i1;
              float v693_data = r3[v690_i1];
              int32_t v700_a = v3_lead + (v690_i1 * 12);
              glb_m2[v700_a] = v693_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

