// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a03a3cd0d, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_8a03a3cd0d, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8a03a3cd0d<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
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
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir1[8]{};
            if (v3_lead < 12) {
              float v49_data = r0[0];
              float v50_data = s0[0];
              float v52_data = ir1[0];
              ir1[0] = (v52_data + (v49_data * v50_data));
              float v55_data = s0[12];
              float v57_data = ir1[1];
              ir1[1] = (v57_data + (v49_data * v55_data));
              float v60_data = s0[24];
              float v62_data = ir1[2];
              ir1[2] = (v62_data + (v49_data * v60_data));
              float v65_data = s0[36];
              float v67_data = ir1[3];
              ir1[3] = (v67_data + (v49_data * v65_data));
              float v70_data = s0[48];
              float v72_data = ir1[4];
              ir1[4] = (v72_data + (v49_data * v70_data));
              float v75_data = s0[60];
              float v77_data = ir1[5];
              ir1[5] = (v77_data + (v49_data * v75_data));
              float v80_data = s0[72];
              float v82_data = ir1[6];
              ir1[6] = (v82_data + (v49_data * v80_data));
              float v85_data = s0[84];
              float v87_data = ir1[7];
              ir1[7] = (v87_data + (v49_data * v85_data));
            }
            if (v3_lead < 12) {
              float v93_data = r0[1];
              float v94_data = s0[1];
              float v96_data = ir1[0];
              ir1[0] = (v96_data + (v93_data * v94_data));
              float v99_data = s0[13];
              float v101_data = ir1[1];
              ir1[1] = (v101_data + (v93_data * v99_data));
              float v104_data = s0[25];
              float v106_data = ir1[2];
              ir1[2] = (v106_data + (v93_data * v104_data));
              float v109_data = s0[37];
              float v111_data = ir1[3];
              ir1[3] = (v111_data + (v93_data * v109_data));
              float v114_data = s0[49];
              float v116_data = ir1[4];
              ir1[4] = (v116_data + (v93_data * v114_data));
              float v119_data = s0[61];
              float v121_data = ir1[5];
              ir1[5] = (v121_data + (v93_data * v119_data));
              float v124_data = s0[73];
              float v126_data = ir1[6];
              ir1[6] = (v126_data + (v93_data * v124_data));
              float v129_data = s0[85];
              float v131_data = ir1[7];
              ir1[7] = (v131_data + (v93_data * v129_data));
            }
            if (v3_lead < 12) {
              float v137_data = r0[2];
              float v138_data = s0[2];
              float v140_data = ir1[0];
              ir1[0] = (v140_data + (v137_data * v138_data));
              float v143_data = s0[14];
              float v145_data = ir1[1];
              ir1[1] = (v145_data + (v137_data * v143_data));
              float v148_data = s0[26];
              float v150_data = ir1[2];
              ir1[2] = (v150_data + (v137_data * v148_data));
              float v153_data = s0[38];
              float v155_data = ir1[3];
              ir1[3] = (v155_data + (v137_data * v153_data));
              float v158_data = s0[50];
              float v160_data = ir1[4];
              ir1[4] = (v160_data + (v137_data * v158_data));
              float v163_data = s0[62];
              float v165_data = ir1[5];
              ir1[5] = (v165_data + (v137_data * v163_data));
              float v168_data = s0[74];
              float v170_data = ir1[6];
              ir1[6] = (v170_data + (v137_data * v168_data));
              float v173_data = s0[86];
              float v175_data = ir1[7];
              ir1[7] = (v175_data + (v137_data * v173_data));
            }
            if (v3_lead < 12) {
              float v181_data = r0[3];
              float v182_data = s0[3];
              float v184_data = ir1[0];
              ir1[0] = (v184_data + (v181_data * v182_data));
              float v187_data = s0[15];
              float v189_data = ir1[1];
              ir1[1] = (v189_data + (v181_data * v187_data));
              float v192_data = s0[27];
              float v194_data = ir1[2];
              ir1[2] = (v194_data + (v181_data * v192_data));
              float v197_data = s0[39];
              float v199_data = ir1[3];
              ir1[3] = (v199_data + (v181_data * v197_data));
              float v202_data = s0[51];
              float v204_data = ir1[4];
              ir1[4] = (v204_data + (v181_data * v202_data));
              float v207_data = s0[63];
              float v209_data = ir1[5];
              ir1[5] = (v209_data + (v181_data * v207_data));
              float v212_data = s0[75];
              float v214_data = ir1[6];
              ir1[6] = (v214_data + (v181_data * v212_data));
              float v217_data = s0[87];
              float v219_data = ir1[7];
              ir1[7] = (v219_data + (v181_data * v217_data));
            }
            if (v3_lead < 12) {
              float v225_data = r0[4];
              float v226_data = s0[4];
              float v228_data = ir1[0];
              ir1[0] = (v228_data + (v225_data * v226_data));
              float v231_data = s0[16];
              float v233_data = ir1[1];
              ir1[1] = (v233_data + (v225_data * v231_data));
              float v236_data = s0[28];
              float v238_data = ir1[2];
              ir1[2] = (v238_data + (v225_data * v236_data));
              float v241_data = s0[40];
              float v243_data = ir1[3];
              ir1[3] = (v243_data + (v225_data * v241_data));
              float v246_data = s0[52];
              float v248_data = ir1[4];
              ir1[4] = (v248_data + (v225_data * v246_data));
              float v251_data = s0[64];
              float v253_data = ir1[5];
              ir1[5] = (v253_data + (v225_data * v251_data));
              float v256_data = s0[76];
              float v258_data = ir1[6];
              ir1[6] = (v258_data + (v225_data * v256_data));
              float v261_data = s0[88];
              float v263_data = ir1[7];
              ir1[7] = (v263_data + (v225_data * v261_data));
            }
            if (v3_lead < 12) {
              float v269_data = r0[5];
              float v270_data = s0[5];
              float v272_data = ir1[0];
              ir1[0] = (v272_data + (v269_data * v270_data));
              float v275_data = s0[17];
              float v277_data = ir1[1];
              ir1[1] = (v277_data + (v269_data * v275_data));
              float v280_data = s0[29];
              float v282_data = ir1[2];
              ir1[2] = (v282_data + (v269_data * v280_data));
              float v285_data = s0[41];
              float v287_data = ir1[3];
              ir1[3] = (v287_data + (v269_data * v285_data));
              float v290_data = s0[53];
              float v292_data = ir1[4];
              ir1[4] = (v292_data + (v269_data * v290_data));
              float v295_data = s0[65];
              float v297_data = ir1[5];
              ir1[5] = (v297_data + (v269_data * v295_data));
              float v300_data = s0[77];
              float v302_data = ir1[6];
              ir1[6] = (v302_data + (v269_data * v300_data));
              float v305_data = s0[89];
              float v307_data = ir1[7];
              ir1[7] = (v307_data + (v269_data * v305_data));
            }
            if (v3_lead < 12) {
              float v313_data = r0[6];
              float v314_data = s0[6];
              float v316_data = ir1[0];
              ir1[0] = (v316_data + (v313_data * v314_data));
              float v319_data = s0[18];
              float v321_data = ir1[1];
              ir1[1] = (v321_data + (v313_data * v319_data));
              float v324_data = s0[30];
              float v326_data = ir1[2];
              ir1[2] = (v326_data + (v313_data * v324_data));
              float v329_data = s0[42];
              float v331_data = ir1[3];
              ir1[3] = (v331_data + (v313_data * v329_data));
              float v334_data = s0[54];
              float v336_data = ir1[4];
              ir1[4] = (v336_data + (v313_data * v334_data));
              float v339_data = s0[66];
              float v341_data = ir1[5];
              ir1[5] = (v341_data + (v313_data * v339_data));
              float v344_data = s0[78];
              float v346_data = ir1[6];
              ir1[6] = (v346_data + (v313_data * v344_data));
              float v349_data = s0[90];
              float v351_data = ir1[7];
              ir1[7] = (v351_data + (v313_data * v349_data));
            }
            if (v3_lead < 12) {
              float v357_data = r0[7];
              float v358_data = s0[7];
              float v360_data = ir1[0];
              ir1[0] = (v360_data + (v357_data * v358_data));
              float v363_data = s0[19];
              float v365_data = ir1[1];
              ir1[1] = (v365_data + (v357_data * v363_data));
              float v368_data = s0[31];
              float v370_data = ir1[2];
              ir1[2] = (v370_data + (v357_data * v368_data));
              float v373_data = s0[43];
              float v375_data = ir1[3];
              ir1[3] = (v375_data + (v357_data * v373_data));
              float v378_data = s0[55];
              float v380_data = ir1[4];
              ir1[4] = (v380_data + (v357_data * v378_data));
              float v383_data = s0[67];
              float v385_data = ir1[5];
              ir1[5] = (v385_data + (v357_data * v383_data));
              float v388_data = s0[79];
              float v390_data = ir1[6];
              ir1[6] = (v390_data + (v357_data * v388_data));
              float v393_data = s0[91];
              float v395_data = ir1[7];
              ir1[7] = (v395_data + (v357_data * v393_data));
            }
            if (v3_lead < 12) {
              float v401_data = r0[8];
              float v402_data = s0[8];
              float v404_data = ir1[0];
              ir1[0] = (v404_data + (v401_data * v402_data));
              float v407_data = s0[20];
              float v409_data = ir1[1];
              ir1[1] = (v409_data + (v401_data * v407_data));
              float v412_data = s0[32];
              float v414_data = ir1[2];
              ir1[2] = (v414_data + (v401_data * v412_data));
              float v417_data = s0[44];
              float v419_data = ir1[3];
              ir1[3] = (v419_data + (v401_data * v417_data));
              float v422_data = s0[56];
              float v424_data = ir1[4];
              ir1[4] = (v424_data + (v401_data * v422_data));
              float v427_data = s0[68];
              float v429_data = ir1[5];
              ir1[5] = (v429_data + (v401_data * v427_data));
              float v432_data = s0[80];
              float v434_data = ir1[6];
              ir1[6] = (v434_data + (v401_data * v432_data));
              float v437_data = s0[92];
              float v439_data = ir1[7];
              ir1[7] = (v439_data + (v401_data * v437_data));
            }
            if (v3_lead < 12) {
              float v445_data = r0[9];
              float v446_data = s0[9];
              float v448_data = ir1[0];
              ir1[0] = (v448_data + (v445_data * v446_data));
              float v451_data = s0[21];
              float v453_data = ir1[1];
              ir1[1] = (v453_data + (v445_data * v451_data));
              float v456_data = s0[33];
              float v458_data = ir1[2];
              ir1[2] = (v458_data + (v445_data * v456_data));
              float v461_data = s0[45];
              float v463_data = ir1[3];
              ir1[3] = (v463_data + (v445_data * v461_data));
              float v466_data = s0[57];
              float v468_data = ir1[4];
              ir1[4] = (v468_data + (v445_data * v466_data));
              float v471_data = s0[69];
              float v473_data = ir1[5];
              ir1[5] = (v473_data + (v445_data * v471_data));
              float v476_data = s0[81];
              float v478_data = ir1[6];
              ir1[6] = (v478_data + (v445_data * v476_data));
              float v481_data = s0[93];
              float v483_data = ir1[7];
              ir1[7] = (v483_data + (v445_data * v481_data));
            }
            if (v3_lead < 12) {
              float v489_data = r0[10];
              float v490_data = s0[10];
              float v492_data = ir1[0];
              ir1[0] = (v492_data + (v489_data * v490_data));
              float v495_data = s0[22];
              float v497_data = ir1[1];
              ir1[1] = (v497_data + (v489_data * v495_data));
              float v500_data = s0[34];
              float v502_data = ir1[2];
              ir1[2] = (v502_data + (v489_data * v500_data));
              float v505_data = s0[46];
              float v507_data = ir1[3];
              ir1[3] = (v507_data + (v489_data * v505_data));
              float v510_data = s0[58];
              float v512_data = ir1[4];
              ir1[4] = (v512_data + (v489_data * v510_data));
              float v515_data = s0[70];
              float v517_data = ir1[5];
              ir1[5] = (v517_data + (v489_data * v515_data));
              float v520_data = s0[82];
              float v522_data = ir1[6];
              ir1[6] = (v522_data + (v489_data * v520_data));
              float v525_data = s0[94];
              float v527_data = ir1[7];
              ir1[7] = (v527_data + (v489_data * v525_data));
            }
            if (v3_lead < 12) {
              float v533_data = r0[11];
              float v534_data = s0[11];
              float v536_data = ir1[0];
              ir1[0] = (v536_data + (v533_data * v534_data));
              float v539_data = s0[23];
              float v541_data = ir1[1];
              ir1[1] = (v541_data + (v533_data * v539_data));
              float v544_data = s0[35];
              float v546_data = ir1[2];
              ir1[2] = (v546_data + (v533_data * v544_data));
              float v549_data = s0[47];
              float v551_data = ir1[3];
              ir1[3] = (v551_data + (v533_data * v549_data));
              float v554_data = s0[59];
              float v556_data = ir1[4];
              ir1[4] = (v556_data + (v533_data * v554_data));
              float v559_data = s0[71];
              float v561_data = ir1[5];
              ir1[5] = (v561_data + (v533_data * v559_data));
              float v564_data = s0[83];
              float v566_data = ir1[6];
              ir1[6] = (v566_data + (v533_data * v564_data));
              float v569_data = s0[95];
              float v571_data = ir1[7];
              ir1[7] = (v571_data + (v533_data * v569_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v577_n1 = 0; v577_n1 < 8; ++v577_n1) {
                int32_t v578_a = 0 + v577_n1;
                float v580_data = ir1[v577_n1];
                int32_t v581_a = 0 + v577_n1;
                r1[v577_n1] = v580_data;
              }
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v588_i1 = 0; v588_i1 < 12; ++v588_i1) {
              int32_t v594_a = v588_i1 * 12;
              int32_t v595_a = v3_lead + v594_a;
              float v603_data = __ldcg(&glb_m5[(v3_lead + v594_a)]);
              int32_t v604_a = 0 + v588_i1;
              r4[v604_a] = v603_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r3[8]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir3[8]{};
            if (v3_lead < 12) {
              float v610_data = r2[0];
              float v611_data = s1[0];
              float v613_data = ir3[0];
              ir3[0] = (v613_data + (v610_data * v611_data));
              float v616_data = s1[12];
              float v618_data = ir3[1];
              ir3[1] = (v618_data + (v610_data * v616_data));
              float v621_data = s1[24];
              float v623_data = ir3[2];
              ir3[2] = (v623_data + (v610_data * v621_data));
              float v626_data = s1[36];
              float v628_data = ir3[3];
              ir3[3] = (v628_data + (v610_data * v626_data));
              float v631_data = s1[48];
              float v633_data = ir3[4];
              ir3[4] = (v633_data + (v610_data * v631_data));
              float v636_data = s1[60];
              float v638_data = ir3[5];
              ir3[5] = (v638_data + (v610_data * v636_data));
              float v641_data = s1[72];
              float v643_data = ir3[6];
              ir3[6] = (v643_data + (v610_data * v641_data));
              float v646_data = s1[84];
              float v648_data = ir3[7];
              ir3[7] = (v648_data + (v610_data * v646_data));
            }
            if (v3_lead < 12) {
              float v654_data = r2[1];
              float v655_data = s1[1];
              float v657_data = ir3[0];
              ir3[0] = (v657_data + (v654_data * v655_data));
              float v660_data = s1[13];
              float v662_data = ir3[1];
              ir3[1] = (v662_data + (v654_data * v660_data));
              float v665_data = s1[25];
              float v667_data = ir3[2];
              ir3[2] = (v667_data + (v654_data * v665_data));
              float v670_data = s1[37];
              float v672_data = ir3[3];
              ir3[3] = (v672_data + (v654_data * v670_data));
              float v675_data = s1[49];
              float v677_data = ir3[4];
              ir3[4] = (v677_data + (v654_data * v675_data));
              float v680_data = s1[61];
              float v682_data = ir3[5];
              ir3[5] = (v682_data + (v654_data * v680_data));
              float v685_data = s1[73];
              float v687_data = ir3[6];
              ir3[6] = (v687_data + (v654_data * v685_data));
              float v690_data = s1[85];
              float v692_data = ir3[7];
              ir3[7] = (v692_data + (v654_data * v690_data));
            }
            if (v3_lead < 12) {
              float v698_data = r2[2];
              float v699_data = s1[2];
              float v701_data = ir3[0];
              ir3[0] = (v701_data + (v698_data * v699_data));
              float v704_data = s1[14];
              float v706_data = ir3[1];
              ir3[1] = (v706_data + (v698_data * v704_data));
              float v709_data = s1[26];
              float v711_data = ir3[2];
              ir3[2] = (v711_data + (v698_data * v709_data));
              float v714_data = s1[38];
              float v716_data = ir3[3];
              ir3[3] = (v716_data + (v698_data * v714_data));
              float v719_data = s1[50];
              float v721_data = ir3[4];
              ir3[4] = (v721_data + (v698_data * v719_data));
              float v724_data = s1[62];
              float v726_data = ir3[5];
              ir3[5] = (v726_data + (v698_data * v724_data));
              float v729_data = s1[74];
              float v731_data = ir3[6];
              ir3[6] = (v731_data + (v698_data * v729_data));
              float v734_data = s1[86];
              float v736_data = ir3[7];
              ir3[7] = (v736_data + (v698_data * v734_data));
            }
            if (v3_lead < 12) {
              float v742_data = r2[3];
              float v743_data = s1[3];
              float v745_data = ir3[0];
              ir3[0] = (v745_data + (v742_data * v743_data));
              float v748_data = s1[15];
              float v750_data = ir3[1];
              ir3[1] = (v750_data + (v742_data * v748_data));
              float v753_data = s1[27];
              float v755_data = ir3[2];
              ir3[2] = (v755_data + (v742_data * v753_data));
              float v758_data = s1[39];
              float v760_data = ir3[3];
              ir3[3] = (v760_data + (v742_data * v758_data));
              float v763_data = s1[51];
              float v765_data = ir3[4];
              ir3[4] = (v765_data + (v742_data * v763_data));
              float v768_data = s1[63];
              float v770_data = ir3[5];
              ir3[5] = (v770_data + (v742_data * v768_data));
              float v773_data = s1[75];
              float v775_data = ir3[6];
              ir3[6] = (v775_data + (v742_data * v773_data));
              float v778_data = s1[87];
              float v780_data = ir3[7];
              ir3[7] = (v780_data + (v742_data * v778_data));
            }
            if (v3_lead < 12) {
              float v786_data = r2[4];
              float v787_data = s1[4];
              float v789_data = ir3[0];
              ir3[0] = (v789_data + (v786_data * v787_data));
              float v792_data = s1[16];
              float v794_data = ir3[1];
              ir3[1] = (v794_data + (v786_data * v792_data));
              float v797_data = s1[28];
              float v799_data = ir3[2];
              ir3[2] = (v799_data + (v786_data * v797_data));
              float v802_data = s1[40];
              float v804_data = ir3[3];
              ir3[3] = (v804_data + (v786_data * v802_data));
              float v807_data = s1[52];
              float v809_data = ir3[4];
              ir3[4] = (v809_data + (v786_data * v807_data));
              float v812_data = s1[64];
              float v814_data = ir3[5];
              ir3[5] = (v814_data + (v786_data * v812_data));
              float v817_data = s1[76];
              float v819_data = ir3[6];
              ir3[6] = (v819_data + (v786_data * v817_data));
              float v822_data = s1[88];
              float v824_data = ir3[7];
              ir3[7] = (v824_data + (v786_data * v822_data));
            }
            if (v3_lead < 12) {
              float v830_data = r2[5];
              float v831_data = s1[5];
              float v833_data = ir3[0];
              ir3[0] = (v833_data + (v830_data * v831_data));
              float v836_data = s1[17];
              float v838_data = ir3[1];
              ir3[1] = (v838_data + (v830_data * v836_data));
              float v841_data = s1[29];
              float v843_data = ir3[2];
              ir3[2] = (v843_data + (v830_data * v841_data));
              float v846_data = s1[41];
              float v848_data = ir3[3];
              ir3[3] = (v848_data + (v830_data * v846_data));
              float v851_data = s1[53];
              float v853_data = ir3[4];
              ir3[4] = (v853_data + (v830_data * v851_data));
              float v856_data = s1[65];
              float v858_data = ir3[5];
              ir3[5] = (v858_data + (v830_data * v856_data));
              float v861_data = s1[77];
              float v863_data = ir3[6];
              ir3[6] = (v863_data + (v830_data * v861_data));
              float v866_data = s1[89];
              float v868_data = ir3[7];
              ir3[7] = (v868_data + (v830_data * v866_data));
            }
            if (v3_lead < 12) {
              float v874_data = r2[6];
              float v875_data = s1[6];
              float v877_data = ir3[0];
              ir3[0] = (v877_data + (v874_data * v875_data));
              float v880_data = s1[18];
              float v882_data = ir3[1];
              ir3[1] = (v882_data + (v874_data * v880_data));
              float v885_data = s1[30];
              float v887_data = ir3[2];
              ir3[2] = (v887_data + (v874_data * v885_data));
              float v890_data = s1[42];
              float v892_data = ir3[3];
              ir3[3] = (v892_data + (v874_data * v890_data));
              float v895_data = s1[54];
              float v897_data = ir3[4];
              ir3[4] = (v897_data + (v874_data * v895_data));
              float v900_data = s1[66];
              float v902_data = ir3[5];
              ir3[5] = (v902_data + (v874_data * v900_data));
              float v905_data = s1[78];
              float v907_data = ir3[6];
              ir3[6] = (v907_data + (v874_data * v905_data));
              float v910_data = s1[90];
              float v912_data = ir3[7];
              ir3[7] = (v912_data + (v874_data * v910_data));
            }
            if (v3_lead < 12) {
              float v918_data = r2[7];
              float v919_data = s1[7];
              float v921_data = ir3[0];
              ir3[0] = (v921_data + (v918_data * v919_data));
              float v924_data = s1[19];
              float v926_data = ir3[1];
              ir3[1] = (v926_data + (v918_data * v924_data));
              float v929_data = s1[31];
              float v931_data = ir3[2];
              ir3[2] = (v931_data + (v918_data * v929_data));
              float v934_data = s1[43];
              float v936_data = ir3[3];
              ir3[3] = (v936_data + (v918_data * v934_data));
              float v939_data = s1[55];
              float v941_data = ir3[4];
              ir3[4] = (v941_data + (v918_data * v939_data));
              float v944_data = s1[67];
              float v946_data = ir3[5];
              ir3[5] = (v946_data + (v918_data * v944_data));
              float v949_data = s1[79];
              float v951_data = ir3[6];
              ir3[6] = (v951_data + (v918_data * v949_data));
              float v954_data = s1[91];
              float v956_data = ir3[7];
              ir3[7] = (v956_data + (v918_data * v954_data));
            }
            if (v3_lead < 12) {
              float v962_data = r2[8];
              float v963_data = s1[8];
              float v965_data = ir3[0];
              ir3[0] = (v965_data + (v962_data * v963_data));
              float v968_data = s1[20];
              float v970_data = ir3[1];
              ir3[1] = (v970_data + (v962_data * v968_data));
              float v973_data = s1[32];
              float v975_data = ir3[2];
              ir3[2] = (v975_data + (v962_data * v973_data));
              float v978_data = s1[44];
              float v980_data = ir3[3];
              ir3[3] = (v980_data + (v962_data * v978_data));
              float v983_data = s1[56];
              float v985_data = ir3[4];
              ir3[4] = (v985_data + (v962_data * v983_data));
              float v988_data = s1[68];
              float v990_data = ir3[5];
              ir3[5] = (v990_data + (v962_data * v988_data));
              float v993_data = s1[80];
              float v995_data = ir3[6];
              ir3[6] = (v995_data + (v962_data * v993_data));
              float v998_data = s1[92];
              float v1000_data = ir3[7];
              ir3[7] = (v1000_data + (v962_data * v998_data));
            }
            if (v3_lead < 12) {
              float v1006_data = r2[9];
              float v1007_data = s1[9];
              float v1009_data = ir3[0];
              ir3[0] = (v1009_data + (v1006_data * v1007_data));
              float v1012_data = s1[21];
              float v1014_data = ir3[1];
              ir3[1] = (v1014_data + (v1006_data * v1012_data));
              float v1017_data = s1[33];
              float v1019_data = ir3[2];
              ir3[2] = (v1019_data + (v1006_data * v1017_data));
              float v1022_data = s1[45];
              float v1024_data = ir3[3];
              ir3[3] = (v1024_data + (v1006_data * v1022_data));
              float v1027_data = s1[57];
              float v1029_data = ir3[4];
              ir3[4] = (v1029_data + (v1006_data * v1027_data));
              float v1032_data = s1[69];
              float v1034_data = ir3[5];
              ir3[5] = (v1034_data + (v1006_data * v1032_data));
              float v1037_data = s1[81];
              float v1039_data = ir3[6];
              ir3[6] = (v1039_data + (v1006_data * v1037_data));
              float v1042_data = s1[93];
              float v1044_data = ir3[7];
              ir3[7] = (v1044_data + (v1006_data * v1042_data));
            }
            if (v3_lead < 12) {
              float v1050_data = r2[10];
              float v1051_data = s1[10];
              float v1053_data = ir3[0];
              ir3[0] = (v1053_data + (v1050_data * v1051_data));
              float v1056_data = s1[22];
              float v1058_data = ir3[1];
              ir3[1] = (v1058_data + (v1050_data * v1056_data));
              float v1061_data = s1[34];
              float v1063_data = ir3[2];
              ir3[2] = (v1063_data + (v1050_data * v1061_data));
              float v1066_data = s1[46];
              float v1068_data = ir3[3];
              ir3[3] = (v1068_data + (v1050_data * v1066_data));
              float v1071_data = s1[58];
              float v1073_data = ir3[4];
              ir3[4] = (v1073_data + (v1050_data * v1071_data));
              float v1076_data = s1[70];
              float v1078_data = ir3[5];
              ir3[5] = (v1078_data + (v1050_data * v1076_data));
              float v1081_data = s1[82];
              float v1083_data = ir3[6];
              ir3[6] = (v1083_data + (v1050_data * v1081_data));
              float v1086_data = s1[94];
              float v1088_data = ir3[7];
              ir3[7] = (v1088_data + (v1050_data * v1086_data));
            }
            if (v3_lead < 12) {
              float v1094_data = r2[11];
              float v1095_data = s1[11];
              float v1097_data = ir3[0];
              ir3[0] = (v1097_data + (v1094_data * v1095_data));
              float v1100_data = s1[23];
              float v1102_data = ir3[1];
              ir3[1] = (v1102_data + (v1094_data * v1100_data));
              float v1105_data = s1[35];
              float v1107_data = ir3[2];
              ir3[2] = (v1107_data + (v1094_data * v1105_data));
              float v1110_data = s1[47];
              float v1112_data = ir3[3];
              ir3[3] = (v1112_data + (v1094_data * v1110_data));
              float v1115_data = s1[59];
              float v1117_data = ir3[4];
              ir3[4] = (v1117_data + (v1094_data * v1115_data));
              float v1120_data = s1[71];
              float v1122_data = ir3[5];
              ir3[5] = (v1122_data + (v1094_data * v1120_data));
              float v1125_data = s1[83];
              float v1127_data = ir3[6];
              ir3[6] = (v1127_data + (v1094_data * v1125_data));
              float v1130_data = s1[95];
              float v1132_data = ir3[7];
              ir3[7] = (v1132_data + (v1094_data * v1130_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v1138_n1 = 0; v1138_n1 < 8; ++v1138_n1) {
                int32_t v1139_a = 0 + v1138_n1;
                float v1141_data = ir3[v1138_n1];
                int32_t v1142_a = 0 + v1138_n1;
                float v1144_data = r1[v1138_n1];
                int32_t v1146_a = 0 + v1138_n1;
                r3[v1138_n1] = (v1144_data + v1141_data);
              }
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          {
            // s2 = load{g>s}(glb_m6[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1153_i1 = 0; v1153_i1 < 12; ++v1153_i1) {
              int32_t v1159_a = v1153_i1 * 12;
              int32_t v1160_a = v3_lead + v1159_a;
              float v1168_data = __ldcg(&glb_m7[(v3_lead + v1159_a)]);
              int32_t v1169_a = 0 + v1153_i1;
              r6[v1169_a] = v1168_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r5[8]{};
          __syncwarp();
          {
            // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            if (v3_lead < 12) {
              float v1175_data = r4[0];
              float v1176_data = s2[0];
              float v1178_data = ir5[0];
              ir5[0] = (v1178_data + (v1175_data * v1176_data));
              float v1181_data = s2[12];
              float v1183_data = ir5[1];
              ir5[1] = (v1183_data + (v1175_data * v1181_data));
              float v1186_data = s2[24];
              float v1188_data = ir5[2];
              ir5[2] = (v1188_data + (v1175_data * v1186_data));
              float v1191_data = s2[36];
              float v1193_data = ir5[3];
              ir5[3] = (v1193_data + (v1175_data * v1191_data));
              float v1196_data = s2[48];
              float v1198_data = ir5[4];
              ir5[4] = (v1198_data + (v1175_data * v1196_data));
              float v1201_data = s2[60];
              float v1203_data = ir5[5];
              ir5[5] = (v1203_data + (v1175_data * v1201_data));
              float v1206_data = s2[72];
              float v1208_data = ir5[6];
              ir5[6] = (v1208_data + (v1175_data * v1206_data));
              float v1211_data = s2[84];
              float v1213_data = ir5[7];
              ir5[7] = (v1213_data + (v1175_data * v1211_data));
            }
            if (v3_lead < 12) {
              float v1219_data = r4[1];
              float v1220_data = s2[1];
              float v1222_data = ir5[0];
              ir5[0] = (v1222_data + (v1219_data * v1220_data));
              float v1225_data = s2[13];
              float v1227_data = ir5[1];
              ir5[1] = (v1227_data + (v1219_data * v1225_data));
              float v1230_data = s2[25];
              float v1232_data = ir5[2];
              ir5[2] = (v1232_data + (v1219_data * v1230_data));
              float v1235_data = s2[37];
              float v1237_data = ir5[3];
              ir5[3] = (v1237_data + (v1219_data * v1235_data));
              float v1240_data = s2[49];
              float v1242_data = ir5[4];
              ir5[4] = (v1242_data + (v1219_data * v1240_data));
              float v1245_data = s2[61];
              float v1247_data = ir5[5];
              ir5[5] = (v1247_data + (v1219_data * v1245_data));
              float v1250_data = s2[73];
              float v1252_data = ir5[6];
              ir5[6] = (v1252_data + (v1219_data * v1250_data));
              float v1255_data = s2[85];
              float v1257_data = ir5[7];
              ir5[7] = (v1257_data + (v1219_data * v1255_data));
            }
            if (v3_lead < 12) {
              float v1263_data = r4[2];
              float v1264_data = s2[2];
              float v1266_data = ir5[0];
              ir5[0] = (v1266_data + (v1263_data * v1264_data));
              float v1269_data = s2[14];
              float v1271_data = ir5[1];
              ir5[1] = (v1271_data + (v1263_data * v1269_data));
              float v1274_data = s2[26];
              float v1276_data = ir5[2];
              ir5[2] = (v1276_data + (v1263_data * v1274_data));
              float v1279_data = s2[38];
              float v1281_data = ir5[3];
              ir5[3] = (v1281_data + (v1263_data * v1279_data));
              float v1284_data = s2[50];
              float v1286_data = ir5[4];
              ir5[4] = (v1286_data + (v1263_data * v1284_data));
              float v1289_data = s2[62];
              float v1291_data = ir5[5];
              ir5[5] = (v1291_data + (v1263_data * v1289_data));
              float v1294_data = s2[74];
              float v1296_data = ir5[6];
              ir5[6] = (v1296_data + (v1263_data * v1294_data));
              float v1299_data = s2[86];
              float v1301_data = ir5[7];
              ir5[7] = (v1301_data + (v1263_data * v1299_data));
            }
            if (v3_lead < 12) {
              float v1307_data = r4[3];
              float v1308_data = s2[3];
              float v1310_data = ir5[0];
              ir5[0] = (v1310_data + (v1307_data * v1308_data));
              float v1313_data = s2[15];
              float v1315_data = ir5[1];
              ir5[1] = (v1315_data + (v1307_data * v1313_data));
              float v1318_data = s2[27];
              float v1320_data = ir5[2];
              ir5[2] = (v1320_data + (v1307_data * v1318_data));
              float v1323_data = s2[39];
              float v1325_data = ir5[3];
              ir5[3] = (v1325_data + (v1307_data * v1323_data));
              float v1328_data = s2[51];
              float v1330_data = ir5[4];
              ir5[4] = (v1330_data + (v1307_data * v1328_data));
              float v1333_data = s2[63];
              float v1335_data = ir5[5];
              ir5[5] = (v1335_data + (v1307_data * v1333_data));
              float v1338_data = s2[75];
              float v1340_data = ir5[6];
              ir5[6] = (v1340_data + (v1307_data * v1338_data));
              float v1343_data = s2[87];
              float v1345_data = ir5[7];
              ir5[7] = (v1345_data + (v1307_data * v1343_data));
            }
            if (v3_lead < 12) {
              float v1351_data = r4[4];
              float v1352_data = s2[4];
              float v1354_data = ir5[0];
              ir5[0] = (v1354_data + (v1351_data * v1352_data));
              float v1357_data = s2[16];
              float v1359_data = ir5[1];
              ir5[1] = (v1359_data + (v1351_data * v1357_data));
              float v1362_data = s2[28];
              float v1364_data = ir5[2];
              ir5[2] = (v1364_data + (v1351_data * v1362_data));
              float v1367_data = s2[40];
              float v1369_data = ir5[3];
              ir5[3] = (v1369_data + (v1351_data * v1367_data));
              float v1372_data = s2[52];
              float v1374_data = ir5[4];
              ir5[4] = (v1374_data + (v1351_data * v1372_data));
              float v1377_data = s2[64];
              float v1379_data = ir5[5];
              ir5[5] = (v1379_data + (v1351_data * v1377_data));
              float v1382_data = s2[76];
              float v1384_data = ir5[6];
              ir5[6] = (v1384_data + (v1351_data * v1382_data));
              float v1387_data = s2[88];
              float v1389_data = ir5[7];
              ir5[7] = (v1389_data + (v1351_data * v1387_data));
            }
            if (v3_lead < 12) {
              float v1395_data = r4[5];
              float v1396_data = s2[5];
              float v1398_data = ir5[0];
              ir5[0] = (v1398_data + (v1395_data * v1396_data));
              float v1401_data = s2[17];
              float v1403_data = ir5[1];
              ir5[1] = (v1403_data + (v1395_data * v1401_data));
              float v1406_data = s2[29];
              float v1408_data = ir5[2];
              ir5[2] = (v1408_data + (v1395_data * v1406_data));
              float v1411_data = s2[41];
              float v1413_data = ir5[3];
              ir5[3] = (v1413_data + (v1395_data * v1411_data));
              float v1416_data = s2[53];
              float v1418_data = ir5[4];
              ir5[4] = (v1418_data + (v1395_data * v1416_data));
              float v1421_data = s2[65];
              float v1423_data = ir5[5];
              ir5[5] = (v1423_data + (v1395_data * v1421_data));
              float v1426_data = s2[77];
              float v1428_data = ir5[6];
              ir5[6] = (v1428_data + (v1395_data * v1426_data));
              float v1431_data = s2[89];
              float v1433_data = ir5[7];
              ir5[7] = (v1433_data + (v1395_data * v1431_data));
            }
            if (v3_lead < 12) {
              float v1439_data = r4[6];
              float v1440_data = s2[6];
              float v1442_data = ir5[0];
              ir5[0] = (v1442_data + (v1439_data * v1440_data));
              float v1445_data = s2[18];
              float v1447_data = ir5[1];
              ir5[1] = (v1447_data + (v1439_data * v1445_data));
              float v1450_data = s2[30];
              float v1452_data = ir5[2];
              ir5[2] = (v1452_data + (v1439_data * v1450_data));
              float v1455_data = s2[42];
              float v1457_data = ir5[3];
              ir5[3] = (v1457_data + (v1439_data * v1455_data));
              float v1460_data = s2[54];
              float v1462_data = ir5[4];
              ir5[4] = (v1462_data + (v1439_data * v1460_data));
              float v1465_data = s2[66];
              float v1467_data = ir5[5];
              ir5[5] = (v1467_data + (v1439_data * v1465_data));
              float v1470_data = s2[78];
              float v1472_data = ir5[6];
              ir5[6] = (v1472_data + (v1439_data * v1470_data));
              float v1475_data = s2[90];
              float v1477_data = ir5[7];
              ir5[7] = (v1477_data + (v1439_data * v1475_data));
            }
            if (v3_lead < 12) {
              float v1483_data = r4[7];
              float v1484_data = s2[7];
              float v1486_data = ir5[0];
              ir5[0] = (v1486_data + (v1483_data * v1484_data));
              float v1489_data = s2[19];
              float v1491_data = ir5[1];
              ir5[1] = (v1491_data + (v1483_data * v1489_data));
              float v1494_data = s2[31];
              float v1496_data = ir5[2];
              ir5[2] = (v1496_data + (v1483_data * v1494_data));
              float v1499_data = s2[43];
              float v1501_data = ir5[3];
              ir5[3] = (v1501_data + (v1483_data * v1499_data));
              float v1504_data = s2[55];
              float v1506_data = ir5[4];
              ir5[4] = (v1506_data + (v1483_data * v1504_data));
              float v1509_data = s2[67];
              float v1511_data = ir5[5];
              ir5[5] = (v1511_data + (v1483_data * v1509_data));
              float v1514_data = s2[79];
              float v1516_data = ir5[6];
              ir5[6] = (v1516_data + (v1483_data * v1514_data));
              float v1519_data = s2[91];
              float v1521_data = ir5[7];
              ir5[7] = (v1521_data + (v1483_data * v1519_data));
            }
            if (v3_lead < 12) {
              float v1527_data = r4[8];
              float v1528_data = s2[8];
              float v1530_data = ir5[0];
              ir5[0] = (v1530_data + (v1527_data * v1528_data));
              float v1533_data = s2[20];
              float v1535_data = ir5[1];
              ir5[1] = (v1535_data + (v1527_data * v1533_data));
              float v1538_data = s2[32];
              float v1540_data = ir5[2];
              ir5[2] = (v1540_data + (v1527_data * v1538_data));
              float v1543_data = s2[44];
              float v1545_data = ir5[3];
              ir5[3] = (v1545_data + (v1527_data * v1543_data));
              float v1548_data = s2[56];
              float v1550_data = ir5[4];
              ir5[4] = (v1550_data + (v1527_data * v1548_data));
              float v1553_data = s2[68];
              float v1555_data = ir5[5];
              ir5[5] = (v1555_data + (v1527_data * v1553_data));
              float v1558_data = s2[80];
              float v1560_data = ir5[6];
              ir5[6] = (v1560_data + (v1527_data * v1558_data));
              float v1563_data = s2[92];
              float v1565_data = ir5[7];
              ir5[7] = (v1565_data + (v1527_data * v1563_data));
            }
            if (v3_lead < 12) {
              float v1571_data = r4[9];
              float v1572_data = s2[9];
              float v1574_data = ir5[0];
              ir5[0] = (v1574_data + (v1571_data * v1572_data));
              float v1577_data = s2[21];
              float v1579_data = ir5[1];
              ir5[1] = (v1579_data + (v1571_data * v1577_data));
              float v1582_data = s2[33];
              float v1584_data = ir5[2];
              ir5[2] = (v1584_data + (v1571_data * v1582_data));
              float v1587_data = s2[45];
              float v1589_data = ir5[3];
              ir5[3] = (v1589_data + (v1571_data * v1587_data));
              float v1592_data = s2[57];
              float v1594_data = ir5[4];
              ir5[4] = (v1594_data + (v1571_data * v1592_data));
              float v1597_data = s2[69];
              float v1599_data = ir5[5];
              ir5[5] = (v1599_data + (v1571_data * v1597_data));
              float v1602_data = s2[81];
              float v1604_data = ir5[6];
              ir5[6] = (v1604_data + (v1571_data * v1602_data));
              float v1607_data = s2[93];
              float v1609_data = ir5[7];
              ir5[7] = (v1609_data + (v1571_data * v1607_data));
            }
            if (v3_lead < 12) {
              float v1615_data = r4[10];
              float v1616_data = s2[10];
              float v1618_data = ir5[0];
              ir5[0] = (v1618_data + (v1615_data * v1616_data));
              float v1621_data = s2[22];
              float v1623_data = ir5[1];
              ir5[1] = (v1623_data + (v1615_data * v1621_data));
              float v1626_data = s2[34];
              float v1628_data = ir5[2];
              ir5[2] = (v1628_data + (v1615_data * v1626_data));
              float v1631_data = s2[46];
              float v1633_data = ir5[3];
              ir5[3] = (v1633_data + (v1615_data * v1631_data));
              float v1636_data = s2[58];
              float v1638_data = ir5[4];
              ir5[4] = (v1638_data + (v1615_data * v1636_data));
              float v1641_data = s2[70];
              float v1643_data = ir5[5];
              ir5[5] = (v1643_data + (v1615_data * v1641_data));
              float v1646_data = s2[82];
              float v1648_data = ir5[6];
              ir5[6] = (v1648_data + (v1615_data * v1646_data));
              float v1651_data = s2[94];
              float v1653_data = ir5[7];
              ir5[7] = (v1653_data + (v1615_data * v1651_data));
            }
            if (v3_lead < 12) {
              float v1659_data = r4[11];
              float v1660_data = s2[11];
              float v1662_data = ir5[0];
              ir5[0] = (v1662_data + (v1659_data * v1660_data));
              float v1665_data = s2[23];
              float v1667_data = ir5[1];
              ir5[1] = (v1667_data + (v1659_data * v1665_data));
              float v1670_data = s2[35];
              float v1672_data = ir5[2];
              ir5[2] = (v1672_data + (v1659_data * v1670_data));
              float v1675_data = s2[47];
              float v1677_data = ir5[3];
              ir5[3] = (v1677_data + (v1659_data * v1675_data));
              float v1680_data = s2[59];
              float v1682_data = ir5[4];
              ir5[4] = (v1682_data + (v1659_data * v1680_data));
              float v1685_data = s2[71];
              float v1687_data = ir5[5];
              ir5[5] = (v1687_data + (v1659_data * v1685_data));
              float v1690_data = s2[83];
              float v1692_data = ir5[6];
              ir5[6] = (v1692_data + (v1659_data * v1690_data));
              float v1695_data = s2[95];
              float v1697_data = ir5[7];
              ir5[7] = (v1697_data + (v1659_data * v1695_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v1703_n1 = 0; v1703_n1 < 8; ++v1703_n1) {
                int32_t v1704_a = 0 + v1703_n1;
                float v1706_data = ir5[v1703_n1];
                int32_t v1707_a = 0 + v1703_n1;
                float v1709_data = r3[v1703_n1];
                int32_t v1711_a = 0 + v1703_n1;
                r5[v1703_n1] = (v1709_data + v1706_data);
              }
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          {
            // s3 = load{g>s}(glb_m8[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r7[8]{};
          __syncwarp();
          {
            // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir7[8]{};
            if (v3_lead < 12) {
              float v1718_data = r6[0];
              float v1719_data = s3[0];
              float v1721_data = ir7[0];
              ir7[0] = (v1721_data + (v1718_data * v1719_data));
              float v1724_data = s3[12];
              float v1726_data = ir7[1];
              ir7[1] = (v1726_data + (v1718_data * v1724_data));
              float v1729_data = s3[24];
              float v1731_data = ir7[2];
              ir7[2] = (v1731_data + (v1718_data * v1729_data));
              float v1734_data = s3[36];
              float v1736_data = ir7[3];
              ir7[3] = (v1736_data + (v1718_data * v1734_data));
              float v1739_data = s3[48];
              float v1741_data = ir7[4];
              ir7[4] = (v1741_data + (v1718_data * v1739_data));
              float v1744_data = s3[60];
              float v1746_data = ir7[5];
              ir7[5] = (v1746_data + (v1718_data * v1744_data));
              float v1749_data = s3[72];
              float v1751_data = ir7[6];
              ir7[6] = (v1751_data + (v1718_data * v1749_data));
              float v1754_data = s3[84];
              float v1756_data = ir7[7];
              ir7[7] = (v1756_data + (v1718_data * v1754_data));
            }
            if (v3_lead < 12) {
              float v1762_data = r6[1];
              float v1763_data = s3[1];
              float v1765_data = ir7[0];
              ir7[0] = (v1765_data + (v1762_data * v1763_data));
              float v1768_data = s3[13];
              float v1770_data = ir7[1];
              ir7[1] = (v1770_data + (v1762_data * v1768_data));
              float v1773_data = s3[25];
              float v1775_data = ir7[2];
              ir7[2] = (v1775_data + (v1762_data * v1773_data));
              float v1778_data = s3[37];
              float v1780_data = ir7[3];
              ir7[3] = (v1780_data + (v1762_data * v1778_data));
              float v1783_data = s3[49];
              float v1785_data = ir7[4];
              ir7[4] = (v1785_data + (v1762_data * v1783_data));
              float v1788_data = s3[61];
              float v1790_data = ir7[5];
              ir7[5] = (v1790_data + (v1762_data * v1788_data));
              float v1793_data = s3[73];
              float v1795_data = ir7[6];
              ir7[6] = (v1795_data + (v1762_data * v1793_data));
              float v1798_data = s3[85];
              float v1800_data = ir7[7];
              ir7[7] = (v1800_data + (v1762_data * v1798_data));
            }
            if (v3_lead < 12) {
              float v1806_data = r6[2];
              float v1807_data = s3[2];
              float v1809_data = ir7[0];
              ir7[0] = (v1809_data + (v1806_data * v1807_data));
              float v1812_data = s3[14];
              float v1814_data = ir7[1];
              ir7[1] = (v1814_data + (v1806_data * v1812_data));
              float v1817_data = s3[26];
              float v1819_data = ir7[2];
              ir7[2] = (v1819_data + (v1806_data * v1817_data));
              float v1822_data = s3[38];
              float v1824_data = ir7[3];
              ir7[3] = (v1824_data + (v1806_data * v1822_data));
              float v1827_data = s3[50];
              float v1829_data = ir7[4];
              ir7[4] = (v1829_data + (v1806_data * v1827_data));
              float v1832_data = s3[62];
              float v1834_data = ir7[5];
              ir7[5] = (v1834_data + (v1806_data * v1832_data));
              float v1837_data = s3[74];
              float v1839_data = ir7[6];
              ir7[6] = (v1839_data + (v1806_data * v1837_data));
              float v1842_data = s3[86];
              float v1844_data = ir7[7];
              ir7[7] = (v1844_data + (v1806_data * v1842_data));
            }
            if (v3_lead < 12) {
              float v1850_data = r6[3];
              float v1851_data = s3[3];
              float v1853_data = ir7[0];
              ir7[0] = (v1853_data + (v1850_data * v1851_data));
              float v1856_data = s3[15];
              float v1858_data = ir7[1];
              ir7[1] = (v1858_data + (v1850_data * v1856_data));
              float v1861_data = s3[27];
              float v1863_data = ir7[2];
              ir7[2] = (v1863_data + (v1850_data * v1861_data));
              float v1866_data = s3[39];
              float v1868_data = ir7[3];
              ir7[3] = (v1868_data + (v1850_data * v1866_data));
              float v1871_data = s3[51];
              float v1873_data = ir7[4];
              ir7[4] = (v1873_data + (v1850_data * v1871_data));
              float v1876_data = s3[63];
              float v1878_data = ir7[5];
              ir7[5] = (v1878_data + (v1850_data * v1876_data));
              float v1881_data = s3[75];
              float v1883_data = ir7[6];
              ir7[6] = (v1883_data + (v1850_data * v1881_data));
              float v1886_data = s3[87];
              float v1888_data = ir7[7];
              ir7[7] = (v1888_data + (v1850_data * v1886_data));
            }
            if (v3_lead < 12) {
              float v1894_data = r6[4];
              float v1895_data = s3[4];
              float v1897_data = ir7[0];
              ir7[0] = (v1897_data + (v1894_data * v1895_data));
              float v1900_data = s3[16];
              float v1902_data = ir7[1];
              ir7[1] = (v1902_data + (v1894_data * v1900_data));
              float v1905_data = s3[28];
              float v1907_data = ir7[2];
              ir7[2] = (v1907_data + (v1894_data * v1905_data));
              float v1910_data = s3[40];
              float v1912_data = ir7[3];
              ir7[3] = (v1912_data + (v1894_data * v1910_data));
              float v1915_data = s3[52];
              float v1917_data = ir7[4];
              ir7[4] = (v1917_data + (v1894_data * v1915_data));
              float v1920_data = s3[64];
              float v1922_data = ir7[5];
              ir7[5] = (v1922_data + (v1894_data * v1920_data));
              float v1925_data = s3[76];
              float v1927_data = ir7[6];
              ir7[6] = (v1927_data + (v1894_data * v1925_data));
              float v1930_data = s3[88];
              float v1932_data = ir7[7];
              ir7[7] = (v1932_data + (v1894_data * v1930_data));
            }
            if (v3_lead < 12) {
              float v1938_data = r6[5];
              float v1939_data = s3[5];
              float v1941_data = ir7[0];
              ir7[0] = (v1941_data + (v1938_data * v1939_data));
              float v1944_data = s3[17];
              float v1946_data = ir7[1];
              ir7[1] = (v1946_data + (v1938_data * v1944_data));
              float v1949_data = s3[29];
              float v1951_data = ir7[2];
              ir7[2] = (v1951_data + (v1938_data * v1949_data));
              float v1954_data = s3[41];
              float v1956_data = ir7[3];
              ir7[3] = (v1956_data + (v1938_data * v1954_data));
              float v1959_data = s3[53];
              float v1961_data = ir7[4];
              ir7[4] = (v1961_data + (v1938_data * v1959_data));
              float v1964_data = s3[65];
              float v1966_data = ir7[5];
              ir7[5] = (v1966_data + (v1938_data * v1964_data));
              float v1969_data = s3[77];
              float v1971_data = ir7[6];
              ir7[6] = (v1971_data + (v1938_data * v1969_data));
              float v1974_data = s3[89];
              float v1976_data = ir7[7];
              ir7[7] = (v1976_data + (v1938_data * v1974_data));
            }
            if (v3_lead < 12) {
              float v1982_data = r6[6];
              float v1983_data = s3[6];
              float v1985_data = ir7[0];
              ir7[0] = (v1985_data + (v1982_data * v1983_data));
              float v1988_data = s3[18];
              float v1990_data = ir7[1];
              ir7[1] = (v1990_data + (v1982_data * v1988_data));
              float v1993_data = s3[30];
              float v1995_data = ir7[2];
              ir7[2] = (v1995_data + (v1982_data * v1993_data));
              float v1998_data = s3[42];
              float v2000_data = ir7[3];
              ir7[3] = (v2000_data + (v1982_data * v1998_data));
              float v2003_data = s3[54];
              float v2005_data = ir7[4];
              ir7[4] = (v2005_data + (v1982_data * v2003_data));
              float v2008_data = s3[66];
              float v2010_data = ir7[5];
              ir7[5] = (v2010_data + (v1982_data * v2008_data));
              float v2013_data = s3[78];
              float v2015_data = ir7[6];
              ir7[6] = (v2015_data + (v1982_data * v2013_data));
              float v2018_data = s3[90];
              float v2020_data = ir7[7];
              ir7[7] = (v2020_data + (v1982_data * v2018_data));
            }
            if (v3_lead < 12) {
              float v2026_data = r6[7];
              float v2027_data = s3[7];
              float v2029_data = ir7[0];
              ir7[0] = (v2029_data + (v2026_data * v2027_data));
              float v2032_data = s3[19];
              float v2034_data = ir7[1];
              ir7[1] = (v2034_data + (v2026_data * v2032_data));
              float v2037_data = s3[31];
              float v2039_data = ir7[2];
              ir7[2] = (v2039_data + (v2026_data * v2037_data));
              float v2042_data = s3[43];
              float v2044_data = ir7[3];
              ir7[3] = (v2044_data + (v2026_data * v2042_data));
              float v2047_data = s3[55];
              float v2049_data = ir7[4];
              ir7[4] = (v2049_data + (v2026_data * v2047_data));
              float v2052_data = s3[67];
              float v2054_data = ir7[5];
              ir7[5] = (v2054_data + (v2026_data * v2052_data));
              float v2057_data = s3[79];
              float v2059_data = ir7[6];
              ir7[6] = (v2059_data + (v2026_data * v2057_data));
              float v2062_data = s3[91];
              float v2064_data = ir7[7];
              ir7[7] = (v2064_data + (v2026_data * v2062_data));
            }
            if (v3_lead < 12) {
              float v2070_data = r6[8];
              float v2071_data = s3[8];
              float v2073_data = ir7[0];
              ir7[0] = (v2073_data + (v2070_data * v2071_data));
              float v2076_data = s3[20];
              float v2078_data = ir7[1];
              ir7[1] = (v2078_data + (v2070_data * v2076_data));
              float v2081_data = s3[32];
              float v2083_data = ir7[2];
              ir7[2] = (v2083_data + (v2070_data * v2081_data));
              float v2086_data = s3[44];
              float v2088_data = ir7[3];
              ir7[3] = (v2088_data + (v2070_data * v2086_data));
              float v2091_data = s3[56];
              float v2093_data = ir7[4];
              ir7[4] = (v2093_data + (v2070_data * v2091_data));
              float v2096_data = s3[68];
              float v2098_data = ir7[5];
              ir7[5] = (v2098_data + (v2070_data * v2096_data));
              float v2101_data = s3[80];
              float v2103_data = ir7[6];
              ir7[6] = (v2103_data + (v2070_data * v2101_data));
              float v2106_data = s3[92];
              float v2108_data = ir7[7];
              ir7[7] = (v2108_data + (v2070_data * v2106_data));
            }
            if (v3_lead < 12) {
              float v2114_data = r6[9];
              float v2115_data = s3[9];
              float v2117_data = ir7[0];
              ir7[0] = (v2117_data + (v2114_data * v2115_data));
              float v2120_data = s3[21];
              float v2122_data = ir7[1];
              ir7[1] = (v2122_data + (v2114_data * v2120_data));
              float v2125_data = s3[33];
              float v2127_data = ir7[2];
              ir7[2] = (v2127_data + (v2114_data * v2125_data));
              float v2130_data = s3[45];
              float v2132_data = ir7[3];
              ir7[3] = (v2132_data + (v2114_data * v2130_data));
              float v2135_data = s3[57];
              float v2137_data = ir7[4];
              ir7[4] = (v2137_data + (v2114_data * v2135_data));
              float v2140_data = s3[69];
              float v2142_data = ir7[5];
              ir7[5] = (v2142_data + (v2114_data * v2140_data));
              float v2145_data = s3[81];
              float v2147_data = ir7[6];
              ir7[6] = (v2147_data + (v2114_data * v2145_data));
              float v2150_data = s3[93];
              float v2152_data = ir7[7];
              ir7[7] = (v2152_data + (v2114_data * v2150_data));
            }
            if (v3_lead < 12) {
              float v2158_data = r6[10];
              float v2159_data = s3[10];
              float v2161_data = ir7[0];
              ir7[0] = (v2161_data + (v2158_data * v2159_data));
              float v2164_data = s3[22];
              float v2166_data = ir7[1];
              ir7[1] = (v2166_data + (v2158_data * v2164_data));
              float v2169_data = s3[34];
              float v2171_data = ir7[2];
              ir7[2] = (v2171_data + (v2158_data * v2169_data));
              float v2174_data = s3[46];
              float v2176_data = ir7[3];
              ir7[3] = (v2176_data + (v2158_data * v2174_data));
              float v2179_data = s3[58];
              float v2181_data = ir7[4];
              ir7[4] = (v2181_data + (v2158_data * v2179_data));
              float v2184_data = s3[70];
              float v2186_data = ir7[5];
              ir7[5] = (v2186_data + (v2158_data * v2184_data));
              float v2189_data = s3[82];
              float v2191_data = ir7[6];
              ir7[6] = (v2191_data + (v2158_data * v2189_data));
              float v2194_data = s3[94];
              float v2196_data = ir7[7];
              ir7[7] = (v2196_data + (v2158_data * v2194_data));
            }
            if (v3_lead < 12) {
              float v2202_data = r6[11];
              float v2203_data = s3[11];
              float v2205_data = ir7[0];
              ir7[0] = (v2205_data + (v2202_data * v2203_data));
              float v2208_data = s3[23];
              float v2210_data = ir7[1];
              ir7[1] = (v2210_data + (v2202_data * v2208_data));
              float v2213_data = s3[35];
              float v2215_data = ir7[2];
              ir7[2] = (v2215_data + (v2202_data * v2213_data));
              float v2218_data = s3[47];
              float v2220_data = ir7[3];
              ir7[3] = (v2220_data + (v2202_data * v2218_data));
              float v2223_data = s3[59];
              float v2225_data = ir7[4];
              ir7[4] = (v2225_data + (v2202_data * v2223_data));
              float v2228_data = s3[71];
              float v2230_data = ir7[5];
              ir7[5] = (v2230_data + (v2202_data * v2228_data));
              float v2233_data = s3[83];
              float v2235_data = ir7[6];
              ir7[6] = (v2235_data + (v2202_data * v2233_data));
              float v2238_data = s3[95];
              float v2240_data = ir7[7];
              ir7[7] = (v2240_data + (v2202_data * v2238_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v2246_n1 = 0; v2246_n1 < 8; ++v2246_n1) {
                int32_t v2247_a = 0 + v2246_n1;
                float v2249_data = ir7[v2246_n1];
                int32_t v2250_a = 0 + v2246_n1;
                float v2252_data = r5[v2246_n1];
                int32_t v2254_a = 0 + v2246_n1;
                r7[v2246_n1] = (v2252_data + v2249_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2260_i1 = 0; v2260_i1 < 8; ++v2260_i1) {
              int32_t v2261_a = 0 + v2260_i1;
              float v2263_data = r7[v2260_i1];
              int32_t v2270_a = v3_lead + (v2260_i1 * 12);
              glb_m0[v2270_a] = v2263_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

